"""Train the ParT CVAE on SB pseudo-data (data pool; SR blinded).

    python -m cvae.train --run-dir $NGR_STORE/runs/<tag> [--seed 0]

Resumable: <run-dir>/last.pt (model, optimizer, LR schedule, history, RNG) is
written after every epoch and picked up on restart, so a run longer than one
interactive allocation is just relaunched with the same command (the config
must match, else it refuses). <run-dir>/DONE marks a finished run; rerunning
then skips training.

The optimisation recipe (AdamW param groups, warmup -> cosine -> floor LR,
bf16 autocast, grad clipping, GPU-resident weighted batch sampling) is the
contrastive_encoder's, reused directly. Batches are drawn with probability
∝ event_weight via contrastive's _GpuBatchLoader (uniform for unweighted
pseudo-data, so the in-batch dCor is a plain estimate on the physical mix).

Loss = v2_model vae_loss (masked recon summed over targets, BCE on binary
       heads, + KL_WEIGHT * KL(q(z|x) || N(0,I)))  [+ DISCO_WEIGHT * dCor^2(mu, myy), off]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from contextlib import nullcontext
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

from src.embedding_models.contrastive_encoder.train import _GpuBatchLoader
from src.embedding_models.v2_model._lib.losses import vae_loss
from src.embedding_models.v2_model.train import _save_reconstruction_pdf

from cvae import config as C
from cvae import data as D
from cvae.losses import distance_corr, hetero_vae_loss
from cvae.model import ParTCVAE

logger = logging.getLogger("cvae.train")


def _make_optimizer(model):
    # Same grouping as contrastive_encoder.train: no decay on 1-D params and class_token.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if (p.ndim == 1 or n.endswith("class_token")) else decay).append(p)
    return optim.AdamW(
        [{"params": decay, "weight_decay": C.WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
        lr=C.LR,
    )


def _make_scheduler(optimizer, total_steps):
    warmup_steps = min(C.WARMUP_STEPS, max(1, total_steps - 1))
    t_max = max(1, total_steps - warmup_steps)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=C.ETA_MIN)
    floor = optim.lr_scheduler.ConstantLR(optimizer, factor=C.ETA_MIN / C.LR, total_iters=10**9)
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine, floor], milestones=[warmup_steps, total_steps]
    )


def _loss_terms(model, batch, is_bin, lam, is_btag=None):
    pl, inter, hl, mask, tgt, tm, cond = batch
    if is_btag is not None and C.BTAG_POS_WEIGHT != 1.0:
        # Per-element weight inside the mask: vae_loss takes a masked weighted mean.
        tm = tm * (1.0 + (C.BTAG_POS_WEIGHT - 1.0) * tgt * is_btag)
    x_hat, z, mu, logvar = model((pl, inter, hl, mask), cond)
    # v2_model's masked VAE loss: per-event masked mean rescaled to a sum over
    # targets (so KL_WEIGHT means the same as for the v2 VAE), BCE on binary dims.
    # Batches are already drawn ∝ event_weight, hence unit weights here.
    dec_logvar = getattr(model, "dec_logvar", None)  # torch.compile forwards attribute access
    if dec_logvar is not None:
        vae, rec, kl = hetero_vae_loss(tgt, x_hat.float(), dec_logvar.float(), mu.float(), logvar.float(),
                                       C.KL_WEIGHT, tm, is_bin)
    else:
        w = torch.ones(len(tgt), device=tgt.device)
        vae, rec, kl = vae_loss(tgt, x_hat.float(), mu.float(), logvar.float(), w, C.KL_WEIGHT,
                                mask=tm, binary_dims=is_bin)
    lat = mu if C.DISCO_ON == "mu" else z
    nd = min(C.DISCO_BATCH, len(lat))
    dc = distance_corr(lat[:nd], cond[:nd])
    total = vae + lam * dc
    return total, rec, kl, dc, x_hat


def disco_lambda(epoch):
    """0 during warm-up, then a linear ramp to DISCO_WEIGHT."""
    if epoch < C.DISCO_WARMUP_EPOCHS:
        return 0.0
    ramp = max(C.DISCO_RAMP_EPOCHS, 1)
    return C.DISCO_WEIGHT * min(1.0, (epoch - C.DISCO_WARMUP_EPOCHS + 1) / ramp)


def _plot_history(hist, path):
    keys = ["total", "recon", "kl", "dcor"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, k in zip(axes, keys):
        ax.plot(hist["epoch"], hist[f"train_{k}"], label="train")
        ax.plot(hist["epoch"], hist[f"val_{k}"], label="val")
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train(run_dir, seed=0, device="cuda"):
    os.makedirs(run_dir, exist_ok=True)
    ckpt = os.path.join(run_dir, "model.pt")
    last = os.path.join(run_dir, "last.pt")
    done = os.path.join(run_dir, "DONE")
    if os.path.exists(done):
        logger.info("%s exists — training already finished, nothing to do", done)
        return ckpt
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if (C.USE_AMP and is_cuda) else nullcontext

    cfg = C.snapshot()
    resume = torch.load(last, map_location="cpu", weights_only=False) if os.path.exists(last) else None
    if resume is not None:
        old = json.loads(json.dumps(resume["config"], default=str))
        new = json.loads(json.dumps(cfg, default=str))
        diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}
        if diff or resume["seed"] != seed:
            raise RuntimeError(f"refusing to resume {last}: config/seed differ: {diff} seed {resume['seed']}->{seed}")
    else:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"seed": seed, **cfg}, f, indent=2, default=str)

    # --- data ----------------------------------------------------------------
    t0 = time.time()
    df = D.load_splits(seed)
    codes = df["region"].to_numpy()
    tr_idx = np.flatnonzero(codes == D.SB_TRAIN)
    va_idx = np.flatnonzero(codes == D.SB_VAL)
    if C.MAX_TRAIN_EVENTS > 0:
        rng = np.random.default_rng(seed)
        tr_idx = np.sort(rng.choice(tr_idx, min(C.MAX_TRAIN_EVENTS, len(tr_idx)), replace=False))
        va_idx = np.sort(rng.choice(va_idx, min(max(C.MAX_TRAIN_EVENTS // 4, 1), len(va_idx)), replace=False))
    norm_params = D.load_norm_params()
    inter_norm = D.interaction_norm_from(norm_params)
    tr = D.build_inputs(df.iloc[tr_idx], norm_params)
    va = D.build_inputs(df.iloc[va_idx], norm_params)
    del df
    logger.info("data: SB train=%d val=%d (test fold + SR held out), %.0fs",
                len(tr_idx), len(va_idx), time.time() - t0)

    tr_ds = D.CVAEDataSet(tr, device, inter_norm)
    va_ds = D.CVAEDataSet(va, device, inter_norm)
    train_loader = _GpuBatchLoader(tr_ds, tr["weight"], C.BATCH_SIZE, device)
    # Fixed weighted draw for validation -> comparable val loss across epochs.
    g = torch.Generator(device=device).manual_seed(12345)
    n_vb = max(1, len(va_ds) // C.BATCH_SIZE)
    bs_val = min(C.BATCH_SIZE, len(va_ds))
    val_idx = torch.multinomial(
        torch.as_tensor(va["weight"], dtype=torch.double, device=device), n_vb * bs_val, replacement=True, generator=g
    ).view(n_vb, bs_val)

    is_bin = torch.as_tensor(D.binary_target_mask(), device=device)
    is_btag = torch.as_tensor([n.endswith("_btag") for n in D.target_names()], device=device, dtype=torch.float32)
    n_targets = len(D.target_names())

    # --- model ----------------------------------------------------------------
    base_model = ParTCVAE(n_targets=n_targets).to(device)
    model = torch.compile(base_model, dynamic=False) if C.USE_COMPILE else base_model
    logger.info("ParTCVAE: %d params, latent=%d, targets=%d",
                sum(p.numel() for p in base_model.parameters()), C.LATENT_DIM, n_targets)

    optimizer = _make_optimizer(base_model)
    total_steps = len(train_loader) * C.EPOCHS
    sched = _make_scheduler(optimizer, total_steps)

    def _save(state, best_epoch, best_val):
        # Written on every improvement (atomic rename), so an interrupted run
        # still leaves its best checkpoint usable by encode / closure.
        tmp = ckpt + ".tmp"
        torch.save(
            {
                "model": state,
                "n_targets": n_targets,
                "target_names": D.target_names(),
                "config": cfg,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val": best_val,
                "history": hist,
            },
            tmp,
        )
        os.replace(tmp, ckpt)

    def _save_last(epoch):
        tmp = last + ".tmp"
        torch.save(
            {
                "epoch": epoch,
                "model": base_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sched": sched.state_dict(),
                "hist": hist,
                "best": (best_val, best_epoch, bad),
                "config": cfg,
                "seed": seed,
                "rng": (torch.get_rng_state(), torch.cuda.get_rng_state_all() if is_cuda else None,
                        np.random.get_state()),
            },
            tmp,
        )
        os.replace(tmp, last)

    hist = {k: [] for k in ["epoch"] + [f"{s}_{t}" for s in ("train", "val") for t in ("total", "recon", "kl", "dcor")]}
    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    start_epoch = 0
    if resume is not None:
        base_model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["sched"])
        hist = resume["hist"]
        best_val, best_epoch, bad = resume["best"]
        torch.set_rng_state(resume["rng"][0])
        if is_cuda and resume["rng"][1] is not None:
            torch.cuda.set_rng_state_all(resume["rng"][1])
        np.random.set_state(resume["rng"][2])
        if os.path.exists(ckpt):
            best_state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        start_epoch = resume["epoch"] + 1
        logger.info("resuming from %s at epoch %d (best epoch %d, val %.4f)", last, start_epoch, best_epoch, best_val)
        del resume

    stopped_early = False
    for epoch in range(start_epoch, C.EPOCHS):
        te = time.time()
        lam = disco_lambda(epoch)
        base_model.train()
        acc = torch.zeros(4, device=device)
        nb = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                total, rec, kl, dc, _ = _loss_terms(model, batch, is_bin, lam, is_btag)
            total.backward()
            if C.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), C.GRAD_CLIP)
            optimizer.step()
            sched.step()
            acc += torch.stack([total.detach(), rec.detach(), kl.detach(), dc.detach()]).float()
            nb += 1
        tr_m = (acc / max(nb, 1)).tolist()

        base_model.eval()
        vacc = torch.zeros(4, device=device)
        with torch.no_grad():
            for b in range(n_vb):
                with amp_ctx():
                    total, rec, kl, dc, _ = _loss_terms(model, va_ds.get_batch(val_idx[b]), is_bin, lam, is_btag)
                vacc += torch.stack([total, rec, kl, dc]).float()
        va_m = (vacc / n_vb).tolist()

        hist["epoch"].append(epoch)
        for s, m in (("train", tr_m), ("val", va_m)):
            for k, v in zip(("total", "recon", "kl", "dcor"), m):
                hist[f"{s}_{k}"].append(v)
        logger.info(
            "epoch %3d  lam=%g  train tot=%.4f rec=%.4f kl=%.3f dcor=%.4f | val tot=%.4f rec=%.4f kl=%.3f dcor=%.4f | lr=%.2e %.0fs",
            epoch, lam, *tr_m, *va_m, sched.get_last_lr()[0], time.time() - te,
        )

        # Only checkpoints at the full lambda are comparable (and are what we want).
        if lam >= C.DISCO_WEIGHT:
            if va_m[0] < best_val:
                best_val, best_state, best_epoch, bad = va_m[0], deepcopy(base_model.state_dict()), epoch, 0
                _save(best_state, best_epoch, best_val)
            else:
                bad += 1
                if bad >= C.PATIENCE:
                    logger.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                    stopped_early = True
        _save_last(epoch)
        if stopped_early:
            break

    base_model.load_state_dict(best_state)
    _save(best_state, best_epoch, best_val)  # final write carries the full history
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(hist, f)
    _plot_history(hist, os.path.join(run_dir, "loss_curve.pdf"))

    # Reconstruction QA (val, best checkpoint), reusing v2_model's plotter.
    n_qa = min(200_000, len(va_ds))
    outs = []
    with torch.no_grad():
        for i in range(0, n_qa, C.BATCH_SIZE):
            idx = torch.arange(i, min(i + C.BATCH_SIZE, n_qa), device=device)
            pl, inter, hl, mask, _tgt, _tm, cond = va_ds.get_batch(idx)
            with amp_ctx():
                outs.append(base_model.decode(base_model.encode((pl, inter, hl, mask), cond), cond).float().cpu().numpy())
    x_hat = np.concatenate(outs)
    binm = D.binary_target_mask()
    x_hat[:, binm] = 1.0 / (1.0 + np.exp(-x_hat[:, binm]))
    _save_reconstruction_pdf(
        va["targets"][:n_qa], x_hat, va["tmask"][:n_qa], D.target_names(),
        os.path.join(run_dir, "reconstruction.pdf"), binary_dims=binm,
    )
    with open(done, "w") as f:
        f.write(f"best_epoch {best_epoch} best_val {best_val}\n")
    logger.info("saved %s (best epoch %d, val %.4f)", ckpt, best_epoch, best_val)
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    train(args.run_dir, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
