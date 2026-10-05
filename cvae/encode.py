"""Encode every SB (train / val / test fold) and SR pseudo-data event with a trained CVAE.

    python -m cvae.encode --run-dir $NGR_STORE/runs/<tag>

Writes ``<run-dir>/latents.npz`` with mu (deterministic latent), a posterior
sample z, and myy / event_weight / process_id / event_number / region code.
The architecture and preprocessing (phi rotation) are rebuilt from the
checkpoint's own config, not from the current env.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch

from cvae import config as C
from cvae import data as D
from cvae.model import ParTCVAE

logger = logging.getLogger("cvae.encode")


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    # The slot layout is fixed at import time from the env (data.py); it must
    # match the checkpoint's.
    if cfg.get("PHOTON_SLOTS", False) != C.PHOTON_SLOTS:
        raise RuntimeError(f"{ckpt_path} was trained with PHOTON_SLOTS={cfg.get('PHOTON_SLOTS', False)}: "
                           f"set CVAE_PHOTON_SLOTS={int(cfg.get('PHOTON_SLOTS', False))}")
    model = ParTCVAE(
        n_targets=ck["n_targets"],
        latent_dim=cfg["LATENT_DIM"],
        dec_hidden=tuple(cfg["DEC_HIDDEN"]),
        cond_embed_dim=cfg["COND_EMBED_DIM"],
        cond_embed=cfg.get("COND_EMBED", "mlp"),
        encoder_kwargs=cfg["ENCODER_KWARGS"],
        enc_cond=cfg.get("ENC_COND", False),  # absent in pre-ENC_COND checkpoints
        dec_learn_var=cfg.get("DEC_LEARN_VAR", False),
        decoder=cfg.get("DECODER", "mlp"),
        myy_head=cfg.get("MYY_HEAD", False),
        dec_myy=cfg.get("DEC_MYY", "true"),
        adversary=cfg.get("ADV_WEIGHT", 0.0) > 0,
    )
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def encode_arrays(model, arrays, inter_norm, device, batch_size=16384, seed=0):
    ds = D.CVAEDataSet(arrays, device, inter_norm)
    torch.manual_seed(seed)
    mus, zs, ms = [], [], []
    for i in range(0, len(ds), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(ds)), device=device)
        pl, inter, hl, mask, _tgt, _tm, cond = ds.get_batch(idx)
        m_hat, mu, logvar = model.encode_full((pl, inter, hl, mask), cond)
        mus.append(mu.float().cpu().numpy())
        zs.append(model.encoder.reparameterize(mu, logvar).float().cpu().numpy())
        ms.append(m_hat[:, 0].float().cpu().numpy() if m_hat is not None else np.full(len(mu), np.nan, np.float32))
    del ds
    return np.concatenate(mus), np.concatenate(zs), np.concatenate(ms)


def encode_run(run_dir, device="cuda", chunk=1_000_000):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, ck = load_model(os.path.join(run_dir, "model.pt"), device)
    df = D.load_splits(ck["seed"])  # same inputs + seed -> same k-fold split as training
    norm_params = D.load_norm_params()
    inter_norm = D.interaction_norm_from(norm_params)
    rotate = bool(ck["config"]["ROTATE_PHI"])

    mus, zs, ms = [], [], []
    for s in range(0, len(df), chunk):
        arr = D.build_inputs(df.iloc[s:s + chunk], norm_params, rotate=rotate)
        mu, z, m = encode_arrays(model, arr, inter_norm, device, seed=ck["seed"] + s)
        mus.append(mu)
        zs.append(z)
        ms.append(m)
        logger.info("encoded %d / %d", min(s + chunk, len(df)), len(df))

    out = os.path.join(run_dir, "latents.npz")
    np.savez(
        out,
        mu=np.concatenate(mus),
        z=np.concatenate(zs),
        m_hat=np.concatenate(ms),  # fixed myy dim, in c = (myy - 125 GeV) / 20 GeV units (NaN without MYY_HEAD)
        myy=df["diphoton_mass"].to_numpy(np.float32),
        weight=df["event_weight"].to_numpy(np.float32),
        process_id=df["process_id"].to_numpy(np.int64),
        event_number=df["event_number"].to_numpy(),
        region=df["region"].to_numpy(),
    )
    logger.info("saved %s", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    encode_run(args.run_dir, device=args.device)


if __name__ == "__main__":
    main()
