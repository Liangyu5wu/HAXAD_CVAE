"""Diagnostic plots of a trained CVAE's latent space and reconstruction.

    python -m cvae.plots --run-dir $NGR_STORE/runs/<tag> [--n 200000]

Needs <run-dir>/latents.npz (cvae.encode). Writes <run-dir>/diagnostics.pdf
(and each page as diagnostics/pNN.png):

  1. per-dim activity: KL(q(z_i|x) || N(0,1)) and mean posterior sigma (SB test)
  2. 1D histograms of mu per dim, by split: SB train / val / test, SR
  3. 1D histograms of mu per dim, by myy: SB-low / SR / SB-high (test fold + SR)
  4. mean of mu_i vs myy (SB test + SR), the per-dim myy dependence
  5. correlation matrix of mu (SB test) and Pearson corr(mu_i, myy)
  6. aggregate posterior vs prior: sampled z (SB test) vs N(0,1), per dim
  7. reconstruction vs truth (2D) for selected targets (SB test)
  8. targets decoded from z ~ N(0,1) at the test events' myy vs the true
     targets (SB test): how far the prior is from the aggregate posterior

Histograms are event-weighted; SM Higgs (resonant) events are dropped.
"""
from __future__ import annotations

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from cvae import config as C
from cvae import data as D
from cvae.encode import load_model

logger = logging.getLogger("cvae.plots")

REGION_LABEL = {D.SB_TRAIN: "SB train", D.SB_VAL: "SB val", D.SB_TEST: "SB test", D.SR: "SR"}
RECO_TARGETS = ["jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt", "jet3_m", "fatjet1_pt",
                "fatjet1_m", "fatjet1_tau32", "diphoton_pt", "n_jets"]
MYY_BINS = np.linspace(C.SB_LOW, C.SB_HIGH, 21)


def _grid(n, ncol=4, size=3.0):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * size, nrow * size * 0.8), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def _hist(ax, x, w, bins, label, **kw):
    h, _ = np.histogram(x, bins, weights=w)
    h = h / max(h.sum() * np.diff(bins)[0], 1e-30)
    ax.stairs(h, bins, label=label, **kw)


def _save(pdf, fig, title):
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    png_dir = os.path.splitext(pdf._file.fh.name)[0]
    os.makedirs(png_dir, exist_ok=True)
    fig.savefig(os.path.join(png_dir, f"p{pdf.get_pagecount():02d}.png"), dpi=90)
    plt.close(fig)


@torch.no_grad()
def _test_pass(model, arrays, inter_norm, device, seed, batch_size=16384):
    """mu, logvar, decoded mu, decoded prior samples for one array dict."""
    ds = D.CVAEDataSet(arrays, device, inter_norm)
    g = torch.Generator(device=device).manual_seed(seed)
    mus, lv, rec, prior = [], [], [], []
    for i in range(0, len(ds), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(ds)), device=device)
        pl, it, hl, m, _t, _tm, c = ds.get_batch(idx)
        mu, logvar = model.encode_mulogvar((pl, it, hl, m), c)
        mus.append(mu.float().cpu().numpy())
        lv.append(logvar.float().cpu().numpy())
        rec.append(model.decode(mu, c).float().cpu().numpy())
        z0 = torch.randn(mu.shape, device=device, generator=g)
        prior.append(model.decode(z0, c).float().cpu().numpy())
    del ds
    return np.concatenate(mus), np.concatenate(lv), np.concatenate(rec), np.concatenate(prior)


def _decoded(x_hat, binm):
    out = x_hat.copy()
    out[:, binm] = 1.0 / (1.0 + np.exp(-x_hat[:, binm]))
    return out


def run(run_dir, n=200_000, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    lat = np.load(os.path.join(run_dir, "latents.npz"))
    model, ck = load_model(os.path.join(run_dir, "model.pt"), device)
    keep = ~np.isin(lat["process_id"], D.resonant_pids())
    mu, z, myy, w, reg = (lat[k][keep] for k in ("mu", "z", "myy", "weight", "region"))
    w = w.astype(np.float64)
    L = mu.shape[1]
    tag = os.path.basename(os.path.normpath(run_dir))

    # SB-test subsample re-encoded for logvar, reconstruction and prior decoding.
    df = D.load_splits(ck["seed"])
    df = df[(df["region"] == D.SB_TEST) & ~df["process_id"].isin(D.resonant_pids())]
    rng = np.random.default_rng(0)
    if len(df) > n:
        df = df.iloc[np.sort(rng.choice(len(df), n, replace=False))]
    norm = D.load_norm_params()
    arrays = D.build_inputs(df, norm, rotate=bool(ck["config"]["ROTATE_PHI"]))
    mu_sub, logvar, rec, prior = _test_pass(model, arrays, D.interaction_norm_from(norm), device, ck["seed"])
    binm = D.binary_target_mask()
    rec, prior = _decoded(rec, binm), _decoded(prior, binm)
    tgt, tm = arrays["targets"], arrays["tmask"].astype(bool)
    names = D.target_names()
    tw = arrays["weight"].astype(np.float64)

    test = reg == D.SB_TEST
    mu_t, w_t = mu[test], w[test]
    sig2 = np.exp(logvar)
    out = os.path.join(run_dir, "diagnostics.pdf")
    with PdfPages(out) as pdf:
        # 1. activity --------------------------------------------------------
        kl = 0.5 * (mu_sub**2 + sig2 - logvar - 1.0)
        kl_dim = np.average(kl, axis=0, weights=tw)
        sig_dim = np.average(np.sqrt(sig2), axis=0, weights=tw)
        order = np.argsort(-kl_dim)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.bar(range(L), kl_dim[order])
        a1.set_xticks(range(L), [f"z{i}" for i in order], fontsize=7)
        a1.set_ylabel("KL per dim [nats]")
        a1.set_title(f"total KL {kl_dim.sum():.1f} nats")
        a2.bar(range(L), sig_dim[order])
        a2.set_xticks(range(L), [f"z{i}" for i in order], fontsize=7)
        a2.set_ylabel("mean posterior σ")
        a2.axhline(1, color="k", lw=0.8, ls="--")
        a2.set_title("σ ≈ 1 → inactive (posterior = prior)")
        _save(pdf, fig, f"{tag}: latent activity (SB test, sorted by KL)")

        # 2. mu by split -----------------------------------------------------
        fig, axes = _grid(L)
        for i, ax in enumerate(axes):
            if i >= L:
                break
            lo, hi = np.percentile(mu[:, i], [0.1, 99.9])
            bins = np.linspace(lo, hi, 60)
            for r, ls in ((D.SB_TRAIN, "-"), (D.SB_VAL, "--"), (D.SB_TEST, ":"), (D.SR, "-")):
                s = reg == r
                _hist(ax, mu[s, i], w[s], bins, REGION_LABEL[r], ls=ls, lw=1.2,
                      color="C3" if r == D.SR else None)
            ax.set_title(f"μ{i}  (KL {kl_dim[i]:.2f})", fontsize=8)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.legend(fontsize=6)
        _save(pdf, fig, f"{tag}: μ per dim by split (unit area)")

        # 3. mu by myy region (test fold + SR) ---------------------------------
        groups = {"SB low (105-120)": test & (myy < C.SR_LOW), "SR (120-130)": reg == D.SR,
                  "SB high (130-145)": test & (myy > C.SR_HIGH)}
        fig, axes = _grid(L)
        for i, ax in enumerate(axes):
            if i >= L:
                break
            lo, hi = np.percentile(mu[:, i], [0.1, 99.9])
            bins = np.linspace(lo, hi, 60)
            for (lab, s), col in zip(groups.items(), ("C0", "C3", "C2")):
                _hist(ax, mu[s, i], w[s], bins, lab, color=col, lw=1.2)
            ax.set_title(f"μ{i}", fontsize=8)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.legend(fontsize=6)
        _save(pdf, fig, f"{tag}: μ per dim by myy region (SB test fold, SR)")

        # 4. <mu_i> vs myy -----------------------------------------------------
        use = test | (reg == D.SR)
        ib = np.digitize(myy[use], MYY_BINS) - 1
        ok = (ib >= 0) & (ib < len(MYY_BINS) - 1)
        cen = 0.5 * (MYY_BINS[1:] + MYY_BINS[:-1]) / 1e3
        mu_u, w_u = mu[use][ok], w[use][ok]
        ib = ib[ok]
        fig, axes = _grid(L)
        for i, ax in enumerate(axes):
            if i >= L:
                break
            m = np.array([np.average(mu_u[ib == b, i], weights=w_u[ib == b]) for b in range(len(cen))])
            sd = np.array([np.sqrt(np.average((mu_u[ib == b, i] - m[b]) ** 2, weights=w_u[ib == b]))
                           for b in range(len(cen))])
            ax.plot(cen, (m - m.mean()) / sd.mean(), "o-", ms=2)
            ax.axvspan(C.SR_LOW / 1e3, C.SR_HIGH / 1e3, color="grey", alpha=0.15)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"μ{i}", fontsize=8)
            ax.tick_params(labelsize=6)
        _save(pdf, fig, f"{tag}: (⟨μ_i⟩(myy) − mean) / σ_i  vs myy [GeV]  (SB test + SR; grey = SR)")

        # 5. correlations ------------------------------------------------------
        X = np.column_stack([mu_t, myy[test]])
        cw = np.cov(X.T, aweights=w_t)
        d = np.sqrt(np.diag(cw))
        corr = cw / np.outer(d, d)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})
        im = a1.imshow(corr[:L, :L], cmap="RdBu_r", vmin=-1, vmax=1)
        a1.set_xticks(range(L), [f"{i}" for i in range(L)], fontsize=7)
        a1.set_yticks(range(L), [f"{i}" for i in range(L)], fontsize=7)
        a1.set_title("corr(μ_i, μ_j)")
        fig.colorbar(im, ax=a1, fraction=0.046)
        a2.bar(range(L), corr[:L, L])
        a2.set_xticks(range(L), [f"z{i}" for i in range(L)], fontsize=7)
        a2.axhline(0, color="k", lw=0.6)
        a2.set_title("Pearson corr(μ_i, myy)")
        _save(pdf, fig, f"{tag}: latent correlations (SB test)")

        # 6. aggregate posterior vs prior ---------------------------------------
        fig, axes = _grid(L)
        g = np.linspace(-4, 4, 200)
        for i, ax in enumerate(axes):
            if i >= L:
                break
            bins = np.linspace(-5, 5, 80)
            _hist(ax, z[test, i], w_t, bins, "z ~ q(z|x), SB test", color="C0", lw=1.2)
            ax.plot(g, np.exp(-g**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=0.8, label="N(0,1)")
            ax.set_title(f"z{i}  std {np.sqrt(np.average(z[test, i]**2, weights=w_t)):.2f}", fontsize=8)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.legend(fontsize=6)
        _save(pdf, fig, f"{tag}: aggregate posterior vs prior")

        # 7. reconstruction vs truth -------------------------------------------
        sel = [names.index(t) for t in RECO_TARGETS if t in names]
        fig, axes = _grid(len(sel), ncol=5, size=2.8)
        for ax, j in zip(axes, sel):
            v = tm[:, j]
            y, yh = tgt[v, j], rec[v, j]
            lo, hi = np.percentile(y, [0.5, 99.5])
            ax.hist2d(y, yh, bins=60, range=[[lo, hi], [lo, hi]], weights=tw[v], cmap="viridis",
                      norm=matplotlib.colors.LogNorm())
            ax.plot([lo, hi], [lo, hi], "w--", lw=0.6)
            r2 = 1 - np.mean((yh - y) ** 2) / max(np.var(y), 1e-12)
            ax.set_title(f"{names[j]}  R²={r2:.2f}", fontsize=8)
            ax.set_xlabel("true", fontsize=7)
            ax.set_ylabel("recon", fontsize=7)
            ax.tick_params(labelsize=6)
        _save(pdf, fig, f"{tag}: reconstruction vs truth (SB test, normalized units)")

        # 8. prior decoding ------------------------------------------------------
        cont = [j for j in range(len(names)) if not binm[j]]
        for s in range(0, len(cont), 16):
            page = cont[s:s + 16]
            fig, axes = _grid(len(page))
            for ax, j in zip(axes, page):
                v = tm[:, j]
                lo, hi = np.percentile(tgt[v, j], [0.5, 99.5])
                bins = np.linspace(lo, hi, 50)
                _hist(ax, tgt[v, j], tw[v], bins, "true", color="k", lw=1.2)
                _hist(ax, rec[v, j], tw[v], bins, "recon (μ)", color="C0", lw=1.0)
                _hist(ax, prior[v, j], tw[v], bins, "decoded z~N(0,1)", color="C3", lw=1.0)
                ax.set_title(names[j], fontsize=8)
                ax.set_yscale("log")
                ax.tick_params(labelsize=6)
            axes[0].legend(fontsize=6)
            _save(pdf, fig, f"{tag}: true vs recon vs prior-decoded (SB test; prior uses the event's own myy)")
        fig, ax = plt.subplots(figsize=(10, 4))
        bj = np.flatnonzero(binm)
        ax.plot(range(len(bj)), np.average(tgt[:, bj], axis=0, weights=tw), "ko", label="true")
        ax.plot(range(len(bj)), np.average(rec[:, bj], axis=0, weights=tw), "C0x", label="recon (μ)")
        ax.plot(range(len(bj)), np.average(prior[:, bj], axis=0, weights=tw), "C3^", label="decoded z~N(0,1)")
        ax.set_xticks(range(len(bj)), [names[j] for j in bj], rotation=60, fontsize=7)
        ax.set_ylabel("mean (rate)")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        _save(pdf, fig, f"{tag}: binary targets (exist, btag), mean over SB test (btag over all slots incl. absent)")
    logger.info("saved %s", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(args.run_dir, args.n, args.device)


if __name__ == "__main__":
    main()
