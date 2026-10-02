"""Closure tests for the "SR template = SB latents" assumption (pseudo-data, no injection).

    python -m cvae.closure --run-dir $NGR_STORE/runs/<tag> [--latent mu|z]
    python -m cvae.closure --run-dir ... --features raw    # same tests on raw inputs (reference)

Everything uses events the CVAE never saw: SB_TEST (the held-out k-fold) and SR.
SM Higgs events (resbkg processes; in the pipeline they enter the template via
ResMCEvents) are removed from both, so the tests probe the continuum only.

1. dCor^2(latent, myy) on SB_TEST and SB_TEST ∪ SR, with a permuted-myy null.
2. Classifier AUCs (HistGradientBoosting, k-fold, event-weighted):
     sb_low_vs_high : SB_TEST below vs above the SR  -> 0.5 iff no myy info
     sr_vs_sb       : SR vs SB_TEST                  -> 0.5 iff the SB latent
                      distribution is a valid SR background template
3. Sculpting: for top-quantile cuts on each latent dim and on |latent|, the
   pass fraction vs myy (should be flat) and eps_SR / eps_SB (should be 1).

Writes <run-dir>/closure_<tag>.json and closure_<tag>.pdf.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cvae import config as C
from cvae import data as D
from cvae.losses import distance_corr

logger = logging.getLogger("cvae.closure")

RAW_FEATURES = ["diphoton_pt", "diphoton_delta_R", "met_pt", "n_jets", "n_bjets", "n_leptons",
                "jet1_pt", "jet2_pt", "jet1_eta", "fatjet1_pt"]
QUANTILES = (0.5, 0.9, 0.99)
MYY_BINS = np.linspace(C.SB_LOW, C.SB_HIGH, 21)


def _dcor(x, y, w, rng, n=8000, reps=5):
    p = w / w.sum()
    vals = []
    for _ in range(reps):
        i = rng.choice(len(x), n, replace=True, p=p)
        vals.append(float(distance_corr(torch.as_tensor(x[i]), torch.as_tensor(y[i, None]))))
    return float(np.mean(vals)), float(np.std(vals))


def _auc_test(X, y, w, rng, max_per_class=300_000, folds=3):
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y == c), min(max_per_class, int((y == c).sum())), replace=False) for c in (0, 1)
    ])
    X, y, w = X[idx], y[idx], w[idx]
    # Balance classes in the fit; keep physical weights within each class.
    wfit = w / np.array([w[y == 0].sum(), w[y == 1].sum()])[y]
    aucs = []
    for k, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y)):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True,
                                             validation_fraction=0.1, random_state=k)
        clf.fit(X[tr], y[tr], sample_weight=wfit[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1], sample_weight=w[te]))
    return float(np.mean(aucs)), float(np.std(aucs)), int(len(y))


def _sculpt(X, names, myy, w, region, pdf):
    sb = region == D.SB_TEST
    use = sb | (region == D.SR)
    is_sr = region == D.SR
    scores = {n: X[:, i] for i, n in enumerate(names)}
    scores["|latent|"] = np.linalg.norm(X, axis=1)
    out = {}
    tot, _ = np.histogram(myy[use], MYY_BINS, weights=w[use])
    centers = 0.5 * (MYY_BINS[1:] + MYY_BINS[:-1]) / 1e3
    for name, s in scores.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        out[name] = {}
        for q in QUANTILES:
            thr = np.quantile(s[sb], q)  # threshold fixed in the SB, like a real analysis
            ps = s > thr
            num, _ = np.histogram(myy[use & ps], MYY_BINS, weights=w[use & ps])
            frac = num / np.clip(tot, 1e-12, None)
            eps_sr = w[is_sr & ps].sum() / w[is_sr].sum()
            eps_sb = w[sb & ps].sum() / w[sb].sum()
            # chi2 of pass fraction vs a constant (binomial errors from raw counts)
            n_raw, _ = np.histogram(myy[use], MYY_BINS)
            err = np.sqrt(np.clip(frac * (1 - frac), 1e-12, None) / np.clip(n_raw, 1, None))
            c = np.average(frac, weights=1 / err**2)
            chi2 = float(np.sum(((frac - c) / err) ** 2))
            out[name][str(q)] = {"eps_sr_over_sb": float(eps_sr / eps_sb), "chi2_flat": chi2, "ndf": len(frac) - 1}
            ax.errorbar(centers, frac / c, yerr=err / c, fmt="o-", ms=3,
                        label=f"top {100 * (1 - q):.0f}%: εSR/εSB={eps_sr / eps_sb:.3f}, χ²={chi2:.0f}/{len(frac) - 1}")
        ax.axvspan(C.SR_LOW / 1e3, C.SR_HIGH / 1e3, color="grey", alpha=0.15)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xlabel("myy [GeV]")
        ax.set_ylabel("pass fraction / mean")
        ax.set_title(f"sculpting: cut on {name}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    return out


def run(run_dir, latent="mu", features=None, seed=0):
    rng = np.random.default_rng(seed)
    if features == "raw":
        with open(os.path.join(run_dir, "config.json")) as fh:
            df = D.load_splits(json.load(fh)["seed"])
        X = df[RAW_FEATURES].to_numpy(np.float64)
        names, myy, w = RAW_FEATURES, df["diphoton_mass"].to_numpy(np.float64), df["event_weight"].to_numpy(np.float64)
        region, pid = df["region"].to_numpy(), df["process_id"].to_numpy()
        tag = "raw"
    else:
        f = np.load(os.path.join(run_dir, "latents.npz"))
        X = f[latent].astype(np.float64)
        names = [f"latent_{i}" for i in range(X.shape[1])]
        myy, w, region, pid = f["myy"].astype(np.float64), f["weight"].astype(np.float64), f["region"], f["process_id"]
        tag = latent

    keep = ~np.isin(pid, D.resonant_pids())
    X, myy, w, region = X[keep], myy[keep], w[keep], region[keep]
    sb = region == D.SB_TEST
    sr = region == D.SR
    both = sb | sr
    cond = D.myy_cond(myy)[:, 0].astype(np.float64)
    Xs = (X - X[sb].mean(0)) / X[sb].std(0).clip(1e-12)

    res = {"n_sb_test": int(sb.sum()), "n_sr": int(sr.sum()), "latent": tag}
    res["dcor_sb_test"] = _dcor(Xs[sb], cond[sb], w[sb], rng)
    res["dcor_sb_test_sr"] = _dcor(Xs[both], cond[both], w[both], rng)
    res["dcor_null"] = _dcor(Xs[both], rng.permutation(cond[both]), w[both], rng)
    lo_hi = myy[sb] > C.SR_HIGH
    res["auc_sb_low_vs_high"] = _auc_test(Xs[sb], lo_hi.astype(int), w[sb], rng)
    res["auc_sr_vs_sb"] = _auc_test(Xs[both], sr[both].astype(int), w[both], rng)
    logger.info("dCor^2 SB_TEST=%.4f±%.4f  SB∪SR=%.4f±%.4f  null=%.4f±%.4f",
                *res["dcor_sb_test"], *res["dcor_sb_test_sr"], *res["dcor_null"])
    logger.info("AUC sb_low_vs_high=%.4f±%.4f (n=%d)   sr_vs_sb=%.4f±%.4f (n=%d)",
                *res["auc_sb_low_vs_high"], *res["auc_sr_vs_sb"])

    with PdfPages(os.path.join(run_dir, f"closure_{tag}.pdf")) as pdf:
        res["sculpting"] = _sculpt(Xs, names, myy, w, region, pdf)
    with open(os.path.join(run_dir, f"closure_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--latent", default="mu", choices=["mu", "z"])
    ap.add_argument("--features", default=None, choices=[None, "raw"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(args.run_dir, latent=args.latent, features=args.features)


if __name__ == "__main__":
    main()
