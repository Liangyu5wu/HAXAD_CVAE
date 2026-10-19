"""Figures for the deck (slides/figs/*.png), from figs/summary.json and the run outputs.

    python slides/make_figs.py      (from the repo root, after `source env.sh` and slides/collect.py)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvae import config as C
from cvae import data as D

FIG = os.path.join(os.path.dirname(__file__), "figs")
STORE = os.path.join(os.environ.get("NGR_STORE", "store"), "runs")
S = json.load(open(os.path.join(FIG, "summary.json")))
STYLE = {"step1": ("Step 1 (no photons, no adversary)", "#7F7F7F", "s"),
         "adv0.03": ("photons + adversary λ = 0.03", "#4C72B0", "o"),
         "adv0.05": ("photons + adversary λ = 0.05", "#8C1515", "D")}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def tradeoff():
    """Closure AUCs vs reconstruction, one point per seed; tuning-scan configs as open markers."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, key, lab in ((axes[0], "auc_srsb", "closure AUC, SR vs SB"),
                         (axes[1], "auc_lohi", "closure AUC, SB-low vs SB-high")):
        scan = [S["runs"][t] for t in ["s1f_adv0p01", "s1f_adv0p1_late", "s1f_lat32", "s1f_stradv", "s1f_kl0p003",
                                       "s1e_ph_adv0p1", "s1e_ph_adv0p3"]]
        ax.scatter([r["r2_sb"] for r in scan], [r[key] for r in scan], facecolors="none", edgecolors="k",
                   s=40, lw=0.8, label="tuning scan (other configs, seed 0)")
        for g, (name, col, mk) in STYLE.items():
            rs = [S["runs"][t] for t in S["groups"][g]["tags"]]
            x, y = np.array([r["r2_sb"] for r in rs]), np.array([r[key] for r in rs])
            ax.scatter(x, y, color=col, marker=mk, s=42, label=f"{name}, {len(rs)} seeds", zorder=3)
            ax.errorbar(x.mean(), y.mean(), xerr=x.std(ddof=1), yerr=y.std(ddof=1), color=col, lw=1.6,
                        capsize=3, zorder=4)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xlabel("core R² median, SB test (reconstruction)")
        ax.set_ylabel(lab)
    axes[0].set_ylim(0.495, 0.56)
    axes[1].set_ylim(0.495, 0.68)
    axes[1].legend(fontsize=8.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tradeoff.png"), dpi=200)
    plt.close(fig)


def mu_vs_myy():
    """<mu_k>(myy) of each latent dimension, in units of its standard deviation (SB test + SR)."""
    runs = [("step1_seed0", "Step 1"), ("s1f_adv0p05", "baseline: photons + adversary λ = 0.05")]
    edges = np.linspace(C.SB_LOW, C.SB_HIGH, 17)
    cen = 0.5 * (edges[1:] + edges[:-1]) / 1e3
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, (tag, title) in zip(axes, runs):
        z = np.load(os.path.join(STORE, tag, "latents.npz"))
        keep = np.isin(z["region"], [D.SB_TEST, D.SR]) & ~np.isin(z["process_id"], D.resonant_pids())
        mu, myy, w = z["mu"][keep], z["myy"][keep], z["weight"][keep]
        mu = (mu - np.average(mu, axis=0, weights=w)) / mu.std(0)
        b = np.digitize(myy, edges) - 1
        prof = np.stack([np.average(mu[b == k], axis=0, weights=w[b == k]) for k in range(len(cen))])
        order = np.argsort(-np.abs(prof[-1] - prof[0]))
        for i, k in enumerate(order):
            ax.plot(cen, prof[:, k], color="#8C1515" if i < 3 else "#BFBFBF", lw=1.6 if i < 3 else 0.9,
                    zorder=3 if i < 3 else 2)
        ax.axvspan(C.SR_LOW / 1e3, C.SR_HIGH / 1e3, color="#F2F2F2", zorder=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    axes[0].set_ylabel(r"$\langle\mu_k\rangle\,/\,\sigma(\mu_k)$")
    axes[0].text(0.02, 0.95, "16 latent dims; the 3 most myy-dependent in red; SR shaded",
                 transform=axes[0].transAxes, fontsize=8.5, va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mu_vs_myy.png"), dpi=200)
    plt.close(fig)


def myy_response():
    """Decoded high-level targets at fixed z vs myy, compared with the data trend."""
    runs = [("step1_seed0", "Step 1", "#7F7F7F"), ("s1f_adv0p05", "baseline λ = 0.05", "#8C1515")]
    targets = ["diphoton_pt", "met_pt", "jet1_pt", "n_jets"]
    fig, axes = plt.subplots(1, len(targets), figsize=(12, 3.3))
    for j, (ax, t) in enumerate(zip(axes, targets)):
        for i, (tag, lab, col) in enumerate(runs):
            m = json.load(open(os.path.join(STORE, tag, "myy_response.json")))
            d, dec = np.array(m["response"][t]["data"]), np.array(m["response"][t]["decoded_fixed_z"])
            if i == 0:
                ax.plot(m["data_bins_gev"], d - d.mean(), "ko", ms=3, label="data: mean vs $m_{\gamma\gamma}$")
            ax.plot(m["myy_grid_gev"], dec - dec.mean(), "-", color=col, lw=1.8, label=f"{lab}: decoded, z fixed")
        ax.axvspan(C.SR_LOW / 1e3, C.SR_HIGH / 1e3, color="#F2F2F2", zorder=0)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    axes[0].set_ylabel("shift (normalized units)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIG, "myy_response.png"), dpi=200)
    plt.close(fig)


def seeds_strip():
    """Per-seed closure AUCs and R^2 for the three multi-seed configurations."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (key, lab) in zip(axes, (("auc_srsb", "AUC SR vs SB"), ("auc_lohi", "AUC SB-low vs SB-high"),
                                     ("r2_sb", "core R² SB test"))):
        for i, (g, (name, col, mk)) in enumerate(STYLE.items()):
            y = [S["runs"][t][key] for t in S["groups"][g]["tags"]]
            ax.scatter(np.full(len(y), i) + np.linspace(-0.12, 0.12, len(y)), y, color=col, marker=mk, s=30)
        ax.set_xticks(range(3), ["step 1", "λ 0.03", "λ 0.05"])
        ax.set_title(lab, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "seeds_strip.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    tradeoff()
    mu_vs_myy()
    myy_response()
    seeds_strip()
    print("figures in", FIG)
