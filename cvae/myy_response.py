"""How much does the decoder use its myy input?

    python -m cvae.myy_response --run-dir $NGR_STORE/runs/<tag> [--n 100000]

On SB-test events (SM Higgs removed):
  1. shuffle test: reconstruction R^2 with each event's own myy vs with myy
     permuted across events (z unchanged). The drop is what myy contributes
     to the reconstruction beyond z.
  2. response: keep every event's z (= mu) fixed, decode at myy = 105 ... 145
     GeV, and compare the mean decoded high-level targets with the data's
     mean in myy bins (SB test + SR).
Writes <run-dir>/myy_response.json and myy_response.pdf.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cvae import config as C
from cvae import data as D
from cvae.encode import load_model
from cvae.evaluate import metrics, region_frame

RESPONSE_TARGETS = ["diphoton_pt", "diphoton_delta_R", "met_pt", "jet1_pt", "n_jets"]
MYY_GRID = np.linspace(C.SB_LOW, C.SB_HIGH, 9)


@torch.no_grad()
def _encode_decode(model, ds, cond_override=None, bs=16384):
    mus, outs = [], []
    for i in range(0, len(ds), bs):
        idx = torch.arange(i, min(i + bs, len(ds)), device=ds.device)
        pl, it, hl, m, _t, _tm, c = ds.get_batch(idx)
        mu = model.encode((pl, it, hl, m), c)
        cc = c if cond_override is None else cond_override[idx]
        outs.append(model.decode(mu, cc).float().cpu().numpy())
        mus.append(mu)
    return torch.cat(mus), np.concatenate(outs)


@torch.no_grad()
def _decode_at(model, mu, c_value, bs=16384):
    outs = []
    for i in range(0, len(mu), bs):
        z = mu[i:i + bs]
        c = torch.full((len(z), 1), float(c_value), device=z.device)
        outs.append(model.decode(z, c).float().cpu().numpy())
    return np.concatenate(outs)


def _decoded_myy(x_hat, names, norm):
    """myy (GeV) computed from the decoded photon slots (PHOTON_SLOTS models)."""
    pp = norm["particle_level"]["pt"]
    j = {f: names.index(f) for f in names if f.startswith("photon")}
    pt = [np.exp(x_hat[:, j[f"photon{i}_pt"]] * pp["std"] + pp["mean"]) for i in (1, 2)]
    eta = [x_hat[:, j[f"photon{i}_eta"]] for i in (1, 2)]
    phi = [np.arctan2(x_hat[:, j[f"photon{i}_sinphi"]], x_hat[:, j[f"photon{i}_cosphi"]]) for i in (1, 2)]
    m2 = 2 * pt[0] * pt[1] * (np.cosh(eta[0] - eta[1]) - np.cos(phi[0] - phi[1]))
    return np.sqrt(np.clip(m2, 0, None)) / 1e3


def run(run_dir, n=100_000, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, ck = load_model(os.path.join(run_dir, "model.pt"), device)
    norm = D.load_norm_params()
    inter = D.interaction_norm_from(norm)
    rotate = bool(ck["config"]["ROTATE_PHI"])
    names = D.target_names()
    rng = np.random.default_rng(0)

    df = region_frame("sb_test", ck["seed"], n, rng)
    arr = D.build_inputs(df, norm, rotate=rotate)
    ds = D.CVAEDataSet(arr, device, inter)

    # 1. shuffle test
    mu, x_true = _encode_decode(model, ds)
    perm = torch.as_tensor(rng.permutation(len(df)), device=device)
    _, x_shuf = _encode_decode(model, ds, cond_override=ds.cond[perm])
    rows_t, s_true = metrics(arr, x_true)
    rows_s, s_shuf = metrics(arr, x_shuf)
    out = {"core_r2_median_true_myy": s_true["core_r2_median"],
           "core_r2_median_shuffled_myy": s_shuf["core_r2_median"],
           "r2_true_vs_shuffled": {k: (rows_t[k]["r2"], rows_s[k]["r2"]) for k in RESPONSE_TARGETS + ["jet2_pt", "jet3_pt"]
                                   if k in rows_t and "r2" in rows_t[k]}}

    if "photon1_pt" in names:
        # Which myy do the decoded photons carry: the one given to the decoder,
        # or the event's true one (which can only come through z)?
        true = arr["myy"] / 1e3
        given_shuf = true[perm.cpu().numpy()]
        m_true_in = _decoded_myy(x_true, names, norm)
        m_shuf_in = _decoded_myy(x_shuf, names, norm)
        cc = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
        out["decoded_photon_myy"] = {
            "given_true: corr(decoded, true)": cc(m_true_in, true),
            "given_shuffled: corr(decoded, given)": cc(m_shuf_in, given_shuf),
            "given_shuffled: corr(decoded, true)": cc(m_shuf_in, true),
            "given_true: rms(decoded - true) GeV": float(np.sqrt(np.mean((m_true_in - true) ** 2))),
        }

    # 2. response at fixed z vs data
    cols = [names.index(t) for t in RESPONSE_TARGETS]
    dec = np.stack([_decode_at(model, mu, D.myy_cond(np.array([m]))[0, 0])[:, cols].mean(0) for m in MYY_GRID])
    dd = region_frame("sb_test", ck["seed"], 10**9, rng)
    sr = region_frame("sr", ck["seed"], 10**9, rng)
    both = pd.concat([dd, sr])
    both = D.build_inputs(both.sample(min(400_000, len(both)), random_state=0), norm, rotate=rotate)
    edges = np.linspace(C.SB_LOW, C.SB_HIGH, 17)
    cen = 0.5 * (edges[1:] + edges[:-1])
    b = np.digitize(both["myy"], edges) - 1
    data = np.stack([[both["targets"][b == k, j].mean() for j in cols] for k in range(len(cen))])
    out["response"] = {t: {"decoded_fixed_z": dec[:, i].tolist(), "data": data[:, i].tolist()}
                       for i, t in enumerate(RESPONSE_TARGETS)}
    out["myy_grid_gev"] = (MYY_GRID / 1e3).tolist()
    out["data_bins_gev"] = (cen / 1e3).tolist()

    fig, axes = plt.subplots(1, len(RESPONSE_TARGETS), figsize=(4 * len(RESPONSE_TARGETS), 3.4))
    for i, (ax, t) in enumerate(zip(axes, RESPONSE_TARGETS)):
        ax.plot(cen / 1e3, data[:, i] - data[:, i].mean(), "ko", ms=3, label="data ⟨x⟩(myy)")
        ax.plot(MYY_GRID / 1e3, dec[:, i] - dec[:, i].mean(), "C3-", label="decoded, z fixed")
        ax.axvspan(C.SR_LOW / 1e3, C.SR_HIGH / 1e3, color="grey", alpha=0.15)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("myy [GeV]")
    axes[0].set_ylabel("mean − overall mean (normalized units)")
    axes[0].legend(fontsize=7)
    tag = os.path.basename(os.path.normpath(run_dir))
    fig.suptitle(f"{tag}: myy response   core R² true myy {s_true['core_r2_median']:.3f}, "
                 f"shuffled myy {s_shuf['core_r2_median']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "myy_response.pdf"))
    plt.close(fig)
    with open(os.path.join(run_dir, "myy_response.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{tag}: core R2 median true myy {s_true['core_r2_median']:.4f}  shuffled {s_shuf['core_r2_median']:.4f}")
    for t, (a, b_) in out["r2_true_vs_shuffled"].items():
        print(f"  {t:18s} R2 true {a:.4f}  shuffled {b_:.4f}")
    for k, v in out.get("decoded_photon_myy", {}).items():
        print(f"  myy from decoded photons, {k}: {v:.3f}")
    for i, t in enumerate(RESPONSE_TARGETS):
        print(f"  {t:18s} 105->145 GeV change: decoded(z fixed) {dec[-1, i] - dec[0, i]:+.4f}   "
              f"data {data[-1, i] - data[0, i]:+.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(args.run_dir, args.n, args.device)


if __name__ == "__main__":
    main()
