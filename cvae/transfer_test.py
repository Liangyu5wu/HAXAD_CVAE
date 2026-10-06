"""Generation-level test: move SB-low events to SB-high myy (and back) through the decoder.

    python -m cvae.transfer_test --run-dir $NGR_STORE/runs/<tag> [--n 50000]

SB-test events (SM Higgs removed) are split into A = 105-120 GeV and
B = 130-145 GeV. For A -> B:
  recon_A    = decode(z_A, myy_A)       A as the model sees it
  recon_B    = decode(z_B, myy_B)       target: B as the model sees it
  transfer   = decode(z_A, myy ~ B)     A's latents with myy drawn from B
The decoder outputs conditional means, so transfer is compared with recon_B
(not with the raw targets). If z were independent of myy and the decoder
carried all myy dependence, transfer and recon_B would have the same
distribution.

Reported per target: the fraction of the A -> B mean shift that the
transfer reproduces, (<transfer> - <recon_A>) / (<recon_B> - <recon_A>);
and a BDT AUC on all decoded outputs: transfer vs recon_B (0.5 = perfect),
with recon_A vs recon_B as the no-correction reference.
Writes <run-dir>/transfer_test.json and transfer_test.pdf.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from cvae import config as C
from cvae import data as D
from cvae.closure import _auc_test
from cvae.encode import load_model
from cvae.evaluate import region_frame

SHOW = ["diphoton_pt", "diphoton_delta_R", "met_pt", "n_jets", "n_bjets", "jet1_pt", "jet2_pt",
        "jet1_exist", "jet2_exist", "jet3_exist", "fatjet1_exist", "photon1_pt", "photon2_pt"]


@torch.no_grad()
def _encode(model, ds, bs=8192):
    out = []
    for i in range(0, len(ds), bs):
        idx = torch.arange(i, min(i + bs, len(ds)), device=ds.device)
        pl, it, hl, m, _t, _tm, c = ds.get_batch(idx)
        out.append(model.encode((pl, it, hl, m), c))
    return torch.cat(out)


@torch.no_grad()
def _decode(model, z, c, binm, bs=8192):
    out = []
    for i in range(0, len(z), bs):
        out.append(model.decode(z[i:i + bs], c[i:i + bs]).float().cpu().numpy())
    x = np.concatenate(out)
    x[:, binm] = 1.0 / (1.0 + np.exp(-x[:, binm]))
    return x


def run(run_dir, n=50_000, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, ck = load_model(os.path.join(run_dir, "model.pt"), device)
    norm = D.load_norm_params()
    inter = D.interaction_norm_from(norm)
    names = D.target_names()
    binm = D.binary_target_mask()
    rng = np.random.default_rng(0)

    df = region_frame("sb_test", ck["seed"], 10**9, rng)
    side = {"A": df[df["diphoton_mass"] < C.SR_LOW], "B": df[df["diphoton_mass"] > C.SR_HIGH]}
    lat, cond, w = {}, {}, {}
    for k, d in side.items():
        d = d.iloc[np.sort(rng.choice(len(d), min(n, len(d)), replace=False))]
        arr = D.build_inputs(d, norm, rotate=bool(ck["config"]["ROTATE_PHI"]))
        ds = D.CVAEDataSet(arr, device, inter)
        lat[k], cond[k], w[k] = _encode(model, ds), ds.cond.clone(), arr["weight"].astype(np.float64)
        del ds

    tag = os.path.basename(os.path.normpath(run_dir))
    out = {}
    pdf = PdfPages(os.path.join(run_dir, "transfer_test.pdf"))
    for src, dst in (("A", "B"), ("B", "A")):
        rec_src = _decode(model, lat[src], cond[src], binm)
        rec_dst = _decode(model, lat[dst], cond[dst], binm)
        draw = torch.as_tensor(rng.integers(0, len(cond[dst]), len(cond[src])), device=device)
        trans = _decode(model, lat[src], cond[dst][draw], binm)
        mean = lambda x, ww: np.average(x, axis=0, weights=ww)  # noqa: E731
        m_src, m_dst, m_tr = mean(rec_src, w[src]), mean(rec_dst, w[dst]), mean(trans, w[src])
        shift = m_dst - m_src
        frac = {nm: float((m_tr[j] - m_src[j]) / shift[j]) if abs(shift[j]) > 1e-3 else None
                for j, nm in enumerate(names)}
        X = np.concatenate([trans, rec_dst])
        y = np.r_[np.zeros(len(trans), int), np.ones(len(rec_dst), int)]
        ww = np.r_[w[src], w[dst]]
        auc_tr = _auc_test(X, y, ww, rng, 100_000)[0]
        auc_no = _auc_test(np.concatenate([rec_src, rec_dst]), y, ww, rng, 100_000)[0]  # same sizes / weights
        key = f"{src}->{dst}"
        out[key] = {"auc_transfer_vs_target": auc_tr, "auc_no_correction": auc_no,
                    "mean_shift_fraction": {k: frac[k] for k in SHOW if k in frac},
                    "mean_shift": {k: float(shift[names.index(k)]) for k in SHOW if k in names}}
        print(f"{tag} {key}: BDT AUC transfer vs target {auc_tr:.3f}  (no correction {auc_no:.3f})")
        for k in SHOW:
            if k in names and frac[k] is not None:
                print(f"    {k:16s} shift {shift[names.index(k)]:+.4f}   reproduced fraction {frac[k]:+.2f}")

        sel = [k for k in SHOW if k in names]
        fig, axes = plt.subplots(3, 5, figsize=(17, 9))
        for ax, k in zip(axes.flat, sel):
            j = names.index(k)
            lo, hi = np.percentile(np.r_[rec_src[:, j], rec_dst[:, j]], [0.5, 99.5])
            bins = np.linspace(lo, hi + 1e-6, 40)
            for x, ws, lab, col, ls in ((rec_src[:, j], w[src], f"recon {src}", "C0", "-"),
                                        (rec_dst[:, j], w[dst], f"recon {dst} (target)", "k", "-"),
                                        (trans[:, j], w[src], f"z_{src} + myy_{dst}", "C3", "--")):
                ax.hist(x, bins, weights=ws, histtype="step", density=True, color=col, ls=ls, label=lab)
            ax.set_title(f"{k}  (frac {frac[k]:+.2f})" if frac[k] is not None else k, fontsize=8)
            ax.set_yscale("log")
            ax.tick_params(labelsize=6)
        for ax in axes.flat[len(sel):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=6)
        fig.suptitle(f"{tag}  {key} (A = 105-120, B = 130-145 GeV): AUC transfer vs target {auc_tr:.3f}, "
                     f"no correction {auc_no:.3f}", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    with open(os.path.join(run_dir, "transfer_test.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(args.run_dir, args.n, args.device)


if __name__ == "__main__":
    main()
