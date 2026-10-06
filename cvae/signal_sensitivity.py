"""Does the latent still separate signals from the SR background?

    python -m cvae.signal_sensitivity --run-dir $NGR_STORE/runs/<tag> [--n 100000]
    python -m cvae.signal_sensitivity --run-dir ... --features raw   # reference on raw inputs

For each signal (MC pool, 120-130 GeV) a BDT (same k-fold HistGradientBoosting
as the closure test) separates signal mu from SR pseudo-data background mu
(SM Higgs removed). AUC near 1 = the latent keeps what distinguishes the
signal; 0.5 = the information is gone. Writes <run-dir>/signal_sensitivity[_raw].json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from cvae import config as C
from cvae import data as D
from cvae.closure import RAW_FEATURES, _auc_test
from cvae.encode import encode_arrays, load_model
from cvae.evaluate import load_signal, region_frame

SIGNALS = ["XHH_500", "HVT_VcXjjHyy_2000_300", "WN_HyyN_300", "TT_tZNtHyyN_1000_205_60"]


def run(run_dir, n=100_000, features="latent", device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    rng = np.random.default_rng(0)
    seed = json.load(open(os.path.join(run_dir, "config.json")))["seed"]
    bkg = region_frame("sr", seed, n, rng)
    frames = {"bkg": bkg}
    for s in SIGNALS:
        df = load_signal(s)
        df = df[(df["diphoton_mass"] >= C.SR_LOW) & (df["diphoton_mass"] <= C.SR_HIGH)]
        frames[s] = df.iloc[np.sort(rng.choice(len(df), min(n, len(df)), replace=False))]

    if features == "raw":
        X = {k: f[RAW_FEATURES].to_numpy(np.float64) for k, f in frames.items()}
    else:
        model, ck = load_model(os.path.join(run_dir, "model.pt"), device)
        norm = D.load_norm_params()
        inter = D.interaction_norm_from(norm)
        X = {}
        for k, f in frames.items():
            arr = D.build_inputs(f, norm, rotate=bool(ck["config"]["ROTATE_PHI"]))
            X[k] = encode_arrays(model, arr, inter, device)[0].astype(np.float64)

    wb = frames["bkg"]["event_weight"].to_numpy(np.float64)
    out = {}
    for s in SIGNALS:
        ws = frames[s]["event_weight"].to_numpy(np.float64)
        Xa = np.concatenate([X["bkg"], X[s]])
        y = np.r_[np.zeros(len(X["bkg"]), int), np.ones(len(X[s]), int)]
        auc, sd, _ = _auc_test(Xa, y, np.r_[wb, ws], rng, n)
        out[s] = {"auc": auc, "std": sd, "n_signal": int(len(X[s]))}
        print(f"{os.path.basename(os.path.normpath(run_dir))} [{features}] {s:26s} AUC {auc:.4f} ± {sd:.4f} (n_sig {len(X[s])})")
    name = "signal_sensitivity" + ("_raw" if features == "raw" else "") + ".json"
    with open(os.path.join(run_dir, name), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--features", default="latent", choices=["latent", "raw"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(args.run_dir, args.n, args.features, args.device)


if __name__ == "__main__":
    main()
