"""Per-target reconstruction quality of a trained CVAE.

    python -m cvae.evaluate --run-dir $NGR_STORE/runs/<tag> \\
        [--regions sb_test,sr,signal:XHH_500,...] [--n 100000]

For each region: R^2 (continuous targets, over events where the object
exists) and accuracy + ROC AUC (binary targets: exist flags, btag), plus
summaries. Writes <run-dir>/eval_<region>.txt and eval.json.

Regions
  sb_val, sb_test : held-out SB folds (sb_test was never used, not even for
                    early stopping)
  sr              : SR pseudo-data. The decoder only ever saw SB myy values,
                    so this measures its interpolation in myy.
  signal:<name>   : signal MC from the MC pool (NormalizeMC output of the
                    contrastive branch, process <name>, 105-145 GeV). Shows
                    how well background-trained reconstruction carries over.
SM Higgs (resonant) events are dropped from the background regions.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.data.data import Data

from cvae import config as C
from cvae import data as D
from cvae.encode import load_model

logger = logging.getLogger("cvae.evaluate")

DEFAULT_REGIONS = "sb_test,sr,signal:XHH_500,signal:HVT_VcXjjHyy_2000_300,signal:WN_HyyN_300,signal:TT_tZNtHyyN_1000_205_60"
# Summary set: jet and leading-fatjet kinematics (leptons / fatjet2 are almost
# never present in the SB background, their R^2 is not meaningful there).
CORE_PREFIXES = ("jet", "fatjet1")  # photon targets are reported but not in the summary


def load_signal(name):
    """Signal MC events of process ``name`` from the MC pool, cached as parquet."""
    from src.tasks.selection import NormalizeMC

    cache = os.path.join(C.NGR_STORE, "cache", C.VERSION, "signals", f"{name}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    task = NormalizeMC(version=C.VERSION, encoder_type=C.PIPELINE_ENCODER_TYPE)
    pid = task.analysis.processes.get(name).id
    data = Data(path=task.output()["normalized_mc"].path)

    def _keep(df):
        m = df["diphoton_mass"].to_numpy()
        return df.loc[(df["process_id"].to_numpy() == pid) & (m >= C.SB_LOW) & (m <= C.SB_HIGH), D.LOAD_COLS]

    data.load_filtered(_keep)
    df = data.all.reset_index(drop=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_parquet(cache, index=False)
    logger.info("signal %s (pid %d): %d events -> %s", name, pid, len(df), cache)
    return df


def region_frame(region, seed, n, rng):
    if region.startswith("signal:"):
        df = load_signal(region.split(":", 1)[1])
    else:
        code = {"sb_val": D.SB_VAL, "sb_test": D.SB_TEST, "sr": D.SR}[region]
        df = D.load_splits(seed)
        df = df[(df["region"] == code) & ~df["process_id"].isin(D.resonant_pids())]
    if len(df) > n:
        df = df.iloc[np.sort(rng.choice(len(df), n, replace=False))]
    return df


@torch.no_grad()
def reconstruct(model, arrays, inter_norm, device, batch_size=16384):
    ds = D.CVAEDataSet(arrays, device, inter_norm)
    outs, mh = [], []
    for i in range(0, len(ds), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(ds)), device=device)
        pl, it, hl, m, _t, _tm, c = ds.get_batch(idx)
        x_hat, _z, _mu, _lv, m_hat = model((pl, it, hl, m), c)
        outs.append(x_hat.float().cpu().numpy())
        if m_hat is not None:
            mh.append(m_hat[:, 0].float().cpu().numpy())
    del ds
    return np.concatenate(outs), (np.concatenate(mh) if mh else None)


def metrics(arrays, x_hat):
    tgt, tm = arrays["targets"], arrays["tmask"].astype(bool)
    binm = D.binary_target_mask()
    rows = {}
    for j, name in enumerate(D.target_names()):
        v = tm[:, j]
        if v.sum() < 50:
            continue
        y = tgt[v, j]
        if binm[j]:
            p = 1.0 / (1.0 + np.exp(-x_hat[v, j]))
            r = {"acc": float(((p > 0.5) == (y > 0.5)).mean()), "pos_frac": float(y.mean())}
            if 0 < y.mean() < 1:
                r["auc"] = float(roc_auc_score(y, p))
        else:
            r = {"r2": float(1.0 - ((x_hat[v, j] - y) ** 2).mean() / (y.var() + 1e-12))}
        r["valid_frac"] = float(v.mean())
        rows[name] = r
    core = [r["r2"] for n, r in rows.items() if "r2" in r and n.startswith(CORE_PREFIXES)]
    btag_auc = [r["auc"] for n, r in rows.items() if n.endswith("_btag") and "auc" in r]
    summary = {
        "n": int(len(tgt)),
        "core_r2_median": float(np.median(core)) if core else float("nan"),
        "core_r2_min": float(np.min(core)) if core else float("nan"),
        "btag_auc_min": float(np.min(btag_auc)) if btag_auc else float("nan"),
    }
    return rows, summary


def _fmt(region, rows, summary):
    extra = (f"  myy dim R2 {summary['myy_head_r2']:.4f} (rms {summary['myy_head_rms_gev']:.2f} GeV)"
             if "myy_head_r2" in summary else "")
    lines = [f"===== {region}  n={summary['n']}  core R2 median {summary['core_r2_median']:.3f} "
             f"min {summary['core_r2_min']:.3f}  btag AUC min {summary['btag_auc_min']:.3f}{extra}",
             f"{'target':22s} {'R2/acc':>7s} {'auc':>7s} {'valid':>7s}"]
    for n, r in rows.items():
        main = r.get("r2", r.get("acc"))
        auc = f"{r['auc']:7.3f}" if "auc" in r else f"{'':7s}"
        lines.append(f"{n:22s} {main:7.3f} {auc} {r['valid_frac']:7.3f}")
    return "\n".join(lines)


def evaluate(run_dir, regions=DEFAULT_REGIONS, n=100_000, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, ck = load_model(os.path.join(run_dir, "model.pt"), device)
    norm = D.load_norm_params()
    inter_norm = D.interaction_norm_from(norm)
    rotate = bool(ck["config"]["ROTATE_PHI"])
    rng = np.random.default_rng(0)
    out = {}
    for region in [r for r in regions.split(",") if r]:
        df = region_frame(region, ck["seed"], n, rng)
        arrays = D.build_inputs(df, norm, rotate=rotate)
        x_hat, m_hat = reconstruct(model, arrays, inter_norm, device)
        rows, summary = metrics(arrays, x_hat)
        if m_hat is not None:  # the fixed myy latent dim vs the true myy
            c = arrays["cond"][:, 0]
            summary["myy_head_r2"] = float(1.0 - ((m_hat - c) ** 2).mean() / c.var())
            summary["myy_head_rms_gev"] = float(np.sqrt(((m_hat - c) ** 2).mean()) * C.MYY_SCALE / 1e3)
        out[region] = {"summary": summary, "targets": rows}
        text = _fmt(region, rows, summary)
        with open(os.path.join(run_dir, f"eval_{region.replace(':', '_')}.txt"), "w") as f:
            f.write(text + "\n")
        logger.info("%s", text.splitlines()[0])
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--regions", default=DEFAULT_REGIONS)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    evaluate(args.run_dir, args.regions, args.n, args.device)


if __name__ == "__main__":
    main()
