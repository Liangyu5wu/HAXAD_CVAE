"""Collect the metrics of every run quoted on the slides into figs/summary.json.

    python slides/collect.py        (from the repo root, after `source env.sh`)

Per run: closure AUCs of mu (SB-low vs SB-high, SR vs SB), core R^2 median on SB test / SR / signals,
myy-head R^2, shuffle test, myy of the decoded photons, validation loss at the selected epoch.
Per group (same config, different seeds): mean, std (ddof 1) and range.
"""
import json
import os

import numpy as np

STORE = os.path.join(os.environ.get("NGR_STORE", "store"), "runs")
OUT = os.path.join(os.path.dirname(__file__), "figs", "summary.json")

GROUPS = {
    # multi-seed baseline (400 epochs; photons + myy head; adversary 20 inner steps, hidden 256)
    "step1": ["step1_seed0", "step1_seed1", "base_step1_s2"],
    "adv0.03": ["s1e_ph_adv0p03", "s1f_adv0p03_s1"] + [f"base_adv0p03_s{i}" for i in range(2, 7)],
    "adv0.05": ["s1f_adv0p05"] + [f"base_adv0p05_s{i}" for i in range(1, 7)],
    # tuning scan (400 epochs, seed 0 unless noted)
    "scan": ["s1f_adv0p01", "s1f_adv0p1_late", "s1f_bigdec", "s1f_lat32", "s1f_stradv", "s1f_kl0p003",
             "s1e_ph_adv0p1", "s1e_ph_adv0p3"],
    # step-1b development (150 epochs)
    "dev": ["ce_mlp_s0", "ce_none_s0", "ce_ph_mlp_s0", "ce_ph_none_s0", "s1c_b0p1", "s1c_adv1", "s1c_adv10",
            "s1d_adv0p3", "s1d_adv0p1", "s1d_nophot_adv1", "s1d_nophot_adv0p3"],
}
CFG_KEYS = ["seed", "EPOCHS", "LATENT_DIM", "KL_WEIGHT", "PHOTON_SLOTS", "MYY_HEAD", "COND_EMBED", "ADV_WEIGHT",
            "ADV_STEPS", "ADV_HIDDEN", "ADV_WARMUP_EPOCHS", "SLOT_DEC_DIM", "SLOT_DEC_LAYERS"]
METRICS = ["auc_lohi", "auc_srsb", "r2_sb", "r2_sr"]


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def run_metrics(tag):
    d = os.path.join(STORE, tag)
    cfg, cl, ev = _load(f"{d}/config.json"), _load(f"{d}/closure_mu.json"), _load(f"{d}/eval.json")
    mr, hist = _load(f"{d}/myy_response.json"), _load(f"{d}/history.json")
    out = {"tag": tag, "config": {k: cfg.get(k) for k in CFG_KEYS},
           "auc_lohi": cl["auc_sb_low_vs_high"][0], "auc_lohi_err": cl["auc_sb_low_vs_high"][1],
           "auc_srsb": cl["auc_sr_vs_sb"][0], "auc_srsb_err": cl["auc_sr_vs_sb"][1],
           "r2_sb": ev["sb_test"]["summary"]["core_r2_median"], "r2_sr": ev["sr"]["summary"]["core_r2_median"],
           "r2_signal": {k.split(":", 1)[1]: v["summary"]["core_r2_median"] for k, v in ev.items()
                         if k.startswith("signal:")},
           "myy_head_r2": ev["sb_test"]["summary"].get("myy_head_r2"),
           "myy_head_rms_gev": ev["sb_test"]["summary"].get("myy_head_rms_gev")}
    if mr:
        out["shuffle_r2"] = [mr["core_r2_median_true_myy"], mr["core_r2_median_shuffled_myy"]]
        dp = mr.get("decoded_photon_myy", {})
        if dp:
            out["photon_myy_corr_given"] = dp["given_shuffled: corr(decoded, given)"]
            out["photon_myy_corr_true_when_shuffled"] = dp["given_shuffled: corr(decoded, true)"]
            out["photon_myy_rms_gev"] = dp["given_true: rms(decoded - true) GeV"]
        out["response_change_105_145"] = {t: [v["decoded_fixed_z"][-1] - v["decoded_fixed_z"][0],
                                              v["data"][-1] - v["data"][0]] for t, v in mr["response"].items()}
    if hist:
        sel = np.asarray(hist.get("val_select", hist["val_total"]), float)  # older runs selected on val_total
        i = int(np.nanargmin(sel))
        out["best_epoch"] = int(hist["epoch"][i])
        out["val_recon_best"] = float(hist["val_recon"][i])
    for name in ("transfer_test", "signal_sensitivity", "signal_sensitivity_raw"):
        j = _load(f"{d}/{name}.json")
        if j:
            out[name] = j
    return out


def main():
    runs, groups = {}, {}
    for g, tags in GROUPS.items():
        for t in tags:
            runs[t] = run_metrics(t)
        if g in ("scan", "dev"):
            continue
        a = np.array([[runs[t][m] for m in METRICS] for t in tags])
        groups[g] = {"n": len(tags), "tags": tags,
                     **{m: {"mean": float(a[:, j].mean()), "std": float(a[:, j].std(ddof=1)),
                            "min": float(a[:, j].min()), "max": float(a[:, j].max())}
                        for j, m in enumerate(METRICS)}}
        rms = [runs[t]["photon_myy_rms_gev"] for t in tags if "photon_myy_rms_gev" in runs[t]]
        if rms:
            groups[g]["photon_myy_rms_gev"] = {"mean": float(np.mean(rms)), "min": float(min(rms)),
                                               "max": float(max(rms))}
        # does the SB-validation loss pick the best-reconstructing seed?
        vr = np.array([runs[t]["val_recon_best"] for t in tags])
        r2 = np.array([runs[t]["r2_sb"] for t in tags])
        groups[g]["spearman_valrecon_r2"] = float(np.corrcoef(np.argsort(np.argsort(vr)),
                                                              np.argsort(np.argsort(-r2)))[0, 1])
        groups[g]["best_by_val"] = tags[int(vr.argmin())]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"runs": runs, "groups": groups}, open(OUT, "w"), indent=1)

    print(f"{'run':20s} {'seed':>4s} {'lo/hi':>6s} {'SR/SB':>6s} {'R2 SB':>6s} {'R2 SR':>6s} {'rms':>5s} {'valrec':>7s}")
    for g, tags in GROUPS.items():
        for t in tags:
            r = runs[t]
            print(f"{t:20s} {r['config']['seed']:>4} {r['auc_lohi']:.3f} {r['auc_srsb']:.3f} {r['r2_sb']:.3f}  "
                  f"{r['r2_sr']:.3f} {r.get('photon_myy_rms_gev', float('nan')):5.2f} {r.get('val_recon_best', float('nan')):.4f}")
        if g in groups:
            s = groups[g]
            print(f"  {g}: " + "  ".join(f"{m} {s[m]['mean']:.3f}±{s[m]['std']:.3f}" for m in METRICS)
                  + f"  | spearman(val recon, R2) {s['spearman_valrecon_r2']:+.2f}, best by val: {s['best_by_val']}")


if __name__ == "__main__":
    main()
