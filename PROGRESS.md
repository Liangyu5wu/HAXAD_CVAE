# new_gen_R progress log

State, decisions and results of the single-CVAE study; newest at the bottom. Technical setup in `README.md`,
working rules in `WORKFLOW.md`. Every number below is in `slides/figs/summary.json` (`slides/collect.py`) unless a
source is named.

## Goal and decisions (agreed with the physicist)

- One ParT CVAE with latent [z (16 dims), m_γγ (one fixed dim)] replaces the HAXAD embedding + separate
  m_γγ-conditioned generator.
- Only m_γγ is fixed in the latent; conditioning on multiplicities was rejected (it would have to be specified at
  generation).
- The downstream anomaly classifier uses z, so the SR background template is the SB z distribution. The metrics
  that matter: closure AUC (SR vs SB, SB-low vs SB-high, on μ) and signal information in z. Decoder fidelity of
  the m_γγ dependence is not a goal for now.
- Compute: interactive allocations only (gpu_interactive, gpu_shared_interactive), never batch jobs.

## Step 1: reconstruction (2026-10-01 – 10-05)

- ParT (DPTModel) encoder + slot-query decoder, KL 0.01, 400 epochs: core R² 0.965 ± 0.004 (3 seeds).
- z carries m_γγ: closure lo/hi 0.648 ± 0.010, SR/SB 0.535 ± 0.004 (raw inputs 0.669 / 0.543, README).
- The decoder ignores m_γγ: shuffle test core R² 0.967 → 0.966 (`cvae/myy_response.py`).

## Steps 1b–1c: fixed m_γγ dim and decorrelation (2026-10-02 – 10-05)

- Photon slots (p_T, η, φ; photon-pair interaction mass = m_γγ) + m̂ head: m̂ R² 0.9995, rms 0.29 GeV.
- Without a counter-force the photons raise the leak (lo/hi 0.86 at 150 epochs); β = 0.1 does not help (0.90).
- One-step gradient reversal fails (encoder dodges). Working adversary: own Adam, 20 inner steps, hidden 256,
  encoder term KL(bin prior ‖ adversary), on after 5 epochs, ramp 10.
- λ ≥ 1 reaches the closure floor but destroys reconstruction (R² 0.32 at λ = 1).
- Without photons the decoder ignores m_γγ even with the adversary; branch dropped.
- COND_EMBED: MLP(1 → 32) kept over direct concatenation (same m_γγ behaviour, better recon).

## Tuning scan (2026-10-05 – 10-06)

400 epochs, seed 0: latent 32, KL 0.003, 40 adversary steps, late adversary, λ 0.01–0.3, decoder 256 × 4. A
plateau: every variant except the larger decoder (R² 0.83) lies inside the λ = 0.03 seed spread.

## Multi-seed baseline (2026-10-06)

| config (400 epochs) | seeds | AUC SR/SB | AUC lo/hi | core R² SB |
|---|---|---|---|---|
| step 1 | 3 | 0.535 ± 0.004 | 0.648 ± 0.010 | 0.965 ± 0.004 |
| photons + adversary λ = 0.03 | 7 | 0.530 ± 0.014 | 0.567 ± 0.010 | 0.923 ± 0.009 |
| photons + adversary λ = 0.05 | 7 | 0.518 ± 0.002 | 0.559 ± 0.005 | 0.897 ± 0.027 |

- Decision (to review): **baseline = photons + m̂ head + adversary λ = 0.05**; closure SR/SB 0.516–0.521 on all
  seeds (≈ 4× the step-1 seed spread below step 1). λ = 0.03 is bimodal (0.516 / 0.55).
- Price: R² 0.85–0.93, seed-dependent. The SB validation reconstruction loss ranks the test R² with Spearman 0.96
  over the 7 λ = 0.05 seeds, so seeds are chosen by validation loss; it picks `s1f_adv0p05` (SR/SB 0.516,
  lo/hi 0.557, R² 0.926 / 0.915).
- Closure estimator noise: ±0.001 (`store/closure_noise.py`, `logs/closure_noise.log`).
- Supervised signal vs SR background AUC ≥ 0.994 for every model and raw inputs: uninformative.
- Decoder of the baseline: decoded photons carry the given m_γγ (corr 0.989); diphoton p_T and MET shift with m_γγ
  in the data's direction; n_jets does not; jet1 p_T gets a slope the data do not have.

## Slides (2026-10-19)

`slides/CVAE_HAXAD_status.pptx`, 9 main + divider + 5 backup slides, built from the SLAC template by
`slides/build_deck.py`; figures by `slides/make_figs.py` from `slides/collect.py`. Rendered with the LibreOffice
shifter image and checked slide by slide.

## Next actions (to be agreed)

- A (proposed): weak-supervision injection test on z (CWoLa-style, several signal strengths, null test without
  signal), step 1 vs baseline vs raw inputs.
- B: recover reconstruction at λ = 0.05 only if A shows R² limits the sensitivity.
- C: fake excess from the residual closure 0.518 = the null of A.
- D: generation (step 3) after A.
