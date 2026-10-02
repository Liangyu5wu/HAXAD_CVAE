# new_gen_R — ParT CVAE: one model for the HAXAD embedding and generation

Goal: replace the separate embedding model + myy-conditioned generative model
with one conditional VAE whose latent space is

    [ z (16 free dims, from the encoder) , myy (fixed dim, given) ]

so that events can later be generated at a *specified* myy instead of relying
on a generator to interpolate into the signal region.

- **Step 1 (this version): a CVAE that reconstructs events well.** Done (production runs `step1_seed{0,1}`); see
  results below.
- Step 2 (next): build the generative model on top of this latent
  (e.g. model p(z | myy) in the SB, or resample, then decode / classify).

## Model

```
 encoder  q(z | x)                                decoder  p(x | z, myy)
 11 particle slots (11x12) ─┐                      myy ─► MLP(1→32) ─┐
 pair interactions (11x11x6)├─► ParT (DPTModel) ─► μ, logσ² (16)      ├─► slot-query transformer ─► 76 targets
 7 high-level features ─────┘     z = μ + σ·ε  ───────────────────────┘
```

- **Encoder**: the contrastive_encoder's `DPTModel`, unchanged (4 self-attention
  blocks with pairwise interaction bias, 2 class-attention blocks). It does
  **not** see the photons' four-vectors or myy.
- **Decoder**: slot-query transformer (after
  [Patchouligoo/particle_transformer_ae](https://github.com/Patchouligoo/particle_transformer_ae)):
  12 learned query tokens (11 slots + 1 event token) + broadcast `[z, emb(myy)]`,
  3 self-attention blocks, per-slot heads.
- **Inputs** (identical to the contrastive encoder):
  - slots: jet1–4 (pt, η, φ, m, btag), fatjet1–2 (+ τ32, τ43), el1–2, mu1–2,
    MET (pt, φ); each with a 5-dim type one-hot; absent objects masked
  - pairwise (every two present slots): Δη, Δφ, ΔR, pt ratio, pair pt, pair mass
  - high level: diphoton_pt, diphoton_ΔR, met_pt, n_jets, n_bjets, n_leptons, n_photons
  - one addition: every φ is measured from the diphoton azimuth φ_yy.
- **Targets (76)**: per slot exist (BCE), pt, η, cos φ, sin φ, m, btag (BCE),
  τ32, τ43 as applicable (MSE, masked where the object is absent), plus the 7
  high-level features.
- **Loss**: v2_model `vae_loss` (masked recon summed over targets) +
  0.01·KL(q(z|x) ‖ N(0, I)); positive btag entries weighted ×10.
- 1.63 M parameters. AdamW 2.8e-4, warmup → cosine, batch 16384, bf16, 400 epochs.

## Data (all through the haxad_demonstrator pipeline; nothing hand-built)

| | source |
|---|---|
| events | data pool of the 50/50 `ProcessDelphesEvents` split → `UnweightEvents(sb/sr_higgs_10, sig_significance=0)` → `NormalizeData(contrastive_encoder)` (paths from the law task objects, version `prod12mlm_f19_bce_nloxsec`) |
| split | SB k-folded with `Data.kfold(5, seed)`: fold_0 test, fold_1 val (early stopping), rest train (958k); SR (528k) never trained on |
| normalization | `NormalizeMC(contrastive_encoder)` pooled ParT params |
| preprocessing | contrastive `build_arrays`, `apply_*_norm`, `build_interaction_torch` |
| signals (evaluation only) | MC pool `NormalizeMC` output, filtered by process id |

## Code

| file | content |
|---|---|
| `env.sh` | sources the demonstrator's `setup.sh`; sets `PYTHONNOUSERSITE=1` |
| `cvae/config.py` | all knobs; **defaults = the recipe above** (`CVAE_*` env vars override) |
| `cvae/data.py` | pipeline loading, k-fold split, φ rotation, targets, GPU dataset |
| `cvae/model.py` | `ParTCVAE` (DPTModel encoder, slot / MLP decoder) |
| `cvae/losses.py` | dCor (rejected option), learned-variance NLL (rejected option) |
| `cvae/train.py` | training; resumable from `last.pt`, `DONE` marks completion |
| `cvae/evaluate.py` | per-target R² / acc / AUC on sb_test, sr, signals |
| `cvae/encode.py` | latents (μ, z) for all SB + SR events |
| `cvae/closure.py` | myy-independence tests of the latent (input to step 2) |
| `cvae/plots.py` | diagnostics.pdf (+ PNGs): latent activity, μ per dim by split / myy, ⟨μ⟩ vs myy, correlations, posterior vs prior, recon vs truth, prior-decoded targets |

Reused from haxad_demonstrator (imported, never modified): `setup.sh`, the law
tasks for paths, `Data`, contrastive `build_arrays` / norms / `DPTModel` /
`DPTDataSet` / `_GpuBatchLoader` / optimizer recipe, v2 `vae_loss` and
`_save_reconstruction_pdf`.

## Running (interactive GPU nodes only, no batch jobs)

```bash
salloc -A m2616 -C gpu -q interactive -N 1 --gpus-per-node=4 -c 128 -t 4:00:00 --no-shell
# then from a login node (detached, survives closing the terminal):
setsid nohup srun --jobid=<JOBID> --overlap -N1 -n1 -c 32 --gpus=1 bash -c '
  source env.sh; R=$NGR_STORE/runs/<tag>
  python -m cvae.train --run-dir $R --seed 0 &&
  python -m cvae.evaluate --run-dir $R &&
  python -m cvae.encode --run-dir $R &&
  python -m cvae.closure --run-dir $R' >> logs/<tag>.log 2>&1 < /dev/null &
```

400 epochs take longer than one 4 h allocation: re-run the same command in a
new allocation and training resumes from `last.pt`.

## Step-1 results (SB test fold unless noted; R², btag = ROC AUC)

Production runs (`store/runs/step1_seed{0,1}`, config defaults, 400 epochs,
best epoch 378 for both). "core" = jet1–4 + fatjet1 targets.

| region | core R² median: seed 0 / seed 1 | slot 250 ep (dev) |
|---|---|---|
| SB test | 0.967 / 0.966 | 0.948 |
| SR | 0.969 / 0.970 | 0.950 |
| signal XHH_500 | 0.773 / 0.766 | 0.690 |
| signal HVT_VcXjjHyy_2000_300 | 0.762 / 0.763 | 0.619 |
| signal WN_HyyN_300 | 0.792 / 0.812 | 0.754 |
| signal TT_tZNtHyyN_1000_205_60 | 0.275 / 0.451 | 0.316 |

SB test per target (seed 0 / seed 1): jet1 pt 0.99/0.99, jet2 pt 0.97/0.98,
jet3 pt 0.85/0.86, jet4 pt 0.63/0.81, jet4 m 0.82/0.84, fatjet1 pt 0.72/0.77,
fatjet1 m 0.34/0.37, τ32 0.34/0.35; btag AUC 1.000; MET / high-level ≥ 0.99.
Closure of μ: SB-low vs SB-high AUC 0.655 / 0.653, SR vs SB 0.538 / 0.536.

Development scan (latent 16, β 0.01, btag ×10 unless noted):

| | MLP 150 ep | MLP 250 ep | slot 150 ep | **slot 250 ep** |
|---|---|---|---|---|
| jet1 pt / m | 0.96 / 0.93 | 0.97 / 0.97 | 0.97 / 0.97 | **0.98 / 0.98** |
| jet2 pt / m | 0.89 / 0.87 | 0.93 / 0.94 | 0.92 / 0.95 | **0.95 / 0.95** |
| jet3 pt / m | 0.61 / 0.65 | 0.68 / 0.74 | 0.74 / 0.85 | **0.76 / 0.88** |
| jet4 pt / m | 0.40 / 0.13 | 0.43 / 0.41 | 0.51 / 0.74 | **0.52 / 0.75** |
| btag AUC jet2–4 | ≥ 0.99 | 1.00 | 1.00 | 1.00 |
| fatjet1 pt / η / m / τ32 | 0.45 / 0.73 / 0.25 / 0.20 | 0.62 / 0.83 / 0.33 / 0.28 | 0.49 / 0.89 / 0.29 / 0.24 | 0.62 / 0.94 / 0.29 / 0.30 |
| MET, diphoton pt/ΔR, n_jets | ≥ 0.99 | ≥ 0.99 | ≥ 0.99 | ≥ 0.99 |
| median R² (jets + fatjet1) | 0.73 | 0.86 | 0.92 | **0.95** |

Slot 250 ep on other regions: **SR 0.950** (= SB, so the decoder interpolates
in myy without loss); signals (never trained on) 0.62–0.75.

Weak spot: fatjet substructure (m, τ32 R² ≈ 0.3) — only ~1 % of background
events have a fatjet.

### What did not work (kept as off-by-default options in `config.py`)

- **β = 0.1**: latent starved (~11 nats), btag never learned. Latent size 8→32 does not help; β does.
- **DisCo decorrelation** λ·dCor²(μ, myy): unstable at λ ≥ 300 (latent turns
  into large noise, KL explodes); with warm-up stable but closure AUC only
  0.562 → 0.551 for 2× worse reconstruction.
- **myy into the encoder**: myy leaks into z (SB low-vs-high AUC 0.85–0.93).
- **learned-variance Gaussian decoder**: KL collapses to ~4 nats.
- Training length matters more than any of the above (60 → 150 → 250 epochs keeps improving).

### Closure (myy dependence of z; input for step 2)

AUC of a classifier on z: SB-low vs SB-high / SR vs SB (0.5 = no myy information).
Raw input features: 0.669 / 0.543. Step-1 latent: ≈ 0.643 / 0.533. The better
the reconstruction, the more myy-correlated information z keeps; handling this
is part of step 2.
