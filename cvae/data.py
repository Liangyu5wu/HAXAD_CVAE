"""Data for the ParT CVAE — everything upstream of the model is the pipeline's.

Source    : data-pool pseudo-data exactly as the pipeline feeds its encoders —
            ProcessDelphesEvents (50/50 data/MC pools) -> CategorizeEvents ->
            UnweightEvents(category, sig_type, sig_significance) ->
            NormalizeData(encoder_type=contrastive_encoder). Paths come from
            instantiating those law tasks; nothing is hard-coded.
Norm      : NormalizeMC(contrastive_encoder) pooled ParT params (MC pool).
Split     : SB events k-folded with the pipeline's Data.kfold (event_number
            based, same helper the classifiers use). SR is never trained on.
Inputs    : contrastive_encoder's build_arrays + apply_*_norm + per-batch
            build_interaction_torch (via its DPTDataSet).

On top of that we build the flat reconstruction target the decoder regresses:

  per slot   jets     : exist, pt, eta, cos phi, sin phi, m, btag
             fatjets  : exist, pt, eta, cos phi, sin phi, m, btag, tau32, tau43
             leptons  : exist, pt, eta, cos phi, sin phi
             met      : pt, cos phi, sin phi
  high level : the 7 contrastive high-level features

in the encoder's normalized units. ``exist`` / ``btag`` get a BCE head; every
other target is MSE and is masked out where the object is absent (the decoder
never has to "fit zeros" for missing objects).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch

from src.data.data import Data
from src.embedding_models import get_encoder
from src.embedding_models.contrastive_encoder.encode import _load_norm_params
from src.embedding_models.contrastive_encoder.features import (
    NUM_PARTICLES,
    _HIGHLEVEL_FEATURE_ORDER,
    _PARTICLE_SLOTS,
    _RAW_HIGHLEVEL_COLUMNS,
    _RAW_KINEMATIC_COLUMNS,
    apply_highlevel_norm,
    apply_particle_norm,
    build_arrays,
    build_interaction_norm,
)
from src.embedding_models.contrastive_encoder.model import DPTDataSet
from src.embedding_models.contrastive_encoder.model.utils import build_interaction_torch

from cvae import config as C

logger = logging.getLogger(__name__)

PHOTON_COLS = [f"photon{i}_{v}" for i in (1, 2) for v in ("pt", "eta", "phi")]
META_COLS = ["process_id", "diphoton_mass", "event_weight", "event_number"]
LOAD_COLS = list(dict.fromkeys(_RAW_KINEMATIC_COLUMNS + _RAW_HIGHLEVEL_COLUMNS + PHOTON_COLS + META_COLS))

# Region codes stored alongside every event.
SB_TRAIN, SB_VAL, SB_TEST, SR = 0, 1, 2, 3
REGION_NAMES = {SB_TRAIN: "sb_train", SB_VAL: "sb_val", SB_TEST: "sb_test", SR: "sr"}


# ---------------------------------------------------------------------------
# Pipeline inputs (resolved through the demonstrator's law tasks)
# ---------------------------------------------------------------------------


def normalize_data_task(category):
    from src.tasks.selection import NormalizeData

    kw = dict(
        version=C.VERSION,
        category_name=category,
        encoder_type=C.PIPELINE_ENCODER_TYPE,
        sig_significance=C.SIG_SIGNIFICANCE,
    )
    if C.SIG_SIGNIFICANCE > 0:
        kw["sig_type"] = C.SIG_TYPE
    return NormalizeData(**kw)


def load_category(category):
    """NormalizeData output for ``category``. If that target was never run for
    the contrastive branch, apply NormalizeData.run's exact transform
    (encoder.normalize + fillna(0)) in memory to its UnweightEvents input,
    without writing into the pipeline store."""
    task = normalize_data_task(category)
    if task.output().exists():
        data = Data(path=task.output().path)
        data.load(columns=LOAD_COLS)
    else:
        sample = task.requires()["sample"].output()
        if not sample.exists():
            raise FileNotFoundError(
                f"{sample.path} missing — produce it with the pipeline, e.g.\n"
                f"  law run NormalizeData --version {C.VERSION} --category-name {category} "
                f"--encoder-type {C.PIPELINE_ENCODER_TYPE} --sig-significance {C.SIG_SIGNIFICANCE}"
                + (f" --sig-type {C.SIG_TYPE}" if C.SIG_SIGNIFICANCE > 0 else "")
            )
        norm = task.requires()["normalization_param"].output()["norm_params"].load()
        encoder = get_encoder(C.PIPELINE_ENCODER_TYPE)
        data = Data(path=sample.path)
        data.load(columns=LOAD_COLS)
        data.data = {k: encoder.normalize(df, norm[k], task.analysis) for k, df in data.data.items()}
        data.map(lambda x: x.fillna(0))
    return data.all.reset_index(drop=True), task.analysis


def load_norm_params():
    from src.tasks.selection import NormalizeMC

    task = NormalizeMC(version=C.VERSION, encoder_type=C.PIPELINE_ENCODER_TYPE)
    return _load_norm_params(task.output()["norm_params"].path)


def resonant_pids():
    """SM Higgs process ids (same rule as ResMCEvents.keep_res)."""
    analysis = normalize_data_task(C.SR_CATEGORY).analysis
    return [p.id for p in analysis.processes if p.aux.get("resbkg", False)]


def _check_windows(analysis):
    sr = analysis.categories[C.SR_CATEGORY].aux["range"]
    sb = analysis.categories[C.SB_CATEGORY].aux["range"]
    assert tuple(sr) == (C.SR_LOW, C.SR_HIGH), f"SR window {sr} != config"
    assert tuple(sb) == ((C.SB_LOW, C.SR_LOW), (C.SR_HIGH, C.SB_HIGH)), f"SB window {sb} != config"


def load_splits(seed):
    """SB + SR pseudo-data in one frame with a ``region`` column.

    SB is k-folded by the pipeline's Data.kfold(N_FOLDS, seed); with the
    Data fold convention at fold=0: test_keys -> SB_TEST, val_keys -> SB_VAL,
    train_keys -> SB_TRAIN. Deterministic in (inputs, seed), so train /
    encode / closure all reproduce the same split. Cached as parquet.
    """
    tag = f"{C.SIG_TYPE}_{C.SIG_SIGNIFICANCE:g}"
    cache = os.path.join(C.NGR_STORE, "cache", C.VERSION, f"{C.SB_CATEGORY}_{C.SR_CATEGORY}", tag,
                         f"splits_k{C.N_FOLDS}_seed{seed}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    sb, analysis = load_category(C.SB_CATEGORY)
    sr, _ = load_category(C.SR_CATEGORY)
    _check_windows(analysis)

    folded = Data(data={"all": sb})
    folded.kfold(C.N_FOLDS, seed)
    view = Data(data=folded.data, fold=0, folds=C.N_FOLDS)
    parts = []
    for key, fdf in folded.data.items():
        code = SB_TEST if key in view.test_keys else SB_VAL if key in view.val_keys else SB_TRAIN
        parts.append(fdf.assign(region=np.int8(code)))
    parts.append(sr.assign(region=np.int8(SR)))
    df = pd.concat(parts, ignore_index=True)
    logger.info("splits: %s", {REGION_NAMES[k]: int(v) for k, v in df["region"].value_counts().sort_index().items()})

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def myy_cond(myy):
    return ((np.asarray(myy, dtype=np.float64) - C.MYY_CENTER) / C.MYY_SCALE).astype(np.float32)[:, None]


# ---------------------------------------------------------------------------
# Inputs + reconstruction targets
# ---------------------------------------------------------------------------


def _wrap(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def diphoton_phi(df):
    px = sum(df[f"photon{i}_pt"].to_numpy(np.float64) * np.cos(df[f"photon{i}_phi"].to_numpy(np.float64)) for i in (1, 2))
    py = sum(df[f"photon{i}_pt"].to_numpy(np.float64) * np.sin(df[f"photon{i}_phi"].to_numpy(np.float64)) for i in (1, 2))
    return np.arctan2(py, px)


def rotate_phi(df):
    """Return a copy of ``df`` with every object phi measured from phi_yy."""
    ref = diphoton_phi(df)
    df = df.copy()
    for col in _RAW_KINEMATIC_COLUMNS:
        if col.endswith("_phi"):
            df[col] = _wrap(df[col].to_numpy(np.float64) - ref).astype(np.float32)
    return df


def _slot_kind(prefix):
    if prefix.startswith("fatjet"):
        return "fatjet"
    if prefix.startswith("jet"):
        return "jet"
    if prefix == "met":
        return "met"
    return "lepton"


# Per-kind target fields; (field, is_binary).
_KIND_FIELDS = {
    "jet": [("exist", True), ("pt", False), ("eta", False), ("cosphi", False), ("sinphi", False), ("m", False), ("btag", True)],
    "fatjet": [("exist", True), ("pt", False), ("eta", False), ("cosphi", False), ("sinphi", False), ("m", False),
               ("btag", True), ("tau32", False), ("tau43", False)],
    "lepton": [("exist", True), ("pt", False), ("eta", False), ("cosphi", False), ("sinphi", False)],
    "met": [("pt", False), ("cosphi", False), ("sinphi", False)],
}
# Column in the 12-dim normalized particle vector for each regressed field.
_PL_IDX = {"pt": 0, "eta": 1, "m": 3, "btag": 4, "tau32": 5, "tau43": 6}


def target_spec():
    """[(name, is_binary, slot_idx | None, field)] in target-vector order."""
    spec = []
    for s, (prefix, *_rest) in enumerate(_PARTICLE_SLOTS):
        for field, is_bin in _KIND_FIELDS[_slot_kind(prefix)]:
            spec.append((f"{prefix}_{field}", is_bin, s, field))
    for h in _HIGHLEVEL_FEATURE_ORDER:
        spec.append((h, False, None, h))
    return spec


# Union of per-slot fields, in a fixed order: the slot decoder emits this many
# values per slot and ``slot_gather_index`` maps them onto the target vector.
SLOT_FIELDS = ["exist", "pt", "eta", "cosphi", "sinphi", "m", "btag", "tau32", "tau43"]


def slot_gather_index():
    """(T,) indices into [slot outputs (NUM_PARTICLES * len(SLOT_FIELDS)), high-level (7)]."""
    nf = len(SLOT_FIELDS)
    idx = []
    for _name, _is_bin, s, field in target_spec():
        if s is None:
            idx.append(NUM_PARTICLES * nf + _HIGHLEVEL_FEATURE_ORDER.index(field))
        else:
            idx.append(s * nf + SLOT_FIELDS.index(field))
    return idx


def target_names():
    return [n for n, *_ in target_spec()]


def binary_target_mask():
    return np.array([b for _, b, *_ in target_spec()], dtype=bool)


def build_targets(particle_level, kin_raw, high_level, mask):
    """Flat (N, T) target + (N, T) validity mask from the *normalized* arrays."""
    spec = target_spec()
    n = len(mask)
    tgt = np.zeros((n, len(spec)), dtype=np.float32)
    tm = np.zeros((n, len(spec)), dtype=np.float32)
    phi = np.nan_to_num(kin_raw[:, :, 2], nan=0.0)
    for j, (_name, _is_bin, s, field) in enumerate(spec):
        if s is None:
            tgt[:, j] = high_level[:, _HIGHLEVEL_FEATURE_ORDER.index(field)]
            tm[:, j] = 1.0
            continue
        present = mask[:, s]
        if field == "exist":
            tgt[:, j] = present
            tm[:, j] = 1.0
            continue
        if field == "cosphi":
            tgt[:, j] = np.cos(phi[:, s])
        elif field == "sinphi":
            tgt[:, j] = np.sin(phi[:, s])
        else:
            tgt[:, j] = particle_level[:, s, _PL_IDX[field]]
        tm[:, j] = present
    return tgt, tm


def build_inputs(df, norm_params, rotate=C.ROTATE_PHI):
    """DataFrame -> dict of numpy arrays (encoder inputs, targets, cond, meta)."""
    if rotate:
        df = rotate_phi(df)
    particle_level, kin_raw, high_level, mask = build_arrays(df, build_interaction=False)
    particle_level = apply_particle_norm(particle_level, norm_params)
    high_level = apply_highlevel_norm(high_level, norm_params)
    targets, tmask = build_targets(particle_level, kin_raw, high_level, mask)
    myy = df["diphoton_mass"].to_numpy(np.float64)
    return {
        "particle_level": particle_level,
        "kin_raw": kin_raw,
        "high_level": high_level,
        "mask": mask,
        "targets": targets,
        "tmask": tmask,
        "cond": myy_cond(myy),
        "myy": myy.astype(np.float32),
        "weight": df["event_weight"].to_numpy(np.float32),
    }


def subset(arrays, sel):
    return {k: v[sel] for k, v in arrays.items()}


# ---------------------------------------------------------------------------
# GPU-resident dataset (extends the contrastive DPTDataSet)
# ---------------------------------------------------------------------------


class CVAEDataSet(DPTDataSet):
    """DPTDataSet + reconstruction targets and myy conditioning.

    ``get_batch`` keeps the parent's whole-batch gather + on-device interaction
    rebuild, so it plugs straight into contrastive_encoder's _GpuBatchLoader.
    """

    def __init__(self, arrays, device, interaction_norm):
        n = len(arrays["mask"])
        dummy = np.zeros(n, dtype=np.float32)
        super().__init__(
            particle_level=arrays["particle_level"],
            kin_raw=arrays["kin_raw"],
            high_level=arrays["high_level"],
            mask_data=arrays["mask"],
            labels={"group_id": dummy, "process_id": dummy, "sb_id": dummy},
            device=device,
            interaction_norm=interaction_norm,
        )
        self.targets = torch.tensor(arrays["targets"], dtype=torch.float32, device=device)
        self.tmask = torch.tensor(arrays["tmask"], dtype=torch.float32, device=device)
        self.cond = torch.tensor(arrays["cond"], dtype=torch.float32, device=device)

    def get_batch(self, idx):
        idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        interaction = build_interaction_torch(self.kin_raw_tensor[idx], self.interaction_norm)
        return (
            self.particle_level_tensor[idx],
            interaction,
            self.high_level_tensor[idx],
            self.mask_tensor[idx],
            self.targets[idx],
            self.tmask[idx],
            self.cond[idx],
        )


def interaction_norm_from(norm_params):
    return build_interaction_norm(norm_params["interaction"])


assert NUM_PARTICLES == len(_PARTICLE_SLOTS)
