"""Knobs for the ParT CVAE. The defaults ARE the configuration (the step-1
recipe, see README); every knob is overridable via the matching ``CVAE_*`` env
var for one-off experiments (same convention as the contrastive_encoder's
env-var sweep surface). The ParT encoder defaults are imported from
contrastive_encoder.train so both encoders stay identical unless overridden.
"""
import os

from src.embedding_models.contrastive_encoder import train as _ce
from src.embedding_models.contrastive_encoder.train import _env_flag, _env_float, _env_int

NGR_STORE = os.environ.get("NGR_STORE", os.path.join(os.path.dirname(os.path.dirname(__file__)), "store"))

# --- Source data --------------------------------------------------------------
# Inputs are resolved through the demonstrator's own law tasks (never by
# hand-built paths): NormalizeData(category, encoder_type=contrastive_encoder)
# for the data-pool pseudo-data (ProcessDelphesEvents 50/50 data pool ->
# UnweightEvents -> passthrough normalize), NormalizeMC(contrastive_encoder)
# for the pooled ParT normalization params (MC pool).
VERSION = os.environ.get("CVAE_VERSION", "prod12mlm_f19_bce_nloxsec")
PIPELINE_ENCODER_TYPE = "contrastive_encoder"  # whose preprocessing / norm params are reused
SB_CATEGORY = os.environ.get("CVAE_SB_CATEGORY", "sb_higgs_10")
SR_CATEGORY = os.environ.get("CVAE_SR_CATEGORY", "sr_higgs_10")
# Signal injection into the data pool (UnweightEvents params); 0 = background only.
SIG_TYPE = os.environ.get("CVAE_SIG_TYPE", "none")
SIG_SIGNIFICANCE = _env_float("CVAE_SIG_SIGNIFICANCE", 0.0)
# SB events are k-folded with the pipeline's Data.kfold (event_number based):
# fold_0 = test (closure / evaluation only), fold_1 = val (early stopping), rest = train.
N_FOLDS = _env_int("CVAE_N_FOLDS", 5)

# sb/sr_higgs_10 window in MeV — asserted against the analysis categories at load.
SB_LOW, SR_LOW, SR_HIGH, SB_HIGH = 105_000.0, 120_000.0, 130_000.0, 145_000.0
MYY_CENTER, MYY_SCALE = 125_000.0, 20_000.0  # c = (myy - 125 GeV) / 20 GeV ∈ [-1, 1]

# Rotate every object's phi by -phi_yy (diphoton azimuth) so the encoder and the
# reconstruction target carry only physically meaningful relative azimuths; the
# global phi of an event is a pure nuisance a latent would otherwise encode.
ROTATE_PHI = _env_flag("CVAE_ROTATE_PHI", True)

# --- Model --------------------------------------------------------------------
# Latent = [z (LATENT_DIM free dims), myy (fixed dim, fed to the decoder only)].
# The encoder never sees myy (with it, myy leaks into z: SB low-vs-high AUC 0.93).
LATENT_DIM = _env_int("CVAE_LATENT_DIM", 16)
COND_EMBED_DIM = _env_int("CVAE_COND_EMBED_DIM", 32)
# Decoder: "slot" (default) = slot-query transformer, after
# Patchouligoo/particle_transformer_ae: learned per-slot queries + broadcast
# [z, emb(myy)], self-attention among slots, per-slot heads, one extra event
# token for the high-level targets. Shared weights let rare slots (jet3/4,
# fatjets) borrow from common ones. "mlp" = one MLP -> all targets (baseline;
# median jet/fatjet R^2 0.86 vs 0.95 for slot at 250 epochs).
DECODER = os.environ.get("CVAE_DECODER", "slot").strip().lower()
SLOT_DEC_DIM = _env_int("CVAE_SLOT_DEC_DIM", 128)
SLOT_DEC_LAYERS = _env_int("CVAE_SLOT_DEC_LAYERS", 3)
SLOT_DEC_HEADS = _env_int("CVAE_SLOT_DEC_HEADS", 8)
DEC_HIDDEN = tuple(int(t) for t in os.environ.get("CVAE_DEC_HIDDEN", "512,512,256").split(","))  # mlp only

ENCODER_KWARGS = dict(
    num_self_attn_layers=_ce.NUM_SELF_ATTN_LAYERS,
    num_class_attn_layers=_ce.NUM_CLASS_ATTN_LAYERS,
    num_heads=_ce.NUM_HEADS,
    expansion=_ce.EXPANSION,
    layer_scale_ini_val=_env_float("CVAE_LAYER_SCALE_INI_VAL", _ce.LAYER_SCALE_INI_VAL),
    drop_rate=_env_float("CVAE_DROP_RATE", _ce.DROP_RATE),
    stochastic_depth_drop_rate=_env_float("CVAE_STOCHASTIC_DEPTH", _ce.STOCHASTIC_DEPTH_DROP_RATE),
    embedding_dim=_ce.EMBEDDING_DIM,
    num_embeding_layers=_ce.NUM_EMBEDDING_LAYERS,
    conv_embed_latent_dim=_ce.CONV_EMBED_LATENT_DIM,
    num_conv_layers=_ce.NUM_CONV_LAYERS,
    interaction_norm_type=os.environ.get("CVAE_INTERACTION_NORM", "layernorm").strip().lower(),
)
INPUT_SHAPE = _ce.INPUT_SHAPE  # without photon slots; the model uses data.INPUT_SHAPE

# --- Loss ---------------------------------------------------------------------
# v2_model vae_loss: masked recon (MSE continuous, BCE exist/btag) as a
# per-event sum over targets + KL_WEIGHT * KL. 0.1 starved the latent
# (~11 nats); 0.01 ~ 23 nats; 0.001 doubles the KL for no recon gain.
KL_WEIGHT = _env_float("CVAE_KL_WEIGHT", 0.01)
# Up-weight positive btag entries in the BCE (only ~3% of jets are b-tagged).
# Applied as a per-element weight inside the vae_loss mask. 10 lifts the rare
# jet3/jet4 btag AUC to 1.000 (0.977 / 0.989 at weight 1).
BTAG_POS_WEIGHT = _env_float("CVAE_BTAG_POS_WEIGHT", 10.0)

# --- Training -----------------------------------------------------------------
BATCH_SIZE = _env_int("CVAE_BATCH_SIZE", _ce.BATCH_SIZE)
# Recon keeps improving with the cosine horizon (best epoch == last at 60, 150
# and 250 epochs), hence long runs. Training resumes from <run>/last.pt, so a
# run longer than one 4 h interactive allocation is simply relaunched.
EPOCHS = _env_int("CVAE_EPOCHS", 400)
PATIENCE = _env_int("CVAE_PATIENCE", 30)
LR = _env_float("CVAE_LR", _ce.LR)
WEIGHT_DECAY = _env_float("CVAE_WEIGHT_DECAY", _ce.WEIGHT_DECAY)
GRAD_CLIP = _env_float("CVAE_GRAD_CLIP", _ce.GRAD_CLIP)
WARMUP_STEPS = _env_int("CVAE_WARMUP_STEPS", _ce.WARMUP_STEPS)
ETA_MIN = _env_float("CVAE_ETA_MIN", _ce.ETA_MIN)
# Cap on SB training events (0 = all); handy for smoke tests.
MAX_TRAIN_EVENTS = _env_int("CVAE_MAX_TRAIN_EVENTS", 0)
USE_AMP = _env_flag("CVAE_USE_AMP", True)
USE_COMPILE = _env_flag("CVAE_USE_COMPILE", False)

# --- Step 1b: myy-decoupled latent (under study; off by default) ----------------
# The step-1 latent carries myy information (SB low-vs-high AUC 0.65) because
# it must encode diphoton pT and dR, which together depend on myy, while the
# encoder cannot see myy to encode them *relative to* myy. Three pieces:
# Add the two photons as particle slots (pt, eta, phi; own type flag) to the
# encoder inputs and to the reconstruction targets. myy is then computable
# from the inputs (it is the photon-photon pair mass of the interaction
# features), so the encoder can remove it from z.
PHOTON_SLOTS = _env_flag("CVAE_PHOTON_SLOTS", False)
# Fixed latent dim: one extra encoder output m_hat regressed onto
# c = (myy - 125 GeV) / 20 GeV (MSE x MYY_HEAD_WEIGHT, no KL, not sampled).
# DEC_MYY: what the decoder's myy input is during training — "true" (the
# event's c; m_hat is then only a check that myy is readable from the inputs)
# or "pred" (m_hat itself, i.e. the myy dim of the latent). At generation the
# dim is set by hand either way.
MYY_HEAD = _env_flag("CVAE_MYY_HEAD", False)
MYY_HEAD_WEIGHT = _env_float("CVAE_MYY_HEAD_WEIGHT", 10.0)
DEC_MYY = os.environ.get("CVAE_DEC_MYY", "true").strip().lower()
# Adversarial decoupling of z from myy (Louppe et al. 2017, "learning to
# pivot"): an MLP classifies the myy bin (ADV_BINS equal bins over the SB) from
# mu — the same question the closure classifier asks. Per batch the adversary
# first takes ADV_STEPS own Adam steps on the detached mu (so it stays close to
# optimal), then the encoder gets ADV_WEIGHT x KL(bin prior || adversary
# output), which is 0 iff the adversary can do no better than the prior.
# (Plain gradient reversal with one shared step failed: the encoder dodged the
# adversary, which stayed at chance while z leaked myy at AUC 0.95.)
ADV_WEIGHT = _env_float("CVAE_ADV_WEIGHT", 0.0)
ADV_BINS = _env_int("CVAE_ADV_BINS", 8)
ADV_HIDDEN = _env_int("CVAE_ADV_HIDDEN", 128)
ADV_WARMUP_EPOCHS = _env_int("CVAE_ADV_WARMUP_EPOCHS", 5)  # weight 0, then
ADV_RAMP_EPOCHS = _env_int("CVAE_ADV_RAMP_EPOCHS", 10)      # linear ramp
ADV_STEPS = _env_int("CVAE_ADV_STEPS", 5)
ADV_LR = _env_float("CVAE_ADV_LR", 1e-3)

# --- Explored and rejected (kept for reproducibility; off by default) -----------
# Decorrelation penalty lambda * dCor^2(mu, myy) (DisCo). At lambda >= 300 it
# drives the latent into large uninformative noise (KL explodes, recon stalls;
# dCor^2 is scale-invariant and its in-batch estimate has a ~0.002 noise floor).
# With warm-up it is stable but buys little: closure AUC 0.562/0.515 -> 0.551/0.512
# for ~2x worse reconstruction. Decorrelation belongs to step 2, not here.
DISCO_WEIGHT = _env_float("CVAE_DISCO_WEIGHT", 0.0)
DISCO_WARMUP_EPOCHS = _env_int("CVAE_DISCO_WARMUP_EPOCHS", 10)  # lambda = 0, then
DISCO_RAMP_EPOCHS = _env_int("CVAE_DISCO_RAMP_EPOCHS", 10)      # linear ramp
DISCO_BATCH = _env_int("CVAE_DISCO_BATCH", 4096)  # dCor is O(B^2): subsample of each batch
DISCO_ON = os.environ.get("CVAE_DISCO_ON", "mu")   # "mu" or "z"
# Feed myy to the encoder too (standard CVAE q(z|x,c)). Rejected: myy leaks into
# z non-linearly (SB low-vs-high AUC 0.85-0.93 although dCor^2 looked small).
ENC_COND = _env_flag("CVAE_ENC_COND", False)
# Gaussian decoder with learned per-target variance (NLL), KL_WEIGHT = 1.
# Rejected: it inflates the variance of hard features and the KL collapses to ~4 nats.
DEC_LEARN_VAR = _env_flag("CVAE_DEC_LEARN_VAR", False)


def snapshot():
    """All knobs as a JSON-able dict (stored in every checkpoint / run dir)."""
    return {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in globals().items()
        if k.isupper() and isinstance(v, (int, float, str, bool, tuple, dict))
    }
