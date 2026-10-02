"""ParT CVAE = contrastive_encoder's DPTModel (encoder) + myy-conditioned decoder (MLP or slot-query)."""
from __future__ import annotations

import torch
from torch import nn

from src.embedding_models.contrastive_encoder.model import DPTModel

from cvae import config as C
from cvae import data as D


def _mlp(dims, act=nn.GELU):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class SlotDecoder(nn.Module):
    """Parallel slot-query decoder (non-autoregressive), after
    Patchouligoo/particle_transformer_ae: h_i = Q_i + W [z, emb(c)], a few
    bidirectional self-attention blocks over the slot tokens (no pairwise
    bias: those features are functions of the unknown outputs), then per-slot
    heads. One extra "event" query token emits the high-level targets.
    Output is gathered into the same flat target order as the MLP decoder."""

    def __init__(self, in_dim, d=C.SLOT_DEC_DIM, n_layers=C.SLOT_DEC_LAYERS, n_heads=C.SLOT_DEC_HEADS):
        super().__init__()
        self.n_slots = D.NUM_PARTICLES
        self.slot_query = nn.Parameter(torch.randn(1, self.n_slots + 1, d) * 0.02)
        self.from_latent = _mlp([in_dim, d, d])
        layer = nn.TransformerEncoderLayer(d, n_heads, 3 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.slot_head = nn.Linear(d, len(D.SLOT_FIELDS))
        self.event_head = nn.Linear(d, len(D._HIGHLEVEL_FEATURE_ORDER))
        self.register_buffer("gather_idx", torch.as_tensor(D.slot_gather_index(), dtype=torch.long),
                             persistent=False)

    def forward(self, zc):
        h = self.norm(self.blocks(self.slot_query + self.from_latent(zc)[:, None, :]))
        out = torch.cat([self.slot_head(h[:, :-1]).flatten(1), self.event_head(h[:, -1])], dim=1)
        return out[:, self.gather_idx]


class ParTCVAE(nn.Module):
    """q(z|x[, c]) from the ParT encoder; p(x|z, c) from an MLP.

    With ``enc_cond`` the condition c is appended to the encoder's high-level
    features (DPTModel itself is unchanged, it just sees one more high-level
    input). ``encode(x, cond)`` returns mu (deterministic) — the latent handed
    downstream and the one the decorrelation penalty acts on by default.
    """

    def __init__(
        self,
        n_targets,
        latent_dim=C.LATENT_DIM,
        n_cond=1,
        dec_hidden=C.DEC_HIDDEN,
        cond_embed_dim=C.COND_EMBED_DIM,
        encoder_kwargs=None,
        input_shape=C.INPUT_SHAPE,
        enc_cond=C.ENC_COND,
        dec_learn_var=C.DEC_LEARN_VAR,
        decoder=C.DECODER,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_cond = bool(enc_cond)
        if self.enc_cond:
            (n_hl,) = input_shape[2]
            input_shape = (input_shape[0], input_shape[1], (n_hl + n_cond,))
        self.encoder = DPTModel(
            input_shape=input_shape,
            output_size=latent_dim,
            output_activation="none",
            **(encoder_kwargs if encoder_kwargs is not None else C.ENCODER_KWARGS),
        )
        self.cond_embed = _mlp([n_cond, cond_embed_dim, cond_embed_dim])
        if decoder == "slot":
            self.decoder = SlotDecoder(latent_dim + cond_embed_dim)
            assert len(self.decoder.gather_idx) == n_targets
        elif decoder == "mlp":
            self.decoder = _mlp([latent_dim + cond_embed_dim, *dec_hidden, n_targets])
        else:
            raise ValueError(f"CVAE_DECODER={decoder!r}: expected 'mlp' or 'slot'")
        # Learned per-target log-variance of the Gaussian decoder (continuous targets).
        self.dec_logvar = nn.Parameter(torch.zeros(n_targets)) if dec_learn_var else None

    def _enc_inputs(self, x, cond):
        if not self.enc_cond:
            return x
        pl, inter, hl, mask = x
        return (pl, inter, torch.cat([hl, cond.to(hl.dtype)], dim=-1), mask)

    def encode_mulogvar(self, x, cond):
        return self.encoder.encode_mulogvar(self._enc_inputs(x, cond))

    def decode(self, z, cond):
        return self.decoder(torch.cat([z, self.cond_embed(cond)], dim=-1))

    def forward(self, x, cond):
        mu, logvar = self.encode_mulogvar(x, cond)
        z = self.encoder.reparameterize(mu, logvar) if self.training else mu
        return self.decode(z, cond), z, mu, logvar

    def encode(self, x, cond, sample=False):
        mu, logvar = self.encode_mulogvar(x, cond)
        return self.encoder.reparameterize(mu, logvar) if sample else mu
