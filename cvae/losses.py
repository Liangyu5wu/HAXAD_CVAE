"""Distance correlation (DisCo) penalty, and the learned-variance variant of
the reconstruction loss. The default reconstruction + KL is v2_model's
``vae_loss`` (masked, BCE binary heads), reused as is."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def hetero_vae_loss(x, x_hat, logvar_dec, mu, logvar, kl_weight, mask, binary_dims):
    """Same conventions as v2_model ``vae_loss`` (per-event masked mean rescaled
    to a sum over targets, KL to N(0, I)), but continuous targets use a Gaussian
    NLL with a learned per-target log-variance ``logvar_dec`` (T,)."""
    r2 = (x_hat - x) ** 2
    nll = 0.5 * (r2 * torch.exp(-logvar_dec) + logvar_dec)
    bce = F.binary_cross_entropy_with_logits(x_hat, x, reduction="none")
    per = torch.where(binary_dims, bce, nll) * mask
    recon = (per.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0) * mask.shape[1]).mean()
    kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return recon + kl_weight * kl, recon, kl


def _centered_dist(x):
    sq = (x * x).sum(dim=1, keepdim=True)
    d2 = (sq + sq.T - 2.0 * x @ x.T).clamp_min(0.0)
    # eps keeps the sqrt gradient finite on the diagonal / duplicate points.
    d = torch.sqrt(d2 + 1e-12)
    return d - d.mean(dim=0, keepdim=True) - d.mean(dim=1, keepdim=True) + d.mean()


def distance_corr(x, y):
    """Squared distance correlation dCor^2(x, y) in [0, 1] (Szekely et al.;
    DisCo, Kasieczka & Shih 2020). ``x`` (B, d), ``y`` (B, k); 0 iff independent
    (population). Computed in fp32 regardless of autocast.

    Inputs are centred and divided by their global std first. dCor^2 is
    invariant to that (translation + global scale), but it keeps the
    |x|^2 + |y|^2 - 2 x.y distance expansion well conditioned: without it a
    growing latent scale (which dCor^2 does not penalise) turns the pairwise
    distances into rounding noise and the penalty gradient into garbage — the
    runaway-KL divergence seen with the myy-conditioned encoder."""

    def _std(t):
        t = t.float()
        t = t - t.mean(dim=0, keepdim=True)
        return t / t.std().clamp_min(1e-12)

    with torch.autocast(device_type=x.device.type, enabled=False):
        a = _centered_dist(_std(x))
        b = _centered_dist(_std(y))
        dcov2 = (a * b).mean()
        return dcov2 / torch.sqrt((a * a).mean() * (b * b).mean()).clamp_min(1e-12)
