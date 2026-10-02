"""ParT conditional VAE: one model that is both the HAXAD embedding and the
background generator.

Encoder = the contrastive_encoder's ParT (DPTModel) on the same inputs (no
photon / myy information). Decoder reconstructs the event from [z, c] with
c = normalized myy. A distance-correlation penalty makes z independent of myy
on background, so the SR background template in latent space is simply the
SB latent distribution — no conditional generator has to interpolate into
the SR.
"""
