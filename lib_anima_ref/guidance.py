# Anima Reference for Forge Neo
# Separate text and reference guidance, in x0 space, with APG or CFG-rescale correction.
#   pos  = prompt + reference, neg = negative + reference, base = negative without the reference
#   guided = base + ref_guidance * (neg - base) + cfg * (pos - neg)
# At ref_guidance 1 the reference is in both branches and base is not needed (plain CFG).
# Ported from ComfyUI-Anima-Ref-Adapter (ElectronGirl, Apache-2.0), __init__.py, AnimaRefGuiderImpl.predict_noise.

import torch
import torch.nn.functional as F

CORRECTIONS = ("apg", "rescale", "none")


def _flat_dims(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, x.ndim))


def needs_base(ref_guidance: float) -> bool:
    return ref_guidance != 1.0


def combine(pos: torch.Tensor, neg: torch.Tensor, base: torch.Tensor | None, cfg: float, ref_guidance: float, correction: str = "apg", rescale_phi: float = 0.5) -> torch.Tensor:
    if base is None:
        base = neg
    guided = base + ref_guidance * (neg - base) + cfg * (pos - neg)
    if correction == "apg":
        update = (guided - pos).double()
        unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
        parallel = (update * unit).sum(dim=_flat_dims(pos), keepdim=True) * unit
        guided = pos + (update - parallel).to(pos.dtype)
    elif correction == "rescale":
        ratio = pos.std(dim=_flat_dims(pos), keepdim=True) / guided.std(dim=_flat_dims(guided), keepdim=True)
        guided = rescale_phi * guided * ratio + (1 - rescale_phi) * guided
    return guided
