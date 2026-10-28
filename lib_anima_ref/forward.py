# Anima Reference for Forge Neo
# Anima's forward pass with the reference as a second latent frame: the reference goes in clean (timestep 0) and its
# RoPE (h, w) coordinates are shifted ref_offset_h target heights down, so the model reads it without copying it
# pixel-aligned (no pose copying). Only the target frame comes out.
# Ported from ComfyUI-Anima-Ref-Adapter (ElectronGirl, Apache-2.0), __init__.py: _frame_positions,
# _rope_from_positions, _pick_ref, _ref_forward. Works on Forge Neo's backend.nn.anima.Anima, which keeps ComfyUI's
# module names (pos_embedder, t_embedder, blocks, final_layer, prepare_embedded_sequence, unpatchify).

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("anima_ref")
_warned_sizes: set = set()


def pad_to_patch_size(x: torch.Tensor, patch: tuple[int, ...]) -> torch.Tensor:
    """ComfyUI's comfy.ldm.common_dit.pad_to_patch_size (circular padding up to a multiple of the patch)."""
    pad = ()
    for i in range(x.ndim - 2):
        pad = (0, (patch[i] - x.shape[i + 2] % patch[i]) % patch[i]) + pad
    return F.pad(x, pad, mode="circular") if any(pad) else x


def frame_positions(t_index: int, height: int, width: int, h0: int, w0: int, device) -> tuple[torch.Tensor, ...]:
    hh, ww = torch.meshgrid(torch.arange(height, device=device) + h0, torch.arange(width, device=device) + w0, indexing="ij")
    return torch.full_like(hh, t_index).flatten(), hh.flatten(), ww.flatten()


def rope_from_positions(pe, t: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """VideoRopePosition3DEmb's embeddings for explicit per-token (t, h, w) positions, frequencies in fp32 as the
    adapter was trained (Forge's own embedder computes them in fp16)."""
    device = t.device
    h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(device))
    w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(device))
    t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(device))

    def rotations(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)

    emb = torch.cat([rotations(t, t_freqs), rotations(h, h_freqs), rotations(w, w_freqs)], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def pick_ref(refs: list[torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
    """The reference latent (B, C, 1, h, w) for a target latent of this (h, w): an exact-size encode if there is one,
    else the closest one resized in latent space (lower fidelity, warned once per size)."""
    refs = [r.unsqueeze(2) if r.ndim == 4 else r for r in refs]
    for ref in refs:
        if tuple(ref.shape[-2:]) == size:
            return ref
    ref = min(refs, key=lambda r: abs(r.shape[-2] - size[0]) + abs(r.shape[-1] - size[1]))
    if size not in _warned_sizes:
        _warned_sizes.add(size)
        logger.warning(f"[Anima Ref] no reference encoded at latent size {size}; resizing the {tuple(ref.shape[-2:])} one (lower fidelity)")
    shrink = ref.shape[-2] > size[0]
    return F.interpolate(ref[:, :, 0].float(), size=size, mode="area" if shrink else "bilinear").unsqueeze(2)


def patch_grid(dm, latent: torch.Tensor) -> tuple[int, int]:
    """(h, w) of the patch grid of a padded latent."""
    return latent.shape[-2] // dm.patch_spatial, latent.shape[-1] // dm.patch_spatial


def run_dit(dm, frames: torch.Tensor, timesteps_B_T: torch.Tensor, rope: torch.Tensor, context: torch.Tensor, transformer_options: dict, final: bool = True) -> torch.Tensor | None:
    """Anima.forward's body with given RoPE and per-frame timesteps; final=False stops after the blocks."""
    x_emb, _, extra = dm.prepare_embedded_sequence(frames, padding_mask=None)
    t_emb, adaln = dm.t_embedder[1](dm.t_embedder[0](timesteps_B_T).to(x_emb.dtype))
    t_emb = dm.t_embedding_norm(t_emb)
    kwargs = {"rope_emb_L_1_1_D": rope.unsqueeze(1).unsqueeze(0), "adaln_lora_B_T_3D": adaln, "extra_per_block_pos_emb": extra, "transformer_options": transformer_options}
    if x_emb.dtype == torch.float16:
        x_emb = x_emb.float()  # residual stream in fp32, as Anima.forward does for fp16
    for block in dm.blocks:
        x_emb = block(x_emb, t_emb, context, **kwargs)
    if not final:
        return None
    out = dm.final_layer(x_emb.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln)
    return dm.unpatchify(out)


def ref_forward(dm, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, refs: list[torch.Tensor], offset_ratio: float, transformer_options: dict) -> torch.Tensor:
    """Anima.forward with the reference as frame 1 (timestep 0, RoPE shifted by offset_ratio target heights)."""
    orig_shape = list(x.shape)
    x = pad_to_patch_size(x, (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
    ref = pick_ref(refs, tuple(x.shape[-2:])).to(device=x.device, dtype=x.dtype)
    if ref.shape[0] != x.shape[0]:
        ref = ref[:1].expand(x.shape[0], -1, -1, -1, -1)
    frames = torch.cat([x[:, :, :1], ref[:, :, :1]], dim=2)
    hp, wp = patch_grid(dm, x)
    positions = [frame_positions(0, hp, wp, 0, 0, x.device), frame_positions(1, hp, wp, round(offset_ratio * hp), 0, x.device)]
    rope = rope_from_positions(dm.pos_embedder, *(torch.cat(axis) for axis in zip(*positions)))

    timesteps = timesteps.reshape(x.shape[0], -1)[:, :1]
    timesteps = torch.cat([timesteps, torch.zeros_like(timesteps)], dim=1)
    out = run_dit(dm, frames, timesteps, rope, context, transformer_options)[:, :, :1]
    return out[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]
