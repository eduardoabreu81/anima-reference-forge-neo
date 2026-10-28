# Anima Reference for Forge Neo
# Role Ref KV method: the reference goes through the model once (timestep 0, frame index 10, its grid centred on the
# target's) and every block keeps its self-attention keys and values; at each step the positive rows attend to them
# next to their own. The method's LoRA is on in every pass, the negative included, and the q/k norms use the
# Diffusers runtime's eps the adapter was trained with.
# Ported from Anima-Role-Ref-KV (AnticPan, Apache-2.0): anima_role_ref_kv/reference.py (rope_at, ref_positions,
# encode_reference, velocity) and pipeline.py (resize_reference). Runs on Forge Neo's backend.nn.anima, whose
# attention computes q/k after norm and RoPE (compute_qkv) and takes keys longer than the queries
# (torch_attention_op), so the cache goes in between the two.

import torch
from PIL import Image, ImageOps

from lib_anima_ref.forward import pad_to_patch_size, patch_grid, rope_from_positions, run_dit

REF_FRAME = 10
QK_EPS = 1e-5  # Diffusers' Cosmos attention; Anima on ComfyUI and Forge Neo uses 1e-6
REF_AREA = 768 * 768
OPTION_KEY = "anima_ref_kv"
COND = 0


def resize_reference(image: Image.Image, area: int = REF_AREA) -> Image.Image:
    """At most `area` pixels, sides rounded down to multiples of 16 (no letterbox)."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    s = min(1.0, (area / (image.width * image.height)) ** 0.5)
    size = tuple(max(16, int(n * s) // 16 * 16) for n in image.size)
    return image if image.size == size else image.resize(size, Image.LANCZOS)


def grid_positions(t_index: int, hs: torch.Tensor, ws: torch.Tensor) -> tuple[torch.Tensor, ...]:
    hh, ww = torch.meshgrid(hs, ws, indexing="ij")
    return torch.full_like(hh, t_index).flatten(), hh.flatten(), ww.flatten()


def reference_positions(target_hw: tuple[int, int], ref_hw: tuple[int, int], device) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference's patch grid centred on the target's (positions may be negative or past the target)."""
    (ht, wt), (hr, wr) = target_hw, ref_hw
    return torch.arange(hr, device=device) + (ht // 2 - hr // 2), torch.arange(wr, device=device) + (wt // 2 - wr // 2)


class State:
    """What the patched self-attention reads during one forward: capture the cache, or inject it into cond rows."""

    def __init__(self, mode: str, cache: list | None = None, cond_rows: list[bool] | None = None):
        self.mode, self.cache, self.cond_rows, self.index = mode, cache if cache is not None else [], cond_rows, 0


def cond_rows(transformer_options: dict, batch: int) -> list[bool]:
    """Which rows of Forge's cond/uncond batch are positive (cond_or_uncond lists one entry per chunk)."""
    chunks = transformer_options.get("cond_or_uncond")
    if not chunks or batch % len(chunks):
        return [True] * batch
    n = batch // len(chunks)
    return [chunk == COND for chunk in chunks for _ in range(n)]


def self_attention(attn, x: torch.Tensor, rope_emb, transformer_options: dict, state: State) -> torch.Tensor:
    """SelfCrossAttention.forward for a self-attention layer, with the reference cache."""
    q, k, v = attn.compute_qkv(x, None, rope_emb=rope_emb)  # (B, S, heads, head_dim), normed and rotated
    index = state.index
    state.index += 1
    if state.mode == "capture":
        state.cache.append((k.detach(), v.detach()))
        return attn.compute_attention(q, k, v, transformer_options=transformer_options)
    rows = state.cond_rows or [True] * x.shape[0]
    if not any(rows) or index >= len(state.cache):
        return attn.compute_attention(q, k, v, transformer_options=transformer_options)
    ref_k, ref_v = state.cache[index]
    cond = [i for i, r in enumerate(rows) if r]

    def with_reference(idx):
        n = len(idx)
        kk = torch.cat([ref_k[:1].expand(n, -1, -1, -1).to(k.dtype), k[idx]], dim=1)
        vv = torch.cat([ref_v[:1].expand(n, -1, -1, -1).to(v.dtype), v[idx]], dim=1)
        return attn.compute_attention(q[idx], kk, vv, transformer_options=transformer_options)

    if len(cond) == len(rows):
        return with_reference(cond)
    uncond = [i for i, r in enumerate(rows) if not r]
    out_cond = with_reference(cond)
    out_uncond = attn.compute_attention(q[uncond], k[uncond], v[uncond], transformer_options=transformer_options)
    out = out_cond.new_empty((x.shape[0], *out_cond.shape[1:]))
    out[cond], out[uncond] = out_cond, out_uncond.to(out.dtype)
    return out


class qk_eps:
    """Context manager: every attention q/k norm of the DiT uses `eps` inside the block."""

    def __init__(self, dm, eps: float = QK_EPS):
        self.norms = [getattr(getattr(block, a), n) for block in dm.blocks for a in ("self_attn", "cross_attn") for n in ("q_norm", "k_norm")]
        self.eps = eps

    def __enter__(self):
        self.saved = [norm.eps for norm in self.norms]
        for norm in self.norms:
            norm.eps = self.eps

    def __exit__(self, *exc):
        for norm, eps in zip(self.norms, self.saved):
            norm.eps = eps
        return False


def capture(dm, ref: torch.Tensor, context: torch.Tensor, target_hw: tuple[int, int], transformer_options: dict) -> list:
    """The per-block (k, v) of the reference latent (1, C, 1, h, w) for a padded target latent of size target_hw."""
    ref = pad_to_patch_size(ref if ref.ndim == 5 else ref.unsqueeze(2), (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
    hr, wr = patch_grid(dm, ref)
    target = (target_hw[0] // dm.patch_spatial, target_hw[1] // dm.patch_spatial)
    hs, ws = reference_positions(target, (hr, wr), ref.device)
    rope = rope_from_positions(dm.pos_embedder, *grid_positions(REF_FRAME, hs, ws))
    state = State("capture")
    timesteps = torch.zeros(1, 1, device=ref.device)
    run_dit(dm, ref[:, :, :1], timesteps, rope, context[:1], {**transformer_options, OPTION_KEY: state}, final=False)
    return state.cache


def kv_forward(dm, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, cache: list, transformer_options: dict) -> torch.Tensor:
    """Anima.forward for the target frame, positive rows attending to the reference cache (RoPE in fp32)."""
    orig_shape = list(x.shape)
    x = pad_to_patch_size(x, (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
    hp, wp = patch_grid(dm, x)
    rope = rope_from_positions(dm.pos_embedder, *grid_positions(0, torch.arange(hp, device=x.device), torch.arange(wp, device=x.device)))
    state = State("inject", cache, cond_rows(transformer_options, x.shape[0]))
    timesteps = timesteps.reshape(x.shape[0], -1)[:, :1]
    out = run_dit(dm, x[:, :, :1], timesteps, rope, context, {**transformer_options, OPTION_KEY: state})
    return out[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]
