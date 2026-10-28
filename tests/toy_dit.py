"""A tiny DiT with the interface ref_forward uses on Forge Neo's backend.nn.anima.Anima, for CPU tests.

Module names follow Anima (blocks.N.self_attn.q_proj, mlp.layer1, ...) so the adapter's hooks attach the same way;
the math is simplified (no RMS-norm RoPE kernel, which comes from comfy_kitchen on Forge). VideoRopePosition3DEmb is
Forge's (backend/nn/anima.py, neo 2900d59) with the frequencies in fp32.
"""

import math

import torch
from einops import rearrange, repeat
from torch import nn


class VideoRopePosition3DEmb(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        dim_h = head_dim // 6 * 2
        dim_t = head_dim - 2 * dim_h
        self.register_buffer("dim_spatial_range", torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h, persistent=False)
        self.register_buffer("dim_temporal_range", torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t, persistent=False)
        self.h_ntk_factor = self.w_ntk_factor = self.t_ntk_factor = 1.0

    def forward(self, x_B_T_H_W_C: torch.Tensor, device) -> torch.Tensor:
        _, T, H, W, _ = x_B_T_H_W_C.shape
        h_freqs = 1.0 / ((10000.0 * self.h_ntk_factor) ** self.dim_spatial_range.to(device))
        w_freqs = 1.0 / ((10000.0 * self.w_ntk_factor) ** self.dim_spatial_range.to(device))
        t_freqs = 1.0 / ((10000.0 * self.t_ntk_factor) ** self.dim_temporal_range.to(device))
        seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)

        def stack(e):
            return torch.stack([torch.cos(e), -torch.sin(e), torch.sin(e), torch.cos(e)], dim=-1)

        half_h, half_w, half_t = stack(torch.outer(seq[:H], h_freqs)), stack(torch.outer(seq[:W], w_freqs)), stack(torch.outer(seq[:T], t_freqs))
        em = torch.cat(
            [
                repeat(half_t, "t d x -> t h w d x", h=H, w=W),
                repeat(half_h, "h d x -> t h w d x", t=T, w=W),
                repeat(half_w, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )
        return rearrange(em, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


class Timesteps(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.seen = []

    def forward(self, t_B_T):
        assert t_B_T.ndim == 2
        self.seen.append(t_B_T.detach().clone())
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        emb = t_B_T.float()[..., None] * freqs
        return torch.cat([emb.cos(), emb.sin()], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, sample):
        return sample, self.linear_1(sample)


class Attention(nn.Module):
    """Forge's SelfCrossAttention API: compute_qkv gives (B, S, heads, head_dim) after the q/k norms and, for
    self-attention, the split-half RoPE (what comfy_kitchen's rms_rope_split_half does); compute_attention takes keys
    longer than the queries."""

    def __init__(self, dim, context_dim=None, heads=2):
        super().__init__()
        self.is_SelfAttn = context_dim is None
        self.n_heads, self.head_dim = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(context_dim or dim, dim, bias=False)
        self.v_proj = nn.Linear(context_dim or dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    @staticmethod
    def rotate(x, rope_emb):
        r = rope_emb[0, :, 0][None, :, None].to(x.dtype)  # (1, L, 1, D/2, 2, 2)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([r[..., 0, 0] * x1 + r[..., 0, 1] * x2, r[..., 1, 0] * x1 + r[..., 1, 1] * x2], dim=-1)

    def compute_qkv(self, x, context=None, rope_emb=None):
        q, k, v = self.q_proj(x), self.k_proj(x if context is None else context), self.v_proj(x if context is None else context)
        q, k, v = (rearrange(t, "b s (h d) -> b s h d", h=self.n_heads) for t in (q, k, v))
        q, k = self.q_norm(q), self.k_norm(k)
        if self.is_SelfAttn and rope_emb is not None:
            q, k = self.rotate(q, rope_emb), self.rotate(k, rope_emb)
        return q, k, v

    def compute_attention(self, q, k, v, transformer_options={}):
        out = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)))
        return self.output_proj(rearrange(out, "b h s d -> b s (h d)"))

    def forward(self, x, context=None, rope_emb=None, transformer_options={}):
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v, transformer_options=transformer_options)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, 2 * dim, bias=False)
        self.layer2 = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x):
        return self.layer2(torch.nn.functional.gelu(self.layer1(x)))


class Block(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.self_attn = Attention(dim)
        self.cross_attn = Attention(dim, context_dim)
        self.mlp = FeedForward(dim)
        self.calls = []

    def forward(self, x, emb, context, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None, extra_per_block_pos_emb=None, transformer_options={}):
        self.calls.append({"x": x.shape, "emb": emb.shape, "rope": rope_emb_L_1_1_D.shape, "options": transformer_options})
        _, T, H, W, _ = x.shape
        scale = rearrange(emb, "b t d -> b t 1 1 d")
        flat = rearrange(x * (1 + scale), "b t h w d -> b (t h w) d")
        flat = flat + self.self_attn(flat, None, rope_emb=rope_emb_L_1_1_D, transformer_options=transformer_options)
        flat = flat + self.cross_attn(flat, context, transformer_options=transformer_options)
        flat = flat + self.mlp(flat)
        return rearrange(flat, "b (t h w) d -> b t h w d", t=T, h=H, w=W)


class FinalLayer(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.linear = nn.Linear(dim, out, bias=False)

    def forward(self, x, emb, adaln_lora_B_T_3D=None):
        return self.linear(x + rearrange(emb, "b t d -> b t 1 1 d"))


class ToyAnima(nn.Module):
    def __init__(self, channels=4, dim=24, heads=2, context_dim=8, num_blocks=2):
        super().__init__()
        self.patch_temporal, self.patch_spatial = 1, 2
        self.channels = channels
        self.pos_embedder = VideoRopePosition3DEmb(dim // heads)
        self.t_embedder = nn.Sequential(Timesteps(dim), TimestepEmbedding(dim))
        self.t_embedding_norm = nn.RMSNorm(dim, eps=1e-6)
        self.x_embedder = nn.Linear((channels + 1) * 4, dim, bias=False)
        self.blocks = nn.ModuleList([Block(dim, context_dim) for _ in range(num_blocks)])
        self.final_layer = FinalLayer(dim, channels * 4)

    def prepare_embedded_sequence(self, x, padding_mask=None):
        mask = torch.zeros(x.shape[0], 1, *x.shape[2:], dtype=x.dtype)
        x = torch.cat([x, mask], dim=1)
        x = rearrange(x, "b c t (h m) (w n) -> b t h w (c m n)", m=2, n=2)
        emb = self.x_embedder(x)
        return emb, self.pos_embedder(emb, device=x.device), None

    def unpatchify(self, x):
        return rearrange(x, "B T H W (p1 p2 C) -> B C T (H p1) (W p2)", p1=2, p2=2)
