# Anima Reference for Forge Neo
# The methods' LoRAs, applied through forward hooks and only inside a reference forward pass: the weights must stay
# off for the branch without the reference (Ref Adapter's reference guidance) and for every generation that does
# not use the panel, so they never go through Forge's own LoRA patching.
# Ported from ComfyUI-Anima-Ref-Adapter (ElectronGirl, Apache-2.0), __init__.py, class _Adapter, and from
# Anima-Role-Ref-KV (AnticPan, Apache-2.0), anima_role_ref_kv/lora.py, LoRALinear (the delta in the adapter's dtype).

import json
import os
import struct

import torch
import torch.nn.functional as F

REF_ADAPTER_KEY = "anima_ref_adapter"  # ElectronGirl's metadata: rank, targets, ref_offset_h
ROLE_REF_KV_KEY = "charref_version"  # AnticPan's metadata (with requires_reference_runtime)
PREFIX = "diffusion_model."
HOOK_ATTR = "_anima_ref_hook"  # on each hooked module: the hook handle

_active = {"adapter": None}  # the adapter whose LoRA the hooks apply right now (None: hooks do nothing)
_cache: dict[tuple, "Adapter"] = {}


def read_header_metadata(path: str) -> dict | None:
    """The safetensors header's __metadata__ ({} without one), or None when the file is not a safetensors file.
    No tensor is read."""
    try:
        with open(path, "rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            if size > 64 * 1024 * 1024:
                return None
            header = json.loads(f.read(size))
        return header.get("__metadata__") or {}
    except Exception:
        return None


def read_metadata(path: str) -> dict | None:
    """ElectronGirl's adapter metadata, or None for any other file."""
    value = (read_header_metadata(path) or {}).get(REF_ADAPTER_KEY)
    try:
        return json.loads(value) if value else None
    except Exception:
        return None


class Adapter:
    def __init__(self, tensors: dict[str, torch.Tensor], name: str = "adapter", compute_dtype: torch.dtype | None = None):
        self.name = name
        self.compute_dtype = compute_dtype  # None: the input's dtype; torch.float32: as AnticPan's LoRALinear
        self.weights: dict[str, list] = {}  # module name -> [down, up, alpha / rank]
        for key, down in tensors.items():
            if not key.endswith(".lora_down.weight"):
                continue
            module = key[: -len(".lora_down.weight")]
            if module.startswith(PREFIX):
                module = module[len(PREFIX) :]
            up = tensors[key.replace("lora_down", "lora_up")]
            alpha = tensors.get(key.replace("lora_down.weight", "alpha"))
            alpha = float(alpha) if alpha is not None else float(down.shape[0])
            self.weights[module] = [down, up, alpha / down.shape[0]]
        if not self.weights:
            raise ValueError(f"{name} has no lora_down/lora_up weights")
        self.strength = 1.0

    @classmethod
    def load(cls, path: str, compute_dtype: torch.dtype | None = None) -> "Adapter":
        from safetensors.torch import load_file

        key = (os.path.abspath(path), os.path.getmtime(path), compute_dtype)
        if key not in _cache:
            _cache.clear()  # one adapter in memory at a time
            _cache[key] = cls(load_file(path), name=os.path.basename(path), compute_dtype=compute_dtype)
        return _cache[key]

    def nbytes(self, dtype: torch.dtype) -> int:
        """Size of the weights on the GPU when they run in this dtype (the input's, unless compute_dtype is set)."""
        size = torch.finfo(self.compute_dtype or dtype).bits // 8
        return sum((entry[0].numel() + entry[1].numel()) * size for entry in self.weights.values())

    def install(self, diffusion_model: torch.nn.Module) -> int:
        """Hooks every module this adapter targets; a module hooked before (by any adapter) is left as it is."""
        installed = 0
        for name in self.weights:
            module = diffusion_model.get_submodule(name)
            if getattr(module, HOOK_ATTR, None) is None:
                setattr(module, HOOK_ATTR, module.register_forward_hook(_make_hook(name)))
                installed += 1
        return installed


def _make_hook(name: str):
    def hook(module, inputs, output):
        adapter = _active["adapter"]
        if adapter is None:
            return output
        entry = adapter.weights.get(name)
        if entry is None:
            return output
        x = inputs[0]
        dtype = adapter.compute_dtype or x.dtype
        if entry[0].device != x.device or entry[0].dtype != dtype:
            entry[0] = entry[0].to(device=x.device, dtype=dtype)
            entry[1] = entry[1].to(device=x.device, dtype=dtype)
        delta = F.linear(F.linear(x.to(dtype), entry[0]), entry[1]) * (entry[2] * adapter.strength)
        return output + delta.to(output.dtype)

    return hook


class active:
    """Context manager: the hooks apply this adapter, at this strength, inside the block."""

    def __init__(self, adapter: Adapter, strength: float):
        self.adapter, self.strength = adapter, strength

    def __enter__(self):
        self.previous = _active["adapter"]
        self.adapter.strength = self.strength
        _active["adapter"] = self.adapter
        return self.adapter

    def __exit__(self, *exc):
        _active["adapter"] = self.previous
        return False
