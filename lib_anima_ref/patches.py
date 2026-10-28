# Anima Reference for Forge Neo
# Everything that touches Forge Neo, all in memory (no Forge file is edited):
#   * Anima.forward hands the pass to the generation's reference (RefAdapterReference: two-frame pass;
#     RoleRefKVReference: reference cache) when the sampling's transformer_options carry one, and runs Forge's own
#     forward otherwise; the options live on a clone of the unet patcher, which Forge throws away after each
#     generation, so nothing stays switched on;
#   * SelfCrossAttention.forward reads or adds the Role Ref KV cache when the block's options carry kv.State;
#   * for Ref Adapter, a pre-CFG function keeps the negative conditioning and a CFG function adds the branch without
#     the reference (base) for reference guidance, then APG / rescale; Forge still runs post-CFG functions after it.
# Forge imports stay inside functions, so the rest of lib_anima_ref is testable without Forge.

import inspect

import torch

from lib_anima_ref import guidance, kv
from lib_anima_ref.adapter import Adapter, active
from lib_anima_ref.forward import pad_to_patch_size, ref_forward

OPTION_KEY = "anima_ref"
TAG = "[Anima Ref]"

REQUIRED = {
    "backend.nn.anima": ("Anima", "SelfCrossAttention"),
    "backend.diffusion_engine.anima": ("Anima",),
    "backend.sampling.sampling_function": ("calc_cond_uncond_batch",),
    "backend.patcher.unet": ("UnetPatcher",),
}
UNET_METHODS = ("clone", "set_transformer_option", "set_model_sampler_cfg_function", "set_model_sampler_pre_cfg_function")
DIT_ATTRS = ("prepare_embedded_sequence", "unpatchify", "forward")
ATTENTION_ATTRS = ("compute_qkv", "compute_attention", "forward")


def check() -> list[str]:
    import importlib

    problems: list[str] = []
    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")
    if problems:
        return problems

    from backend.nn.anima import Anima, SelfCrossAttention
    from backend.patcher.unet import UnetPatcher

    missing = [m for m in UNET_METHODS if not hasattr(UnetPatcher, m)]
    if missing:
        problems.append(f"UnetPatcher is missing {', '.join(missing)}")
    missing = [a for a in DIT_ATTRS if not hasattr(Anima, a)]
    if missing:
        problems.append(f"backend.nn.anima.Anima is missing {', '.join(missing)}")
    params = list(inspect.signature(getattr(Anima, "_anima_ref_original", Anima.forward)).parameters)
    if params[:4] != ["self", "x", "timesteps", "context"]:
        problems.append(f"backend.nn.anima.Anima.forward has an unexpected signature {params}")
    missing = [a for a in ATTENTION_ATTRS if not hasattr(SelfCrossAttention, a)]
    if missing:
        problems.append(f"backend.nn.anima.SelfCrossAttention is missing {', '.join(missing)}")
    return problems


def report(problems: list[str]):
    print(f"{TAG} disabled, this Forge Neo is missing what the extension needs (Forge Neo is unchanged):")
    for problem in problems:
        print(f"    - {problem}")


class RefAdapterReference:
    """Ref Adapter (ElectronGirl): what the patched forward needs during one sampling pass."""

    def __init__(self, latents: list[torch.Tensor], adapter: Adapter, strength: float, offset_h: float):
        self.latents, self.adapter, self.strength, self.offset_h = latents, adapter, strength, offset_h

    def run(self, dm, x, timesteps, context, transformer_options):
        self.adapter.install(dm, block_mappings())  # no-op once hooked; covers a model reloaded since then
        with active(self.adapter, self.strength):
            return ref_forward(dm, x, timesteps, context, self.latents, self.offset_h, transformer_options)


Reference = RefAdapterReference


class RoleRefKVReference:
    """Role Ref KV (AnticPan): the reference latent, and its cache once the first positive pass has made it."""

    def __init__(self, latent: torch.Tensor, adapter: Adapter, strength: float):
        self.latent, self.adapter, self.strength = latent, adapter, strength
        self.caches: dict[tuple, list] = {}  # padded target latent size -> per-block (k, v)

    def run(self, dm, x, timesteps, context, transformer_options):
        self.adapter.install(dm, block_mappings())
        options = transformer_options or {}
        with active(self.adapter, self.strength), kv.qk_eps(dm):
            size = tuple(pad_to_patch_size(x[:1, :1], (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial)).shape[-2:])
            cache = self.caches.get(size)
            if cache is None:
                rows = kv.cond_rows(options, x.shape[0])
                if not any(rows):  # a negative-only pass before any positive one: nothing to attend to yet
                    return kv.kv_forward(dm, x, timesteps, context, [], options)
                first = rows.index(True)
                latent = self.latent.to(device=x.device, dtype=x.dtype)
                cache = self.caches[size] = kv.capture(dm, latent, context[first : first + 1], size, options)
            return kv.kv_forward(dm, x, timesteps, context, cache, options)


def apply():
    from backend.nn.anima import Anima, SelfCrossAttention

    if getattr(Anima, "_anima_ref_original", None) is None:
        original = Anima.forward

        def forward(self, x, timesteps, context, padding_mask=None, **kwargs):
            reference = (kwargs.get("transformer_options") or {}).get(OPTION_KEY)
            if reference is None:
                return original(self, x, timesteps, context, padding_mask, **kwargs)
            return reference.run(self, x, timesteps, context, kwargs.get("transformer_options"))

        Anima._anima_ref_original = original
        Anima.forward = forward

    if getattr(SelfCrossAttention, "_anima_ref_original", None) is None:
        original_attention = SelfCrossAttention.forward

        def attention_forward(self, x, context=None, rope_emb=None, transformer_options={}):
            state = (transformer_options or {}).get(kv.OPTION_KEY)
            if state is None or context is not None:
                return original_attention(self, x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            return kv.self_attention(self, x, rope_emb, transformer_options, state)

        SelfCrossAttention._anima_ref_original = original_attention
        SelfCrossAttention.forward = attention_forward


def block_mappings() -> dict:
    """Forge Neo's Anima block mappings ((trained blocks, model blocks) -> source block per model block)."""
    try:
        import backend.nn.anima
    except ImportError:  # only outside Forge (tests)
        return {}
    return getattr(backend.nn.anima, "BLOCK_MAPPINGS", {})


def prepare(sd_model, adapter: Adapter) -> str | None:
    """Hooks the adapter into the loaded Anima; returns why it cannot run there, or None."""
    dm = sd_model.forge_objects.unet.model.diffusion_model
    blocks = len(dm.blocks)
    try:
        adapter.install(dm, block_mappings())
    except ValueError as e:
        return str(e)
    if adapter.blocks and blocks != adapter.blocks and blocks not in adapter.announced:
        adapter.announced.add(blocks)
        print(f"{TAG} {adapter.name} was trained for {adapter.blocks} blocks; spreading it over this model's {blocks} blocks, as Forge Neo does for Anima LoRAs")
    return None


def is_anima(sd_model) -> bool:
    from backend.diffusion_engine.anima import Anima

    return isinstance(sd_model, Anima)


@torch.inference_mode()
def encode(sd_model, pixels: torch.Tensor) -> torch.Tensor:
    """(1, H, W, 3) in [0, 1] -> Anima latent, as the engine's encode_first_stage does it but without its own
    reference bookkeeping (opts.anima_do_reference)."""
    vae = sd_model.forge_objects.vae
    latent = vae.encode(pixels)
    return vae.first_stage_model.process_in(latent).cpu()


def without_reference(model_options: dict) -> dict:
    options = model_options.copy()
    transformer_options = options.get("transformer_options", {}).copy()
    transformer_options.pop(OPTION_KEY, None)
    options["transformer_options"] = transformer_options
    return options


def preserve_memory(unet, adapter: Adapter):
    """Tell Forge's memory manager about the adapter weights the hooks put on the GPU."""
    if hasattr(unet, "add_extra_preserved_memory_during_sampling"):
        dtype = getattr(unet.model, "computation_dtype", None)  # KModel's: what the hooks' inputs come in
        unet.add_extra_preserved_memory_during_sampling(adapter.nbytes(dtype if isinstance(dtype, torch.dtype) else torch.float16))


def patch_unet_kv(unet, reference: RoleRefKVReference):
    """A clone of the unet patcher that samples with the Role Ref KV reference (plain CFG)."""
    unet = unet.clone()
    unet.set_transformer_option(OPTION_KEY, reference)
    preserve_memory(unet, reference.adapter)
    return unet


def patch_unet(unet, reference: RefAdapterReference, ref_guidance: float, correction: str, rescale_phi: float):
    """A clone of the unet patcher that samples with the Ref Adapter reference and the reference guidance."""
    from backend.sampling.sampling_function import calc_cond_uncond_batch

    unet = unet.clone()
    unet.set_transformer_option(OPTION_KEY, reference)
    preserve_memory(unet, reference.adapter)
    captured = {}

    def pre_cfg(model, cond, uncond, x, timestep, model_options):
        captured["uncond"] = uncond
        return model, cond, uncond, x, timestep, model_options

    def cfg(args):
        x, pos, neg = args["input"], args["cond_denoised"], args["uncond_denoised"]
        base = None
        if guidance.needs_base(ref_guidance) and captured.get("uncond") is not None:
            _, base = calc_cond_uncond_batch(args["model"], [], captured["uncond"], x, args["timestep"], without_reference(args["model_options"]))
        return x - guidance.combine(pos, neg, base, args["cond_scale"], ref_guidance, correction, rescale_phi)

    # with ref guidance the negative branch is needed even at CFG 1
    unet.set_model_sampler_pre_cfg_function(pre_cfg, disable_cfg1_optimization=guidance.needs_base(ref_guidance))
    unet.set_model_sampler_cfg_function(cfg, disable_cfg1_optimization=guidance.needs_base(ref_guidance))
    return unet
