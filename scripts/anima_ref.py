# Anima Reference for Forge Neo
# Entry point: Forge Neo imports every scripts/*.py at startup. The Anima patches go in at import time (they do
# nothing until a generation asks for a reference); any failure leaves Forge Neo unchanged and says so in the
# console. There is no panel: a method is switched on by its LoRA tag in the prompt (the tag's weight is the
# strength) with the reference picture in ImageStitch Integrated, as Forge Neo's Anima Edit works, and the few
# settings live in Settings > Stable Diffusion > Anima Reference.
#   Ref Adapter (ElectronGirl): reference as a second frame; Role Ref KV (AnticPan): reference key/value cache.

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import torch
from modules import script_callbacks, scripts, shared

ENABLED = False
TAG = "[Anima Ref]"

try:
    from lib_anima_ref import patches

    _problems = patches.check()
    if _problems:
        patches.report(_problems)
    else:
        patches.apply()
        ENABLED = True
except Exception:
    print(f"{TAG} failed to enable, Forge Neo is unchanged:")
    traceback.print_exc()

from lib_anima_ref import files, kv, stitch, tags
from lib_anima_ref.files import METHODS, REF_ADAPTER, ROLE_REF_KV
from lib_anima_ref.guidance import CORRECTIONS
from lib_anima_ref.letterbox import letterbox, to_pixels

FOUND_ATTR = "anima_ref_tags"  # on the processing object: [(name, path, weight)] of this generation's adapter tags
SECTION = ("anima_reference", "Anima Reference")
CATEGORY = "sd"
# option -> (default, label, component, component args, info, infotext key); all are Ref Adapter's
OPTIONS = {
    "anima_ref_guidance": (4.5, "[Anima Reference] Reference guidance", gr.Slider, {"minimum": 0.0, "maximum": 10.0, "step": 0.1}, "Ref Adapter: 1 = reference in both CFG branches (faster); higher = more character detail and style; needs CFG above 1", "Anima Ref guidance"),
    "anima_ref_correction": ("apg", "[Anima Reference] Correction", gr.Radio, {"choices": CORRECTIONS}, "Ref Adapter: apg keeps colors in check at high reference guidance; rescale is the alternative", "Anima Ref correction"),
    "anima_ref_rescale_phi": (0.5, "[Anima Reference] Rescale phi", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}, "Ref Adapter, rescale correction only", "Anima Ref rescale phi"),
    "anima_ref_offset": (1.0, "[Anima Reference] Reference RoPE offset (picture heights)", gr.Slider, {"minimum": 0.0, "maximum": 4.0, "step": 0.05}, "Ref Adapter: where the reference sits next to the picture; training used 1.0", "Anima Ref offset"),
    "anima_ref_fill": (255, "[Anima Reference] Letterbox fill (grey level)", gr.Slider, {"minimum": 0, "maximum": 255, "step": 1}, "Ref Adapter: around the reference when its shape differs from the picture; training used 255 (white)", "Anima Ref fill"),
}

_adapters: dict[str, dict[str, str]] = {method: {} for method in METHODS}
_names: dict[str, str] = {}  # lowercase tag name -> adapter path


def lora_dirs() -> list[str]:
    return [shared.cmd_opts.lora_dir, *getattr(shared.cmd_opts, "lora_dirs", [])]


def refresh_adapters():
    _adapters.clear()
    _adapters.update(files.find_adapters(lora_dirs()))
    _names.clear()
    for found in _adapters.values():
        for path in found.values():
            for name in tags.lora_names(path, lora_dirs()):
                _names[name] = path


def resolve(name: str) -> str | None:
    key = name.replace("\\", "/").lower()
    if key not in _names and "anima" in key.replace("-", "_"):
        refresh_adapters()  # a file added since startup (adapter files always have "anima" in their name)
    return _names.get(key)


def hook_lora():
    """Wraps Forge's LoRA extra network so the adapter tags never load as ordinary LoRAs (once)."""
    from modules import extra_networks

    lora = extra_networks.extra_network_registry.get("lora")
    if lora is None or getattr(lora, "_anima_ref_original", None) is not None:
        return
    original = lora.activate

    def activate(p, params_list):
        keep, found = tags.split(params_list, resolve)
        if found and not getattr(p, FOUND_ATTR, None):
            setattr(p, FOUND_ATTR, found)
        return original(p, keep)

    lora._anima_ref_original = original
    lora.activate = activate


def on_ui_settings():
    from modules.options import OptionInfo

    for key, (default, label, component, args, info, infotext) in OPTIONS.items():
        if key not in shared.opts.data_labels:
            shared.opts.add_option(key, OptionInfo(default, label, component, args, section=SECTION, category_id=CATEGORY, infotext=infotext).info(info))


def option(key: str):
    default, *_ = OPTIONS[key]
    return type(default)(getattr(shared.opts, key, default))


if ENABLED:
    script_callbacks.on_ui_settings(on_ui_settings)
    refresh_adapters()


class AnimaRefScript(scripts.Script):
    def __init__(self):
        self._reset()

    def _reset(self):
        self._latents: dict[tuple, object] = {}
        self._state = None  # None: not looked at yet; False: no reference this generation; dict: ready

    def title(self):
        return "Anima Reference"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if ENABLED else None

    def ui(self, is_img2img):
        return []

    def process(self, p, *args):
        self._reset()
        hook_lora()  # Forge registers its LoRA network after the scripts load, so the wrap happens here

    def _prepare(self, p):
        """Reads this generation's adapter tag, picture and settings; False when there is nothing to do."""
        found = getattr(p, FOUND_ATTR, None)
        if not found:
            return False
        name, path, strength = found[0]
        if len(found) > 1:
            print(f"{TAG} more than one reference tag in the prompt; using <lora:{name}>")
        method = files.method_of(path)
        if method not in METHODS:
            return False
        if not patches.is_anima(p.sd_model):
            print(f"{TAG} the loaded checkpoint is not Anima; <lora:{name}> is skipped")
            return False
        from modules.api.api import decode_base64_to_image

        image, count, problem = stitch.reference(getattr(p.scripts, "alwayson_scripts", None), p.script_args, decode_base64_to_image)
        if image is None:
            print(f"{TAG} <lora:{name}> needs a reference picture: {problem}; the reference is skipped")
            return False
        if count > 1:
            print(f"{TAG} {stitch.TITLE} has {count} pictures; {method} uses the first one")
        if getattr(shared.opts, "anima_do_reference", False):
            print(f"{TAG} [Anima] Enable Reference is on: {stitch.TITLE} also gives the picture to Forge's own reference (made for the Edit LoRA); turn it off for {method}")
        # Role Ref KV's LoRA runs in fp32, as its runtime does (LoRALinear)
        adapter = patches.Adapter.load(path, torch.float32 if method == ROLE_REF_KV else None)
        state = {"method": method, "adapter": adapter, "image": image.convert("RGB"), "strength": strength}
        params = {"Anima Ref method": method}
        if method == REF_ADAPTER:
            state.update({key: option(key) for key in OPTIONS})
            params.update(
                {
                    "Anima Ref guidance": state["anima_ref_guidance"],
                    "Anima Ref correction": state["anima_ref_correction"],
                    "Anima Ref rescale phi": state["anima_ref_rescale_phi"] if state["anima_ref_correction"] == "rescale" else None,
                    "Anima Ref offset": state["anima_ref_offset"] if state["anima_ref_offset"] != 1.0 else None,
                    "Anima Ref fill": state["anima_ref_fill"] if state["anima_ref_fill"] != 255 else None,
                }
            )
        p.extra_generation_params.update(params)
        return state

    def process_before_every_sampling(self, p, *args, **kwargs):
        if self._state is None:
            self._state = self._prepare(p)
        state = self._state
        if not state or not patches.is_anima(p.sd_model):
            return
        problem = patches.prepare(p.sd_model, state["adapter"])
        if problem:
            print(f"{TAG} {problem}; the reference is skipped")
            return
        unet = p.sd_model.forge_objects.unet
        if state["method"] == ROLE_REF_KV:
            if "kv" not in self._latents:  # one encode at the reference's own size; each pass size gets its cache
                self._latents["kv"] = patches.encode(p.sd_model, to_pixels(kv.resize_reference(state["image"])))
            p.sd_model.forge_objects.unet = patches.patch_unet_kv(unet, patches.RoleRefKVReference(self._latents["kv"], state["adapter"], state["strength"]))
            return

        ref_guidance, fill = state["anima_ref_guidance"], state["anima_ref_fill"]
        x = kwargs["x"]
        height, width = x.shape[-2] * 8, x.shape[-1] * 8  # this pass's size (hires fix runs a second, larger one)
        key = (width, height, fill)
        if key not in self._latents:
            self._latents[key] = patches.encode(p.sd_model, to_pixels(letterbox(state["image"], width, height, fill)))
        if kwargs.get("uc") is None and ref_guidance != 1.0:
            # at CFG 1 Forge drops the negative prompt (setup_conds), and reference guidance needs it
            print(f"{TAG} CFG is 1, so this pass has no negative prompt: reference guidance {ref_guidance} runs as 1")
            ref_guidance = 1.0
            if not getattr(p, "is_hr_pass", False):
                p.extra_generation_params["Anima Ref guidance"] = 1.0
        reference = patches.RefAdapterReference([self._latents[key]], state["adapter"], state["strength"], state["anima_ref_offset"])
        p.sd_model.forge_objects.unet = patches.patch_unet(unet, reference, ref_guidance, state["anima_ref_correction"], state["anima_ref_rescale_phi"])

    def postprocess(self, p, processed, *args):
        self._reset()
