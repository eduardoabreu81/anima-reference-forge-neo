# Anima Reference for Forge Neo
# Entry point: Forge Neo imports every scripts/*.py at startup. The Anima patches go in at import time (they do
# nothing until a generation asks for a reference); any failure leaves Forge Neo unchanged and says so in the
# console. The panel is an accordion under the generation settings, in txt2img and img2img, with two methods:
# Ref Adapter (ElectronGirl: reference as a second frame) and Role Ref KV (AnticPan: reference key/value cache).

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import torch
from modules import scripts, shared
from modules.ui_components import InputAccordion, ToolButton

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

from lib_anima_ref import __version__, files, kv
from lib_anima_ref.files import METHODS, REF_ADAPTER, ROLE_REF_KV
from lib_anima_ref.guidance import CORRECTIONS
from lib_anima_ref.letterbox import letterbox, to_pixels

_adapters: dict[str, dict[str, str]] = {method: {} for method in METHODS}


def lora_dirs() -> list[str]:
    return [shared.cmd_opts.lora_dir, *getattr(shared.cmd_opts, "lora_dirs", [])]


def refresh_adapters():
    _adapters.clear()
    _adapters.update(files.find_adapters(lora_dirs()))


def adapter_update(method: str):
    choices = list(_adapters.get(method, {}))
    return gr.update(choices=choices, value=next(iter(choices), None))


def as_image(value):
    """The reference as a PIL image: from the panel, or from the API (base64 string, or a {"image": ...} dict)."""
    from PIL import Image

    if value is None or value == "":
        return None
    if isinstance(value, dict):
        value = value.get("image") or value.get("composite") or value.get("background")
    if isinstance(value, str):
        from modules.api.api import decode_base64_to_image

        value = decode_base64_to_image(value)
    if not isinstance(value, Image.Image):
        import numpy as np

        value = Image.fromarray(np.asarray(value).astype("uint8"))
    return value.convert("RGB")


class AnimaRefScript(scripts.Script):
    def __init__(self):
        self._latents: dict[tuple, object] = {}
        self._active = False

    def title(self):
        return "Anima Reference"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if ENABLED else None

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        refresh_adapters()
        default = next((m for m in METHODS if _adapters.get(m)), REF_ADAPTER)
        choices = list(_adapters.get(default, {}))
        with InputAccordion(False, label=f"{self.title()} v{__version__}", elem_id=f"anima_ref_{tab}") as enable:
            with gr.Row():
                image = gr.Image(type="pil", label="Reference image", height=256, elem_id=f"anima_ref_image_{tab}")
                with gr.Column():
                    method = gr.Radio(list(METHODS), value=default, label="Method", info="Ref Adapter: ElectronGirl's adapter, with reference guidance. Role Ref KV: AnticPan's, plain CFG", elem_id=f"anima_ref_method_{tab}")
                    with gr.Row():
                        adapter = gr.Dropdown(choices=choices, value=next(iter(choices), None), label="Adapter weights", info="the method's .safetensors in the LoRA folder", elem_id=f"anima_ref_adapter_{tab}")
                        refresh = ToolButton(value="\U0001f504", elem_id=f"anima_ref_refresh_{tab}")
                    strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Adapter strength")
                    with gr.Group(visible=default == REF_ADAPTER) as ref_adapter_group:
                        ref_guidance = gr.Slider(0.0, 10.0, value=4.5, step=0.1, label="Reference guidance", info="1 = reference in both CFG branches (faster); higher = more character detail and style. Needs CFG above 1")
                        with gr.Row():
                            correction = gr.Radio(list(CORRECTIONS), value="apg", label="Correction")
                            rescale_phi = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Rescale phi", info="only for rescale")
            with gr.Accordion("Advanced (Ref Adapter)", open=False, visible=default == REF_ADAPTER) as advanced:
                with gr.Row():
                    offset_h = gr.Slider(0.0, 4.0, value=1.0, step=0.05, label="Reference RoPE offset (target heights)", info="training used 1.0")
                    fill = gr.Slider(0, 255, value=255, step=1, label="Letterbox fill (grey level)")

        def on_method(m):
            return adapter_update(m), gr.update(visible=m == REF_ADAPTER), gr.update(visible=m == REF_ADAPTER)

        method.change(on_method, inputs=[method], outputs=[adapter, ref_adapter_group, advanced], show_progress=False)

        def on_refresh(m, current):
            refresh_adapters()
            update = adapter_update(m)
            if current in _adapters.get(m, {}):
                update["value"] = current
            return update

        refresh.click(on_refresh, inputs=[method, adapter], outputs=[adapter], show_progress=False)

        self.infotext_fields = [
            (enable, lambda d: "Anima Ref adapter" in d),
            (method, "Anima Ref method"),
            (adapter, "Anima Ref adapter"),
            (strength, "Anima Ref strength"),
            (ref_guidance, "Anima Ref guidance"),
            (correction, "Anima Ref correction"),
            (rescale_phi, "Anima Ref rescale phi"),
            (offset_h, "Anima Ref offset"),
            (fill, "Anima Ref fill"),
        ]
        args = [enable, method, image, adapter, strength, ref_guidance, correction, rescale_phi, offset_h, fill]
        for component in args:
            component.do_not_save_to_config = True
        return args

    def process(self, p, enable, method, image, adapter, strength, ref_guidance, correction, rescale_phi, offset_h, fill, *args):
        self._active = False
        self._latents = {}
        if not enable:
            return
        method = method if method in METHODS else REF_ADAPTER
        self._image = as_image(image)
        if self._image is None:
            print(f"{TAG} no reference image, the reference is skipped")
            return
        if not patches.is_anima(p.sd_model):
            print(f"{TAG} the loaded checkpoint is not Anima, the reference is skipped")
            return
        if adapter not in _adapters.get(method, {}):
            refresh_adapters()  # a file added since the panel was built, or a name pasted from infotext / the API
        path = _adapters.get(method, {}).get(adapter)
        if not path:
            print(f"{TAG} {method} weights {adapter!r} not found in the LoRA folders, the reference is skipped")
            return
        if getattr(shared.opts, "anima_do_reference", False):
            print(f"{TAG} Forge's own Anima reference setting is on; its reference frames are ignored while {method} runs")
        # Role Ref KV's LoRA runs in fp32, as its runtime does (LoRALinear)
        self._adapter = patches.Adapter.load(path, torch.float32 if method == ROLE_REF_KV else None)
        self._method = method
        self._settings = (float(strength), float(ref_guidance), str(correction), float(rescale_phi), float(offset_h), int(fill))
        self._active = True
        params = {"Anima Ref method": method, "Anima Ref adapter": adapter, "Anima Ref strength": strength}
        if method == REF_ADAPTER:
            params.update(
                {
                    "Anima Ref guidance": ref_guidance,
                    "Anima Ref correction": correction,
                    "Anima Ref rescale phi": rescale_phi if correction == "rescale" else None,
                    "Anima Ref offset": offset_h if offset_h != 1.0 else None,
                    "Anima Ref fill": fill if fill != 255 else None,
                }
            )
        p.extra_generation_params.update(params)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if not self._active or not patches.is_anima(p.sd_model):
            return
        problem = patches.prepare(p.sd_model, self._adapter)
        if problem:
            print(f"{TAG} {problem}; the reference is skipped")
            return
        unet = p.sd_model.forge_objects.unet
        strength, ref_guidance, correction, rescale_phi, offset_h, fill = self._settings
        if self._method == ROLE_REF_KV:
            if "kv" not in self._latents:  # one encode at the reference's own size; each pass size gets its cache
                self._latents["kv"] = patches.encode(p.sd_model, to_pixels(kv.resize_reference(self._image)))
            p.sd_model.forge_objects.unet = patches.patch_unet_kv(unet, patches.RoleRefKVReference(self._latents["kv"], self._adapter, strength))
            return

        x = kwargs["x"]
        height, width = x.shape[-2] * 8, x.shape[-1] * 8  # this pass's size (hires fix runs a second, larger one)
        key = (width, height, fill)
        if key not in self._latents:
            self._latents[key] = patches.encode(p.sd_model, to_pixels(letterbox(self._image, width, height, fill)))
        if kwargs.get("uc") is None and ref_guidance != 1.0:
            # at CFG 1 Forge drops the negative prompt (setup_conds), and reference guidance needs it
            print(f"{TAG} CFG is 1, so this pass has no negative prompt: reference guidance {ref_guidance} runs as 1")
            ref_guidance = 1.0
            if not getattr(p, "is_hr_pass", False):
                p.extra_generation_params["Anima Ref guidance"] = 1.0
        reference = patches.RefAdapterReference([self._latents[key]], self._adapter, strength, offset_h)
        p.sd_model.forge_objects.unet = patches.patch_unet(unet, reference, ref_guidance, correction, rescale_phi)

    def postprocess(self, p, processed, *args):
        self._active = False
        self._latents = {}
        self._image = None
