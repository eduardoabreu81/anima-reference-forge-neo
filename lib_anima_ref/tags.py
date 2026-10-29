# Anima Reference for Forge Neo
# The methods are switched on by their own LoRA tag in the prompt, <lora:anima_ref_adapter_pilot_r32:1>, as Forge
# Neo's Anima Edit is: the extension takes those tags out of what Forge's LoRA system loads (their weights must only
# run inside a reference pass) and keeps their names and weights for the sampling. The tag stays in the prompt and
# in the infotext.

import os


def lora_names(path: str, directories: list[str]) -> set[str]:
    """The names a <lora:...> tag may use for this file: its file name, or its path under a LoRA folder, without
    the extension (case-insensitive, either slash)."""
    names = {os.path.splitext(os.path.basename(path))[0].lower()}
    for directory in directories:
        if directory:
            try:
                rel = os.path.relpath(path, directory)
            except ValueError:  # another drive
                continue
            if not rel.startswith(".."):
                names.add(os.path.splitext(rel)[0].replace("\\", "/").lower())
    return names


def multiplier(positional: list, named: dict) -> float:
    """The weight Forge's LoRA uses for the model: <lora:name:te:unet>, te: / unet: named, 1 when absent."""
    te = (0.0 if "@" in str(positional[1]) else float(positional[1])) if len(positional) > 1 else 1.0
    te = float(named.get("te", te))
    unet = float(positional[2]) if len(positional) > 2 else te
    return float(named.get("unet", unet))


def split(params_list: list, resolve) -> tuple[list, list]:
    """(params Forge's LoRA should still load, [(name, path, weight)] of the adapter tags); resolve(name) gives an
    adapter file's path or None."""
    keep, found = [], []
    for params in params_list:
        positional = list(getattr(params, "positional", None) or getattr(params, "items", None) or [])
        path = resolve(str(positional[0])) if positional else None
        if path is None:
            keep.append(params)
        else:
            found.append((str(positional[0]), path, multiplier(positional, getattr(params, "named", {}) or {})))
    return keep, found
