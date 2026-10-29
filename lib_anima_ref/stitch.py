# Anima Reference for Forge Neo
# The reference picture comes from Forge Neo's ImageStitch Integrated, the usual place for reference pictures: the
# panel reads that script's arguments (enable, gallery, maximum side length) from the generation and takes the first
# picture of its gallery. ImageStitch itself only feeds Anima's references with "[Anima] Enable Reference" on (for
# the Edit LoRA), so with that setting off it leaves the picture to this extension.

TITLE = "ImageStitch Integrated"


def find_args(alwayson_scripts, script_args) -> tuple | None:
    """ImageStitch's (enable, gallery, max_dim) for this generation, or None when it is not installed."""
    for script in alwayson_scripts or []:
        try:
            if script.title() == TITLE:
                return tuple(script_args[script.args_from : script.args_to])
        except Exception:
            continue
    return None


def gallery_images(gallery, decode=None) -> list:
    """The pictures of a Gradio gallery value: (image, caption) pairs from the UI, base64 strings from the API,
    or {"image": ...} / {"name": ...} dicts."""
    images = []
    for item in gallery or []:
        if isinstance(item, (tuple, list)):
            item = item[0] if item else None
        if isinstance(item, dict):
            item = item.get("image") or item.get("name") or item.get("path")
        if isinstance(item, str) and decode is not None:
            item = decode(item)
        if item is not None:
            images.append(item)
    return images


def reference(alwayson_scripts, script_args, decode=None) -> tuple:
    """(first picture or None, number of pictures, why there is none)."""
    args = find_args(alwayson_scripts, script_args)
    if args is None:
        return None, 0, f"{TITLE} is not available in this Forge Neo"
    enable, gallery = (args + (None, None))[:2]
    if not enable:
        return None, 0, f"turn on {TITLE} and put the reference picture in its gallery"
    images = gallery_images(gallery, decode)
    if not images:
        return None, 0, f"{TITLE} has no picture: add the reference to its gallery"
    return images[0], len(images), None
