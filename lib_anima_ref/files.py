# Anima Reference for Forge Neo
# Finds each method's weights among the LoRA files. Only names with "anima" in them are looked at (a LoRA folder
# can hold thousands of files); of those, the released files say what they are in their safetensors metadata
# (anima_ref_adapter for Ref Adapter, charref_version for Role Ref KV), and a re-saved copy without metadata is
# still listed when its name says so ("anima_ref" / "role_ref_kv").

import os

from lib_anima_ref.adapter import REF_ADAPTER_KEY, ROLE_REF_KV_KEY, read_header_metadata

REF_ADAPTER = "Ref Adapter"
ROLE_REF_KV = "Role Ref KV"
METHODS = (REF_ADAPTER, ROLE_REF_KV)

_listing: dict[str, tuple[float, dict | None]] = {}  # path -> (mtime, metadata), so a refresh reads new files only


def metadata(path: str) -> dict | None:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _listing.get(path)
    if cached is None or cached[0] != mtime:
        cached = _listing[path] = (mtime, read_header_metadata(path))
    return cached[1]


def method_of(path: str) -> str | None:
    """Which method a LoRA file belongs to, or None."""
    name = os.path.basename(path).lower()
    if "anima" not in name:
        return None
    meta = metadata(path) or {}
    if REF_ADAPTER_KEY in meta:
        return REF_ADAPTER
    if ROLE_REF_KV_KEY in meta:
        return ROLE_REF_KV
    simple = name.replace("-", "_")
    if "role_ref_kv" in simple:
        return ROLE_REF_KV
    if "anima_ref" in simple:
        return REF_ADAPTER
    return None


def find_adapters(directories: list[str]) -> dict[str, dict[str, str]]:
    """{method: {name shown in the UI: full path}}, each sorted by name."""
    found: dict[str, dict[str, str]] = {method: {} for method in METHODS}
    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue
        for root, _, names in os.walk(directory, followlinks=True):
            for name in names:
                if not name.lower().endswith(".safetensors"):
                    continue
                path = os.path.join(root, name)
                method = method_of(path)
                if method:
                    found[method].setdefault(os.path.relpath(path, directory).replace(os.sep, "/"), path)
    return {method: dict(sorted(files.items(), key=lambda item: item[0].lower())) for method, files in found.items()}
