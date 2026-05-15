"""
Per-character .npz sidecar for persisting shadow/barrier edits and the
upscaled frame cache. Lives next to the source .frm/.npy as
``{name}_sprite_data.npz``.

Arrays in the .npz (all optional):
    frames           — edited RGB (6, N, H, W, 3) uint8
    frames_pal_idx   — edited palette indices (6, N, H, W) uint8 (FRM only)
    upscaled         — upscaled frames (6, N, H', W', 3) uint8
    upscaled_backend — 0-d str array naming the backend that produced
                       ``upscaled`` (e.g. "edsr", "realesrgan", "torch")
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def sidecar_path(char) -> Optional[str]:
    if not getattr(char, "source_path", None) or not getattr(char, "name", None):
        return None
    base_dir = os.path.dirname(char.source_path) or os.getcwd()
    return os.path.join(base_dir, f"{char.name}_sprite_data.npz")


def save_sidecar(char, *, upscaled_backend: Optional[str] = None) -> Optional[str]:
    """Write a sidecar containing whatever non-None state the char holds.
    Returns the path written, or None if no source_path/name."""
    p = sidecar_path(char)
    if p is None:
        return None
    arrays: dict[str, np.ndarray] = {}
    if getattr(char, "frames", None) is not None:
        arrays["frames"] = char.frames
    if getattr(char, "frames_pal_idx", None) is not None:
        arrays["frames_pal_idx"] = char.frames_pal_idx
    if getattr(char, "upscaled_frames", None) is not None:
        arrays["upscaled"] = char.upscaled_frames
        if upscaled_backend:
            arrays["upscaled_backend"] = np.array(upscaled_backend)
        else:
            existing = getattr(char, "_sidecar_upscaled_backend", "") or ""
            if existing:
                arrays["upscaled_backend"] = np.array(existing)
    if not arrays:
        return None
    try:
        np.savez_compressed(p, **arrays)
    except Exception:
        return None
    if upscaled_backend:
        char._sidecar_upscaled_backend = upscaled_backend
    return p


def load_sidecar(char) -> Optional[dict]:
    p = sidecar_path(char)
    if p is None or not os.path.exists(p):
        return None
    try:
        with np.load(p, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except Exception:
        return None


def delete_sidecar(char) -> bool:
    p = sidecar_path(char)
    if p and os.path.exists(p):
        try:
            os.remove(p)
            return True
        except Exception:
            return False
    return False


def apply_sidecar_to_char(char) -> tuple[bool, bool]:
    """Apply sidecar contents to ``char`` in place.

    The current ``char.frames``/``char.frames_pal_idx`` (the freshly-loaded
    originals) are stashed into ``frames_backup``/``frames_pal_idx_backup``
    so Restore reverts to the source-file state.

    Returns (edits_loaded, upscaled_loaded).
    """
    data = load_sidecar(char)
    if data is None:
        return False, False

    edits_loaded = False
    if "frames" in data and char.frames is not None:
        sf = data["frames"]
        if sf.shape == char.frames.shape and sf.dtype == char.frames.dtype:
            char.frames_backup = char.frames.copy()
            char.frames = sf.copy()
            edits_loaded = True

    if ("frames_pal_idx" in data
            and getattr(char, "frames_pal_idx", None) is not None):
        sp = data["frames_pal_idx"]
        if sp.shape == char.frames_pal_idx.shape:
            char.frames_pal_idx_backup = char.frames_pal_idx.copy()
            char.frames_pal_idx = sp.copy()
            edits_loaded = True

    upscaled_loaded = False
    if "upscaled" in data:
        char.upscaled_frames = data["upscaled"].copy()
        upscaled_loaded = True
        if "upscaled_backend" in data:
            try:
                char._sidecar_upscaled_backend = str(data["upscaled_backend"])
            except Exception:
                pass

    return edits_loaded, upscaled_loaded
