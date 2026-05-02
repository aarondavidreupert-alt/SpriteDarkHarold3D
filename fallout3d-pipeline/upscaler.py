"""
Unified upscaler backends for the Fallout3D pipeline.

Supported backends
------------------
"edsr"       — OpenCV DNN Super Resolution (EDSR).
               Requires: pip install opencv-contrib-python
               Model:    EDSR_x4.pb from the OpenCV model zoo.

"realesrgan" — Real-ESRGAN neural network.
               Requires: pip install realesrgan basicsr

"torch"      — Generic passthrough for any PyTorch SR model that accepts
               a (B, C, H, W) float32 tensor in [0, 1].
               Useful for SwinIR, SPAN, HAT, etc.

Public API
----------
    upscale_sequence(frames, backend, *, model_path, model_name,
                     torch_model, scale, progress_cb) -> np.ndarray

    REALESRGAN_MODELS   — dict of available Real-ESRGAN model configs
    REALESRGAN_DEFAULT  — default model key
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Real-ESRGAN model registry
# ---------------------------------------------------------------------------

REALESRGAN_MODELS = {
    "RealESRGAN_x4plus": {
        "scale": 4,
        "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "scale": 4,
        "num_block": 6,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    },
    "RealESRGAN_x2plus": {
        "scale": 2,
        "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
}

REALESRGAN_DEFAULT = "RealESRGAN_x4plus_anime_6B"


# ---------------------------------------------------------------------------
# EDSR (OpenCV DNN Super Resolution)
# ---------------------------------------------------------------------------

def _make_edsr(model_path: str, scale: int = 4):
    """Load an EDSR DnnSuperResImpl from a .pb model file."""
    try:
        from cv2 import dnn_superres
    except ImportError:
        raise ImportError(
            "opencv-contrib-python is required for the EDSR backend.\n"
            "Run:  pip install opencv-contrib-python"
        )
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", scale)
    # Uncomment to enable CUDA acceleration:
    # import cv2
    # sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return sr


def _upscale_edsr_frame(frame_rgb: np.ndarray, sr) -> np.ndarray:
    bgr     = frame_rgb[..., ::-1].copy()
    out_bgr = sr.upsample(bgr)
    return out_bgr[..., ::-1]


# ---------------------------------------------------------------------------
# Real-ESRGAN
# ---------------------------------------------------------------------------

def _make_realesrgan(model_name: str):
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        raise ImportError(
            "Real-ESRGAN is not installed.\n"
            "Run:  pip install realesrgan"
        )
    cfg   = REALESRGAN_MODELS[model_name]
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=cfg["num_block"], num_grow_ch=32, scale=cfg["scale"],
    )
    upsampler = RealESRGANer(
        scale=cfg["scale"],
        model_path=cfg["url"],
        model=model,
        tile=0, tile_pad=10, pre_pad=0, half=False,
    )
    return upsampler, cfg["scale"]


def _upscale_realesrgan_frame(frame_rgb: np.ndarray, upsampler, scale: int) -> np.ndarray:
    bgr     = frame_rgb[..., ::-1].copy()
    out_bgr, _ = upsampler.enhance(bgr, outscale=scale)
    return out_bgr[..., ::-1]


# ---------------------------------------------------------------------------
# Generic PyTorch passthrough
# ---------------------------------------------------------------------------

def _upscale_torch_frame(frame_rgb: np.ndarray, model) -> np.ndarray:
    """
    Call a PyTorch SR model.  The model must accept (1, 3, H, W) float32
    in [0, 1] and return (1, 3, H', W').
    """
    import torch
    x   = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
    x   = x.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    out_np = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (out_np * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public batch API
# ---------------------------------------------------------------------------

def upscale_sequence(
    frames: np.ndarray,
    backend: str,
    *,
    model_path: str = "",
    model_name: str = REALESRGAN_DEFAULT,
    torch_model=None,
    scale: int = 4,
    progress_cb=None,
) -> np.ndarray:
    """
    Upscale a full (n_dirs, n_frames, H, W, 3) uint8 RGB array.

    Parameters
    ----------
    frames      : (n_dirs, n_frames, H, W, 3) uint8 RGB
    backend     : "edsr" | "realesrgan" | "torch"
    model_path  : path to EDSR .pb file  (required for "edsr")
    model_name  : Real-ESRGAN model key  (for "realesrgan")
    torch_model : PyTorch model callable (for "torch")
    scale       : upscale factor; EDSR defaults to 4×; ignored for "torch"
    progress_cb : callable(done: int, total: int, msg: str) | None

    Returns
    -------
    np.ndarray (n_dirs, n_frames, H', W', 3) uint8 RGB
    """
    if backend == "edsr":
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"EDSR model not found: '{model_path}'\n"
                "Download EDSR_x4.pb from the OpenCV model zoo and point the\n"
                "model path field to it."
            )
        sr   = _make_edsr(model_path, scale=scale)
        _upscale = lambda img: _upscale_edsr_frame(img, sr)

    elif backend == "realesrgan":
        upsampler, scale = _make_realesrgan(model_name)
        _upscale = lambda img: _upscale_realesrgan_frame(img, upsampler, scale)

    elif backend == "torch":
        if torch_model is None:
            raise ValueError("torch_model must be provided for backend='torch'")
        _upscale = lambda img: _upscale_torch_frame(img, torch_model)

    else:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Choose 'edsr', 'realesrgan', or 'torch'."
        )

    n_dirs, n_frames = frames.shape[:2]
    total = n_dirs * n_frames

    # Probe first frame so we know the exact output shape
    first_out = _upscale(frames[0, 0])
    out_h, out_w = first_out.shape[:2]

    result      = np.zeros((n_dirs, n_frames, out_h, out_w, 3), dtype=np.uint8)
    result[0, 0] = first_out
    done = 1
    if progress_cb:
        progress_cb(done, total, f"Dir 1/{n_dirs}  Frame 1/{n_frames}")

    for d in range(n_dirs):
        for fi in range(n_frames):
            if d == 0 and fi == 0:
                continue
            out = _upscale(frames[d, fi])
            # Guard against rare ±1 pixel size variation
            if out.shape[:2] != (out_h, out_w):
                from PIL import Image as _PIL
                out = np.array(
                    _PIL.fromarray(out).resize((out_w, out_h), _PIL.LANCZOS)
                )
            result[d, fi] = out
            done += 1
            if progress_cb:
                progress_cb(
                    done, total,
                    f"Dir {d + 1}/{n_dirs}  Frame {fi + 1}/{n_frames}",
                )

    return result
