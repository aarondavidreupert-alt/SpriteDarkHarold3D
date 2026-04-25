"""
FRMWriter — write valid Fallout 2 .FRM binary files.

Format (big-endian):
  UINT32  version         (4 = Fallout 2)
  UINT16  fps
  UINT16  action_frame
  UINT16  frames_per_dir
  INT16   shift_x[6]
  INT16   shift_y[6]
  UINT32  dir_offset[6]   (byte offset from start of frame-data block)
  UINT32  data_size       (total bytes in frame-data block)
  --- frame data ---
  For each direction d in 0..5:
    For each frame f in 0..frames_per_dir-1:
      UINT16  width
      UINT16  height
      UINT32  pixel_count  (= width * height)
      INT16   offset_x
      INT16   offset_y
      UINT8   pixels[pixel_count]   (palette indices)
"""

import struct
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


# -----------------------------------------------------------------------
# Palette I/O
# -----------------------------------------------------------------------

def read_pal(path: str) -> np.ndarray:
    """
    Read a Fallout .pal file.

    Values are stored in [0, 63] and scaled to [0, 255].
    Returns (256, 3) uint8 array.
    """
    with open(path, "rb") as f:
        raw = f.read(768)
    pal = np.frombuffer(raw[:768], dtype=np.uint8).reshape(256, 3)
    return np.clip(pal.astype(np.uint16) * 4, 0, 255).astype(np.uint8)


def build_default_palette() -> np.ndarray:
    """Create a simple 256-colour greyscale/rainbow palette as fallback."""
    pal = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        pal[i] = [i, i, i]   # greyscale
    return pal


# -----------------------------------------------------------------------
# Palette quantisation
# -----------------------------------------------------------------------

def quantize_frame(
    frame_rgb: np.ndarray,   # (H, W, 3) uint8
    palette: np.ndarray,     # (256, 3) uint8
) -> np.ndarray:
    """
    Map every pixel to the nearest palette entry using squared Euclidean distance.

    Returns (H, W) uint8 — palette indices.
    """
    h, w = frame_rgb.shape[:2]
    pixels = frame_rgb.reshape(-1, 3).astype(np.float32)  # (H*W, 3)
    pal_f  = palette.astype(np.float32)                    # (256, 3)

    # Batch nearest-neighbour: compute distances for chunks to save memory
    chunk = 4096
    indices = np.zeros(len(pixels), dtype=np.uint8)
    for start in range(0, len(pixels), chunk):
        end  = min(start + chunk, len(pixels))
        diff = pixels[start:end, np.newaxis] - pal_f[np.newaxis]   # (c, 256, 3)
        dist = (diff ** 2).sum(axis=-1)                             # (c, 256)
        indices[start:end] = dist.argmin(axis=-1).astype(np.uint8)

    return indices.reshape(h, w)


def quantize_batch(
    frames: np.ndarray,   # (N, H, W, 3) uint8
    palette: np.ndarray,  # (256, 3) uint8
) -> np.ndarray:
    """Quantise a batch of frames. Returns (N, H, W) uint8."""
    return np.stack([quantize_frame(frames[i], palette) for i in range(len(frames))])


# -----------------------------------------------------------------------
# Software isometric renderer
# -----------------------------------------------------------------------

def render_isometric_frame(
    verts: np.ndarray,       # (V, 3)
    faces: np.ndarray,       # (F, 3)
    uvs: np.ndarray,         # (V, 2) in [0,1]
    texture: np.ndarray,     # (T_H, T_W, 3) uint8
    P: np.ndarray,           # (3, 4) projection matrix
    image_size: Tuple[int, int] = (200, 300),
) -> np.ndarray:
    """
    Software rasteriser: returns (H, W, 4) RGBA uint8.
    Background is transparent (alpha = 0).
    """
    H, W = image_size
    img    = np.zeros((H, W, 4), dtype=np.uint8)
    z_buf  = np.full((H, W), np.inf, dtype=np.float32)
    T_H, T_W = texture.shape[:2]

    # Project all vertices
    v_h  = np.hstack([verts, np.ones((len(verts), 1), dtype=np.float64)])
    p_h  = (P @ v_h.T).T           # (V, 3)
    w    = p_h[:, 2:3]
    safe = np.abs(w) > 1e-8
    v2d  = np.where(safe, p_h[:, :2] / np.where(safe, w, 1.0), 0.0)  # (V, 2)
    z_v  = p_h[:, 2]                                                   # (V,)

    for face in faces:
        i0, i1, i2 = face
        p0, p1, p2 = v2d[i0], v2d[i1], v2d[i2]
        z0, z1, z2 = float(z_v[i0]), float(z_v[i1]), float(z_v[i2])
        uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]

        # Bounding box clipped to image
        x0 = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        x1 = min(W - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        y0 = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        y1 = min(H - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
        if x0 > x1 or y0 > y1:
            continue

        # Pixel grid
        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        px = gx.ravel(); py = gy.ravel()

        # Signed-area edge function: (b-a) × (p-a)
        def _edge(ax, ay, bx, by):
            return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

        area = _edge(p0[0], p0[1], p1[0], p1[1])
        area = _edge(p0[0], p0[1], p1[0], p1[1])

        # Barycentric weights
        w0 = _edge(p1[0], p1[1], p2[0], p2[1])
        w1 = _edge(p2[0], p2[1], p0[0], p0[1])
        w2 = _edge(p0[0], p0[1], p1[0], p1[1])
        total = w0 + w1 + w2
        eps = -1e-4 * (np.abs(total).max() + 1e-10)
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue

        total_safe = np.where(np.abs(total) > 1e-12, total, 1.0)
        b0 = w0 / total_safe
        b1 = w1 / total_safe
        b2 = w2 / total_safe

        z_interp = b0 * z0 + b1 * z1 + b2 * z2

        pxi = px[inside].astype(int)
        pyi = py[inside].astype(int)
        zi  = z_interp[inside]
        b0i = b0[inside]; b1i = b1[inside]; b2i = b2[inside]

        cur_z = z_buf[pyi, pxi]
        upd   = zi < cur_z
        if not upd.any():
            continue

        pxu = pxi[upd]; pyu = pyi[upd]
        b0u = b0i[upd]; b1u = b1i[upd]; b2u = b2i[upd]

        z_buf[pyu, pxu] = zi[upd]

        u = (b0u * uv0[0] + b1u * uv1[0] + b2u * uv2[0])
        v = (b0u * uv0[1] + b1u * uv1[1] + b2u * uv2[1])
        tx = np.clip((u * T_W).astype(int), 0, T_W - 1)
        ty = np.clip((v * T_H).astype(int), 0, T_H - 1)

        img[pyu, pxu, :3] = texture[ty, tx]
        img[pyu, pxu,  3] = 255

    return img


def generate_shadow_mesh(
    verts: np.ndarray,      # (V, 3)
    faces: np.ndarray,      # (F, 3)
    ground_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project mesh vertices onto the ground plane and return a flat shadow mesh
    (convex hull of the silhouette).
    """
    shadow_verts = verts.copy()
    shadow_verts[:, 1] = ground_y + 0.001  # barely above ground
    return shadow_verts, faces


# -----------------------------------------------------------------------
# FRM Writer
# -----------------------------------------------------------------------

class FRMWriter:
    """
    Converts (6, N, H, W, 3) uint8 RGB frames into a valid Fallout 2 .FRM.

    Usage
    -----
    writer = FRMWriter()
    writer.load_palette("data/color.pal")
    writer.write("critter.frm", frames_rgb, fps=10)
    """

    def __init__(self):
        self.palette: Optional[np.ndarray] = None   # (256, 3) uint8

    def load_palette(self, path: str):
        self.palette = read_pal(path)

    def write(
        self,
        output_path: str,
        frames: np.ndarray,          # (6, N, H, W, 3) uint8
        fps: int = 10,
        action_frame: int = 0,
        shift_x: Optional[List[int]] = None,
        shift_y: Optional[List[int]] = None,
    ):
        """
        Write frames as a valid .FRM file.

        Parameters
        ----------
        frames        : (6, N, H, W, 3) uint8 RGB
        fps           : animation playback speed
        action_frame  : frame index where the action trigger fires
        shift_x/y     : per-direction pixel offsets (default: centred)
        """
        if self.palette is None:
            self.palette = build_default_palette()

        assert frames.ndim == 5, "Expected (6, N, H, W, 3)"
        n_dirs, n_frames, H, W = frames.shape[:4]
        assert n_dirs == 6, "FRM always has 6 directions"

        if shift_x is None:
            shift_x = [0] * 6
        if shift_y is None:
            shift_y = [0] * 6

        # Quantise all frames to palette indices
        pal_frames = np.zeros((n_dirs, n_frames, H, W), dtype=np.uint8)
        for d in range(n_dirs):
            for f in range(n_frames):
                pal_frames[d, f] = quantize_frame(frames[d, f], self.palette)

        # Build per-direction frame data blobs
        dir_blobs: List[bytes] = []
        for d in range(n_dirs):
            blob = bytearray()
            for f in range(n_frames):
                blob += struct.pack(">HH", W, H)                         # width, height
                blob += struct.pack(">I", W * H)                         # pixel_count
                blob += struct.pack(">hh", shift_x[d], shift_y[d])      # offset_x/y
                blob += pal_frames[d, f].tobytes()
            dir_blobs.append(bytes(blob))

        # Compute directory offsets (from start of frame-data block)
        dir_offsets = [0] * 6
        cursor = 0
        for d in range(n_dirs):
            dir_offsets[d] = cursor
            cursor += len(dir_blobs[d])

        data_size = cursor

        # Build header
        header = struct.pack(">I", 4)                                  # version
        header += struct.pack(">HHH", fps, action_frame, n_frames)    # fps/action/count
        header += struct.pack(">6h", *shift_x)                        # shift_x[6]
        header += struct.pack(">6h", *shift_y)                        # shift_y[6]
        header += struct.pack(">6I", *dir_offsets)                    # dir_offset[6]
        header += struct.pack(">I", data_size)                        # data_size

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(header)
            for blob in dir_blobs:
                f.write(blob)

    # ------------------------------------------------------------------
    # Spritesheet PNG export
    # ------------------------------------------------------------------

    def write_spritesheet(
        self,
        output_path: str,
        frames: np.ndarray,       # (6, N, H, W, 3)
    ) -> np.ndarray:
        """
        Export all frames as a single PNG spritesheet (6 rows × N columns).
        Returns the spritesheet as (6*H, N*W, 3) uint8.
        """
        import cv2
        n_dirs, n_frames, H, W = frames.shape[:4]
        sheet = np.zeros((H * n_dirs, W * n_frames, 3), dtype=np.uint8)
        for d in range(n_dirs):
            for f in range(n_frames):
                sheet[d*H:(d+1)*H, f*W:(f+1)*W] = frames[d, f]
        cv2.imwrite(output_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        return sheet

    # ------------------------------------------------------------------
    # imageMap JSON (DarkHarold2 format)
    # ------------------------------------------------------------------

    def write_image_map(
        self,
        output_path: str,
        frames: np.ndarray,       # (6, N, H, W, 3)
        name: str = "critter",
        fps: int = 10,
    ):
        """
        Write a .json imageMap compatible with DarkHarold2's sprite loader.
        """
        import json
        n_dirs, n_frames, H, W = frames.shape[:4]
        data = {
            "name": name,
            "fps": fps,
            "directions": n_dirs,
            "frames": n_frames,
            "frameWidth": W,
            "frameHeight": H,
            "frames_data": [
                {
                    "direction": d,
                    "frame": f,
                    "x": f * W,
                    "y": d * H,
                    "w": W,
                    "h": H,
                }
                for d in range(n_dirs)
                for f in range(n_frames)
            ],
        }
        with open(output_path, "w") as fp:
            json.dump(data, fp, indent=2)
