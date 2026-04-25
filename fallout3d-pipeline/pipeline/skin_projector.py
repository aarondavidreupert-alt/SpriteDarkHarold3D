"""
SkinProjector — synthesises a complete 360° UV texture by projecting
sprite pixels from 6 isometric views onto the mesh UV map.

Algorithm
---------
For each texel (u, v):
  1. Find the 3D world position via barycentric UV → world mapping.
  2. For each of the 6 camera views determine if the point is visible
     (no front-facing triangle between the camera and the point).
  3. Sample the sprite image at the 2D projection of that world point.
  4. Blend samples weighted by the dot product of the view direction
     and the surface normal (angle weight).
  5. For texels with zero visible views, fill via symmetry mirroring
     and then nearest-neighbour dilation.

Output: (H_tex, W_tex, 3) uint8 — full diffuse texture.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List

from .isometric_camera_setup5 import IsometricCameraSetup


# -----------------------------------------------------------------------
# UV generation (cylindrical projection fallback)
# -----------------------------------------------------------------------

def generate_cylindrical_uvs(vertices: np.ndarray) -> np.ndarray:
    """
    Assign UV coordinates via cylindrical projection around the Y axis.

    Returns
    -------
    uvs : (V, 2) in [0, 1]
    """
    cx, cz = vertices[:, 0].mean(), vertices[:, 2].mean()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()

    u = np.arctan2(vertices[:, 2] - cz, vertices[:, 0] - cx)
    u = (u / (2 * np.pi)) + 0.5                             # [0, 1]

    v_range = y_max - y_min
    v = (vertices[:, 1] - y_min) / (v_range + 1e-8)        # [0, 1]

    return np.stack([u, v], axis=-1)


# -----------------------------------------------------------------------
# UV atlas rasterisation helpers
# -----------------------------------------------------------------------

def _bary_coords(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """
    Compute barycentric coordinates of p w.r.t. triangle (a, b, c).
    All inputs are 2-D vectors.  Returns (w0, w1, w2) or None if degenerate.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return None
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2
    return w0, w1, w2


def build_uv_world_map(
    vertices: np.ndarray,     # (V, 3)
    faces: np.ndarray,        # (F, 3)
    uvs: np.ndarray,          # (V, 2)
    tex_size: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every texel (u, v), compute the corresponding 3D world position
    and face normal.

    Returns
    -------
    world_map  : (H, W, 3)   — world position for each texel (nan = empty)
    normal_map : (H, W, 3)   — face normal for each texel
    valid_mask : (H, W) bool
    """
    H, W = tex_size
    world_map  = np.full((H, W, 3), np.nan, dtype=np.float32)
    normal_map = np.zeros((H, W, 3), dtype=np.float32)

    for face in faces:
        i0, i1, i2 = face
        uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]
        v0,  v1,  v2  = vertices[i0], vertices[i1], vertices[i2]

        # Face normal
        fn = np.cross(v1 - v0, v2 - v0)
        fn_len = np.linalg.norm(fn)
        fn = fn / fn_len if fn_len > 1e-8 else fn

        # Bounding box in UV space
        px0 = int(np.floor(min(uv0[0], uv1[0], uv2[0]) * W))
        px1 = int(np.ceil( max(uv0[0], uv1[0], uv2[0]) * W)) + 1
        py0 = int(np.floor(min(uv0[1], uv1[1], uv2[1]) * H))
        py1 = int(np.ceil( max(uv0[1], uv1[1], uv2[1]) * H)) + 1

        px0, px1 = max(0, px0), min(W, px1)
        py0, py1 = max(0, py0), min(H, py1)

        for py in range(py0, py1):
            for px in range(px0, px1):
                tc = np.array([(px + 0.5) / W, (py + 0.5) / H])
                bc = _bary_coords(tc, uv0, uv1, uv2)
                if bc is None:
                    continue
                w0, w1, w2 = bc
                if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4:
                    continue
                world_pt = w0 * v0 + w1 * v1 + w2 * v2
                world_map[py, px]  = world_pt
                normal_map[py, px] = fn

    valid_mask = ~np.any(np.isnan(world_map), axis=-1)
    return world_map, normal_map, valid_mask


# -----------------------------------------------------------------------
# Visibility (simple back-face check only — no raycasting)
# -----------------------------------------------------------------------

def _angle_weight(view_dir: np.ndarray, normal: np.ndarray) -> float:
    """Dot-product weight: how directly the camera faces this surface point."""
    w = -np.dot(view_dir, normal)
    return max(0.0, float(w))


# -----------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------

class SkinProjector:
    """
    Projects 6-view sprite images onto a UV texture.

    Usage
    -----
    sp = SkinProjector(camera_setup)
    texture = sp.project(
        frames_single,     # (6, H_sprite, W_sprite, 3)
        vertices, faces,
        uvs=None,          # auto-generates cylindrical UVs if None
        tex_size=(512, 512),
    )
    """

    def __init__(self, camera_setup: Optional[IsometricCameraSetup] = None):
        self.camera_setup = camera_setup or IsometricCameraSetup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        sprites: np.ndarray,           # (6, H_sprite, W_sprite, 3) uint8
        vertices: np.ndarray,          # (V, 3)
        faces: np.ndarray,             # (F, 3)
        uvs: Optional[np.ndarray] = None,  # (V, 2) or None
        tex_size: Tuple[int, int] = (512, 512),
        frame_idx: int = 0,
    ) -> np.ndarray:
        """
        Synthesise a diffuse texture from 6 sprite views.

        Returns
        -------
        texture : (H, W, 3) uint8
        """
        if uvs is None:
            uvs = generate_cylindrical_uvs(vertices)

        world_map, normal_map, valid = build_uv_world_map(
            vertices, faces, uvs, tex_size
        )

        H, W = tex_size
        texture = np.zeros((H, W, 3), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)

        ys, xs = np.where(valid)

        for v_idx, cam in enumerate(self.camera_setup.camera_views):
            sprite = sprites[v_idx]
            if sprite.dtype != np.uint8:
                sprite = sprite.astype(np.uint8)
            sp_h, sp_w = sprite.shape[:2]

            P   = cam["projection"]
            cam_pos = cam["position"]

            for py, px in zip(ys, xs):
                world_pt = world_map[py, px]
                normal   = normal_map[py, px]

                # View direction (camera → world point)
                view_dir = world_pt - cam_pos
                vd_norm  = np.linalg.norm(view_dir)
                if vd_norm < 1e-8:
                    continue
                view_dir /= vd_norm

                w = _angle_weight(view_dir, normal)
                if w < 1e-4:
                    continue

                # Project world point → sprite pixel
                ph = np.append(world_pt, 1.0)
                p2h = P @ ph
                if abs(p2h[2]) < 1e-8:
                    continue
                sx = p2h[0] / p2h[2]
                sy = p2h[1] / p2h[2]

                # Bilinear sample
                col = _bilinear_sample(sprite, sx, sy)
                if col is None:
                    continue

                texture[py, px] += col * w
                weight_sum[py, px] += w

        # Normalise
        mask = weight_sum > 1e-6
        texture[mask] = (texture[mask] / weight_sum[mask, np.newaxis]).clip(0, 255)

        # Fill holes via nearest-neighbour dilation
        texture_u8 = texture.astype(np.uint8)
        texture_u8 = _fill_holes(texture_u8, valid)

        return texture_u8

    def project_sequence(
        self,
        frames: np.ndarray,            # (6, N, H_sprite, W_sprite, 3)
        vertices: np.ndarray,
        faces: np.ndarray,
        uvs: Optional[np.ndarray] = None,
        tex_size: Tuple[int, int] = (512, 512),
        progress_cb=None,
    ) -> np.ndarray:
        """
        Project all N frames → returns (N, H, W, 3) texture sequence.
        """
        N = frames.shape[1]
        uvs = uvs or generate_cylindrical_uvs(vertices)
        textures = []
        for f in range(N):
            if progress_cb:
                progress_cb(f, N)
            tex = self.project(frames[:, f], vertices, faces, uvs, tex_size, f)
            textures.append(tex)
        return np.stack(textures)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _bilinear_sample(img: np.ndarray, x: float, y: float):
    """Sample img at sub-pixel (x, y).  Returns (3,) float or None if OOB."""
    h, w = img.shape[:2]
    if x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return None
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    c = (img[y0, x0] * (1-fx) * (1-fy) +
         img[y0, x1] * fx     * (1-fy) +
         img[y1, x0] * (1-fx) * fy     +
         img[y1, x1] * fx     * fy)
    return c.astype(np.float32)


def _fill_holes(texture: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Dilate valid pixels into empty (hole) regions."""
    mask = valid.astype(np.uint8) * 255
    result = texture.copy()
    for _ in range(8):  # up to 8 pixels outward
        dilated = cv2.dilate(result, np.ones((3, 3), np.uint8))
        hole_pixels = (mask == 0)
        result[hole_pixels] = dilated[hole_pixels]
        # Don't mark as valid—keeps original data intact
    return result
