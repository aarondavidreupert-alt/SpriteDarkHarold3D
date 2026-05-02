"""
ShadowSpriteGenerator — orthographic ground-plane shadows for an animated mesh.
"""

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _box_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Approximate a Gaussian blur with iterated box convolution (3 passes).
    Used as a numpy-only fallback when scipy is missing.
    """
    radius = max(1, int(round(sigma)))
    k = 2 * radius + 1
    kernel_1d = np.ones(k, dtype=np.float32) / k

    out = image.astype(np.float32)
    for _ in range(3):
        # Horizontal
        padded = np.pad(out, ((0, 0), (radius, radius)), mode="edge")
        out = np.apply_along_axis(
            lambda r: np.convolve(r, kernel_1d, mode="valid"), 1, padded
        )
        # Vertical
        padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
        out = np.apply_along_axis(
            lambda c: np.convolve(c, kernel_1d, mode="valid"), 0, padded
        )
    return out


class ShadowSpriteGenerator:
    """
    Project a mesh's silhouette onto the Y=0 ground plane and rasterise it
    as a soft alpha shadow.
    """

    def __init__(
        self,
        img_size: int = 128,
        blur_sigma: float = 4.0,
        opacity: float = 0.6,
    ):
        self.img_size = int(img_size)
        self.blur_sigma = float(blur_sigma)
        self.opacity = float(np.clip(opacity, 0.0, 1.0))

    # ------------------------------------------------------------------

    def generate(
        self,
        verts: np.ndarray,
        direction_angle_deg: float,
    ) -> np.ndarray:
        """
        Orthographic projection of mesh vertices onto Y=0 along the
        camera's view direction at `direction_angle_deg`.

        Returns
        -------
        rgba : (img_size, img_size, 4) uint8 — black shadow with soft alpha
        """
        S = self.img_size
        verts = np.asarray(verts, dtype=float)
        if verts.size == 0:
            return np.zeros((S, S, 4), dtype=np.uint8)

        # Camera "forward" direction in 3D (matches IsometricCameraSetup convention)
        a = np.radians(direction_angle_deg)
        cam_pos = np.array([np.cos(a), np.sin(a), 0.0])
        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-12)

        # World "up" is +Z in the pipeline's coords; project verts to ground (z=0).
        # We drop the Z (vertical) component and keep the (x, y) ground footprint,
        # but rotated so the camera's view direction aligns with image -Y axis.
        right = np.array([-np.sin(a), np.cos(a), 0.0])     # tangent in ground plane
        forward_xy = np.array([np.cos(a), np.sin(a), 0.0]) # away from camera
        # u = right · v  (image x), v = forward_xy · v  (image y)
        u = verts @ right
        v = verts @ forward_xy

        # Centre & scale into [0, S)
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()
        span = max(u_max - u_min, v_max - v_min, 1e-6) * 1.1
        u_centre = 0.5 * (u_min + u_max)
        v_centre = 0.5 * (v_min + v_max)

        u_norm = (u - u_centre) / span + 0.5
        v_norm = (v - v_centre) / span + 0.5
        px = np.clip((u_norm * S).astype(int), 0, S - 1)
        py = np.clip((v_norm * S).astype(int), 0, S - 1)

        mask = np.zeros((S, S), dtype=np.float32)
        mask[py, px] = 1.0

        # Soft shadow blur
        if _SCIPY_AVAILABLE:
            blurred = gaussian_filter(mask, sigma=self.blur_sigma)
        else:
            blurred = _box_blur(mask, self.blur_sigma)

        max_val = blurred.max()
        if max_val > 1e-9:
            blurred = blurred / max_val

        alpha = (blurred * self.opacity * 255.0).astype(np.uint8)

        rgba = np.zeros((S, S, 4), dtype=np.uint8)
        rgba[..., 3] = alpha
        return rgba

    # ------------------------------------------------------------------

    def generate_all_directions(self, verts: np.ndarray) -> list:
        """Return six RGBA images at angles 0°, 60°, ..., 300°."""
        return [self.generate(verts, a) for a in (0, 60, 120, 180, 240, 300)]
