"""
NormalMapBaker — generates tangent-space normal maps and depth maps
from 2D sprite images using Sobel gradient analysis.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional


class NormalMapBaker:
    """
    Bakes normal maps and depth maps from sprite frames.

    For each sprite:
    1. Convert to grayscale as a height-field proxy.
    2. Compute Sobel gradients → tangent-space normals.
    3. Invert and gamma-correct for optimal visual quality.
    4. Optionally apply a depth pass via distance-transform smoothing.
    """

    def __init__(
        self,
        strength: float = 2.0,
        blur_radius: int = 1,
        depth_blur: int = 5,
    ):
        self.strength = strength
        self.blur_radius = blur_radius
        self.depth_blur = depth_blur

    # ------------------------------------------------------------------
    # Core baking
    # ------------------------------------------------------------------

    def bake_normal_map(self, sprite: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        sprite : (H, W, 3|4) uint8 or float

        Returns
        -------
        normal_map : (H, W, 3) uint8  — RGB encoded normal in [0, 255]
        """
        gray = self._to_gray(sprite)

        # Optional blur to reduce noise
        if self.blur_radius > 0:
            ksize = 2 * self.blur_radius + 1
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        gray_f = gray.astype(float) / 255.0

        # Sobel gradients (tangent-space components)
        gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)

        # Build normal vector N = normalize((-Gx*s, -Gy*s, 1))
        nx = -gx * self.strength
        ny = -gy * self.strength
        nz = np.ones_like(nx)

        length = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx /= length
        ny /= length
        nz /= length

        # Map [-1,1] → [0, 255]
        normal_map = np.stack([
            (nx + 1) / 2,
            (ny + 1) / 2,
            (nz + 1) / 2,
        ], axis=-1)
        return (np.clip(normal_map, 0, 1) * 255).astype(np.uint8)

    def bake_depth_map(self, sprite: np.ndarray) -> np.ndarray:
        """
        Estimate a depth map from sprite alpha/luminance.

        Bright pixels = close, dark = far (sprite convention).
        Returns (H, W) uint8 in [0, 255].
        """
        if sprite.shape[2] == 4:
            # Use inverted alpha as depth proxy
            alpha = sprite[:, :, 3].astype(float)
            depth = alpha
        else:
            gray = self._to_gray(sprite).astype(float)
            depth = gray

        if self.depth_blur > 0:
            ksize = 2 * self.depth_blur + 1
            depth = cv2.GaussianBlur(depth.astype(np.float32), (ksize, ksize), 0)

        depth = depth - depth.min()
        dmax = depth.max()
        if dmax > 1e-6:
            depth = depth / dmax
        return (depth * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Batch baking
    # ------------------------------------------------------------------

    def bake_sequence(
        self,
        frames: np.ndarray,
        output_dir: Optional[str] = None,
    ):
        """
        Bake normal + depth maps for every (direction, frame) pair.

        Parameters
        ----------
        frames     : (6, N, H, W, 3|4)
        output_dir : if provided, save PNGs there

        Returns
        -------
        normal_maps : (6, N, H, W, 3) uint8
        depth_maps  : (6, N, H, W)    uint8
        """
        n_dirs, n_frames = frames.shape[0], frames.shape[1]
        H, W = frames.shape[2], frames.shape[3]
        C = frames.shape[4] if frames.ndim == 5 else 3

        normal_maps = np.zeros((n_dirs, n_frames, H, W, 3), dtype=np.uint8)
        depth_maps = np.zeros((n_dirs, n_frames, H, W), dtype=np.uint8)

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for d in range(n_dirs):
            for f in range(n_frames):
                sprite = frames[d, f]
                if sprite.dtype != np.uint8:
                    sprite = (np.clip(sprite, 0, 255)).astype(np.uint8)

                nm = self.bake_normal_map(sprite)
                dm = self.bake_depth_map(sprite)

                normal_maps[d, f] = nm
                depth_maps[d, f] = dm

                if output_dir:
                    cv2.imwrite(f"{output_dir}/normal_d{d+1}_f{f:03d}.png", nm)
                    cv2.imwrite(f"{output_dir}/depth_d{d+1}_f{f:03d}.png", dm)

        return normal_maps, depth_maps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.shape[2] == 4:
            # Composite onto white background
            alpha = img[:, :, 3:4].astype(float) / 255.0
            rgb = img[:, :, :3].astype(float)
            composited = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
            return cv2.cvtColor(composited, cv2.COLOR_RGB2GRAY)
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img[:, :, 0]
