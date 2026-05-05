"""
Iterative 4D Visual Hull — Per-Bone Voxel Carving.

Each BoneSausage owns a voxel grid initialised as a capsule in bone-local
space.  For every animation frame, the bone's current world transform is
used to project all occupied voxels into the 6 silhouette masks.  Voxels
that fall outside the silhouette are permanently removed.  After all frames
have been carved, each sausage's surviving voxels are baked to a triangle
mesh via marching cubes.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from .skeleton_builder import BONE_HIERARCHY, BONE_NAMES

_logger = logging.getLogger(__name__)

# Joints too small/degenerate to carve usefully
_SKIP_JOINTS = frozenset({0, 7, 8, 17, 18, 19, 20, 21, 22})


# ---------------------------------------------------------------------------
# BoneSausage
# ---------------------------------------------------------------------------

class BoneSausage:
    """Voxel grid (NxNxN) living in bone-local space."""

    def __init__(self, joint_idx: int, parent_idx: int, radius: float,
                 resolution: int = 32):
        self.joint_idx  = joint_idx
        self.parent_idx = parent_idx
        self.radius     = radius
        self.resolution = resolution

        self.voxels: np.ndarray = np.zeros((resolution, resolution, resolution),
                                           dtype=bool)
        self.grid_origin:  np.ndarray = np.zeros(3, dtype=np.float64)
        self.voxel_size:   float = 1.0

        self.verts_local: Optional[np.ndarray] = None
        self.faces:       Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def _init_voxels(self, head_local: np.ndarray, tail_local: np.ndarray):
        """Initialise voxels as a capsule in bone-local space."""
        r = self.radius
        N = self.resolution

        # In bone-local space, head is at origin and tail is along +Z
        bone_len = float(np.linalg.norm(tail_local - head_local))

        # Grid extents
        x_min, x_max = -r * 1.5,  r * 1.5
        y_min, y_max = -r * 1.5,  r * 1.5
        z_min, z_max = -r * 0.5,  bone_len + r * 0.5

        # voxel_size is uniform
        extent = max(x_max - x_min, y_max - y_min, z_max - z_min)
        voxel_size = extent / N
        self.voxel_size = voxel_size
        self.grid_origin = np.array([x_min, y_min, z_min], dtype=np.float64)

        # Voxel centre coordinates in local space: (N,N,N,3)
        ii = np.arange(N)
        gx, gy, gz = np.meshgrid(ii, ii, ii, indexing="ij")
        centres = (self.grid_origin +
                   np.stack([gx, gy, gz], axis=-1) * voxel_size)  # (N,N,N,3)
        pts = centres.reshape(-1, 3)  # (M, 3)

        # Capsule signed-distance: distance to the segment [head_local, tail_local]
        seg = tail_local - head_local  # (3,) along Z
        seg_len = float(np.linalg.norm(seg)) + 1e-12
        seg_unit = seg / seg_len
        # Project pts onto the segment
        t = np.clip(np.dot(pts - head_local, seg_unit), 0.0, seg_len)
        closest = head_local + np.outer(t, seg_unit)  # (M,3)
        dist = np.linalg.norm(pts - closest, axis=-1)  # (M,)

        self.voxels = (dist <= r).reshape(N, N, N)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_bone_matrix(head_world: np.ndarray,
                           tail_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (local_to_world (4,4), world_to_local (4,4)).
        Local frame: origin at head_world, Z along bone direction.
        """
        z = tail_world - head_world
        z_len = np.linalg.norm(z)
        if z_len < 1e-12:
            z = np.array([0.0, 0.0, 1.0])
        else:
            z = z / z_len

        # Gram-Schmidt: pick up = [0,0,1], fall back to [1,0,0] if Z is vertical
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z, up)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
        x = up - np.dot(up, z) * z
        x /= np.linalg.norm(x) + 1e-12
        y = np.cross(z, x)
        y /= np.linalg.norm(y) + 1e-12

        l2w = np.eye(4, dtype=np.float64)
        l2w[:3, 0] = x
        l2w[:3, 1] = y
        l2w[:3, 2] = z
        l2w[:3, 3] = head_world

        w2l = np.linalg.inv(l2w)
        return l2w, w2l

    # ------------------------------------------------------------------

    def carve_frame(self, head_world: np.ndarray, tail_world: np.ndarray,
                    silhouette_masks, camera_setup):
        """
        Remove voxels that fall outside any silhouette in this frame.

        silhouette_masks: list of 6 (H, W) binary arrays.
        camera_setup: IsometricCameraSetup with .camera_views and .image_size.
        """
        if not self.voxels.any():
            return

        l2w, _w2l = self._build_bone_matrix(head_world, tail_world)

        # Enumerate occupied voxel centres in local space
        idx = np.argwhere(self.voxels)              # (M, 3) int indices
        if len(idx) == 0:
            return
        local_pts = self.grid_origin + idx * self.voxel_size  # (M, 3)

        # Transform to world space
        M = len(local_pts)
        local_h = np.hstack([local_pts, np.ones((M, 1))])  # (M, 4)
        world_pts = (l2w @ local_h.T).T[:, :3]              # (M, 3)

        W_img, H_img = camera_setup.image_size
        pts_h = np.hstack([world_pts, np.ones((M, 1))])     # (M, 4)

        # Accumulate "survive" mask: a voxel survives if it's inside EVERY
        # direction's silhouette (or out of bounds — treat as silhouette)
        carve = np.zeros(M, dtype=bool)

        for d, view in enumerate(camera_setup.camera_views):
            P = view["projection"]                  # (3, 4)
            p2h = (P @ pts_h.T).T                  # (M, 3)
            w_coord = p2h[:, 2]
            safe = np.abs(w_coord) > 1e-12
            px = np.full(M, -1.0)
            py = np.full(M, -1.0)
            px[safe] = p2h[safe, 0] / w_coord[safe]
            py[safe] = p2h[safe, 1] / w_coord[safe]

            in_bounds = (safe &
                         (px >= 0) & (px < W_img) &
                         (py >= 0) & (py < H_img))

            mask_d = silhouette_masks[d]            # (H, W) binary
            ipx = np.clip(px.astype(int), 0, W_img - 1)
            ipy = np.clip(py.astype(int), 0, H_img - 1)
            hit = mask_d[ipy, ipx].astype(bool)    # (M,)

            # Carve if in-bounds AND outside silhouette
            carve |= in_bounds & ~hit

            if not self.voxels.any():
                break

        if carve.any():
            self.voxels[idx[carve, 0], idx[carve, 1], idx[carve, 2]] = False

    # ------------------------------------------------------------------

    def bake_mesh(self):
        """Run marching cubes on the surviving voxels."""
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:
            raise ImportError(
                "scikit-image is required for bake_mesh — "
                "pip install scikit-image"
            ) from exc

        if not self.voxels.any():
            self.verts_local = np.zeros((0, 3), dtype=np.float64)
            self.faces = np.zeros((0, 3), dtype=np.int32)
            return

        vol = self.voxels.astype(np.float32)
        try:
            verts_idx, faces, _, _ = marching_cubes(vol, level=0.5)
        except ValueError:
            self.verts_local = np.zeros((0, 3), dtype=np.float64)
            self.faces = np.zeros((0, 3), dtype=np.int32)
            return

        self.verts_local = self.grid_origin + verts_idx * self.voxel_size
        self.faces = faces.astype(np.int32)


# ---------------------------------------------------------------------------
# VoxelCarver
# ---------------------------------------------------------------------------

class VoxelCarver:
    """
    Build one BoneSausage per skeletal bone, carve all frames, bake meshes.
    """

    def __init__(self, skeleton_builder, camera_setup, resolution: int = 32):
        self.skeleton_builder = skeleton_builder
        self.camera_setup     = camera_setup
        self.resolution       = resolution
        self.sausages: List[BoneSausage] = []

        self._build_sausages()

    # ------------------------------------------------------------------

    def _build_sausages(self):
        sb = self.skeleton_builder
        poses = sb.poses
        if poses is None or poses.shape[0] == 0:
            return

        pose0 = poses[0]  # (36, 3) world-space joints at frame 0

        # Compute body height from frame 0 for radius estimation
        nose  = pose0[0] if pose0.shape[0] > 0 else pose0[0]
        feet_mid = (pose0[27] + pose0[28]) * 0.5 if pose0.shape[0] > 28 else pose0[0]
        body_height = float(np.linalg.norm(pose0[0] - feet_mid))
        if body_height < 1e-6:
            body_height = 1.0
        base_r = body_height * 0.045

        for joint_idx, parent_idx in BONE_HIERARCHY.items():
            if parent_idx is None:
                continue  # skip root (no parent → no bone)
            if joint_idx in _SKIP_JOINTS:
                continue

            if joint_idx >= pose0.shape[0] or parent_idx >= pose0.shape[0]:
                continue

            head_world = pose0[parent_idx]
            tail_world = pose0[joint_idx]
            bone_len = float(np.linalg.norm(tail_world - head_world))
            if bone_len < 0.01:
                _logger.debug("Skipping joint %d — bone too short (%.4f)", joint_idx, bone_len)
                continue

            sausage = BoneSausage(joint_idx, parent_idx, base_r, self.resolution)

            # In bone-local space the head is at origin, tail is along +Z
            head_local = np.zeros(3, dtype=np.float64)
            tail_local = np.array([0.0, 0.0, bone_len], dtype=np.float64)
            sausage._init_voxels(head_local, tail_local)

            self.sausages.append(sausage)

        _logger.info("VoxelCarver: %d sausages initialised at res=%d",
                     len(self.sausages), self.resolution)

    # ------------------------------------------------------------------

    def carve_all(self, all_silhouette_masks):
        """
        Carve all frames.

        all_silhouette_masks: list of F entries, each a list of 6 (H,W) arrays.
        """
        sb  = self.skeleton_builder
        poses = sb.poses
        if poses is None:
            return

        N = poses.shape[0]
        for f in range(N):
            masks_f = all_silhouette_masks[f]
            for sausage in self.sausages:
                head_w = poses[f, sausage.parent_idx]
                tail_w = poses[f, sausage.joint_idx]
                sausage.carve_frame(head_w, tail_w, masks_f, self.camera_setup)
            if f % 10 == 0:
                _logger.info("VoxelCarver: carved frame %d/%d", f + 1, N)

        _logger.info("VoxelCarver: baking meshes …")
        for sausage in self.sausages:
            sausage.bake_mesh()
        _logger.info("VoxelCarver: done")

    # ------------------------------------------------------------------

    def to_glb_meshes(self) -> List[Dict]:
        """Return one dict per sausage with verts/faces in bone-local space."""
        result = []
        for s in self.sausages:
            if s.verts_local is None or len(s.verts_local) == 0:
                continue
            result.append({
                "joint_idx":   s.joint_idx,
                "bone_name":   BONE_NAMES.get(s.joint_idx, str(s.joint_idx)),
                "verts_local": s.verts_local,
                "faces":       s.faces,
                "skin_weight": 1.0,
            })
        return result
