"""
Per-bone voxel sausage carver.

Each bone owns a 3-D boolean grid (NxNxN) initialised as a capsule in
bone-local space.  For every frame the bone's world transform is applied to
project all occupied voxels into the 6 silhouette masks; voxels that fall
outside any silhouette are removed.  Voxels live in bone-local space
permanently — "following the bone" is free because the bone matrix is
recomputed fresh each frame.

Weighted mode: keep a float32 hit-count grid.  After all frames, voxels with
count >= threshold * n_frames are kept.  This tolerates brief occlusions.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from .skeleton_builder import BONE_HIERARCHY, BONE_NAMES

_logger = logging.getLogger(__name__)

# Joints too small / degenerate to carve usefully
SKIP_JOINTS: frozenset = frozenset({7, 8, 17, 18, 19, 20, 21, 22})
_HEAD_JOINTS: frozenset = frozenset({0, 36})   # Nose + Head Top — get 2× base radius


# ---------------------------------------------------------------------------
# BoneSausage
# ---------------------------------------------------------------------------

class BoneSausage:
    """NxNxN voxel grid in bone-local space."""

    def __init__(self, joint_idx: int, parent_idx: int, radius: float,
                 resolution: int = 32):
        self.joint_idx   = joint_idx
        self.parent_idx  = parent_idx
        self.radius      = radius          # mutable — updated by user sliders
        self.resolution  = resolution

        self.voxels:     Optional[np.ndarray] = None   # bool  (N,N,N)
        self.weights:    Optional[np.ndarray] = None   # float32 (N,N,N) — weighted mode
        self.grid_origin: np.ndarray = np.zeros(3, dtype=np.float64)
        self.voxel_size:  float = 1.0
        self._bone_len_f0: float = 0.0  # stored for reset()

        self.verts_local: Optional[np.ndarray] = None
        self.faces:       Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def _init_voxels(self, head_local: np.ndarray, tail_local: np.ndarray):
        """
        Grid spans:
          X, Y: [-radius*1.5 .. radius*1.5]
          Z:    [-radius*0.5 .. bone_length + radius*0.5]
        Capsule init is fully vectorised (no Python loop over voxels).
        """
        r = self.radius
        N = self.resolution
        bone_len = float(np.linalg.norm(tail_local - head_local))
        self._bone_len_f0 = bone_len

        x_min, x_max = -r * 1.5, r * 1.5
        y_min, y_max = -r * 1.5, r * 1.5
        z_min, z_max = -r * 0.5, bone_len + r * 0.5

        extent = max(x_max - x_min, y_max - y_min, z_max - z_min)
        voxel_size = extent / max(N - 1, 1)
        self.voxel_size  = voxel_size
        self.grid_origin = np.array([x_min, y_min, z_min], dtype=np.float64)

        # Voxel centres  (N,N,N,3)
        ii = np.arange(N)
        gi, gj, gk = np.meshgrid(ii, ii, ii, indexing="ij")
        centres = self.grid_origin + np.stack([gi, gj, gk], axis=-1) * voxel_size
        pts = centres.reshape(-1, 3)

        # Capsule SDF
        seg_len = float(np.linalg.norm(tail_local - head_local)) + 1e-12
        seg_unit = (tail_local - head_local) / seg_len
        t = np.clip(np.dot(pts - head_local, seg_unit), 0.0, seg_len)
        closest = head_local + np.outer(t, seg_unit)
        dist = np.linalg.norm(pts - closest, axis=-1)

        self.voxels  = (dist <= r).reshape(N, N, N)
        self.weights = None

    # ------------------------------------------------------------------

    @staticmethod
    def _build_bone_matrix(head_world: np.ndarray,
                           tail_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (local→world 4×4, world→local 4×4).
        Origin at head_world, local-Z along bone axis.
        """
        z = tail_world - head_world
        z_len = np.linalg.norm(z)
        z = (z / z_len) if z_len > 1e-12 else np.array([0., 0., 1.])

        up = np.array([0., 0., 1.])
        if abs(np.dot(z, up)) > 0.99:
            up = np.array([1., 0., 0.])
        x = up - np.dot(up, z) * z
        x /= np.linalg.norm(x) + 1e-12
        y  = np.cross(z, x)
        y /= np.linalg.norm(y) + 1e-12

        l2w = np.eye(4, dtype=np.float64)
        l2w[:3, 0] = x
        l2w[:3, 1] = y
        l2w[:3, 2] = z
        l2w[:3, 3] = head_world

        return l2w, np.linalg.inv(l2w)

    # ------------------------------------------------------------------

    def carve_frame(self, head_world: np.ndarray, tail_world: np.ndarray,
                    silhouette_masks, camera_setup, weighted: bool = False):
        """
        Fully vectorised carve (no Python loop over voxels).

        Boolean mode  — voxels outside any camera's silhouette → cleared.
        Weighted mode — voxels inside ALL cameras' silhouettes → weights++.
        """
        if self.voxels is None or not self.voxels.any():
            return

        l2w, _ = self._build_bone_matrix(head_world, tail_world)

        active_idx = np.argwhere(self.voxels)    # (M, 3)
        if len(active_idx) == 0:
            return

        local_pts = self.grid_origin + active_idx * self.voxel_size  # (M, 3)
        M = len(local_pts)
        world_pts = (l2w @ np.hstack([local_pts, np.ones((M, 1))]).T).T[:, :3]

        W_img, H_img = camera_setup.image_size
        pts_h = np.hstack([world_pts, np.ones((M, 1))])   # (M, 4)

        if weighted:
            if self.weights is None:
                self.weights = np.zeros(self.voxels.shape, dtype=np.float32)
            # A voxel is counted if inside ALL cameras' silhouettes.
            # Out-of-bounds voxels get benefit of the doubt (treated as inside).
            all_inside = np.ones(M, dtype=bool)
            for d, view in enumerate(camera_setup.camera_views):
                P   = view["projection"]
                p2h = (P @ pts_h.T).T
                w_c = p2h[:, 2]
                safe = np.abs(w_c) > 1e-12
                px = np.where(safe, p2h[:, 0] / np.where(safe, w_c, 1.), -1.)
                py = np.where(safe, p2h[:, 1] / np.where(safe, w_c, 1.), -1.)
                in_b = safe & (px >= 0) & (px < W_img) & (py >= 0) & (py < H_img)
                ipx  = np.clip(px.astype(int), 0, W_img - 1)
                ipy  = np.clip(py.astype(int), 0, H_img - 1)
                hit  = silhouette_masks[d][ipy, ipx].astype(bool)
                # Not inside this camera = in-bounds AND outside mask
                all_inside &= ~in_b | hit
            self.weights[active_idx[all_inside, 0],
                         active_idx[all_inside, 1],
                         active_idx[all_inside, 2]] += 1.0
        else:
            carve = np.zeros(M, dtype=bool)
            for d, view in enumerate(camera_setup.camera_views):
                P   = view["projection"]
                p2h = (P @ pts_h.T).T
                w_c = p2h[:, 2]
                safe = np.abs(w_c) > 1e-12
                px = np.where(safe, p2h[:, 0] / np.where(safe, w_c, 1.), -1.)
                py = np.where(safe, p2h[:, 1] / np.where(safe, w_c, 1.), -1.)
                in_b = safe & (px >= 0) & (px < W_img) & (py >= 0) & (py < H_img)
                ipx  = np.clip(px.astype(int), 0, W_img - 1)
                ipy  = np.clip(py.astype(int), 0, H_img - 1)
                hit  = silhouette_masks[d][ipy, ipx].astype(bool)
                carve |= in_b & ~hit
                if not self.voxels.any():
                    break
            if carve.any():
                self.voxels[active_idx[carve, 0],
                             active_idx[carve, 1],
                             active_idx[carve, 2]] = False

    # ------------------------------------------------------------------

    def finalise_weighted(self, n_frames: int, threshold: float = 0.3):
        """Convert accumulated hit-counts to boolean voxels."""
        if self.weights is None:
            return
        self.voxels  = self.weights >= (threshold * max(n_frames, 1))
        self.weights = None

    # ------------------------------------------------------------------

    def bake_mesh(self):
        """Run marching cubes; store verts_local and faces."""
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:
            raise ImportError("scikit-image required: pip install scikit-image") from exc

        if self.voxels is None or not self.voxels.any():
            self.verts_local = np.zeros((0, 3), dtype=np.float64)
            self.faces       = np.zeros((0, 3), dtype=np.int32)
            return

        try:
            vi, faces, _, _ = marching_cubes(self.voxels.astype(np.float32), level=0.5)
        except ValueError:
            self.verts_local = np.zeros((0, 3), dtype=np.float64)
            self.faces       = np.zeros((0, 3), dtype=np.int32)
            return

        self.verts_local = self.grid_origin + vi * self.voxel_size
        self.faces       = faces.astype(np.int32)

    # ------------------------------------------------------------------

    def occupied_count(self) -> int:
        return int(self.voxels.sum()) if self.voxels is not None else 0

    def max_radial_distance(self) -> float:
        """
        Maximum distance of any surviving voxel centre from the bone axis
        (local Z axis).  Equals sqrt(x² + y²) in local space.
        Returns 0.0 if no voxels survive.
        """
        if self.voxels is None or not self.voxels.any():
            return 0.0
        idx = np.argwhere(self.voxels)                      # (M, 3)
        pts = self.grid_origin + idx * self.voxel_size       # (M, 3)
        radial = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        return float(radial.max())


# ---------------------------------------------------------------------------
# VoxelCarver
# ---------------------------------------------------------------------------

class VoxelCarver:
    """Build one BoneSausage per skeletal bone, carve frames, bake meshes."""

    def __init__(self, skeleton_builder, camera_setup,
                 bone_radii: Optional[Dict[int, float]] = None,
                 resolution: int = 32):
        self.skeleton_builder = skeleton_builder
        self.camera_setup     = camera_setup
        self.resolution       = resolution
        self._bone_radii      = bone_radii or {}
        self.sausages: Dict[int, BoneSausage] = {}
        self._build_sausages()

    # ------------------------------------------------------------------

    def _build_sausages(self):
        sb    = self.skeleton_builder
        poses = sb.poses
        if poses is None or poses.shape[0] == 0:
            return

        pose0 = poses[0]
        feet_mid    = (pose0[27] + pose0[28]) * 0.5 if pose0.shape[0] > 28 else pose0[0]
        body_height = float(np.linalg.norm(pose0[0] - feet_mid))
        if body_height < 1e-6:
            body_height = 1.0
        base_r = body_height * 0.045

        for joint_idx, parent_idx in BONE_HIERARCHY.items():
            if parent_idx is None or joint_idx in SKIP_JOINTS:
                continue
            if joint_idx >= pose0.shape[0] or parent_idx >= pose0.shape[0]:
                continue
            head_w   = pose0[parent_idx]
            tail_w   = pose0[joint_idx]
            bone_len = float(np.linalg.norm(tail_w - head_w))
            if bone_len < 0.01:
                continue

            default = base_r * 2.0 if joint_idx in _HEAD_JOINTS else base_r
            radius  = self._bone_radii.get(joint_idx, default)
            sausage = BoneSausage(joint_idx, parent_idx, radius, self.resolution)
            sausage._init_voxels(
                np.zeros(3, dtype=np.float64),
                np.array([0., 0., bone_len], dtype=np.float64),
            )
            self.sausages[joint_idx] = sausage

        _logger.info("VoxelCarver: %d sausages at res=%d", len(self.sausages), self.resolution)

    # ------------------------------------------------------------------

    def reset(self):
        """Re-initialise all voxel grids from Frame-0 (keeps current radii)."""
        for sausage in self.sausages.values():
            sausage._init_voxels(
                np.zeros(3, dtype=np.float64),
                np.array([0., 0., sausage._bone_len_f0], dtype=np.float64),
            )

    # ------------------------------------------------------------------

    def carve_frame(self, frame_idx: int, silhouette_masks,
                    weighted: bool = False):
        """Carve a single animation frame for all sausages."""
        poses = self.skeleton_builder.poses
        if poses is None or frame_idx >= poses.shape[0]:
            return
        for sausage in self.sausages.values():
            head_w = poses[frame_idx, sausage.parent_idx]
            tail_w = poses[frame_idx, sausage.joint_idx]
            sausage.carve_frame(head_w, tail_w, silhouette_masks,
                                self.camera_setup, weighted=weighted)

    def carve_all_frames(self, all_masks, weighted: bool = False,
                         progress_cb=None):
        """
        all_masks: list[F] of list[6] of (H,W) uint8 arrays.
        progress_cb(done, total) called after each frame.
        """
        n = len(all_masks)
        for f, masks_f in enumerate(all_masks):
            self.carve_frame(f, masks_f, weighted=weighted)
            if progress_cb is not None:
                progress_cb(f + 1, n)

    # ------------------------------------------------------------------

    def bake_all(self):
        """Run marching cubes on every sausage that has surviving voxels."""
        for sausage in self.sausages.values():
            sausage.bake_mesh()

    # ------------------------------------------------------------------

    def bake_world_grid(
        self,
        resolution: int = 64,
        progress_cb=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bake per-frame meshes through a single fixed world-space grid.

        Stable topology across frames is achieved by:
          * computing the bounding box from ALL transformed voxel points,
          * cubic voxels (same voxel_size on all 3 axes),
          * a single grid_shape used for every frame.

        Frames whose marching-cubes vertex count differs from frame 0 fall
        back to the frame-0 mesh (a warning is logged).

        Returns (mesh_frames float32 (N, V, 3), faces int32 (F, 3)).
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:
            raise ImportError("scikit-image required: pip install scikit-image") from exc

        poses = self.skeleton_builder.poses
        empty = (np.zeros((0, 0, 3), np.float32), np.zeros((0, 3), np.int32))
        if poses is None or poses.shape[0] == 0 or not self.sausages:
            return empty
        n_frames = int(poses.shape[0])

        # ── Step 1: collect transformed voxel points + bbox over all frames ──
        per_frame_points: list[list[np.ndarray]] = []
        bbox_min = np.full(3,  np.inf, np.float64)
        bbox_max = np.full(3, -np.inf, np.float64)
        for f in range(n_frames):
            frame_pts: list[np.ndarray] = []
            for s in self.sausages.values():
                if s.voxels is None or not s.voxels.any():
                    continue
                idx = np.argwhere(s.voxels)
                if len(idx) == 0:
                    continue
                local_pts = s.grid_origin + idx * s.voxel_size
                head_w = poses[f, s.parent_idx]
                tail_w = poses[f, s.joint_idx]
                l2w, _ = BoneSausage._build_bone_matrix(head_w, tail_w)
                M = len(local_pts)
                world_pts = (l2w @ np.hstack([local_pts, np.ones((M, 1))]).T).T[:, :3]
                frame_pts.append(world_pts)
                bbox_min = np.minimum(bbox_min, world_pts.min(axis=0))
                bbox_max = np.maximum(bbox_max, world_pts.max(axis=0))
            per_frame_points.append(frame_pts)

        if not np.all(np.isfinite(bbox_min)):
            return empty

        # ── Step 2: fixed cubic grid ──────────────────────────────────────
        voxel_size = float((bbox_max - bbox_min).max()) / max(resolution - 1, 1)
        if voxel_size < 1e-12:
            return empty
        pad = 2.0 * voxel_size
        grid_lo = bbox_min - pad
        grid_hi = bbox_max + pad
        grid_shape = np.ceil((grid_hi - grid_lo) / voxel_size).astype(int) + 1
        nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])

        def _rasterise(frame_pts: list[np.ndarray]) -> np.ndarray:
            grid = np.zeros((nx, ny, nz), dtype=bool)
            for world_pts in frame_pts:
                gi = np.round((world_pts - grid_lo) / voxel_size).astype(int)
                valid = (np.all(gi >= 0, axis=1) &
                         (gi[:, 0] < nx) & (gi[:, 1] < ny) & (gi[:, 2] < nz))
                gi = gi[valid]
                if len(gi):
                    grid[gi[:, 0], gi[:, 1], gi[:, 2]] = True
            return grid

        # ── Step 3: marching cubes per frame ─────────────────────────────
        grid0 = _rasterise(per_frame_points[0]).astype(np.float32)
        if not grid0.any():
            return empty
        try:
            v0, faces_ref, _, _ = marching_cubes(grid0, level=0.5)
        except ValueError:
            return empty
        V = int(len(v0))
        rest_world = (v0 * voxel_size + grid_lo).astype(np.float32)
        mesh_frames: list[np.ndarray] = [rest_world]
        if progress_cb is not None:
            progress_cb(1, n_frames)

        mismatches = 0
        for f in range(1, n_frames):
            grid_f = _rasterise(per_frame_points[f]).astype(np.float32)
            verts_world = None
            if grid_f.any():
                try:
                    vf, _, _, _ = marching_cubes(grid_f, level=0.5)
                    if len(vf) == V:
                        verts_world = (vf * voxel_size + grid_lo).astype(np.float32)
                    else:
                        _logger.warning(
                            "bake_world_grid frame %d: %d verts ≠ %d (rest pose)",
                            f, len(vf), V)
                except ValueError:
                    pass
            if verts_world is None:
                verts_world = rest_world
                mismatches += 1
            mesh_frames.append(verts_world)
            if progress_cb is not None:
                progress_cb(f + 1, n_frames)

        if mismatches:
            _logger.info("bake_world_grid: %d/%d frames fell back to rest pose",
                         mismatches, n_frames)

        return np.stack(mesh_frames, axis=0), faces_ref.astype(np.int32)

    # ------------------------------------------------------------------

    def world_bounds(self, n_frames: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bounding box of all bone capsules in world space across N frames.
        Uses bone positions + radius (cheap, no voxel expansion needed).
        Returns (lo, hi) as float64 (3,) arrays with a small margin.
        """
        poses = self.skeleton_builder.poses
        if poses is None or not self.sausages:
            return np.zeros(3, np.float64), np.ones(3, np.float64)
        n = int(poses.shape[0]) if n_frames is None else min(n_frames, int(poses.shape[0]))

        lo = np.full(3,  np.inf, np.float64)
        hi = np.full(3, -np.inf, np.float64)
        for f in range(n):
            for s in self.sausages.values():
                r = max(s.radius * 1.6, s.voxel_size * s.resolution * 0.5 + 0.01)
                for pidx in (s.parent_idx, s.joint_idx):
                    p = poses[f, pidx].astype(np.float64)
                    lo = np.minimum(lo, p - r)
                    hi = np.maximum(hi, p + r)
        margin = max(float(np.max(hi - lo)) * 0.03, 0.02)
        return (lo - margin).astype(np.float64), (hi + margin).astype(np.float64)

    def bake_frame_world(
        self,
        frame_idx: int,
        world_lo: np.ndarray,
        world_hi: np.ndarray,
        world_res: int = 64,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Rasterise all sausage voxels for one frame into a fixed world-space
        grid, then run marching cubes.

        The grid dims are determined by world_lo/hi + world_res — identical
        across frames so topology is consistent.
        Returns (verts float32 (V,3), faces int32 (F,3)) or (None, None).
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:
            raise ImportError("scikit-image required: pip install scikit-image") from exc

        poses = self.skeleton_builder.poses
        if poses is None or frame_idx >= poses.shape[0]:
            return None, None

        extent = (world_hi - world_lo).astype(np.float64)
        voxel_size = float(np.max(extent)) / max(world_res - 1, 1)
        if voxel_size < 1e-12:
            return None, None

        nx = max(int(np.ceil(extent[0] / voxel_size)) + 2, 4)
        ny = max(int(np.ceil(extent[1] / voxel_size)) + 2, 4)
        nz = max(int(np.ceil(extent[2] / voxel_size)) + 2, 4)
        grid = np.zeros((nx, ny, nz), dtype=np.float32)

        for s in self.sausages.values():
            if s.voxels is None or not s.voxels.any():
                continue
            idx = np.argwhere(s.voxels)
            if len(idx) == 0:
                continue
            local_pts = s.grid_origin + idx * s.voxel_size      # (M, 3)
            head_w = poses[frame_idx, s.parent_idx]
            tail_w = poses[frame_idx, s.joint_idx]
            l2w, _ = BoneSausage._build_bone_matrix(head_w, tail_w)
            M = len(local_pts)
            pts_h = np.hstack([local_pts, np.ones((M, 1))])
            world_pts = (l2w @ pts_h.T).T[:, :3]

            gi = np.round((world_pts - world_lo) / voxel_size).astype(int)
            valid = (np.all(gi >= 0, axis=1) &
                     (gi[:, 0] < nx) & (gi[:, 1] < ny) & (gi[:, 2] < nz))
            gi = gi[valid]
            if len(gi) > 0:
                grid[gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0

        if not grid.any():
            return None, None

        # Dilate by one voxel to close gaps between bones
        from scipy.ndimage import binary_dilation
        grid = binary_dilation(grid).astype(np.float32)

        try:
            vi, faces, _, _ = marching_cubes(grid, level=0.5)
        except ValueError:
            return None, None

        verts_world = (world_lo + vi * voxel_size).astype(np.float32)
        return verts_world, faces.astype(np.int32)

    # ------------------------------------------------------------------

    def to_combined_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Concatenate all sausage meshes in world space (Frame-0 transform).
        Returns (verts_world float32 (V,3), faces int32 (F,3),
                 bone_weights int32 (V,) — joint_idx per vertex).
        """
        poses = self.skeleton_builder.poses
        _empty = (np.zeros((0, 3), np.float32),
                  np.zeros((0, 3), np.int32),
                  np.zeros(0, np.int32))
        if poses is None:
            return _empty

        all_v, all_f, all_bw = [], [], []
        offset = 0
        for jidx, s in self.sausages.items():
            if s.verts_local is None or len(s.verts_local) == 0:
                continue
            head_w = poses[0, s.parent_idx]
            tail_w = poses[0, s.joint_idx]
            l2w, _ = BoneSausage._build_bone_matrix(head_w, tail_w)
            M = len(s.verts_local)
            lh = np.hstack([s.verts_local, np.ones((M, 1))])
            all_v.append((l2w @ lh.T).T[:, :3].astype(np.float32))
            all_f.append(s.faces + offset)
            all_bw.append(np.full(M, jidx, dtype=np.int32))
            offset += M

        if not all_v:
            return _empty
        return (np.concatenate(all_v),
                np.concatenate(all_f).astype(np.int32),
                np.concatenate(all_bw))

    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save voxels + metadata to compressed .npz."""
        data: Dict[str, np.ndarray] = {}
        sl = list(self.sausages.values())
        data["n_sausages"] = np.array(len(sl), dtype=np.int32)
        for i, s in enumerate(sl):
            p = f"s{i}_"
            data[p + "joint_idx"]  = np.array(s.joint_idx,   np.int32)
            data[p + "parent_idx"] = np.array(s.parent_idx,  np.int32)
            data[p + "radius"]     = np.array(s.radius,       np.float64)
            data[p + "resolution"] = np.array(s.resolution,  np.int32)
            data[p + "bone_len"]   = np.array(s._bone_len_f0, np.float64)
            data[p + "grid_origin"]= s.grid_origin
            data[p + "voxel_size"] = np.array(s.voxel_size,  np.float64)
            if s.voxels is not None:
                data[p + "voxels"] = s.voxels
        if self.skeleton_builder.poses is not None:
            data["skeleton_poses"] = self.skeleton_builder.poses
        np.savez_compressed(path, **data)
        _logger.info("VoxelCarver saved to %s", path)

    @classmethod
    def load(cls, path: str, skeleton_builder, camera_setup) -> "VoxelCarver":
        """Restore from .npz."""
        d   = np.load(path, allow_pickle=False)
        n   = int(d["n_sausages"])
        obj = cls.__new__(cls)
        obj.skeleton_builder = skeleton_builder
        obj.camera_setup     = camera_setup
        obj.sausages         = {}
        obj.resolution       = 32
        obj._bone_radii      = {}
        for i in range(n):
            p    = f"s{i}_"
            jidx = int(d[p + "joint_idx"])
            pidx = int(d[p + "parent_idx"])
            r    = float(d[p + "radius"])
            res  = int(d[p + "resolution"])
            obj.resolution = res
            s = BoneSausage(jidx, pidx, r, res)
            s._bone_len_f0 = float(d[p + "bone_len"])
            s.grid_origin  = d[p + "grid_origin"].copy()
            s.voxel_size   = float(d[p + "voxel_size"])
            key = p + "voxels"
            if key in d:
                s.voxels = d[key].astype(bool)
            obj.sausages[jidx] = s
        _logger.info("VoxelCarver loaded from %s (%d sausages)", path, n)
        return obj

    # ------------------------------------------------------------------

    def to_glb_meshes(self) -> List[Dict]:
        """Compatibility helper (used by MeshFitter.carve_voxel_sausages)."""
        result = []
        for s in self.sausages.values():
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
