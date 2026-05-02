"""
SkeletonBuilder — rigid bone lengths, bind pose, constrained 3D poses, and frame interpolation.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# -----------------------------------------------------------------------
# Hierarchy: child → parent (None = root)
# Index 33 is the virtual Hip Root (midpoint of hips 23 & 24)
# -----------------------------------------------------------------------

BONE_HIERARCHY: dict[int, int | None] = {
    33: None,
    23: 33, 24: 33,           # hips
    25: 23, 26: 24,           # knees
    27: 25, 28: 26,           # ankles
    29: 27, 30: 28,           # heels
    31: 29, 32: 30,           # feet
    11: 33, 12: 33,           # shoulders (via virtual spine)
    13: 11, 14: 12,           # elbows
    15: 13, 16: 14,           # wrists
    17: 15, 19: 15, 21: 15,   # left fingers
    18: 16, 20: 16, 22: 16,   # right fingers
    0: 33,                    # nose → spine
    7: 0,  8: 0,              # ears
}

BONE_NAMES: dict[int, str] = {
    33: "Hip Root",
    23: "L-Hip",      24: "R-Hip",
    25: "L-Knee",     26: "R-Knee",
    27: "L-Ankle",    28: "R-Ankle",
    29: "L-Heel",     30: "R-Heel",
    31: "L-Foot",     32: "R-Foot",
    11: "L-Shoulder", 12: "R-Shoulder",
    13: "L-Elbow",    14: "R-Elbow",
    15: "L-Wrist",    16: "R-Wrist",
    17: "L-Pinky",    18: "R-Pinky",
    19: "L-Index",    20: "R-Index",
    21: "L-Thumb",    22: "R-Thumb",
    0:  "Nose",
    7:  "L-Ear",      8: "R-Ear",
}

# BFS-ordered list of joint indices (root first) for top-down traversal
_TRAVERSE_ORDER: list[int] = []
_seen: set[int] = set()
_queue: list[int] = [33]
while _queue:
    _node = _queue.pop(0)
    if _node in _seen:
        continue
    _seen.add(_node)
    _TRAVERSE_ORDER.append(_node)
    for child, parent in BONE_HIERARCHY.items():
        if parent == _node and child not in _seen:
            _queue.append(child)


def _vec_to_rotvec(v: np.ndarray) -> np.ndarray:
    """Return a rotation-vector that rotates [0,0,1] onto the unit vector v."""
    v = v / (np.linalg.norm(v) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    cross = np.cross(ref, v)
    n = np.linalg.norm(cross)
    if n < 1e-9:
        return np.zeros(3) if v[2] > 0 else np.array([np.pi, 0.0, 0.0])
    return cross / n * np.arcsin(np.clip(n, 0.0, 1.0))


class SkeletonBuilder:
    """
    Locks bone lengths from a 3D skeleton sequence and applies rigid constraints.

    Usage
    -----
    sb = SkeletonBuilder()
    sb.build(skeleton_3d, mode="median")   # (N, 33, 3)
    # sb.poses  → (N, 34, 3)  rigidly constrained, index 33 = Hip Root
    """

    def __init__(self):
        self.bone_lengths: dict[int, float] = {}
        self.bind_pose: np.ndarray | None = None    # (34, 3)
        self.poses: np.ndarray | None = None        # (N, 34, 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        skeleton_3d: np.ndarray,
        mode: str = "median",
        manual_lengths: dict[int, float] | None = None,
    ) -> "SkeletonBuilder":
        """
        Parameters
        ----------
        skeleton_3d : (N, 33, 3)
        mode        : "frame0" | "median" | "manual"
        manual_lengths : required when mode=="manual"
        """
        N = skeleton_3d.shape[0]

        # Build (N, 34, 3) by appending the virtual hip root at index 33
        full = self._add_hip_root(skeleton_3d)  # (N, 34, 3)

        # Compute bone lengths
        if mode == "manual" and manual_lengths is not None:
            self.bone_lengths = dict(manual_lengths)
        elif mode == "frame0":
            self.bone_lengths = self._measure_lengths(full[0:1])
        else:  # median
            self.bone_lengths = self._measure_lengths(full)

        # Bind pose
        if mode == "frame0":
            self.bind_pose = full[0].copy()
        else:
            self.bind_pose = np.median(full, axis=0)

        # Apply rigid constraints to every frame
        constrained = np.zeros_like(full)
        for i in range(N):
            constrained[i] = self._apply_rigid_constraints(full[i].copy())
        self.poses = constrained

        return self

    def interpolate(self, frame_a: int, frame_b: int, t: float) -> np.ndarray:
        """
        Interpolate between two frames using slerp on bone rotations.

        Parameters
        ----------
        frame_a, frame_b : frame indices into self.poses
        t                : blend factor in [0, 1]

        Returns
        -------
        (34, 3) interpolated pose
        """
        if self.poses is None:
            raise ValueError("Call build() first.")

        pa = self.poses[frame_a]
        pb = self.poses[frame_b]

        out = np.zeros((34, 3))

        # Root position: linear interpolation
        out[33] = (1.0 - t) * pa[33] + t * pb[33]

        for joint_idx in _TRAVERSE_ORDER:
            if joint_idx == 33:
                continue
            parent_idx = BONE_HIERARCHY[joint_idx]
            if parent_idx is None:
                continue

            p_parent_a = pa[parent_idx]
            p_parent_b = pb[parent_idx]
            p_child_a  = pa[joint_idx]
            p_child_b  = pb[joint_idx]

            dir_a = p_child_a - p_parent_a
            dir_b = p_child_b - p_parent_b
            len_a = np.linalg.norm(dir_a)
            len_b = np.linalg.norm(dir_b)

            if len_a < 1e-9 or len_b < 1e-9:
                out[joint_idx] = (1.0 - t) * p_child_a + t * p_child_b
                continue

            rv_a = _vec_to_rotvec(dir_a / len_a)
            rv_b = _vec_to_rotvec(dir_b / len_b)

            try:
                rots = Rotation.from_rotvec(np.stack([rv_a, rv_b]))
                slerp = Slerp([0.0, 1.0], rots)
                dir_interp = slerp(t).apply(np.array([0.0, 0.0, 1.0]))
            except Exception:
                dir_interp = ((1.0 - t) * dir_a / len_a + t * dir_b / len_b)
                norm = np.linalg.norm(dir_interp)
                dir_interp = dir_interp / (norm + 1e-12)

            bone_len = self.bone_lengths.get(joint_idx, (len_a + len_b) * 0.5)
            parent_pos = out[parent_idx]
            out[joint_idx] = parent_pos + dir_interp * bone_len

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_hip_root(skeleton_3d: np.ndarray) -> np.ndarray:
        """Append virtual hip root (mean of joints 23 & 24) as index 33."""
        N = skeleton_3d.shape[0]
        hip_root = ((skeleton_3d[:, 23, :] + skeleton_3d[:, 24, :]) / 2.0)[:, np.newaxis, :]
        return np.concatenate([skeleton_3d, hip_root], axis=1)  # (N, 34, 3)

    def _measure_lengths(self, full: np.ndarray) -> dict[int, float]:
        """Compute median bone length for each joint across all frames."""
        lengths: dict[int, float] = {}
        for joint_idx, parent_idx in BONE_HIERARCHY.items():
            if parent_idx is None:
                continue
            diffs = full[:, joint_idx, :] - full[:, parent_idx, :]
            norms = np.linalg.norm(diffs, axis=1)
            nonzero = norms[norms > 1e-9]
            lengths[joint_idx] = float(np.median(nonzero)) if len(nonzero) else 0.0
        return lengths

    def _apply_rigid_constraints(self, positions: np.ndarray) -> np.ndarray:
        """
        Walk the hierarchy top-down and reproject each child to exactly
        bone_length distance from its parent.
        """
        for joint_idx in _TRAVERSE_ORDER:
            parent_idx = BONE_HIERARCHY.get(joint_idx)
            if parent_idx is None:
                continue
            bone_len = self.bone_lengths.get(joint_idx, 0.0)
            if bone_len < 1e-9:
                continue

            parent_pos = positions[parent_idx]
            child_pos  = positions[joint_idx]
            direction  = child_pos - parent_pos
            dist = np.linalg.norm(direction)
            if dist < 1e-9:
                # zero-length — pick arbitrary direction
                direction = np.array([0.0, 1.0, 0.0])
                dist = 1.0
            positions[joint_idx] = parent_pos + (direction / dist) * bone_len

        return positions
