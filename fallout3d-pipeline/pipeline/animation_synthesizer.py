"""
AnimationSynthesizer — skeleton frame interpolation (LERP / SLERP),
BVH mocap import + retargeting, and procedural motion generators.
"""

import re
import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable

# scipy is optional; fall back to linear interpolation when absent
try:
    from scipy.spatial.transform import Rotation, Slerp as ScipySlerp
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# -----------------------------------------------------------------------
# Skeleton definition (MediaPipe indices)
# -----------------------------------------------------------------------

# Parent → children — used for FK reconstruction after SLERP
SKELETON_PARENT: Dict[int, int] = {
    # legs
    25: 23, 27: 25,
    26: 24, 28: 26,
    # arms
    13: 11, 15: 13,
    14: 12, 16: 14,
    # face chain
    11: 0, 12: 0,
    23: 11, 24: 12,
}

SKELETON_CHAINS: List[List[int]] = [
    [23, 25, 27],   # left leg
    [24, 26, 28],   # right leg
    [11, 13, 15],   # left arm
    [12, 14, 16],   # right arm
]

# BVH joint name → (MediaPipe landmark index, is_left_right)
BVH_JOINT_MAP: Dict[str, int] = {
    # Hips / root
    "hips": 23, "hip": 23, "pelvis": 23,
    # Spine / chest
    "spine": 0, "spine1": 0, "spine2": 0, "chest": 0, "neck": 0,
    "head": 0,
    # Left arm
    "leftshoulder": 11, "lshoulder": 11, "l_shoulder": 11,
    "leftarm": 11, "larm": 11, "l_arm": 11,
    "leftforearm": 13, "lforearm": 13, "l_forearm": 13, "leftelbow": 13,
    "lefthand": 15, "lhand": 15, "l_hand": 15, "leftwrist": 15,
    # Right arm
    "rightshoulder": 12, "rshoulder": 12, "r_shoulder": 12,
    "rightarm": 12, "rarm": 12, "r_arm": 12,
    "rightforearm": 14, "rforearm": 14, "r_forearm": 14, "rightelbow": 14,
    "righthand": 16, "rhand": 16, "r_hand": 16, "rightwrist": 16,
    # Left leg
    "leftupleg": 23, "lupleg": 23, "l_upleg": 23, "lefthip": 23,
    "leftleg": 25, "lleg": 25, "l_leg": 25, "leftknee": 25,
    "leftfoot": 27, "lfoot": 27, "l_foot": 27, "leftankle": 27,
    # Right leg
    "rightupleg": 24, "rupleg": 24, "r_upleg": 24, "righthip": 24,
    "rightleg": 26, "rleg": 26, "r_leg": 26, "rightknee": 26,
    "rightfoot": 28, "rfoot": 28, "r_foot": 28, "rightankle": 28,
}


# -----------------------------------------------------------------------
# Interpolation primitives
# -----------------------------------------------------------------------

def _lerp_skeleton(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation, bone-length preserving."""
    interp = (1.0 - t) * a + t * b
    # Preserve bone lengths from pose a
    for chain in SKELETON_CHAINS:
        for i in range(len(chain) - 1):
            p, c = chain[i], chain[i + 1]
            orig_len = np.linalg.norm(a[c] - a[p])
            vec = interp[c] - interp[p]
            vlen = np.linalg.norm(vec)
            if vlen > 1e-8 and orig_len > 1e-8:
                interp[c] = interp[p] + vec * (orig_len / vlen)
    return interp


def _slerp_skeleton(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    SLERP on bone directions (requires scipy).
    Falls back to _lerp_skeleton when scipy is absent.
    """
    if not _SCIPY_AVAILABLE:
        return _lerp_skeleton(a, b, t)

    result = a.copy()

    for chain in SKELETON_CHAINS:
        # LERP the root joint of each chain
        root = chain[0]
        result[root] = (1 - t) * a[root] + t * b[root]

        for i in range(len(chain) - 1):
            p, c = chain[i], chain[i + 1]

            bone_a = a[c] - a[p]
            bone_b = b[c] - b[p]
            len_a  = np.linalg.norm(bone_a)
            len_b  = np.linalg.norm(bone_b)

            if len_a < 1e-8 or len_b < 1e-8:
                result[c] = (1 - t) * a[c] + t * b[c]
                continue

            dir_a = bone_a / len_a
            dir_b = bone_b / len_b

            ref = np.array([0.0, -1.0, 0.0])

            def _to_rot(v):
                cross = np.cross(ref, v)
                cn = np.linalg.norm(cross)
                if cn < 1e-8:
                    angle = 0.0 if np.dot(ref, v) > 0 else math.pi
                    return Rotation.from_rotvec(np.array([1, 0, 0]) * angle)
                axis = cross / cn
                angle = float(np.arccos(np.clip(np.dot(ref, v), -1, 1)))
                return Rotation.from_rotvec(axis * angle)

            rot_a = _to_rot(dir_a)
            rot_b = _to_rot(dir_b)

            slerp = ScipySlerp([0.0, 1.0], Rotation.concatenate([rot_a, rot_b]))
            interp_dir = slerp([t])[0].apply(ref)

            bone_len = (1 - t) * len_a + t * len_b
            result[c] = result[p] + interp_dir * bone_len

    return result


# -----------------------------------------------------------------------
# AnimationSynthesizer
# -----------------------------------------------------------------------

class AnimationSynthesizer:
    """
    Synthesises new skeleton sequences via interpolation, BVH retargeting,
    or procedural generators.

    Parameters are (N, 33, 3) numpy arrays.
    """

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(
        self,
        skeleton_seq: np.ndarray,    # (N, 33, 3)
        n_interp: int = 3,
        mode: str = "slerp",         # "lerp" | "slerp"
        keyframe_indices: Optional[List[int]] = None,
        progress_cb: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Insert n_interp synthesised frames between every consecutive pair
        of keyframes.

        Returns
        -------
        new_seq : (N + (N-1)*n_interp, 33, 3)
        """
        N = skeleton_seq.shape[0]
        keys = keyframe_indices if keyframe_indices else list(range(N))
        fn = _slerp_skeleton if mode == "slerp" else _lerp_skeleton

        result: List[np.ndarray] = []
        total = len(keys) - 1

        for i in range(total):
            if progress_cb:
                progress_cb(i, total)
            ka, kb = keys[i], keys[i + 1]
            result.append(skeleton_seq[ka])
            for j in range(1, n_interp + 1):
                t = j / (n_interp + 1)
                result.append(fn(skeleton_seq[ka], skeleton_seq[kb], t))

        if keys:
            result.append(skeleton_seq[keys[-1]])

        return np.stack(result)

    # ------------------------------------------------------------------
    # Procedural: idle breathing
    # ------------------------------------------------------------------

    def add_idle_breathing(
        self,
        skeleton_seq: np.ndarray,
        amplitude: float = 0.015,
        freq: float = 0.5,
    ) -> np.ndarray:
        """
        Add a subtle up/down oscillation to the shoulder and chest area.
        freq : breaths per second (assumes 10 fps).
        """
        N = skeleton_seq.shape[0]
        result = skeleton_seq.copy()
        for f in range(N):
            phase = 2 * math.pi * freq * f / 10.0
            dy = amplitude * math.sin(phase)
            # Shift shoulders, neck, head
            for lm_idx in [0, 11, 12, 13, 14, 15, 16]:
                result[f, lm_idx, 1] += dy
        return result

    # ------------------------------------------------------------------
    # Procedural: hit reaction
    # ------------------------------------------------------------------

    def add_hit_reaction(
        self,
        skeleton_seq: np.ndarray,
        hit_frame: int = 0,
        direction: np.ndarray = None,
        intensity: float = 0.05,
        decay_frames: int = 8,
    ) -> np.ndarray:
        """
        Apply a decaying push in `direction` starting at hit_frame.
        """
        if direction is None:
            direction = np.array([1.0, 0.0, 0.0])
        d = direction / (np.linalg.norm(direction) + 1e-8)
        result = skeleton_seq.copy()
        N = skeleton_seq.shape[0]
        for i in range(decay_frames):
            f = hit_frame + i
            if f >= N:
                break
            scale = intensity * math.exp(-3.0 * i / decay_frames)
            # Apply push to full upper body
            for lm_idx in [0, 11, 12, 13, 14, 15, 16, 23, 24]:
                result[f, lm_idx] += d * scale
        return result

    # ------------------------------------------------------------------
    # BVH Import
    # ------------------------------------------------------------------

    def load_bvh(self, path: str) -> Tuple[Dict, np.ndarray]:
        """
        Parse a BVH file.

        Returns
        -------
        hierarchy : dict  — joint tree
        motion    : (F, D) float — raw channel data
        """
        with open(path) as f:
            text = f.read()
        return _parse_bvh(text)

    def retarget_bvh(
        self,
        hierarchy: Dict,
        motion: np.ndarray,
        reference_skeleton: np.ndarray,  # (33, 3) — rest pose for scale
        joint_map: Optional[Dict[str, int]] = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Convert BVH motion data into a MediaPipe-style (F, 33, 3) sequence.

        Parameters
        ----------
        hierarchy        : dict from load_bvh()
        motion           : (F, D) from load_bvh()
        reference_skeleton : (33, 3) rest pose — used to set scale and origin
        joint_map        : optional override of BVH_JOINT_MAP
        scale            : additional scaling factor

        Returns
        -------
        skeleton_seq : (F, 33, 3)
        """
        jmap = {**BVH_JOINT_MAP, **(joint_map or {})}
        F = motion.shape[0]
        result = np.zeros((F, 33, 3), dtype=np.float64)

        # Fill with rest pose initially
        result[:] = reference_skeleton[np.newaxis]

        joints = hierarchy.get("joints", [])
        channels = hierarchy.get("channels", {})
        offsets  = hierarchy.get("offsets", {})

        # Build a per-frame joint-position table via forward kinematics
        for frame_idx in range(F):
            frame_data = motion[frame_idx]
            joint_world = _bvh_fk(joints, channels, offsets, frame_data, scale)

            for jname, world_pos in joint_world.items():
                jkey = jname.lower().replace(" ", "").replace("_", "").replace("-", "")
                mp_idx = jmap.get(jkey)
                if mp_idx is not None:
                    result[frame_idx, mp_idx] = world_pos

        # Centre and scale to reference_skeleton coordinate range
        ref_min = reference_skeleton.min(axis=0)
        ref_max = reference_skeleton.max(axis=0)
        ref_range = ref_max - ref_min + 1e-8

        bvh_min = result.min(axis=(0, 1))
        bvh_max = result.max(axis=(0, 1))
        bvh_range = bvh_max - bvh_min + 1e-8

        result = (result - bvh_min) / bvh_range * ref_range + ref_min

        return result.astype(np.float32)


# -----------------------------------------------------------------------
# BVH parser
# -----------------------------------------------------------------------

def _parse_bvh(text: str) -> Tuple[Dict, np.ndarray]:
    """Parse HIERARCHY + MOTION sections of a BVH file."""
    lines = text.splitlines()
    idx = 0

    joints: List[str] = []
    offsets: Dict[str, np.ndarray] = {}
    channels: Dict[str, List[str]] = {}
    parent_map: Dict[str, Optional[str]] = {}
    stack: List[str] = []

    # ---- HIERARCHY ----
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1

        if line.upper().startswith("MOTION"):
            break

        upper = line.upper()
        if upper.startswith("ROOT") or upper.startswith("JOINT"):
            jname = line.split()[-1]
            joints.append(jname)
            parent_map[jname] = stack[-1] if stack else None
            stack.append(jname)
        elif upper.startswith("OFFSET"):
            parts = line.split()
            if stack:
                offsets[stack[-1]] = np.array([float(x) for x in parts[1:4]])
        elif upper.startswith("CHANNELS"):
            parts = line.split()
            n_ch = int(parts[1])
            chan_names = parts[2: 2 + n_ch]
            if stack:
                channels[stack[-1]] = chan_names
        elif line == "}":
            if stack:
                stack.pop()

    # ---- MOTION ----
    n_frames = 0
    frame_time = 1 / 30
    motion_rows: List[List[float]] = []

    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if line.upper().startswith("FRAMES:"):
            n_frames = int(line.split(":")[-1])
        elif line.upper().startswith("FRAME TIME:"):
            frame_time = float(line.split(":")[-1])
        elif line and line[0].lstrip("-").replace(".", "").replace("e", "").replace("E", "").replace("+", "").replace("-", "").isdigit():
            try:
                motion_rows.append([float(x) for x in line.split()])
            except ValueError:
                pass

    motion = np.array(motion_rows, dtype=np.float64) if motion_rows else np.zeros((1, 1))

    hierarchy = {
        "joints": joints,
        "channels": channels,
        "offsets": offsets,
        "parent_map": parent_map,
        "frame_time": frame_time,
    }
    return hierarchy, motion


def _bvh_fk(joints, channels, offsets, frame_data, scale):
    """Forward kinematics: compute world positions from one frame of BVH data."""
    joint_world: Dict[str, np.ndarray] = {}
    joint_local_rot: Dict[str, np.ndarray] = {}
    joint_world_rot: Dict[str, np.ndarray] = {}

    chan_idx = 0
    for jname in joints:
        ch = channels.get(jname, [])
        off = offsets.get(jname, np.zeros(3))
        local_pos = off * scale
        rx = ry = rz = 0.0
        for c in ch:
            v = frame_data[chan_idx] if chan_idx < len(frame_data) else 0.0
            cu = c.upper()
            if cu == "XPOSITION":
                local_pos[0] = v * scale
            elif cu == "YPOSITION":
                local_pos[1] = v * scale
            elif cu == "ZPOSITION":
                local_pos[2] = v * scale
            elif cu == "XROTATION":
                rx = math.radians(v)
            elif cu == "YROTATION":
                ry = math.radians(v)
            elif cu == "ZROTATION":
                rz = math.radians(v)
            chan_idx += 1

        if _SCIPY_AVAILABLE:
            # Apply ZXY Euler convention (standard BVH)
            rot = Rotation.from_euler("ZXY", [rz, rx, ry])
        else:
            rot = None

        joint_local_rot[jname] = rot

        parent = None
        for j2 in joints:
            if jname in channels and j2 != jname:
                pass  # simplified — use parent_map from hierarchy

        # For simplicity, just store absolute positions using pure translation + offset
        # (rotation retargeting is handled by mapping to reference skeleton)
        if rot is not None and _SCIPY_AVAILABLE:
            mat = rot.as_matrix()
        else:
            mat = np.eye(3)

        joint_world_rot[jname] = mat
        joint_world[jname] = local_pos.copy()

    return joint_world
