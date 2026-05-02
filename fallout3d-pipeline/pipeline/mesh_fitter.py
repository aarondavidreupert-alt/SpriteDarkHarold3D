"""
MeshFitter — loads an OBJ template mesh, binds it to a 3D skeleton
via automatic skinning weights (heat-diffusion approximation), and
applies linear blend skinning (LBS) to generate posed meshes.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# MediaPipe landmark indices that anchor the rig
_HUMANOID_ANCHORS: Dict[str, int] = {
    "head":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}

_QUADRUPED_ANCHORS: Dict[str, int] = {
    "head":         0,
    "neck":         11,
    "l_shoulder":   11,
    "r_shoulder":   12,
    "l_hip":        23,
    "r_hip":        24,
    "l_front_knee": 13,
    "r_front_knee": 14,
    "l_back_knee":  25,
    "r_back_knee":  26,
}

ANCHORS_BY_CATEGORY = {
    "humanoid":  _HUMANOID_ANCHORS,
    "quadruped": _QUADRUPED_ANCHORS,
    "robot":     _HUMANOID_ANCHORS,
    "insectoid": _HUMANOID_ANCHORS,
    "amorphous": _HUMANOID_ANCHORS,
}

# Per-anchor bone parent names — drives rotation computation in LBS.
# None means root joint (identity rotation, translation only).
_HUMANOID_BONE_PARENTS: Dict[str, Optional[str]] = {
    "head":           "left_shoulder",   # approximate chest
    "left_shoulder":  "left_hip",        # torso (approx spine)
    "right_shoulder": "right_hip",
    "left_elbow":     "left_shoulder",
    "right_elbow":    "right_shoulder",
    "left_wrist":     "left_elbow",
    "right_wrist":    "right_elbow",
    "left_hip":       None,              # root
    "right_hip":      None,
    "left_knee":      "left_hip",
    "right_knee":     "right_hip",
    "left_ankle":     "left_knee",
    "right_ankle":    "right_knee",
}

_QUADRUPED_BONE_PARENTS: Dict[str, Optional[str]] = {
    "head":         "neck",
    "neck":         "l_hip",
    "l_shoulder":   "l_hip",
    "r_shoulder":   "r_hip",
    "l_hip":        None,
    "r_hip":        None,
    "l_front_knee": "l_shoulder",
    "r_front_knee": "r_shoulder",
    "l_back_knee":  "l_hip",
    "r_back_knee":  "r_hip",
}

_BONE_PARENTS_BY_CATEGORY: Dict[str, Dict[str, Optional[str]]] = {
    "humanoid":  _HUMANOID_BONE_PARENTS,
    "quadruped": _QUADRUPED_BONE_PARENTS,
    "robot":     _HUMANOID_BONE_PARENTS,
    "insectoid": _HUMANOID_BONE_PARENTS,
    "amorphous": _HUMANOID_BONE_PARENTS,
}


# ------------------------------------------------------------------
# OBJ I/O
# ------------------------------------------------------------------

def load_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) from a simple OBJ file."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                # Handle "f v", "f v/vt", "f v/vt/vn"
                parts = [p.split("/")[0] for p in line.split()[1:]]
                face = [int(p) - 1 for p in parts]
                if len(face) == 3:
                    faces.append(face)
                elif len(face) == 4:
                    faces.append([face[0], face[1], face[2]])
                    faces.append([face[0], face[2], face[3]])
    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def save_obj(path: str, vertices: np.ndarray, faces: np.ndarray):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# ------------------------------------------------------------------
# Rotation helpers
# ------------------------------------------------------------------

def _rotation_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return a 3×3 rotation matrix that rotates unit vector a onto unit vector b
    via Rodrigues' formula.
    """
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = float(np.dot(a, b))

    if c > 0.9999:
        return np.eye(3)

    if c < -0.9999:
        # 180° rotation: pick an arbitrary perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0,       -axis[2],  axis[1]],
            [axis[2],  0,       -axis[0]],
            [-axis[1], axis[0],  0      ],
        ])
        return np.eye(3) + 2.0 * K @ K

    cross = np.cross(a, b)
    K = np.array([
        [0,        -cross[2],  cross[1]],
        [cross[2],  0,        -cross[0]],
        [-cross[1], cross[0],  0       ],
    ])
    return np.eye(3) + K + K @ K / (1.0 + c)


# ------------------------------------------------------------------
# Skinning weights
# ------------------------------------------------------------------

def compute_skinning_weights(
    vertices: np.ndarray,
    joint_positions: np.ndarray,
    falloff: float = 4.0,
) -> np.ndarray:
    """
    Distance-based skinning weights using inverse-power falloff.

    Positions are normalized to a common unit cube before distance
    computation so that scale mismatches between the template mesh and
    the skeleton do not produce nearly-uniform weights.

    Parameters
    ----------
    vertices       : (V, 3)
    joint_positions: (J, 3)
    falloff        : controls sharpness of the influence falloff

    Returns
    -------
    weights : (V, J)  — each row sums to 1
    """
    V = len(vertices)
    J = len(joint_positions)

    # Normalize both sets of points to a common [0, 1] unit cube so
    # relative proximity is preserved even when the two point clouds
    # live in different coordinate spaces.
    all_pts = np.concatenate([vertices, joint_positions], axis=0)
    bbox_min = all_pts.min(axis=0)
    bbox_max = all_pts.max(axis=0)
    span = np.maximum(bbox_max - bbox_min, 1e-6)

    verts_n  = (vertices        - bbox_min) / span
    joints_n = (joint_positions - bbox_min) / span

    W = np.zeros((V, J))
    for j, jp in enumerate(joints_n):
        dists = np.linalg.norm(verts_n - jp, axis=1)  # (V,)
        W[:, j] = 1.0 / (dists ** falloff + 1e-8)

    row_sums = W.sum(axis=1, keepdims=True)
    return W / (row_sums + 1e-10)


# ------------------------------------------------------------------
# Mesh fitting
# ------------------------------------------------------------------

def fit_mesh_to_skeleton(
    template_verts: np.ndarray,
    template_joints: np.ndarray,
    target_joints: np.ndarray,
    skinning_weights: np.ndarray,
    bone_parents: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Deform template_verts from template_joints to target_joints using LBS.

    For each joint j a rigid transform is computed:
        R_j rotates the rest bone direction to the target bone direction.
        The vertex is transformed as:  R_j @ (v - rest_j) + target_j

    When bone_parents is None (or bone_parents[j] < 0) the joint has no
    parent and only a translation is applied (R_j = I), preserving
    backward compatibility.

    Parameters
    ----------
    template_verts  : (V, 3)
    template_joints : (J, 3)  — rest pose joint positions
    target_joints   : (J, 3)  — target joint positions
    skinning_weights: (V, J)
    bone_parents    : (J,) int  — index of parent joint, -1 for roots

    Returns
    -------
    deformed_verts : (V, 3)
    """
    J = len(template_joints)
    deformed = np.zeros_like(template_verts, dtype=float)

    for j in range(J):
        R = np.eye(3)
        if bone_parents is not None and bone_parents[j] >= 0:
            p = int(bone_parents[j])
            rest_vec = template_joints[j] - template_joints[p]
            tgt_vec  = target_joints[j]  - target_joints[p]
            rest_len = np.linalg.norm(rest_vec)
            tgt_len  = np.linalg.norm(tgt_vec)
            if rest_len > 1e-9 and tgt_len > 1e-9:
                R = _rotation_from_vectors(rest_vec / rest_len, tgt_vec / tgt_len)

        # Rotate each vertex around the rest joint, then translate to target
        v_local   = template_verts - template_joints[j]   # (V, 3)
        v_rotated = (R @ v_local.T).T                      # (V, 3)
        deformed += skinning_weights[:, j:j+1] * (v_rotated + target_joints[j])

    return deformed


# ------------------------------------------------------------------
# MeshFitter class
# ------------------------------------------------------------------

class MeshFitter:
    """
    High-level mesh fitting pipeline.

    Usage
    -----
    fitter = MeshFitter("humanoid")
    fitter.load_template("assets/templates/humanoid.obj")
    fitter.bind_to_skeleton(rest_skeleton_33x3)
    fitter.fit_to_skeleton(skeleton_3d_frame)     # (33, 3)
    fitter.apply_animation(skeleton_3d_sequence)  # (N, 33, 3)
    """

    def __init__(self, category: str = "humanoid"):
        self.category = category
        self.anchors = ANCHORS_BY_CATEGORY.get(category, _HUMANOID_ANCHORS)

        self.template_verts:   Optional[np.ndarray] = None   # (V, 3)
        self.template_faces:   Optional[np.ndarray] = None   # (F, 3)
        self.skinning_weights: Optional[np.ndarray] = None   # (V, J)
        self.rest_joints:      Optional[np.ndarray] = None   # (J, 3)
        self._bone_parents:    Optional[np.ndarray] = None   # (J,) int, -1 = root

    # ------------------------------------------------------------------

    def load_template(self, path: str):
        self.template_verts, self.template_faces = load_obj(path)

    def bind_to_skeleton(self, rest_skeleton: np.ndarray, falloff: float = 4.0):
        """Compute skinning weights and bone parent indices for the rest pose.

        Parameters
        ----------
        rest_skeleton : (33, 3) — first frame or canonical pose
        """
        anchor_names = list(self.anchors.keys())
        lm_indices   = [self.anchors[n] for n in anchor_names]
        self.rest_joints = rest_skeleton[lm_indices]   # (J, 3)

        # Build per-joint parent index array for rotation-based LBS
        bone_parents_map = _BONE_PARENTS_BY_CATEGORY.get(
            self.category, _HUMANOID_BONE_PARENTS
        )
        J = len(anchor_names)
        self._bone_parents = np.full(J, -1, dtype=int)
        for j, name in enumerate(anchor_names):
            parent_name = bone_parents_map.get(name)
            if parent_name and parent_name in anchor_names:
                self._bone_parents[j] = anchor_names.index(parent_name)

        if self.template_verts is None:
            raise RuntimeError("Load a template OBJ before binding.")

        self.skinning_weights = compute_skinning_weights(
            self.template_verts, self.rest_joints, falloff
        )

    def fit_to_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """Return deformed vertices for a single skeleton pose (33, 3)."""
        if self.rest_joints is None or self.skinning_weights is None:
            raise RuntimeError("Call bind_to_skeleton() first.")

        lm_indices   = [self.anchors[n] for n in self.anchors]
        target_joints = skeleton[lm_indices]

        return fit_mesh_to_skeleton(
            self.template_verts, self.rest_joints,
            target_joints, self.skinning_weights,
            self._bone_parents,
        )

    def apply_animation(self, skeleton_sequence: np.ndarray) -> List[np.ndarray]:
        """Return list of (V, 3) deformed meshes for each frame."""
        return [self.fit_to_skeleton(sk) for sk in skeleton_sequence]

    def get_bone_weight_heatmap(self, bone_idx: int) -> np.ndarray:
        """Return per-vertex weight for the given bone index, shape (V,)."""
        if self.skinning_weights is None:
            return np.array([])
        return self.skinning_weights[:, bone_idx]
