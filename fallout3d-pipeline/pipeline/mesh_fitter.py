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
    "head":         0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow":   13,
    "right_elbow":  14,
    "left_wrist":   15,
    "right_wrist":  16,
    "left_hip":     23,
    "right_hip":    24,
    "left_knee":    25,
    "right_knee":   26,
    "left_ankle":   27,
    "right_ankle":  28,
}

_QUADRUPED_ANCHORS: Dict[str, int] = {
    "head":       0,
    "neck":       11,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_hip":      23,
    "r_hip":      24,
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
# Skinning weights
# ------------------------------------------------------------------

def compute_skinning_weights(
    vertices: np.ndarray,
    joint_positions: np.ndarray,
    falloff: float = 4.0,
) -> np.ndarray:
    """
    Simple distance-based skinning weights using inverse-square falloff.

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
    W = np.zeros((V, J))

    for j, jp in enumerate(joint_positions):
        dists = np.linalg.norm(vertices - jp, axis=1)  # (V,)
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
) -> np.ndarray:
    """
    Deform template_verts so each joint moves from template_joints[j]
    to target_joints[j] using LBS.

    Parameters
    ----------
    template_verts  : (V, 3)
    template_joints : (J, 3)  — rest pose joint positions
    target_joints   : (J, 3)  — target joint positions
    skinning_weights: (V, J)

    Returns
    -------
    deformed_verts : (V, 3)
    """
    J = len(template_joints)
    deformed = np.zeros_like(template_verts)

    for j in range(J):
        delta = target_joints[j] - template_joints[j]
        deformed += skinning_weights[:, j:j+1] * (template_verts + delta)

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
    fitter.fit_to_skeleton(skeleton_3d_frame)     # (33, 3)
    fitter.apply_animation(skeleton_3d_sequence)  # (N, 33, 3)
    """

    def __init__(self, category: str = "humanoid"):
        self.category = category
        self.anchors = ANCHORS_BY_CATEGORY.get(category, _HUMANOID_ANCHORS)

        self.template_verts: Optional[np.ndarray] = None   # (V, 3)
        self.template_faces: Optional[np.ndarray] = None   # (F, 3)
        self.skinning_weights: Optional[np.ndarray] = None # (V, J)
        self.rest_joints: Optional[np.ndarray] = None      # (J, 3)

    # ------------------------------------------------------------------

    def load_template(self, path: str):
        self.template_verts, self.template_faces = load_obj(path)

    def bind_to_skeleton(self, rest_skeleton: np.ndarray, falloff: float = 4.0):
        """Compute skinning weights for the rest-pose skeleton.

        Parameters
        ----------
        rest_skeleton : (33, 3) — first frame or canonical pose
        """
        anchor_names = list(self.anchors.keys())
        lm_indices = [self.anchors[n] for n in anchor_names]
        self.rest_joints = rest_skeleton[lm_indices]   # (J, 3)

        if self.template_verts is None:
            raise RuntimeError("Load a template OBJ before binding.")

        self.skinning_weights = compute_skinning_weights(
            self.template_verts, self.rest_joints, falloff
        )

    def fit_to_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """Return deformed vertices for a single skeleton pose (33, 3)."""
        if self.rest_joints is None or self.skinning_weights is None:
            raise RuntimeError("Call bind_to_skeleton() first.")

        lm_indices = [self.anchors[n] for n in self.anchors]
        target_joints = skeleton[lm_indices]

        return fit_mesh_to_skeleton(
            self.template_verts, self.rest_joints,
            target_joints, self.skinning_weights,
        )

    def apply_animation(self, skeleton_sequence: np.ndarray) -> List[np.ndarray]:
        """Return list of (V, 3) deformed meshes for each frame."""
        return [self.fit_to_skeleton(sk) for sk in skeleton_sequence]

    def get_bone_weight_heatmap(self, bone_idx: int) -> np.ndarray:
        """Return per-vertex weight for the given bone index, shape (V,)."""
        if self.skinning_weights is None:
            return np.array([])
        return self.skinning_weights[:, bone_idx]
