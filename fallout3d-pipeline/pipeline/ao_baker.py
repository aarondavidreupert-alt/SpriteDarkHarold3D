"""
AmbientOcclusionBaker — per-vertex ambient occlusion via Möller–Trumbore
ray casting.  Pure NumPy; no OpenGL, trimesh, or open3d.
"""

import numpy as np


def _moller_trumbore_batch(
    ray_origins: np.ndarray,    # (R, 3)
    ray_dirs: np.ndarray,       # (R, 3)
    v0: np.ndarray,             # (F, 3)
    v1: np.ndarray,             # (F, 3)
    v2: np.ndarray,             # (F, 3)
    max_dist: float,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Vectorised Möller–Trumbore intersection.

    Returns
    -------
    hits : (R,) bool — True if at least one triangle intersected within max_dist
    """
    R = ray_origins.shape[0]
    F = v0.shape[0]
    if R == 0 or F == 0:
        return np.zeros(R, dtype=bool)

    edge1 = v1 - v0   # (F, 3)
    edge2 = v2 - v0   # (F, 3)

    # Cross product of ray dirs (R,3) with edge2 (F,3) → (R, F, 3)
    h = np.cross(ray_dirs[:, None, :], edge2[None, :, :])
    a = np.einsum("fj,rfj->rf", edge1, h)         # (R, F)

    parallel = np.abs(a) < eps
    inv_a = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, a))

    s = ray_origins[:, None, :] - v0[None, :, :]  # (R, F, 3)
    u = inv_a * np.einsum("rfj,rfj->rf", s, h)

    q = np.cross(s, edge1[None, :, :])            # (R, F, 3)
    v = inv_a * np.einsum("rj,rfj->rf", ray_dirs, q)

    t = inv_a * np.einsum("fj,rfj->rf", edge2, q)

    valid = (
        ~parallel
        & (u >= 0.0) & (u <= 1.0)
        & (v >= 0.0) & ((u + v) <= 1.0)
        & (t > eps) & (t < max_dist)
    )
    return valid.any(axis=1)


def _cosine_hemisphere_samples(n: int, normal: np.ndarray) -> np.ndarray:
    """Cosine-weighted hemisphere sampling around `normal` (unit vector)."""
    # Sample on the unit disk, then lift to the hemisphere
    u1 = np.random.random(n)
    u2 = np.random.random(n)
    r = np.sqrt(u1)
    theta = 2.0 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(np.maximum(0.0, 1.0 - u1))
    local = np.stack([x, y, z], axis=1)   # (n, 3) in local frame, +Z = normal

    # Build orthonormal frame around `normal`
    n_unit = normal / (np.linalg.norm(normal) + 1e-12)
    if abs(n_unit[2]) < 0.999:
        tangent = np.cross(n_unit, np.array([0.0, 0.0, 1.0]))
    else:
        tangent = np.cross(n_unit, np.array([1.0, 0.0, 0.0]))
    tangent /= np.linalg.norm(tangent) + 1e-12
    bitangent = np.cross(n_unit, tangent)

    # Transform local samples to world space
    return (
        local[:, 0:1] * tangent
        + local[:, 1:2] * bitangent
        + local[:, 2:3] * n_unit
    )


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals."""
    V = verts.shape[0]
    if faces.size == 0:
        return np.tile(np.array([0.0, 0.0, 1.0]), (V, 1))

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)   # (F, 3) area-weighted

    vn = np.zeros_like(verts)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)

    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.where(norms > 1e-12, vn / norms, np.array([0.0, 0.0, 1.0]))


class AmbientOcclusionBaker:
    """
    Per-vertex ambient occlusion via cosine-weighted hemisphere ray casting.

    Usage
    -----
    baker = AmbientOcclusionBaker(num_rays=64, max_dist=0.5)
    ao = baker.bake_vertex_ao(verts, faces)   # (V,) in [0, 1]
    """

    def __init__(self, num_rays: int = 64, max_dist: float = 0.5):
        self.num_rays = int(num_rays)
        self.max_dist = float(max_dist)

    # ------------------------------------------------------------------

    def bake_vertex_ao(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        progress_cb=None,
    ) -> np.ndarray:
        """
        Cast `num_rays` cosine-weighted hemisphere rays from each vertex.
        Returns per-vertex AO factor in [0, 1] (1 = fully lit).
        """
        verts = np.asarray(verts, dtype=float)
        faces = np.asarray(faces, dtype=int)
        V = verts.shape[0]
        if V == 0 or faces.shape[0] == 0:
            return np.ones(V, dtype=float)

        normals = _vertex_normals(verts, faces)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        ao = np.ones(V, dtype=float)
        eps_offset = max(self.max_dist * 1e-3, 1e-5)

        for i in range(V):
            origin = verts[i] + normals[i] * eps_offset
            dirs = _cosine_hemisphere_samples(self.num_rays, normals[i])
            origins = np.tile(origin, (self.num_rays, 1))
            hits = _moller_trumbore_batch(
                origins, dirs, v0, v1, v2, self.max_dist
            )
            ao[i] = 1.0 - hits.mean()

            if progress_cb and (i % max(1, V // 100) == 0):
                progress_cb(i, V)

        if progress_cb:
            progress_cb(V, V)
        return ao

    # ------------------------------------------------------------------

    def bake_face_ao(self, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Average vertex AO across each triangle's three corners → shape (F,)."""
        v_ao = self.bake_vertex_ao(verts, faces)
        if faces.size == 0:
            return np.array([], dtype=float)
        return v_ao[faces].mean(axis=1)

    # ------------------------------------------------------------------

    def ao_to_vertex_colors(self, ao: np.ndarray) -> np.ndarray:
        """Map AO [0,1] to RGBA uint8 (white = lit, dark = occluded)."""
        ao = np.clip(np.asarray(ao, dtype=float), 0.0, 1.0)
        v = (ao * 255.0).astype(np.uint8)
        rgba = np.empty((ao.shape[0], 4), dtype=np.uint8)
        rgba[:, 0] = v
        rgba[:, 1] = v
        rgba[:, 2] = v
        rgba[:, 3] = 255
        return rgba
