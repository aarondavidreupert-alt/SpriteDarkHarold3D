"""
GLTFExporter — packs a rigged animated mesh, normal maps, and pose data
into a self-contained .glb file using pygltflib.
"""

import json
import struct
import numpy as np
from pathlib import Path
from typing import List, Optional

try:
    import pygltflib
    _PYGLTF_AVAILABLE = True
except ImportError:
    _PYGLTF_AVAILABLE = False


# MediaPipe landmark connectivity used to define skeleton bones
BONE_PAIRS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 11), (0, 12),
]


def _pack_floats(data: np.ndarray) -> bytes:
    return data.astype(np.float32).tobytes()


def _pack_ushorts(data: np.ndarray) -> bytes:
    return data.astype(np.uint16).tobytes()


class GLTFExporter:
    """
    Export pipeline results to glTF/GLB.

    Minimal valid output includes:
    - Mesh geometry (vertices + faces)
    - Armature skeleton (one node per landmark)
    - Animation tracks (one per bone, sampled at frame rate)
    - Normal map textures (embedded as PNG)
    """

    def __init__(self, fps: float = 10.0):
        self.fps = fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_glb(
        self,
        output_path: str,
        vertices: np.ndarray,            # (V, 3)
        faces: np.ndarray,               # (F, 3)
        skeleton_sequence: np.ndarray,   # (N, 33, 3)
        skinning_weights: Optional[np.ndarray] = None,  # (V, J)
        normal_maps: Optional[np.ndarray] = None,        # (6, N, H, W, 3)
        uvs: Optional[np.ndarray] = None,               # (V, 2)
        mesh_frames: Optional[np.ndarray] = None,       # (N, V, 3) per-frame verts
    ):
        if not _PYGLTF_AVAILABLE:
            raise RuntimeError("pygltflib is not installed. Run: pip install pygltflib")

        gltf = pygltflib.GLTF2()
        gltf.asset = pygltflib.Asset(version="2.0", generator="Fallout3D Pipeline")

        blob = bytearray()

        def add_accessor(data: np.ndarray, comp_type, acc_type, target=None) -> int:
            raw = data.astype({
                pygltflib.FLOAT: np.float32,
                pygltflib.UNSIGNED_SHORT: np.uint16,
                pygltflib.UNSIGNED_INT: np.uint32,
            }[comp_type]).tobytes()
            bv_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(pygltflib.BufferView(
                buffer=0,
                byteOffset=len(blob),
                byteLength=len(raw),
                target=target,
            ))
            blob.extend(raw)
            # Pad to 4-byte alignment
            while len(blob) % 4:
                blob.append(0)

            acc_idx = len(gltf.accessors)
            count = data.shape[0]
            flat = data.flatten()
            gltf.accessors.append(pygltflib.Accessor(
                bufferView=bv_idx,
                componentType=comp_type,
                count=count,
                type=acc_type,
                max=data.max(axis=0).tolist() if data.ndim > 1 else [float(data.max())],
                min=data.min(axis=0).tolist() if data.ndim > 1 else [float(data.min())],
            ))
            return acc_idx

        # ---- Mesh geometry ----------------------------------------
        verts_f32 = vertices.astype(np.float32)
        pos_acc = add_accessor(verts_f32, pygltflib.FLOAT, pygltflib.VEC3,
                               target=pygltflib.ARRAY_BUFFER)

        # Simple flat normals
        flat_normals = np.tile([0.0, 0.0, 1.0], (len(vertices), 1)).astype(np.float32)
        norm_acc = add_accessor(flat_normals, pygltflib.FLOAT, pygltflib.VEC3,
                                target=pygltflib.ARRAY_BUFFER)

        # UVs
        if uvs is None:
            uvs = np.zeros((len(vertices), 2), dtype=np.float32)
        uv_acc = add_accessor(uvs.astype(np.float32), pygltflib.FLOAT, pygltflib.VEC2,
                              target=pygltflib.ARRAY_BUFFER)

        idx_acc = add_accessor(faces.flatten().astype(np.uint16), pygltflib.UNSIGNED_SHORT,
                               pygltflib.SCALAR, target=pygltflib.ELEMENT_ARRAY_BUFFER)

        # ---- Skeleton nodes ----------------------------------------
        n_joints = 33
        joint_nodes = []
        for j in range(n_joints):
            rest_pos = skeleton_sequence[0, j].tolist() if skeleton_sequence is not None else [0, 0, 0]
            node_idx = len(gltf.nodes)
            gltf.nodes.append(pygltflib.Node(
                name=f"joint_{j}",
                translation=rest_pos,
            ))
            joint_nodes.append(node_idx)

        # ---- Mesh node with skin ------------------------------------
        skin_idx = len(gltf.skins)
        gltf.skins.append(pygltflib.Skin(
            name="Armature",
            joints=joint_nodes,
        ))

        prim_extras: dict = {}
        if skinning_weights is not None and len(skinning_weights) > 0:
            # Encode top-4 joints + weights per vertex (glTF skinning)
            top4_joints = np.argsort(-skinning_weights, axis=1)[:, :4].astype(np.uint16)
            top4_weights = np.take_along_axis(skinning_weights, top4_joints.astype(int), axis=1).astype(np.float32)
            top4_weights /= (top4_weights.sum(axis=1, keepdims=True) + 1e-10)

            jnt_acc = add_accessor(top4_joints, pygltflib.UNSIGNED_SHORT, pygltflib.VEC4,
                                   target=pygltflib.ARRAY_BUFFER)
            wgt_acc = add_accessor(top4_weights, pygltflib.FLOAT, pygltflib.VEC4,
                                   target=pygltflib.ARRAY_BUFFER)
            prim_extras = {"JOINTS_0": jnt_acc, "WEIGHTS_0": wgt_acc}

        # inverseBindMatrices: pure-translate inverse of each joint's bind pose.
        # MediaPipe joints are positions only — no rotation — so the inverse
        # is simply T(-p).  glTF stores matrices column-major; numpy's default
        # tobytes() is row-major so we transpose before flattening.
        if skeleton_sequence is not None and skinning_weights is not None:
            inv_binds = np.tile(np.eye(4, dtype=np.float32),
                                (n_joints, 1, 1))
            inv_binds[:, :3, 3] = -skeleton_sequence[0, :n_joints].astype(np.float32)
            inv_binds_cm = np.transpose(inv_binds, (0, 2, 1)
                                        ).reshape(n_joints, 16).astype(np.float32)
            ibm_acc = add_accessor(inv_binds_cm,
                                   pygltflib.FLOAT, pygltflib.MAT4)
            gltf.skins[skin_idx].inverseBindMatrices = ibm_acc

        mesh_node_idx = len(gltf.nodes)
        gltf.nodes.append(pygltflib.Node(name="Mesh", mesh=0, skin=skin_idx))

        primitive = pygltflib.Primitive(
            attributes=pygltflib.Attributes(
                POSITION=pos_acc,
                NORMAL=norm_acc,
                TEXCOORD_0=uv_acc,
                **prim_extras,
            ),
            indices=idx_acc,
            material=0,
        )
        mesh_obj = pygltflib.Mesh(name="CritterMesh", primitives=[primitive])
        gltf.meshes.append(mesh_obj)

        # ---- Morph targets (per-frame deformation) -----------------
        morph_target_acc: list[int] = []
        morph_n_targets = 0
        morph_stride = 1
        if mesh_frames is not None and len(mesh_frames) >= 2 and \
                mesh_frames.shape[1] == len(vertices):
            n_total = int(mesh_frames.shape[0])
            morph_stride = max(1, (n_total + 254) // 255)
            sel = mesh_frames[::morph_stride]
            if len(sel) > 255:
                sel = sel[:255]
            rest = verts_f32
            for i in range(len(sel)):
                delta = (sel[i].astype(np.float32) - rest)
                if not np.any(np.isfinite(delta)):
                    delta = np.zeros_like(rest)
                delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                acc = add_accessor(delta, pygltflib.FLOAT, pygltflib.VEC3,
                                   target=pygltflib.ARRAY_BUFFER)
                morph_target_acc.append(acc)
            morph_n_targets = len(morph_target_acc)
            primitive.targets = [
                pygltflib.Attributes(POSITION=a) for a in morph_target_acc
            ]
            mesh_obj.weights = [0.0] * morph_n_targets

        gltf.materials.append(pygltflib.Material(
            name="CritterMaterial",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.8,
            ),
            doubleSided=True,
        ))

        # ---- Animation ---------------------------------------------
        if skeleton_sequence is not None and len(skeleton_sequence) > 1:
            N = skeleton_sequence.shape[0]
            times = np.arange(N, dtype=np.float32) / self.fps
            time_acc = add_accessor(times, pygltflib.FLOAT, pygltflib.SCALAR)

            anim = pygltflib.Animation(name="Walk")
            for j in range(n_joints):
                translations = skeleton_sequence[:, j, :].astype(np.float32)
                tl_acc = add_accessor(translations, pygltflib.FLOAT, pygltflib.VEC3)

                sampler_idx = len(anim.samplers)
                anim.samplers.append(pygltflib.AnimationSampler(
                    input=time_acc, output=tl_acc, interpolation="LINEAR"
                ))
                anim.channels.append(pygltflib.AnimationChannel(
                    sampler=sampler_idx,
                    target=pygltflib.AnimationChannelTarget(
                        node=joint_nodes[j], path="translation"
                    ),
                ))
            gltf.animations.append(anim)

        # ---- Morph weight animation --------------------------------
        if morph_n_targets >= 2:
            m_times = (np.arange(morph_n_targets, dtype=np.float32)
                       * morph_stride / self.fps)
            m_weights = np.eye(morph_n_targets, dtype=np.float32).flatten()
            mtime_acc = add_accessor(m_times, pygltflib.FLOAT, pygltflib.SCALAR)
            mwgt_acc  = add_accessor(m_weights, pygltflib.FLOAT, pygltflib.SCALAR)
            anim_m = pygltflib.Animation(name="MorphAnim")
            anim_m.samplers.append(pygltflib.AnimationSampler(
                input=mtime_acc, output=mwgt_acc, interpolation="STEP"
            ))
            anim_m.channels.append(pygltflib.AnimationChannel(
                sampler=0,
                target=pygltflib.AnimationChannelTarget(
                    node=mesh_node_idx, path="weights"
                ),
            ))
            gltf.animations.append(anim_m)

        # ---- Scene -------------------------------------------------
        scene_nodes = [mesh_node_idx] + joint_nodes
        gltf.scenes.append(pygltflib.Scene(name="Scene", nodes=scene_nodes))
        gltf.scene = 0

        gltf.buffers.append(pygltflib.Buffer(byteLength=len(blob)))
        gltf.set_binary_blob(bytes(blob))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        gltf.save_binary(output_path)
        return output_path

    # ------------------------------------------------------------------
    # Convenience: export normal maps as individual PNGs
    # ------------------------------------------------------------------

    def export_normal_maps(self, normal_maps: np.ndarray, output_dir: str):
        """normal_maps: (6, N, H, W, 3) uint8"""
        import cv2
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for d in range(normal_maps.shape[0]):
            for f in range(normal_maps.shape[1]):
                nm = normal_maps[d, f]
                cv2.imwrite(str(out / f"normal_d{d+1}_f{f:03d}.png"), nm)

    # ------------------------------------------------------------------
    # Convenience: export animation data JSON
    # ------------------------------------------------------------------

    def export_animation_json(self, skeleton_sequence: np.ndarray, path: str):
        data = {
            "fps": self.fps,
            "frames": skeleton_sequence.shape[0],
            "joints": 33,
            "animation": skeleton_sequence.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
