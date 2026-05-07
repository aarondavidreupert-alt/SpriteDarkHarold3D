"""
Tab 9 — Mesh Export

Marching cubes + corrective shape keys + skinning + GLB export.
Source can be either the master sausage (from Tab 8) or the current
character's VoxelCarver directly.
"""

import os
import logging
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFileDialog, QRadioButton, QButtonGroup, QSpinBox,
    QDoubleSpinBox,
)
from PyQt6.QtCore import QThread, QObject, pyqtSignal

from gui.main_window import AppState
from gui.mesh_tab import MeshViewer3D
from pipeline.voxel_carver import VoxelCarver, BoneSausage

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class _CorrectiveWorker(QObject):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, carver: VoxelCarver):
        super().__init__()
        self._carver = carver

    def run(self):
        try:
            self.finished.emit(self._carver.bake_corrective_frames())
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Helper: build a synthetic VoxelCarver from a master_sausage dict
# ---------------------------------------------------------------------------

def _carver_from_master(master: dict, source_carver: VoxelCarver) -> VoxelCarver:
    """
    Build a fresh VoxelCarver where each BoneSausage's voxel grid is replaced
    by the master grid for that joint.  Geometry (grid_origin, voxel_size,
    bone_len) is inherited from `source_carver` so the master grid is placed
    correctly in bone-local space.
    """
    new = VoxelCarver(
        source_carver.skeleton_builder,
        source_carver.camera_setup,
        bone_radii={j: s.radius for j, s in source_carver.sausages.items()},
        resolution=source_carver.resolution,
    )
    for jidx, master_grid in master.items():
        if jidx not in new.sausages:
            continue
        s = new.sausages[jidx]
        # Resize the source-carver's grid_origin/voxel_size for the new shape
        src_s = source_carver.sausages[jidx]
        N_master = master_grid.shape[0]
        N_src    = src_s.voxels.shape[0] if src_s.voxels is not None else N_master
        scale = N_src / max(N_master, 1)
        s.voxels = master_grid.astype(bool)
        s.grid_origin = src_s.grid_origin.copy()
        s.voxel_size  = float(src_s.voxel_size * scale)
        s._bone_len_f0 = src_s._bone_len_f0
    return new


# ---------------------------------------------------------------------------
# MeshExportTab
# ---------------------------------------------------------------------------

class MeshExportTab(QWidget):

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self._carver: VoxelCarver | None = None
        self._verts_0: np.ndarray | None = None
        self._faces:   np.ndarray | None = None
        self._bone_ids: np.ndarray | None = None
        self._shape_keys: np.ndarray | None = None
        self._mesh_frames: list | None = None

        self._corr_thread: QThread | None = None
        self._corr_worker: _CorrectiveWorker | None = None

        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QWidget()
        v = QVBoxLayout(left)
        v.setContentsMargins(4, 4, 4, 4)

        # ── Source ────────────────────────────────────────────────────
        grp_src = QGroupBox("Source")
        h_src = QHBoxLayout(grp_src)
        self.rb_master = QRadioButton("Master Sausage")
        self.rb_current = QRadioButton("Current Character")
        self.rb_master.setChecked(True)
        self._src_group = QButtonGroup(self)
        self._src_group.addButton(self.rb_master)
        self._src_group.addButton(self.rb_current)
        h_src.addWidget(self.rb_master)
        h_src.addWidget(self.rb_current)
        h_src.addStretch()
        v.addWidget(grp_src)

        # ── Step 1: marching cubes ────────────────────────────────────
        grp1 = QGroupBox("Step 1 — Marching Cubes")
        v1 = QVBoxLayout(grp1)
        h_res = QHBoxLayout()
        h_res.addWidget(QLabel("Resolution:"))
        self.cb_res = QComboBox()
        for r in (32, 64, 96):
            self.cb_res.addItem(str(r), r)
        self.cb_res.setCurrentIndex(1)
        h_res.addWidget(self.cb_res, 1)
        v1.addLayout(h_res)
        self.btn_bake0 = QPushButton("Bake Frame 0 Mesh")
        self.btn_bake0.setStyleSheet("font-weight: bold;")
        self.btn_bake0.clicked.connect(self._bake_frame0)
        v1.addWidget(self.btn_bake0)
        self.lbl_mc = QLabel("—")
        v1.addWidget(self.lbl_mc)
        v.addWidget(grp1)

        # ── Step 2: corrective shape keys ─────────────────────────────
        grp2 = QGroupBox("Step 2 — Corrective Shape Keys")
        v2 = QVBoxLayout(grp2)
        h_k = QHBoxLayout()
        h_k.addWidget(QLabel("k-Neighbours:"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 8)
        self.spin_k.setValue(1)
        h_k.addWidget(self.spin_k)
        h_k.addWidget(QLabel("Max Dist Factor:"))
        self.spin_md = QDoubleSpinBox()
        self.spin_md.setRange(0.5, 8.0)
        self.spin_md.setValue(2.0)
        self.spin_md.setSingleStep(0.5)
        h_k.addWidget(self.spin_md)
        v2.addLayout(h_k)
        self.btn_corr = QPushButton("Bake Corrective Frames")
        self.btn_corr.clicked.connect(self._bake_corrective)
        v2.addWidget(self.btn_corr)
        self.lbl_corr = QLabel("—")
        v2.addWidget(self.lbl_corr)
        v.addWidget(grp2)

        # ── Step 3: export ────────────────────────────────────────────
        grp3 = QGroupBox("Step 3 — Export")
        v3 = QVBoxLayout(grp3)
        h_fps = QHBoxLayout()
        h_fps.addWidget(QLabel("FPS:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(10)
        h_fps.addWidget(self.spin_fps)
        h_fps.addStretch()
        v3.addLayout(h_fps)
        self.btn_save_frames = QPushButton("Save All Frames (N × .glb)")
        self.btn_save_frames.clicked.connect(self._save_all_frames)
        v3.addWidget(self.btn_save_frames)
        self.btn_save_corr = QPushButton("Save Corrective GLB")
        self.btn_save_corr.clicked.connect(self._save_corrective_glb)
        v3.addWidget(self.btn_save_corr)
        self.btn_save_skinned = QPushButton("Save Skinned GLB")
        self.btn_save_skinned.clicked.connect(self._save_skinned_glb)
        v3.addWidget(self.btn_save_skinned)
        v.addWidget(grp3)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#88f; font-style: italic;")
        self.status_lbl.setWordWrap(True)
        v.addWidget(self.status_lbl)
        v.addStretch()

        left.setMaximumWidth(420)
        root.addWidget(left)

        self._viewer = MeshViewer3D()
        root.addWidget(self._viewer, 1)

    # ------------------------------------------------------------------
    # Source resolution
    # ------------------------------------------------------------------

    def _resolve_carver(self) -> VoxelCarver | None:
        char = self.state.current_character
        if char is None:
            self._set_status("No character loaded.")
            return None
        src_carver = getattr(char, "voxel_carver", None)
        if src_carver is None:
            self._set_status("Current character has no VoxelCarver — "
                             "carve it in Tab 7c first.")
            return None

        if self.rb_master.isChecked():
            master = getattr(self.state, "master_sausage", None)
            if not master:
                self._set_status("No master sausage — build it in Tab 8 first.")
                return None
            return _carver_from_master(master, src_carver)
        return src_carver

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def _bake_frame0(self):
        carver = self._resolve_carver()
        if carver is None:
            return
        try:
            carver.bake_all()
        except Exception as exc:
            self._set_status(f"Bake error: {exc}")
            return
        verts, faces, bone_ids = carver.to_combined_mesh()
        if len(verts) == 0:
            self._set_status("Mesh is empty — no voxels survived.")
            return
        self._carver   = carver
        self._verts_0  = verts
        self._faces    = faces
        self._bone_ids = bone_ids
        self._shape_keys = None
        try:
            self._viewer.set_mesh(verts, faces, None, 0)
        except Exception:
            pass
        self.lbl_mc.setText(f"{len(verts)} verts, {len(faces)} faces")
        self._set_status(f"Frame-0 mesh baked — {len(verts)} verts.")

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    def _bake_corrective(self):
        if self._carver is None or self._verts_0 is None:
            self._set_status("Bake Frame-0 mesh first (Step 1).")
            return
        self.btn_corr.setEnabled(False)
        self._set_status("Computing corrective shape keys (background)…")
        self._corr_worker = _CorrectiveWorker(self._carver)
        self._corr_thread = QThread(self)
        self._corr_worker.moveToThread(self._corr_thread)
        self._corr_thread.started.connect(self._corr_worker.run)
        self._corr_worker.finished.connect(self._on_corrective_done)
        self._corr_worker.error.connect(self._on_corrective_error)
        self._corr_thread.start()

    def _on_corrective_done(self, result):
        if self._corr_thread is not None:
            self._corr_thread.quit()
        self.btn_corr.setEnabled(True)
        verts_0, faces, bone_ids, shape_keys = result
        self._verts_0  = verts_0
        self._faces    = faces
        self._bone_ids = bone_ids
        self._shape_keys = shape_keys
        n = int(shape_keys.shape[0])
        self.lbl_corr.setText(f"{n} frames baked, {shape_keys.shape[1]} verts")
        self._set_status(f"Corrective shape keys ready ({n} frames).")

    def _on_corrective_error(self, msg: str):
        if self._corr_thread is not None:
            self._corr_thread.quit()
        self.btn_corr.setEnabled(True)
        self._set_status(f"Corrective error: {msg}")

    # ------------------------------------------------------------------
    # Step 3 — save
    # ------------------------------------------------------------------

    def _pick_save_dir(self) -> str | None:
        return QFileDialog.getExistingDirectory(self, "Choose output directory") or None

    def _char_name(self) -> str:
        char = self.state.current_character
        return getattr(char, "name", "character") if char else "character"

    def _build_skin_w(self, n_verts: int) -> np.ndarray:
        n_joints = 33
        skin_w = np.zeros((n_verts, n_joints), dtype=np.float32)
        if self._bone_ids is None:
            return skin_w
        for vi, jidx in enumerate(self._bone_ids):
            col = int(jidx) if int(jidx) < n_joints else 0
            skin_w[vi, col] = 1.0
        return skin_w

    def _save_all_frames(self):
        if self._carver is None:
            self._set_status("Bake Frame-0 mesh first (Step 1).")
            return
        try:
            frames = self._carver.bake_world_grid(
                resolution=int(self.cb_res.currentData()))
        except Exception as exc:
            self._set_status(f"World-grid bake error: {exc}")
            return
        if not frames:
            self._set_status("No frames produced.")
            return
        save_dir = self._pick_save_dir()
        if not save_dir:
            return
        from pipeline.gltf_exporter import GLTFExporter
        exp = GLTFExporter()
        name = self._char_name()
        saved = 0
        for f, (verts, faces) in enumerate(frames):
            if verts is None or len(verts) == 0:
                continue
            path = os.path.join(save_dir, f"{name}_frame_{f:03d}.glb")
            try:
                exp.export_glb(path, verts.astype(np.float32),
                               faces.astype(np.int32),
                               skeleton_sequence=None,
                               skinning_weights=None)
                saved += 1
            except Exception as exc:
                _logger.warning("frame %d save failed: %s", f, exc)
        self._set_status(f"Saved {saved} GLB files to {save_dir}.")

    def _save_corrective_glb(self):
        if self._verts_0 is None or self._shape_keys is None:
            self._set_status("Bake corrective frames first (Step 2).")
            return
        sb = self._carver.skeleton_builder if self._carver else None
        if sb is None or sb.poses is None:
            self._set_status("No skeleton.")
            return
        save_dir = self._pick_save_dir()
        if not save_dir:
            return
        from pipeline.gltf_exporter import GLTFExporter
        skel_seq = sb.poses[:, :33, :].astype(np.float32)
        skin_w = self._build_skin_w(len(self._verts_0))
        path = os.path.join(save_dir, f"{self._char_name()}_corrective.glb")
        try:
            GLTFExporter(fps=float(self.spin_fps.value())).export_glb(
                path, self._verts_0.astype(np.float32),
                self._faces.astype(np.int32),
                skeleton_sequence=skel_seq,
                skinning_weights=skin_w,
                shape_keys=self._shape_keys,
            )
        except Exception as exc:
            self._set_status(f"Corrective GLB save failed: {exc}")
            return
        self._set_status(f"Corrective GLB saved → {path}.")

    def _save_skinned_glb(self):
        if self._verts_0 is None or self._faces is None:
            self._set_status("Bake Frame-0 mesh first (Step 1).")
            return
        sb = self._carver.skeleton_builder if self._carver else None
        if sb is None or sb.poses is None:
            self._set_status("No skeleton.")
            return
        save_dir = self._pick_save_dir()
        if not save_dir:
            return
        from pipeline.gltf_exporter import GLTFExporter
        skel_seq = sb.poses[:, :33, :].astype(np.float32)
        skin_w = self._build_skin_w(len(self._verts_0))
        path = os.path.join(save_dir, f"{self._char_name()}_skinned.glb")
        try:
            GLTFExporter(fps=float(self.spin_fps.value())).export_glb(
                path, self._verts_0.astype(np.float32),
                self._faces.astype(np.int32),
                skeleton_sequence=skel_seq,
                skinning_weights=skin_w,
            )
        except Exception as exc:
            self._set_status(f"Skinned GLB save failed: {exc}")
            return
        self._set_status(f"Skinned GLB saved → {path}.")

    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        self.status_lbl.setText(text)
        _logger.info("MeshExport: %s", text)
