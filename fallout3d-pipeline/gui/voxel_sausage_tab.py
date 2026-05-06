"""
Tab 7c — Voxel Sausage Carver

Workflow:
  1. Import Skeleton    — pull from AppState (same pattern as Tab 7b)
  2. Generate Ragdoll   — build per-bone voxel grids initialised as capsules
  3. Adjust Radii       — per-bone QDoubleSpinBox; "Rebuild" re-inits grids
  4. Play Animation     — frame slider + play/pause button
  5. Carve Voxels       — single-frame / all-bool / all-weighted + progress bar
  6. Bake Mesh          — marching cubes → combined world-space mesh
  7. Save               — .npz voxels + .glb mesh + skeleton .json
"""

import os
import logging
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSlider, QGroupBox, QSplitter, QProgressBar,
    QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea, QRadioButton,
    QButtonGroup, QSizePolicy, QFileDialog,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage

from gui.main_window import AppState
from gui.mesh_tab import MeshViewer3D
from pipeline.pose_triangulator import POSE_CONNECTIONS
from pipeline.skeleton_builder import BONE_NAMES, BONE_HIERARCHY
from pipeline.voxel_carver import VoxelCarver, BoneSausage, SKIP_JOINTS

try:
    import pyqtgraph.opengl as gl
    _GL = True
except ImportError:
    _GL = False

_logger = logging.getLogger(__name__)

# Distinct colours for the per-bone scatter plot (RGB 0-1)
_BONE_COLOURS = [
    (1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.5, 1.0),
    (1.0, 1.0, 0.2), (1.0, 0.5, 0.0), (0.8, 0.2, 1.0),
    (0.0, 1.0, 1.0), (1.0, 0.0, 0.5), (0.5, 1.0, 0.0),
    (0.0, 0.5, 1.0), (1.0, 0.8, 0.8), (0.6, 0.9, 0.6),
]


# ---------------------------------------------------------------------------
# CarveWorker
# ---------------------------------------------------------------------------

class CarveWorker(QObject):
    progress = pyqtSignal(int, int)   # (done, total)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, carver: VoxelCarver, all_masks: list, weighted: bool):
        super().__init__()
        self._carver  = carver
        self._masks   = all_masks
        self._weighted = weighted

    def run(self):
        try:
            self._carver.carve_all_frames(
                self._masks,
                weighted=self._weighted,
                progress_cb=lambda done, total: self.progress.emit(done, total),
            )
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# VoxelSausageTab
# ---------------------------------------------------------------------------

class VoxelSausageTab(QWidget):

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self._skeleton_builder = None        # SkeletonBuilder when loaded
        self._carver: VoxelCarver | None = None
        self._current_frame = 0
        self._mesh_verts: np.ndarray | None = None
        self._mesh_faces: np.ndarray | None = None
        self._bone_weights: np.ndarray | None = None
        self._scatter_items: list = []       # GLScatterPlotItem refs

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)
        self._carve_thread: QThread | None = None
        self._carve_worker: CarveWorker | None = None

        self._radius_spins: dict[int, QDoubleSpinBox] = {}

        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.character_updated.connect(self._on_char_updated)
        self._on_char_changed(self.state.selected_idx)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([380, 820])

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        vbox  = QVBoxLayout(panel)
        vbox.setContentsMargins(4, 4, 4, 4)

        # ── Step 1: Import Skeleton ──────────────────────────────────
        grp1 = QGroupBox("Step 1: Import Skeleton")
        v1   = QVBoxLayout(grp1)
        self.skeleton_lbl = QLabel("Skeleton: not loaded")
        self.skeleton_lbl.setWordWrap(True)
        v1.addWidget(self.skeleton_lbl)
        btn_use = QPushButton("← Use skeleton from Tab 5b")
        btn_use.clicked.connect(self._use_skeleton)
        v1.addWidget(btn_use)
        vbox.addWidget(grp1)

        # ── Step 2: Generate Voxel Ragdoll ───────────────────────────
        grp2 = QGroupBox("Step 2: Generate Voxel Ragdoll")
        v2   = QVBoxLayout(grp2)
        row_res = QHBoxLayout()
        row_res.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        for r in (16, 32, 64):
            self.resolution_combo.addItem(str(r), r)
        self.resolution_combo.setCurrentIndex(1)  # default 32
        row_res.addWidget(self.resolution_combo, 1)
        v2.addLayout(row_res)
        btn_gen = QPushButton("Generate Voxel Ragdoll")
        btn_gen.setStyleSheet("font-weight: bold;")
        btn_gen.clicked.connect(self._generate_ragdoll)
        v2.addWidget(btn_gen)
        self.ragdoll_lbl = QLabel("—")
        v2.addWidget(self.ragdoll_lbl)
        vbox.addWidget(grp2)

        # ── Step 3: Adjust Radii ─────────────────────────────────────
        grp3 = QGroupBox("Step 3: Adjust Radii")
        v3   = QVBoxLayout(grp3)

        row_all = QHBoxLayout()
        row_all.addWidget(QLabel("Set all:"))
        self._all_radius_spin = QDoubleSpinBox()
        self._all_radius_spin.setRange(0.001, 0.5)
        self._all_radius_spin.setSingleStep(0.005)
        self._all_radius_spin.setDecimals(3)
        self._all_radius_spin.setValue(0.045)
        row_all.addWidget(self._all_radius_spin)
        btn_set_all = QPushButton("Apply to All")
        btn_set_all.clicked.connect(self._set_all_radii)
        row_all.addWidget(btn_set_all)
        v3.addLayout(row_all)

        self._radius_scroll = QScrollArea()
        self._radius_scroll.setWidgetResizable(True)
        self._radius_scroll.setFixedHeight(160)
        self._radius_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._radius_container = QWidget()
        self._radius_layout    = QVBoxLayout(self._radius_container)
        self._radius_layout.setContentsMargins(2, 2, 2, 2)
        self._radius_layout.setSpacing(2)
        self._radius_layout.addStretch()
        self._radius_scroll.setWidget(self._radius_container)
        v3.addWidget(self._radius_scroll)
        btn_rebuild = QPushButton("Rebuild Ragdoll")
        btn_rebuild.clicked.connect(self._rebuild_ragdoll)
        v3.addWidget(btn_rebuild)
        vbox.addWidget(grp3)

        # ── Step 4: Play Animation ────────────────────────────────────
        grp4 = QGroupBox("Step 4: Play Animation")
        v4   = QVBoxLayout(grp4)
        row_fr = QHBoxLayout()
        row_fr.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        row_fr.addWidget(self.frame_slider)
        self.frame_lbl = QLabel("0 / 0")
        self.frame_lbl.setFixedWidth(60)
        row_fr.addWidget(self.frame_lbl)
        v4.addLayout(row_fr)
        row_p = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._on_play_toggled)
        row_p.addWidget(self.btn_play)
        row_p.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(
            lambda v: self._play_timer.setInterval(max(1, 1000 // v))
            if self._play_timer.isActive() else None
        )
        row_p.addWidget(self.fps_spin)
        row_p.addStretch()
        v4.addLayout(row_p)
        vbox.addWidget(grp4)

        # ── Step 5: Carve Voxels ──────────────────────────────────────
        grp5 = QGroupBox("Step 5: Carve Voxels")
        v5   = QVBoxLayout(grp5)

        self._mode_group  = QButtonGroup(self)
        self.radio_single = QRadioButton("Single frame (current)")
        self.radio_bool   = QRadioButton("All frames — sequential")
        self.radio_weighted = QRadioButton("All frames — weighted")
        self.radio_bool.setChecked(True)
        for rb in (self.radio_single, self.radio_bool, self.radio_weighted):
            self._mode_group.addButton(rb)
            v5.addWidget(rb)
        self.radio_weighted.toggled.connect(self._on_mode_changed)

        row_thr = QHBoxLayout()
        row_thr.addWidget(QLabel("Threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(30)
        self.threshold_lbl = QLabel("0.30")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_lbl.setText(f"{v / 100:.2f}")
        )
        row_thr.addWidget(self.threshold_slider)
        row_thr.addWidget(self.threshold_lbl)
        v5.addLayout(row_thr)
        self._threshold_row_widget = QWidget()
        self._threshold_row_widget.setLayout(row_thr)
        self._threshold_row_widget.setVisible(False)
        v5.addWidget(self._threshold_row_widget)

        self.btn_carve = QPushButton("▶ Carve")
        self.btn_carve.setStyleSheet("font-weight: bold;")
        self.btn_carve.clicked.connect(self._run_carve)
        v5.addWidget(self.btn_carve)
        self.carve_progress = QProgressBar()
        self.carve_progress.setRange(0, 100)
        self.carve_progress.setVisible(False)
        v5.addWidget(self.carve_progress)
        btn_reset = QPushButton("Reset Voxels")
        btn_reset.clicked.connect(self._reset_voxels)
        v5.addWidget(btn_reset)
        vbox.addWidget(grp5)

        # ── Step 6: Bake Mesh ─────────────────────────────────────────
        grp6 = QGroupBox("Step 6: Bake Mesh")
        v6   = QVBoxLayout(grp6)
        btn_bake = QPushButton("Bake Mesh (marching cubes)")
        btn_bake.setStyleSheet("font-weight: bold;")
        btn_bake.clicked.connect(self._bake_mesh)
        v6.addWidget(btn_bake)
        self.mesh_lbl = QLabel("—")
        v6.addWidget(self.mesh_lbl)
        vbox.addWidget(grp6)

        # ── Step 7: Save ─────────────────────────────────────────────
        grp7 = QGroupBox("Step 7: Save")
        v7   = QVBoxLayout(grp7)
        btn_save = QPushButton("Save…")
        btn_save.clicked.connect(self._save)
        v7.addWidget(btn_save)
        vbox.addWidget(grp7)

        # Status
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#88f; font-style: italic;")
        self.status_lbl.setWordWrap(True)
        vbox.addWidget(self.status_lbl)
        vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(420)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _build_right_panel(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Vertical)

        self._mesh_viewer = MeshViewer3D()
        splitter.addWidget(self._mesh_viewer)

        self._dir_labels: list[QLabel] = []
        strip_widget = QWidget()
        strip_layout = QHBoxLayout(strip_widget)
        strip_layout.setContentsMargins(2, 2, 2, 2)
        strip_layout.setSpacing(2)
        for d in range(6):
            lbl = QLabel(f"Dir {d}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedWidth(120)
            lbl.setMinimumHeight(90)
            lbl.setStyleSheet("background:#222; color:#aaa; font-size:10px;")
            strip_layout.addWidget(lbl)
            self._dir_labels.append(lbl)
        strip_layout.addStretch()
        splitter.addWidget(strip_widget)

        splitter.setSizes([520, 140])
        return splitter

    # ------------------------------------------------------------------
    # Step 1: Import Skeleton
    # ------------------------------------------------------------------

    def _animation_skeleton(self) -> np.ndarray | None:
        """Return (N, 33, 3) per-frame skeleton, preferring SkeletonBuilder."""
        char = self.state.current_character
        if char is None:
            return None
        if char.skeleton is not None and getattr(char.skeleton, "poses", None) is not None:
            return char.skeleton.poses[:, :33, :]
        return char.skeleton_3d

    def _use_skeleton(self):
        char = self.state.current_character
        if char is None:
            self._set_status("No character loaded.")
            return
        sb = getattr(char, "skeleton", None)
        if sb is None or getattr(sb, "poses", None) is None:
            self._set_status("No skeleton — run triangulation + skeleton build first.")
            return
        self._skeleton_builder = sb
        n = sb.poses.shape[0]
        self.frame_slider.setRange(0, max(0, n - 1))
        self.frame_slider.setValue(0)
        self._current_frame = 0
        self.frame_lbl.setText(f"1 / {n}")
        self.skeleton_lbl.setText(f"Skeleton: {n} frames loaded")
        self._set_status(f"Skeleton ready ({n} frames).")

    # ------------------------------------------------------------------
    # Step 2: Generate Voxel Ragdoll
    # ------------------------------------------------------------------

    def _generate_ragdoll(self):
        if self._skeleton_builder is None:
            self._set_status("Import a skeleton first (Step 1).")
            return
        cam = self._get_camera_setup()
        if cam is None:
            self._set_status("Camera setup unavailable.")
            return

        resolution = self.resolution_combo.currentData()
        self._carver = VoxelCarver(
            self._skeleton_builder, cam,
            bone_radii=None,
            resolution=resolution,
        )
        n = len(self._carver.sausages)
        self.ragdoll_lbl.setText(f"{n} bones, {resolution}³ voxels each")
        self._populate_radius_panel()
        self._show_voxel_cloud()
        self._set_status(f"Voxel ragdoll ready — {n} bones at {resolution}³.")

    def _populate_radius_panel(self):
        """Fill radius spinboxes from current carver sausages."""
        # Clear existing widgets (except stretch)
        while self._radius_layout.count() > 1:
            item = self._radius_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._radius_spins.clear()

        if self._carver is None:
            return

        for jidx, sausage in self._carver.sausages.items():
            name = BONE_NAMES.get(jidx, str(jidx))
            row  = QWidget()
            hl   = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(QLabel(name))
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 0.5)
            spin.setSingleStep(0.005)
            spin.setDecimals(3)
            spin.setValue(sausage.radius)
            hl.addWidget(spin)
            self._radius_layout.insertWidget(
                self._radius_layout.count() - 1, row
            )
            self._radius_spins[jidx] = spin

    # ------------------------------------------------------------------
    # Step 3: Rebuild Ragdoll with new radii
    # ------------------------------------------------------------------

    def _rebuild_ragdoll(self):
        if self._carver is None:
            self._set_status("Generate a ragdoll first (Step 2).")
            return
        for jidx, spin in self._radius_spins.items():
            if jidx in self._carver.sausages:
                self._carver.sausages[jidx].radius = spin.value()
        self._carver.reset()
        self._clear_scatter()
        self._show_voxel_cloud()
        self._set_status("Ragdoll rebuilt with updated radii.")

    def _set_all_radii(self):
        r = self._all_radius_spin.value()
        for spin in self._radius_spins.values():
            spin.setValue(r)

    def _write_back_radii(self):
        """Update radius spinboxes from max radial extent of carved voxels."""
        if self._carver is None:
            return
        for jidx, sausage in self._carver.sausages.items():
            r = sausage.max_radial_distance()
            if r > 0.001 and jidx in self._radius_spins:
                self._radius_spins[jidx].setValue(round(r, 4))
        self._set_status(
            "Radii updated from carved voxels — adjust if needed, "
            "then click Rebuild + Carve for a tighter pass."
        )

    # ------------------------------------------------------------------
    # Step 4: Play Animation
    # ------------------------------------------------------------------

    def _on_play_toggled(self, playing: bool):
        if playing:
            self.btn_play.setText("⏸ Pause")
            self._play_timer.start(max(1, 1000 // self.fps_spin.value()))
        else:
            self.btn_play.setText("▶ Play")
            self._play_timer.stop()

    def _advance_frame(self):
        skel = self._animation_skeleton()
        if skel is None:
            self.btn_play.setChecked(False)
            return
        n = skel.shape[0]
        if n == 0:
            self.btn_play.setChecked(False)
            return
        next_f = (self._current_frame + 1) % n
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(next_f)
        self.frame_slider.blockSignals(False)
        self._current_frame = next_f
        self._update_for_frame()

    def _on_frame_slider(self, val: int):
        self._current_frame = val
        self._update_for_frame()

    def _update_for_frame(self):
        skel = self._animation_skeleton()
        if skel is not None:
            n = skel.shape[0]
            f = max(0, min(self._current_frame, n - 1))
            self.frame_lbl.setText(f"{f + 1} / {n}")
            try:
                self._mesh_viewer.set_skeleton_overlay(skel[f], POSE_CONNECTIONS)
            except Exception:
                pass
            if self._carver is not None:
                self._show_voxel_cloud(frame_idx=self._current_frame)
        self._update_sprite_strip(self._current_frame)

    # ------------------------------------------------------------------
    # Step 5: Carve Voxels
    # ------------------------------------------------------------------

    def _on_mode_changed(self, weighted: bool):
        self._threshold_row_widget.setVisible(weighted)

    def _collect_silhouette_masks(
        self, frame_indices: list
    ) -> list:
        """Return list[f] of list[6] of (H,W) uint8 arrays."""
        char = self.state.current_character
        if char is None:
            return []
        usc = getattr(char, "upscaled_frames", None)  # (6, N, H, W, C)
        frm = getattr(char, "frames", None)            # (6, N, H, W, 3)
        result = []
        for f in frame_indices:
            masks_f = []
            for d in range(6):
                try:
                    if usc is not None:
                        img = usc[d, f]
                    elif frm is not None:
                        img = frm[d, f]
                    else:
                        masks_f.append(np.zeros((64, 64), dtype=np.uint8))
                        continue
                    if img.ndim == 3 and img.shape[-1] == 4:
                        mask = (img[:, :, 3] > 0).astype(np.uint8)
                    else:
                        mask = (img.mean(axis=-1) > 10).astype(np.uint8)
                    masks_f.append(mask)
                except Exception:
                    masks_f.append(np.zeros((64, 64), dtype=np.uint8))
            result.append(masks_f)
        return result

    def _run_carve(self):
        if self._carver is None:
            self._set_status("Generate a ragdoll first (Step 2).")
            return
        char = self.state.current_character
        if char is None:
            self._set_status("No character loaded.")
            return

        n_frames = self._skeleton_builder.poses.shape[0]
        weighted = self.radio_weighted.isChecked()

        if self.radio_single.isChecked():
            frame_indices = [self._current_frame]
        else:
            frame_indices = list(range(n_frames))

        all_masks = self._collect_silhouette_masks(frame_indices)
        if not all_masks:
            self._set_status("No silhouette masks available.")
            return

        # Reset voxels before carving so repeated runs are independent
        self._carver.reset()

        self.carve_progress.setVisible(True)
        self.carve_progress.setValue(0)
        self.btn_carve.setEnabled(False)

        self._carve_worker = CarveWorker(self._carver, all_masks, weighted)
        self._carve_thread = QThread(self)
        self._carve_worker.moveToThread(self._carve_thread)
        self._carve_thread.started.connect(self._carve_worker.run)
        self._carve_worker.progress.connect(self._on_carve_progress)
        self._carve_worker.finished.connect(
            lambda: self._on_carve_done(len(frame_indices), weighted)
        )
        self._carve_worker.error.connect(self._on_carve_error)
        self._carve_thread.start()

    def _on_carve_progress(self, done: int, total: int):
        if total > 0:
            self.carve_progress.setValue(int(100 * done / total))

    def _on_carve_done(self, n_frames: int, weighted: bool):
        self._carve_thread.quit()
        self.carve_progress.setVisible(False)
        self.btn_carve.setEnabled(True)
        self.carve_progress.setValue(100)

        if weighted:
            thr = self.threshold_slider.value() / 100.0
            for s in self._carver.sausages.values():
                s.finalise_weighted(n_frames, thr)

        counts = {BONE_NAMES.get(j, str(j)): s.occupied_count()
                  for j, s in self._carver.sausages.items()}
        summary = ", ".join(f"{n}={c}" for n, c in list(counts.items())[:6])
        self._set_status(f"Carved — {summary} …")

        self._clear_scatter()
        self._show_voxel_cloud()
        self._write_back_radii()

    def _on_carve_error(self, msg: str):
        self._carve_thread.quit()
        self.carve_progress.setVisible(False)
        self.btn_carve.setEnabled(True)
        self._set_status(f"Carve error: {msg}")

    def _reset_voxels(self):
        if self._carver is None:
            return
        self._carver.reset()
        self._clear_scatter()
        self._show_voxel_cloud()
        self._set_status("Voxels reset to initial capsule.")

    # ------------------------------------------------------------------
    # Step 6: Bake Mesh
    # ------------------------------------------------------------------

    def _bake_mesh(self):
        if self._carver is None:
            self._set_status("Carve voxels first (Step 5).")
            return
        try:
            self._carver.bake_all()
        except ImportError as exc:
            self._set_status(str(exc))
            return
        except Exception as exc:
            self._set_status(f"Bake error: {exc}")
            _logger.error("bake_all error: %s", exc)
            return

        verts, faces, bw = self._carver.to_combined_mesh()
        if len(verts) == 0:
            self._set_status("No voxels survived — mesh is empty.")
            return

        self._mesh_verts  = verts
        self._mesh_faces  = faces
        self._bone_weights = bw
        self._clear_scatter()
        self._mesh_viewer.set_mesh(verts, faces, None, 0)
        self.mesh_lbl.setText(f"{len(verts)} verts, {len(faces)} faces")
        self._set_status(f"Mesh baked — {len(verts)} verts, {len(faces)} faces.")

    # ------------------------------------------------------------------
    # Step 7: Save
    # ------------------------------------------------------------------

    def _save(self):
        if self._carver is None:
            self._set_status("Nothing to save yet.")
            return
        char = self.state.current_character
        name = getattr(char, "name", "character") if char else "character"

        save_dir = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if not save_dir:
            return

        # ── .npz voxels
        npz_path = os.path.join(save_dir, f"{name}_voxels.npz")
        try:
            self._carver.save(npz_path)
        except Exception as exc:
            self._set_status(f"Save voxels error: {exc}")
            return

        # ── skeleton JSON
        sb = self._skeleton_builder
        if sb is not None:
            import json
            skel_path = os.path.join(save_dir, f"{name}_skeleton.json")
            try:
                with open(skel_path, "w") as fh:
                    json.dump(sb.to_dict(include_poses=False), fh, indent=2)
            except Exception as exc:
                _logger.warning("skeleton JSON save failed: %s", exc)

        # ── .glb mesh
        if self._mesh_verts is not None and self._mesh_faces is not None:
            glb_path = os.path.join(save_dir, f"{name}_sausages.glb")
            try:
                from pipeline.gltf_exporter import GLTFExporter
                skel_seq = sb.poses[:, :33, :].astype(np.float32) if sb is not None else None
                n_joints = 33
                skin_w = np.zeros((len(self._mesh_verts), n_joints), dtype=np.float32)
                for vi, jidx in enumerate(self._bone_weights):
                    col = int(jidx) if int(jidx) < n_joints else 0
                    skin_w[vi, col] = 1.0
                exp = GLTFExporter()
                exp.export_glb(
                    glb_path,
                    self._mesh_verts.astype(np.float32),
                    self._mesh_faces.astype(np.int32),
                    skel_seq,
                    skinning_weights=skin_w,
                )
            except Exception as exc:
                _logger.warning("GLB save failed: %s", exc)

        self._set_status(f"Saved to {save_dir}.")

    # ------------------------------------------------------------------
    # 3D viewer helpers
    # ------------------------------------------------------------------

    def _clear_scatter(self):
        """Remove any voxel scatter items from the 3D viewer."""
        try:
            self._mesh_viewer.clear_extra_items()
        except Exception:
            pass
        self._scatter_items.clear()

    def _show_voxel_cloud(self, frame_idx: int = 0):
        """Display occupied voxel centres at the given frame's bone poses."""
        if not _GL or self._carver is None:
            return
        self._clear_scatter()

        poses = self._skeleton_builder.poses if self._skeleton_builder else None
        if poses is None or poses.shape[0] == 0:
            return

        f = max(0, min(frame_idx, poses.shape[0] - 1))
        animating = (f != 0)
        cap = 1500 if (animating and len(self._carver.sausages) > 12) else 4000

        for ci, (jidx, sausage) in enumerate(self._carver.sausages.items()):
            if sausage.voxels is None or not sausage.voxels.any():
                continue
            idx = np.argwhere(sausage.voxels)          # (M, 3)
            if len(idx) == 0:
                continue
            local_pts = sausage.grid_origin + idx * sausage.voxel_size  # (M, 3)

            head_w = poses[f, sausage.parent_idx]
            tail_w = poses[f, sausage.joint_idx]
            l2w, _ = BoneSausage._build_bone_matrix(head_w, tail_w)
            M = len(local_pts)
            world_pts = (l2w @ np.hstack([local_pts, np.ones((M, 1))]).T).T[:, :3]

            if len(world_pts) > cap:
                rng = np.random.default_rng(jidx)
                world_pts = world_pts[rng.choice(len(world_pts), cap, replace=False)]

            r, g, b = _BONE_COLOURS[ci % len(_BONE_COLOURS)]
            try:
                scatter = gl.GLScatterPlotItem(
                    pos=world_pts.astype(np.float32),
                    color=(r, g, b, 0.8),
                    size=2.5,
                )
                self._mesh_viewer.add_extra_item(scatter)
                self._scatter_items.append(scatter)
            except Exception as exc:
                _logger.debug("scatter plot failed for bone %d: %s", jidx, exc)

    # ------------------------------------------------------------------
    # Sprite strip
    # ------------------------------------------------------------------

    def _update_sprite_strip(self, frame_idx: int):
        char = self.state.current_character
        if char is None:
            return
        usc = getattr(char, "upscaled_frames", None)  # (6, N, H, W, C)
        frm = getattr(char, "frames", None)            # (6, N, H, W, 3)

        for d, lbl in enumerate(self._dir_labels):
            try:
                if usc is not None and frame_idx < usc.shape[1]:
                    img = usc[d, frame_idx]
                elif frm is not None and frame_idx < frm.shape[1]:
                    img = frm[d, frame_idx]
                else:
                    lbl.setText(f"Dir {d}")
                    continue

                img_rgb = img[:, :, :3].astype(np.uint8)

                # Green silhouette tint (30 % opacity)
                if img.ndim == 3 and img.shape[-1] == 4:
                    mask = img[:, :, 3] > 0
                else:
                    mask = img_rgb.mean(axis=-1) > 10
                overlay = img_rgb.copy()
                overlay[mask] = (
                    overlay[mask] * 0.7 +
                    np.array([0, 200, 0], dtype=np.float32) * 0.3
                ).clip(0, 255).astype(np.uint8)

                h, w = overlay.shape[:2]
                tgt_w, tgt_h = 116, max(1, int(h * 116 / max(w, 1)))
                scaled = cv2.resize(overlay, (tgt_w, tgt_h))
                qi = QImage(scaled.data, scaled.shape[1], scaled.shape[0],
                            int(scaled.strides[0]), QImage.Format.Format_RGB888)
                lbl.setPixmap(QPixmap.fromImage(qi.copy()))
            except Exception:
                lbl.setText(f"Dir {d}")

    # ------------------------------------------------------------------
    # Camera setup (copied from MeshBuilderTab)
    # ------------------------------------------------------------------

    def _get_camera_setup(self, char=None):
        p = self.parent()
        while p is not None:
            if hasattr(p, "tab_recon"):
                return p.tab_recon._triangulator.camera_setup
            p = p.parent()
        if char is None:
            char = self.state.current_character
        if char is None or char.frames is None or char.frames.size == 0:
            return None
        from pipeline.pose_triangulator import PoseTriangulator
        h, w = char.frames[0, 0].shape[:2]
        return PoseTriangulator(image_size=(w, h)).camera_setup

    # ------------------------------------------------------------------
    # AppState hooks
    # ------------------------------------------------------------------

    def _on_char_changed(self, idx: int):
        self._skeleton_builder = None
        self._carver = None
        self._mesh_verts = None
        self._mesh_faces = None
        self._clear_scatter()
        char = self.state.current_character
        if char is None:
            self.skeleton_lbl.setText("Skeleton: not loaded")
            return
        sb = getattr(char, "skeleton", None)
        if sb is not None and getattr(sb, "poses", None) is not None:
            n = sb.poses.shape[0]
            self.skeleton_lbl.setText(
                f"Skeleton: {n} frames available — click Use to load"
            )
        else:
            self.skeleton_lbl.setText("Skeleton: not loaded")

    def _on_char_updated(self, idx: int):
        if idx == self.state.selected_idx:
            self._on_char_changed(idx)

    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        self.status_lbl.setText(text)
        _logger.info("VoxelSausage: %s", text)
