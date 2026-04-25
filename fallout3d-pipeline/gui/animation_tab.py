"""
Tab 6 — Animation Synthesis
Frame interpolation (LERP / SLERP), BVH mocap import & retargeting,
procedural idle/hit motions, and a side-by-side 3D preview.
"""

import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QComboBox, QGroupBox, QSplitter, QProgressBar,
    QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QScrollArea, QFrame,
    QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage

from gui.main_window import AppState
from pipeline.animation_synthesizer import AnimationSynthesizer

try:
    import pyqtgraph.opengl as gl
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False

from pipeline.pose_triangulator import POSE_CONNECTIONS


# -----------------------------------------------------------------------
# Workers
# -----------------------------------------------------------------------

class InterpolationWorker(QObject):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, synth, skeleton_seq, n_interp, mode, keyframes):
        super().__init__()
        self.synth = synth
        self.skeleton_seq = skeleton_seq
        self.n_interp = n_interp
        self.mode = mode
        self.keyframes = keyframes

    def run(self):
        try:
            result = self.synth.interpolate(
                self.skeleton_seq,
                n_interp=self.n_interp,
                mode=self.mode,
                keyframe_indices=self.keyframes if self.keyframes else None,
                progress_cb=lambda f, t: self.progress.emit(f, t),
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class ProceduralWorker(QObject):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, synth, skeleton_seq, mode, **kwargs):
        super().__init__()
        self.synth = synth
        self.skeleton_seq = skeleton_seq
        self.mode = mode
        self.kwargs = kwargs

    def run(self):
        try:
            if self.mode == "breathing":
                result = self.synth.add_idle_breathing(self.skeleton_seq, **self.kwargs)
            else:
                result = self.synth.add_hit_reaction(self.skeleton_seq, **self.kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class BVHWorker(QObject):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, synth, path, reference, scale):
        super().__init__()
        self.synth = synth
        self.path = path
        self.reference = reference
        self.scale = scale

    def run(self):
        try:
            hierarchy, motion = self.synth.load_bvh(self.path)
            result = self.synth.retarget_bvh(
                hierarchy, motion, self.reference, scale=self.scale
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# 3D skeleton viewer (reused from reconstruction_tab pattern)
# -----------------------------------------------------------------------

_PART_RGBA = {
    "face":  (1.0, 0.0, 1.0, 1.0),
    "torso": (0.2, 0.9, 0.2, 1.0),
    "arms":  (0.3, 0.5, 1.0, 1.0),
    "legs":  (1.0, 1.0, 0.2, 1.0),
}
_LM_PART = {**{i: "face" for i in range(11)},
            **{i: "torso" for i in [11, 12, 23, 24]},
            **{i: "arms" for i in [13, 14, 15, 16]},
            **{i: "legs" for i in [25, 26, 27, 28]}}


class MiniSkeletonViewer(QWidget):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if title:
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)

        if _GL_AVAILABLE:
            self._view = gl.GLViewWidget()
            self._view.setBackgroundColor("k")
            self._view.setCameraPosition(distance=4, elevation=20, azimuth=45)
            grid = gl.GLGridItem()
            grid.scale(0.5, 0.5, 0.5)
            self._view.addItem(grid)
            self._items: list = []
            layout.addWidget(self._view)
        else:
            layout.addWidget(QLabel("pyqtgraph not available."))

    def set_skeleton(self, skeleton: np.ndarray, color=(0.4, 0.9, 0.4)):
        if not _GL_AVAILABLE:
            return
        for item in self._items:
            self._view.removeItem(item)
        self._items = []

        for s, e in POSE_CONNECTIONS:
            if np.all(skeleton[s] == 0) or np.all(skeleton[e] == 0):
                continue
            pts = np.array([skeleton[s], skeleton[e]], dtype=np.float32)
            col = np.array([[*color, 1.0]] * 2, dtype=np.float32)
            line = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True)
            self._view.addItem(line)
            self._items.append(line)

        valid = ~np.all(skeleton == 0, axis=1)
        if valid.any():
            sc = gl.GLScatterPlotItem(
                pos=skeleton[valid].astype(np.float32),
                color=np.array([[*color, 1.0]] * valid.sum(), dtype=np.float32),
                size=7, pxMode=True,
            )
            self._view.addItem(sc)
            self._items.append(sc)


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class AnimationTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._synth = AnimationSynthesizer()
        self._thread: QThread | None = None

        # Synthesised sequence (separate from character's original)
        self._synth_sequence: np.ndarray | None = None
        self._current_frame: int = 0

        self._build_ui()
        self.state.selection_changed.connect(self._on_char_changed)
        self.state.character_updated.connect(self._on_char_updated)

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ---- Left controls panel ------------------------------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(340)
        left_scroll.setMaximumWidth(400)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_scroll.setWidget(left_widget)
        root.addWidget(left_scroll)

        # ---- Section: Frame Interpolation ---------------------------
        interp_box = QGroupBox("Frame Interpolation")
        interp_layout = QVBoxLayout(interp_box)

        # Mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("SLERP (smooth rotations)", "slerp")
        self.mode_combo.addItem("LERP (linear positions)", "lerp")
        mode_row.addWidget(self.mode_combo)
        interp_layout.addLayout(mode_row)

        # N interp
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("Frames between keyframes:"))
        self.n_interp_spin = QSpinBox()
        self.n_interp_spin.setRange(1, 30)
        self.n_interp_spin.setValue(3)
        n_row.addWidget(self.n_interp_spin)
        interp_layout.addLayout(n_row)

        # Keyframe selection hint
        interp_layout.addWidget(QLabel("(Leave empty = use all frames as keyframes)"))

        self.btn_interpolate = QPushButton("Generate Interpolated Sequence")
        self.btn_interpolate.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_interpolate.clicked.connect(self._run_interpolation)
        interp_layout.addWidget(self.btn_interpolate)

        left_layout.addWidget(interp_box)

        # ---- Section: Procedural Motions ----------------------------
        proc_box = QGroupBox("Procedural Motions")
        proc_layout = QVBoxLayout(proc_box)

        # Idle breathing
        breath_row = QHBoxLayout()
        self.chk_breathing = QCheckBox("Idle Breathing")
        self.chk_breathing.setChecked(True)
        breath_row.addWidget(self.chk_breathing)
        breath_row.addWidget(QLabel("Amplitude:"))
        self.breath_amp = QDoubleSpinBox()
        self.breath_amp.setRange(0.001, 0.1)
        self.breath_amp.setSingleStep(0.005)
        self.breath_amp.setValue(0.015)
        breath_row.addWidget(self.breath_amp)
        proc_layout.addLayout(breath_row)

        # Hit reaction
        hit_row = QHBoxLayout()
        self.chk_hit = QCheckBox("Hit Reaction at frame:")
        hit_row.addWidget(self.chk_hit)
        self.hit_frame_spin = QSpinBox()
        self.hit_frame_spin.setRange(0, 999)
        self.hit_frame_spin.setValue(0)
        hit_row.addWidget(self.hit_frame_spin)
        proc_layout.addLayout(hit_row)

        hit_int_row = QHBoxLayout()
        hit_int_row.addWidget(QLabel("Intensity:"))
        self.hit_intensity = QDoubleSpinBox()
        self.hit_intensity.setRange(0.01, 0.5)
        self.hit_intensity.setSingleStep(0.01)
        self.hit_intensity.setValue(0.05)
        hit_int_row.addWidget(self.hit_intensity)
        proc_layout.addLayout(hit_int_row)

        self.btn_procedural = QPushButton("Apply Procedural Motions")
        self.btn_procedural.clicked.connect(self._run_procedural)
        proc_layout.addWidget(self.btn_procedural)

        left_layout.addWidget(proc_box)

        # ---- Section: BVH Import ------------------------------------
        bvh_box = QGroupBox("BVH Mocap Import")
        bvh_layout = QVBoxLayout(bvh_box)

        bvh_file_row = QHBoxLayout()
        self.bvh_path_lbl = QLabel("No file selected.")
        self.bvh_path_lbl.setWordWrap(True)
        bvh_file_row.addWidget(self.bvh_path_lbl)
        btn_pick_bvh = QPushButton("Browse…")
        btn_pick_bvh.clicked.connect(self._pick_bvh)
        bvh_file_row.addWidget(btn_pick_bvh)
        bvh_layout.addLayout(bvh_file_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale:"))
        self.bvh_scale_spin = QDoubleSpinBox()
        self.bvh_scale_spin.setRange(0.001, 100.0)
        self.bvh_scale_spin.setValue(0.01)
        self.bvh_scale_spin.setSingleStep(0.005)
        scale_row.addWidget(self.bvh_scale_spin)
        bvh_layout.addLayout(scale_row)

        self.btn_import_bvh = QPushButton("Import & Retarget")
        self.btn_import_bvh.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_import_bvh.clicked.connect(self._run_bvh_import)
        bvh_layout.addWidget(self.btn_import_bvh)

        self._bvh_path: str = ""
        left_layout.addWidget(bvh_box)

        # ---- Commit to character ------------------------------------
        commit_box = QGroupBox("Commit Synthesised Sequence")
        commit_layout = QVBoxLayout(commit_box)
        self.btn_commit = QPushButton("Replace Character Skeleton with Synthesised")
        self.btn_commit.clicked.connect(self._commit_to_character)
        commit_layout.addWidget(self.btn_commit)
        self.btn_append = QPushButton("Append to Character Skeleton")
        self.btn_append.clicked.connect(self._append_to_character)
        commit_layout.addWidget(self.btn_append)
        left_layout.addWidget(commit_box)

        # Progress + status
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)
        self.status_lbl = QLabel("")
        left_layout.addWidget(self.status_lbl)
        left_layout.addStretch()

        # ---- Right: side-by-side 3D preview -------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        root.addWidget(right, 1)

        # Frame navigation
        nav_row = QHBoxLayout()
        nav_row.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider)
        nav_row.addWidget(self.frame_slider)
        self.frame_lbl = QLabel("0 / 0")
        nav_row.addWidget(self.frame_lbl)
        right_layout.addLayout(nav_row)

        # Two 3D viewers side by side
        viewers_splitter = QSplitter(Qt.Orientation.Horizontal)
        right_layout.addWidget(viewers_splitter, 1)

        self._orig_viewer  = MiniSkeletonViewer("Original", parent=self)
        self._synth_viewer = MiniSkeletonViewer("Synthesised", parent=self)
        viewers_splitter.addWidget(self._orig_viewer)
        viewers_splitter.addWidget(self._synth_viewer)

        # Stats
        self.stats_lbl = QLabel("—")
        self.stats_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.stats_lbl)

    # ------------------------------------------------------------------
    # Character state callbacks
    # ------------------------------------------------------------------

    def _on_char_changed(self, idx: int):
        char = self.state.current_character
        if char is None or char.skeleton_3d is None:
            return
        n = char.skeleton_3d.shape[0]
        self.frame_slider.setRange(0, max(0, n - 1))
        self.hit_frame_spin.setMaximum(n - 1)
        self._update_views(0)

    def _on_char_updated(self, idx: int):
        if idx == self.state.selected_idx:
            self._on_char_changed(idx)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _run_interpolation(self):
        char = self.state.current_character
        if char is None or char.skeleton_3d is None:
            self.status_lbl.setText("Load and triangulate a character first.")
            return
        mode = self.mode_combo.currentData()
        n_interp = self.n_interp_spin.value()

        self.btn_interpolate.setEnabled(False)
        self.progress.setVisible(True)
        self.status_lbl.setText("Interpolating…")

        self._worker = InterpolationWorker(
            self._synth, char.skeleton_3d, n_interp, mode, []
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_interp_done)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.progress.setValue(int(100 * current / (total + 1e-6)))

    def _on_interp_done(self, result: np.ndarray):
        self._finish_thread()
        self._synth_sequence = result
        n = result.shape[0]
        self.frame_slider.setRange(0, max(0, n - 1))
        orig_n = self.state.current_character.skeleton_3d.shape[0] if self.state.current_character else 0
        self.status_lbl.setText(f"Done — {orig_n} → {n} frames.")
        self.stats_lbl.setText(
            f"Original: {orig_n} frames  →  Synthesised: {n} frames  "
            f"({self.n_interp_spin.value()} interpolated between each pair)"
        )
        self._update_views(0)

    # ------------------------------------------------------------------
    # Procedural
    # ------------------------------------------------------------------

    def _run_procedural(self):
        base = self._synth_sequence if self._synth_sequence is not None else (
            self.state.current_character.skeleton_3d if self.state.current_character else None
        )
        if base is None:
            self.status_lbl.setText("No skeleton available.")
            return
        seq = base.copy()
        try:
            if self.chk_breathing.isChecked():
                seq = self._synth.add_idle_breathing(seq, amplitude=self.breath_amp.value())
            if self.chk_hit.isChecked():
                seq = self._synth.add_hit_reaction(
                    seq,
                    hit_frame=self.hit_frame_spin.value(),
                    intensity=self.hit_intensity.value(),
                )
        except Exception as exc:
            self.status_lbl.setText(f"Error: {exc}")
            return
        self._synth_sequence = seq
        n = seq.shape[0]
        self.frame_slider.setRange(0, max(0, n - 1))
        self.status_lbl.setText("Procedural motions applied.")
        self._update_views(self._current_frame)

    # ------------------------------------------------------------------
    # BVH
    # ------------------------------------------------------------------

    def _pick_bvh(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open BVH File", "", "BVH (*.bvh)")
        if path:
            self._bvh_path = path
            self.bvh_path_lbl.setText(os.path.basename(path))

    def _run_bvh_import(self):
        if not self._bvh_path or not os.path.exists(self._bvh_path):
            self.status_lbl.setText("Select a .bvh file first.")
            return
        char = self.state.current_character
        if char is None or char.skeleton_3d is None:
            self.status_lbl.setText("Load and triangulate a character first.")
            return

        self.btn_import_bvh.setEnabled(False)
        self.progress.setVisible(True)
        self.status_lbl.setText("Importing BVH…")

        self._bvh_worker = BVHWorker(
            self._synth, self._bvh_path,
            char.skeleton_3d[0], self.bvh_scale_spin.value()
        )
        self._bvh_thread = QThread(self)
        self._bvh_worker.moveToThread(self._bvh_thread)
        self._bvh_thread.started.connect(self._bvh_worker.run)
        self._bvh_worker.finished.connect(self._on_bvh_done)
        self._bvh_worker.error.connect(self._on_bvh_error)
        self._bvh_thread.start()

    def _on_bvh_done(self, result: np.ndarray):
        self._bvh_thread.quit()
        self.progress.setVisible(False)
        self.btn_import_bvh.setEnabled(True)
        self._synth_sequence = result
        n = result.shape[0]
        self.frame_slider.setRange(0, max(0, n - 1))
        self.status_lbl.setText(f"BVH retargeted — {n} frames.")
        self._update_views(0)

    def _on_bvh_error(self, msg: str):
        self._bvh_thread.quit()
        self.progress.setVisible(False)
        self.btn_import_bvh.setEnabled(True)
        self.status_lbl.setText(f"BVH error: {msg}")

    # ------------------------------------------------------------------
    # Commit to character
    # ------------------------------------------------------------------

    def _commit_to_character(self):
        if self._synth_sequence is None:
            return
        char = self.state.current_character
        if char is None:
            return
        char.skeleton_3d = self._synth_sequence.copy()
        self.state.character_updated.emit(self.state.selected_idx)
        self.status_lbl.setText("Skeleton replaced.")

    def _append_to_character(self):
        if self._synth_sequence is None:
            return
        char = self.state.current_character
        if char is None:
            return
        if char.skeleton_3d is not None:
            char.skeleton_3d = np.concatenate([char.skeleton_3d, self._synth_sequence], axis=0)
        else:
            char.skeleton_3d = self._synth_sequence.copy()
        self.state.character_updated.emit(self.state.selected_idx)
        n = char.skeleton_3d.shape[0]
        self.status_lbl.setText(f"Appended → {n} frames total.")

    # ------------------------------------------------------------------
    # View helpers
    # ------------------------------------------------------------------

    def _on_slider(self, val: int):
        self._current_frame = val
        self._update_views(val)

    def _update_views(self, frame: int):
        char = self.state.current_character

        # Original view
        if char and char.skeleton_3d is not None:
            f = min(frame, char.skeleton_3d.shape[0] - 1)
            self._orig_viewer.set_skeleton(char.skeleton_3d[f], color=(0.4, 0.9, 0.4))
        else:
            self._orig_viewer.set_skeleton(np.zeros((33, 3)))

        # Synthesised view
        if self._synth_sequence is not None:
            f = min(frame, self._synth_sequence.shape[0] - 1)
            self._synth_viewer.set_skeleton(self._synth_sequence[f], color=(1.0, 0.6, 0.2))
            total = self._synth_sequence.shape[0]
            self.frame_lbl.setText(f"{f + 1} / {total}")
        else:
            self._synth_viewer.set_skeleton(np.zeros((33, 3)))

    def _finish_thread(self):
        if self._thread:
            self._thread.quit()
        self.progress.setVisible(False)
        self.btn_interpolate.setEnabled(True)

    def _on_error(self, msg: str):
        self._finish_thread()
        self.status_lbl.setText(f"Error: {msg}")
