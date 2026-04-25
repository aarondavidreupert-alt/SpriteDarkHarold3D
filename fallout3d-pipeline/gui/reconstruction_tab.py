"""
Tab 3 — 3D Reconstruction
Triangulates the 2D poses into a 3D skeleton, shows a rotatable
pyqtgraph OpenGL skeleton viewer, and colour-codes back-projection error.
"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QTableWidget, QTableWidgetItem, QSplitter,
    QGroupBox, QSlider,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

from gui.main_window import AppState
from pipeline.pose_triangulator import POSE_CONNECTIONS

# Optional pyqtgraph OpenGL
try:
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False


# -----------------------------------------------------------------------
# Skeleton colours
# -----------------------------------------------------------------------

_PART_RGBA = {
    "face":  (1.0, 0.0, 1.0, 1.0),
    "torso": (0.0, 1.0, 0.0, 1.0),
    "arms":  (0.2, 0.4, 1.0, 1.0),
    "legs":  (1.0, 1.0, 0.0, 1.0),
}
_LM_PART = {**{i: "face" for i in range(11)},
            **{i: "torso" for i in [11, 12, 23, 24]},
            **{i: "arms" for i in [13, 14, 15, 16]},
            **{i: "legs" for i in [25, 26, 27, 28]}}


def _error_color(err: float, max_err: float = 20.0) -> np.ndarray:
    """Map reprojection error in pixels → RGBA array."""
    t = min(1.0, err / max(max_err, 1e-6))
    return np.array([t, 1.0 - t, 0.0, 1.0])


# -----------------------------------------------------------------------
# Triangulation worker
# -----------------------------------------------------------------------

class TriangulationWorker(QObject):
    finished = pyqtSignal(object)  # np.ndarray (N, 33, 3)
    error    = pyqtSignal(str)

    def __init__(self, triangulator, char):
        super().__init__()
        self.triangulator = triangulator
        self.char = char

    def run(self):
        try:
            self.triangulator.load_animation_sequence(self.char.frames)
            self.triangulator.poses_sequence = self.char.poses_2d
            result = self.triangulator.triangulate_sequence()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# 3D Viewer (pyqtgraph OpenGL)
# -----------------------------------------------------------------------

class SkeletonViewer3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _GL_AVAILABLE:
            self._view = gl.GLViewWidget()
            self._view.setBackgroundColor("k")
            self._view.setCameraPosition(distance=5, elevation=20, azimuth=45)

            # Grid
            grid = gl.GLGridItem()
            grid.scale(0.5, 0.5, 0.5)
            self._view.addItem(grid)

            self._scatter: gl.GLScatterPlotItem | None = None
            self._lines: list[gl.GLLinePlotItem] = []
            layout.addWidget(self._view)
        else:
            layout.addWidget(QLabel(
                "pyqtgraph not installed.\n"
                "Run: pip install pyqtgraph PyOpenGL\n"
                "to enable 3D visualisation."
            ))

    def set_skeleton(self, skeleton: np.ndarray, errors: np.ndarray | None = None):
        """
        skeleton : (33, 3)
        errors   : (33,) reprojection error per landmark, or None
        """
        if not _GL_AVAILABLE:
            return

        # Remove old items
        if self._scatter:
            self._view.removeItem(self._scatter)
            self._scatter = None
        for item in self._lines:
            self._view.removeItem(item)
        self._lines = []

        # Landmark colours
        max_err = errors.max() if errors is not None and errors.max() > 0 else 20.0
        colors = np.zeros((33, 4))
        for i in range(33):
            if errors is not None:
                colors[i] = _error_color(errors[i], max_err)
            else:
                part = _LM_PART.get(i, "torso")
                colors[i] = _PART_RGBA[part]

        valid = ~np.all(skeleton == 0, axis=1)
        if valid.any():
            self._scatter = gl.GLScatterPlotItem(
                pos=skeleton[valid].astype(np.float32),
                color=colors[valid].astype(np.float32),
                size=8,
                pxMode=True,
            )
            self._view.addItem(self._scatter)

        # Bones
        for s, e in POSE_CONNECTIONS:
            if np.all(skeleton[s] == 0) or np.all(skeleton[e] == 0):
                continue
            pts = np.array([skeleton[s], skeleton[e]], dtype=np.float32)
            part = _LM_PART.get(s, "torso")
            col = np.array([_PART_RGBA[part]] * 2, dtype=np.float32)
            line = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True)
            self._view.addItem(line)
            self._lines.append(line)


# -----------------------------------------------------------------------
# Error table
# -----------------------------------------------------------------------

class ErrorTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Landmark", "Error (px)", "Quality"])
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setMaximumWidth(320)

    def set_errors(self, errors: np.ndarray):
        self.setRowCount(33)
        for i, err in enumerate(errors):
            self.setItem(i, 0, QTableWidgetItem(str(i)))
            self.setItem(i, 1, QTableWidgetItem(f"{err:.2f}"))
            quality = "Good" if err < 5 else "Fair" if err < 15 else "Poor"
            qi = QTableWidgetItem(quality)
            if quality == "Poor":
                qi.setForeground(Qt.GlobalColor.red)
            elif quality == "Fair":
                qi.setForeground(Qt.GlobalColor.yellow)
            else:
                qi.setForeground(Qt.GlobalColor.green)
            self.setItem(i, 2, qi)


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class ReconstructionTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread: QThread | None = None
        from pipeline import PoseTriangulator
        self._triangulator = PoseTriangulator()
        self._skeleton_sequence: np.ndarray | None = None  # (N, 33, 3)
        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.frame_changed.connect(self._on_frame_changed)
        self.state.character_updated.connect(self._on_char_updated)

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Top controls
        ctrl = QHBoxLayout()
        root.addLayout(ctrl)

        self.btn_tri = QPushButton("Triangulate 3D Skeleton")
        self.btn_tri.setStyleSheet("font-weight: bold; padding: 5px 12px;")
        self.btn_tri.clicked.connect(self.run_triangulation)
        ctrl.addWidget(self.btn_tri)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress)

        ctrl.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider)
        ctrl.addWidget(self.frame_slider)

        self.frame_lbl = QLabel("0 / 0")
        ctrl.addWidget(self.frame_lbl)

        self.status_lbl = QLabel("")
        ctrl.addWidget(self.status_lbl)

        # Splitter: 3D view left, error table right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        self._viewer = SkeletonViewer3D()
        splitter.addWidget(self._viewer)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        stats_box = QGroupBox("Reprojection Error")
        stats_layout = QVBoxLayout(stats_box)
        self.mean_err_lbl = QLabel("Mean: —")
        self.max_err_lbl  = QLabel("Max:  —")
        stats_layout.addWidget(self.mean_err_lbl)
        stats_layout.addWidget(self.max_err_lbl)
        right_layout.addWidget(stats_box)

        self._error_table = ErrorTable()
        right_layout.addWidget(self._error_table, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([800, 320])

    # ------------------------------------------------------------------

    def run_triangulation(self):
        char = self.state.current_character
        if char is None:
            self.status_lbl.setText("No character loaded.")
            return
        if char.poses_2d is None:
            self.status_lbl.setText("Run pose detection first (Tab 2).")
            return

        self.btn_tri.setEnabled(False)
        self.progress.setVisible(True)
        self.status_lbl.setText("Triangulating…")

        self._worker = TriangulationWorker(self._triangulator, char)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_tri_done)
        self._worker.error.connect(self._on_tri_error)
        self._thread.start()

    def _on_tri_done(self, result: np.ndarray):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_tri.setEnabled(True)

        char = self.state.current_character
        if char:
            char.skeleton_3d = result
            self.state.character_updated.emit(self.state.selected_idx)

        self._skeleton_sequence = result
        n = result.shape[0]
        self.frame_slider.setRange(0, max(0, n - 1))
        self.frame_slider.setValue(0)
        self.status_lbl.setText(f"Done — {n} frames triangulated.")
        self._show_frame(0)

    def _on_tri_error(self, msg: str):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_tri.setEnabled(True)
        self.status_lbl.setText(f"Error: {msg}")

    # ------------------------------------------------------------------

    def _on_char_changed(self, idx: int):
        char = self.state.current_character
        if char and char.skeleton_3d is not None:
            self._skeleton_sequence = char.skeleton_3d
            n = char.skeleton_3d.shape[0]
            self.frame_slider.setRange(0, max(0, n - 1))
            self._show_frame(self.state.current_frame)
        else:
            self._skeleton_sequence = None

    def _on_char_updated(self, idx: int):
        if idx == self.state.selected_idx:
            self._on_char_changed(idx)

    def _on_frame_changed(self, frame: int):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame)
        self.frame_slider.blockSignals(False)
        self._show_frame(frame)

    def _on_slider(self, value: int):
        self.state.set_frame(value)

    def _show_frame(self, frame: int):
        if self._skeleton_sequence is None:
            return
        n = self._skeleton_sequence.shape[0]
        frame = max(0, min(frame, n - 1))
        self.frame_lbl.setText(f"{frame + 1} / {n}")

        skeleton = self._skeleton_sequence[frame]

        # Compute reprojection error if poses are available
        errors = None
        char = self.state.current_character
        if char and char.poses_2d is not None:
            errors = self._triangulator.get_backprojection_error(frame, skeleton)
            self.mean_err_lbl.setText(f"Mean: {errors.mean():.2f} px")
            self.max_err_lbl.setText(f"Max:  {errors.max():.2f} px")
            self._error_table.set_errors(errors)

        self._viewer.set_skeleton(skeleton, errors)
