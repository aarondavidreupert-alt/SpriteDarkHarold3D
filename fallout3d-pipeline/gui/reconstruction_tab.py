"""
Tab 5 — 3D Reconstruction
Triangulates 2D poses into a 3D skeleton. Layout:
  LEFT:  rotatable pyqtgraph OpenGL skeleton viewer with Play/Pause animation.
  RIGHT: 2×3 grid of the 6 direction views showing char.frames with the
         back-projected 3D skeleton overlaid (blue=left, red=right, white=center).
"""

import sys
import subprocess
import logging
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QScrollArea, QSlider,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject

from gui.main_window import AppState
from pipeline.pose_triangulator import POSE_CONNECTIONS

_logger = logging.getLogger(__name__)

# Optional pyqtgraph OpenGL
try:
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False

_DIR_LABELS = ["NE", "E", "SE", "SW", "W", "NW"]

# Landmark side → RGB draw color (cv2 on RGB array: tuple = (R, G, B))
_LM_SIDE: dict[int, str] = {}
for _i in [11, 13, 15, 23, 25, 27, 29, 31, 1, 2, 3, 7, 9]:
    _LM_SIDE[_i] = "left"
for _i in [12, 14, 16, 24, 26, 28, 30, 32, 4, 5, 6, 8, 10]:
    _LM_SIDE[_i] = "right"

_SIDE_COLOR = {
    "left":   (0,   100, 255),   # blue
    "right":  (255,  60,   0),   # red
    "center": (255, 255, 255),   # white
}


# -----------------------------------------------------------------------
# 3D viewer colors
# -----------------------------------------------------------------------

_PART_RGBA = {
    "face":  (1.0, 0.0, 1.0, 1.0),
    "torso": (0.0, 1.0, 0.0, 1.0),
    "arms":  (0.2, 0.4, 1.0, 1.0),
    "legs":  (1.0, 1.0, 0.0, 1.0),
}
_LM_PART = {
    **{i: "face"  for i in range(11)},
    **{i: "torso" for i in [11, 12, 23, 24]},
    **{i: "arms"  for i in [13, 14, 15, 16]},
    **{i: "legs"  for i in [25, 26, 27, 28]},
}


def _error_color(err: float, max_err: float = 20.0) -> np.ndarray:
    t = min(1.0, err / max(max_err, 1e-6))
    return np.array([t, 1.0 - t, 0.0, 1.0])


def _arr_to_pixmap(rgb: np.ndarray, max_side: int = 240) -> QPixmap:
    """RGB (H,W,3) uint8 → QPixmap, downscaled if larger than max_side."""
    h, w = rgb.shape[:2]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    data = rgb.tobytes()
    qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
    pix  = QPixmap.fromImage(qimg)
    if max(h, w) > max_side:
        pix = pix.scaled(
            max_side, max_side,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pix


# -----------------------------------------------------------------------
# Workers
# -----------------------------------------------------------------------

class TriangulationWorker(QObject):
    finished = pyqtSignal(object)   # np.ndarray (N, 33, 3)
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


class _InstallWorker(QObject):
    finished = pyqtSignal(bool)

    def run(self):
        _logger.info("Installing pyqtgraph and PyOpenGL…")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "pyqtgraph", "PyOpenGL"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                stripped = line.rstrip()
                if stripped:
                    _logger.info("[pip] %s", stripped)
            proc.wait()
            if proc.returncode == 0:
                _logger.info(
                    "Install complete — restart the application to enable 3D view."
                )
                self.finished.emit(True)
            else:
                _logger.error("pip exited with code %d.", proc.returncode)
                self.finished.emit(False)
        except Exception as exc:
            _logger.error("Install failed: %s", exc)
            self.finished.emit(False)


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

            grid = gl.GLGridItem()
            grid.scale(0.5, 0.5, 0.5)
            self._view.addItem(grid)

            self._scatter: gl.GLScatterPlotItem | None = None
            self._lines: list[gl.GLLinePlotItem] = []
            layout.addWidget(self._view)
        else:
            layout.addWidget(QLabel(
                "pyqtgraph / PyOpenGL not installed — 3D viewer disabled."
            ))
            self._btn_install = QPushButton("Install pyqtgraph + PyOpenGL")
            self._btn_install.clicked.connect(self._start_install)
            layout.addWidget(self._btn_install)
            self._install_lbl = QLabel("")
            layout.addWidget(self._install_lbl)
            layout.addStretch()
            self._install_thread: QThread | None = None

    def _start_install(self):
        self._btn_install.setEnabled(False)
        self._install_lbl.setText("Installing… check Console Log for progress.")
        self._install_worker = _InstallWorker()
        self._install_thread = QThread(self)
        self._install_worker.moveToThread(self._install_thread)
        self._install_thread.started.connect(self._install_worker.run)
        self._install_worker.finished.connect(self._on_install_done)
        self._install_thread.start()

    def _on_install_done(self, success: bool):
        self._install_thread.quit()
        if success:
            self._install_lbl.setText(
                "Restart the application to enable 3D view."
            )
        else:
            self._btn_install.setEnabled(True)
            self._install_lbl.setText(
                "Install failed — see Console Log for details."
            )

    def set_skeleton(self, skeleton: np.ndarray, errors: np.ndarray | None = None):
        """skeleton: (33, 3)  errors: (33,) or None"""
        if not _GL_AVAILABLE:
            return

        if self._scatter:
            self._view.removeItem(self._scatter)
            self._scatter = None
        for item in self._lines:
            self._view.removeItem(item)
        self._lines = []

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

        for s, e in POSE_CONNECTIONS:
            if np.all(skeleton[s] == 0) or np.all(skeleton[e] == 0):
                continue
            pts = np.array([skeleton[s], skeleton[e]], dtype=np.float32)
            part = _LM_PART.get(s, "torso")
            col  = np.array([_PART_RGBA[part]] * 2, dtype=np.float32)
            line = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True)
            self._view.addItem(line)
            self._lines.append(line)


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class ReconstructionTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread: QThread | None = None
        self._skeleton_sequence: np.ndarray | None = None   # (N, 33, 3)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)

        from pipeline import PoseTriangulator
        self._triangulator = PoseTriangulator()

        if not _GL_AVAILABLE:
            _logger.warning(
                "pyqtgraph/PyOpenGL not installed — 3D viewer disabled. "
                "Click 'Install' in the Reconstruction tab or run: "
                "pip install pyqtgraph PyOpenGL"
            )

        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.frame_changed.connect(self._on_frame_changed)
        self.state.character_updated.connect(self._on_char_updated)

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Controls row
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

        self._btn_play = QPushButton("▶ Play")
        self._btn_play.setCheckable(True)
        self._btn_play.setFixedWidth(70)
        self._btn_play.toggled.connect(self._on_play_toggled)
        ctrl.addWidget(self._btn_play)

        ctrl.addWidget(QLabel("FPS:"))
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 30)
        self._speed_slider.setValue(8)
        self._speed_slider.setFixedWidth(80)
        self._speed_slider.valueChanged.connect(self._on_fps_changed)
        ctrl.addWidget(self._speed_slider)

        self._fps_lbl = QLabel("8")
        self._fps_lbl.setFixedWidth(24)
        ctrl.addWidget(self._fps_lbl)

        self.mean_err_lbl = QLabel("Mean: —")
        ctrl.addWidget(self.mean_err_lbl)

        self.max_err_lbl = QLabel("Max: —")
        ctrl.addWidget(self.max_err_lbl)

        self.status_lbl = QLabel("")
        ctrl.addWidget(self.status_lbl)
        ctrl.addStretch()

        # Views row — same structure as Pose Editor:
        # Dir 1–3  |  [3D Wireframe]  |  Dir 4–6
        views_scroll = QScrollArea()
        views_scroll.setWidgetResizable(True)
        views_container = QWidget()
        views_layout = QHBoxLayout(views_container)
        views_scroll.setWidget(views_container)
        root.addWidget(views_scroll, 1)

        self._bp_labels: list[QLabel] = []

        for v in range(3):
            vbox = QVBoxLayout()
            dir_lbl = QLabel(f"Dir {v + 1} — {_DIR_LABELS[v]}")
            dir_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(dir_lbl)
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setMinimumSize(150, 150)
            img_lbl.setStyleSheet("background:#111;")
            vbox.addWidget(img_lbl, 1)
            views_layout.addLayout(vbox, 1)
            self._bp_labels.append(img_lbl)

        center_vbox = QVBoxLayout()
        center_title = QLabel("3D Skeleton")
        center_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_vbox.addWidget(center_title)
        self._viewer = SkeletonViewer3D()
        self._viewer.setMinimumSize(400, 300)
        center_vbox.addWidget(self._viewer, 1)
        views_layout.addLayout(center_vbox, 2)

        for v in range(3, 6):
            vbox = QVBoxLayout()
            dir_lbl = QLabel(f"Dir {v + 1} — {_DIR_LABELS[v]}")
            dir_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(dir_lbl)
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setMinimumSize(150, 150)
            img_lbl.setStyleSheet("background:#111;")
            vbox.addWidget(img_lbl, 1)
            views_layout.addLayout(vbox, 1)
            self._bp_labels.append(img_lbl)

    # ── Playback ──────────────────────────────────────────────────────

    def _on_play_toggled(self, playing: bool):
        if playing:
            self._btn_play.setText("⏸ Pause")
            self._play_timer.start(max(1, 1000 // self._speed_slider.value()))
        else:
            self._btn_play.setText("▶ Play")
            self._play_timer.stop()

    def _on_fps_changed(self, fps: int):
        self._fps_lbl.setText(str(fps))
        if self._play_timer.isActive():
            self._play_timer.setInterval(max(1, 1000 // fps))

    def _play_tick(self):
        char = self.state.current_character
        if char is None or char.n_frames == 0:
            self._btn_play.setChecked(False)
            return
        n = (
            self._skeleton_sequence.shape[0]
            if self._skeleton_sequence is not None
            else char.n_frames
        )
        next_frame = (self.state.current_frame + 1) % n
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(next_frame)
        self.frame_slider.blockSignals(False)
        self.state.set_frame(next_frame)

    # ── Triangulation ─────────────────────────────────────────────────

    def run_triangulation(self):
        char = self.state.current_character
        if char is None:
            self.status_lbl.setText("No character loaded.")
            return
        if char.poses_2d is None:
            self.status_lbl.setText("Run pose detection first (Tab 3).")
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

    # ── Selection / frame updates ─────────────────────────────────────

    def _on_char_changed(self, idx: int):
        char = self.state.current_character
        if char is None:
            self._skeleton_sequence = None
            return
        n_frames = char.n_frames
        self.frame_slider.setRange(0, max(0, n_frames - 1))
        self._skeleton_sequence = char.skeleton_3d   # may be None
        self._show_frame(min(self.state.current_frame, max(0, n_frames - 1)))

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

    # ── Frame display ─────────────────────────────────────────────────

    def _show_frame(self, frame: int):
        char = self.state.current_character
        if char is None:
            return

        self.frame_lbl.setText(f"{frame + 1} / {char.n_frames}")

        skeleton = None
        if self._skeleton_sequence is not None:
            fi_skel = max(0, min(frame, self._skeleton_sequence.shape[0] - 1))
            skeleton = self._skeleton_sequence[fi_skel]

            errors = None
            if char.poses_2d is not None:
                errors = self._triangulator.get_backprojection_error(fi_skel, skeleton)
                self.mean_err_lbl.setText(f"Mean: {errors.mean():.2f} px")
                self.max_err_lbl.setText(f"Max:  {errors.max():.2f} px")

            self._viewer.set_skeleton(skeleton, errors)

        self._refresh_bp_grid(frame, skeleton)

    def _refresh_bp_grid(self, frame: int, skeleton: np.ndarray | None):
        char = self.state.current_character
        if char is None:
            return

        cam_w, cam_h = self._triangulator.camera_setup.image_size
        bp_views = None
        if skeleton is not None:
            bp_views = self._triangulator.camera_setup.back_project_points(skeleton)

        n_dirs = char.frames.shape[0]
        fi     = max(0, min(frame, char.frames.shape[1] - 1))

        for v in range(min(6, n_dirs)):
            img = char.frames[v, fi]
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            else:
                img = img.copy()

            if bp_views is not None:
                img = self._render_backprojection(img, bp_views[v], cam_w, cam_h)

            self._bp_labels[v].setPixmap(_arr_to_pixmap(img))

    @staticmethod
    def _render_backprojection(
        img: np.ndarray,
        bp_pts: np.ndarray,
        cam_w: int,
        cam_h: int,
    ) -> np.ndarray:
        """Draw back-projected skeleton onto an RGB uint8 image."""
        out = img.copy()
        fh, fw = out.shape[:2]
        sx, sy = fw / cam_w, fh / cam_h

        for s, e in POSE_CONNECTIONS:
            ps, pe = bp_pts[s], bp_pts[e]
            if np.all(ps == 0) or np.all(pe == 0):
                continue
            x1, y1 = int(ps[0] * sx), int(ps[1] * sy)
            x2, y2 = int(pe[0] * sx), int(pe[1] * sy)
            cv2.line(out, (x1, y1), (x2, y2), (180, 180, 180), 1)

        for i, pt in enumerate(bp_pts):
            if np.all(pt == 0):
                continue
            x = max(0, min(int(pt[0] * sx), fw - 1))
            y = max(0, min(int(pt[1] * sy), fh - 1))
            color = _SIDE_COLOR.get(_LM_SIDE.get(i, "center"), (255, 255, 255))
            cv2.circle(out, (x, y), 3, color, -1)

        return out
