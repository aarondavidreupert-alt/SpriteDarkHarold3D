"""
Tab 2 — Pose Detection & Editor
Runs MediaPipe detection in background, then displays all 6 views with
draggable landmarks overlaid in a confidence heatmap.
"""

import logging as _logging_mod

import numpy as np
import cv2

_logger = _logging_mod.getLogger(__name__)

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QFrame, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsItem,
    QSlider, QSplitter, QProgressBar, QCheckBox, QGroupBox,
)
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainter, QKeyEvent
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QPointF, QRectF

from gui.main_window import AppState
from pipeline.pose_triangulator import POSE_CONNECTIONS


# -----------------------------------------------------------------------
# Landmark confidence → colour (red→green)
# -----------------------------------------------------------------------

def _confidence_color(conf: float) -> QColor:
    """conf in [0,1] → QColor from red (0) through yellow to green (1)."""
    conf = max(0.0, min(1.0, conf))
    r = int((1 - conf) * 255)
    g = int(conf * 255)
    return QColor(r, g, 0)


# -----------------------------------------------------------------------
# Draggable landmark item
# -----------------------------------------------------------------------

class LandmarkItem(QGraphicsEllipseItem):
    RADIUS = 5

    def __init__(self, lm_idx: int, x: float, y: float, conf: float, view_idx: int):
        super().__init__(-self.RADIUS, -self.RADIUS, 2 * self.RADIUS, 2 * self.RADIUS)
        self.lm_idx = lm_idx
        self.view_idx = view_idx
        self.setPos(x, y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(2)
        self.set_confidence(conf)
        self.setToolTip(f"Landmark {lm_idx}  conf={conf:.2f}")
        self._on_move = None   # callback(lm_idx, view_idx, x, y)

    def set_confidence(self, conf: float):
        color = _confidence_color(conf)
        self.setBrush(QBrush(color))
        pen = QPen(color.darker(150))
        pen.setWidth(1)
        self.setPen(pen)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self._on_move:
            pos = self.scenePos()
            self._on_move(self.lm_idx, self.view_idx, pos.x(), pos.y())
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        from PyQt6.QtWidgets import QMenu
        menu = QMenu()
        del_act = menu.addAction("Delete Landmark")
        act = menu.exec(event.screenPos().toPoint())
        if act == del_act and self._on_move:
            self._on_move(self.lm_idx, self.view_idx, -1, -1)  # sentinel for delete
        event.accept()


# -----------------------------------------------------------------------
# Single-view canvas (QGraphicsView with sprite + landmarks)
# -----------------------------------------------------------------------

class ViewCanvas(QGraphicsView):
    landmark_moved = pyqtSignal(int, int, float, float)  # lm_idx, view_idx, x, y

    def __init__(self, view_idx: int, parent=None):
        super().__init__(parent)
        self.view_idx = view_idx
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMinimumSize(200, 200)
        self._lm_items: dict[int, LandmarkItem] = {}
        self._conn_items: list[QGraphicsLineItem] = []
        self._show_heatmap = False

        self._scene.mousePressEvent = self._scene_mouse_press

    def set_frame(self, img: np.ndarray, pose: np.ndarray | None, show_heatmap: bool = False):
        """
        img  : (H, W, 3) uint8
        pose : (33, 3) [x_px, y_px, z_conf] or None
        """
        self._scene.clear()
        self._lm_items = {}
        self._conn_items = []
        self._show_heatmap = show_heatmap

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        h, w, c = img.shape

        # Optional confidence heatmap overlay
        if show_heatmap and pose is not None:
            img = self._draw_heatmap(img.copy(), pose)

        qimg = QImage(img.data, w, h, w * c, QImage.Format.Format_RGB888)
        pix = self._scene.addPixmap(QPixmap.fromImage(qimg))
        pix.setZValue(0)

        if pose is not None:
            self._draw_pose(pose, w, h)

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _draw_pose(self, pose: np.ndarray, img_w: int, img_h: int):
        # Connections
        pen = QPen(QColor(255, 255, 255, 180))
        pen.setWidth(1)
        for s, e in POSE_CONNECTIONS:
            ps, pe = pose[s], pose[e]
            if np.all(ps == 0) or np.all(pe == 0):
                continue
            line = self._scene.addLine(ps[0], ps[1], pe[0], pe[1], pen)
            line.setZValue(1)
            self._conn_items.append(line)

        # Landmarks
        for i, lm in enumerate(pose):
            if np.all(lm == 0):
                continue
            conf = float(abs(lm[2])) if len(lm) > 2 else 1.0
            item = LandmarkItem(i, float(lm[0]), float(lm[1]), conf, self.view_idx)
            item._on_move = self._lm_moved
            self._scene.addItem(item)
            self._lm_items[i] = item

    def _draw_heatmap(self, img: np.ndarray, pose: np.ndarray) -> np.ndarray:
        hmap = np.zeros(img.shape[:2], dtype=np.float32)
        h, w = img.shape[:2]
        for lm in pose:
            if np.all(lm == 0):
                continue
            x, y = int(lm[0]), int(lm[1])
            conf = float(abs(lm[2])) if len(lm) > 2 else 1.0
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            cv2.circle(hmap, (x, y), 12, conf, -1)

        hmap_blur = cv2.GaussianBlur(hmap, (25, 25), 0)
        hmap_norm = np.clip(hmap_blur / (hmap_blur.max() + 1e-6), 0, 1)
        color_hmap = cv2.applyColorMap((hmap_norm * 255).astype(np.uint8), cv2.COLORMAP_RdYlGn)
        color_hmap = cv2.cvtColor(color_hmap, cv2.COLOR_BGR2RGB)
        blend = (0.6 * img + 0.4 * color_hmap).astype(np.uint8)
        return blend

    def _lm_moved(self, lm_idx: int, view_idx: int, x: float, y: float):
        self.landmark_moved.emit(lm_idx, view_idx, x, y)
        self._redraw_connections()

    def _redraw_connections(self):
        for item in self._conn_items:
            self._scene.removeItem(item)
        self._conn_items = []
        pen = QPen(QColor(255, 255, 255, 180))
        pen.setWidth(1)
        for s, e in POSE_CONNECTIONS:
            if s not in self._lm_items or e not in self._lm_items:
                continue
            ps = self._lm_items[s].scenePos()
            pe = self._lm_items[e].scenePos()
            line = self._scene.addLine(ps.x(), ps.y(), pe.x(), pe.y(), pen)
            line.setZValue(1)
            self._conn_items.append(line)

    def _scene_mouse_press(self, event):
        # Right-click on empty space → add landmark (not implemented fully)
        QGraphicsScene.mousePressEvent(self._scene, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# -----------------------------------------------------------------------
# Detection worker
# -----------------------------------------------------------------------

class DetectionWorker(QObject):
    """
    Runs MediaPipe Holistic (or Pose as fallback) on every view×frame,
    draws landmarks onto annotated frames, and stores results in CharacterData.
    """
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, char):
        super().__init__()
        self.char = char

    def run(self):
        try:
            import mediapipe as mp
        except ImportError:
            msg = "mediapipe not installed — run: pip install mediapipe"
            _logger.error(msg)
            self.error.emit(msg)
            return

        try:
            frames = (
                self.char.upscaled_frames
                if self.char.upscaled_frames is not None
                else self.char.frames
            )
            n_dirs   = int(frames.shape[0])
            n_frames = int(frames.shape[1])
            h        = int(frames.shape[2])
            w        = int(frames.shape[3])

            poses_out = np.zeros((n_frames, n_dirs, 33, 3))
            annotated = np.zeros((n_dirs, n_frames, h, w, 3), dtype=np.uint8)

            # Try Holistic first (face + body + hands); fall back to Pose only
            try:
                detector_ctx = mp.solutions.holistic.Holistic(
                    static_image_mode=True, min_detection_confidence=0.5
                )
                mode = "holistic"
            except AttributeError:
                detector_ctx = mp.solutions.pose.Pose(
                    static_image_mode=True, min_detection_confidence=0.5
                )
                mode = "pose"

            _logger.info(
                "Starting detection — %d views × %d frames (%d×%d px) "
                "using MediaPipe %s",
                n_dirs, n_frames, w, h, mode,
            )

            drawing       = mp.solutions.drawing_utils
            pose_cx       = mp.solutions.pose.POSE_CONNECTIONS
            lm_spec       = drawing.DrawingSpec(color=(0, 180, 0),   thickness=1, circle_radius=2)
            conn_spec     = drawing.DrawingSpec(color=(200, 200, 200), thickness=1)

            with detector_ctx as detector:
                for frame_idx in range(n_frames):
                    n_detected = 0
                    for dir_idx in range(n_dirs):
                        img_rgb = frames[dir_idx, frame_idx]
                        if img_rgb.dtype != np.uint8:
                            img_rgb = (
                                (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
                                if img_rgb.max() <= 1.0
                                else img_rgb.astype(np.uint8)
                            )

                        result   = detector.process(img_rgb)
                        pose_lms = result.pose_landmarks

                        # draw_landmarks expects BGR; convert, draw, convert back
                        img_bgr = img_rgb[..., ::-1].copy()
                        if pose_lms is not None:
                            drawing.draw_landmarks(
                                img_bgr, pose_lms, pose_cx, lm_spec, conn_spec
                            )
                            for lm_idx, lm in enumerate(pose_lms.landmark):
                                px = max(0, min(int(lm.x * w), w - 1))
                                py = max(0, min(int(lm.y * h), h - 1))
                                poses_out[frame_idx, dir_idx, lm_idx] = [
                                    px, py, lm.visibility
                                ]
                                _logger.debug(
                                    "F%d D%d LM%02d: (%d,%d) vis=%.2f",
                                    frame_idx, dir_idx, lm_idx, px, py, lm.visibility,
                                )
                            n_detected += 1
                        else:
                            _logger.warning(
                                "No pose detected — frame=%d dir=%d",
                                frame_idx, dir_idx,
                            )

                        annotated[dir_idx, frame_idx] = img_bgr[..., ::-1]

                    _logger.info(
                        "Frame %d/%d — pose in %d/%d views",
                        frame_idx + 1, n_frames, n_detected, n_dirs,
                    )
                    self.progress.emit(frame_idx + 1, n_frames)

            self.char.poses_2d        = poses_out
            self.char.annotated_frames = annotated

            z     = np.abs(poses_out[:, :, :, 2])
            z_max = z.max() + 1e-6
            self.char.confidences = (z / z_max).mean(axis=1)  # (N, 33)

            _logger.info("Pose detection complete.")
            self.finished.emit()

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            _logger.error("Detection failed: %s\n%s", exc, tb)
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class PoseEditorTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread: QThread | None = None
        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.frame_changed.connect(self._refresh_views)

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Controls row
        ctrl = QHBoxLayout()
        root.addLayout(ctrl)

        self.btn_detect = QPushButton("Run Pose Detection")
        self.btn_detect.setStyleSheet("font-weight: bold; padding: 5px 12px;")
        self.btn_detect.clicked.connect(self.run_detection)
        ctrl.addWidget(self.btn_detect)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress)

        ctrl.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider_moved)
        ctrl.addWidget(self.frame_slider)

        self.frame_lbl = QLabel("0 / 0")
        ctrl.addWidget(self.frame_lbl)

        self.heatmap_chk = QCheckBox("Confidence Heatmap")
        self.heatmap_chk.stateChanged.connect(lambda _: self._refresh_views(self.state.current_frame))
        ctrl.addWidget(self.heatmap_chk)

        self.btn_apply_all_frames = QPushButton("Apply Correction → All Frames")
        self.btn_apply_all_frames.clicked.connect(self._apply_to_all_frames)
        ctrl.addWidget(self.btn_apply_all_frames)

        self.btn_apply_all_chars = QPushButton("Apply → All Characters")
        self.btn_apply_all_chars.clicked.connect(self._apply_to_all_chars)
        ctrl.addWidget(self.btn_apply_all_chars)

        self.status_lbl = QLabel("")
        ctrl.addWidget(self.status_lbl)

        # 6-view grid
        views_scroll = QScrollArea()
        views_scroll.setWidgetResizable(True)
        views_container = QWidget()
        views_layout = QHBoxLayout(views_container)
        views_scroll.setWidget(views_container)
        root.addWidget(views_scroll, 1)

        self._canvases: list[ViewCanvas] = []
        for v in range(6):
            vbox = QVBoxLayout()
            lbl = QLabel(f"Dir {v+1}  ({v*60}°)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(lbl)
            canvas = ViewCanvas(v)
            canvas.landmark_moved.connect(self._on_landmark_moved)
            vbox.addWidget(canvas)
            views_layout.addLayout(vbox)
            self._canvases.append(canvas)

    # ------------------------------------------------------------------
    # Key navigation
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        char = self.state.current_character
        if char is None:
            return
        frame = self.state.current_frame
        if event.key() == Qt.Key.Key_Right:
            frame = min(frame + 1, char.n_frames - 1)
        elif event.key() == Qt.Key.Key_Left:
            frame = max(frame - 1, 0)
        else:
            super().keyPressEvent(event)
            return
        self.state.set_frame(frame)
        self.frame_slider.setValue(frame)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def run_detection(self):
        char = self.state.current_character
        if char is None:
            self.status_lbl.setText("No character loaded.")
            return

        self.btn_detect.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_lbl.setText("Running MediaPipe…")

        self._worker = DetectionWorker(char)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_detect_progress)
        self._worker.finished.connect(self._on_detect_done)
        self._worker.error.connect(self._on_detect_error)
        self._thread.start()

    def _on_detect_progress(self, current: int, total: int):
        if total > 0:
            self.progress.setValue(int(100 * current / total))

    def _on_detect_done(self):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_detect.setEnabled(True)
        self.status_lbl.setText("Detection complete.")
        char = self.state.current_character
        if char:
            idx = self.state.selected_idx
            self.state.character_updated.emit(idx)
            self._on_char_changed(idx)

    def _on_detect_error(self, msg: str):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_detect.setEnabled(True)
        self.status_lbl.setText(f"Error: {msg}")

    # ------------------------------------------------------------------
    # View refresh
    # ------------------------------------------------------------------

    def _on_char_changed(self, idx: int):
        char = self.state.current_character
        if char is None:
            return
        self.frame_slider.setRange(0, max(0, char.n_frames - 1))
        self.frame_slider.setValue(0)
        self._refresh_views(0)

    def _on_slider_moved(self, value: int):
        self.state.set_frame(value)

    def _refresh_views(self, frame_idx: int):
        char = self.state.current_character
        if char is None:
            return
        show_heatmap = self.heatmap_chk.isChecked()
        n = char.n_frames
        self.frame_lbl.setText(f"{frame_idx + 1} / {n}")

        ann = char.annotated_frames    # (6, N, H, W, 3) or None
        usc = char.upscaled_frames     # (6, N, H, W, 3) or None

        for v in range(min(6, char.frames.shape[0])):
            fi = min(frame_idx, char.frames.shape[1] - 1)
            if ann is not None and v < ann.shape[0] and fi < ann.shape[1]:
                img = ann[v, fi]
            elif usc is not None and v < usc.shape[0] and fi < usc.shape[1]:
                img = usc[v, fi]
            else:
                img = char.frames[v, fi]
            pose = None
            if char.poses_2d is not None:
                pose = char.poses_2d[frame_idx, v]
            self._canvases[v].set_frame(img, pose, show_heatmap)

    # ------------------------------------------------------------------
    # Landmark editing
    # ------------------------------------------------------------------

    def _on_landmark_moved(self, lm_idx: int, view_idx: int, x: float, y: float):
        char = self.state.current_character
        if char is None or char.poses_2d is None:
            return
        frame = self.state.current_frame
        if x < 0:
            # Delete
            char.poses_2d[frame, view_idx, lm_idx] = 0
        else:
            char.poses_2d[frame, view_idx, lm_idx, :2] = [x, y]
        self.state.character_updated.emit(self.state.selected_idx)

    def _apply_to_all_frames(self):
        """Copy current frame's pose corrections to all other frames."""
        char = self.state.current_character
        if char is None or char.poses_2d is None:
            return
        frame = self.state.current_frame
        ref = char.poses_2d[frame].copy()
        for f in range(char.n_frames):
            if f != frame:
                char.poses_2d[f] = ref.copy()
        self.status_lbl.setText("Correction applied to all frames.")

    def _apply_to_all_chars(self):
        """Copy current character's pose to all other characters of same category."""
        char = self.state.current_character
        if char is None or char.poses_2d is None:
            return
        for other in self.state.characters:
            if other is not char and other.category == char.category:
                if other.poses_2d is not None:
                    n = min(other.n_frames, char.n_frames)
                    other.poses_2d[:n] = char.poses_2d[:n]
        self.status_lbl.setText("Correction applied to all matching characters.")
