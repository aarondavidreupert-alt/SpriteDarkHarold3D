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
    QComboBox, QDoubleSpinBox,
)
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainter, QKeyEvent
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject, QPointF, QRectF

from gui.main_window import AppState
from pipeline.pose_triangulator import POSE_CONNECTIONS

_DIR_LABELS = ["NE", "E", "SE", "SW", "W", "NW"]

_LM_SIDE: dict[int, str] = {}
for _i in [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 1, 2, 3, 7, 9]:
    _LM_SIDE[_i] = "left"
for _i in [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 4, 5, 6, 8, 10]:
    _LM_SIDE[_i] = "right"
for _i in [0]:
    _LM_SIDE[_i] = "center"


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
    _SIDE_BASE = {
        "left":   QColor(0, 100, 255),
        "right":  QColor(255, 60, 60),
        "center": QColor(255, 255, 255),
    }

    def __init__(self, lm_idx: int, x: float, y: float, conf: float, view_idx: int,
                 side: str = "center"):
        super().__init__(-self.RADIUS, -self.RADIUS, 2 * self.RADIUS, 2 * self.RADIUS)
        self.lm_idx   = lm_idx
        self.view_idx = view_idx
        self.side     = side  # must be set before set_confidence()
        self.setPos(x, y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(2)
        self.set_confidence(conf)
        self.setToolTip(f"Landmark {lm_idx}  conf={conf:.2f}")
        self._on_move = None   # callback(lm_idx, view_idx, x, y)

    def set_confidence(self, conf: float):
        conf  = max(0.0, min(1.0, conf))
        base  = self._SIDE_BASE.get(getattr(self, "side", "center"), QColor(255, 255, 255))
        alpha = int(40 + conf * 215)   # 40 @ conf=0 → 255 @ conf=1
        color = QColor(base.red(), base.green(), base.blue(), alpha)
        self.setBrush(QBrush(color))
        pen = QPen(QColor(base.red(), base.green(), base.blue(), min(255, alpha + 40)))
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
            side = _LM_SIDE.get(i, "center")
            item = LandmarkItem(i, float(lm[0]), float(lm[1]), conf, self.view_idx, side=side)
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
    """Runs MediaPipe pose detection on every view×frame using the Tasks API."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, char, bidirectional: bool = False,
                 weight_mode: str = "visibility", threshold: float = 0.3):
        super().__init__()
        self.char          = char
        self.bidirectional = bidirectional
        self.weight_mode   = weight_mode
        self.threshold     = threshold

    def run(self):
        try:
            import mediapipe as mp
        except ImportError:
            msg = "mediapipe not installed — run: pip install mediapipe"
            _logger.error(msg)
            self.error.emit(msg)
            return

        try:
            import os, urllib.request
            from pipeline.pose_triangulator import (
                _MODEL_PATH, _MODEL_URL, _MODEL_DIR,
                POSE_CONNECTIONS as _PC, _normalized_to_pixel, _landmark_conf,
            )
            if not os.path.exists(_MODEL_PATH):
                os.makedirs(_MODEL_DIR, exist_ok=True)
                _logger.info("Downloading pose model…")
                urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

            from mediapipe.tasks import python as _mptasks
            from mediapipe.tasks.python import vision as _mpvision

            options = _mpvision.PoseLandmarkerOptions(
                base_options=_mptasks.BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=_mpvision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detector = _mpvision.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            import traceback
            _logger.error("Failed to create detector: %s\n%s", exc, traceback.format_exc())
            self.error.emit(str(exc))
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

            total = n_frames * (2 if self.bidirectional else 1)

            _logger.info(
                "Starting %s detection — %d views × %d frames (%d×%d px)",
                "bidirectional" if self.bidirectional else "forward",
                n_dirs, n_frames, w, h,
            )

            def _run_pass(pass_frames, prog_offset):
                poses = np.zeros((n_frames, n_dirs, 33, 3))
                ann   = np.zeros((n_dirs, n_frames, h, w, 3), dtype=np.uint8)
                for frame_idx in range(n_frames):
                    n_detected = 0
                    for dir_idx in range(n_dirs):
                        img_rgb = pass_frames[dir_idx, frame_idx]
                        if img_rgb.dtype != np.uint8:
                            img_rgb = (
                                (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
                                if img_rgb.max() <= 1.0
                                else img_rgb.astype(np.uint8)
                            )
                        if len(img_rgb.shape) == 2:
                            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
                        elif img_rgb.shape[2] == 4:
                            img_rgb = img_rgb[:, :, :3]

                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(img_rgb),
                        )
                        result = detector.detect(mp_image)
                        img_ann = img_rgb.copy()

                        if result.pose_landmarks:
                            lm_arr = np.zeros((33, 3), dtype=float)
                            for lm_i, lm in enumerate(result.pose_landmarks[0]):
                                conf = _landmark_conf(lm, self.weight_mode)
                                if conf >= self.threshold:
                                    px, py = _normalized_to_pixel(lm.x, lm.y, w, h)
                                    lm_arr[lm_i] = [px, py, conf]
                            for s, e in _PC:
                                if not (np.all(lm_arr[s] == 0) or np.all(lm_arr[e] == 0)):
                                    cv2.line(
                                        img_ann,
                                        (int(lm_arr[s, 0]), int(lm_arr[s, 1])),
                                        (int(lm_arr[e, 0]), int(lm_arr[e, 1])),
                                        (200, 200, 200), 1,
                                    )
                            for lm_i, pt in enumerate(lm_arr):
                                if not np.all(pt == 0):
                                    cv2.circle(img_ann, (int(pt[0]), int(pt[1])), 3, (0, 180, 0), -1)
                            poses[frame_idx, dir_idx] = lm_arr
                            n_detected += 1
                        else:
                            _logger.warning("No pose — frame=%d dir=%d", frame_idx, dir_idx)

                        ann[dir_idx, frame_idx] = img_ann

                    _logger.info("Frame %d/%d — pose in %d/%d views",
                                 frame_idx + 1, n_frames, n_detected, n_dirs)
                    self.progress.emit(prog_offset + frame_idx + 1, total)
                return poses, ann

            if self.bidirectional:
                fwd, fwd_ann = _run_pass(frames, 0)
                bwd_rev, _   = _run_pass(frames[:, ::-1], n_frames)
                bwd          = bwd_rev[::-1]
                fwd_zero     = np.all(fwd == 0, axis=-1, keepdims=True)
                bwd_zero     = np.all(bwd == 0, axis=-1, keepdims=True)
                poses_out    = np.where(
                    ~fwd_zero & ~bwd_zero, (fwd + bwd) / 2,
                    np.where(~fwd_zero, fwd, bwd),
                )
                annotated = fwd_ann
            else:
                poses_out, annotated = _run_pass(frames, 0)

            detector.close()

            self.char.poses_2d         = poses_out
            self.char.annotated_frames = annotated

            self.char.confidences = poses_out[:, :, :, 2].mean(axis=1)  # (N, 33)

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
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
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

        self.btn_bidir = QPushButton("Run Bidirectional Detection")
        self.btn_bidir.setStyleSheet("padding: 5px 12px;")
        self.btn_bidir.clicked.connect(self.run_bidirectional_detection)
        ctrl.addWidget(self.btn_bidir)

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

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setFixedWidth(70)
        self.btn_play.toggled.connect(self._on_play_toggled)
        ctrl.addWidget(self.btn_play)

        ctrl.addWidget(QLabel("FPS:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 30)
        self.speed_slider.setValue(8)
        self.speed_slider.setFixedWidth(80)
        self.speed_slider.valueChanged.connect(self._on_fps_changed)
        ctrl.addWidget(self.speed_slider)

        self.fps_lbl = QLabel("8")
        self.fps_lbl.setFixedWidth(24)
        ctrl.addWidget(self.fps_lbl)

        self.heatmap_chk = QCheckBox("Confidence Heatmap")
        self.heatmap_chk.stateChanged.connect(lambda _: self._refresh_views(self.state.current_frame))
        ctrl.addWidget(self.heatmap_chk)

        ctrl.addWidget(QLabel("Weight by:"))
        self._weight_combo = QComboBox()
        self._weight_combo.addItem("visibility",   "visibility")
        self._weight_combo.addItem("presence",     "presence")
        self._weight_combo.addItem("vis × pres",   "vis×pres")
        self._weight_combo.addItem("z depth",      "z")
        self._weight_combo.setFixedWidth(100)
        ctrl.addWidget(self._weight_combo)

        ctrl.addWidget(QLabel("Threshold:"))
        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(0.0, 1.0)
        self._thresh_spin.setSingleStep(0.05)
        self._thresh_spin.setValue(0.3)
        self._thresh_spin.setDecimals(2)
        self._thresh_spin.setFixedWidth(64)
        ctrl.addWidget(self._thresh_spin)

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
            lbl = QLabel(f"Dir {v+1} — {_DIR_LABELS[v]}")
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

        self._play_timer.stop()
        self.btn_play.setChecked(False)
        self.btn_detect.setEnabled(False)
        self.btn_bidir.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_lbl.setText("Running MediaPipe…")

        self._worker = DetectionWorker(
            char, bidirectional=False,
            weight_mode=self._weight_combo.currentData(),
            threshold=self._thresh_spin.value(),
        )
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
        self.btn_bidir.setEnabled(True)
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
        self.btn_bidir.setEnabled(True)
        self.status_lbl.setText(f"Error: {msg}")

    def run_bidirectional_detection(self):
        char = self.state.current_character
        if char is None:
            self.status_lbl.setText("No character loaded.")
            return

        self._play_timer.stop()
        self.btn_play.setChecked(False)
        self.btn_detect.setEnabled(False)
        self.btn_bidir.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_lbl.setText("Running bidirectional detection…")

        self._worker = DetectionWorker(
            char, bidirectional=True,
            weight_mode=self._weight_combo.currentData(),
            threshold=self._thresh_spin.value(),
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_detect_progress)
        self._worker.finished.connect(self._on_detect_done)
        self._worker.error.connect(self._on_detect_error)
        self._thread.start()

    def _on_play_toggled(self, playing: bool):
        if playing:
            self.btn_play.setText("⏸ Pause")
            self._play_timer.start(max(1, 1000 // self.speed_slider.value()))
        else:
            self.btn_play.setText("▶ Play")
            self._play_timer.stop()

    def _on_fps_changed(self, fps: int):
        self.fps_lbl.setText(str(fps))
        if self._play_timer.isActive():
            self._play_timer.setInterval(max(1, 1000 // fps))

    def _play_tick(self):
        char = self.state.current_character
        if char is None or char.n_frames == 0:
            self.btn_play.setChecked(False)
            return
        next_frame = (self.state.current_frame + 1) % char.n_frames
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(next_frame)
        self.frame_slider.blockSignals(False)
        self.state.set_frame(next_frame)

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
