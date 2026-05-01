"""
Tab 4 — Manual Pose Editor
6-direction overview (same layout as Pose Detector) with click-to-edit.
Drag landmarks, flip L/R, save back to char.poses_2d.
"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QGroupBox, QSplitter, QSlider, QGridLayout,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent

from gui.main_window import AppState
from gui.pose_editor_tab import ViewCanvas, _DIR_LABELS

_FLIP_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28),
    (29, 30), (31, 32),
]

# (button label, list-of-index-pairs) for the individual swap buttons
_SWAP_GROUPS = [
    ("Eyes L/R",  [(1, 4), (2, 5), (3, 6)]),
    ("Mouth L/R", [(9, 10)]),
    ("Ears L/R",  [(7, 8)]),
    ("Shoulders", [(11, 12)]),
    ("Elbows",    [(13, 14)]),
    ("Wrists",    [(15, 16)]),
    ("Fingers",   [(17, 18), (19, 20), (21, 22)]),
    ("Hips",      [(23, 24)]),
    ("Knees",     [(25, 26)]),
    ("Ankles",    [(27, 28)]),
    ("Heels",     [(29, 30)]),
    ("Feet",      [(31, 32)]),
]


class _ThumbCanvas(ViewCanvas):
    """ViewCanvas that emits thumb_clicked on left press."""
    thumb_clicked = pyqtSignal(int)  # view_idx

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.thumb_clicked.emit(self.view_idx)


class PoseManualEditorTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._selected_dir = 0
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.frame_changed.connect(self._refresh_views)
        self.state.character_updated.connect(self._on_char_data_updated)

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter)

        # Upper: controls + 6-view overview
        upper = QWidget()
        upper_lay = QVBoxLayout(upper)
        upper_lay.setContentsMargins(0, 0, 0, 0)

        ctrl = QHBoxLayout()
        upper_lay.addLayout(ctrl)

        ctrl.addWidget(QLabel("Frame:"))
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.valueChanged.connect(self._on_slider_moved)
        ctrl.addWidget(self._frame_slider)

        self._frame_lbl = QLabel("0 / 0")
        ctrl.addWidget(self._frame_lbl)

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

        ctrl.addWidget(QLabel("  Click a direction below to edit it."))
        ctrl.addStretch()

        views_scroll = QScrollArea()
        views_scroll.setWidgetResizable(True)
        views_container = QWidget()
        views_layout = QHBoxLayout(views_container)
        views_scroll.setWidget(views_container)
        upper_lay.addWidget(views_scroll, 1)

        self._thumbs: list[_ThumbCanvas] = []
        for v in range(6):
            vbox = QVBoxLayout()
            lbl = QLabel(f"Dir {v+1} — {_DIR_LABELS[v]}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(lbl)
            thumb = _ThumbCanvas(v)
            thumb.setMinimumSize(120, 120)
            thumb.thumb_clicked.connect(self._on_thumb_clicked)
            vbox.addWidget(thumb)
            views_layout.addLayout(vbox)
            self._thumbs.append(thumb)

        splitter.addWidget(upper)

        # Lower: editor panel
        self._editor_box = QGroupBox(
            "Editor — Direction 1 (NE)  [click a direction above]"
        )
        editor_lay = QVBoxLayout(self._editor_box)

        self._editor_canvas = ViewCanvas(0)
        self._editor_canvas.setMinimumSize(300, 300)
        self._editor_canvas.landmark_moved.connect(self._on_editor_lm_moved)
        editor_lay.addWidget(self._editor_canvas, 1)

        # Swap pairs section — 4 columns × 3 rows
        swap_box = QGroupBox("Swap Pairs")
        swap_grid = QGridLayout(swap_box)
        swap_grid.setSpacing(4)
        for idx, (label, pairs) in enumerate(_SWAP_GROUPS):
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, p=pairs: self._swap_pairs(p))
            swap_grid.addWidget(btn, idx // 4, idx % 4)
        editor_lay.addWidget(swap_box)

        btn_row = QHBoxLayout()
        self._btn_flip = QPushButton("Flip All L/R")
        self._btn_flip.clicked.connect(self._flip_pose)
        btn_row.addWidget(self._btn_flip)

        self._btn_save = QPushButton("Save Poses")
        self._btn_save.setStyleSheet("font-weight: bold; padding: 4px 12px;")
        self._btn_save.clicked.connect(self._save_poses)
        btn_row.addWidget(self._btn_save)

        self._status_lbl = QLabel("")
        btn_row.addWidget(self._status_lbl)
        btn_row.addStretch()
        editor_lay.addLayout(btn_row)

        splitter.addWidget(self._editor_box)
        splitter.setSizes([300, 400])

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
        next_frame = (self.state.current_frame + 1) % char.n_frames
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(next_frame)
        self._frame_slider.blockSignals(False)
        self.state.set_frame(next_frame)

    def _on_slider_moved(self, value: int):
        self.state.set_frame(value)

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
        self._frame_slider.setValue(frame)

    # ── Direction selection ────────────────────────────────────────────

    def _on_thumb_clicked(self, view_idx: int):
        self._selected_dir = view_idx
        self._editor_box.setTitle(
            f"Editor — Direction {view_idx+1} ({_DIR_LABELS[view_idx]})"
        )
        self._refresh_editor()

    # ── View refresh ──────────────────────────────────────────────────

    def _on_char_changed(self, _idx: int = -1):
        char = self.state.current_character
        if char is None:
            return
        self._frame_slider.setRange(0, max(0, char.n_frames - 1))
        self._frame_slider.setValue(0)
        self._refresh_views(0)

    def _on_char_data_updated(self, _idx: int = -1):
        self._refresh_views(self.state.current_frame)

    def _refresh_views(self, frame_idx: int):
        char = self.state.current_character
        if char is None:
            return
        self._frame_lbl.setText(f"{frame_idx + 1} / {char.n_frames}")
        self._refresh_thumbs(frame_idx)
        self._refresh_editor()

    def _refresh_thumbs(self, frame_idx: int):
        char = self.state.current_character
        if char is None:
            return
        ann = char.annotated_frames
        usc = char.upscaled_frames
        for v in range(min(6, char.frames.shape[0])):
            fi = min(frame_idx, char.frames.shape[1] - 1)
            if ann is not None and v < ann.shape[0] and fi < ann.shape[1]:
                img = ann[v, fi]
            elif usc is not None and v < usc.shape[0] and fi < usc.shape[1]:
                img = usc[v, fi]
            else:
                img = char.frames[v, fi]
            pose = char.poses_2d[frame_idx, v] if char.poses_2d is not None else None
            self._thumbs[v].set_frame(img, pose)

    def _refresh_editor(self):
        char = self.state.current_character
        if char is None:
            return
        frame_idx = self.state.current_frame
        d  = self._selected_dir
        fi = min(frame_idx, char.frames.shape[1] - 1)
        ann = char.annotated_frames
        usc = char.upscaled_frames
        if ann is not None and d < ann.shape[0] and fi < ann.shape[1]:
            img = ann[d, fi]
        elif usc is not None and d < usc.shape[0] and fi < usc.shape[1]:
            img = usc[d, fi]
        else:
            img = char.frames[d, fi]
        pose = char.poses_2d[frame_idx, d] if char.poses_2d is not None else None
        self._editor_canvas.set_frame(img, pose)

    # ── Editing ────────────────────────────────────────────────────────

    def _on_editor_lm_moved(self, lm_idx: int, _view_idx: int, x: float, y: float):
        char = self.state.current_character
        if char is None or char.poses_2d is None:
            return
        frame = self.state.current_frame
        if x < 0:
            char.poses_2d[frame, self._selected_dir, lm_idx] = 0
        else:
            char.poses_2d[frame, self._selected_dir, lm_idx, :2] = [x, y]
        self._refresh_thumbs(frame)

    def _swap_pairs(self, pairs: list):
        char = self.state.current_character
        if char is None or char.poses_2d is None:
            self._status_lbl.setText("No poses.")
            return
        frame = self.state.current_frame
        pose = char.poses_2d[frame, self._selected_dir]
        for a, b in pairs:
            pose[a], pose[b] = pose[b].copy(), pose[a].copy()
        self._refresh_editor()
        self._refresh_thumbs(frame)
        self._status_lbl.setText(f"Swapped {', '.join(f'({a},{b})' for a, b in pairs)}.")

    def _flip_pose(self):
        char = self.state.current_character
        if char is None or char.poses_2d is None:
            self._status_lbl.setText("No poses to flip.")
            return
        frame = self.state.current_frame
        pose = char.poses_2d[frame, self._selected_dir]
        for a, b in _FLIP_PAIRS:
            pose[a], pose[b] = pose[b].copy(), pose[a].copy()
        self._refresh_editor()
        self._refresh_thumbs(frame)
        self._status_lbl.setText("Flipped L/R.")

    def _save_poses(self):
        char = self.state.current_character
        if char is None or char.poses_2d is None:
            self._status_lbl.setText("No poses to save.")
            return
        self.state.character_updated.emit(self.state.selected_idx)
        self._status_lbl.setText("Poses saved.")
