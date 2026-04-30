"""
Tab 1b — FRM Registration Viewer
Lets the user inspect spatially-registered FRM animation frames before
upscaling and pose detection.  Shows all 6 directions simultaneously in a
2×3 grid with playback controls and a per-direction L/R flip tool.
"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QGridLayout, QFrame, QSplitter,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from gui.main_window import AppState

_DIR_NAMES = ["NE", "E", "SE", "SW", "W", "NW"]


def _to_pixmap(img: np.ndarray, max_side: int = 220) -> QPixmap:
    """RGB (H, W, 3) uint8 ndarray → scaled QPixmap."""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())   # .copy() owns the pixel data
    if max(h, w) > max_side:
        pix = pix.scaled(
            max_side, max_side,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pix


class _DirCell(QFrame):
    """One cell in the 2×3 grid.  Shows current frame for one direction."""
    selected_signal = pyqtSignal(int)   # dir_idx

    def __init__(self, dir_idx: int, parent=None):
        super().__init__(parent)
        self.dir_idx = dir_idx
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(2)
        self._set_border(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)

        name_lbl = QLabel(f"Dir {dir_idx + 1}  {_DIR_NAMES[dir_idx]}")
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet("font-size: 10px; color: #aaa; border: none;")
        lay.addWidget(name_lbl)

        self._img = QLabel()
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setMinimumSize(140, 140)
        self._img.setStyleSheet("background: #0d0d0d; border: none;")
        lay.addWidget(self._img, 1)

    def _set_border(self, selected: bool):
        color = "#ffdd00" if selected else "#444"
        self.setStyleSheet(f"border: 2px solid {color};")

    def set_selected(self, sel: bool):
        self._set_border(sel)

    def set_frame(self, img: np.ndarray | None):
        if img is None:
            self._img.clear()
        else:
            self._img.setPixmap(_to_pixmap(img))

    def mousePressEvent(self, event):
        self.selected_signal.emit(self.dir_idx)
        super().mousePressEvent(event)


class FrmViewerTab(QWidget):
    """
    Displays all 6 FRM directions simultaneously, with playback and
    per-direction L/R flip for manual registration correction.
    """

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._selected_dir = 0
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self._build_ui()
        self._select_dir(0)

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.frame_changed.connect(self._on_frame_changed)
        self.state.character_updated.connect(self._on_char_updated)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── LEFT PANEL ────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(220)
        left.setMaximumWidth(300)
        ll = QVBoxLayout(left)
        ll.setSpacing(6)
        ll.setContentsMargins(0, 0, 4, 0)

        # Direction selector
        dir_box = QGroupBox("Direction")
        dir_grid = QGridLayout(dir_box)
        dir_grid.setSpacing(3)
        self._dir_btns: list[QPushButton] = []
        for i in range(6):
            btn = QPushButton(f"Dir {i+1} {_DIR_NAMES[i]}")
            btn.setCheckable(True)
            btn.setStyleSheet("padding: 3px 6px;")
            btn.clicked.connect(lambda _checked, idx=i: self._select_dir(idx))
            dir_grid.addWidget(btn, i // 2, i % 2)
            self._dir_btns.append(btn)
        ll.addWidget(dir_box)

        # Playback
        pb_box = QGroupBox("Playback")
        pb_l = QVBoxLayout(pb_box)

        nav = QHBoxLayout()
        self._btn_first = QPushButton("◀◀")
        self._btn_first.setFixedWidth(34)
        self._btn_first.setToolTip("First frame")
        self._btn_first.clicked.connect(self._go_first)
        nav.addWidget(self._btn_first)

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(28)
        self._btn_prev.setToolTip("Previous frame")
        self._btn_prev.clicked.connect(self._go_prev)
        nav.addWidget(self._btn_prev)

        self._btn_play = QPushButton("▶ Play")
        self._btn_play.setCheckable(True)
        self._btn_play.setFixedWidth(66)
        self._btn_play.toggled.connect(self._on_play_toggled)
        nav.addWidget(self._btn_play)

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(28)
        self._btn_next.setToolTip("Next frame")
        self._btn_next.clicked.connect(self._go_next)
        nav.addWidget(self._btn_next)

        self._btn_last = QPushButton("▶▶")
        self._btn_last.setFixedWidth(34)
        self._btn_last.setToolTip("Last frame")
        self._btn_last.clicked.connect(self._go_last)
        nav.addWidget(self._btn_last)
        pb_l.addLayout(nav)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self._fps_slider = QSlider(Qt.Orientation.Horizontal)
        self._fps_slider.setRange(1, 24)
        self._fps_slider.setValue(8)
        self._fps_slider.valueChanged.connect(self._on_fps_changed)
        fps_row.addWidget(self._fps_slider, 1)
        self._fps_lbl = QLabel("8")
        self._fps_lbl.setFixedWidth(24)
        fps_row.addWidget(self._fps_lbl)
        pb_l.addLayout(fps_row)

        self._frame_lbl = QLabel("Frame — / —")
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pb_l.addWidget(self._frame_lbl)
        ll.addWidget(pb_box)

        # Manual L/R flip
        flip_box = QGroupBox("Manual Correction")
        flip_l = QVBoxLayout(flip_box)
        self._btn_flip = QPushButton("↔ Flip L/R for Dir 1 NE")
        self._btn_flip.setToolTip(
            "Horizontally mirror all frames for the selected direction.\n"
            "Use this when the FRM sprite was stored mirrored."
        )
        self._btn_flip.clicked.connect(self._flip_current_dir)
        flip_l.addWidget(self._btn_flip)
        ll.addWidget(flip_box)

        # Status
        self._status_lbl = QLabel("Load a character first.")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color: #999; font-size: 10px;")
        ll.addWidget(self._status_lbl)
        ll.addStretch()

        splitter.addWidget(left)

        # ── RIGHT PANEL: 2×3 grid ─────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(400)
        grid = QGridLayout(right)
        grid.setSpacing(6)
        grid.setContentsMargins(6, 6, 6, 6)

        self._cells: list[_DirCell] = []
        for i in range(6):
            cell = _DirCell(i)
            cell.selected_signal.connect(self._select_dir)
            grid.addWidget(cell, i // 3, i % 3)
            self._cells.append(cell)

        splitter.addWidget(right)
        splitter.setSizes([250, 900])

    # ------------------------------------------------------------------
    # Direction selection
    # ------------------------------------------------------------------

    def _select_dir(self, idx: int):
        self._selected_dir = idx
        for i, btn in enumerate(self._dir_btns):
            btn.setChecked(i == idx)
        for i, cell in enumerate(self._cells):
            cell.set_selected(i == idx)
        self._btn_flip.setText(
            f"↔ Flip L/R for Dir {idx + 1} {_DIR_NAMES[idx]}"
        )
        self._refresh_status()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _on_play_toggled(self, playing: bool):
        if playing:
            self._btn_play.setText("⏸ Pause")
            self._play_timer.start(max(1, 1000 // self._fps_slider.value()))
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
        nxt = (self.state.current_frame + 1) % char.n_frames
        self.state.set_frame(nxt)

    def _go_first(self):
        self._btn_play.setChecked(False)
        self.state.set_frame(0)

    def _go_last(self):
        char = self.state.current_character
        if char:
            self._btn_play.setChecked(False)
            self.state.set_frame(char.n_frames - 1)

    def _go_prev(self):
        self._btn_play.setChecked(False)
        char = self.state.current_character
        if char:
            self.state.set_frame(max(0, self.state.current_frame - 1))

    def _go_next(self):
        self._btn_play.setChecked(False)
        char = self.state.current_character
        if char:
            self.state.set_frame(min(char.n_frames - 1, self.state.current_frame + 1))

    # ------------------------------------------------------------------
    # L/R flip operation
    # ------------------------------------------------------------------

    def _flip_current_dir(self):
        char = self.state.current_character
        if char is None:
            return
        d = self._selected_dir
        if d >= char.frames.shape[0]:
            return
        # Flip all frames for this direction along the W axis (horizontal mirror).
        # char.frames[d] has shape (n_frames, H, W, 3); W is axis 2.
        char.frames[d] = np.flip(char.frames[d], axis=2).copy()
        # Mirror frm_offsets x-coords if present
        if char.frm_offsets is not None and d < len(char.frm_offsets):
            cw = char.frames.shape[3]
            char.frm_offsets[d] = [
                (cw - 1 - ox, oy) for ox, oy in char.frm_offsets[d]
            ]
        self.state.character_updated.emit(self.state.selected_idx)
        self._refresh_cells(self.state.current_frame)
        self._status_lbl.setText(
            f"Dir {d + 1} {_DIR_NAMES[d]} flipped horizontally."
        )

    # ------------------------------------------------------------------
    # State signal handlers
    # ------------------------------------------------------------------

    def _on_char_changed(self, _idx: int):
        char = self.state.current_character
        if char is None:
            self._frame_lbl.setText("Frame — / —")
            self._status_lbl.setText("Load a character first.")
            for cell in self._cells:
                cell.set_frame(None)
            return
        self._refresh_cells(self.state.current_frame)
        self._refresh_status()

    def _on_char_updated(self, idx: int):
        if idx == self.state.selected_idx:
            self._on_char_changed(idx)

    def _on_frame_changed(self, frame: int):
        char = self.state.current_character
        if char is None:
            return
        self._refresh_cells(frame)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _refresh_cells(self, frame_idx: int):
        char = self.state.current_character
        if char is None:
            return
        n_dirs = char.frames.shape[0]
        fi = max(0, min(frame_idx, char.n_frames - 1))
        self._frame_lbl.setText(f"Frame {fi + 1} / {char.n_frames}")
        for d in range(6):
            if d < n_dirs:
                self._cells[d].set_frame(char.frames[d, fi])
            else:
                self._cells[d].set_frame(None)

    def _refresh_status(self):
        char = self.state.current_character
        if char is None:
            return
        frame = self.state.current_frame
        d = self._selected_dir
        cw, ch = char.frames.shape[3], char.frames.shape[2]
        offset_info = ""
        if (char.frm_offsets is not None
                and d < len(char.frm_offsets)
                and frame < len(char.frm_offsets[d])):
            ox, oy = char.frm_offsets[d][frame]
            offset_info = f"  offset ({ox}, {oy})"
        self._status_lbl.setText(
            f"'{char.name}'  Dir {d+1} {_DIR_NAMES[d]}\n"
            f"Frame {frame + 1}/{char.n_frames}  "
            f"canvas {cw}×{ch}{offset_info}"
        )
