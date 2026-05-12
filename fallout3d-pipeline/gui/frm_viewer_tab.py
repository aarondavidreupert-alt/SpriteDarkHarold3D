"""
Tab 1b — FRM Registration Viewer
Lets the user inspect spatially-registered FRM animation frames before
upscaling and pose detection.  Shows all 6 directions simultaneously in a
2×3 grid with playback controls and a per-direction L/R flip tool.
"""

import sys
import os
import numpy as np

# ── example_scripts bootstrap (same as asset_loader_tab) ────────────────
_GUI_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_GUI_DIR))
_SCRIPTS   = os.path.join(_REPO_ROOT, "example_scripts")
_PAL_PATH  = os.path.join(_REPO_ROOT, "color", "color.pal")

if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import pal as _pal_mod
import frmpixels as _frmpixels

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QGridLayout, QFrame, QSplitter, QSpinBox,
    QRadioButton, QButtonGroup, QMessageBox, QScrollArea,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
)
from PyQt6.QtGui import QPixmap, QImage, QTransform
from PyQt6.QtCore import Qt, QTimer, QEvent, QRectF, pyqtSignal

from gui.main_window import AppState

_DIR_NAMES = ["NE", "E", "SE", "SW", "W", "NW"]


def _flood_fill_mask(grid: np.ndarray, sx: int, sy: int,
                     target: int) -> np.ndarray:
    """
    4-connected iterative flood fill on a 2-D uint8 palette-index array.
    Returns a boolean mask of all pixels reachable from (sx, sy) with value == target.
    """
    H, W = grid.shape
    mask = np.zeros((H, W), dtype=bool)
    if grid[sy, sx] != target:
        return mask
    stack = [(sx, sy)]
    while stack:
        x, y = stack.pop()
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        if mask[y, x] or grid[y, x] != target:
            continue
        mask[y, x] = True
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    return mask


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


def _to_pixmap_raw(
    img: np.ndarray,
    bg_rgb: tuple[int, int, int] = (13, 13, 13),
) -> QPixmap:
    """RGB (H, W, 3) uint8 → QPixmap at NATIVE resolution (no scaling).
    Used by _DirCell so the QGraphicsView handles zoom itself."""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    # Composite sprite over background colour (display-only, no mutation)
    if bg_rgb != (0, 0, 0):
        bg_mask = np.all(img == 0, axis=2)   # True where pixel is black bg
        canvas = np.empty_like(img)
        canvas[:] = bg_rgb
        canvas[~bg_mask] = img[~bg_mask]
        img = canvas
    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class _DirCell(QFrame):
    """One cell in the 2×3 grid.  Shows current frame for one direction."""
    selected_signal = pyqtSignal(int)               # dir_idx
    pixel_clicked   = pyqtSignal(int, float, float) # dir_idx, x_frac, y_frac

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

        self._scene = QGraphicsScene(self)
        self._view  = QGraphicsView(self._scene)
        self._view.setMinimumSize(140, 140)
        self._bg_color = "#0d0d0d"
        self._bg_rgb: tuple[int, int, int] = (13, 13, 13)
        self._view.setStyleSheet("background: #0d0d0d; border: none;")
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self._view.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self._scene.addItem(self._pix_item)
        lay.addWidget(self._view, 1)

        # last raw frame array (for bg re-render)
        self._last_img: np.ndarray | None = None

        # zoom / pan state
        self._zoom_factor = 1.0
        self._MIN_ZOOM    = 0.5
        self._MAX_ZOOM    = 16.0
        self._pan_active  = False
        self._pan_start   = None

        self._view.viewport().installEventFilter(self)
        self._view.viewport().setMouseTracking(True)

    # ------------------------------------------------------------------
    # Border / selection
    # ------------------------------------------------------------------

    def _set_border(self, selected: bool):
        color = "#ffdd00" if selected else "#444"
        self.setStyleSheet(f"border: 2px solid {color};")

    def set_selected(self, sel: bool):
        self._set_border(sel)

    def set_bg_color(self, hex_color: str):
        self._bg_color = hex_color
        self._view.setStyleSheet(f"background: {hex_color}; border: none;")
        h = hex_color.lstrip("#")
        try:
            self._bg_rgb = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        except (ValueError, IndexError):
            self._bg_rgb = (13, 13, 13)
        if self._pix_item.pixmap() and not self._pix_item.pixmap().isNull():
            self._rerender_bg()

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def set_frame(self, img: np.ndarray | None):
        self._last_img = img
        if img is None:
            self._pix_item.setPixmap(QPixmap())
            self._scene.setSceneRect(QRectF())
            return
        pix = _to_pixmap_raw(img, bg_rgb=self._bg_rgb)
        self._pix_item.setPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self._fit_in_view()

    def _rerender_bg(self):
        if self._last_img is None:
            return
        pix = _to_pixmap_raw(self._last_img, bg_rgb=self._bg_rgb)
        self._pix_item.setPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))

    def _fit_in_view(self):
        """Reset zoom/pan so the full sprite fits the cell."""
        sr = self._scene.sceneRect()
        if sr.isEmpty():
            return
        vr = self._view.viewport().rect()
        if vr.width() == 0 or vr.height() == 0:
            return
        sx = vr.width()  / sr.width()
        sy = vr.height() / sr.height()
        scale = min(sx, sy)
        self._zoom_factor = scale
        t = QTransform()
        t.scale(scale, scale)
        self._view.setTransform(t)
        self._view.centerOn(self._pix_item)

    # ------------------------------------------------------------------
    # Event filter — wheel zoom + pan + click forwarding
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is not self._view.viewport():
            return False

        etype = event.type()

        # ── wheel → zoom centered on cursor ───────────────────────────
        if etype == QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else (1.0 / 1.15)
            new_zoom = max(self._MIN_ZOOM,
                           min(self._MAX_ZOOM, self._zoom_factor * factor))
            if new_zoom == self._zoom_factor:
                return True
            mouse_pos = event.position().toPoint()
            scene_pos = self._view.mapToScene(mouse_pos)
            ratio = new_zoom / self._zoom_factor
            self._zoom_factor = new_zoom
            t = self._view.transform()
            t.scale(ratio, ratio)
            self._view.setTransform(t)
            new_scene_pos = self._view.mapToScene(mouse_pos)
            delta_scene = new_scene_pos - scene_pos
            self._view.translate(delta_scene.x(), delta_scene.y())
            return True

        # ── middle-button press → start pan ───────────────────────────
        if (etype == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.MiddleButton):
            self._pan_active = True
            self._pan_start  = event.position().toPoint()
            self._view.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return True

        # ── mouse move → pan ──────────────────────────────────────────
        if etype == QEvent.Type.MouseMove and self._pan_active:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self._view.translate(
                delta.x() / self._zoom_factor,
                delta.y() / self._zoom_factor,
            )
            return True

        # ── middle-button release → stop pan ──────────────────────────
        if (etype == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.MiddleButton):
            self._pan_active = False
            self._view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            return True

        # ── double left-click → reset zoom ────────────────────────────
        if (etype == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton):
            self._fit_in_view()
            return True

        # ── left-click → select cell + pixel_clicked ──────────────────
        if (etype == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton):
            self.selected_signal.emit(self.dir_idx)
            vp_pt = event.position().toPoint()
            scene_pt = self._view.mapToScene(vp_pt)
            sr = self._scene.sceneRect()
            if sr.width() > 0 and sr.height() > 0:
                xf = max(0.0, min(1.0, scene_pt.x() / sr.width()))
                yf = max(0.0, min(1.0, scene_pt.y() / sr.height()))
                self.pixel_clicked.emit(self.dir_idx, xf, yf)
            return False  # let event propagate

        return False

    def mousePressEvent(self, event):
        # Frame-level press — just ensures selection if the viewport
        # event was somehow not caught.
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
        self._removed_colors: dict[int, tuple[int, int, int]] = {}

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.frame_changed.connect(self._on_frame_changed)
        self.state.character_updated.connect(self._on_char_updated)
        self.state.character_added.connect(self._on_char_added)

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

        # View controls (zoom reset)
        view_box = QGroupBox("View")
        view_l = QVBoxLayout(view_box)
        self._btn_reset_zoom = QPushButton("⊡ Reset Zoom (all cells)")
        self._btn_reset_zoom.setToolTip(
            "Double-click any cell to reset that cell.\n"
            "This button resets all 6 at once.")
        self._btn_reset_zoom.clicked.connect(self._reset_all_zoom)
        view_l.addWidget(self._btn_reset_zoom)

        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("BG:"))
        for label, color in [("⬛ Black", "#0d0d0d"),
                              ("🟦 Blue",  "#0a1a3a"),
                              ("🟩 Green", "#0a2a0a")]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.clicked.connect(
                lambda _checked, c=color: self._set_bg_color(c)
            )
            bg_row.addWidget(btn)
        view_l.addLayout(bg_row)
        ll.addWidget(view_box)

        # Paint-bucket shadow removal
        shadow_box = QGroupBox("Shadow Removal")
        sv = QVBoxLayout(shadow_box)

        self._btn_shadow_mode = QPushButton("🪣 Shadow Removal: OFF")
        self._btn_shadow_mode.setCheckable(True)
        self._btn_shadow_mode.setToolTip(
            "Click a shadow pixel on any frame to flood-fill it\n"
            "with transparent (index 0).")
        self._btn_shadow_mode.toggled.connect(self._on_shadow_mode_toggled)
        sv.addWidget(self._btn_shadow_mode)

        scope_row = QHBoxLayout()
        self._shadow_scope_current = QRadioButton("This frame only")
        self._shadow_scope_all = QRadioButton("All frames")
        self._shadow_scope_all.setChecked(True)
        _sg = QButtonGroup(self)
        _sg.addButton(self._shadow_scope_current)
        _sg.addButton(self._shadow_scope_all)
        scope_row.addWidget(self._shadow_scope_current)
        scope_row.addWidget(self._shadow_scope_all)
        scope_row.addStretch()
        sv.addLayout(scope_row)

        self._btn_shadow_restore = QPushButton("↩ Restore Original")
        self._btn_shadow_restore.clicked.connect(self._restore_shadow)
        sv.addWidget(self._btn_shadow_restore)

        self._btn_clean_orphans = QPushButton("✦ Remove Orphan Pixels")
        self._btn_clean_orphans.setToolTip(
            "Erases isolated non-background pixels with no non-background\n"
            "4-connected neighbour. Applied to ALL frames and ALL directions."
        )
        self._btn_clean_orphans.clicked.connect(self._remove_orphans)
        sv.addWidget(self._btn_clean_orphans)

        sv.addWidget(QLabel("Removed colours:"))
        self._swatch_scroll = QScrollArea()
        self._swatch_scroll.setWidgetResizable(True)
        self._swatch_scroll.setFixedHeight(72)
        self._swatch_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._swatch_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._swatch_inner = QWidget()
        self._swatch_layout = QHBoxLayout(self._swatch_inner)
        self._swatch_layout.setContentsMargins(2, 2, 2, 2)
        self._swatch_layout.setSpacing(4)
        self._swatch_layout.addStretch()
        self._swatch_scroll.setWidget(self._swatch_inner)
        sv.addWidget(self._swatch_scroll)

        ll.addWidget(shadow_box)

        # Canvas size control
        canvas_box = QGroupBox("Canvas Size")
        canvas_l = QVBoxLayout(canvas_box)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size:"))
        self._canvas_spin = QSpinBox()
        self._canvas_spin.setRange(50, 800)
        self._canvas_spin.setValue(100)
        self._canvas_spin.setSingleStep(10)
        self._canvas_spin.setSuffix(" px")
        size_row.addWidget(self._canvas_spin, 1)
        canvas_l.addLayout(size_row)
        self._btn_apply_canvas = QPushButton("Apply Canvas")
        self._btn_apply_canvas.setToolTip(
            "Recomposite FRM frames onto a canvas of the selected size.\n"
            "Smaller canvas = faster upscaling."
        )
        self._btn_apply_canvas.clicked.connect(self._apply_canvas)
        canvas_l.addWidget(self._btn_apply_canvas)
        ll.addWidget(canvas_box)

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
            cell.pixel_clicked.connect(self._on_pixel_clicked)
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
        self._refresh_status()

    # ------------------------------------------------------------------
    # Zoom reset
    # ------------------------------------------------------------------

    def _reset_all_zoom(self):
        for cell in self._cells:
            cell._fit_in_view()

    def _set_bg_color(self, hex_color: str):
        for cell in self._cells:
            cell.set_bg_color(hex_color)

    def _remove_orphans(self):
        char = self.state.current_character
        if char is None:
            self._status_lbl.setText("No character loaded.")
            return

        if char.frames_backup is None:
            char.frames_backup = char.frames.copy()
            if char.frames_pal_idx is not None:
                char.frames_pal_idx_backup = char.frames_pal_idx.copy()

        frames = char.frames                      # (n_dirs, n_frames, H, W, 3)
        n_dirs, n_frames, H, W, _ = frames.shape
        total_removed = 0

        for d in range(n_dirs):
            for fi in range(n_frames):
                img = frames[d, fi]               # (H, W, 3)
                fg = np.any(img > 0, axis=2)      # foreground mask

                # Count fg 4-connected neighbours per pixel
                nb = np.zeros((H, W), dtype=np.int32)
                nb[:-1, :] += fg[1:,  :]
                nb[1:,  :] += fg[:-1, :]
                nb[:, :-1] += fg[:,  1:]
                nb[:,  1:] += fg[:, :-1]

                orphan = fg & (nb == 0)
                if not orphan.any():
                    continue

                total_removed += int(orphan.sum())

                # Record unique orphan colours before zeroing
                orphan_pixels = img[orphan]
                for rgb_row in np.unique(orphan_pixels, axis=0):
                    rgb = (int(rgb_row[0]), int(rgb_row[1]), int(rgb_row[2]))
                    key = -(rgb[0] * 65536 + rgb[1] * 256 + rgb[2] + 1)
                    self._add_swatch(key, rgb)

                frames[d, fi][orphan] = 0
                if char.frames_pal_idx is not None:
                    char.frames_pal_idx[d, fi][orphan] = 0

        self.state.character_updated.emit(self.state.selected_idx)
        self._status_lbl.setText(
            f"Removed {total_removed} orphan pixel(s) across all frames.")

    # ------------------------------------------------------------------
    # Removed-colours swatch helpers
    # ------------------------------------------------------------------

    def _get_pal_table(self) -> np.ndarray | None:
        """Return the 256×3 uint8 palette table, or None on failure."""
        try:
            if not os.path.exists(_PAL_PATH):
                return None
            with open(_PAL_PATH, "rb") as f:
                return np.array(
                    [(r, g, b) for r, g, b in _pal_mod.readPAL(f)],
                    dtype=np.uint8,
                )
        except Exception:
            return None

    def _add_swatch(self, pal_idx: int, rgb: tuple[int, int, int]):
        """Add a colour chip to the removed-colours panel if not already present."""
        if pal_idx in self._removed_colors:
            return
        self._removed_colors[pal_idx] = rgb
        r, g, b = rgb
        chip = QLabel()
        chip.setFixedSize(36, 52)
        chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chip.setToolTip(f"Palette index: {pal_idx}\nRGB: ({r}, {g}, {b})")
        chip.setStyleSheet(
            f"background: rgb({r},{g},{b});"
            f"border: 1px solid #555;"
            f"color: {'#000' if (r + g + b) > 380 else '#fff'};"
            f"font-size: 9px;"
        )
        chip.setText(f"#{pal_idx}" if pal_idx >= 0 else "orph")
        count = self._swatch_layout.count()
        self._swatch_layout.insertWidget(count - 1, chip)

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
    # Shadow removal (paint-bucket flood fill)
    # ------------------------------------------------------------------

    def _on_shadow_mode_toggled(self, active: bool):
        self._btn_shadow_mode.setText(
            "🪣 Shadow Removal: ON" if active else "🪣 Shadow Removal: OFF")
        cursor = Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor
        for cell in self._cells:
            cell._view.viewport().setCursor(cursor)

    def _on_pixel_clicked(self, dir_idx: int, xf: float, yf: float):
        if not self._btn_shadow_mode.isChecked():
            return
        char = self.state.current_character
        if char is None:
            return
        if char.frames_pal_idx is None:
            self._status_lbl.setText(
                "Shadow removal needs a .frm source. Re-load as .frm.")
            return

        fi = self.state.current_frame
        frame_pal = char.frames_pal_idx[dir_idx, fi]   # (H, W) uint8
        H, W = frame_pal.shape
        px = max(0, min(int(xf * W), W - 1))
        py = max(0, min(int(yf * H), H - 1))
        target_idx = int(frame_pal[py, px])
        if target_idx == 0:
            self._status_lbl.setText("Clicked background (index 0) — nothing to do.")
            return

        # Backup before first modification
        if char.frames_backup is None:
            char.frames_backup         = char.frames.copy()
            char.frames_pal_idx_backup = char.frames_pal_idx.copy()

        if self._shadow_scope_all.isChecked():
            n_dirs   = char.frames_pal_idx.shape[0]
            n_frames = char.frames_pal_idx.shape[1]
            targets  = [(d, f) for d in range(n_dirs) for f in range(n_frames)]
        else:
            targets = [(dir_idx, fi)]

        for d, f in targets:
            mask = _flood_fill_mask(char.frames_pal_idx[d, f], px, py, target_idx)
            if not mask.any():
                continue
            char.frames_pal_idx[d, f][mask] = 0
            char.frames[d, f][mask] = 0   # black in RGB

        scope = "all frames" if self._shadow_scope_all.isChecked() else "this frame"
        self.state.character_updated.emit(self.state.selected_idx)
        self._status_lbl.setText(
            f"Removed palette index {target_idx} ({scope}).")

        pal_table = self._get_pal_table()
        if pal_table is not None and target_idx < len(pal_table):
            rgb = tuple(int(v) for v in pal_table[target_idx])
        else:
            rgb = (80, 80, 80)
        self._add_swatch(target_idx, rgb)

    def _restore_shadow(self):
        char = self.state.current_character
        if char is None:
            self._status_lbl.setText("No character loaded.")
            return
        if char.frames_backup is None:
            self._status_lbl.setText("No backup available.")
            return
        char.frames = char.frames_backup.copy()
        if char.frames_pal_idx_backup is not None:
            char.frames_pal_idx = char.frames_pal_idx_backup.copy()
        char.frames_backup         = None
        char.frames_pal_idx_backup = None
        self.state.character_updated.emit(self.state.selected_idx)
        self._status_lbl.setText("Original frames restored.")

        self._removed_colors.clear()
        while self._swatch_layout.count() > 1:
            item = self._swatch_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ------------------------------------------------------------------
    # Canvas recomposite
    # ------------------------------------------------------------------

    def _on_char_added(self, idx: int):
        """Auto-apply canvas when a new FRM character is loaded."""
        chars = self.state.characters
        char = chars[idx] if 0 <= idx < len(chars) else None
        if char is None:
            return
        src = char.source_path or ""
        if src.lower().endswith(".frm") and os.path.exists(src):
            QTimer.singleShot(50, self._apply_canvas)

    def _apply_canvas(self):
        char = self.state.current_character
        if char is None:
            self._status_lbl.setText("No character loaded.")
            return
        src = char.source_path or ""
        if not src.lower().endswith(".frm") or not os.path.exists(src):
            self._status_lbl.setText("Canvas resize only works for .frm files.")
            return

        canvas_size = self._canvas_spin.value()
        self._btn_apply_canvas.setEnabled(False)
        self._status_lbl.setText(f"Recompositing at {canvas_size}×{canvas_size}…")

        try:
            pal_path = _PAL_PATH if os.path.exists(_PAL_PATH) else None
            if pal_path is None:
                self._status_lbl.setText("color.pal not found — cannot recomposite.")
                self._btn_apply_canvas.setEnabled(True)
                return

            with open(pal_path, "rb") as f:
                pal_table = np.array(
                    [(r, g, b) for r, g, b in _pal_mod.readPAL(f)], dtype=np.uint8
                )

            with open(src, "rb") as f:
                info = _frmpixels.readFRMInfo(f, exportImage=True)

            n_dirs   = info['numDirections']
            n_frames = info['numFrames']
            offsets  = info['frameOffsets']
            pixels   = info['framePixels']

            cw = canvas_size
            ch = canvas_size
            anchor_x = cw // 2
            anchor_y = ch * 3 // 4

            new_frames  = np.zeros((6, n_frames, ch, cw, 3),  dtype=np.uint8)
            new_pal_idx = np.zeros((6, n_frames, ch, cw),     dtype=np.uint8)

            for d in range(n_dirs):
                ox, oy = 0, 0
                for fi in range(n_frames):
                    fo = offsets[d][fi]
                    fw, fh = fo['w'], fo['h']
                    ox += fo['x']
                    oy += fo['y']

                    left = anchor_x - (fw // 2 - ox)
                    top  = anchor_y - (fh - oy)

                    x0 = max(left, 0);  y0 = max(top, 0)
                    x1 = min(left + fw, cw);  y1 = min(top + fh, ch)
                    sx0 = x0 - left;  sy0 = y0 - top

                    if x1 > x0 and y1 > y0:
                        raw = pixels[d][fi].reshape(fh, fw)
                        patch = raw[sy0:sy0+(y1-y0), sx0:sx0+(x1-x0)]
                        new_frames [d, fi, y0:y1, x0:x1] = pal_table[patch]
                        new_pal_idx[d, fi, y0:y1, x0:x1] = patch

            char.frames         = new_frames
            char.frames_pal_idx = new_pal_idx
            char.frm_offsets = None
            self.state.character_updated.emit(self.state.selected_idx)
            self._status_lbl.setText(
                f"Recomposited at {canvas_size}×{canvas_size} px."
            )
        except Exception as exc:
            self._status_lbl.setText(f"Error: {exc}")
        finally:
            self._btn_apply_canvas.setEnabled(True)

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
