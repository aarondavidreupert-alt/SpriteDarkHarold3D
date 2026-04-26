"""
Tab 1 — Asset Loader
Loads .npy / .png / .frm sprite sheets, shows thumbnail grids,
and manages the character list.  FRM files are decoded using the
verified pal.py + frmpixels.py pipeline from example_scripts/.
"""

import sys
import os
import numpy as np

# ── example_scripts bootstrap ───────────────────────────────────────────
_GUI_DIR    = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_GUI_DIR))   # gui/ → pipeline/ → repo
_SCRIPTS    = os.path.join(_REPO_ROOT, "example_scripts")
_PAL_PATH   = os.path.join(_REPO_ROOT, "color", "color.pal")

if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import pal as _pal_mod
import frmpixels as _frmpixels

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QListWidget, QListWidgetItem, QFileDialog,
    QScrollArea, QGridLayout, QGroupBox, QSizePolicy, QFrame,
    QProgressBar, QLineEdit, QSpinBox, QSlider,
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

from gui.main_window import AppState, CharacterData, CRITTER_CATEGORIES


# -----------------------------------------------------------------------
# Palette / pixel helpers
# -----------------------------------------------------------------------

def _load_palette(pal_path: str) -> np.ndarray:
    """Return (256, 3) uint8 RGB table via the verified pal.readPAL()."""
    with open(pal_path, "rb") as f:
        tuples = _pal_mod.readPAL(f)
    return np.array([(r, g, b) for r, g, b in tuples], dtype=np.uint8)


def _indices_to_rgba(indices: np.ndarray, pal_table: np.ndarray) -> np.ndarray:
    """(H, W) palette-index array → (H, W, 4) RGBA; index 0 is transparent."""
    rgba = np.zeros((*indices.shape, 4), dtype=np.uint8)
    rgba[..., :3] = pal_table[indices]
    rgba[..., 3]  = np.where(indices == 0, 0, 255).astype(np.uint8)
    return rgba


def _to_pixmap(rgba: np.ndarray) -> QPixmap:
    h, w = rgba.shape[:2]
    data = rgba.tobytes()
    qimg = QImage(data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


# -----------------------------------------------------------------------
# Worker thread for background loading (CharacterData pipeline)
# -----------------------------------------------------------------------

class LoadWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)   # CharacterData or None
    error    = pyqtSignal(str)

    def __init__(self, path: str, name: str, category: str, pal_path: str = ""):
        super().__init__()
        self.path     = path
        self.name     = name
        self.category = category
        self.pal_path = pal_path

    def run(self):
        try:
            ext = os.path.splitext(self.path)[1].lower()
            self.progress.emit(f"Loading {os.path.basename(self.path)}…")

            if ext == ".npy":
                arr = np.load(self.path)
                if arr.ndim == 4:
                    arr = np.stack([arr] * 3, axis=-1)
                frames = arr.astype(np.uint8)

            elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
                import cv2
                img = cv2.imread(self.path)
                if img is None:
                    raise IOError(f"Cannot read image: {self.path}")
                img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames = np.tile(img[np.newaxis, np.newaxis], (6, 1, 1, 1, 1))

            elif ext == ".frm":
                pal_path  = (self.pal_path if self.pal_path and os.path.exists(self.pal_path)
                             else _PAL_PATH)
                pal_table = _load_palette(pal_path)           # (256, 3) uint8

                with open(self.path, "rb") as f:
                    info = _frmpixels.readFRMInfo(f, exportImage=True)

                n_dirs   = info['numDirections']
                n_frames = info['numFrames']
                offsets  = info['frameOffsets']   # [dir][frame] → {'w','h',...}
                pixels   = info['framePixels']    # [dir][frame] → 1-D np.uint8

                max_w = max(fo['w'] for d in offsets for fo in d)
                max_h = max(fo['h'] for d in offsets for fo in d)

                frames = np.zeros((6, n_frames, max_h, max_w, 3), dtype=np.uint8)
                for d in range(n_dirs):
                    for fi in range(n_frames):
                        fo = offsets[d][fi]
                        w, h = fo['w'], fo['h']
                        idx  = pixels[d][fi].reshape(h, w)
                        frames[d, fi, :h, :w] = pal_table[idx]

            else:
                raise ValueError(f"Unsupported file type: {ext}")

            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0, 255).astype(np.uint8)

            char = CharacterData(name=self.name, category=self.category, frames=frames)
            self.finished.emit(char)

        except Exception as exc:
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# Thumbnail widget (CharacterData sprite preview)
# -----------------------------------------------------------------------

class ThumbnailGrid(QScrollArea):
    """One row per direction, one thumbnail per frame."""

    THUMB_SIZE = 64

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(2)
        self.setWidget(self._container)
        self.setMinimumHeight(200)

    def set_frames(self, frames: np.ndarray):
        """frames: (6, N, H, W, 3)"""
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        n_dirs, n_frames = frames.shape[0], frames.shape[1]
        for d in range(n_dirs):
            lbl = QLabel(f"Dir {d+1}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._grid.addWidget(lbl, d, 0)

            for f in range(min(n_frames, 30)):
                img = frames[d, f]
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                h, w, c = img.shape
                qimg = QImage(img.data, w, h, w * c, QImage.Format.Format_RGB888)
                pix  = QPixmap.fromImage(qimg).scaled(
                    self.THUMB_SIZE, self.THUMB_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                cell = QLabel()
                cell.setPixmap(pix)
                cell.setToolTip(f"Dir {d+1}, Frame {f+1}")
                self._grid.addWidget(cell, d, f + 1)


# -----------------------------------------------------------------------
# FRM Viewer
# -----------------------------------------------------------------------

class _DirectionRow(QWidget):
    """One direction row: frame display + spinbox/slider scrubber."""

    DISPLAY_SIZE = 160      # max px for frame thumbnail

    def __init__(self, dir_idx, frame_pixels, frame_offsets, pal_table, parent=None):
        super().__init__(parent)
        self._pixels  = frame_pixels   # list[np.ndarray]  (1-D indexed)
        self._offsets = frame_offsets  # list[dict]  with 'w' / 'h'
        self._pal     = pal_table      # (256, 3) uint8
        self._n       = len(frame_pixels)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)

        dir_lbl = QLabel(f"Dir {dir_idx + 1}")
        dir_lbl.setFixedWidth(44)
        lay.addWidget(dir_lbl)

        self._img = QLabel()
        self._img.setFixedSize(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet("background:#111;border:1px solid #444;")
        lay.addWidget(self._img)

        ctrl = QVBoxLayout()

        self._counter = QLabel(f"1 / {self._n}")
        self._counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrl.addWidget(self._counter)

        scrub = QHBoxLayout()
        self._spin = QSpinBox()
        self._spin.setRange(1, self._n)
        self._spin.setFixedWidth(56)
        scrub.addWidget(self._spin)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._n - 1)
        scrub.addWidget(self._slider)

        ctrl.addLayout(scrub)
        lay.addLayout(ctrl)

        self._spin.valueChanged.connect(self._on_spin)
        self._slider.valueChanged.connect(self._on_slide)
        self._show(0)

    # --- scrubber wiring (mutual blocking to avoid recursion) ---

    def _on_spin(self, v):
        self._slider.blockSignals(True)
        self._slider.setValue(v - 1)
        self._slider.blockSignals(False)
        self._show(v - 1)

    def _on_slide(self, v):
        self._spin.blockSignals(True)
        self._spin.setValue(v + 1)
        self._spin.blockSignals(False)
        self._show(v)

    def _show(self, idx):
        fo = self._offsets[idx]
        w, h = fo['w'], fo['h']
        indices = self._pixels[idx].reshape(h, w)
        rgba = _indices_to_rgba(indices, self._pal)
        pix  = _to_pixmap(rgba).scaled(
            self.DISPLAY_SIZE, self.DISPLAY_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._img.setPixmap(pix)
        self._counter.setText(f"{idx + 1} / {self._n}")


class FrmViewerPanel(QGroupBox):
    """Load FRM → display all directions with per-direction frame scrubbers."""

    def __init__(self, pal_path: str, parent=None):
        super().__init__("FRM Viewer", parent)
        self._pal_path  = pal_path
        self._pal_table = self._init_palette()
        self._build_ui()

    def _init_palette(self) -> np.ndarray:
        if os.path.exists(self._pal_path):
            return _load_palette(self._pal_path)
        # greyscale fallback so the viewer is usable even without color.pal
        return np.array([(i, i, i) for i in range(256)], dtype=np.uint8)

    def _build_ui(self):
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        self._btn_load = QPushButton("Load FRM…")
        self._btn_load.clicked.connect(self._open)
        top.addWidget(self._btn_load)

        self._info_lbl = QLabel("No FRM loaded.")
        self._info_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self._info_lbl, 1)
        lay.addLayout(top)

        # Scrollable area that holds one _DirectionRow per direction
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(350)
        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setSpacing(4)
        self._rows_layout.addStretch()          # always last item
        self._scroll.setWidget(self._rows_container)
        lay.addWidget(self._scroll)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open FRM File", "",
            "Fallout FRM (*.frm *.FRM);;All Files (*)",
        )
        if path:
            self._load(path)

    def _load(self, path: str):
        try:
            with open(path, "rb") as f:
                info = _frmpixels.readFRMInfo(f, exportImage=True)
        except Exception as exc:
            self._info_lbl.setText(f"Error: {exc}")
            return

        n_dirs   = info['numDirections']
        n_frames = info['numFrames']
        self._info_lbl.setText(
            f"{os.path.basename(path)}  —  "
            f"{n_dirs} dir{'s' if n_dirs != 1 else ''} "
            f"× {n_frames} frame{'s' if n_frames != 1 else ''}"
        )

        # Remove old direction rows (everything except the trailing stretch)
        while self._rows_layout.count() > 1:
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for d in range(n_dirs):
            row = _DirectionRow(
                d,
                info['framePixels'][d],
                info['frameOffsets'][d],
                self._pal_table,
                self._rows_container,
            )
            self._rows_layout.insertWidget(d, row)


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class AssetLoaderTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread: QThread | None = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ── Left panel: controls + character list ────────────────────────
        left = QVBoxLayout()
        root.addLayout(left, 1)

        load_box = QGroupBox("Load Asset")
        load_layout = QVBoxLayout(load_box)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path to .npy / .png / .frm …")
        load_layout.addWidget(self.path_edit)

        btn_row = QHBoxLayout()
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self.open_file_dialog)
        btn_row.addWidget(self.btn_browse)
        self.btn_browse_dir = QPushButton("Browse Folder…")
        self.btn_browse_dir.clicked.connect(self.open_dir_dialog)
        btn_row.addWidget(self.btn_browse_dir)
        load_layout.addLayout(btn_row)

        meta_row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Character name…")
        meta_row.addWidget(QLabel("Name:"))
        meta_row.addWidget(self.name_edit)
        load_layout.addLayout(meta_row)

        cat_row = QHBoxLayout()
        self.cat_combo = QComboBox()
        for cat in CRITTER_CATEGORIES:
            self.cat_combo.addItem(cat.title(), cat)
        cat_row.addWidget(QLabel("Category:"))
        cat_row.addWidget(self.cat_combo)
        load_layout.addLayout(cat_row)

        self.pal_edit = QLineEdit()
        self.pal_edit.setPlaceholderText("Optional: color.pal path for .frm files")
        load_layout.addWidget(self.pal_edit)

        self.btn_load = QPushButton("Load Character")
        self.btn_load.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_load.clicked.connect(self._load_character)
        load_layout.addWidget(self.btn_load)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        load_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("")
        load_layout.addWidget(self.status_lbl)

        left.addWidget(load_box)

        char_box = QGroupBox("Loaded Characters")
        char_layout = QVBoxLayout(char_box)
        self.char_list = QListWidget()
        self.char_list.currentRowChanged.connect(self._on_char_selected)
        char_layout.addWidget(self.char_list)
        char_btn_row = QHBoxLayout()
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self._remove_character)
        char_btn_row.addWidget(self.btn_remove)
        char_layout.addLayout(char_btn_row)
        left.addWidget(char_box)

        # ── Right panel: sprite preview + FRM viewer ─────────────────────
        right = QVBoxLayout()
        root.addLayout(right, 3)

        info_box = QGroupBox("Sprite Preview")
        info_layout = QVBoxLayout(info_box)
        self.info_lbl = QLabel("No character loaded.")
        info_layout.addWidget(self.info_lbl)
        self.thumbnail_grid = ThumbnailGrid()
        info_layout.addWidget(self.thumbnail_grid)
        right.addWidget(info_box, 1)

        self.frm_viewer = FrmViewerPanel(_PAL_PATH)
        right.addWidget(self.frm_viewer, 2)

        # Signals
        self.state.character_added.connect(self._refresh_char_list)
        self.state.character_removed.connect(self._refresh_char_list)
        self.state.selection_changed.connect(self._show_character)

    # ── File dialogs ──────────────────────────────────────────────────────

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Sprite Asset", "",
            "Sprite Assets (*.npy *.png *.jpg *.bmp *.frm);;All Files (*)"
        )
        if path:
            self.path_edit.setText(path)
            if not self.name_edit.text():
                self.name_edit.setText(os.path.splitext(os.path.basename(path))[0])

    def open_dir_dialog(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Sprite Directory")
        if directory:
            self._load_directory(directory)

    # ── Loading ───────────────────────────────────────────────────────────

    def _load_character(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.exists(path):
            self.status_lbl.setText("Please select a valid file.")
            return

        name     = self.name_edit.text().strip() or os.path.splitext(os.path.basename(path))[0]
        category = self.cat_combo.currentData()
        pal_path = self.pal_edit.text().strip()

        self.btn_load.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_lbl.setText("")

        self._worker = LoadWorker(path, name, category, pal_path)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status_lbl.setText)
        self._worker.finished.connect(self._on_load_done)
        self._worker.error.connect(self._on_load_error)
        self._thread.start()

    def _on_load_done(self, char: CharacterData):
        self._thread.quit()
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        if char:
            self.state.add_character(char)
            self.status_lbl.setText(
                f"Loaded '{char.name}' — {char.n_frames} frames × 6 views"
            )
        else:
            self.status_lbl.setText("Load failed.")

    def _on_load_error(self, msg: str):
        self._thread.quit()
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self.status_lbl.setText(f"Error: {msg}")

    def _load_directory(self, directory: str):
        for fname in sorted(os.listdir(directory)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".npy", ".frm", ".png"):
                self.path_edit.setText(os.path.join(directory, fname))
                self.name_edit.setText(os.path.splitext(fname)[0])
                self._load_character()

    # ── Character list ────────────────────────────────────────────────────

    def _refresh_char_list(self, _=None):
        self.char_list.clear()
        for c in self.state.characters:
            item = QListWidgetItem(f"{c.name}  [{c.category}]  {c.n_frames}f")
            self.char_list.addItem(item)

    def _on_char_selected(self, row: int):
        if row >= 0:
            self.state.set_selected(row)

    def _remove_character(self):
        row = self.char_list.currentRow()
        if row >= 0:
            self.state.remove_character(row)

    def _show_character(self, idx: int):
        char = self.state.current_character
        if char is None:
            self.info_lbl.setText("No character selected.")
            return
        shape = char.frames.shape
        self.info_lbl.setText(
            f"{char.name}  |  category: {char.category}  |  "
            f"shape: {shape[0]}dirs × {shape[1]}frames × {shape[2]}×{shape[3]}px"
        )
        self.thumbnail_grid.set_frames(char.frames)
