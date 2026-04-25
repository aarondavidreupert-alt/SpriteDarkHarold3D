"""
Tab 1 — Asset Loader
Loads .npy / .png / .frm sprite sheets, shows thumbnail grids,
and manages the character list.
"""

import os
import struct
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QListWidget, QListWidgetItem, QFileDialog,
    QScrollArea, QGridLayout, QGroupBox, QSizePolicy, QFrame,
    QProgressBar, QLineEdit,
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

from gui.main_window import AppState, CharacterData, CRITTER_CATEGORIES


# -----------------------------------------------------------------------
# FRM reader
# -----------------------------------------------------------------------

def _read_pal(path: str):
    """Read a Fallout .pal file — 256 × 3 bytes, values in [0, 63] → scale ×4."""
    with open(path, "rb") as f:
        raw = f.read(768)
    return [(raw[i * 3] * 4, raw[i * 3 + 1] * 4, raw[i * 3 + 2] * 4) for i in range(256)]


def _read_frm(frm_path: str, palette) -> np.ndarray:
    """
    Parse a Fallout 2 .FRM file.
    Returns ndarray of shape (6, N, H, W, 3) uint8.
    """
    with open(frm_path, "rb") as f:
        data = f.read()

    offset = 0
    version         = struct.unpack_from(">I", data, offset)[0]; offset += 4
    fps             = struct.unpack_from(">H", data, offset)[0]; offset += 2
    action_frame    = struct.unpack_from(">H", data, offset)[0]; offset += 2
    frames_per_dir  = struct.unpack_from(">H", data, offset)[0]; offset += 2
    shift_x = struct.unpack_from(">6h", data, offset); offset += 12
    shift_y = struct.unpack_from(">6h", data, offset); offset += 12
    dir_offsets = struct.unpack_from(">6I", data, offset); offset += 24
    data_size = struct.unpack_from(">I", data, offset)[0]; offset += 4

    base = offset  # start of frame data
    dirs = []
    for d in range(6):
        pos = base + dir_offsets[d]
        frames = []
        for _ in range(frames_per_dir):
            w, h = struct.unpack_from(">HH", data, pos); pos += 4
            pixel_size = struct.unpack_from(">I", data, pos)[0]; pos += 4
            ox, oy = struct.unpack_from(">hh", data, pos); pos += 4
            pixels = np.frombuffer(data, dtype=np.uint8, count=w * h, offset=pos)
            pos += w * h
            # Palette lookup → RGB
            rgb = np.array([palette[p] for p in pixels], dtype=np.uint8).reshape(h, w, 3)
            frames.append(rgb)
        dirs.append(frames)

    # Pad to uniform size
    max_h = max(f.shape[0] for d in dirs for f in d)
    max_w = max(f.shape[1] for d in dirs for f in d)
    result = np.zeros((6, frames_per_dir, max_h, max_w, 3), dtype=np.uint8)
    for d, frames in enumerate(dirs):
        for fi, frame in enumerate(frames):
            h, w = frame.shape[:2]
            result[d, fi, :h, :w] = frame
    return result


# -----------------------------------------------------------------------
# Worker thread for background loading
# -----------------------------------------------------------------------

class LoadWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)   # CharacterData or None
    error    = pyqtSignal(str)

    def __init__(self, path: str, name: str, category: str, pal_path: str = ""):
        super().__init__()
        self.path = path
        self.name = name
        self.category = category
        self.pal_path = pal_path

    def run(self):
        try:
            ext = os.path.splitext(self.path)[1].lower()
            self.progress.emit(f"Loading {os.path.basename(self.path)}…")

            if ext == ".npy":
                arr = np.load(self.path)
                # Expected: (6, N, H, W, 3) or (6, N, H, W)
                if arr.ndim == 4:
                    arr = np.stack([arr] * 3, axis=-1)
                frames = arr.astype(np.uint8)

            elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
                import cv2
                img = cv2.imread(self.path)
                if img is None:
                    raise IOError(f"Cannot read image: {self.path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Treat single image as 1 direction × 1 frame
                frames = img[np.newaxis, np.newaxis]   # (1, 1, H, W, 3)
                # Tile to 6 directions
                frames = np.tile(frames, (6, 1, 1, 1, 1))

            elif ext == ".frm":
                if self.pal_path and os.path.exists(self.pal_path):
                    pal = _read_pal(self.pal_path)
                else:
                    pal = [(i, i, i) for i in range(256)]  # greyscale fallback
                frames = _read_frm(self.path, pal)

            else:
                raise ValueError(f"Unsupported file type: {ext}")

            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0, 255).astype(np.uint8)

            char = CharacterData(
                name=self.name,
                category=self.category,
                frames=frames,
            )
            self.finished.emit(char)

        except Exception as exc:
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# Thumbnail widget
# -----------------------------------------------------------------------

class ThumbnailGrid(QScrollArea):
    """Displays one row per direction, one thumbnail per frame."""

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
        # Clear
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        n_dirs, n_frames = frames.shape[0], frames.shape[1]
        for d in range(n_dirs):
            dir_label = QLabel(f"Dir {d+1}")
            dir_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._grid.addWidget(dir_label, d, 0)

            for f in range(min(n_frames, 30)):   # cap at 30 thumbnails per row
                img = frames[d, f]
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                h, w, c = img.shape
                qimg = QImage(img.data, w, h, w * c, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(
                    self.THUMB_SIZE, self.THUMB_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                lbl = QLabel()
                lbl.setPixmap(pix)
                lbl.setToolTip(f"Dir {d+1}, Frame {f+1}")
                self._grid.addWidget(lbl, d, f + 1)


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class AssetLoaderTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread: QThread | None = None
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ---- Left panel: controls + character list -------------------
        left = QVBoxLayout()
        root.addLayout(left, 1)

        # Load controls
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

        # Name + category
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

        # PAL path (for FRM)
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

        # Character list
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

        # ---- Right panel: thumbnail grid ----------------------------
        right = QVBoxLayout()
        root.addLayout(right, 3)

        info_box = QGroupBox("Sprite Preview")
        info_layout = QVBoxLayout(info_box)

        self.info_lbl = QLabel("No character loaded.")
        info_layout.addWidget(self.info_lbl)

        self.thumbnail_grid = ThumbnailGrid()
        info_layout.addWidget(self.thumbnail_grid)

        right.addWidget(info_box)

        # Connect state signals
        self.state.character_added.connect(self._refresh_char_list)
        self.state.character_removed.connect(self._refresh_char_list)
        self.state.selection_changed.connect(self._show_character)

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_character(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.exists(path):
            self.status_lbl.setText("Please select a valid file.")
            return

        name = self.name_edit.text().strip() or os.path.splitext(os.path.basename(path))[0]
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
        """Load all .npy / .frm files from a directory as one character each."""
        for fname in sorted(os.listdir(directory)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".npy", ".frm", ".png"):
                self.path_edit.setText(os.path.join(directory, fname))
                self.name_edit.setText(os.path.splitext(fname)[0])
                self._load_character()

    # ------------------------------------------------------------------
    # Character list management
    # ------------------------------------------------------------------

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
