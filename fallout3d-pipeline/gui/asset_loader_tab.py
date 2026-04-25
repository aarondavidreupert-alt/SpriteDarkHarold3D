"""
Tab 1 — Asset Loader
Loads .npy / .png / .frm sprite sheets, shows thumbnail grids,
and manages the character list.
"""

import json
import os
import struct
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QListWidget, QListWidgetItem, QFileDialog,
    QScrollArea, QGridLayout, QGroupBox, QSizePolicy, QFrame,
    QProgressBar, QLineEdit, QMessageBox,
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

from gui.main_window import AppState, CharacterData, CRITTER_CATEGORIES


# -----------------------------------------------------------------------
# PAL / FRM helpers
# -----------------------------------------------------------------------

def _read_pal(path: str) -> np.ndarray:
    """Read a Fallout .pal file — returns (256, 3) uint8, values scaled ×4."""
    with open(path, "rb") as f:
        raw = f.read(768)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(256, 3).copy()
    return np.clip(arr.astype(np.uint16) * 4, 0, 255).astype(np.uint8)


def _read_frm(frm_path: str, palette: np.ndarray) -> np.ndarray:
    """
    Parse a Fallout 2 .FRM file.
    Returns ndarray of shape (6, N, H, W, 3) uint8.
    palette must be (256, 3) uint8.
    """
    with open(frm_path, "rb") as f:
        data = f.read()

    offset = 0
    _version        = struct.unpack_from(">I", data, offset)[0]; offset += 4
    _fps            = struct.unpack_from(">H", data, offset)[0]; offset += 2
    _action_frame   = struct.unpack_from(">H", data, offset)[0]; offset += 2
    frames_per_dir  = struct.unpack_from(">H", data, offset)[0]; offset += 2
    _shift_x        = struct.unpack_from(">6h", data, offset);   offset += 12
    _shift_y        = struct.unpack_from(">6h", data, offset);   offset += 12
    dir_offsets     = struct.unpack_from(">6I", data, offset);   offset += 24
    _data_size      = struct.unpack_from(">I",  data, offset)[0]; offset += 4

    base = offset  # start of frame data
    dirs = []
    for d in range(6):
        pos = base + dir_offsets[d]
        frames = []
        for _ in range(frames_per_dir):
            w, h        = struct.unpack_from(">HH", data, pos); pos += 4
            _pixel_size = struct.unpack_from(">I",  data, pos)[0]; pos += 4
            _ox, _oy    = struct.unpack_from(">hh", data, pos); pos += 4
            pixels = np.frombuffer(data, dtype=np.uint8, count=w * h, offset=pos)
            pos += w * h
            rgb = palette[pixels].reshape(h, w, 3)
            frames.append(rgb)
        dirs.append(frames)

    # Pad all frames to uniform size
    max_h = max(f.shape[0] for d in dirs for f in d)
    max_w = max(f.shape[1] for d in dirs for f in d)
    result = np.zeros((6, frames_per_dir, max_h, max_w, 3), dtype=np.uint8)
    for d, frames in enumerate(dirs):
        for fi, frame in enumerate(frames):
            h, w = frame.shape[:2]
            result[d, fi, :h, :w] = frame
    return result


# -----------------------------------------------------------------------
# PNG spritesheet splitter
# -----------------------------------------------------------------------

def split_spritesheet(png_path: str, image_map: dict, key: str) -> np.ndarray:
    """
    Split a horizontal spritesheet using imageMap.json metadata.

    image_map[key]["frameOffsets"] is a list-of-lists:
        frameOffsets[dir_idx][frame_idx] = {"sx": x, "w": w, "h": h, ...}

    Returns ndarray of shape (n_dirs, n_frames, max_H, max_W, 3) uint8.
    """
    from PIL import Image

    meta = image_map.get(key)
    if meta is None:
        raise KeyError(f"Key '{key}' not found in imageMap.json")

    frame_offsets = meta.get("frameOffsets")
    if not frame_offsets:
        raise ValueError(f"imageMap['{key}'] has no frameOffsets")

    sheet = Image.open(png_path).convert("RGB")

    dirs_pil: list[list[np.ndarray]] = []
    for dir_entries in frame_offsets:
        dir_frames = []
        for entry in dir_entries:
            sx = entry["sx"]
            w  = entry["w"]
            h  = entry["h"]
            crop = sheet.crop((sx, 0, sx + w, h))
            dir_frames.append(np.array(crop, dtype=np.uint8))
        dirs_pil.append(dir_frames)

    n_dirs   = len(dirs_pil)
    n_frames = max(len(d) for d in dirs_pil)
    max_h    = max(f.shape[0] for d in dirs_pil for f in d)
    max_w    = max(f.shape[1] for d in dirs_pil for f in d)

    result = np.zeros((n_dirs, n_frames, max_h, max_w, 3), dtype=np.uint8)
    for d, dir_frames in enumerate(dirs_pil):
        for fi, frame in enumerate(dir_frames):
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

    def __init__(
        self,
        path: str,
        name: str,
        category: str,
        palette: "np.ndarray | None" = None,
        image_map: "dict | None" = None,
        image_map_key: str = "",
    ):
        super().__init__()
        self.path          = path
        self.name          = name
        self.category      = category
        self.palette       = palette
        self.image_map     = image_map
        self.image_map_key = image_map_key

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
                if self.image_map and self.image_map_key:
                    self.progress.emit(f"Splitting spritesheet with imageMap.json…")
                    frames = split_spritesheet(self.path, self.image_map, self.image_map_key)
                else:
                    import cv2
                    img = cv2.imread(self.path)
                    if img is None:
                        raise IOError(f"Cannot read image: {self.path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames = np.tile(img[np.newaxis, np.newaxis], (6, 1, 1, 1, 1))

            elif ext == ".frm":
                frames = _read_frm(self.path, self.palette)

            else:
                raise ValueError(f"Unsupported file type: {ext}")

            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0, 255).astype(np.uint8)

            char = CharacterData(name=self.name, category=self.category, frames=frames)
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
        """frames: (n_dirs, n_frames, H, W, 3)"""
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        n_dirs, n_frames = frames.shape[0], frames.shape[1]
        for d in range(n_dirs):
            dir_label = QLabel(f"Dir {d + 1}")
            dir_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._grid.addWidget(dir_label, d, 0)

            for f in range(min(n_frames, 30)):
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
                lbl.setToolTip(f"Dir {d + 1}, Frame {f + 1}")
                self._grid.addWidget(lbl, d, f + 1)


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class AssetLoaderTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread: QThread | None = None
        self._image_map: dict | None = None
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ---- Left panel: controls + character list -------------------
        left = QVBoxLayout()
        root.addLayout(left, 1)

        # ---- Palette section ----------------------------------------
        pal_box = QGroupBox("Colour Palette (.pal)")
        pal_layout = QVBoxLayout(pal_box)

        pal_path_row = QHBoxLayout()
        self.pal_edit = QLineEdit()
        self.pal_edit.setPlaceholderText("color.pal — required for .frm files")
        self.pal_edit.setReadOnly(True)
        pal_path_row.addWidget(self.pal_edit)
        btn_browse_pal = QPushButton("Browse…")
        btn_browse_pal.clicked.connect(self._browse_pal)
        pal_path_row.addWidget(btn_browse_pal)
        pal_layout.addLayout(pal_path_row)

        pal_btn_row = QHBoxLayout()
        self.btn_load_pal = QPushButton("Load Palette")
        self.btn_load_pal.clicked.connect(self._load_palette)
        pal_btn_row.addWidget(self.btn_load_pal)
        self.pal_status_lbl = QLabel("No palette loaded.")
        self.pal_status_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        pal_btn_row.addWidget(self.pal_status_lbl, 1)
        pal_layout.addLayout(pal_btn_row)

        left.addWidget(pal_box)

        # ---- Load controls ------------------------------------------
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

        # imageMap.json picker (for PNG spritesheet splitting)
        imap_row = QHBoxLayout()
        self.image_map_edit = QLineEdit()
        self.image_map_edit.setPlaceholderText("imageMap.json — optional, for PNG spritesheets")
        self.image_map_edit.setReadOnly(True)
        imap_row.addWidget(self.image_map_edit)
        btn_imap = QPushButton("Browse…")
        btn_imap.clicked.connect(self._browse_image_map)
        imap_row.addWidget(btn_imap)
        load_layout.addLayout(imap_row)

        self.image_map_status_lbl = QLabel("")
        self.image_map_status_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        load_layout.addWidget(self.image_map_status_lbl)

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
    # Palette controls
    # ------------------------------------------------------------------

    def _browse_pal(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Palette File", "",
            "Fallout Palette (*.pal);;All Files (*)"
        )
        if path:
            self.pal_edit.setText(path)

    def _load_palette(self):
        path = self.pal_edit.text().strip()
        if not path or not os.path.exists(path):
            self.pal_status_lbl.setText("Select a .pal file first.")
            self.pal_status_lbl.setStyleSheet("color: #e06060; font-size: 11px;")
            return
        try:
            pal = _read_pal(path)
            self.state.palette = pal
            self.state.palette_path = path
            self.pal_status_lbl.setText(
                f"Loaded: 256 colours from {os.path.basename(path)}"
            )
            self.pal_status_lbl.setStyleSheet("color: #60c060; font-size: 11px;")
        except Exception as exc:
            self.pal_status_lbl.setText(f"Error: {exc}")
            self.pal_status_lbl.setStyleSheet("color: #e06060; font-size: 11px;")

    # ------------------------------------------------------------------
    # imageMap.json controls
    # ------------------------------------------------------------------

    def _browse_image_map(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select imageMap.json", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._image_map = json.load(f)
            self.image_map_edit.setText(path)
            n_keys = len(self._image_map)
            self.image_map_status_lbl.setText(
                f"Loaded imageMap with {n_keys} key(s)."
            )
            self.image_map_status_lbl.setStyleSheet("color: #60c060; font-size: 11px;")
        except Exception as exc:
            self._image_map = None
            self.image_map_status_lbl.setText(f"Error: {exc}")
            self.image_map_status_lbl.setStyleSheet("color: #e06060; font-size: 11px;")

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

        ext = os.path.splitext(path)[1].lower()

        # Guard: FRM requires palette
        if ext == ".frm" and self.state.palette is None:
            self.status_lbl.setText(
                "Please load color.pal first (see Colour Palette section above)."
            )
            self.status_lbl.setStyleSheet("color: #e06060;")
            return
        self.status_lbl.setStyleSheet("")

        name     = self.name_edit.text().strip() or os.path.splitext(os.path.basename(path))[0]
        category = self.cat_combo.currentData()

        # imageMap key derived from file stem; try a few common formats
        image_map_key = ""
        if self._image_map and ext in (".png", ".jpg", ".jpeg", ".bmp"):
            stem = os.path.splitext(os.path.basename(path))[0].lower()
            # Try exact stem, then stem prefixed with common art paths
            candidates = [stem] + [f"art/critters/{stem}", f"art/{stem}"]
            for c in candidates:
                if c in self._image_map:
                    image_map_key = c
                    break
            if not image_map_key:
                self.status_lbl.setText(
                    f"imageMap loaded but key for '{stem}' not found — loading as single image."
                )

        self.btn_load.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_lbl.setText("")

        self._worker = LoadWorker(
            path=path,
            name=name,
            category=category,
            palette=self.state.palette,
            image_map=self._image_map if image_map_key else None,
            image_map_key=image_map_key,
        )
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
                f"Loaded '{char.name}' — {char.n_frames} frames × {char.frames.shape[0]} views"
            )
        else:
            self.status_lbl.setText("Load failed.")

    def _on_load_error(self, msg: str):
        self._thread.quit()
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self.status_lbl.setText(f"Error: {msg}")
        self.status_lbl.setStyleSheet("color: #e06060;")

    def _load_directory(self, directory: str):
        """Load all .npy / .frm / .png files from a directory as one character each."""
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
