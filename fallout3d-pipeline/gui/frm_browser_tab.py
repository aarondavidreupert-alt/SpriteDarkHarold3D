"""
Tab 1c — FRM Browser

Scan a folder for Fallout 2 critter .frm files, show them in a
configurable matrix of thumbnails, preview a selected file's six
directions with playback, and load it into the pipeline with one click.
"""

from __future__ import annotations

import os
import sys
import logging
import numpy as np

# example_scripts bootstrap (mirrors frm_viewer_tab.py)
_GUI_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_GUI_DIR))
_SCRIPTS   = os.path.join(_REPO_ROOT, "example_scripts")
_PAL_PATH  = os.path.join(_REPO_ROOT, "color", "color.pal")

if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

try:
    import pal as _pal_mod  # type: ignore
    import frmpixels as _frmpixels  # type: ignore
    _FRM_AVAILABLE = True
except Exception:
    _FRM_AVAILABLE = False

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QGroupBox, QComboBox, QCheckBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QGridLayout,
    QSpinBox, QSlider, QSizePolicy,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import (
    Qt, QObject, pyqtSignal, QRunnable, QThreadPool, QTimer,
)

from gui.main_window import AppState
from pipeline.frm_catalog import FrmCatalog, FrmEntry, TYPE_CODES

_logger = logging.getLogger(__name__)
_DIR_NAMES = ["NE", "E", "SE", "SW", "W", "NW"]


# ---------------------------------------------------------------------------
# Palette helper
# ---------------------------------------------------------------------------

def _load_palette() -> np.ndarray | None:
    if not _FRM_AVAILABLE or not os.path.exists(_PAL_PATH):
        return None
    try:
        with open(_PAL_PATH, "rb") as f:
            return np.array(
                [(r, g, b) for r, g, b in _pal_mod.readPAL(f)], dtype=np.uint8)
    except Exception as exc:
        _logger.warning("Failed to load palette: %s", exc)
        return None


def _frm_first_frame_rgb(path: str, pal_table: np.ndarray | None) -> np.ndarray | None:
    """Decode FRM dir 0, frame 0 as RGB (H,W,3) uint8."""
    if not _FRM_AVAILABLE:
        return None
    try:
        with open(path, "rb") as f:
            info = _frmpixels.readFRMInfo(f, exportImage=True)
        offsets = info["frameOffsets"]
        pixels  = info["framePixels"]
        if not offsets or not pixels or not offsets[0] or not pixels[0]:
            return None
        fo = offsets[0][0]
        idx = pixels[0][0].reshape(fo["h"], fo["w"])
        if pal_table is not None:
            return pal_table[idx]
        # fall back to greyscale
        g = idx.astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    except Exception as exc:
        _logger.debug("FRM thumbnail decode failed for %s: %s", path, exc)
        return None


def _frm_all_dirs_first_frame(path: str, pal_table: np.ndarray | None
                              ) -> list[np.ndarray] | None:
    """All 6 (or fewer) directions, frame 0, as list of RGB arrays."""
    if not _FRM_AVAILABLE:
        return None
    try:
        with open(path, "rb") as f:
            info = _frmpixels.readFRMInfo(f, exportImage=True)
        offsets = info["frameOffsets"]
        pixels  = info["framePixels"]
        out: list[np.ndarray] = []
        for d in range(min(6, len(offsets))):
            if not pixels[d]:
                continue
            fo = offsets[d][0]
            idx = pixels[d][0].reshape(fo["h"], fo["w"])
            if pal_table is not None:
                out.append(pal_table[idx])
            else:
                g = idx.astype(np.uint8)
                out.append(np.stack([g, g, g], axis=-1))
        return out
    except Exception as exc:
        _logger.debug("FRM preview decode failed for %s: %s", path, exc)
        return None


def _rgb_to_pixmap(rgb: np.ndarray, max_side: int = 48) -> QPixmap:
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qi = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qi.copy())
    if max(h, w) > max_side:
        pix = pix.scaled(max_side, max_side,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    return pix


# ---------------------------------------------------------------------------
# Thumbnail worker (QThreadPool)
# ---------------------------------------------------------------------------

class _ThumbSignals(QObject):
    done = pyqtSignal(str, object)   # (path, QPixmap or None)


class _ThumbTask(QRunnable):
    def __init__(self, path: str, pal_table: np.ndarray | None,
                 signals: _ThumbSignals):
        super().__init__()
        self.path = path
        self.pal  = pal_table
        self.signals = signals

    def run(self):
        rgb = _frm_first_frame_rgb(self.path, self.pal)
        pix = _rgb_to_pixmap(rgb) if rgb is not None else None
        self.signals.done.emit(self.path, pix)


# ---------------------------------------------------------------------------
# Preview cell (mirrors _DirCell pattern from frm_viewer_tab)
# ---------------------------------------------------------------------------

class _PreviewCell(QFrame):
    def __init__(self, dir_idx: int):
        super().__init__()
        self.dir_idx = dir_idx
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(1)
        lbl = QLabel(f"Dir {dir_idx + 1} {_DIR_NAMES[dir_idx]}")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("font-size: 9px; color: #aaa;")
        v.addWidget(lbl)
        self._img = QLabel()
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setMinimumSize(80, 80)
        self._img.setStyleSheet("background:#0d0d0d;")
        v.addWidget(self._img, 1)

    def set_frame(self, img: np.ndarray | None):
        if img is None:
            self._img.clear()
        else:
            self._img.setPixmap(_rgb_to_pixmap(img, max_side=120))


# ---------------------------------------------------------------------------
# FrmBrowserTab
# ---------------------------------------------------------------------------

_AXIS_FIELDS = [
    ("Character",      "char_label"),
    ("Animation",      "anim_label"),
    ("Type (label)",   "type_label"),
    ("Animation code", "anim_code"),
    ("Type code",      "type_code"),
]


class FrmBrowserTab(QWidget):

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state    = state
        self._catalog = FrmCatalog()
        self._pal     = _load_palette()
        self._thumb_cache: dict[str, QPixmap] = {}
        self._thumb_pending: set[str] = set()
        self._thumb_signals = _ThumbSignals()
        self._thumb_signals.done.connect(self._on_thumb_done)
        self._pool = QThreadPool.globalInstance()

        self._cur_entry: FrmEntry | None = None
        self._preview_dirs: list[np.ndarray] = []
        self._preview_anim_timer = QTimer(self)
        self._preview_anim_timer.timeout.connect(self._preview_tick)
        self._preview_frame_idx = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_centre())
        splitter.addWidget(self._build_right())
        splitter.setSizes([240, 700, 320])

    def _build_left(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)

        # Folder
        grp_dir = QGroupBox("FRM Folder")
        gv = QVBoxLayout(grp_dir)
        self.txt_folder = QLineEdit(getattr(self.state, "frm_folder", "") or "")
        gv.addWidget(self.txt_folder)
        h = QHBoxLayout()
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_folder)
        h.addWidget(btn_browse)
        btn_scan = QPushButton("Scan")
        btn_scan.setStyleSheet("font-weight:bold;")
        btn_scan.clicked.connect(self._scan_folder)
        h.addWidget(btn_scan)
        gv.addLayout(h)
        v.addWidget(grp_dir)

        # Type filter
        grp_filter = QGroupBox("Filter — Type")
        fv = QVBoxLayout(grp_filter)
        self._type_checks: dict[str, QCheckBox] = {}
        for code, label in TYPE_CODES.items():
            cb = QCheckBox(f"{code} — {label}")
            cb.setChecked(True)
            cb.toggled.connect(self._refresh_matrix)
            fv.addWidget(cb)
            self._type_checks[code] = cb
        v.addWidget(grp_filter)

        # Axes
        grp_axes = QGroupBox("Matrix Axes")
        av = QVBoxLayout(grp_axes)
        h_row = QHBoxLayout()
        h_row.addWidget(QLabel("Rows (Y):"))
        self.cb_row = QComboBox()
        for label, field in _AXIS_FIELDS:
            self.cb_row.addItem(label, field)
        self.cb_row.setCurrentIndex(0)  # Character
        self.cb_row.currentIndexChanged.connect(self._refresh_matrix)
        h_row.addWidget(self.cb_row, 1)
        av.addLayout(h_row)
        h_col = QHBoxLayout()
        h_col.addWidget(QLabel("Cols (X):"))
        self.cb_col = QComboBox()
        for label, field in _AXIS_FIELDS:
            self.cb_col.addItem(label, field)
        self.cb_col.setCurrentIndex(1)  # Animation
        self.cb_col.currentIndexChanged.connect(self._refresh_matrix)
        h_col.addWidget(self.cb_col, 1)
        av.addLayout(h_col)
        v.addWidget(grp_axes)

        # Status
        self.status_lbl = QLabel("—")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#88f; font-style:italic; font-size:11px;")
        v.addWidget(self.status_lbl)
        v.addStretch()
        w.setMinimumWidth(220)
        w.setMaximumWidth(280)
        return w

    def _build_centre(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed)
        self.table.setIconSize(self.table.iconSize() * 1)  # default
        self.table.cellClicked.connect(self._on_cell_clicked)
        # Rebuild thumb requests when scrolling
        self.table.verticalScrollBar().valueChanged.connect(self._populate_visible)
        self.table.horizontalScrollBar().valueChanged.connect(self._populate_visible)
        v.addWidget(self.table)
        return w

    def _build_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        grp = QGroupBox("Preview")
        gv = QVBoxLayout(grp)
        self.preview_name_lbl = QLabel("—")
        self.preview_name_lbl.setStyleSheet("font-weight:bold;")
        gv.addWidget(self.preview_name_lbl)
        self.preview_meta_lbl = QLabel("")
        self.preview_meta_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        gv.addWidget(self.preview_meta_lbl)

        # 2×3 grid of dirs
        grid_box = QFrame()
        grid = QGridLayout(grid_box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        self._preview_cells: list[_PreviewCell] = []
        for i in range(6):
            cell = _PreviewCell(i)
            grid.addWidget(cell, i // 3, i % 3)
            self._preview_cells.append(cell)
        gv.addWidget(grid_box)

        # Playback controls (decorative — first frame only is decoded above,
        # but we still expose Stop/Play uniformity with viewer)
        h_pb = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._on_play_toggled)
        h_pb.addWidget(self.btn_play)
        h_pb.addWidget(QLabel("FPS:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 24)
        self.spin_fps.setValue(8)
        h_pb.addWidget(self.spin_fps)
        h_pb.addStretch()
        gv.addLayout(h_pb)

        # Canvas size + load button
        h_cv = QHBoxLayout()
        h_cv.addWidget(QLabel("Canvas:"))
        self.spin_canvas = QSpinBox()
        self.spin_canvas.setRange(50, 800)
        self.spin_canvas.setValue(100)
        self.spin_canvas.setSingleStep(10)
        self.spin_canvas.setSuffix(" px")
        h_cv.addWidget(self.spin_canvas)
        h_cv.addStretch()
        gv.addLayout(h_cv)

        self.btn_load = QPushButton("→ Load into Pipeline")
        self.btn_load.setStyleSheet("font-weight:bold;")
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self._load_into_pipeline)
        gv.addWidget(self.btn_load)

        v.addWidget(grp)
        v.addStretch()
        w.setMinimumWidth(280)
        w.setMaximumWidth(360)
        return w

    # ------------------------------------------------------------------
    # Folder & scan
    # ------------------------------------------------------------------

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose FRM folder",
                                             self.txt_folder.text() or "")
        if d:
            self.txt_folder.setText(d)

    def _scan_folder(self):
        folder = self.txt_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            self._set_status("Folder not found.")
            return
        self.state.frm_folder = folder
        self._catalog.scan(folder)
        self._thumb_cache.clear()
        self._thumb_pending.clear()
        n = len(self._catalog.entries)
        self._set_status(f"Scanned {n} FRM files.")
        self._refresh_matrix()

    # ------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------

    def _active_type_codes(self) -> list[str]:
        return [c for c, cb in self._type_checks.items() if cb.isChecked()]

    def _refresh_matrix(self):
        if not self._catalog.entries:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        type_codes = self._active_type_codes()
        entries = self._catalog.filter(type_codes=type_codes)
        if not entries:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self._set_status("No FRM matches current filter.")
            return

        row_field = self.cb_row.currentData()
        col_field = self.cb_col.currentData()
        rows, cols, cell_map = self._catalog.as_matrix(
            row_field, col_field, entries=entries)

        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setVerticalHeaderLabels(rows)
        cell_size = 56
        for r in range(len(rows)):
            self.table.setRowHeight(r, cell_size)
        for c in range(len(cols)):
            self.table.setColumnWidth(c, cell_size)

        self._matrix_rows = rows
        self._matrix_cols = cols
        self._matrix_map  = cell_map

        for r, rv in enumerate(rows):
            for c, cv in enumerate(cols):
                entry = cell_map.get((rv, cv))
                if entry is None:
                    item = QTableWidgetItem("")
                    item.setBackground(Qt.GlobalColor.darkGray)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                else:
                    item = QTableWidgetItem("")
                    item.setToolTip(
                        f"{entry.filename}\n{entry.type_label} · "
                        f"{entry.char_label} · {entry.anim_label}")
                    item.setData(Qt.ItemDataRole.UserRole, entry.path)
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)

        self._set_status(
            f"{len(entries)} FRM in matrix ({len(rows)} rows × {len(cols)} cols).")
        self._populate_visible()

    # ------------------------------------------------------------------
    # Lazy thumbnails
    # ------------------------------------------------------------------

    def _populate_visible(self):
        if self.table.rowCount() == 0 or self.table.columnCount() == 0:
            return
        vp = self.table.viewport().rect()
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                rect = self.table.visualRect(self.table.model().index(r, c))
                if not vp.intersects(rect):
                    continue
                item = self.table.item(r, c)
                if item is None:
                    continue
                path = item.data(Qt.ItemDataRole.UserRole)
                if not path:
                    continue
                pix = self._thumb_cache.get(path)
                if pix is not None:
                    if item.icon().isNull():
                        from PyQt6.QtGui import QIcon
                        item.setIcon(QIcon(pix))
                        self.table.setIconSize(pix.size())
                    continue
                if path in self._thumb_pending:
                    continue
                self._thumb_pending.add(path)
                self._pool.start(_ThumbTask(path, self._pal, self._thumb_signals))

    def _on_thumb_done(self, path: str, pix):
        self._thumb_pending.discard(path)
        if pix is None:
            return
        self._thumb_cache[path] = pix
        # Find item displaying this path (linear — small for visible page)
        from PyQt6.QtGui import QIcon
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item is not None and item.data(Qt.ItemDataRole.UserRole) == path:
                    item.setIcon(QIcon(pix))
                    self.table.setIconSize(pix.size())

    # ------------------------------------------------------------------
    # Cell click → preview
    # ------------------------------------------------------------------

    def _on_cell_clicked(self, row: int, col: int):
        item = self.table.item(row, col)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        # find FrmEntry from catalog
        entry = next((e for e in self._catalog.entries if e.path == path), None)
        if entry is None:
            return
        self._cur_entry = entry
        self.preview_name_lbl.setText(entry.filename)
        self.preview_meta_lbl.setText(
            f"{entry.type_label} · {entry.char_label} · {entry.anim_label}")
        dirs = _frm_all_dirs_first_frame(entry.path, self._pal)
        self._preview_dirs = dirs or []
        self._preview_frame_idx = 0
        for i, cell in enumerate(self._preview_cells):
            cell.set_frame(self._preview_dirs[i] if i < len(self._preview_dirs) else None)
        self.btn_load.setEnabled(True)

    def _on_play_toggled(self, playing: bool):
        # Decorative animation — flashes between dirs since we only decoded
        # frame 0 cheaply.  Real playback happens once loaded into pipeline.
        if playing:
            self.btn_play.setText("⏸ Pause")
            self._preview_anim_timer.start(max(1, 1000 // self.spin_fps.value()))
        else:
            self.btn_play.setText("▶ Play")
            self._preview_anim_timer.stop()

    def _preview_tick(self):
        if not self._preview_dirs:
            return
        # Cycle highlight by hiding all but one cell briefly
        self._preview_frame_idx = (self._preview_frame_idx + 1) % 6
        for i, cell in enumerate(self._preview_cells):
            cell.setStyleSheet(
                "QFrame { border: 2px solid #ffdd00; }"
                if i == self._preview_frame_idx else
                "QFrame { border: 1px solid #444; }")

    # ------------------------------------------------------------------
    # Load into pipeline
    # ------------------------------------------------------------------

    def _load_into_pipeline(self):
        if self._cur_entry is None:
            return
        mw = self.window()
        asset_tab = getattr(mw, "tab_asset", None)
        tabs = getattr(mw, "tabs", None)
        if asset_tab is None or tabs is None:
            self._set_status("Cannot find Asset Loader tab.")
            return
        try:
            asset_tab.path_edit.setText(self._cur_entry.path)
            stem = os.path.splitext(self._cur_entry.filename)[0]
            asset_tab.name_edit.setText(stem)
            tabs.setCurrentIndex(0)
            asset_tab._load_character()
        except Exception as exc:
            self._set_status(f"Load error: {exc}")
            return
        self._set_status(f"Loading {self._cur_entry.filename} into pipeline…")

    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        self.status_lbl.setText(text)
        _logger.info("FrmBrowser: %s", text)
