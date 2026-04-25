"""
Tab 4 — Pose Library & Averaging
Matches frames across multiple characters of the same category,
computes a confidence-weighted master skeleton, and visualises
all characters overlaid in the same 3D view.
"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QComboBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QSplitter, QSlider,
    QDoubleSpinBox, QProgressBar, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QColor

from gui.main_window import AppState, CRITTER_CATEGORIES
from pipeline.pose_library import PoseLibrary

try:
    import pyqtgraph.opengl as gl
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False


# -----------------------------------------------------------------------
# Colour palette for overlaid characters
# -----------------------------------------------------------------------

_CHAR_COLORS = [
    (1.0, 0.5, 0.0),
    (0.2, 0.8, 1.0),
    (0.8, 0.2, 0.8),
    (0.2, 1.0, 0.4),
    (1.0, 0.9, 0.1),
    (1.0, 0.3, 0.3),
]

from pipeline.pose_triangulator import POSE_CONNECTIONS


def _rgba(rgb, a=0.9):
    return (*rgb, a)


# -----------------------------------------------------------------------
# Averaging worker
# -----------------------------------------------------------------------

class AverageWorker(QObject):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, library: PoseLibrary, category: str, threshold: float):
        super().__init__()
        self.library = library
        self.category = category
        self.threshold = threshold

    def run(self):
        try:
            master = self.library.compute_master_skeleton(self.category, self.threshold)
            self.finished.emit(master)
        except Exception as exc:
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# Multi-skeleton 3D overlay view
# -----------------------------------------------------------------------

class OverlayViewer3D(QWidget):
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
            self._items: list = []
            layout.addWidget(self._view)
        else:
            layout.addWidget(QLabel("pyqtgraph not available — install pyqtgraph + PyOpenGL."))

    def set_skeletons(
        self,
        skeletons: list[np.ndarray],   # each (33, 3)
        colors: list[tuple],
        master: np.ndarray | None = None,
    ):
        if not _GL_AVAILABLE:
            return

        for item in self._items:
            self._view.removeItem(item)
        self._items = []

        def _add_skel(skel, rgb, alpha=0.9, width=2):
            for s, e in POSE_CONNECTIONS:
                if np.all(skel[s] == 0) or np.all(skel[e] == 0):
                    continue
                pts = np.array([skel[s], skel[e]], dtype=np.float32)
                col = np.array([[*rgb, alpha]] * 2, dtype=np.float32)
                line = gl.GLLinePlotItem(pos=pts, color=col, width=width, antialias=True)
                self._view.addItem(line)
                self._items.append(line)

            valid = ~np.all(skel == 0, axis=1)
            if valid.any():
                sc = gl.GLScatterPlotItem(
                    pos=skel[valid].astype(np.float32),
                    color=np.array([[*rgb, alpha]] * valid.sum(), dtype=np.float32),
                    size=6, pxMode=True,
                )
                self._view.addItem(sc)
                self._items.append(sc)

        for skel, rgb in zip(skeletons, colors):
            _add_skel(skel, rgb, alpha=0.5, width=1)

        if master is not None:
            _add_skel(master, (1.0, 1.0, 1.0), alpha=1.0, width=3)


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class PoseLibraryTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._master_skeleton: np.ndarray | None = None  # (N, 33, 3)
        self._thread: QThread | None = None
        self._build_ui()

        self.state.character_added.connect(self._refresh)
        self.state.character_removed.connect(self._refresh)
        self.state.character_updated.connect(self._refresh)

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # ---- Left: controls -----------------------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        cat_box = QGroupBox("Category Filter")
        cat_layout = QHBoxLayout(cat_box)
        cat_layout.addWidget(QLabel("Show:"))
        self.cat_combo = QComboBox()
        self.cat_combo.addItem("All", "")
        for cat in CRITTER_CATEGORIES:
            self.cat_combo.addItem(cat.title(), cat)
        self.cat_combo.currentIndexChanged.connect(self._refresh)
        cat_layout.addWidget(self.cat_combo)
        left_layout.addWidget(cat_box)

        # Character list
        char_box = QGroupBox("Characters in Library")
        char_layout = QVBoxLayout(char_box)
        self.char_list = QListWidget()
        self.char_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        char_layout.addWidget(self.char_list)
        left_layout.addWidget(char_box, 1)

        # Averaging controls
        avg_box = QGroupBox("Pose Averaging")
        avg_layout = QVBoxLayout(avg_box)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Outlier threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1.0, 5.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(2.5)
        thr_row.addWidget(self.threshold_spin)
        avg_layout.addLayout(thr_row)

        self.btn_avg = QPushButton("Compute Master Skeleton")
        self.btn_avg.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_avg.clicked.connect(self._compute_average)
        avg_layout.addWidget(self.btn_avg)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        avg_layout.addWidget(self.progress)

        self.status_lbl = QLabel("")
        avg_layout.addWidget(self.status_lbl)
        left_layout.addWidget(avg_box)

        # Matching table
        match_box = QGroupBox("Pose Matches (reference → other)")
        match_layout = QVBoxLayout(match_box)
        self.match_table = QTableWidget(0, 4)
        self.match_table.setHorizontalHeaderLabels(["Ref frame", "Match char", "Match frame", "Similarity"])
        self.match_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        match_layout.addWidget(self.match_table)

        self.btn_match = QPushButton("Find Pose Matches")
        self.btn_match.clicked.connect(self._find_matches)
        match_layout.addWidget(self.btn_match)
        left_layout.addWidget(match_box, 1)

        # Save/load library
        io_box = QGroupBox("Library")
        io_layout = QHBoxLayout(io_box)
        self.btn_save = QPushButton("Save Library…")
        self.btn_save.clicked.connect(self._save_library)
        io_layout.addWidget(self.btn_save)
        self.btn_load_lib = QPushButton("Load Library…")
        self.btn_load_lib.clicked.connect(self._load_library)
        io_layout.addWidget(self.btn_load_lib)
        left_layout.addWidget(io_box)

        # ---- Right: 3D overlay --------------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)

        self._viewer = OverlayViewer3D()
        right_layout.addWidget(self._viewer, 1)

        # Frame slider
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider)
        frame_row.addWidget(self.frame_slider)
        self.frame_lbl = QLabel("0")
        frame_row.addWidget(self.frame_lbl)
        right_layout.addLayout(frame_row)

        splitter.setSizes([400, 800])

    # ------------------------------------------------------------------

    def _refresh(self, _=None):
        cat_filter = self.cat_combo.currentData()
        self.char_list.clear()
        for i, c in enumerate(self.state.characters):
            if cat_filter and c.category != cat_filter:
                continue
            has_skel = "✓ skeleton" if c.skeleton_3d is not None else "  no skeleton"
            item = QListWidgetItem(f"{c.name}  [{c.category}]  {has_skel}")
            color = _CHAR_COLORS[i % len(_CHAR_COLORS)]
            item.setForeground(QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255)))
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.char_list.addItem(item)
            item.setSelected(c.skeleton_3d is not None)

        self._update_viewer(0)

    def _selected_char_indices(self) -> list[int]:
        return [
            self.char_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.char_list.count())
            if self.char_list.item(i).isSelected()
        ]

    # ------------------------------------------------------------------
    # Averaging
    # ------------------------------------------------------------------

    def _compute_average(self):
        cat_filter = self.cat_combo.currentData()
        if not cat_filter:
            self.status_lbl.setText("Please select a category.")
            return

        # Sync library from state
        self._sync_library()

        self.btn_avg.setEnabled(False)
        self.progress.setVisible(True)
        self.status_lbl.setText("Computing master skeleton…")

        self._worker = AverageWorker(
            self.state.pose_library, cat_filter, self.threshold_spin.value()
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_avg_done)
        self._worker.error.connect(self._on_avg_error)
        self._thread.start()

    def _sync_library(self):
        lib = self.state.pose_library
        lib.characters.clear()
        for c in self.state.characters:
            if c.skeleton_3d is None:
                continue
            confs = c.confidences if c.confidences is not None else np.ones((c.n_frames, 33))
            lib.add_character(c.name, c.category, c.skeleton_3d, confs)

    def _on_avg_done(self, master: np.ndarray | None):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_avg.setEnabled(True)
        if master is None:
            self.status_lbl.setText("No characters with skeletons found.")
            return
        self._master_skeleton = master
        n = master.shape[0]
        self.frame_slider.setRange(0, max(0, n - 1))
        self.status_lbl.setText(f"Master skeleton computed — {n} frames.")
        self._update_viewer(0)

    def _on_avg_error(self, msg: str):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_avg.setEnabled(True)
        self.status_lbl.setText(f"Error: {msg}")

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _on_slider(self, val: int):
        self.frame_lbl.setText(str(val + 1))
        self._update_viewer(val)

    def _update_viewer(self, frame: int):
        selected_indices = self._selected_char_indices()
        skeletons, colors = [], []
        for i in selected_indices:
            char = self.state.characters[i]
            if char.skeleton_3d is None:
                continue
            f = min(frame, char.skeleton_3d.shape[0] - 1)
            skeletons.append(char.skeleton_3d[f])
            colors.append(_CHAR_COLORS[i % len(_CHAR_COLORS)])

        master_frame = None
        if self._master_skeleton is not None:
            f = min(frame, self._master_skeleton.shape[0] - 1)
            master_frame = self._master_skeleton[f]

        self._viewer.set_skeletons(skeletons, colors, master_frame)

    # ------------------------------------------------------------------
    # Pose matching
    # ------------------------------------------------------------------

    def _find_matches(self):
        cat_filter = self.cat_combo.currentData()
        if not cat_filter:
            self.status_lbl.setText("Select a category first.")
            return
        self._sync_library()
        lib = self.state.pose_library
        members = lib.get_by_category(cat_filter)
        if len(members) < 2:
            self.status_lbl.setText("Need at least 2 characters with skeletons.")
            return

        self.match_table.setRowCount(0)
        ref_idx, ref_char = members[0]
        for other_idx, other_char in members[1:]:
            matches = lib.match_poses(ref_idx, other_idx)
            for fa, fb, sim in matches:
                row = self.match_table.rowCount()
                self.match_table.insertRow(row)
                self.match_table.setItem(row, 0, QTableWidgetItem(str(fa)))
                self.match_table.setItem(row, 1, QTableWidgetItem(other_char.name))
                self.match_table.setItem(row, 2, QTableWidgetItem(str(fb)))
                self.match_table.setItem(row, 3, QTableWidgetItem(f"{sim:.3f}"))

        self.status_lbl.setText(f"Found {self.match_table.rowCount()} matches.")

    # ------------------------------------------------------------------
    # Library I/O
    # ------------------------------------------------------------------

    def _save_library(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pose Library", "pose_library.json", "JSON (*.json)"
        )
        if path:
            self._sync_library()
            self.state.pose_library.save(path)
            self.status_lbl.setText(f"Saved: {path}")

    def _load_library(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pose Library", "", "JSON (*.json)"
        )
        if path:
            self.state.pose_library.load(path)
            self.status_lbl.setText(f"Loaded: {path}")
