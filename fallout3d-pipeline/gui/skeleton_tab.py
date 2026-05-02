"""
Tab 5b — Skeleton
Bone hierarchy, rigid bone lengths, 3D visualization, and frame interpolation.
"""

import logging
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSplitter, QGroupBox, QSlider, QRadioButton, QButtonGroup,
    QCheckBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QSizePolicy,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, QTimer

from gui.main_window import AppState
from pipeline.skeleton_builder import (
    SkeletonBuilder, BONE_HIERARCHY, BONE_NAMES, _TRAVERSE_ORDER,
)

_logger = logging.getLogger(__name__)

try:
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False

# -----------------------------------------------------------------------
# Color helpers
# -----------------------------------------------------------------------

# Joint index → side string (34=Spine Mid, 35=Chest are center)
_JOINT_SIDE: dict[int, str] = {}
for _i in [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 7]:
    _JOINT_SIDE[_i] = "left"
for _i in [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 8]:
    _JOINT_SIDE[_i] = "right"

_SIDE_RGBA = {
    "left":   np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32),
    "right":  np.array([1.0, 0.2, 0.1, 1.0], dtype=np.float32),
    "center": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
}

_SIDE_QCOLOR = {
    "left":   QColor(100, 140, 255),
    "right":  QColor(255, 100, 80),
    "center": QColor(220, 220, 220),
}

def _joint_side(idx: int) -> str:
    return _JOINT_SIDE.get(idx, "center")

def _bone_rgba(child_idx: int) -> np.ndarray:
    return _SIDE_RGBA[_joint_side(child_idx)]


# -----------------------------------------------------------------------
# 3D Viewer
# -----------------------------------------------------------------------

class _SkeletonGLView(QWidget):
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
            self._scatter = None
            self._lines: list = []
            layout.addWidget(self._view)
        else:
            layout.addWidget(QLabel(
                "pyqtgraph / PyOpenGL not installed — 3D viewer disabled.\n"
                "Run: pip install pyqtgraph PyOpenGL"
            ))
            layout.addStretch()

    def show_pose(self, positions: np.ndarray):
        """positions: (36, 3) — indices 33/34/35 are virtual joints."""
        if not _GL_AVAILABLE:
            return

        if self._scatter is not None:
            self._view.removeItem(self._scatter)
            self._scatter = None
        for item in self._lines:
            self._view.removeItem(item)
        self._lines = []

        n_joints = positions.shape[0]
        colors = np.array([_bone_rgba(i) for i in range(n_joints)], dtype=np.float32)
        valid = ~np.all(positions == 0, axis=1)
        if valid.any():
            self._scatter = gl.GLScatterPlotItem(
                pos=positions[valid].astype(np.float32),
                color=colors[valid],
                size=8,
                pxMode=True,
            )
            self._view.addItem(self._scatter)

        for child_idx, parent_idx in BONE_HIERARCHY.items():
            if parent_idx is None:
                continue
            p = positions[parent_idx]
            c = positions[child_idx]
            if np.all(p == 0) or np.all(c == 0):
                continue
            pts = np.array([p, c], dtype=np.float32)
            col = np.array([_bone_rgba(child_idx)] * 2, dtype=np.float32)
            line = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True)
            self._view.addItem(line)
            self._lines.append(line)


# -----------------------------------------------------------------------
# Main Tab
# -----------------------------------------------------------------------

class SkeletonTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self._current_frame: int = 0
        self._interp_poses: np.ndarray | None = None   # (M, 34, 3) after interpolation

        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.character_updated.connect(self._on_char_updated)
        self.state.frame_changed.connect(self._on_frame_changed)

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())

        self._gl_view = _SkeletonGLView()
        splitter.addWidget(self._build_right_panel())

        splitter.setSizes([340, 760])

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(400)
        vbox = QVBoxLayout(panel)

        # ── Bone Lengths group ──────────────────────────────────────
        grp_bl = QGroupBox("Bone Lengths")
        vbl = QVBoxLayout(grp_bl)

        self._rb_frame0 = QRadioButton("Frame 0")
        self._rb_median = QRadioButton("Median")
        self._rb_manual = QRadioButton("Manual")
        self._rb_median.setChecked(True)

        self._rb_group = QButtonGroup(self)
        self._rb_group.addButton(self._rb_frame0, 0)
        self._rb_group.addButton(self._rb_median, 1)
        self._rb_group.addButton(self._rb_manual, 2)

        radio_row = QHBoxLayout()
        radio_row.addWidget(self._rb_frame0)
        radio_row.addWidget(self._rb_median)
        radio_row.addWidget(self._rb_manual)
        vbl.addLayout(radio_row)

        self._btn_calc = QPushButton("Calculate Bone Lengths")
        self._btn_calc.setStyleSheet("font-weight: bold; padding: 4px 8px;")
        self._btn_calc.clicked.connect(self._calculate)
        vbl.addWidget(self._btn_calc)

        # Bone length table
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Bone", "Length (px)"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setMinimumHeight(220)
        self._table.itemChanged.connect(self._on_table_item_changed)
        vbl.addWidget(self._table)

        self._rb_group.idToggled.connect(self._on_mode_changed)

        vbox.addWidget(grp_bl)

        # ── Rigid Bones group ───────────────────────────────────────
        grp_rb = QGroupBox("Rigid Bones")
        vrb = QVBoxLayout(grp_rb)
        self._chk_rigid = QCheckBox("Enable Rigid Constraints")
        self._chk_rigid.setChecked(True)
        vrb.addWidget(self._chk_rigid)
        vbox.addWidget(grp_rb)

        # ── Smoothing group ─────────────────────────────────────────
        grp_sm = QGroupBox("Smoothing")
        vsm = QVBoxLayout(grp_sm)

        self._chk_filter = QCheckBox("Low-pass Filter")
        self._chk_filter.setChecked(False)
        vsm.addWidget(self._chk_filter)

        row_sig = QHBoxLayout()
        row_sig.addWidget(QLabel("Sigma:"))
        self._sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self._sigma_slider.setRange(1, 10)   # value × 0.5 → [0.5 … 5.0]
        self._sigma_slider.setValue(3)       # default 1.5
        self._sigma_slider.setEnabled(False)
        row_sig.addWidget(self._sigma_slider)
        self._sigma_lbl = QLabel("1.5")
        self._sigma_lbl.setFixedWidth(30)
        row_sig.addWidget(self._sigma_lbl)
        vsm.addLayout(row_sig)

        self._chk_filter.toggled.connect(self._on_filter_changed)
        self._sigma_slider.valueChanged.connect(self._on_sigma_changed)
        vbox.addWidget(grp_sm)

        # ── Interpolation group ─────────────────────────────────────
        grp_int = QGroupBox("Interpolation")
        vint = QVBoxLayout(grp_int)

        row_n = QHBoxLayout()
        row_n.addWidget(QLabel("Insert N frames between each frame:"))
        self._spin_n = QSpinBox()
        self._spin_n.setRange(1, 10)
        self._spin_n.setValue(1)
        self._spin_n.setFixedWidth(56)
        row_n.addWidget(self._spin_n)
        vint.addLayout(row_n)

        self._btn_interp = QPushButton("Generate Interpolated Frames")
        self._btn_interp.clicked.connect(self._generate_interpolated)
        vint.addWidget(self._btn_interp)

        self._interp_lbl = QLabel("Result: —")
        vint.addWidget(self._interp_lbl)
        vbox.addWidget(grp_int)

        vbox.addStretch()
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)

        # Playback controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Frame:"))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider)
        ctrl.addWidget(self._slider)

        self._frame_lbl = QLabel("0 / 0")
        self._frame_lbl.setFixedWidth(60)
        ctrl.addWidget(self._frame_lbl)

        self._btn_play = QPushButton("▶ Play")
        self._btn_play.setCheckable(True)
        self._btn_play.setFixedWidth(70)
        self._btn_play.toggled.connect(self._on_play_toggled)
        ctrl.addWidget(self._btn_play)

        ctrl.addWidget(QLabel("FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 30)
        self._fps_spin.setValue(8)
        self._fps_spin.setFixedWidth(50)
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        ctrl.addWidget(self._fps_spin)

        self._status_lbl = QLabel("")
        ctrl.addWidget(self._status_lbl)
        ctrl.addStretch()
        vbox.addLayout(ctrl)

        self._gl_view = _SkeletonGLView()
        vbox.addWidget(self._gl_view, 1)

        return panel

    # ── Mode helpers ──────────────────────────────────────────────────

    def _mode_str(self) -> str:
        bid = self._rb_group.checkedId()
        return {0: "frame0", 1: "median", 2: "manual"}.get(bid, "median")

    def _on_mode_changed(self, bid: int, checked: bool):
        if not checked:
            return
        is_manual = bid == 2
        editable = (
            QAbstractItemView.EditTrigger.DoubleClicked
            if is_manual
            else QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setEditTriggers(editable)

    # ── Calculate ─────────────────────────────────────────────────────

    def _sigma_value(self) -> float | None:
        """Return current sigma if filter is enabled, else None."""
        if not self._chk_filter.isChecked():
            return None
        return self._sigma_slider.value() * 0.5

    def _calculate(self):
        char = self.state.current_character
        if char is None or char.skeleton_3d is None:
            self._status_lbl.setText("No 3D skeleton — run Reconstruction first.")
            return

        mode = self._mode_str()
        manual_lengths: dict[int, float] | None = None

        if mode == "manual":
            manual_lengths = self._read_table_lengths()

        sb = SkeletonBuilder()
        sb.build(
            char.skeleton_3d,
            mode=mode,
            manual_lengths=manual_lengths,
            lowpass_sigma=self._sigma_value(),
        )
        char.skeleton = sb

        self._interp_poses = None
        self._interp_lbl.setText("Result: —")
        self._populate_table(sb)
        self._update_slider()
        self._show_frame(self._current_frame)
        self.state.character_updated.emit(self.state.selected_idx)
        self._status_lbl.setText(f"Built ({mode}) — {sb.poses.shape[0]} frames.")

    def _read_table_lengths(self) -> dict[int, float]:
        lengths: dict[int, float] = {}
        for row in range(self._table.rowCount()):
            joint_item = self._table.item(row, 0)
            len_item   = self._table.item(row, 1)
            if joint_item is None or len_item is None:
                continue
            joint_idx = joint_item.data(Qt.ItemDataRole.UserRole)
            try:
                lengths[joint_idx] = float(len_item.text())
            except ValueError:
                pass
        return lengths

    # ── Table ─────────────────────────────────────────────────────────

    def _populate_table(self, sb: SkeletonBuilder):
        self._table.blockSignals(True)
        self._table.setRowCount(0)

        # Walk in BFS order so hierarchy is visually grouped
        rows = []
        for joint_idx in _TRAVERSE_ORDER:
            parent_idx = BONE_HIERARCHY.get(joint_idx)
            if parent_idx is None:
                continue   # root has no bone length
            length = sb.bone_lengths.get(joint_idx, 0.0)
            rows.append((joint_idx, BONE_NAMES.get(joint_idx, str(joint_idx)), length))

        self._table.setRowCount(len(rows))
        for row, (joint_idx, name, length) in enumerate(rows):
            side = _joint_side(joint_idx)
            bg   = _SIDE_QCOLOR[side]

            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.ItemDataRole.UserRole, joint_idx)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            name_item.setBackground(bg)
            self._table.setItem(row, 0, name_item)

            len_item = QTableWidgetItem(f"{length:.2f}")
            len_item.setData(Qt.ItemDataRole.UserRole, joint_idx)
            len_item.setBackground(bg)
            self._table.setItem(row, 1, len_item)

        self._table.blockSignals(False)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        # Only relevant in manual mode; rebuild on edit
        if self._mode_str() != "manual":
            return
        char = self.state.current_character
        if char is None or char.skeleton_3d is None:
            return
        manual_lengths = self._read_table_lengths()
        sb = SkeletonBuilder()
        sb.build(
            char.skeleton_3d,
            mode="manual",
            manual_lengths=manual_lengths,
            lowpass_sigma=self._sigma_value(),
        )
        char.skeleton = sb
        self._update_slider()
        self._show_frame(self._current_frame)

    # ── Smoothing callbacks ───────────────────────────────────────────

    def _on_filter_changed(self, checked: bool):
        self._sigma_slider.setEnabled(checked)
        self._recalculate_if_built()

    def _on_sigma_changed(self, value: int):
        self._sigma_lbl.setText(f"{value * 0.5:.1f}")
        if self._chk_filter.isChecked():
            self._recalculate_if_built()

    def _recalculate_if_built(self):
        char = self.state.current_character
        if char is not None and char.skeleton_3d is not None and char.skeleton is not None:
            self._calculate()

    # ── Interpolation ─────────────────────────────────────────────────

    def _generate_interpolated(self):
        char = self.state.current_character
        if char is None or char.skeleton is None:
            self._status_lbl.setText("Calculate bone lengths first.")
            return

        sb = char.skeleton
        n_orig = sb.poses.shape[0]
        n_insert = self._spin_n.value()

        frames_out = []
        for i in range(n_orig - 1):
            frames_out.append(sb.poses[i])
            for k in range(1, n_insert + 1):
                t = k / (n_insert + 1)
                frames_out.append(sb.interpolate(i, i + 1, t))
        frames_out.append(sb.poses[-1])

        self._interp_poses = np.array(frames_out)  # (M, 34, 3)
        n_result = len(frames_out)
        self._interp_lbl.setText(f"Result: {n_result} frames ({n_orig} orig + {n_result - n_orig} interp)")
        self._update_slider()
        self._show_frame(0)

    # ── Playback ──────────────────────────────────────────────────────

    def _on_play_toggled(self, playing: bool):
        if playing:
            self._btn_play.setText("⏸ Pause")
            fps = self._fps_spin.value()
            self._play_timer.start(max(1, 1000 // fps))
        else:
            self._btn_play.setText("▶ Play")
            self._play_timer.stop()

    def _on_fps_changed(self, fps: int):
        if self._play_timer.isActive():
            self._play_timer.setInterval(max(1, 1000 // fps))

    def _play_tick(self):
        n = self._slider.maximum() + 1
        if n <= 0:
            self._btn_play.setChecked(False)
            return
        next_frame = (self._current_frame + 1) % n
        self._slider.setValue(next_frame)

    def _on_slider(self, value: int):
        self._current_frame = value
        self._show_frame(value)

    # ── Frame display ─────────────────────────────────────────────────

    def _update_slider(self):
        char = self.state.current_character
        source = self._active_poses()
        if source is None:
            self._slider.setRange(0, 0)
            self._frame_lbl.setText("0 / 0")
            return
        n = source.shape[0]
        self._slider.setRange(0, max(0, n - 1))
        self._frame_lbl.setText(f"{min(self._current_frame + 1, n)} / {n}")

    def _active_poses(self) -> "np.ndarray | None":
        """Return interp poses if available, else the skeleton poses."""
        if self._interp_poses is not None:
            return self._interp_poses
        char = self.state.current_character
        if char and char.skeleton:
            return char.skeleton.poses
        return None

    def _show_frame(self, frame: int):
        source = self._active_poses()
        if source is None:
            return
        n = source.shape[0]
        frame = max(0, min(frame, n - 1))
        self._frame_lbl.setText(f"{frame + 1} / {n}")

        positions = source[frame]
        self._gl_view.show_pose(positions)

    # ── State change signals ──────────────────────────────────────────

    def _on_char_changed(self, idx: int):
        self._interp_poses = None
        self._interp_lbl.setText("Result: —")
        char = self.state.current_character
        if char and char.skeleton:
            self._populate_table(char.skeleton)
        else:
            self._table.setRowCount(0)
        self._current_frame = 0
        self._update_slider()
        self._show_frame(0)

    def _on_char_updated(self, idx: int):
        if idx == self.state.selected_idx:
            self._on_char_changed(idx)

    def _on_frame_changed(self, frame: int):
        # Sync with global frame only when not using interpolated data
        if self._interp_poses is None:
            self._slider.blockSignals(True)
            self._slider.setValue(frame)
            self._slider.blockSignals(False)
            self._current_frame = frame
            self._show_frame(frame)
