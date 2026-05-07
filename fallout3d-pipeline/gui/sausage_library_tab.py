"""
Tab 8 — Sausage Library

Collects per-(character, animation) VoxelCarver snapshots and
weight-averages them into a single master voxel grid per bone.
"""

import logging
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QGroupBox, QFileDialog,
    QHeaderView,
)
from PyQt6.QtCore import Qt

from gui.main_window import AppState
from pipeline.sausage_library import SausageEntry, SausageLibrary

_logger = logging.getLogger(__name__)


class SausageLibraryTab(QWidget):

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._library = SausageLibrary()
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Add row ───────────────────────────────────────────────────
        grp_add = QGroupBox("Add Current Sausage")
        h = QHBoxLayout(grp_add)
        self.btn_add = QPushButton("Add Current Sausage")
        self.btn_add.clicked.connect(self._add_current)
        h.addWidget(self.btn_add)
        h.addWidget(QLabel("Character:"))
        self.txt_char = QLineEdit()
        self.txt_char.setMaximumWidth(120)
        h.addWidget(self.txt_char)
        h.addWidget(QLabel("Anim:"))
        self.txt_anim = QLineEdit()
        self.txt_anim.setMaximumWidth(120)
        h.addWidget(self.txt_anim)
        h.addWidget(QLabel("Weight:"))
        self.spin_weight = QDoubleSpinBox()
        self.spin_weight.setRange(0.0, 100.0)
        self.spin_weight.setValue(1.0)
        self.spin_weight.setSingleStep(0.1)
        h.addWidget(self.spin_weight)
        h.addStretch()
        root.addWidget(grp_add)

        # ── Library table ─────────────────────────────────────────────
        grp_tbl = QGroupBox("Library Entries")
        v_tbl = QVBoxLayout(grp_tbl)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["#", "Character", "Animation", "Weight", "Bones", ""])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        v_tbl.addWidget(self.table)
        root.addWidget(grp_tbl, 1)

        # ── Build master ──────────────────────────────────────────────
        grp_build = QGroupBox("Build Master Sausage")
        gh = QHBoxLayout(grp_build)
        gh.addWidget(QLabel("Filter Character:"))
        self.txt_filter = QLineEdit()
        self.txt_filter.setMaximumWidth(160)
        self.txt_filter.setPlaceholderText("(empty = all)")
        gh.addWidget(self.txt_filter)
        gh.addWidget(QLabel("Threshold:"))
        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.05, 0.95)
        self.spin_thr.setValue(0.30)
        self.spin_thr.setSingleStep(0.05)
        gh.addWidget(self.spin_thr)
        gh.addWidget(QLabel("Resolution:"))
        self.spin_res = QDoubleSpinBox()
        self.spin_res.setRange(8, 128)
        self.spin_res.setDecimals(0)
        self.spin_res.setValue(32)
        gh.addWidget(self.spin_res)
        self.btn_build = QPushButton("Build Master Sausage")
        self.btn_build.setStyleSheet("font-weight: bold;")
        self.btn_build.clicked.connect(self._build_master)
        gh.addWidget(self.btn_build)
        gh.addStretch()
        root.addWidget(grp_build)

        # ── Status ────────────────────────────────────────────────────
        self.status_lbl = QLabel("—")
        self.status_lbl.setStyleSheet("color:#88f; font-style: italic;")
        self.status_lbl.setWordWrap(True)
        root.addWidget(self.status_lbl)

        # ── Save / load library ───────────────────────────────────────
        h_io = QHBoxLayout()
        btn_save = QPushButton("Save Library…")
        btn_save.clicked.connect(self._save_library)
        h_io.addWidget(btn_save)
        btn_load = QPushButton("Load Library…")
        btn_load.clicked.connect(self._load_library)
        h_io.addWidget(btn_load)
        h_io.addStretch()
        root.addLayout(h_io)

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def _add_current(self):
        char = self.state.current_character
        if char is None:
            self._set_status("No character loaded.")
            return
        carver = getattr(char, "voxel_carver", None)
        if carver is None or not getattr(carver, "sausages", None):
            self._set_status("Current character has no VoxelCarver — "
                             "carve it in Tab 7c first.")
            return
        char_name = self.txt_char.text().strip() or getattr(char, "name", "char")
        anim_name = self.txt_anim.text().strip() or "default"
        weight    = float(self.spin_weight.value())
        entry = SausageEntry.from_carver(carver, char_name, anim_name, weight)
        self._library.add(entry)
        self._refresh_table()
        self._set_status(f"Added {char_name}/{anim_name} (w={weight}, "
                         f"{entry.n_bones} bones).")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._library.entries))
        for i, e in enumerate(self._library.entries):
            def _ro(text):
                it = QTableWidgetItem(str(text))
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                return it
            self.table.setItem(i, 0, _ro(i))
            self.table.setItem(i, 1, _ro(e.character))
            self.table.setItem(i, 2, _ro(e.animation))
            self.table.setItem(i, 3, QTableWidgetItem(f"{e.weight:.3f}"))
            self.table.setItem(i, 4, _ro(e.n_bones))
            btn = QPushButton("✕")
            btn.setMaximumWidth(28)
            btn.clicked.connect(lambda _checked, idx=i: self._remove_row(idx))
            self.table.setCellWidget(i, 5, btn)
        self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem):
        if item.column() != 3:
            return
        idx = item.row()
        if not (0 <= idx < len(self._library.entries)):
            return
        try:
            self._library.entries[idx].weight = float(item.text())
        except ValueError:
            item.setText(f"{self._library.entries[idx].weight:.3f}")

    def _remove_row(self, idx: int):
        self._library.remove(idx)
        self._refresh_table()
        self._set_status(f"Removed entry {idx}.")

    # ------------------------------------------------------------------
    # Build master
    # ------------------------------------------------------------------

    def _build_master(self):
        if not self._library.entries:
            self._set_status("Library is empty — add an entry first.")
            return
        flt = self.txt_filter.text().strip() or None
        thr = float(self.spin_thr.value())
        res = int(self.spin_res.value())
        try:
            master = self._library.build_master(
                filter_character=flt, resolution=res, threshold=thr)
        except Exception as exc:
            self._set_status(f"Build error: {exc}")
            return
        if not master:
            self._set_status("Master is empty — check filter / threshold.")
            return
        self.state.master_sausage = master
        densities = [float(g.mean()) for g in master.values()]
        avg = float(np.mean(densities)) if densities else 0.0
        self._set_status(
            f"Master built — {len(master)} bones, avg density {avg:.3f} "
            f"@ res={res}, thr={thr:.2f}.")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_library(self):
        if not self._library.entries:
            self._set_status("Nothing to save.")
            return
        d = QFileDialog.getExistingDirectory(self, "Choose library directory")
        if not d:
            return
        try:
            self._library.save_library(d)
        except Exception as exc:
            self._set_status(f"Save error: {exc}")
            return
        self._set_status(f"Library saved ({len(self._library.entries)} entries) → {d}.")

    def _load_library(self):
        d = QFileDialog.getExistingDirectory(self, "Choose library directory")
        if not d:
            return
        try:
            self._library.load_library(d)
        except Exception as exc:
            self._set_status(f"Load error: {exc}")
            return
        self._refresh_table()
        self._set_status(f"Loaded library — {len(self._library.entries)} entries total.")

    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        self.status_lbl.setText(text)
        _logger.info("SausageLibrary: %s", text)
