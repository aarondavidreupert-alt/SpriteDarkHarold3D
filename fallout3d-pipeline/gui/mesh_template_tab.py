"""
Tab 7c — Mesh Template
Pull the rigid skeleton from AppState (same as Tab 7b), generate a per-bone
ragdoll, adjust capsule radii interactively, and save as a template JSON.
"""

import os
import json
import logging
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QScrollArea, QSplitter, QFileDialog,
    QGroupBox, QFormLayout,
)
from PyQt6.QtCore import Qt

from gui.main_window import AppState
from gui.mesh_tab import MeshViewer3D
from pipeline.mesh_fitter import MeshFitter

_logger = logging.getLogger(__name__)

_BONE_NAMES = MeshFitter.RAGDOLL_BONE_NAMES   # 11 entries (Head + 10 capsules)


class MeshTemplateTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self._rest_pose: np.ndarray | None = None   # (33, 3) first frame
        self._ragdoll_verts: np.ndarray | None = None
        self._ragdoll_faces: np.ndarray | None = None
        self._radius_spins: dict[str, QDoubleSpinBox] = {}

        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.character_updated.connect(self._on_char_updated)

        self._on_char_changed(self.state.selected_idx)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([300, 900])

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # ── Step 1: Skeleton
        grp1 = QGroupBox("Step 1: Skeleton")
        v1 = QVBoxLayout(grp1)
        self.skeleton_lbl = QLabel("Skeleton: not loaded")
        v1.addWidget(self.skeleton_lbl)
        btn_use = QPushButton("← Use skeleton from Tab 5b")
        btn_use.clicked.connect(self._use_skeleton)
        v1.addWidget(btn_use)
        vbox.addWidget(grp1)

        # ── Step 2: Ragdoll
        grp2 = QGroupBox("Step 2: Ragdoll")
        v2 = QVBoxLayout(grp2)
        self.btn_generate = QPushButton("Generate Ragdoll")
        self.btn_generate.setEnabled(False)
        self.btn_generate.setStyleSheet("font-weight: bold; padding: 4px 8px;")
        self.btn_generate.clicked.connect(self._rebuild_ragdoll)
        v2.addWidget(self.btn_generate)
        self.ragdoll_lbl = QLabel("—")
        v2.addWidget(self.ragdoll_lbl)
        vbox.addWidget(grp2)

        # ── Step 3: Per-Bone Radii
        grp3 = QGroupBox("Step 3: Per-Bone Radii")
        form = QFormLayout(grp3)
        form.setSpacing(4)
        for name in _BONE_NAMES:
            spin = QDoubleSpinBox()
            spin.setRange(0.005, 0.5)
            spin.setSingleStep(0.001)
            spin.setDecimals(3)
            spin.setValue(0.045)
            spin.valueChanged.connect(self._rebuild_ragdoll)
            form.addRow(name, spin)
            self._radius_spins[name] = spin
        vbox.addWidget(grp3)

        # ── Step 4: Save
        grp4 = QGroupBox("Step 4: Save")
        v4 = QVBoxLayout(grp4)
        btn_save = QPushButton("Save Template JSON…")
        btn_save.clicked.connect(self._save_template)
        v4.addWidget(btn_save)
        vbox.addWidget(grp4)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: #88f; font-style: italic;")
        vbox.addWidget(self.status_lbl)

        vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMaximumWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _build_right_panel(self) -> QWidget:
        self._mesh_viewer = MeshViewer3D()
        return self._mesh_viewer

    # ------------------------------------------------------------------
    # Skeleton access (mirrors MeshBuilderTab exactly)
    # ------------------------------------------------------------------

    def _animation_skeleton(self) -> np.ndarray | None:
        char = self.state.current_character
        if char is None:
            return None
        if char.skeleton is not None and getattr(char.skeleton, "poses", None) is not None:
            return char.skeleton.poses[:, :33, :]
        return char.skeleton_3d

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def _use_skeleton(self):
        skel = self._animation_skeleton()
        if skel is None:
            self._set_status("No skeleton available — run triangulation first.")
            return

        self._rest_pose = skel[0]

        feet_mid = (skel[0, 27] + skel[0, 28]) * 0.5
        body_height = float(np.linalg.norm(skel[0, 0] - feet_mid))
        if body_height < 1e-6:
            body_height = 1.0
        base_r = body_height * 0.045

        for name, spin in self._radius_spins.items():
            spin.blockSignals(True)
            spin.setValue(base_r * 1.8 if name == "Head" else base_r)
            spin.blockSignals(False)

        self.skeleton_lbl.setText(f"Skeleton: {skel.shape[0]} frames")
        self.btn_generate.setEnabled(True)
        self._set_status(f"Skeleton ready ({skel.shape[0]} frames, body h≈{body_height:.3f}).")
        self._rebuild_ragdoll()

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    def _rebuild_ragdoll(self):
        if self._rest_pose is None:
            return
        per_bone = {name: spin.value() for name, spin in self._radius_spins.items()}
        try:
            verts, faces = MeshFitter.generate_ragdoll(
                self._rest_pose, per_bone_radii=per_bone
            )
        except Exception as exc:
            self._set_status(f"Ragdoll error: {exc}")
            _logger.error("Ragdoll build error: %s", exc)
            return
        self._ragdoll_verts = verts
        self._ragdoll_faces = faces
        self._mesh_viewer.set_mesh(verts, faces, None, 0)
        self.ragdoll_lbl.setText(f"{len(verts)} verts, {len(faces)} faces")

    # ------------------------------------------------------------------
    # Step 4
    # ------------------------------------------------------------------

    def _save_template(self):
        if self._ragdoll_verts is None or self._ragdoll_faces is None:
            self._set_status("Generate a ragdoll first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Template JSON", "", "JSON Files (*.json)"
        )
        if not path:
            return
        char = self.state.current_character
        template = {
            "version": 1,
            "source_character": char.name if char is not None else "",
            "bone_radii": {name: spin.value() for name, spin in self._radius_spins.items()},
            "ragdoll_verts": self._ragdoll_verts.tolist(),
            "ragdoll_faces": self._ragdoll_faces.tolist(),
        }
        try:
            with open(path, "w") as f:
                json.dump(template, f, indent=2)
            self._set_status(f"Saved → {os.path.basename(path)}")
        except Exception as exc:
            self._set_status(f"Save error: {exc}")
            _logger.error("Template save error: %s", exc)

    # ------------------------------------------------------------------
    # AppState signal handlers
    # ------------------------------------------------------------------

    def _on_char_changed(self, idx: int):
        char = self.state.current_character
        if char is None:
            self.skeleton_lbl.setText("Skeleton: not loaded")
            self.btn_generate.setEnabled(False)
            return
        skel = self._animation_skeleton()
        if skel is not None:
            self.skeleton_lbl.setText(
                f"Skeleton: {skel.shape[0]} frames available — click Use to load"
            )
        else:
            self.skeleton_lbl.setText("Skeleton: not loaded")

    def _on_char_updated(self, idx: int):
        if idx == self.state.selected_idx:
            self._on_char_changed(idx)

    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        self.status_lbl.setText(text)
        _logger.info("MeshTemplate: %s", text)
