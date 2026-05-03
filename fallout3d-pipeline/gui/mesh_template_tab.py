"""
Tab 7c — Mesh Template
Load a saved skeleton JSON, generate a per-bone ragdoll, adjust capsule
radii interactively, and save the result as a mesh template JSON.
"""

import os
import json
import logging
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QScrollArea, QSplitter, QFileDialog, QFrame,
    QGroupBox, QFormLayout,
)
from PyQt6.QtCore import Qt

from gui.main_window import AppState
from gui.mesh_tab import MeshViewer3D
from pipeline.mesh_fitter import _make_sphere, _make_capsule
from pipeline.skeleton_builder import SkeletonBuilder

_logger = logging.getLogger(__name__)

# Bone names in the same order as MeshFitter.generate_ragdoll
_RAGDOLL_BONE_NAMES = [
    "Head",
    "Neck",
    "L Upper Arm",
    "R Upper Arm",
    "L Forearm",
    "R Forearm",
    "Torso",
    "L Thigh",
    "R Thigh",
    "L Shin",
    "R Shin",
]

_DEFAULT_RADIUS_SCALE = 0.045


def _build_ragdoll_per_bone(
    skeleton: np.ndarray,
    radii: dict,
    capsule_segments: int = 8,
    capsule_rings: int = 4,
):
    """Build a ragdoll mesh from a (33, 3) skeleton with per-bone radii.

    The bone layout mirrors MeshFitter.generate_ragdoll: one sphere for
    the head plus one capsule per named bone.
    """
    sk = np.asarray(skeleton, dtype=float)

    shoulders_mid = (sk[11] + sk[12]) * 0.5
    hips_mid      = (sk[23] + sk[24]) * 0.5

    # (name, p0, p1) — capsule endpoint pairs
    capsule_defs = [
        ("Neck",        sk[0],         shoulders_mid),
        ("L Upper Arm", sk[11],        sk[13]),
        ("R Upper Arm", sk[12],        sk[14]),
        ("L Forearm",   sk[13],        sk[15]),
        ("R Forearm",   sk[14],        sk[16]),
        ("Torso",       shoulders_mid, hips_mid),
        ("L Thigh",     sk[23],        sk[25]),
        ("R Thigh",     sk[24],        sk[26]),
        ("L Shin",      sk[25],        sk[27]),
        ("R Shin",      sk[26],        sk[28]),
    ]

    all_verts = []
    all_faces = []
    offset = 0

    head_r = float(radii.get("Head", _DEFAULT_RADIUS_SCALE))
    hv, hf = _make_sphere(
        sk[0], head_r,
        segments=capsule_segments,
        rings=max(4, capsule_segments),
    )
    all_verts.append(hv)
    all_faces.append(hf + offset)
    offset += len(hv)

    for name, p0, p1 in capsule_defs:
        r = float(radii.get(name, _DEFAULT_RADIUS_SCALE))
        cv, cf = _make_capsule(p0, p1, r, segments=capsule_segments, rings=capsule_rings)
        all_verts.append(cv)
        all_faces.append(cf + offset)
        offset += len(cv)

    return (
        np.concatenate(all_verts, axis=0).astype(np.float32),
        np.concatenate(all_faces, axis=0).astype(np.int32),
    )


class MeshTemplateTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self._skeleton: SkeletonBuilder | None = None
        self._skeleton_path: str = ""
        self._body_height: float = 1.0
        self._mesh_verts: np.ndarray | None = None
        self._mesh_faces: np.ndarray | None = None
        self._radius_spinboxes: dict[str, QDoubleSpinBox] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([300, 900])

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # ── Load skeleton
        grp_load = QGroupBox("Skeleton")
        v_load = QVBoxLayout(grp_load)

        self.btn_load = QPushButton("Load Skeleton JSON…")
        self.btn_load.clicked.connect(self._load_skeleton)
        v_load.addWidget(self.btn_load)

        self.lbl_file = QLabel("No skeleton loaded.")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color: #aaa; font-size: 11px;")
        v_load.addWidget(self.lbl_file)

        self.btn_generate = QPushButton("Generate Ragdoll")
        self.btn_generate.setEnabled(False)
        self.btn_generate.setStyleSheet("font-weight: bold; padding: 4px 8px;")
        self.btn_generate.clicked.connect(self._rebuild_ragdoll)
        v_load.addWidget(self.btn_generate)

        vbox.addWidget(grp_load)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        vbox.addWidget(sep1)

        # ── Per-bone radii
        grp_radii = QGroupBox("Per-Bone Capsule Radius")
        form = QFormLayout(grp_radii)
        form.setSpacing(4)
        for name in _RAGDOLL_BONE_NAMES:
            spin = QDoubleSpinBox()
            spin.setRange(0.005, 0.5)
            spin.setSingleStep(0.001)
            spin.setDecimals(3)
            spin.setValue(_DEFAULT_RADIUS_SCALE)
            spin.valueChanged.connect(self._rebuild_ragdoll)
            form.addRow(name, spin)
            self._radius_spinboxes[name] = spin
        vbox.addWidget(grp_radii)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        vbox.addWidget(sep2)

        # ── Save
        self.btn_save = QPushButton("Save Template JSON…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_template)
        vbox.addWidget(self.btn_save)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #88f; font-style: italic;")
        vbox.addWidget(self.lbl_status)

        vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFixedWidth(280)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _build_right_panel(self) -> QWidget:
        self._viewer = MeshViewer3D()
        return self._viewer

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _load_skeleton(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Skeleton JSON", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            self._skeleton = SkeletonBuilder.load(path)
        except Exception as exc:
            self._set_status(f"Load error: {exc}")
            _logger.error("Skeleton load failed: %s", exc)
            return

        self._skeleton_path = path
        self.lbl_file.setText(os.path.basename(path))
        self.btn_generate.setEnabled(True)
        self.btn_save.setEnabled(True)

        # Derive body height from bind_pose for default radii
        if self._skeleton.bind_pose is not None:
            bp = self._skeleton.bind_pose  # (36, 3)
            feet_mid = (bp[27] + bp[28]) * 0.5
            h = float(np.linalg.norm(bp[0] - feet_mid))
            self._body_height = h if h > 1e-6 else 1.0
        else:
            self._body_height = 1.0

        default_r = self._body_height * _DEFAULT_RADIUS_SCALE
        for spin in self._radius_spinboxes.values():
            spin.blockSignals(True)
            spin.setValue(default_r)
            spin.blockSignals(False)

        self._set_status(f"Skeleton loaded (body height ≈ {self._body_height:.3f}).")
        self._rebuild_ragdoll()

    def _get_radii(self) -> dict:
        return {name: spin.value() for name, spin in self._radius_spinboxes.items()}

    def _rebuild_ragdoll(self):
        if self._skeleton is None:
            return
        if self._skeleton.bind_pose is None:
            self._set_status("Skeleton has no bind_pose.")
            return

        skeleton_33 = self._skeleton.bind_pose[:33]
        try:
            verts, faces = _build_ragdoll_per_bone(skeleton_33, self._get_radii())
        except Exception as exc:
            self._set_status(f"Ragdoll error: {exc}")
            _logger.error("Ragdoll build error: %s", exc)
            return

        self._mesh_verts = verts
        self._mesh_faces = faces
        self._viewer.set_mesh(verts, faces)
        self._set_status(f"Ragdoll — {len(verts)} verts, {len(faces)} faces.")

    def _save_template(self):
        if self._mesh_verts is None or self._mesh_faces is None:
            self._set_status("Generate a ragdoll first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Template JSON", "", "JSON Files (*.json)"
        )
        if not path:
            return

        template = {
            "version": 1,
            "source_skeleton": self._skeleton_path,
            "bone_radii": self._get_radii(),
            "verts": self._mesh_verts.tolist(),
            "faces": self._mesh_faces.tolist(),
        }
        try:
            with open(path, "w") as f:
                json.dump(template, f, indent=2)
            self._set_status(f"Saved → {os.path.basename(path)}")
        except Exception as exc:
            self._set_status(f"Save error: {exc}")
            _logger.error("Template save error: %s", exc)

    def _set_status(self, text: str):
        self.lbl_status.setText(text)
        _logger.info("MeshTemplate: %s", text)
