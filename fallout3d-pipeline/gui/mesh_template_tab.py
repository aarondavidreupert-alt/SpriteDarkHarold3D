"""
Tab 7c — Mesh Template
Pull a rigid skeleton from AppState, generate a per-bone ragdoll, edit
capsule radii either by hand or by silhouette-fitting against the sprite
masks, and save the result as a template JSON.

Includes a projection preview + play controls (mirrored from Tab 7b) and
an optional visual-hull carve pass that trims the ragdoll mesh by the
intersection of the per-direction silhouette extrusions.
"""

import os
import json
import logging
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QScrollArea, QSplitter, QFileDialog,
    QGroupBox, QFormLayout, QSlider, QSpinBox, QCheckBox,
    QProgressBar, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, QTimer, QObject, pyqtSignal

from gui.main_window import AppState
from gui.mesh_tab import MeshViewer3D
from gui.mesh_builder_tab import ProjectionPreview
from pipeline.mesh_fitter import MeshFitter

_logger = logging.getLogger(__name__)

_BONE_NAMES = MeshFitter.RAGDOLL_BONE_NAMES   # 11 entries (Head + 10 capsules)


# -----------------------------------------------------------------------
# Rasterisation helper — used by both the optimiser and the preview overlay
# -----------------------------------------------------------------------

def rasterise_capsules_2d(
    verts: np.ndarray,
    faces: np.ndarray,
    cam_setup,
    dir_idx: int,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterise mesh triangles into a binary mask of shape (H, W)."""
    H, W = mask_shape
    cam_w, cam_h = cam_setup.image_size
    bp = cam_setup.back_project_points(verts)            # list of 6 × (V, 3)
    pts = bp[dir_idx][:, :2].copy()
    pts[:, 0] *= W / cam_w
    pts[:, 1] *= H / cam_h
    pts_px = pts.astype(np.int32)

    out = np.zeros((H, W), dtype=np.uint8)
    if len(faces) == 0:
        return out.astype(bool)
    triangles = pts_px[faces]                            # (F, 3, 2)
    cv2.fillPoly(out, triangles, 1)
    return out.astype(bool)


# -----------------------------------------------------------------------
# Silhouette-fit worker
# -----------------------------------------------------------------------

class SilhouetteFitWorker(QObject):
    progress = pyqtSignal(int, int)     # step, total
    finished = pyqtSignal(dict)         # {bone_name: radius}
    error    = pyqtSignal(str)

    def __init__(
        self,
        skel_frames: np.ndarray,        # (N, 33, 3)
        masks: np.ndarray,              # (D, F, H, W) bool
        cam_setup,
        radii_init: dict[str, float],
        max_iter: int = 10,
    ):
        super().__init__()
        self.skel_frames = skel_frames
        self.masks = masks
        self.cam_setup = cam_setup
        self.radii_init = radii_init
        self.max_iter = max_iter
        self._step = 0

    def run(self):
        try:
            from scipy.optimize import minimize
        except Exception as exc:
            self.error.emit(f"scipy not available: {exc}")
            return

        bone_names = list(self.radii_init.keys())
        x0 = np.log(np.array([self.radii_init[b] for b in bone_names], dtype=float))

        D, F = self.masks.shape[:2]
        # Sample every 3rd frame, capped at 20 frames
        all_frames = list(range(0, F, 3))
        if len(all_frames) > 20:
            stride = max(1, len(all_frames) // 20)
            all_frames = all_frames[::stride][:20]
        if not all_frames:
            all_frames = [0]
        n_frames = min(self.skel_frames.shape[0], F)
        frames_idx = [f for f in all_frames if f < n_frames]
        if not frames_idx:
            self.error.emit("No usable frames for fitting.")
            return

        H, W = self.masks.shape[2:]
        target_per = [self.masks[:, f] for f in frames_idx]   # list of (D, H, W)

        def loss(log_radii):
            radii_dict = {b: float(np.exp(log_radii[i])) for i, b in enumerate(bone_names)}
            total = 0.0
            for fi, f in enumerate(frames_idx):
                try:
                    verts, faces = MeshFitter.generate_ragdoll(
                        self.skel_frames[f], per_bone_radii=radii_dict
                    )
                except Exception:
                    return 1e6
                tgt_dir = target_per[fi]
                for d in range(D):
                    proj = rasterise_capsules_2d(verts, faces, self.cam_setup, d, (H, W))
                    inter = np.logical_and(proj, tgt_dir[d]).sum()
                    union = np.logical_or(proj, tgt_dir[d]).sum()
                    iou = inter / max(int(union), 1)
                    total += (1.0 - float(iou))
            return total

        # Crude progress: scipy doesn't expose iteration count cleanly, so
        # emit one tick per outer iteration via callback + initial signal.
        n_total = self.max_iter
        self._step = 0
        self.progress.emit(0, n_total)

        def cb(xk):
            self._step += 1
            self.progress.emit(min(self._step, n_total), n_total)

        # Bounds on log-radius: roughly [0.001, 0.5]
        bounds = [(np.log(0.001), np.log(0.5))] * len(bone_names)

        try:
            res = minimize(
                loss, x0, method="L-BFGS-B",
                bounds=bounds,
                callback=cb,
                options={"maxiter": self.max_iter, "ftol": 1e-3, "eps": 0.05},
            )
        except Exception as exc:
            self.error.emit(f"Optimisation failed: {exc}")
            return

        radii = {b: float(np.exp(res.x[i])) for i, b in enumerate(bone_names)}
        self.progress.emit(n_total, n_total)
        self.finished.emit(radii)


# -----------------------------------------------------------------------
# Main tab
# -----------------------------------------------------------------------

class MeshTemplateTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        self._skel_frames: np.ndarray | None = None     # (N, 33, 3)
        self._rest_pose:   np.ndarray | None = None     # (33, 3) — skel[0]
        self._current_frame: int = 0
        self._ragdoll_verts: np.ndarray | None = None
        self._ragdoll_faces: np.ndarray | None = None
        self._radius_spins:  dict[str, QDoubleSpinBox] = {}
        self._carved: bool = False

        self._masks_cache: np.ndarray | None = None    # (D, F, H, W) bool

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        self._fit_thread: QThread | None = None
        self._fit_worker: SilhouetteFitWorker | None = None

        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.character_updated.connect(self._on_char_updated)

        self._on_char_changed(self.state.selected_idx)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([320, 880])

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
        self.btn_generate.clicked.connect(self._regenerate_from_rest)
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

        # ── Step 4: Animate
        grp4 = QGroupBox("Step 4: Animate")
        v4 = QVBoxLayout(grp4)
        row_fr = QHBoxLayout()
        row_fr.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        row_fr.addWidget(self.frame_slider)
        self.frame_lbl = QLabel("0 / 0")
        self.frame_lbl.setFixedWidth(60)
        row_fr.addWidget(self.frame_lbl)
        v4.addLayout(row_fr)

        row_p = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._on_play_toggled)
        row_p.addWidget(self.btn_play)
        row_p.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        row_p.addWidget(self.fps_spin)
        row_p.addStretch()
        v4.addLayout(row_p)
        vbox.addWidget(grp4)

        # ── Step 5: Shave to Silhouette
        grp5 = QGroupBox("Step 5: Shave to Silhouette")
        v5 = QVBoxLayout(grp5)
        self.btn_fit = QPushButton("Run Silhouette Fit (A)")
        self.btn_fit.clicked.connect(self._run_silhouette_fit)
        v5.addWidget(self.btn_fit)
        self.btn_carve = QPushButton("Carve Visual Hull (B)")
        self.btn_carve.setEnabled(False)
        self.btn_carve.clicked.connect(self._carve_hull)
        v5.addWidget(self.btn_carve)
        self.fit_progress = QProgressBar()
        self.fit_progress.setRange(0, 100)
        self.fit_progress.setValue(0)
        self.fit_progress.setVisible(False)
        v5.addWidget(self.fit_progress)
        self.fit_status_lbl = QLabel("—")
        self.fit_status_lbl.setWordWrap(True)
        v5.addWidget(self.fit_status_lbl)
        vbox.addWidget(grp5)

        # ── Step 6: Overlays
        grp6 = QGroupBox("Step 6: Overlays")
        v6 = QVBoxLayout(grp6)
        self.chk_mask = QCheckBox("Show silhouette mask overlay")
        self.chk_mask.toggled.connect(self._update_projection)
        v6.addWidget(self.chk_mask)
        self.chk_wire = QCheckBox("Show fitted ragdoll overlay")
        self.chk_wire.setChecked(True)
        self.chk_wire.toggled.connect(self._update_projection)
        v6.addWidget(self.chk_wire)
        self.chk_skel = QCheckBox("Show skeleton joints")
        self.chk_skel.toggled.connect(self._update_projection)
        v6.addWidget(self.chk_skel)
        row_d = QHBoxLayout()
        row_d.addWidget(QLabel("Direction:"))
        self.dir_spin = QSpinBox()
        self.dir_spin.setRange(0, 5)
        self.dir_spin.valueChanged.connect(self._update_projection)
        row_d.addWidget(self.dir_spin)
        row_d.addStretch()
        v6.addLayout(row_d)
        vbox.addWidget(grp6)

        # ── Step 7: Save
        grp7 = QGroupBox("Step 7: Save")
        v7 = QVBoxLayout(grp7)
        btn_save = QPushButton("Save Template JSON…")
        btn_save.clicked.connect(self._save_template)
        v7.addWidget(btn_save)
        vbox.addWidget(grp7)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: #88f; font-style: italic;")
        vbox.addWidget(self.status_lbl)

        vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMaximumWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _build_right_panel(self) -> QWidget:
        panel = QSplitter(Qt.Orientation.Vertical)
        self._mesh_viewer = MeshViewer3D()
        panel.addWidget(self._mesh_viewer)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(2, 2, 2, 2)
        self._projection = ProjectionPreview()
        self._projection.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        bl.addWidget(self._projection, 1)
        panel.addWidget(bottom)

        panel.setSizes([520, 320])
        return panel

    # ------------------------------------------------------------------
    # Skeleton + camera helpers (mirror MeshBuilderTab)
    # ------------------------------------------------------------------

    def _animation_skeleton(self) -> np.ndarray | None:
        char = self.state.current_character
        if char is None:
            return None
        if char.skeleton is not None and getattr(char.skeleton, "poses", None) is not None:
            return char.skeleton.poses[:, :33, :]
        return char.skeleton_3d

    def _get_camera_setup(self, char=None):
        p = self.parent()
        while p is not None:
            if hasattr(p, "tab_recon"):
                return p.tab_recon._triangulator.camera_setup
            p = p.parent()
        if char is None:
            char = self.state.current_character
        if char is None or char.frames is None or char.frames.size == 0:
            return None
        from pipeline.pose_triangulator import PoseTriangulator
        h, w = char.frames[0, 0].shape[:2]
        return PoseTriangulator(image_size=(w, h)).camera_setup

    # ------------------------------------------------------------------
    # Step 1: Use skeleton
    # ------------------------------------------------------------------

    def _use_skeleton(self):
        skel = self._animation_skeleton()
        if skel is None:
            self._set_status("No skeleton available — run triangulation first.")
            return

        self._skel_frames = skel
        self._rest_pose = skel[0]
        self._current_frame = 0
        self._carved = False
        self._masks_cache = None

        feet_mid = (skel[0, 27] + skel[0, 28]) * 0.5
        body_height = float(np.linalg.norm(skel[0, 0] - feet_mid))
        if body_height < 1e-6:
            body_height = 1.0
        base_r = body_height * 0.045

        for name, spin in self._radius_spins.items():
            spin.blockSignals(True)
            spin.setValue(base_r * 1.8 if name == "Head" else base_r)
            spin.blockSignals(False)

        n = skel.shape[0]
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, n - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.frame_lbl.setText(f"1 / {n}")

        self.skeleton_lbl.setText(f"Skeleton: {n} frames")
        self.btn_generate.setEnabled(True)
        self._set_status(f"Skeleton ready ({n} frames, body h≈{body_height:.3f}).")
        self._rebuild_ragdoll()

    # ------------------------------------------------------------------
    # Step 2/3: Ragdoll generation
    # ------------------------------------------------------------------

    def _current_skeleton_frame(self) -> np.ndarray | None:
        if self._skel_frames is None:
            return self._rest_pose
        f = max(0, min(self._current_frame, self._skel_frames.shape[0] - 1))
        return self._skel_frames[f]

    def _radii_dict(self) -> dict[str, float]:
        return {name: spin.value() for name, spin in self._radius_spins.items()}

    def _regenerate_from_rest(self):
        """The 'Generate Ragdoll' button: reset carve flag and rebuild."""
        self._carved = False
        self._rebuild_ragdoll()

    def _rebuild_ragdoll(self):
        if self._carved:
            # Carved geometry is destructive — preserve it until the user
            # clicks 'Generate Ragdoll' again.
            return
        sk = self._current_skeleton_frame()
        if sk is None:
            return
        try:
            verts, faces = MeshFitter.generate_ragdoll(
                sk, per_bone_radii=self._radii_dict()
            )
        except Exception as exc:
            self._set_status(f"Ragdoll error: {exc}")
            _logger.error("Ragdoll build error: %s", exc)
            return
        self._ragdoll_verts = verts
        self._ragdoll_faces = faces
        self._mesh_viewer.set_mesh(verts, faces, None, 0)
        self.ragdoll_lbl.setText(f"{len(verts)} verts, {len(faces)} faces")
        self._update_projection()

    # ------------------------------------------------------------------
    # Step 4: Animate
    # ------------------------------------------------------------------

    def _on_frame_slider(self, val: int):
        self._current_frame = val
        if self._skel_frames is not None:
            n = self._skel_frames.shape[0]
            self.frame_lbl.setText(f"{val + 1} / {n}")
        self._rebuild_ragdoll()

    def _on_play_toggled(self, playing: bool):
        if playing:
            self.btn_play.setText("⏸ Pause")
            self._play_timer.start(max(1, 1000 // self.fps_spin.value()))
        else:
            self.btn_play.setText("▶ Play")
            self._play_timer.stop()

    def _on_fps_changed(self, fps: int):
        if self._play_timer.isActive():
            self._play_timer.setInterval(max(1, 1000 // fps))

    def _advance_frame(self):
        if self._skel_frames is None or self._skel_frames.shape[0] == 0:
            self.btn_play.setChecked(False)
            return
        n = self._skel_frames.shape[0]
        nxt = (self._current_frame + 1) % n
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(nxt)
        self.frame_slider.blockSignals(False)
        self._current_frame = nxt
        self.frame_lbl.setText(f"{nxt + 1} / {n}")
        self._rebuild_ragdoll()

    # ------------------------------------------------------------------
    # Silhouette mask extraction
    # ------------------------------------------------------------------

    def _extract_silhouette_masks(self, char) -> np.ndarray:
        """Return (D, F, H, W) bool array."""
        frames = char.frames
        D, F, H, W = frames.shape[:4]
        masks = np.zeros((D, F, H, W), dtype=bool)
        for d in range(D):
            for f in range(F):
                spr = frames[d, f]
                if spr.ndim == 3 and spr.shape[-1] == 4:
                    masks[d, f] = spr[..., 3] > 10
                else:
                    rgb = spr[..., :3] if spr.shape[-1] >= 3 else spr
                    masks[d, f] = np.any(rgb > 10, axis=-1)
        return masks

    # ------------------------------------------------------------------
    # Step 5A: Silhouette fit
    # ------------------------------------------------------------------

    def _run_silhouette_fit(self):
        char = self.state.current_character
        if char is None or char.frames is None:
            self._set_status("No character loaded.")
            return
        if self._skel_frames is None:
            self._set_status("Click 'Use skeleton from Tab 5b' first.")
            return
        cam = self._get_camera_setup(char)
        if cam is None:
            self._set_status("No camera setup available.")
            return

        if self._masks_cache is None:
            try:
                self._masks_cache = self._extract_silhouette_masks(char)
            except Exception as exc:
                self._set_status(f"Mask extraction error: {exc}")
                return

        radii_init = self._radii_dict()

        self._fit_worker = SilhouetteFitWorker(
            self._skel_frames, self._masks_cache, cam, radii_init,
        )
        self._fit_thread = QThread(self)
        self._fit_worker.moveToThread(self._fit_thread)
        self._fit_thread.started.connect(self._fit_worker.run)
        self._fit_worker.progress.connect(self._on_fit_progress)
        self._fit_worker.finished.connect(self._on_fit_finished)
        self._fit_worker.error.connect(self._on_fit_error)

        self.fit_progress.setVisible(True)
        self.fit_progress.setValue(0)
        self.btn_fit.setEnabled(False)
        self.fit_status_lbl.setText("Fitting…")
        self._fit_thread.start()

    def _on_fit_progress(self, step: int, total: int):
        if total > 0:
            self.fit_progress.setValue(int(100 * step / total))

    def _on_fit_finished(self, radii: dict):
        if self._fit_thread is not None:
            self._fit_thread.quit()
        self.fit_progress.setVisible(False)
        self.btn_fit.setEnabled(True)

        for name, r in radii.items():
            spin = self._radius_spins.get(name)
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(float(np.clip(r, 0.005, 0.5)))
                spin.blockSignals(False)

        self._carved = False
        self._rebuild_ragdoll()
        self.btn_carve.setEnabled(True)
        rs = ", ".join(f"{n}={r:.3f}" for n, r in radii.items())
        self.fit_status_lbl.setText(f"Fit done — {rs}")
        self._set_status("Silhouette fit complete.")

    def _on_fit_error(self, msg: str):
        if self._fit_thread is not None:
            self._fit_thread.quit()
        self.fit_progress.setVisible(False)
        self.btn_fit.setEnabled(True)
        self.fit_status_lbl.setText(f"Error: {msg}")
        self._set_status(f"Fit error: {msg}")

    # ------------------------------------------------------------------
    # Step 5B: Visual hull carve
    # ------------------------------------------------------------------

    def _carve_hull(self):
        char = self.state.current_character
        if char is None or self._ragdoll_verts is None or self._ragdoll_faces is None:
            self._set_status("Generate a ragdoll first.")
            return
        if self._masks_cache is None:
            try:
                self._masks_cache = self._extract_silhouette_masks(char)
            except Exception as exc:
                self._set_status(f"Mask extraction error: {exc}")
                return
        cam = self._get_camera_setup(char)
        if cam is None:
            self._set_status("No camera setup available.")
            return

        try:
            verts_c, faces_c = self._carve_visual_hull(
                self._ragdoll_verts, self._ragdoll_faces, self._masks_cache, cam,
            )
        except Exception as exc:
            self._set_status(f"Carve error: {exc}")
            _logger.error("Carve error: %s", exc)
            return

        self._ragdoll_verts = verts_c
        self._ragdoll_faces = faces_c
        self._carved = True
        self._mesh_viewer.set_mesh(verts_c, faces_c, None, 0)
        self.ragdoll_lbl.setText(f"{len(verts_c)} verts, {len(faces_c)} faces (carved)")
        self._update_projection()
        self._set_status(f"Carve done — {len(verts_c)} verts, {len(faces_c)} faces.")

    @staticmethod
    def _carve_visual_hull(verts, faces, masks, cam_setup):
        """Trim faces whose vertices fall outside the union silhouette in any direction."""
        D = masks.shape[0]
        union_masks = [np.any(masks[d], axis=0) for d in range(D)]    # (D, H, W)
        cam_w, cam_h = cam_setup.image_size

        bp = cam_setup.back_project_points(verts)                     # list of D × (V, 3)
        inside = np.ones(len(verts), dtype=bool)
        for d in range(D):
            H, W = union_masks[d].shape
            sx, sy = W / cam_w, H / cam_h
            pts = bp[d][:, :2]
            px = np.clip((pts[:, 0] * sx).astype(int), 0, W - 1)
            py = np.clip((pts[:, 1] * sy).astype(int), 0, H - 1)
            inside &= union_masks[d][py, px]

        # Keep faces where ALL three vertices are inside
        keep = inside[faces].all(axis=1)
        new_faces = faces[keep]
        if len(new_faces) == 0:
            return verts, faces  # nothing left — bail to original

        used = np.unique(new_faces)
        remap = -np.ones(len(verts), dtype=np.int64)
        remap[used] = np.arange(len(used))
        return verts[used], remap[new_faces].astype(faces.dtype)

    # ------------------------------------------------------------------
    # Projection preview
    # ------------------------------------------------------------------

    def _update_projection(self, *_):
        char = self.state.current_character
        if char is None or char.frames is None:
            return

        d = self.dir_spin.value()
        if not (0 <= d < char.frames.shape[0]):
            return

        n_spr = char.frames.shape[1]
        f_spr = max(0, min(self._current_frame, n_spr - 1))

        sprite = char.frames[d, f_spr]
        if sprite.dtype != np.uint8:
            sprite = sprite.astype(np.uint8)
        if sprite.shape[-1] == 4:
            sprite = sprite[..., :3]
        img = sprite.copy()

        # 2. Silhouette mask tint (red, alpha 0.3)
        if self.chk_mask.isChecked():
            try:
                if self._masks_cache is None:
                    self._masks_cache = self._extract_silhouette_masks(char)
                mask = self._masks_cache[d, f_spr]
                tint = np.zeros_like(img)
                tint[..., 0] = 255  # red
                a = 0.3
                img[mask] = (a * tint[mask] + (1 - a) * img[mask]).astype(np.uint8)
            except Exception as exc:
                _logger.warning("mask overlay failed: %s", exc)

        # 3. Mesh wireframe overlay (green)
        if self.chk_wire.isChecked() and self._ragdoll_verts is not None and self._ragdoll_faces is not None:
            cam = self._get_camera_setup(char)
            if cam is not None:
                cam_w, cam_h = cam.image_size
                bp = cam.back_project_points(self._ragdoll_verts)
                pts = bp[d][:, :2]
                fh, fw = img.shape[:2]
                sx, sy = fw / cam_w, fh / cam_h
                faces = self._ragdoll_faces
                if len(faces) > 1500:
                    faces = faces[:: max(1, len(faces) // 1500)]
                n_pts = len(pts)
                for tri in faces:
                    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                    if a >= n_pts or b >= n_pts or c >= n_pts:
                        continue
                    pa = (int(pts[a, 0] * sx), int(pts[a, 1] * sy))
                    pb = (int(pts[b, 0] * sx), int(pts[b, 1] * sy))
                    pc = (int(pts[c, 0] * sx), int(pts[c, 1] * sy))
                    cv2.line(img, pa, pb, (80, 255, 120), 1)
                    cv2.line(img, pb, pc, (80, 255, 120), 1)
                    cv2.line(img, pc, pa, (80, 255, 120), 1)

        # 4. Skeleton joints
        if self.chk_skel.isChecked() and self._skel_frames is not None:
            cam = self._get_camera_setup(char)
            if cam is not None:
                cam_w, cam_h = cam.image_size
                f_sk = max(0, min(self._current_frame, self._skel_frames.shape[0] - 1))
                bp = cam.back_project_points(self._skel_frames[f_sk])
                pts = bp[d][:, :2]
                fh, fw = img.shape[:2]
                sx, sy = fw / cam_w, fh / cam_h
                for p in pts:
                    px = int(p[0] * sx)
                    py = int(p[1] * sy)
                    if 0 <= px < fw and 0 <= py < fh:
                        cv2.circle(img, (px, py), 2, (255, 220, 80), -1)

        self._projection.show_image(img)

    # ------------------------------------------------------------------
    # Save
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
            "bone_radii": self._radii_dict(),
            "ragdoll_verts": self._ragdoll_verts.tolist(),
            "ragdoll_faces": self._ragdoll_faces.tolist(),
            "carved": self._carved,
        }
        try:
            with open(path, "w") as f:
                json.dump(template, f, indent=2)
            self._set_status(f"Saved → {os.path.basename(path)}")
        except Exception as exc:
            self._set_status(f"Save error: {exc}")
            _logger.error("Template save error: %s", exc)

    # ------------------------------------------------------------------
    # AppState handlers
    # ------------------------------------------------------------------

    def _on_char_changed(self, idx: int):
        # Reset cached masks/skeleton when the active character changes
        self._masks_cache = None
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
            self._masks_cache = None
            self._on_char_changed(idx)

    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        self.status_lbl.setText(text)
        _logger.info("MeshTemplate: %s", text)
