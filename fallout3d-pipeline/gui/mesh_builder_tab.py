"""
Tab 7b — Mesh Builder
Step-by-step workflow: skeleton → mesh fit → skinning weights → animate →
project onto sprite → AO bake → shadow sprites.
"""

import os
import logging
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSlider, QGroupBox, QSplitter, QProgressBar,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor

from gui.main_window import AppState, CRITTER_CATEGORIES
from gui.mesh_tab import MeshFitWorker, MeshViewer3D
from pipeline.mesh_fitter import (
    MeshFitter, ANCHORS_BY_CATEGORY, HUMANOID_ANCHORS, compute_skinning_weights,
)
from pipeline.ao_baker import AmbientOcclusionBaker
from pipeline.shadow_sprite import ShadowSpriteGenerator
from pipeline.pose_triangulator import POSE_CONNECTIONS

try:
    import pyqtgraph.opengl as gl  # noqa: F401
    _GL = True
except ImportError:
    _GL = False

_logger = logging.getLogger(__name__)

_DIR_LABELS = ["Dir0 NE", "Dir1 E", "Dir2 SE", "Dir3 SW", "Dir4 W", "Dir5 NW"]


# -----------------------------------------------------------------------
# Workers
# -----------------------------------------------------------------------

class SausageWorker(QObject):
    """Runs VoxelCarver.carve_all in a background thread."""
    progress = pyqtSignal(int, int)   # (done, total) — not yet granular
    finished = pyqtSignal(list)       # list[dict] from to_glb_meshes()
    error    = pyqtSignal(str)

    def __init__(self, skeleton_builder, camera_setup, all_masks, resolution):
        super().__init__()
        self._sb       = skeleton_builder
        self._cam      = camera_setup
        self._masks    = all_masks
        self._res      = resolution

    def run(self):
        try:
            from pipeline.voxel_carver import VoxelCarver
            vc = VoxelCarver(self._sb, self._cam, self._res)
            vc.carve_all(self._masks)
            meshes = vc.to_glb_meshes()
            self.finished.emit(meshes)
        except Exception as exc:
            self.error.emit(str(exc))


class AOBakeWorker(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)   # ao array (V,)
    error    = pyqtSignal(str)

    def __init__(self, baker: AmbientOcclusionBaker, verts: np.ndarray, faces: np.ndarray):
        super().__init__()
        self.baker = baker
        self.verts = verts
        self.faces = faces

    def run(self):
        try:
            ao = self.baker.bake_vertex_ao(
                self.verts, self.faces,
                progress_cb=lambda i, n: self.progress.emit(i, n),
            )
            self.finished.emit(ao)
        except Exception as exc:
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# Projection preview widget
# -----------------------------------------------------------------------

class ProjectionPreview(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background:#111; color:#aaa; border: 1px solid #333;")
        self.setText("No projection yet.")
        self._base_pixmap: QPixmap | None = None

    def show_image(self, rgb: np.ndarray):
        """Display a pre-rendered RGB (H,W,3) uint8 array directly."""
        if rgb is None:
            return
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pix)

    def show_projection(
        self,
        sprite_rgb: np.ndarray,
        edges_2d: list | None = None,
        shadow_rgba: np.ndarray | None = None,
        shadow_alpha: float = 1.0,
    ):
        """
        sprite_rgb : (H, W, 3) uint8
        edges_2d   : list of ((x1, y1), (x2, y2)) tuples in pixel coords
        shadow_rgba: optional (h, w, 4) uint8 image, blended at the bottom centre
        """
        if sprite_rgb is None:
            self.setText("No frame.")
            return
        if sprite_rgb.dtype != np.uint8:
            sprite_rgb = sprite_rgb.astype(np.uint8)
        h, w = sprite_rgb.shape[:2]

        if sprite_rgb.shape[-1] == 4:
            sprite_rgb = sprite_rgb[..., :3]

        composite = sprite_rgb.copy()

        # Composite shadow at bottom-centre
        if shadow_rgba is not None and shadow_rgba.size > 0:
            sh, sw = shadow_rgba.shape[:2]
            x0 = max(0, (w - sw) // 2)
            y0 = max(0, h - sh)
            x1 = min(w, x0 + sw)
            y1 = min(h, y0 + sh)
            sx1 = x1 - x0
            sy1 = y1 - y0
            if sx1 > 0 and sy1 > 0:
                shadow_crop = shadow_rgba[:sy1, :sx1]
                a = (shadow_crop[..., 3:4].astype(np.float32) / 255.0) * shadow_alpha
                fg = shadow_crop[..., :3].astype(np.float32)
                bg = composite[y0:y1, x0:x1].astype(np.float32)
                composite[y0:y1, x0:x1] = (a * fg + (1.0 - a) * bg).astype(np.uint8)

        qimg = QImage(composite.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())

        # Wireframe overlay
        if edges_2d:
            painter = QPainter(pix)
            pen = QPen(QColor(80, 255, 120, 200))
            pen.setWidth(1)
            painter.setPen(pen)
            for (x1, y1), (x2, y2) in edges_2d:
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            painter.end()

        scaled = pix.scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self._base_pixmap = pix


# -----------------------------------------------------------------------
# Main tab
# -----------------------------------------------------------------------

class MeshBuilderTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        # Mesh state
        self._fitter: MeshFitter | None = None
        self._mesh_verts: np.ndarray | None = None
        self._mesh_faces: np.ndarray | None = None
        self._skinning_weights: np.ndarray | None = None
        self._ao: np.ndarray | None = None
        self._shadow_sprites: list[np.ndarray] | None = None

        # Animation state
        self._current_frame = 0
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        # Threads
        self._fit_thread: QThread | None = None
        self._fit_worker: MeshFitWorker | None = None
        self._ao_thread: QThread | None = None
        self._ao_worker: AOBakeWorker | None = None
        self._sausage_thread: QThread | None = None
        self._sausage_worker: SausageWorker | None = None

        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.character_updated.connect(self._on_char_updated)

        self._on_char_changed(self.state.selected_idx)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([380, 820])

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(4, 4, 4, 4)

        # ── Step 1: Skeleton
        grp1 = QGroupBox("Step 1: Skeleton")
        v1 = QVBoxLayout(grp1)
        self.skeleton_lbl = QLabel("Skeleton: not loaded")
        v1.addWidget(self.skeleton_lbl)
        self.btn_use_skeleton = QPushButton("← Use skeleton from Tab 5b")
        self.btn_use_skeleton.clicked.connect(self._use_skeleton)
        v1.addWidget(self.btn_use_skeleton)
        vbox.addWidget(grp1)

        # ── Step 2: Fit Mesh
        grp2 = QGroupBox("Step 2: Fit Mesh")
        v2 = QVBoxLayout(grp2)

        row_mt = QHBoxLayout()
        row_mt.addWidget(QLabel("Mesh type:"))
        self.mesh_type_combo = QComboBox()
        self.mesh_type_combo.addItem("Ragdoll (capsules)", "ragdoll")
        self.mesh_type_combo.addItem("Template OBJ",       "template")
        self.mesh_type_combo.currentIndexChanged.connect(self._on_mesh_type_changed)
        row_mt.addWidget(self.mesh_type_combo, 1)
        v2.addLayout(row_mt)

        row_t = QHBoxLayout()
        self.template_label = QLabel("Template:")
        row_t.addWidget(self.template_label)
        self.template_combo = QComboBox()
        self._refresh_template_combo()
        row_t.addWidget(self.template_combo, 1)
        v2.addLayout(row_t)

        row_f = QHBoxLayout()
        self.falloff_label = QLabel("Falloff:")
        row_f.addWidget(self.falloff_label)
        self.falloff_spin = QDoubleSpinBox()
        self.falloff_spin.setRange(1.0, 10.0)
        self.falloff_spin.setSingleStep(0.5)
        self.falloff_spin.setValue(4.0)
        row_f.addWidget(self.falloff_spin)
        row_f.addStretch()
        v2.addLayout(row_f)

        self.btn_fit = QPushButton("Fit Mesh to Skeleton")
        self.btn_fit.setStyleSheet("font-weight: bold; padding: 4px 8px;")
        self.btn_fit.clicked.connect(self._fit_mesh)
        v2.addWidget(self.btn_fit)
        self.fit_progress = QProgressBar()
        self.fit_progress.setRange(0, 0)
        self.fit_progress.setVisible(False)
        v2.addWidget(self.fit_progress)
        vbox.addWidget(grp2)

        # ── Step 3: Skinning Weights
        grp3 = QGroupBox("Step 3: Skinning Weights")
        v3 = QVBoxLayout(grp3)
        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("Bone:"))
        self.bone_slider = QSlider(Qt.Orientation.Horizontal)
        self.bone_slider.setRange(0, 0)
        self.bone_slider.valueChanged.connect(self._on_bone_changed)
        row_b.addWidget(self.bone_slider)
        self.bone_name_lbl = QLabel("—")
        self.bone_name_lbl.setMinimumWidth(80)
        row_b.addWidget(self.bone_name_lbl)
        v3.addLayout(row_b)
        self.chk_heatmap = QCheckBox("Show weight heatmap in 3D viewer")
        self.chk_heatmap.toggled.connect(self._refresh_3d_view)
        v3.addWidget(self.chk_heatmap)
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

        # ── Step 5: Project onto Sprite
        grp5 = QGroupBox("Step 5: Project onto Sprite")
        v5 = QVBoxLayout(grp5)
        row_dir = QHBoxLayout()
        row_dir.addWidget(QLabel("Direction:"))
        self.dir_combo = QComboBox()
        for lbl in _DIR_LABELS:
            self.dir_combo.addItem(lbl)
        self.dir_combo.currentIndexChanged.connect(self._update_projection)
        row_dir.addWidget(self.dir_combo, 1)
        v5.addLayout(row_dir)

        self.chk_wireframe = QCheckBox("Show mesh wireframe overlay")
        self.chk_wireframe.setChecked(True)
        self.chk_wireframe.toggled.connect(self._update_projection)
        v5.addWidget(self.chk_wireframe)

        self.chk_shadow = QCheckBox("Show shadow sprite")
        self.chk_shadow.toggled.connect(self._update_projection)
        v5.addWidget(self.chk_shadow)

        row_alpha = QHBoxLayout()
        row_alpha.addWidget(QLabel("Overlay alpha:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(80)
        self.alpha_slider.valueChanged.connect(self._update_projection)
        row_alpha.addWidget(self.alpha_slider)
        v5.addLayout(row_alpha)

        self.btn_project_curr = QPushButton("Project Current Frame")
        self.btn_project_curr.clicked.connect(self._update_projection)
        v5.addWidget(self.btn_project_curr)

        self.btn_project_all = QPushButton("Project All Frames → char.projected_frames")
        self.btn_project_all.clicked.connect(self._project_all_frames)
        v5.addWidget(self.btn_project_all)
        vbox.addWidget(grp5)

        # ── Step 6: AO Baking
        grp6 = QGroupBox("Step 6: AO Baking")
        v6 = QVBoxLayout(grp6)
        row_r = QHBoxLayout()
        row_r.addWidget(QLabel("Rays:"))
        self.ao_rays_spin = QSpinBox()
        self.ao_rays_spin.setRange(16, 256)
        self.ao_rays_spin.setValue(64)
        row_r.addWidget(self.ao_rays_spin)
        row_r.addWidget(QLabel("Max dist:"))
        self.ao_dist_spin = QDoubleSpinBox()
        self.ao_dist_spin.setRange(0.1, 2.0)
        self.ao_dist_spin.setSingleStep(0.1)
        self.ao_dist_spin.setValue(0.5)
        row_r.addWidget(self.ao_dist_spin)
        row_r.addStretch()
        v6.addLayout(row_r)

        self.btn_bake_ao = QPushButton("Bake AO")
        self.btn_bake_ao.clicked.connect(self._bake_ao)
        v6.addWidget(self.btn_bake_ao)
        self.ao_progress = QProgressBar()
        self.ao_progress.setRange(0, 100)
        self.ao_progress.setValue(0)
        self.ao_progress.setVisible(False)
        v6.addWidget(self.ao_progress)
        vbox.addWidget(grp6)

        # ── Step 7: Shadow Sprites
        grp7 = QGroupBox("Step 7: Shadow Sprites")
        v7 = QVBoxLayout(grp7)
        row_s = QHBoxLayout()
        row_s.addWidget(QLabel("Image size:"))
        self.shadow_size_spin = QSpinBox()
        self.shadow_size_spin.setRange(64, 512)
        self.shadow_size_spin.setSingleStep(32)
        self.shadow_size_spin.setValue(128)
        row_s.addWidget(self.shadow_size_spin)
        v7.addLayout(row_s)

        row_b2 = QHBoxLayout()
        row_b2.addWidget(QLabel("Blur sigma:"))
        self.shadow_blur_spin = QDoubleSpinBox()
        self.shadow_blur_spin.setRange(0.5, 20.0)
        self.shadow_blur_spin.setSingleStep(0.5)
        self.shadow_blur_spin.setValue(4.0)
        row_b2.addWidget(self.shadow_blur_spin)
        row_b2.addWidget(QLabel("Opacity:"))
        self.shadow_opacity_spin = QDoubleSpinBox()
        self.shadow_opacity_spin.setRange(0.1, 1.0)
        self.shadow_opacity_spin.setSingleStep(0.05)
        self.shadow_opacity_spin.setValue(0.6)
        row_b2.addWidget(self.shadow_opacity_spin)
        v7.addLayout(row_b2)

        self.btn_gen_shadows = QPushButton("Generate Shadow Sprites")
        self.btn_gen_shadows.clicked.connect(self._generate_shadows)
        v7.addWidget(self.btn_gen_shadows)
        self.shadow_status_lbl = QLabel("—")
        v7.addWidget(self.shadow_status_lbl)
        vbox.addWidget(grp7)

        # ── Step 8: Voxel Carve
        grp8 = QGroupBox("Step 8: Voxel Carve Sausages")
        v8 = QVBoxLayout(grp8)

        row_res = QHBoxLayout()
        row_res.addWidget(QLabel("Resolution:"))
        self.sausage_res_combo = QComboBox()
        for r in (16, 32, 64):
            self.sausage_res_combo.addItem(str(r), r)
        self.sausage_res_combo.setCurrentIndex(1)  # default 32
        row_res.addWidget(self.sausage_res_combo, 1)
        v8.addLayout(row_res)

        self.btn_carve_sausages = QPushButton("Carve Voxel Sausages (all frames)")
        self.btn_carve_sausages.setStyleSheet("font-weight: bold; padding: 4px 8px;")
        self.btn_carve_sausages.clicked.connect(self._carve_voxel_sausages)
        v8.addWidget(self.btn_carve_sausages)

        self.sausage_progress = QProgressBar()
        self.sausage_progress.setRange(0, 0)
        self.sausage_progress.setVisible(False)
        v8.addWidget(self.sausage_progress)

        self.sausage_status_lbl = QLabel("—")
        self.sausage_status_lbl.setWordWrap(True)
        v8.addWidget(self.sausage_status_lbl)

        vbox.addWidget(grp8)

        # Status
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#88f; font-style: italic;")
        self.status_lbl.setWordWrap(True)
        vbox.addWidget(self.status_lbl)

        vbox.addStretch()

        # Apply initial mesh-type enabled state (default = ragdoll)
        self._on_mesh_type_changed(self.mesh_type_combo.currentIndex())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(440)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _build_right_panel(self) -> QWidget:
        panel = QSplitter(Qt.Orientation.Vertical)

        # Top: 3D mesh viewer (reused from mesh_tab)
        self._mesh_viewer = MeshViewer3D()
        panel.addWidget(self._mesh_viewer)

        # Bottom: projection preview
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(2, 2, 2, 2)
        self._projection = ProjectionPreview()
        self._projection.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        bottom_layout.addWidget(self._projection, 1)
        panel.addWidget(bottom)

        panel.setSizes([520, 320])
        return panel

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_template_combo(self):
        self.template_combo.clear()
        templates_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets", "templates",
        )
        for cat, info in CRITTER_CATEGORIES.items():
            tmpl = info.get("template")
            if tmpl and os.path.exists(os.path.join(templates_dir, tmpl)):
                self.template_combo.addItem(
                    f"{cat.title()} ({tmpl})",
                    os.path.join(templates_dir, tmpl),
                )
        if self.template_combo.count() == 0:
            self.template_combo.addItem("No templates found", "")

    def _set_status(self, text: str):
        self.status_lbl.setText(text)
        _logger.info("MeshBuilder: %s", text)

    def _on_mesh_type_changed(self, _idx: int):
        is_ragdoll = self.mesh_type_combo.currentData() == "ragdoll"
        for w in (self.template_label, self.template_combo,
                  self.falloff_label, self.falloff_spin):
            w.setEnabled(not is_ragdoll)
        self._set_status("")

    def _animation_skeleton(self) -> np.ndarray | None:
        """Return per-frame (33,3) skeleton frames from char (preferring SkeletonBuilder)."""
        char = self.state.current_character
        if char is None:
            return None
        if char.skeleton is not None and getattr(char.skeleton, "poses", None) is not None:
            # SkeletonBuilder.poses is (N, 37, 3); slice to first 33 to match MediaPipe
            return char.skeleton.poses[:, :33, :]
        return char.skeleton_3d

    # ------------------------------------------------------------------
    # Step 1: Skeleton
    # ------------------------------------------------------------------

    def _use_skeleton(self):
        char = self.state.current_character
        if char is None:
            self._set_status("No character loaded.")
            return
        skel = self._animation_skeleton()
        if skel is None:
            self._set_status("No skeleton available — run triangulation first.")
            return
        self.skeleton_lbl.setText(f"Skeleton: {skel.shape[0]} frames loaded")
        n = skel.shape[0]
        self.frame_slider.setRange(0, max(0, n - 1))
        self.frame_lbl.setText(f"1 / {n}")
        self._set_status(f"Skeleton ready ({n} frames).")

    # ------------------------------------------------------------------
    # Step 2: Fit Mesh
    # ------------------------------------------------------------------

    def _fit_mesh(self):
        char = self.state.current_character
        if char is None:
            self._set_status("No character loaded.")
            return
        skel = self._animation_skeleton()
        if skel is None:
            self._set_status("No skeleton — run triangulation first.")
            return

        rest = skel[0]
        mesh_type = self.mesh_type_combo.currentData()

        if mesh_type == "ragdoll":
            # Build a capsule ragdoll directly — fast enough to skip the worker
            self._fitter = MeshFitter(char.category)
            try:
                verts, faces = MeshFitter.generate_ragdoll(rest)
            except Exception as exc:
                self._set_status(f"Ragdoll generation error: {exc}")
                return

            self._fitter.template_verts = verts
            self._fitter.template_faces = faces
            self._fitter.anchors = HUMANOID_ANCHORS
            try:
                # Use the standard bind path so rest_joints + bone_parents are set
                # and per-frame fit_to_skeleton(...) works correctly.
                self._fitter.bind_to_skeleton(rest, falloff=self.falloff_spin.value())
            except Exception as exc:
                self._set_status(f"Ragdoll skinning error: {exc}")
                return

            self._on_fit_done(verts, faces, self._fitter.skinning_weights)
            return

        # Template OBJ path (existing behaviour)
        tmpl_path = self.template_combo.currentData()
        if not tmpl_path or not os.path.exists(tmpl_path):
            self._set_status("No valid template selected.")
            return

        self._fitter = MeshFitter(char.category)
        try:
            self._fitter.load_template(tmpl_path)
        except Exception as exc:
            self._set_status(f"Template load error: {exc}")
            return

        self.fit_progress.setVisible(True)
        self.btn_fit.setEnabled(False)

        self._fit_worker = MeshFitWorker(self._fitter, rest)
        self._fit_thread = QThread(self)
        self._fit_worker.moveToThread(self._fit_thread)
        self._fit_thread.started.connect(self._fit_worker.run)
        self._fit_worker.finished.connect(self._on_fit_done)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_thread.start()

    def _on_fit_done(self, verts, faces, weights):
        if self._fit_thread is not None:
            self._fit_thread.quit()
        self.fit_progress.setVisible(False)
        self.btn_fit.setEnabled(True)

        self._mesh_verts = verts
        self._mesh_faces = faces
        self._skinning_weights = weights

        char = self.state.current_character
        if char is not None:
            char.mesh_verts = verts
            char.skinning_weights = weights
            self.state.character_updated.emit(self.state.selected_idx)

        n_bones = weights.shape[1] if weights is not None else 0
        self.bone_slider.setRange(0, max(0, n_bones - 1))
        self._on_bone_changed(0)
        self._refresh_3d_view()
        self._set_status(f"Mesh fitted — {len(verts)} verts, {n_bones} bones.")

    def _on_fit_error(self, msg: str):
        self._fit_thread.quit()
        self.fit_progress.setVisible(False)
        self.btn_fit.setEnabled(True)
        self._set_status(f"Fit error: {msg}")

    # ------------------------------------------------------------------
    # Step 3: Skinning weights / bone display
    # ------------------------------------------------------------------

    def _on_bone_changed(self, val: int):
        if self._fitter and self._fitter.anchors:
            names = list(self._fitter.anchors.keys())
            if 0 <= val < len(names):
                self.bone_name_lbl.setText(names[val])
            else:
                self.bone_name_lbl.setText("—")
        self._refresh_3d_view()

    # ------------------------------------------------------------------
    # Step 4: Animation
    # ------------------------------------------------------------------

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
        skel = self._animation_skeleton()
        if skel is None:
            self.btn_play.setChecked(False)
            return
        n = skel.shape[0]
        if n == 0:
            self.btn_play.setChecked(False)
            return
        next_frame = (self._current_frame + 1) % n
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(next_frame)
        self.frame_slider.blockSignals(False)
        self._current_frame = next_frame
        self._update_for_frame()

    def _on_frame_slider(self, val: int):
        self._current_frame = val
        self._update_for_frame()

    def _update_for_frame(self):
        skel = self._animation_skeleton()
        if skel is None:
            return
        n = skel.shape[0]
        f = max(0, min(self._current_frame, n - 1))
        self.frame_lbl.setText(f"{f + 1} / {n}")

        # Recompute mesh verts for this frame
        if self._fitter is not None and self._fitter.skinning_weights is not None:
            try:
                self._mesh_verts = self._fitter.fit_to_skeleton(skel[f])
            except Exception as exc:
                _logger.warning("fit_to_skeleton failed at frame %d: %s", f, exc)

        self._refresh_3d_view()
        self._update_projection()

    # ------------------------------------------------------------------
    # 3D viewer
    # ------------------------------------------------------------------

    def _refresh_3d_view(self):
        if self._mesh_verts is None or self._mesh_faces is None:
            return
        weights = self._skinning_weights if self.chk_heatmap.isChecked() else None
        self._mesh_viewer.set_mesh(
            self._mesh_verts, self._mesh_faces,
            weights, self.bone_slider.value(),
        )

        skel = self._animation_skeleton()
        if skel is not None and skel.shape[0] > 0:
            f = max(0, min(self._current_frame, skel.shape[0] - 1))
            self._mesh_viewer.set_skeleton_overlay(skel[f], POSE_CONNECTIONS)

    # ------------------------------------------------------------------
    # Step 5: Projection
    # ------------------------------------------------------------------

    def _get_camera_setup(self, char=None):
        """Return the IsometricCameraSetup used for triangulation.

        Walks up the widget hierarchy to find MainWindow.tab_recon and
        re-uses its triangulator's camera_setup — the same coordinate system
        that produced the 3D skeleton. Falls back to a fresh PoseTriangulator
        sized to the character's sprite frames when the reconstruction tab is
        not yet accessible (e.g. standalone unit tests).
        """
        p = self.parent()
        while p is not None:
            if hasattr(p, "tab_recon"):
                return p.tab_recon._triangulator.camera_setup
            p = p.parent()
        # Fallback: build a triangulator with image_size matching the sprites
        if char is None:
            char = self.state.current_character
        if char is None or char.frames is None or char.frames.size == 0:
            return None
        from pipeline.pose_triangulator import PoseTriangulator
        h, w = char.frames[0, 0].shape[:2]
        return PoseTriangulator(image_size=(w, h)).camera_setup

    def _update_projection(self, *_):
        """Render the back-projected skeleton (+ optional mesh wireframe) onto
        the sprite for the currently selected direction and frame.

        Works as soon as a skeleton is available; the mesh overlay is added only
        after a mesh has been fitted.
        """
        from gui.reconstruction_tab import ReconstructionTab

        char = self.state.current_character
        if char is None or char.frames is None:
            return
        d = self.dir_combo.currentIndex()
        if not (0 <= d < char.frames.shape[0]):
            return

        skel = self._animation_skeleton()
        if skel is None:
            return
        n = skel.shape[0]
        f       = max(0, min(self._current_frame, n - 1))
        f_spr   = max(0, min(f, char.frames.shape[1] - 1))

        sprite = char.frames[d, f_spr]
        if sprite.dtype != np.uint8:
            sprite = sprite.astype(np.uint8)
        sprite = sprite[..., :3].copy() if sprite.shape[-1] == 4 else sprite.copy()

        cam = self._get_camera_setup(char)
        if cam is None:
            self._projection.show_image(sprite)
            return

        cam_w, cam_h = cam.image_size
        bp_skel = cam.back_project_points(skel[f])   # list of 6 × (33, 3)

        # Skeleton backprojection: colored joints (blue=left, red=right, white=center)
        img = ReconstructionTab._render_backprojection(sprite, bp_skel[d], cam_w, cam_h)

        # Optional mesh wireframe overlay (green lines via cv2)
        if self.chk_wireframe.isChecked() and \
                self._mesh_verts is not None and self._mesh_faces is not None:
            bp_mesh = cam.back_project_points(self._mesh_verts)  # list of 6 × (V, 3)
            pts_2d  = bp_mesh[d][:, :2]
            fh, fw  = img.shape[:2]
            sx, sy  = fw / cam_w, fh / cam_h
            faces   = self._mesh_faces
            if len(faces) > 1500:
                faces = faces[:: len(faces) // 1500]
            n_pts = len(pts_2d)
            for tri in faces:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                if a >= n_pts or b >= n_pts or c >= n_pts:
                    continue
                pa = (int(pts_2d[a, 0] * sx), int(pts_2d[a, 1] * sy))
                pb = (int(pts_2d[b, 0] * sx), int(pts_2d[b, 1] * sy))
                pc = (int(pts_2d[c, 0] * sx), int(pts_2d[c, 1] * sy))
                cv2.line(img, pa, pb, (80, 255, 120), 1)
                cv2.line(img, pb, pc, (80, 255, 120), 1)
                cv2.line(img, pc, pa, (80, 255, 120), 1)

        # Optional shadow composite (drawn last so it sits under the skeleton)
        if self.chk_shadow.isChecked() and self._shadow_sprites is not None:
            if 0 <= d < len(self._shadow_sprites):
                shadow_rgba = self._shadow_sprites[d]
                alpha_f = self.alpha_slider.value() / 100.0
                if shadow_rgba is not None and shadow_rgba.size > 0:
                    sh, sw = shadow_rgba.shape[:2]
                    h_, w_ = img.shape[:2]
                    x0 = max(0, (w_ - sw) // 2)
                    y0 = max(0, h_ - sh)
                    x1 = min(w_, x0 + sw)
                    y1 = min(h_, y0 + sh)
                    if x1 > x0 and y1 > y0:
                        sc = shadow_rgba[: y1 - y0, : x1 - x0]
                        a  = (sc[..., 3:4].astype(np.float32) / 255.0) * alpha_f
                        img[y0:y1, x0:x1] = (
                            a * sc[..., :3].astype(np.float32)
                            + (1.0 - a) * img[y0:y1, x0:x1].astype(np.float32)
                        ).astype(np.uint8)

        self._projection.show_image(img)

    def _project_all_frames(self):
        char = self.state.current_character
        if char is None or char.frames is None:
            self._set_status("No character.")
            return
        if self._mesh_verts is None or self._fitter is None or self._fitter.skinning_weights is None:
            self._set_status("Fit a mesh first.")
            return
        skel = self._animation_skeleton()
        if skel is None:
            self._set_status("No skeleton.")
            return

        cam = self._get_camera_setup(char)
        if cam is None:
            self._set_status("No camera setup.")
            return

        n_dirs = char.frames.shape[0]
        n_frames = min(skel.shape[0], char.frames.shape[1])
        h, w = char.frames[0, 0].shape[:2]
        cam_w, cam_h = cam.image_size
        sx, sy = w / cam_w, h / cam_h

        out = np.zeros((n_dirs, n_frames, h, w, 3), dtype=np.uint8)
        try:
            for fi in range(n_frames):
                verts = self._fitter.fit_to_skeleton(skel[fi])
                bp = cam.back_project_points(verts)
                for d in range(n_dirs):
                    sprite = char.frames[d, fi]
                    if sprite.dtype != np.uint8:
                        sprite = sprite.astype(np.uint8)
                    if sprite.shape[-1] == 4:
                        sprite = sprite[..., :3]
                    composite = sprite.copy()
                    pts_cam = bp[d][:, :2]
                    px = np.clip((pts_cam[:, 0] * sx).astype(int), 0, w - 1)
                    py = np.clip((pts_cam[:, 1] * sy).astype(int), 0, h - 1)
                    composite[py, px] = (80, 255, 120)
                    out[d, fi] = composite
        except Exception as exc:
            self._set_status(f"Projection error: {exc}")
            return

        char.projected_frames = out
        self._set_status(f"Projected {n_frames} frames × {n_dirs} dirs → char.projected_frames.")

    # ------------------------------------------------------------------
    # Step 6: AO baking
    # ------------------------------------------------------------------

    def _bake_ao(self):
        if self._mesh_verts is None or self._mesh_faces is None:
            self._set_status("Fit a mesh first.")
            return
        baker = AmbientOcclusionBaker(
            num_rays=self.ao_rays_spin.value(),
            max_dist=self.ao_dist_spin.value(),
        )
        self.ao_progress.setVisible(True)
        self.ao_progress.setValue(0)
        self.btn_bake_ao.setEnabled(False)

        self._ao_worker = AOBakeWorker(baker, self._mesh_verts, self._mesh_faces)
        self._ao_thread = QThread(self)
        self._ao_worker.moveToThread(self._ao_thread)
        self._ao_thread.started.connect(self._ao_worker.run)
        self._ao_worker.progress.connect(self._on_ao_progress)
        self._ao_worker.finished.connect(self._on_ao_done)
        self._ao_worker.error.connect(self._on_ao_error)
        self._ao_thread.start()

    def _on_ao_progress(self, i: int, n: int):
        if n > 0:
            self.ao_progress.setValue(int(100 * i / n))

    def _on_ao_done(self, ao: np.ndarray):
        self._ao_thread.quit()
        self.ao_progress.setVisible(False)
        self.btn_bake_ao.setEnabled(True)
        self._ao = ao
        self._set_status(f"AO baked — mean={float(ao.mean()):.3f}, min={float(ao.min()):.3f}")

    def _on_ao_error(self, msg: str):
        self._ao_thread.quit()
        self.ao_progress.setVisible(False)
        self.btn_bake_ao.setEnabled(True)
        self._set_status(f"AO error: {msg}")

    # ------------------------------------------------------------------
    # Step 7: Shadow sprites
    # ------------------------------------------------------------------

    def _generate_shadows(self):
        if self._mesh_verts is None:
            self._set_status("Fit a mesh first.")
            return
        gen = ShadowSpriteGenerator(
            img_size=self.shadow_size_spin.value(),
            blur_sigma=self.shadow_blur_spin.value(),
            opacity=self.shadow_opacity_spin.value(),
        )
        try:
            self._shadow_sprites = gen.generate_all_directions(self._mesh_verts)
        except Exception as exc:
            self.shadow_status_lbl.setText(f"Error: {exc}")
            self._set_status(f"Shadow generation error: {exc}")
            return
        self.shadow_status_lbl.setText(
            f"Generated {len(self._shadow_sprites)} shadows ({self.shadow_size_spin.value()}px)."
        )
        self._set_status("Shadow sprites ready.")
        if self.chk_shadow.isChecked():
            self._update_projection()

    # ------------------------------------------------------------------
    # Step 8: Voxel carve sausages
    # ------------------------------------------------------------------

    def _carve_voxel_sausages(self):
        char = self.state.current_character
        if char is None:
            self._set_status("No character loaded.")
            return
        sb = getattr(char, "skeleton", None)
        if sb is None or getattr(sb, "poses", None) is None:
            self._set_status("No skeleton with poses — run triangulation first.")
            return
        usc = getattr(char, "upscaled_frames", None)
        if usc is None:
            self._set_status("No upscaled frames — run upscaler first.")
            return

        cam = self._get_camera_setup(char)
        if cam is None:
            self._set_status("Camera setup not available.")
            return

        # usc is (6, N, H, W, C) — direction-major
        n_dirs, n_frames = usc.shape[0], usc.shape[1]
        all_masks = []
        for f in range(n_frames):
            frame_masks = []
            for d in range(n_dirs):
                img = usc[d, f]
                if img.ndim == 3 and img.shape[-1] == 4:
                    mask = (img[:, :, 3] > 0).astype(np.uint8)
                else:
                    mask = (img.mean(axis=-1) > 10).astype(np.uint8)
                frame_masks.append(mask)
            all_masks.append(frame_masks)

        resolution = self.sausage_res_combo.currentData()

        self.btn_carve_sausages.setEnabled(False)
        self.sausage_progress.setVisible(True)
        self.sausage_status_lbl.setText("Carving …")

        self._sausage_worker = SausageWorker(sb, cam, all_masks, resolution)
        self._sausage_thread = QThread(self)
        self._sausage_worker.moveToThread(self._sausage_thread)
        self._sausage_thread.started.connect(self._sausage_worker.run)
        self._sausage_worker.finished.connect(self._on_sausage_done)
        self._sausage_worker.error.connect(self._on_sausage_error)
        self._sausage_thread.start()

    def _on_sausage_done(self, meshes: list):
        self._sausage_thread.quit()
        self.sausage_progress.setVisible(False)
        self.btn_carve_sausages.setEnabled(True)

        if not meshes:
            self.sausage_status_lbl.setText("Carve produced no meshes.")
            self._set_status("Voxel carve: no surviving voxels.")
            return

        total_v = sum(len(m["verts_local"]) for m in meshes)
        total_f = sum(len(m["faces"]) for m in meshes)
        self.sausage_status_lbl.setText(
            f"{len(meshes)} bones, {total_v} verts, {total_f} faces"
        )
        self._set_status(f"Voxel sausages ready — {len(meshes)} bones.")

        # Merge all bones into a single world-space mesh for the 3D viewer.
        # Each sausage's verts_local are in bone-local space; we concatenate
        # them without re-transforming here (they share world scale because
        # the grid origin is in local metres).
        all_v, all_f = [], []
        offset = 0
        for m in meshes:
            all_v.append(m["verts_local"])
            all_f.append(m["faces"] + offset)
            offset += len(m["verts_local"])

        if all_v:
            merged_v = np.concatenate(all_v, axis=0)
            merged_f = np.concatenate(all_f, axis=0)
            self._mesh_viewer.set_mesh(merged_v, merged_f, None, 0)

    def _on_sausage_error(self, msg: str):
        self._sausage_thread.quit()
        self.sausage_progress.setVisible(False)
        self.btn_carve_sausages.setEnabled(True)
        self.sausage_status_lbl.setText(f"Error: {msg}")
        self._set_status(f"Voxel carve error: {msg}")

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def _on_char_changed(self, idx: int):
        char = self.state.current_character
        if char is None:
            self.skeleton_lbl.setText("Skeleton: not loaded")
            self.frame_slider.setRange(0, 0)
            self.frame_lbl.setText("0 / 0")
            return

        skel = self._animation_skeleton()
        if skel is not None:
            self.skeleton_lbl.setText(f"Skeleton: {skel.shape[0]} frames available")
            n = skel.shape[0]
            self.frame_slider.setRange(0, max(0, n - 1))
            self.frame_lbl.setText(f"1 / {n}")
        else:
            self.skeleton_lbl.setText("Skeleton: not loaded")

        # Reset mesh state if char has a previously fitted mesh
        if char.mesh_verts is not None:
            self._mesh_verts = char.mesh_verts
            self._skinning_weights = char.skinning_weights
            self._refresh_3d_view()

    def _on_char_updated(self, idx: int):
        if idx == self.state.selected_idx:
            self._on_char_changed(idx)

    def run_fit(self):
        """Public entry-point for pipeline automation."""
        self._fit_mesh()
