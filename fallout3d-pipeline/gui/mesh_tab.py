"""
Tab 5 — Mesh & Normal Maps
Fits a template OBJ mesh to the 3D skeleton with automatic LBS skinning,
previews skinning-weight heatmaps, and bakes normal + depth maps.
"""

import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSlider, QGroupBox, QSplitter, QProgressBar,
    QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage

from gui.main_window import AppState, CRITTER_CATEGORIES
from pipeline.mesh_fitter import MeshFitter, ANCHORS_BY_CATEGORY
from pipeline.normal_map_baker import NormalMapBaker

try:
    import pyqtgraph.opengl as gl
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False


# -----------------------------------------------------------------------
# Mesh fitting worker
# -----------------------------------------------------------------------

class MeshFitWorker(QObject):
    finished = pyqtSignal(object, object, object)  # verts, faces, weights
    error    = pyqtSignal(str)

    def __init__(self, fitter: MeshFitter, skeleton: np.ndarray):
        super().__init__()
        self.fitter = fitter
        self.skeleton = skeleton

    def run(self):
        try:
            self.fitter.bind_to_skeleton(self.skeleton)
            verts = self.fitter.fit_to_skeleton(self.skeleton)
            weights = self.fitter.skinning_weights
            self.finished.emit(verts, self.fitter.template_faces, weights)
        except Exception as exc:
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# Normal map baking worker
# -----------------------------------------------------------------------

class BakeWorker(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object, object)   # normal_maps, depth_maps
    error    = pyqtSignal(str)

    def __init__(self, baker: NormalMapBaker, frames: np.ndarray):
        super().__init__()
        self.baker = baker
        self.frames = frames

    def run(self):
        try:
            nm, dm = self.baker.bake_sequence(self.frames)
            self.finished.emit(nm, dm)
        except Exception as exc:
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# 3D mesh preview
# -----------------------------------------------------------------------

class MeshViewer3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _GL_AVAILABLE:
            self._view = gl.GLViewWidget()
            self._view.setBackgroundColor("k")
            self._view.setCameraPosition(distance=4, elevation=20, azimuth=45)
            grid = gl.GLGridItem()
            grid.scale(0.5, 0.5, 0.5)
            self._view.addItem(grid)
            self._mesh_item = None
            layout.addWidget(self._view)
        else:
            layout.addWidget(QLabel("pyqtgraph not available."))

    def set_mesh(self, verts: np.ndarray, faces: np.ndarray, weights: np.ndarray | None = None, bone_idx: int = 0):
        if not _GL_AVAILABLE:
            return
        if self._mesh_item:
            self._view.removeItem(self._mesh_item)
            self._mesh_item = None

        if verts is None or faces is None or len(faces) == 0:
            return

        # Vertex colours from skinning weights if provided
        if weights is not None and bone_idx < weights.shape[1]:
            w = weights[:, bone_idx]
            w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
            colors = np.zeros((len(verts), 4), dtype=np.float32)
            colors[:, 0] = w_norm           # red channel = weight
            colors[:, 2] = 1.0 - w_norm    # blue = 1-weight
            colors[:, 3] = 0.85
        else:
            colors = np.ones((len(verts), 4), dtype=np.float32) * [0.7, 0.7, 0.8, 0.85]

        mesh_data = gl.MeshData(vertexes=verts.astype(np.float32), faces=faces.astype(np.int32))
        self._mesh_item = gl.GLMeshItem(
            meshdata=mesh_data,
            color=(0.7, 0.7, 0.8, 0.85),
            drawFaces=True,
            drawEdges=False,
            smooth=True,
        )
        self._view.addItem(self._mesh_item)


# -----------------------------------------------------------------------
# Normal map image widget
# -----------------------------------------------------------------------

class NormalMapPreview(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setText("No normal map yet.")
        self.setStyleSheet("background: #111; color: #aaa; border: 1px solid #333;")

    def set_normal_map(self, nm: np.ndarray):
        """nm: (H, W, 3) uint8"""
        h, w, c = nm.shape
        qimg = QImage(nm.data, w, h, w * c, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pix)


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class MeshTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._fitter: MeshFitter | None = None
        self._mesh_verts: np.ndarray | None = None
        self._mesh_faces: np.ndarray | None = None
        self._skinning_weights: np.ndarray | None = None
        self._normal_maps: np.ndarray | None = None
        self._depth_maps: np.ndarray | None = None
        self._thread: QThread | None = None
        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.character_updated.connect(self._on_char_updated)

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Controls
        ctrl = QHBoxLayout()
        root.addLayout(ctrl)

        # Template selection
        ctrl.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self._refresh_template_combo()
        ctrl.addWidget(self.template_combo)

        self.btn_fit = QPushButton("Fit Mesh to Skeleton")
        self.btn_fit.setStyleSheet("font-weight: bold; padding: 5px 10px;")
        self.btn_fit.clicked.connect(self._fit_mesh)
        ctrl.addWidget(self.btn_fit)

        ctrl.addWidget(QLabel("Bone:"))
        self.bone_slider = QSlider(Qt.Orientation.Horizontal)
        self.bone_slider.setRange(0, len(ANCHORS_BY_CATEGORY.get("humanoid", {})) - 1)
        self.bone_slider.valueChanged.connect(self._on_bone_slider)
        ctrl.addWidget(self.bone_slider)

        self.bone_lbl = QLabel("0")
        ctrl.addWidget(self.bone_lbl)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress)

        self.status_lbl = QLabel("")
        ctrl.addWidget(self.status_lbl)

        # Splitter: mesh view + normal map panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        self._mesh_viewer = MeshViewer3D()
        splitter.addWidget(self._mesh_viewer)

        # Right: normal map baking
        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)

        bake_box = QGroupBox("Normal Map Baking")
        bake_layout = QVBoxLayout(bake_box)

        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Strength:"))
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.1, 10.0)
        self.strength_spin.setSingleStep(0.5)
        self.strength_spin.setValue(2.0)
        strength_row.addWidget(self.strength_spin)
        bake_layout.addLayout(strength_row)

        blur_row = QHBoxLayout()
        blur_row.addWidget(QLabel("Blur radius:"))
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(0, 5)
        self.blur_spin.setValue(1)
        blur_row.addWidget(self.blur_spin)
        bake_layout.addLayout(blur_row)

        self.btn_bake = QPushButton("Bake Normal Maps")
        self.btn_bake.setStyleSheet("font-weight: bold; padding: 5px 10px;")
        self.btn_bake.clicked.connect(self._bake_normals)
        bake_layout.addWidget(self.btn_bake)

        self.bake_progress = QProgressBar()
        self.bake_progress.setRange(0, 0)
        self.bake_progress.setVisible(False)
        bake_layout.addWidget(self.bake_progress)

        right_layout.addWidget(bake_box)

        # Normal map preview
        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Dir:"))
        self.dir_combo = QComboBox()
        for i in range(6):
            self.dir_combo.addItem(f"Dir {i+1}  ({i*60}°)")
        self.dir_combo.currentIndexChanged.connect(self._refresh_nm_preview)
        dir_row.addWidget(self.dir_combo)

        dir_row.addWidget(QLabel("Frame:"))
        self.nm_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.nm_frame_slider.setRange(0, 0)
        self.nm_frame_slider.valueChanged.connect(self._refresh_nm_preview)
        dir_row.addWidget(self.nm_frame_slider)
        preview_layout.addLayout(dir_row)

        self.nm_preview = NormalMapPreview()
        preview_layout.addWidget(self.nm_preview, 1)

        right_layout.addWidget(preview_box, 1)

        splitter.setSizes([700, 400])

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
                self.template_combo.addItem(f"{cat.title()} ({tmpl})", os.path.join(templates_dir, tmpl))
        if self.template_combo.count() == 0:
            self.template_combo.addItem("No templates found", "")

    # ------------------------------------------------------------------
    # Mesh fitting
    # ------------------------------------------------------------------

    def _fit_mesh(self):
        char = self.state.current_character
        if char is None:
            self.status_lbl.setText("No character loaded.")
            return
        if char.skeleton_3d is None:
            self.status_lbl.setText("Triangulate skeleton first (Tab 3).")
            return

        tmpl_path = self.template_combo.currentData()
        if not tmpl_path or not os.path.exists(tmpl_path):
            self.status_lbl.setText("No valid template selected.")
            return

        self._fitter = MeshFitter(char.category)
        try:
            self._fitter.load_template(tmpl_path)
        except Exception as exc:
            self.status_lbl.setText(f"Template load error: {exc}")
            return

        rest_skel = char.skeleton_3d[0]
        self.progress.setVisible(True)
        self.btn_fit.setEnabled(False)

        self._mesh_worker = MeshFitWorker(self._fitter, rest_skel)
        self._thread = QThread(self)
        self._mesh_worker.moveToThread(self._thread)
        self._thread.started.connect(self._mesh_worker.run)
        self._mesh_worker.finished.connect(self._on_fit_done)
        self._mesh_worker.error.connect(self._on_fit_error)
        self._thread.start()

    def _on_fit_done(self, verts, faces, weights):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_fit.setEnabled(True)

        self._mesh_verts = verts
        self._mesh_faces = faces
        self._skinning_weights = weights

        char = self.state.current_character
        if char:
            char.mesh_verts = verts
            char.skinning_weights = weights

        n_bones = weights.shape[1] if weights is not None else 0
        self.bone_slider.setRange(0, max(0, n_bones - 1))
        self._refresh_mesh_view()
        self.status_lbl.setText(f"Mesh fitted — {len(verts)} verts, {n_bones} bones.")

    def _on_fit_error(self, msg: str):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_fit.setEnabled(True)
        self.status_lbl.setText(f"Error: {msg}")

    # ------------------------------------------------------------------
    # Normal map baking
    # ------------------------------------------------------------------

    def _bake_normals(self):
        char = self.state.current_character
        if char is None:
            self.status_lbl.setText("No character loaded.")
            return

        baker = NormalMapBaker(
            strength=self.strength_spin.value(),
            blur_radius=self.blur_spin.value(),
        )
        self.bake_progress.setVisible(True)
        self.btn_bake.setEnabled(False)

        self._bake_worker = BakeWorker(baker, char.frames)
        bake_thread = QThread(self)
        self._bake_worker.moveToThread(bake_thread)
        bake_thread.started.connect(self._bake_worker.run)
        self._bake_worker.finished.connect(self._on_bake_done)
        self._bake_worker.error.connect(self._on_bake_error)
        self._bake_thread = bake_thread
        bake_thread.start()

    def _on_bake_done(self, nm, dm):
        self._bake_thread.quit()
        self.bake_progress.setVisible(False)
        self.btn_bake.setEnabled(True)
        self._normal_maps = nm
        self._depth_maps = dm
        n_frames = nm.shape[1]
        self.nm_frame_slider.setRange(0, max(0, n_frames - 1))
        self._refresh_nm_preview()
        self.status_lbl.setText(f"Baked {n_frames} frames × 6 directions.")

    def _on_bake_error(self, msg: str):
        self._bake_thread.quit()
        self.bake_progress.setVisible(False)
        self.btn_bake.setEnabled(True)
        self.status_lbl.setText(f"Bake error: {msg}")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _on_bone_slider(self, val: int):
        self.bone_lbl.setText(str(val))
        self._refresh_mesh_view()

    def _refresh_mesh_view(self):
        if self._mesh_verts is not None and self._mesh_faces is not None:
            self._mesh_viewer.set_mesh(
                self._mesh_verts, self._mesh_faces,
                self._skinning_weights, self.bone_slider.value(),
            )

    def _refresh_nm_preview(self, _=None):
        if self._normal_maps is None:
            return
        d = self.dir_combo.currentIndex()
        f = self.nm_frame_slider.value()
        f = min(f, self._normal_maps.shape[1] - 1)
        self.nm_preview.set_normal_map(self._normal_maps[d, f])

    def _on_char_changed(self, idx: int):
        char = self.state.current_character
        if char and char.mesh_verts is not None:
            self._mesh_verts = char.mesh_verts
            self._skinning_weights = char.skinning_weights
            self._refresh_mesh_view()

    def _on_char_updated(self, idx: int):
        if idx == self.state.selected_idx:
            self._on_char_changed(idx)
