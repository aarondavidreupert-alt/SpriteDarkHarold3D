"""
Tab 6 — Export
Exports the full pipeline result: GLB mesh + skeleton + animation,
normal maps as PNGs, animation_data.json, and pose_library.json.
Also renders a WebGL-style preview (lit mesh) using pyqtgraph OpenGL.
"""

import os
import json
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QGroupBox, QCheckBox, QProgressBar,
    QScrollArea, QTextEdit, QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

from gui.main_window import AppState

try:
    import pyqtgraph.opengl as gl
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False


# -----------------------------------------------------------------------
# Export worker
# -----------------------------------------------------------------------

class ExportWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)   # list of exported file paths
    error    = pyqtSignal(str)

    def __init__(self, tasks: dict):
        super().__init__()
        self.tasks = tasks

    def run(self):
        from pipeline import GLTFExporter, NormalMapBaker
        from pipeline.pose_library import PoseLibrary

        exported = []
        try:
            t = self.tasks
            char = t.get("char")
            out_dir = t.get("out_dir", "output")
            os.makedirs(out_dir, exist_ok=True)

            if t.get("export_glb") and char and char.skeleton_3d is not None:
                self.progress.emit("Exporting GLB…")
                exporter = GLTFExporter(fps=t.get("fps", 10.0))
                glb_path = os.path.join(out_dir, f"{char.name}.glb")
                exporter.export_glb(
                    output_path=glb_path,
                    vertices=char.mesh_verts if char.mesh_verts is not None else _stub_verts(char.skeleton_3d[0]),
                    faces=_stub_faces(char.mesh_verts) if char.mesh_verts is not None else np.array([[0, 1, 2]], dtype=int),
                    skeleton_sequence=char.skeleton_3d,
                    skinning_weights=char.skinning_weights,
                )
                exported.append(glb_path)

            if t.get("export_normals"):
                self.progress.emit("Exporting normal maps…")
                baker = NormalMapBaker(strength=t.get("nm_strength", 2.0))
                nm_dir = os.path.join(out_dir, "normal_maps")
                if char is not None:
                    nm, _ = baker.bake_sequence(char.frames, output_dir=nm_dir)
                    exported.append(nm_dir)

            if t.get("export_anim_json") and char and char.skeleton_3d is not None:
                self.progress.emit("Exporting animation_data.json…")
                anim_path = os.path.join(out_dir, "animation_data.json")
                data = {
                    "fps": t.get("fps", 10.0),
                    "frames": int(char.skeleton_3d.shape[0]),
                    "joints": 33,
                    "animation": char.skeleton_3d.tolist(),
                    "poses_2d": char.poses_2d.tolist() if char.poses_2d is not None else [],
                }
                with open(anim_path, "w") as f:
                    json.dump(data, f, indent=2)
                exported.append(anim_path)

            if t.get("export_pose_lib"):
                self.progress.emit("Exporting pose_library.json…")
                lib_path = os.path.join(out_dir, "pose_library.json")
                t["pose_library"].save(lib_path)
                exported.append(lib_path)

            self.finished.emit(exported)

        except Exception as exc:
            self.error.emit(str(exc))


def _stub_verts(skeleton: np.ndarray) -> np.ndarray:
    """Fall-back: use landmark positions as a point cloud mesh."""
    return skeleton[~np.all(skeleton == 0, axis=1)]


def _stub_faces(verts: np.ndarray | None) -> np.ndarray:
    if verts is None or len(verts) < 3:
        return np.array([[0, 1, 2]], dtype=int)
    n = len(verts)
    return np.array([[i, (i+1) % n, (i+2) % n] for i in range(0, n - 2, 3)], dtype=int)


# -----------------------------------------------------------------------
# WebGL-style preview (lit mesh)
# -----------------------------------------------------------------------

class PreviewViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _GL_AVAILABLE:
            self._view = gl.GLViewWidget()
            self._view.setBackgroundColor((20, 20, 30, 255))
            self._view.setCameraPosition(distance=4, elevation=25, azimuth=30)

            grid = gl.GLGridItem()
            grid.scale(0.5, 0.5, 0.5)
            self._view.addItem(grid)

            self._mesh_item = None
            layout.addWidget(self._view)
        else:
            layout.addWidget(QLabel(
                "Install pyqtgraph + PyOpenGL for the live preview.\n"
                "Your exported .glb can be previewed at:\n"
                "gltf-viewer.donmccurdy.com"
            ))

    def set_mesh(self, verts: np.ndarray, faces: np.ndarray):
        if not _GL_AVAILABLE or verts is None or faces is None:
            return
        if self._mesh_item:
            self._view.removeItem(self._mesh_item)
            self._mesh_item = None

        if len(verts) == 0 or len(faces) == 0:
            return

        mesh_data = gl.MeshData(
            vertexes=verts.astype(np.float32),
            faces=faces.astype(np.int32),
        )
        self._mesh_item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=True,
            drawFaces=True,
            drawEdges=False,
            color=(0.6, 0.7, 0.5, 1.0),
            shader="shaded",
        )
        self._view.addItem(self._mesh_item)

    def set_skeleton(self, skeleton: np.ndarray):
        """Fallback when no mesh is available."""
        if not _GL_AVAILABLE or skeleton is None:
            return
        from pipeline.pose_triangulator import POSE_CONNECTIONS
        if self._mesh_item:
            self._view.removeItem(self._mesh_item)
            self._mesh_item = None

        for s, e in POSE_CONNECTIONS:
            if np.all(skeleton[s] == 0) or np.all(skeleton[e] == 0):
                continue
            pts = np.array([skeleton[s], skeleton[e]], dtype=np.float32)
            line = gl.GLLinePlotItem(pos=pts, color=(0.4, 0.9, 0.4, 1.0), width=2)
            self._view.addItem(line)


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class ExportTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread: QThread | None = None
        self._build_ui()

        self.state.selection_changed.connect(self._on_char_changed)
        self.state.character_updated.connect(self._on_char_updated)

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # ---- Left: export controls ----------------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        # Output directory
        dir_box = QGroupBox("Output Directory")
        dir_layout = QHBoxLayout(dir_box)
        self.out_dir_edit = QLineEdit("output")
        dir_layout.addWidget(self.out_dir_edit)
        btn_dir = QPushButton("Browse…")
        btn_dir.clicked.connect(self._pick_out_dir)
        dir_layout.addWidget(btn_dir)
        left_layout.addWidget(dir_box)

        # Export options
        opt_box = QGroupBox("Export Options")
        opt_layout = QVBoxLayout(opt_box)
        self.chk_glb       = QCheckBox("GLB (mesh + skeleton + animation)")
        self.chk_normals   = QCheckBox("Normal maps as PNG (per direction × frame)")
        self.chk_anim_json = QCheckBox("animation_data.json")
        self.chk_pose_lib  = QCheckBox("pose_library.json")
        self.chk_glb.setChecked(True)
        self.chk_normals.setChecked(True)
        self.chk_anim_json.setChecked(True)
        for chk in [self.chk_glb, self.chk_normals, self.chk_anim_json, self.chk_pose_lib]:
            opt_layout.addWidget(chk)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Animation FPS:"))
        self.fps_edit = QLineEdit("10")
        self.fps_edit.setMaximumWidth(60)
        fps_row.addWidget(self.fps_edit)
        fps_row.addStretch()
        opt_layout.addLayout(fps_row)
        left_layout.addWidget(opt_box)

        # Export button + progress
        self.btn_export = QPushButton("Export All")
        self.btn_export.setStyleSheet("font-weight: bold; font-size: 14px; padding: 8px;")
        self.btn_export.clicked.connect(self.export_glb)
        left_layout.addWidget(self.btn_export)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        # Log
        log_box = QGroupBox("Export Log")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        left_layout.addWidget(log_box, 1)

        # ---- Right: preview -----------------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)

        preview_box = QGroupBox("Preview (how it looks in DarkHarold2)")
        preview_layout = QVBoxLayout(preview_box)
        self._preview = PreviewViewer()
        preview_layout.addWidget(self._preview)

        note = QLabel(
            "Exported .glb files can be viewed at:\n"
            "gltf-viewer.donmccurdy.com"
        )
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(note)
        right_layout.addWidget(preview_box, 1)

        splitter.setSizes([420, 780])

    # ------------------------------------------------------------------

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self.out_dir_edit.setText(d)

    def _log(self, msg: str):
        self.log_view.append(msg)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_glb(self):
        char = self.state.current_character
        if char is None:
            self._log("No character selected.")
            return

        try:
            fps = float(self.fps_edit.text())
        except ValueError:
            fps = 10.0

        tasks = {
            "char":         char,
            "out_dir":      self.out_dir_edit.text().strip() or "output",
            "fps":          fps,
            "export_glb":       self.chk_glb.isChecked(),
            "export_normals":   self.chk_normals.isChecked(),
            "export_anim_json": self.chk_anim_json.isChecked(),
            "export_pose_lib":  self.chk_pose_lib.isChecked(),
            "nm_strength":  2.0,
            "pose_library": self.state.pose_library,
        }

        self.btn_export.setEnabled(False)
        self.progress.setVisible(True)
        self._log("Starting export…")

        self._worker = ExportWorker(tasks)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._log)
        self._worker.finished.connect(self._on_export_done)
        self._worker.error.connect(self._on_export_error)
        self._thread.start()

    def _on_export_done(self, paths: list):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_export.setEnabled(True)
        for p in paths:
            self._log(f"  ✓  {p}")
        self._log("Export complete.")
        self._update_preview()

    def _on_export_error(self, msg: str):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_export.setEnabled(True)
        self._log(f"Export error: {msg}")

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self):
        char = self.state.current_character
        if char is None:
            return
        if char.mesh_verts is not None and char.skeleton_3d is not None:
            from pipeline.mesh_fitter import _stub_faces
            faces = _stub_faces(char.mesh_verts)
            self._preview.set_mesh(char.mesh_verts, faces)
        elif char.skeleton_3d is not None:
            self._preview.set_skeleton(char.skeleton_3d[0])

    def _on_char_changed(self, idx: int):
        self._update_preview()

    def _on_char_updated(self, idx: int):
        if idx == self.state.selected_idx:
            self._update_preview()
