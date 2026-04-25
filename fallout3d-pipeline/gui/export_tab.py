"""
Tab 7 — Export (dual pipeline)

Pipeline A — Sprite
  Animated 3D mesh → isometric render (6 dirs × N frames)
  → palette quantisation → .frm + .png spritesheet + .json imageMap

Pipeline B — 3D
  Mesh + skeleton + animation + skin + normal maps
  → .glb (pygltflib)
  → normal_maps/ PNG per dir × frame
  → shadow_mesh/ low-poly OBJ
"""

import os
import json
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QGroupBox, QCheckBox, QProgressBar,
    QTextEdit, QSplitter, QComboBox, QSpinBox, QScrollArea,
    QFrame, QTabWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage

from gui.main_window import AppState

try:
    import pyqtgraph.opengl as gl
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False


# -----------------------------------------------------------------------
# Pipeline A worker — Sprite / FRM
# -----------------------------------------------------------------------

class PipelineAWorker(QObject):
    progress = pyqtSignal(str, int)   # message, percent
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, tasks: dict):
        super().__init__()
        self.t = tasks

    def run(self):
        from pipeline.frm_writer import (
            FRMWriter, render_isometric_frame,
            generate_shadow_mesh,
        )
        from pipeline.skin_projector import SkinProjector, generate_cylindrical_uvs
        from pipeline.isometric_camera_setup5 import IsometricCameraSetup

        t = self.t
        char    = t["char"]
        out_dir = t["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        exported = []

        try:
            # Resolve or generate UVs and texture
            self.progress.emit("Preparing mesh and texture…", 5)

            verts = char.mesh_verts
            faces = _stub_faces(verts) if verts is not None else None

            if verts is None or faces is None:
                # Fall back: just use the original sprites as-is
                frames_rgba = _sprites_to_rgba(char.frames)
            else:
                # Project sprites onto UV texture, then render isometrically
                camera_setup = IsometricCameraSetup(
                    image_size=(t.get("render_w", 200), t.get("render_h", 300))
                )
                uvs = generate_cylindrical_uvs(verts)
                skin_proj = SkinProjector(camera_setup)

                N = char.n_frames
                render_w = t.get("render_w", 200)
                render_h = t.get("render_h", 300)
                frames_rgba = np.zeros((6, N, render_h, render_w, 3), dtype=np.uint8)

                for f in range(N):
                    pct = 5 + int(60 * f / max(N, 1))
                    self.progress.emit(f"Rendering frame {f+1}/{N}…", pct)

                    # Get texture for this frame
                    tex = skin_proj.project(
                        char.frames[:, f], verts, faces, uvs,
                        tex_size=(256, 256)
                    )
                    skel = char.skeleton_3d[f] if char.skeleton_3d is not None else None
                    posed_verts = _apply_skeleton(verts, skel)

                    for d, cam in enumerate(camera_setup.camera_views):
                        rendered = render_isometric_frame(
                            posed_verts, faces, uvs, tex,
                            cam["projection"],
                            image_size=(render_h, render_w),
                        )
                        frames_rgba[d, f] = rendered[:, :, :3]

            writer = FRMWriter()
            pal_path = t.get("pal_path", "")
            if pal_path and os.path.exists(pal_path):
                writer.load_palette(pal_path)

            name = char.name.replace(" ", "_")
            fps  = t.get("fps", 10)

            if t.get("export_frm"):
                self.progress.emit("Writing .frm…", 70)
                frm_path = os.path.join(out_dir, f"{name}.frm")
                writer.write(frm_path, frames_rgba, fps=fps)
                exported.append(frm_path)

            if t.get("export_spritesheet"):
                self.progress.emit("Writing spritesheet PNG…", 80)
                sheet_path = os.path.join(out_dir, f"{name}_sheet.png")
                writer.write_spritesheet(sheet_path, frames_rgba)
                exported.append(sheet_path)

            if t.get("export_image_map"):
                self.progress.emit("Writing imageMap JSON…", 88)
                json_path = os.path.join(out_dir, f"{name}_imageMap.json")
                writer.write_image_map(json_path, frames_rgba, name=name, fps=fps)
                exported.append(json_path)

            self.progress.emit("Pipeline A complete.", 100)
            self.finished.emit(exported)

        except Exception as exc:
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# Pipeline B worker — 3D / GLB
# -----------------------------------------------------------------------

class PipelineBWorker(QObject):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, tasks: dict):
        super().__init__()
        self.t = tasks

    def run(self):
        from pipeline import GLTFExporter, NormalMapBaker
        from pipeline.frm_writer import generate_shadow_mesh
        from pipeline.mesh_fitter import save_obj

        t = self.t
        char    = t["char"]
        out_dir = t["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        exported = []

        try:
            fps = t.get("fps", 10.0)
            exporter = GLTFExporter(fps=fps)

            verts  = char.mesh_verts
            faces  = _stub_faces(verts) if verts is not None else None
            skel   = char.skeleton_3d
            name   = char.name.replace(" ", "_")

            if t.get("export_glb"):
                self.progress.emit("Generating UV coordinates…", 5)
                if verts is not None:
                    from pipeline.skin_projector import generate_cylindrical_uvs
                    uvs = generate_cylindrical_uvs(verts)
                else:
                    verts = skel[0] if skel is not None else np.zeros((3, 3))
                    faces = np.array([[0, 1, 2]])
                    uvs   = None

                self.progress.emit("Exporting GLB…", 15)
                glb_path = os.path.join(out_dir, f"{name}.glb")
                exporter.export_glb(
                    output_path=glb_path,
                    vertices=verts,
                    faces=faces,
                    skeleton_sequence=skel,
                    skinning_weights=char.skinning_weights,
                    uvs=uvs,
                )
                exported.append(glb_path)
                self.progress.emit(f"GLB written: {glb_path}", 40)

            if t.get("export_normals"):
                self.progress.emit("Baking normal maps…", 42)
                quality = t.get("quality", "medium")
                strength = {"low": 1.0, "medium": 2.0, "high": 4.0}.get(quality, 2.0)
                baker = NormalMapBaker(strength=strength)
                nm_dir = os.path.join(out_dir, "normal_maps")
                baker.bake_sequence(char.frames, output_dir=nm_dir)
                exported.append(nm_dir)
                self.progress.emit("Normal maps baked.", 70)

            if t.get("export_shadow"):
                self.progress.emit("Generating shadow mesh…", 72)
                sv = char.mesh_verts if char.mesh_verts is not None else (
                    skel[0] if skel is not None else None
                )
                if sv is not None:
                    sf = _stub_faces(sv)
                    shadow_v, shadow_f = generate_shadow_mesh(sv, sf)
                    shadow_dir = os.path.join(out_dir, "shadow_mesh")
                    os.makedirs(shadow_dir, exist_ok=True)
                    shadow_path = os.path.join(shadow_dir, f"{name}_shadow.obj")
                    save_obj(shadow_path, shadow_v, shadow_f)
                    exported.append(shadow_path)

            if t.get("export_anim_json"):
                self.progress.emit("Exporting animation JSON…", 88)
                anim_path = os.path.join(out_dir, "animation_data.json")
                exporter.export_animation_json(skel, anim_path)
                exported.append(anim_path)

            if t.get("export_pose_lib"):
                self.progress.emit("Saving pose library…", 94)
                lib_path = os.path.join(out_dir, "pose_library.json")
                t["pose_library"].save(lib_path)
                exported.append(lib_path)

            self.progress.emit("Pipeline B complete.", 100)
            self.finished.emit(exported)

        except Exception as exc:
            self.error.emit(str(exc))


# -----------------------------------------------------------------------
# Preview viewer
# -----------------------------------------------------------------------

class PreviewViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _GL_AVAILABLE:
            self._view = gl.GLViewWidget()
            self._view.setBackgroundColor((18, 18, 26, 255))
            self._view.setCameraPosition(distance=4, elevation=25, azimuth=30)
            grid = gl.GLGridItem()
            grid.scale(0.5, 0.5, 0.5)
            self._view.addItem(grid)
            self._items: list = []
            layout.addWidget(self._view)
        else:
            layout.addWidget(QLabel(
                "Install pyqtgraph + PyOpenGL for live preview.\n"
                "View exported .glb at: gltf-viewer.donmccurdy.com"
            ))

    def set_mesh(self, verts: np.ndarray, faces: np.ndarray):
        if not _GL_AVAILABLE or verts is None or faces is None or len(verts) == 0:
            return
        for item in self._items:
            self._view.removeItem(item)
        self._items = []
        try:
            md = gl.MeshData(vertexes=verts.astype(np.float32), faces=faces.astype(np.int32))
            mesh = gl.GLMeshItem(meshdata=md, smooth=True, drawFaces=True,
                                 color=(0.6, 0.7, 0.5, 1.0), shader="shaded")
            self._view.addItem(mesh)
            self._items.append(mesh)
        except Exception:
            pass

    def set_skeleton(self, skeleton: np.ndarray):
        if not _GL_AVAILABLE or skeleton is None:
            return
        from pipeline.pose_triangulator import POSE_CONNECTIONS
        for item in self._items:
            self._view.removeItem(item)
        self._items = []
        for s, e in POSE_CONNECTIONS:
            if np.all(skeleton[s] == 0) or np.all(skeleton[e] == 0):
                continue
            pts = np.array([skeleton[s], skeleton[e]], dtype=np.float32)
            line = gl.GLLinePlotItem(pos=pts, color=(0.4, 0.9, 0.4, 1.0), width=2)
            self._view.addItem(line)
            self._items.append(line)


# -----------------------------------------------------------------------
# Sprite compare widget (original vs. rendered)
# -----------------------------------------------------------------------

class SpriteCompare(QWidget):
    """Side-by-side: original sprite (left) vs. rendered/synthesised (right)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        for attr, title in [("_lbl_orig", "Original"), ("_lbl_rend", "Rendered")]:
            box = QGroupBox(title)
            vl  = QVBoxLayout(box)
            lbl = QLabel("—")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(120, 120)
            lbl.setStyleSheet("background:#111; border:1px solid #333;")
            vl.addWidget(lbl)
            layout.addWidget(box)
            setattr(self, attr, lbl)

    def set_images(self, orig: np.ndarray | None, rend: np.ndarray | None):
        for lbl, img in [(self._lbl_orig, orig), (self._lbl_rend, rend)]:
            if img is None:
                lbl.setText("—")
                continue
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            h, w = img.shape[:2]
            c = img.shape[2] if img.ndim == 3 else 1
            fmt = QImage.Format.Format_RGB888 if c == 3 else QImage.Format.Format_RGBA8888
            qimg = QImage(img.data, w, h, w * c, fmt)
            pix = QPixmap.fromImage(qimg).scaled(
                150, 150,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl.setPixmap(pix)


# -----------------------------------------------------------------------
# Tab widget
# -----------------------------------------------------------------------

class ExportTab(QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread_a: QThread | None = None
        self._thread_b: QThread | None = None
        self._last_rendered_frames: np.ndarray | None = None
        self._build_ui()
        self.state.selection_changed.connect(self._on_char_changed)
        self.state.character_updated.connect(self._on_char_updated)

    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Output directory row
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output directory:"))
        self.out_dir_edit = QLineEdit("output")
        dir_row.addWidget(self.out_dir_edit)
        btn_dir = QPushButton("Browse…")
        btn_dir.clicked.connect(self._pick_out_dir)
        dir_row.addWidget(btn_dir)
        root.addLayout(dir_row)

        # Main splitter: pipelines left, preview right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # ---- Left: two pipeline panels in a QTabWidget --------------
        pip_tabs = QTabWidget()
        splitter.addWidget(pip_tabs)

        pip_tabs.addTab(self._build_pipeline_a(), "Pipeline A — Sprite / FRM")
        pip_tabs.addTab(self._build_pipeline_b(), "Pipeline B — 3D / GLB")

        # ---- Right: preview panels -----------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)

        self._preview = PreviewViewer()
        right_layout.addWidget(self._preview, 3)

        self._sprite_compare = SpriteCompare()
        right_layout.addWidget(self._sprite_compare, 1)

        note = QLabel("View .glb at: gltf-viewer.donmccurdy.com  |  Test .frm at: play.html?artemple")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet("color: #888; font-size: 11px;")
        right_layout.addWidget(note)

        splitter.setSizes([520, 680])

    # ------------------------------------------------------------------
    # Pipeline A panel
    # ------------------------------------------------------------------

    def _build_pipeline_a(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # Render settings
        render_box = QGroupBox("Render Settings")
        rl = QVBoxLayout(render_box)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("W:"))
        self.render_w = QSpinBox(); self.render_w.setRange(32, 1024); self.render_w.setValue(200)
        size_row.addWidget(self.render_w)
        size_row.addWidget(QLabel("H:"))
        self.render_h = QSpinBox(); self.render_h.setRange(32, 1024); self.render_h.setValue(300)
        size_row.addWidget(self.render_h)
        rl.addLayout(size_row)

        pal_row = QHBoxLayout()
        pal_row.addWidget(QLabel("Palette (color.pal):"))
        self.pal_edit = QLineEdit()
        self.pal_edit.setPlaceholderText("Optional — default: greyscale")
        pal_row.addWidget(self.pal_edit)
        btn_pal = QPushButton("Browse…")
        btn_pal.clicked.connect(self._pick_pal)
        pal_row.addWidget(btn_pal)
        rl.addLayout(pal_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self.fps_a = QSpinBox(); self.fps_a.setRange(1, 60); self.fps_a.setValue(10)
        fps_row.addWidget(self.fps_a)
        fps_row.addStretch()
        rl.addLayout(fps_row)

        layout.addWidget(render_box)

        # Output options
        out_box = QGroupBox("Output")
        ol = QVBoxLayout(out_box)
        self.chk_frm        = QCheckBox(".frm   (Fallout 2 binary)")
        self.chk_frm.setChecked(True)
        self.chk_spritesheet = QCheckBox(".png   (spritesheet — 6 rows × N frames)")
        self.chk_spritesheet.setChecked(True)
        self.chk_image_map  = QCheckBox(".json  (imageMap for DarkHarold2)")
        self.chk_image_map.setChecked(True)
        ol.addWidget(self.chk_frm)
        ol.addWidget(self.chk_spritesheet)
        ol.addWidget(self.chk_image_map)
        layout.addWidget(out_box)

        # Run
        self.btn_run_a = QPushButton("▶  Run Pipeline A")
        self.btn_run_a.setStyleSheet("font-weight:bold; font-size:13px; padding:8px; background:#1e4d2b;")
        self.btn_run_a.clicked.connect(self._run_pipeline_a)
        layout.addWidget(self.btn_run_a)

        self.prog_a = QProgressBar()
        self.prog_a.setRange(0, 100)
        self.prog_a.setVisible(False)
        layout.addWidget(self.prog_a)

        self.log_a = QTextEdit(); self.log_a.setReadOnly(True); self.log_a.setMaximumHeight(160)
        layout.addWidget(self.log_a)
        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # Pipeline B panel
    # ------------------------------------------------------------------

    def _build_pipeline_b(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # Quality
        q_box = QGroupBox("Quality")
        ql = QHBoxLayout(q_box)
        ql.addWidget(QLabel("Level:"))
        self.quality_combo = QComboBox()
        for q in ["Low", "Medium", "High"]:
            self.quality_combo.addItem(q, q.lower())
        self.quality_combo.setCurrentIndex(1)
        ql.addWidget(self.quality_combo)

        ql.addWidget(QLabel("FPS:"))
        self.fps_b = QSpinBox(); self.fps_b.setRange(1, 60); self.fps_b.setValue(10)
        ql.addWidget(self.fps_b)
        layout.addWidget(q_box)

        # Output options
        out_box = QGroupBox("Output")
        ol = QVBoxLayout(out_box)
        self.chk_glb        = QCheckBox(".glb   (mesh + skeleton + animation)")
        self.chk_glb.setChecked(True)
        self.chk_normals    = QCheckBox("normal_maps/  PNG per dir × frame")
        self.chk_normals.setChecked(True)
        self.chk_shadow     = QCheckBox("shadow_mesh/  low-poly ground-projection OBJ")
        self.chk_shadow.setChecked(True)
        self.chk_anim_json  = QCheckBox("animation_data.json")
        self.chk_anim_json.setChecked(True)
        self.chk_pose_lib   = QCheckBox("pose_library.json")
        for chk in [self.chk_glb, self.chk_normals, self.chk_shadow, self.chk_anim_json, self.chk_pose_lib]:
            ol.addWidget(chk)
        layout.addWidget(out_box)

        # Run
        self.btn_run_b = QPushButton("▶  Run Pipeline B")
        self.btn_run_b.setStyleSheet("font-weight:bold; font-size:13px; padding:8px; background:#1a2e4a;")
        self.btn_run_b.clicked.connect(self._run_pipeline_b)
        layout.addWidget(self.btn_run_b)

        self.prog_b = QProgressBar()
        self.prog_b.setRange(0, 100)
        self.prog_b.setVisible(False)
        layout.addWidget(self.prog_b)

        self.log_b = QTextEdit(); self.log_b.setReadOnly(True); self.log_b.setMaximumHeight(160)
        layout.addWidget(self.log_b)
        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self.out_dir_edit.setText(d)

    def _pick_pal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Palette", "", "PAL (*.pal);;All (*)")
        if path:
            self.pal_edit.setText(path)

    def _run_pipeline_a(self):
        char = self.state.current_character
        if char is None:
            self.log_a.append("No character selected.")
            return

        tasks = {
            "char":             char,
            "out_dir":          os.path.join(self.out_dir_edit.text(), "sprite"),
            "fps":              self.fps_a.value(),
            "render_w":         self.render_w.value(),
            "render_h":         self.render_h.value(),
            "pal_path":         self.pal_edit.text().strip(),
            "export_frm":        self.chk_frm.isChecked(),
            "export_spritesheet": self.chk_spritesheet.isChecked(),
            "export_image_map":  self.chk_image_map.isChecked(),
        }

        self.btn_run_a.setEnabled(False)
        self.prog_a.setVisible(True)
        self.prog_a.setValue(0)
        self.log_a.clear()
        self.log_a.append("Starting Pipeline A…")

        self._worker_a = PipelineAWorker(tasks)
        self._thread_a = QThread(self)
        self._worker_a.moveToThread(self._thread_a)
        self._thread_a.started.connect(self._worker_a.run)
        self._worker_a.progress.connect(self._on_prog_a)
        self._worker_a.finished.connect(self._on_done_a)
        self._worker_a.error.connect(self._on_err_a)
        self._thread_a.start()

    def _on_prog_a(self, msg: str, pct: int):
        self.log_a.append(msg)
        self.prog_a.setValue(pct)

    def _on_done_a(self, paths: list):
        self._thread_a.quit()
        self.prog_a.setVisible(False)
        self.btn_run_a.setEnabled(True)
        for p in paths:
            self.log_a.append(f"  ✓  {p}")
        self.log_a.append("Pipeline A complete.")
        self._update_sprite_preview()

    def _on_err_a(self, msg: str):
        self._thread_a.quit()
        self.prog_a.setVisible(False)
        self.btn_run_a.setEnabled(True)
        self.log_a.append(f"Error: {msg}")

    def _run_pipeline_b(self):
        char = self.state.current_character
        if char is None:
            self.log_b.append("No character selected.")
            return
        if char.skeleton_3d is None:
            self.log_b.append("Triangulate skeleton first (Tab 3).")
            return

        tasks = {
            "char":          char,
            "out_dir":       os.path.join(self.out_dir_edit.text(), "3d"),
            "fps":           float(self.fps_b.value()),
            "quality":       self.quality_combo.currentData(),
            "export_glb":         self.chk_glb.isChecked(),
            "export_normals":     self.chk_normals.isChecked(),
            "export_shadow":      self.chk_shadow.isChecked(),
            "export_anim_json":   self.chk_anim_json.isChecked(),
            "export_pose_lib":    self.chk_pose_lib.isChecked(),
            "pose_library":  self.state.pose_library,
        }

        self.btn_run_b.setEnabled(False)
        self.prog_b.setVisible(True)
        self.prog_b.setValue(0)
        self.log_b.clear()
        self.log_b.append("Starting Pipeline B…")

        self._worker_b = PipelineBWorker(tasks)
        self._thread_b = QThread(self)
        self._worker_b.moveToThread(self._thread_b)
        self._thread_b.started.connect(self._worker_b.run)
        self._worker_b.progress.connect(self._on_prog_b)
        self._worker_b.finished.connect(self._on_done_b)
        self._worker_b.error.connect(self._on_err_b)
        self._thread_b.start()

    def _on_prog_b(self, msg: str, pct: int):
        self.log_b.append(msg)
        self.prog_b.setValue(pct)

    def _on_done_b(self, paths: list):
        self._thread_b.quit()
        self.prog_b.setVisible(False)
        self.btn_run_b.setEnabled(True)
        for p in paths:
            self.log_b.append(f"  ✓  {p}")
        self.log_b.append("Pipeline B complete.")
        self._update_3d_preview()

    def _on_err_b(self, msg: str):
        self._thread_b.quit()
        self.prog_b.setVisible(False)
        self.btn_run_b.setEnabled(True)
        self.log_b.append(f"Error: {msg}")

    # ------------------------------------------------------------------
    # Preview updates
    # ------------------------------------------------------------------

    def _update_3d_preview(self):
        char = self.state.current_character
        if char is None:
            return
        if char.mesh_verts is not None:
            faces = _stub_faces(char.mesh_verts)
            self._preview.set_mesh(char.mesh_verts, faces)
        elif char.skeleton_3d is not None:
            self._preview.set_skeleton(char.skeleton_3d[0])

    def _update_sprite_preview(self):
        char = self.state.current_character
        if char is None:
            return
        orig = char.frames[0, 0] if char.frames is not None else None
        self._sprite_compare.set_images(orig, None)

    def _on_char_changed(self, idx: int):
        self._update_3d_preview()
        self._update_sprite_preview()

    def _on_char_updated(self, idx: int):
        if idx == self.state.selected_idx:
            self._on_char_changed(idx)

    # ------------------------------------------------------------------
    # Public shortcut (called from toolbar)
    # ------------------------------------------------------------------

    def export_glb(self):
        self._run_pipeline_b()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _stub_faces(verts: np.ndarray | None) -> np.ndarray:
    if verts is None or len(verts) < 3:
        return np.array([[0, 1, 2]], dtype=np.int32)
    n = len(verts)
    return np.array([[i, (i+1) % n, (i+2) % n] for i in range(0, n - 2, 3)], dtype=np.int32)


def _apply_skeleton(verts: np.ndarray, skeleton: np.ndarray | None) -> np.ndarray:
    """Apply skeleton pose to mesh (identity if no skeleton or skinning)."""
    return verts


def _sprites_to_rgba(frames: np.ndarray) -> np.ndarray:
    """Convert (6, N, H, W, 3) to (6, N, H, W, 3) uint8 pass-through."""
    if frames.dtype != np.uint8:
        return np.clip(frames, 0, 255).astype(np.uint8)
    return frames
