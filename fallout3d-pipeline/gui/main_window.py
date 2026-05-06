"""
MainWindow — top-level PyQt6 window with six pipeline tabs and shared AppState.
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QToolBar, QApplication, QSplitter,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# Pipeline backend
_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PIPELINE_DIR)
from pipeline import (
    PoseTriangulator, PoseLibrary, MeshFitter,
    NormalMapBaker, GLTFExporter,
)
from config import load_config


# -----------------------------------------------------------------------
# Shared application state
# -----------------------------------------------------------------------

CRITTER_CATEGORIES = {
    "humanoid":  {"model": "mediapipe_pose",   "template": "humanoid.obj"},
    "quadruped": {"model": "mediapipe_animal",  "template": "quadruped.obj"},
    "insectoid": {"model": "manual",            "template": None},
    "robot":     {"model": "rule_based",        "template": "robot.obj"},
    "amorphous": {"model": "manual",            "template": None},
}


@dataclass
class CharacterData:
    name: str
    category: str
    frames: np.ndarray                          # (6, N, H, W, 3)
    poses_2d: Optional[np.ndarray] = None       # (N, 6, 33, 3)
    skeleton_3d: Optional[np.ndarray] = None    # (N, 33, 3)
    confidences: Optional[np.ndarray] = None    # (N, 33)
    mesh_verts: Optional[np.ndarray] = None     # (V, 3)  rest-pose
    mesh_frames: Optional[np.ndarray] = None    # (N, V, 3) per-frame baked
    skinning_weights: Optional[np.ndarray] = None
    upscaled_frames: Optional[np.ndarray] = None    # (6, N, H', W', 3) after upscaling
    annotated_frames: Optional[np.ndarray] = None   # (6, N, H, W, 3) with MP overlay
    source_path: Optional[str] = None               # original file path for cache naming
    frm_offsets: Optional[list] = None              # [dir][frame] (ox, oy) int tuples — FRM only
    color: Tuple[float, float, float] = (1.0, 0.8, 0.2)
    skeleton: Optional[object] = None               # SkeletonBuilder instance

    @property
    def n_frames(self) -> int:
        return self.frames.shape[1] if self.frames is not None else 0


class AppState(QObject):
    """Central data store; tabs connect to its signals for updates."""

    character_added    = pyqtSignal(int)          # index
    character_removed  = pyqtSignal(int)
    character_updated  = pyqtSignal(int)          # data changed (poses, skel…)
    character_upscaled = pyqtSignal(int)          # upscaled_frames ready
    selection_changed  = pyqtSignal(int)          # selected character index
    frame_changed      = pyqtSignal(int)          # current frame index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.characters: List[CharacterData] = []
        self.pose_library = PoseLibrary()
        self.selected_idx: int = -1
        self.current_frame: int = 0
        self.assets_dir: str = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets", "templates",
        )

    # ------------------------------------------------------------------

    def add_character(self, char: CharacterData) -> int:
        idx = len(self.characters)
        self.characters.append(char)
        self.character_added.emit(idx)
        if self.selected_idx < 0:
            self.set_selected(idx)
        return idx

    def remove_character(self, idx: int):
        self.characters.pop(idx)
        self.character_removed.emit(idx)
        if self.selected_idx >= len(self.characters):
            self.selected_idx = len(self.characters) - 1
        self.selection_changed.emit(self.selected_idx)

    def set_selected(self, idx: int):
        self.selected_idx = idx
        self.selection_changed.emit(idx)

    def set_frame(self, frame: int):
        self.current_frame = frame
        self.frame_changed.emit(frame)

    @property
    def current_character(self) -> Optional[CharacterData]:
        if 0 <= self.selected_idx < len(self.characters):
            return self.characters[self.selected_idx]
        return None

    def template_path(self, category: str) -> Optional[str]:
        tmpl = CRITTER_CATEGORIES.get(category, {}).get("template")
        if tmpl is None:
            return None
        path = os.path.join(self.assets_dir, tmpl)
        return path if os.path.exists(path) else None


# -----------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fallout3D Pipeline Tool — DarkHarold2")
        self.resize(1400, 900)

        self.state = AppState(self)

        self._build_tabs()
        self._build_console()
        self._build_toolbar()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready — load a critter asset to begin.")

    # ------------------------------------------------------------------

    def _build_tabs(self):
        from gui.asset_loader_tab import AssetLoaderTab
        from gui.frm_viewer_tab import FrmViewerTab
        from gui.upscaler_tab import UpscalerTab
        from gui.pose_editor_tab import PoseEditorTab
        from gui.pose_editor_tab2 import PoseManualEditorTab
        from gui.reconstruction_tab import ReconstructionTab
        from gui.skeleton_tab import SkeletonTab
        from gui.pose_library_tab import PoseLibraryTab
        from gui.mesh_tab import MeshTab
        from gui.mesh_builder_tab import MeshBuilderTab
        from gui.voxel_sausage_tab import VoxelSausageTab
        from gui.export_tab import ExportTab

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(False)

        self.tab_asset            = AssetLoaderTab(self.state, self)
        self.tab_frm_viewer       = FrmViewerTab(self.state, self)
        self.tab_upscaler         = UpscalerTab(self.state, self)
        self.tab_pose             = PoseEditorTab(self.state, self)
        self.tab_pose_editor      = PoseManualEditorTab(self.state, self)
        self.tab_recon            = ReconstructionTab(self.state, self)
        self.tab_skeleton         = SkeletonTab(self.state, self)
        self.tab_library          = PoseLibraryTab(self.state, self)
        self.tab_mesh             = MeshTab(self.state, self)
        self.tab_mesh_builder     = MeshBuilderTab(self.state, self)
        self.tab_voxel_sausage    = VoxelSausageTab(self.state, self)
        self.tab_export           = ExportTab(self.state, self)

        self.tabs.addTab(self.tab_asset,            "1 · Asset Loader")
        self.tabs.addTab(self.tab_frm_viewer,       "1b · FRM Viewer")
        self.tabs.addTab(self.tab_upscaler,         "2 · Upscaler")
        self.tabs.addTab(self.tab_pose,             "3 · Pose Detector")
        self.tabs.addTab(self.tab_pose_editor,      "4 · Pose Editor")
        self.tabs.addTab(self.tab_recon,            "5 · 3D Reconstruction")
        self.tabs.addTab(self.tab_skeleton,         "5b · Skeleton")
        self.tabs.addTab(self.tab_library,          "6 · Pose Library")
        self.tabs.addTab(self.tab_mesh,             "7 · Mesh & Normals")
        self.tabs.addTab(self.tab_mesh_builder,     "7b · Mesh Builder")
        self.tabs.addTab(self.tab_voxel_sausage,   "7c · Voxel Sausage")
        self.tabs.addTab(self.tab_export,           "8 · Export")

        # Auto-build skeleton after triangulation completes
        self.state.character_updated.connect(self._auto_build_skeleton)

        # ▶ per-tab play buttons
        self._add_play_buttons()

        # Central widget is set in _build_console() via a QSplitter

        # Forward tab changes so status bar stays informative
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_console(self):
        import logging
        from gui.console_widget import ConsoleWidget

        self._console = ConsoleWidget(self)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self._console)
        splitter.setSizes([720, 160])
        splitter.setCollapsible(1, True)
        self.setCentralWidget(splitter)

        root = logging.getLogger()
        root.addHandler(self._console.handler)
        if root.level == logging.NOTSET or root.level > logging.DEBUG:
            root.setLevel(logging.DEBUG)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        act_load = QAction("Load Asset…", self)
        act_load.setShortcut("Ctrl+O")
        act_load.triggered.connect(lambda: (
            self.tabs.setCurrentIndex(0),
            self.tab_asset.open_file_dialog(),
        ))
        tb.addAction(act_load)

        act_run = QAction("Run Detection", self)
        act_run.setShortcut("Ctrl+R")
        act_run.triggered.connect(lambda: (
            self.tabs.setCurrentIndex(3),
            self.tab_pose.run_detection(),
        ))
        tb.addAction(act_run)

        act_tri = QAction("Triangulate", self)
        act_tri.setShortcut("Ctrl+T")
        act_tri.triggered.connect(lambda: (
            self.tabs.setCurrentIndex(5),
            self.tab_recon.run_triangulation(),
        ))
        tb.addAction(act_tri)

        act_exp = QAction("Export GLB…", self)
        act_exp.setShortcut("Ctrl+E")
        act_exp.triggered.connect(lambda: (
            self.tabs.setCurrentIndex(8),
            self.tab_export.export_glb(),
        ))
        tb.addAction(act_exp)

    def _auto_build_skeleton(self, idx: int):
        """After triangulation, auto-run SkeletonBuilder with mode='median'."""
        from pipeline.skeleton_builder import SkeletonBuilder
        char = self.state.characters[idx] if 0 <= idx < len(self.state.characters) else None
        if char is None or char.skeleton_3d is None or char.skeleton is not None:
            return
        try:
            sb = SkeletonBuilder()
            sb.build(char.skeleton_3d, mode="median")
            char.skeleton = sb
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Auto skeleton build failed: %s", exc)

    def _on_tab_changed(self, idx: int):
        labels = [
            "Load critter sprites (.npy / .png / .frm)",
            "Preview and verify FRM frame registration",
            "Upscale frames with Real-ESRGAN",
            "Run MediaPipe pose detection",
            "Manually drag and correct 2D pose landmarks",
            "Run 3D triangulation and inspect skeleton",
            "View rigid skeleton, bone lengths, and frame interpolation",
            "Average poses across multiple characters",
            "Fit mesh template and bake normal maps",
            "Fit mesh to skeleton, animate, project onto sprite views",
            "Load skeleton JSON, build ragdoll, adjust radii, save template",
            "Export glTF / GLB and animation data",
        ]
        if 0 <= idx < len(labels):
            self.statusBar().showMessage(labels[idx])

    # ------------------------------------------------------------------
    # ▶ Per-tab play buttons
    # ------------------------------------------------------------------

    def _add_play_buttons(self):
        """Attach a small ▶ button to the right side of every tab label."""
        from PyQt6.QtWidgets import QPushButton, QTabBar
        tab_bar = self.tabs.tabBar()
        for i in range(self.tabs.count()):
            btn = QPushButton("▶")
            btn.setFixedSize(22, 18)
            btn.setFlat(True)
            btn.setToolTip(f"Run pipeline up to this tab (step {i + 1})")
            btn.clicked.connect(lambda _checked, idx=i: self.run_pipeline_until(idx))
            tab_bar.setTabButton(i, QTabBar.ButtonPosition.RightSide, btn)

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------

    # Tab index constants (must match addTab order in _build_tabs)
    _TAB_ASSET          = 0
    _TAB_FRM_VIEWER     = 1
    _TAB_UPSCALER       = 2
    _TAB_POSE           = 3
    _TAB_POSE_EDITOR    = 4
    _TAB_RECON          = 5
    _TAB_SKELETON       = 6
    _TAB_LIBRARY        = 7
    _TAB_MESH           = 8
    _TAB_MESH_BUILDER   = 9
    _TAB_VOXEL_SAUSAGE  = 10
    _TAB_EXPORT         = 11

    def run_pipeline_until(self, target: int):
        """Run the full pipeline from asset-load up to and including *target* tab."""
        from PyQt6.QtCore import QEventLoop, QTimer
        from PyQt6.QtWidgets import QMessageBox

        cfg        = load_config()
        input_path = cfg.get("input_path", "").strip()
        input_mode = cfg.get("input_mode", "npy")
        upscale    = bool(cfg.get("upscale", False))

        if not input_path or not os.path.exists(input_path):
            QMessageBox.warning(
                self, "Quick-Start Config",
                "Input file not found.\n"
                "Please set a valid path in Tab 1 → Quick-Start Config.",
            )
            self.tabs.setCurrentIndex(self._TAB_ASSET)
            return

        def _wait(signal, call, timeout_ms: int = 120_000):
            """Connect signal → call → exec local event loop → disconnect."""
            loop = QEventLoop()
            def _quit(*_): loop.quit()
            signal.connect(_quit)
            call()
            QTimer.singleShot(timeout_ms, loop.quit)
            loop.exec()
            try:
                signal.disconnect(_quit)
            except RuntimeError:
                pass

        # ── Step 1: load asset ──────────────────────────────────────────
        self.tabs.setCurrentIndex(self._TAB_ASSET)
        QApplication.processEvents()
        prev_count = len(self.state.characters)
        _wait(self.state.character_added, self.tab_asset.load_default)
        if len(self.state.characters) <= prev_count:
            return  # load failed or user cancelled

        # ── Step 2: upscale (FRM + upscale=True only) ───────────────────
        if upscale and input_mode == "frm" and target >= self._TAB_UPSCALER:
            self.tabs.setCurrentIndex(self._TAB_UPSCALER)
            QApplication.processEvents()
            _wait(self.state.character_upscaled, self.tab_upscaler.run_upscale)

        # ── Step 3: pose detection ──────────────────────────────────────
        if target >= self._TAB_POSE:
            self.tabs.setCurrentIndex(self._TAB_POSE)
            QApplication.processEvents()
            _wait(self.state.character_updated, self.tab_pose.run_detection)

        # ── Step 4: triangulation ───────────────────────────────────────
        if target >= self._TAB_RECON:
            self.tabs.setCurrentIndex(self._TAB_RECON)
            QApplication.processEvents()
            _wait(self.state.character_updated, self.tab_recon.run_triangulation)

        # ── Step 5: skeleton build (synchronous) ────────────────────────
        if target >= self._TAB_SKELETON:
            self.tabs.setCurrentIndex(self._TAB_SKELETON)
            QApplication.processEvents()
            self.tab_skeleton.run_build()
            QApplication.processEvents()

        # ── Step 6: mesh fit ────────────────────────────────────────────
        if target >= self._TAB_MESH_BUILDER:
            self.tabs.setCurrentIndex(self._TAB_MESH_BUILDER)
            QApplication.processEvents()
            _wait(self.state.character_updated, self.tab_mesh_builder.run_fit)

        # ── Step 7: export ──────────────────────────────────────────────
        if target >= self._TAB_EXPORT:
            self.tabs.setCurrentIndex(self._TAB_EXPORT)
            QApplication.processEvents()
            self.tab_export.run_export()

        # ── Switch to target tab when done ──────────────────────────────
        self.tabs.setCurrentIndex(target)
