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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import (
    PoseTriangulator, PoseLibrary, MeshFitter,
    NormalMapBaker, GLTFExporter,
)


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
    skinning_weights: Optional[np.ndarray] = None
    upscaled_frames: Optional[np.ndarray] = None    # (6, N, H', W', 3) after upscaling
    annotated_frames: Optional[np.ndarray] = None   # (6, N, H, W, 3) with MP overlay
    source_path: Optional[str] = None               # original file path for cache naming
    color: Tuple[float, float, float] = (1.0, 0.8, 0.2)

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
        from gui.upscaler_tab import UpscalerTab
        from gui.pose_editor_tab import PoseEditorTab
        from gui.reconstruction_tab import ReconstructionTab
        from gui.pose_library_tab import PoseLibraryTab
        from gui.mesh_tab import MeshTab
        from gui.export_tab import ExportTab

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(False)

        self.tab_asset       = AssetLoaderTab(self.state, self)
        self.tab_upscaler    = UpscalerTab(self.state, self)
        self.tab_pose        = PoseEditorTab(self.state, self)
        self.tab_recon       = ReconstructionTab(self.state, self)
        self.tab_library     = PoseLibraryTab(self.state, self)
        self.tab_mesh        = MeshTab(self.state, self)
        self.tab_export      = ExportTab(self.state, self)

        self.tabs.addTab(self.tab_asset,    "1 · Asset Loader")
        self.tabs.addTab(self.tab_upscaler, "2 · Upscaler")
        self.tabs.addTab(self.tab_pose,     "3 · Pose Editor")
        self.tabs.addTab(self.tab_recon,    "4 · 3D Reconstruction")
        self.tabs.addTab(self.tab_library,  "5 · Pose Library")
        self.tabs.addTab(self.tab_mesh,     "6 · Mesh & Normals")
        self.tabs.addTab(self.tab_export,   "7 · Export")

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
            self.tabs.setCurrentIndex(2),
            self.tab_pose.run_detection(),
        ))
        tb.addAction(act_run)

        act_tri = QAction("Triangulate", self)
        act_tri.setShortcut("Ctrl+T")
        act_tri.triggered.connect(lambda: (
            self.tabs.setCurrentIndex(3),
            self.tab_recon.run_triangulation(),
        ))
        tb.addAction(act_tri)

        act_exp = QAction("Export GLB…", self)
        act_exp.setShortcut("Ctrl+E")
        act_exp.triggered.connect(lambda: (
            self.tabs.setCurrentIndex(6),
            self.tab_export.export_glb(),
        ))
        tb.addAction(act_exp)

    def _on_tab_changed(self, idx: int):
        labels = [
            "Load critter sprites (.npy / .png / .frm)",
            "Upscale frames with Real-ESRGAN",
            "Inspect and correct 2D pose landmarks",
            "Run 3D triangulation and inspect skeleton",
            "Average poses across multiple characters",
            "Fit mesh template and bake normal maps",
            "Export glTF / GLB and animation data",
        ]
        if 0 <= idx < len(labels):
            self.statusBar().showMessage(labels[idx])
