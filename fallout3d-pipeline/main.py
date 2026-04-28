"""
Fallout3D Pipeline Tool — entry point.

Usage
-----
    python main.py

Requirements
------------
    pip install PyQt6 pyqtgraph PyOpenGL mediapipe numpy opencv-python
    pip install trimesh pygltflib scipy
"""

import sys
import os

# ---------------------------------------------------------------------------
# Compatibility shim — torchvision ≥0.17 removed functional_tensor.py but
# basicsr still imports from it.  Patch the missing module before anything
# else loads so no manual file editing is ever needed after reinstalls.
# ---------------------------------------------------------------------------
def _patch_torchvision_functional_tensor() -> None:
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401 — already exists, nothing to do
    except ImportError:
        import importlib
        import types
        from torchvision.transforms.functional import rgb_to_grayscale
        mod = types.ModuleType("torchvision.transforms.functional_tensor")
        mod.__dict__["rgb_to_grayscale"] = rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = mod
        # Back-fill on the parent package so attribute access also works
        tv_transforms = importlib.import_module("torchvision.transforms")
        tv_transforms.functional_tensor = mod  # type: ignore[attr-defined]

_patch_torchvision_functional_tensor()

# Make sure the package root is on the path regardless of cwd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from gui.main_window import MainWindow


def main():
    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Fallout3D Pipeline Tool")
    app.setOrganizationName("DarkHarold2")
    app.setStyle("Fusion")

    # Dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    dark = QColor(30, 30, 35)
    mid  = QColor(50, 50, 58)
    text = QColor(220, 220, 220)
    hi   = QColor(90, 130, 200)
    palette.setColor(QPalette.ColorRole.Window,          dark)
    palette.setColor(QPalette.ColorRole.WindowText,      text)
    palette.setColor(QPalette.ColorRole.Base,            QColor(20, 20, 24))
    palette.setColor(QPalette.ColorRole.AlternateBase,   mid)
    palette.setColor(QPalette.ColorRole.ToolTipBase,     mid)
    palette.setColor(QPalette.ColorRole.ToolTipText,     text)
    palette.setColor(QPalette.ColorRole.Text,            text)
    palette.setColor(QPalette.ColorRole.Button,          mid)
    palette.setColor(QPalette.ColorRole.ButtonText,      text)
    palette.setColor(QPalette.ColorRole.BrightText,      QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Highlight,       hi)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
