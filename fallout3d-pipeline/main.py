"""
Fallout3D Pipeline Tool — entry point.

Quick start (Windows)
---------------------
    setup_env.bat   # once — creates isolated venv
    run.bat         # every time

Manual start (any OS)
---------------------
    pip install -r requirements.txt
    python main.py

If PyQt6 fails to import on Windows see TROUBLESHOOT.md.
"""

import sys
import os

# Package root on path regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# Early DLL-conflict check (Windows only)
# ------------------------------------------------------------------
if sys.platform == "win32":
    import ctypes, ctypes.util
    _qt6core = ctypes.util.find_library("Qt6Core")
    if _qt6core:
        try:
            _lib = ctypes.CDLL(_qt6core)
        except OSError as _e:
            print(
                "\n"
                "╔══════════════════════════════════════════════════════╗\n"
                "║  Qt6Core.dll found but could not be loaded.          ║\n"
                "║                                                      ║\n"
                "║  This usually means a conflicting Qt version is in   ║\n"
                "║  your PATH (Anaconda, PyQt5, OBS, DaVinci Resolve…)  ║\n"
                "║                                                      ║\n"
                "║  Fix:  run  run.bat  instead of  python main.py      ║\n"
                "║  or see  TROUBLESHOOT.md  for manual steps.          ║\n"
                "╚══════════════════════════════════════════════════════╝\n",
                file=sys.stderr,
            )

# ------------------------------------------------------------------
# PyQt6 import with friendly error
# ------------------------------------------------------------------
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
except ImportError as _exc:
    _msg = str(_exc)
    print(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║  Could not import PyQt6.                             ║\n"
        "╠══════════════════════════════════════════════════════╣\n"
        f"║  {_msg[:52].ljust(52)} ║\n"
        "╠══════════════════════════════════════════════════════╣\n"
        "║  Windows DLL conflict?  → run  run.bat               ║\n"
        "║  Not installed?         → pip install PyQt6           ║\n"
        "║  Full guide:              TROUBLESHOOT.md            ║\n"
        "╚══════════════════════════════════════════════════════╝\n",
        file=sys.stderr,
    )
    sys.exit(1)

from gui.main_window import MainWindow


def main():
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
    palette.setColor(QPalette.ColorRole.Window,         dark)
    palette.setColor(QPalette.ColorRole.WindowText,     text)
    palette.setColor(QPalette.ColorRole.Base,           QColor(20, 20, 24))
    palette.setColor(QPalette.ColorRole.AlternateBase,  mid)
    palette.setColor(QPalette.ColorRole.ToolTipBase,    mid)
    palette.setColor(QPalette.ColorRole.ToolTipText,    text)
    palette.setColor(QPalette.ColorRole.Text,           text)
    palette.setColor(QPalette.ColorRole.Button,         mid)
    palette.setColor(QPalette.ColorRole.ButtonText,     text)
    palette.setColor(QPalette.ColorRole.BrightText,     QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Highlight,      hi)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
