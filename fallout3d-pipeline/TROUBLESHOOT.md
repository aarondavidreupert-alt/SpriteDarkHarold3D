# Troubleshooting — Windows DLL conflicts

## Symptom
```
ImportError: DLL load failed while importing QtWidgets:
The specified procedure could not be found.
```

## Why this happens
Windows loads DLLs by searching PATH in order.  
If **any** of these are installed, their Qt DLLs end up in PATH and clash with PyQt6:

| Software | Qt version it ships |
|----------|---------------------|
| Anaconda / conda | Qt 5.x |
| PyQt5 (same env) | Qt 5.x |
| PySide2 / PySide6 | Qt 5 or 6 |
| OBS Studio | Qt 6.x (different build) |
| DaVinci Resolve | Qt 5.x |
| Autodesk products | Various |

---

## Fix A — Use the isolated venv (recommended)

```bat
cd fallout3d-pipeline
setup_env.bat      # creates .\venv with clean PyQt6
run.bat            # launches with a stripped PATH
```

`run.bat` strips everything except `System32` from PATH before launching,
so no external Qt DLLs can interfere.

---

## Fix B — Repair existing environment

```powershell
# From inside your env (activate first if using conda/venv)
pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip -y
pip install PyQt6
python -c "from PyQt6.QtWidgets import QApplication; print('OK')"
```

If that still fails, also uninstall PyQt5 / PySide2 / PySide6 from the same env:

```powershell
pip uninstall PyQt5 PySide2 PySide6 -y
pip install PyQt6
```

---

## Fix C — Conda environment

```powershell
conda create -n fallout3d python=3.11
conda activate fallout3d
pip install PyQt6 numpy opencv-python mediapipe scipy Pillow pygltflib pyqtgraph PyOpenGL
python main.py
```

---

## Fix D — Manual PATH strip (PowerShell one-liner)

Run this before `python main.py` in the same terminal:

```powershell
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
python main.py
```

---

## Checking which Qt DLL is being loaded

```powershell
# Shows which Qt6Core.dll Windows would load
where.exe Qt6Core.dll
# Or with Process Monitor (Sysinternals) — filter on "QtWidgets" + "NAME NOT FOUND"
```

---

## Visual C++ Redistributable

If none of the above helps, install the latest:
https://aka.ms/vs/17/release/vc_redist.x64.exe
