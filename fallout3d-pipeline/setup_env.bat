@echo off
REM -----------------------------------------------------------------------
REM  Fallout3D Pipeline Tool — Windows venv setup
REM  Run once from the fallout3d-pipeline\ directory:
REM      setup_env.bat
REM  Then launch with:
REM      run.bat
REM -----------------------------------------------------------------------

setlocal EnableDelayedExpansion

echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.9+ from https://python.org
    pause & exit /b 1
)
python -c "import sys; assert sys.version_info >= (3,9), 'Python 3.9+ required'" 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.9 or newer is required.
    pause & exit /b 1
)

echo [2/5] Creating virtual environment in .\venv ...
if exist venv (
    echo   venv already exists, removing old one...
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo ERROR: Could not create venv.
    pause & exit /b 1
)

echo [3/5] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [4/5] Installing packages (this takes a minute)...
REM Install PyQt6 FIRST and alone to avoid DLL conflicts
venv\Scripts\pip.exe install "PyQt6>=6.4" --quiet
if errorlevel 1 (
    echo ERROR: PyQt6 install failed. Check internet connection.
    pause & exit /b 1
)
REM Install the rest
venv\Scripts\pip.exe install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: Some packages failed. Check output above.
)

echo [5/5] Verifying PyQt6...
venv\Scripts\python.exe -c "from PyQt6.QtWidgets import QApplication; print('  PyQt6 OK')"
if errorlevel 1 (
    echo ERROR: PyQt6 still failing after install.
    echo        See TROUBLESHOOT.md for manual fixes.
    pause & exit /b 1
)

echo.
echo =====================================================
echo  Setup complete!  Run the tool with:  run.bat
echo =====================================================
pause
