@echo off
REM Launch Fallout3D Pipeline Tool using the isolated venv.
REM If you haven't run setup_env.bat yet, do that first.

if not exist venv\Scripts\python.exe (
    echo venv not found. Run setup_env.bat first.
    pause & exit /b 1
)

REM Strip system-wide Qt DLLs from PATH to avoid DLL conflicts.
REM (Anaconda, OBS, DaVinci Resolve etc. can inject Qt5/Qt6 DLLs)
set "CLEAN_PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem"
set "PATH=%CLEAN_PATH%"

venv\Scripts\python.exe main.py %*
if errorlevel 1 (
    echo.
    echo Tool exited with an error. See output above.
    pause
)
