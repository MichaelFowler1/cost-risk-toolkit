@echo off
rem Double-click to install cost-core for this Windows user and put a
rem shortcut to its window on the desktop. No administrator rights needed.
rem
rem It needs Python 3.9 or newer (python.org, or your organisation's software
rem centre). If a folder named "wheels" sits beside this file, it installs from
rem there without touching the network, for offline and air-gapped machines:
rem make one on a connected machine with
rem     py -m pip download "cost-core[plots]" -d wheels

setlocal
title Install cost-core
cd /d "%~dp0"
echo.
echo  Installing cost-core: cost risk, CERs, phasing, EVM and schedules in Excel.
echo  It installs for you only and needs no administrator rights.
echo.

set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>nul && set "PY=python"
if not defined PY (
    echo  Python 3.9 or newer isn't installed, or isn't on this PC's path.
    echo.
    echo  Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
    echo  or from your organisation's software centre, then double-click this file again.
    echo.
    pause
    exit /b 1
)

if exist "%~dp0wheels\" (
    echo  Installing from the wheels folder beside this file, without the network...
    %PY% -m pip install --user --upgrade --no-index --find-links "%~dp0wheels" "cost-core[plots]"
) else (
    echo  Downloading and installing from PyPI. This takes a minute or two...
    %PY% -m pip install --user --upgrade "cost-core[plots]"
)
if errorlevel 1 (
    echo.
    echo  The install didn't finish; the lines above say why. Behind a proxy, or with no
    echo  internet? See "Using it at a lab" in the cost-core documentation:
    echo  https://github.com/MichaelFowler1/cost-risk-toolkit/blob/main/docs/using-at-a-lab.md
    echo.
    pause
    exit /b 1
)

echo.
%PY% -m cost_core gui --shortcut
if errorlevel 1 (
    echo  cost-core is installed, but the desktop shortcut couldn't be made.
    echo  Open it any time with:  %PY% -m cost_core gui
    echo.
    pause
    exit /b 1
)
%PY% -c "import subprocess, sys, pathlib; w = pathlib.Path(sys.executable).with_name('pythonw.exe'); subprocess.Popen([str(w) if w.exists() else sys.executable, '-m', 'cost_core.gui'])"
echo.
echo  Done. cost-core is opening now, and the cost-core shortcut on your desktop
echo  opens it from now on. This window closes in a few seconds.
timeout /t 8 >nul
exit /b 0
