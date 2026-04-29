@echo off
setlocal
cd /d "%~dp0"

if not "%~1"=="" goto use_arg

python -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 goto no_python

python export_tool.py --capture-api
pause
exit /b %errorlevel%

:use_arg
"%~1" -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 goto bad_arg

"%~1" export_tool.py --capture-api
pause
exit /b %errorlevel%

:no_python
echo.
echo ERROR: Python was not found in this terminal.
echo Open Anaconda Prompt or Miniconda Prompt, then run capture_api.bat again.
pause
exit /b 1

:bad_arg
echo.
echo ERROR: The provided python.exe path cannot run:
echo %~1
pause
exit /b 1
