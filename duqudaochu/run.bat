@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="--inspect" goto inspect_default
if "%~1"=="-inspect" goto inspect_default

if not "%~1"=="" goto use_arg

python -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 goto no_python

python export_tool.py
pause
exit /b %errorlevel%

:use_arg
"%~1" -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 goto bad_arg

"%~1" export_tool.py
pause
exit /b %errorlevel%

:inspect_default
python -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 goto no_python

python export_tool.py --inspect
pause
exit /b %errorlevel%

:no_python
echo.
echo ERROR: Python was not found in this terminal.
echo Open Anaconda Prompt or Miniconda Prompt, then run run.bat again.
echo Or pass a full python.exe path:
echo run.bat "D:\path\to\python.exe"
pause
exit /b 1

:bad_arg
echo.
echo ERROR: The provided python.exe path cannot run:
echo %~1
pause
exit /b 1
