@echo off
setlocal
cd /d "%~dp0"

if not "%~1"=="" goto use_arg

python -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 goto no_python

python -m pip install -r requirements.txt
if errorlevel 1 goto fail

python -m playwright install chromium
if errorlevel 1 goto fail

echo.
echo Install finished.
pause
exit /b 0

:use_arg
"%~1" -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 goto bad_arg

"%~1" -m pip install -r requirements.txt
if errorlevel 1 goto fail

"%~1" -m playwright install chromium
if errorlevel 1 goto fail

echo.
echo Install finished.
pause
exit /b 0

:no_python
echo.
echo ERROR: Python was not found in this terminal.
echo Open Anaconda Prompt or Miniconda Prompt, then run install.bat again.
echo Or pass a full python.exe path:
echo install.bat "D:\path\to\python.exe"
pause
exit /b 1

:bad_arg
echo.
echo ERROR: The provided python.exe path cannot run:
echo %~1
pause
exit /b 1

:fail
echo.
echo ERROR: Install failed. Check the message above.
pause
exit /b 1
