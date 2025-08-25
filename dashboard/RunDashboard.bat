@echo off
setlocal enabledelayedexpansion

REM ----- Jump to the folder this .bat lives in -----
cd /d "%~dp0"

REM ----- Pick a Python launcher -----
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PYTHON=py -3"
) else (
  set "PYTHON=python"
)

REM ----- Create venv if missing -----
if not exist ".venv" (
  echo [Setup] Creating virtual environment...
  %PYTHON% -m venv .venv
  if %ERRORLEVEL% NEQ 0 (
    echo [Error] Could not create a virtual environment. Is Python installed?
    pause
    exit /b 1
  )
)

REM ----- Activate venv -----
call ".venv\Scripts\activate"

REM ----- Install requirements -----
echo [Setup] Installing/upgrading dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt || (
  echo [Error] pip install failed. See messages above.
  pause
  exit /b 1
)

REM ----- Start the Flask app in a separate console so logs are visible -----
echo [Run] Starting server window...
start "Dashboard Server" cmd /k call ".venv\Scripts\activate" ^& python api\app.py

REM ----- Wait for http://127.0.0.1:5000 to respond (max ~15s) -----
powershell -NoProfile -Command ^
  "$u='http://127.0.0.1:5000';for($i=0;$i -lt 30;$i++){try{(Invoke-WebRequest -UseBasicParsing $u -TimeoutSec 1) > $null;exit 0}catch{Start-Sleep -Milliseconds 500}};exit 1"

if errorlevel 1 (
  echo [Error] The server did not come up. Check the "Dashboard Server" window for a traceback.
  pause
  exit /b 1
)

REM ----- Open the browser only after the server is ready -----
start "" http://127.0.0.1:5000/

echo [Info] Server is running in the "Dashboard Server" window. Close it to stop.
