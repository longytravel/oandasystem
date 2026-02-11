@echo off
setlocal enabledelayedexpansion
title OANDA Trading System - VPS Setup
color 0A

echo.
echo ============================================================
echo    OANDA TRADING SYSTEM - ONE-CLICK VPS SETUP
echo ============================================================
echo.
echo This will:
echo   1. Install Python dependencies
echo   2. Configure your OANDA API credentials
echo   3. Install all strategies as Windows services
echo   4. Start trading
echo.
echo Re-run this anytime to add/update strategies.
echo Edit deploy\strategies.json to configure what runs.
echo.
echo ============================================================
echo.

:: Check we're in the right directory
if not exist "scripts\run_live.py" (
    echo ERROR: Run this from the oandasystem root directory.
    echo   cd C:\Trading\oandasystem
    echo   deploy\setup.bat
    pause
    exit /b 1
)

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ from python.org
    echo   Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found

:: Install deps
echo.
echo Installing dependencies...
pip install pandas numpy requests loguru pydantic pydantic-settings python-dotenv numba pandas-ta ta pyarrow fastapi uvicorn jinja2 --quiet
if errorlevel 1 (
    echo WARNING: Some packages may have failed. Continuing...
)
echo [OK] Dependencies installed

:: Setup .env
echo.
if exist ".env" (
    echo [OK] .env file already exists
) else (
    echo --- OANDA API Configuration ---
    echo.
    echo You need your OANDA Practice account API key.
    echo Get it from: https://www.oanda.com/demo-account/tpa/personal_token
    echo.
    set /p "API_KEY=Enter OANDA API Key: "
    set /p "ACCOUNT_ID=Enter OANDA Account ID (e.g. 101-004-12345678-001): "
    echo.
    set /p "TG_TOKEN=Telegram Bot Token (press Enter to skip): "
    set /p "TG_CHAT=Telegram Chat ID (press Enter to skip): "

    (
        echo OANDA_API_KEY=!API_KEY!
        echo OANDA_ACCOUNT_TYPE=practice
        echo OANDA_ACCOUNT_ID=!ACCOUNT_ID!
        echo TELEGRAM_BOT_TOKEN=!TG_TOKEN!
        echo TELEGRAM_CHAT_ID=!TG_CHAT!
    ) > .env

    echo [OK] .env created
)

:: Test OANDA connection
echo.
echo Testing OANDA connection...
python -c "from config.settings import settings; from live.oanda_client import OandaClient; c = OandaClient(); c.account_id = settings.OANDA_ACCOUNT_ID or c.get_accounts()[0]['id']; a = c.get_account_summary(); print(f'[OK] Connected - Balance: {a[\"balance\"]}, Account: {c.account_id}')"
if errorlevel 1 (
    echo ERROR: Could not connect to OANDA. Check your API key in .env
    pause
    exit /b 1
)

:: Find NSSM - check local copy, then PATH, then try to download
echo.
if exist "deploy\nssm\nssm.exe" (
    echo [OK] NSSM already downloaded
    goto :nssm_ready
)

:: Check if nssm is already in PATH (e.g. installed via winget/choco)
where nssm >nul 2>&1
if not errorlevel 1 (
    echo [OK] NSSM found in PATH
    if not exist "deploy\nssm" mkdir deploy\nssm
    REM Copy to local dir so install_services.py can find it
    for /f "delims=" %%i in ('where nssm') do copy "%%i" "deploy\nssm\nssm.exe" >nul
    goto :nssm_ready
)

:: Try winget install first (built-in Windows package manager)
echo NSSM not found. Attempting install...
echo.
echo [Try 1/3] winget install nssm...
winget install nssm --accept-package-agreements --accept-source-agreements >nul 2>&1
if not errorlevel 1 (
    REM winget may put it in PATH - check again
    where nssm >nul 2>&1
    if not errorlevel 1 (
        if not exist "deploy\nssm" mkdir deploy\nssm
        for /f "delims=" %%i in ('where nssm') do copy "%%i" "deploy\nssm\nssm.exe" >nul
        echo [OK] NSSM installed via winget
        goto :nssm_ready
    )
)

:: Try direct download from nssm.cc
echo [Try 2/3] Downloading from nssm.cc...
if not exist "deploy\nssm" mkdir deploy\nssm
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://nssm.cc/release/nssm-2.24.zip' -OutFile 'deploy\nssm\nssm.zip' -UseBasicParsing" >nul 2>&1
if not errorlevel 1 (
    powershell -Command "Expand-Archive -Path 'deploy\nssm\nssm.zip' -DestinationPath 'deploy\nssm\temp' -Force"
    copy "deploy\nssm\temp\nssm-2.24\win64\nssm.exe" "deploy\nssm\nssm.exe" >nul
    rmdir /s /q "deploy\nssm\temp"
    del "deploy\nssm\nssm.zip"
    if exist "deploy\nssm\nssm.exe" (
        echo [OK] NSSM downloaded from nssm.cc
        goto :nssm_ready
    )
)

:: All automatic methods failed - give manual instructions
echo [Try 3/3] Automatic install failed.
echo.
echo   Please install NSSM manually:
echo     Option A: choco install nssm  (if Chocolatey is installed)
echo     Option B: Download https://nssm.cc/release/nssm-2.24.zip
echo               Extract nssm-2.24\win64\nssm.exe to deploy\nssm\nssm.exe
echo.
echo   Then re-run this script.
pause
exit /b 1

:nssm_ready

:: Hand off to Python to install all strategies from strategies.json
echo.
echo ============================================================
echo    Installing strategies from deploy\strategies.json
echo ============================================================
echo.
python deploy\install_services.py
if errorlevel 1 (
    echo.
    echo ERROR: Service installation failed. Check output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    SETUP COMPLETE
echo ============================================================
echo.
echo   Dashboard: http://YOUR_VPS_IP:8080
echo.
echo   Quick commands:
echo     Status:   deploy\nssm\nssm.exe status OandaTrader_STRATEGY_ID
echo     Logs:     type instances\STRATEGY_ID\logs\stderr.log
echo     Stop:     deploy\nssm\nssm.exe stop OandaTrader_STRATEGY_ID
echo     Restart:  deploy\nssm\nssm.exe restart OandaTrader_STRATEGY_ID
echo.
echo   Add a strategy:
echo     1. Edit deploy\strategies.json
echo     2. Put config.json in instances\STRATEGY_ID\
echo     3. Re-run deploy\setup.bat
echo.
echo ============================================================
pause
