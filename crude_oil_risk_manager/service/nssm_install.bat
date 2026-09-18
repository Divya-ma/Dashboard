@echo off
REM ==============================================================================
REM nssm_install.bat
REM Installs the Crude Oil Risk Manager Dash app as a Windows Service using NSSM
REM (Non-Sucking Service Manager - https://nssm.cc/).
REM
REM Prerequisites:
REM   - nssm.exe available on PATH (or update NSSM_EXE below with a full path)
REM   - Python environment with project requirements installed
REM   - Run this script from an elevated (Administrator) command prompt
REM ==============================================================================

setlocal

REM ---- Configuration ----
set SERVICE_NAME=CrudeOilRiskManager
set NSSM_EXE=nssm.exe
REM PROJECT_ROOT resolves to the parent directory of this script (project root)
set PROJECT_ROOT=%~dp0..
set PYTHON_EXE=python.exe
set APP_ENTRY=ui\app.py
set LOG_DIR=%PROJECT_ROOT%\logs

REM Ensure the logs directory exists so NSSM can write stdout/stderr there
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM ---- Install the service ----
REM "nssm install <name> <exe> <args>" registers a new Windows Service that
REM launches the given executable with the given arguments.
"%NSSM_EXE%" install %SERVICE_NAME% "%PYTHON_EXE%" "%APP_ENTRY%"

REM Set the working directory so relative paths (config, db, data) resolve
REM correctly regardless of where the service process is spawned from.
"%NSSM_EXE%" set %SERVICE_NAME% AppDirectory "%PROJECT_ROOT%"

REM Redirect stdout and stderr to log files under logs/ for troubleshooting.
"%NSSM_EXE%" set %SERVICE_NAME% AppStdout "%LOG_DIR%\stdout.log"
"%NSSM_EXE%" set %SERVICE_NAME% AppStderr "%LOG_DIR%\stderr.log"

REM Rotate logs so they don't grow unbounded (NSSM appends by default).
"%NSSM_EXE%" set %SERVICE_NAME% AppRotateFiles 1
"%NSSM_EXE%" set %SERVICE_NAME% AppRotateBytes 10485760

REM If the process exits unexpectedly, wait 5000 ms before restarting it.
"%NSSM_EXE%" set %SERVICE_NAME% AppExit Default Restart
"%NSSM_EXE%" set %SERVICE_NAME% AppRestartDelay 5000

REM Set a friendly display name and description for the Services console.
"%NSSM_EXE%" set %SERVICE_NAME% DisplayName "Crude Oil Risk Manager"
"%NSSM_EXE%" set %SERVICE_NAME% Description "Standalone crude oil portfolio risk manager (Dash application)."

REM Start the service automatically when Windows boots.
"%NSSM_EXE%" set %SERVICE_NAME% Start SERVICE_AUTO_START

echo.
echo Service "%SERVICE_NAME%" installed. Start it with:
echo   nssm start %SERVICE_NAME%
echo or via the Windows Services console.

endlocal
