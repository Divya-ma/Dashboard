@echo off
REM ==============================================================================
REM nssm_uninstall.bat
REM Stops and removes the Crude Oil Risk Manager Windows Service installed via
REM nssm_install.bat. Run from an elevated (Administrator) command prompt.
REM ==============================================================================

setlocal

set SERVICE_NAME=CrudeOilRiskManager
set NSSM_EXE=nssm.exe

REM Stop the service if it is currently running.
"%NSSM_EXE%" stop %SERVICE_NAME%

REM Remove the service registration. The "confirm" flag skips the
REM interactive confirmation prompt NSSM shows by default.
"%NSSM_EXE%" remove %SERVICE_NAME% confirm

echo.
echo Service "%SERVICE_NAME%" removed.

endlocal
