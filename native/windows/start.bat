@echo off
rem Sunk City on Windows PC VR (Oculus Rift via the Meta / Oculus OpenXR runtime, or any
rem OpenXR runtime with Vulkan). Logs to sunk-city.log next to this file.
rem Needs openxr_loader.dll next to sunk-city.exe (Khronos OpenXR SDK release, x64).
cd /d "%~dp0"
if "%EYE_SIZE%"=="" set EYE_SIZE=512
if "%MOVE%"=="" set MOVE=1
if "%WEIGHTS%"=="" if exist net\net.json set WEIGHTS=net
rem Live faders: point this at the av-controls broker, e.g.
rem set CONTROLS=ws://192.168.1.20:8090
sunk-city.exe > sunk-city.log 2>&1
