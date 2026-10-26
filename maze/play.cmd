@echo off
cd /d "%~dp0.."
".venv-maze\Scripts\python.exe" -m maze.demo --size 11 --min-route 24
if errorlevel 1 pause
