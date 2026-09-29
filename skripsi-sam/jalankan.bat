@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
start "" "%~dp0..\Simulasi Skripsi Sam.html"
python demo_server.py
