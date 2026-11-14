@echo off
cd /d "%~dp0"

py -3.12 -c "import sys" >nul 2>&1 || winget install -e --id Python.Python.3.12

if not exist .venv312\Scripts\python.exe py -3.12 -m venv .venv312

if not exist .venv312\.installed (
    .venv312\Scripts\python.exe -m pip install --upgrade pip
    .venv312\Scripts\python.exe -m pip install -r requirements.txt && echo.> .venv312\.installed
)

.venv312\Scripts\python.exe -m streamlit run app.py
pause
