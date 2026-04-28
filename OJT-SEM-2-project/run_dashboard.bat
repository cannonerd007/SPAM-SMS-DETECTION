@echo off
cd /d "%~dp0"
python -m streamlit run 05_dashboard.py --server.port 8501
pause
