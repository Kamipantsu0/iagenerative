@echo off
chcp 65001 >nul
echo.
echo  ============================================
echo   Systeme RAG - IA Generative
echo   Option 1 - RAG Simple
echo  ============================================
echo.
echo  [*] Demarrage de l'application...
echo  [*] Ouvrez votre navigateur sur : http://localhost:8501
echo.
cd /d "%~dp0"
streamlit run app.py
pause
