@echo off
echo.
echo  ====================================================
echo   SmartParkTN - Quick Start
echo  ====================================================
echo.

echo [1/4] Running bootstrap (creating project files)...
python bootstrap.py
if %errorlevel% neq 0 goto error

echo.
echo [2/4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 goto error

echo.
echo [3/4] Seeding test vehicles...
python scripts\seed_vehicles.py
if %errorlevel% neq 0 goto error

echo.
echo [4/4] Done! Now:
echo.
echo   Terminal 1:  uvicorn main:app --reload --port 8000
echo   Terminal 2:  streamlit run ui\dashboard.py
echo   Demo:        python demo\demo.py --simulate
echo.
pause
goto end

:error
echo ERROR! Check the output above.
pause

:end
