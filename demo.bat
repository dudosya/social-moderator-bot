@echo off
echo =======================================================
echo.
echo      Running Intelligent Triage Assistant Demo
echo.
echo =======================================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Running the main application...
echo This may take a moment to load the AI models.
echo.

python -m app.main --url "https://www.youtube.com/watch?v=aZE5KPgiHIE"

echo.
echo =======================================================
echo.
echo      Demo script finished.
echo      Check the /reports/ folder for the output CSV.
echo.
echo =======================================================

pause