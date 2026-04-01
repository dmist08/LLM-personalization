@echo off
REM ============================================================
REM scripts/init_git.bat — Initialize git repo and make first commit
REM ============================================================
REM RUN: scripts\init_git.bat

cd /d D:\HDD\Project\DL

echo === Initializing Git Repository ===
git init

echo === Adding all files ===
git add .

echo === First commit ===
git commit -m "feat: project scaffold - config, utils, data pipeline, scraping infra"

echo.
echo === Status ===
git status
git log --oneline -5

echo.
echo === DONE ===
echo Next: Create a GitHub repo and run:
echo   git remote add origin https://github.com/YOUR_USERNAME/cold-start-stylevector.git
echo   git push -u origin main
pause
