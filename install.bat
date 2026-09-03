@echo off
REM ===================================================================
REM  ComfyUI-Music-to-Video - dependency installer (Windows)
REM  Geekatplay Studio - Vladimir Chopine
REM
REM  Run this from inside the node folder. It uses ComfyUI's embedded
REM  Python when present, otherwise the system Python.
REM ===================================================================
setlocal

set "EMBEDDED_PY=%~dp0..\..\..\python_embeded\python.exe"

if exist "%EMBEDDED_PY%" (
    echo Using ComfyUI's embedded Python...
    "%EMBEDDED_PY%" -m pip install -r "%~dp0requirements.txt"
) else (
    echo Embedded Python not found - using the system Python.
    echo If ComfyUI runs in a venv, activate it first, or run:
    echo    ^<your-python^> -m pip install -r requirements.txt
    python -m pip install -r "%~dp0requirements.txt"
)

echo.
echo -------------------------------------------------------------------
echo  ffmpeg check
echo -------------------------------------------------------------------
echo  ffmpeg joins the clips and muxes your song into the final video.
echo  ffprobe measures songs given as a file path.
where ffmpeg >nul 2>nul && (echo  ffmpeg  : FOUND) || (echo  ffmpeg  : NOT FOUND - get it from https://ffmpeg.org/download.html)
where ffprobe >nul 2>nul && (echo  ffprobe : FOUND) || (echo  ffprobe : NOT FOUND - ships with ffmpeg)

echo.
echo -------------------------------------------------------------------
echo  Done. Restart ComfyUI, then open one of the workflows in
echo  example_workflows\ - start with music_video_ALL_IN_ONE.json
echo -------------------------------------------------------------------
echo.
echo  Thank you for your support!  Star the project on GitHub and
echo  subscribe on YouTube: @geekatplay  @geekatplay-ru  @v-code-studio
echo.
pause
endlocal
