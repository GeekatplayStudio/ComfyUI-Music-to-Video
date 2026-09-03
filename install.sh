#!/usr/bin/env bash
# ===================================================================
#  ComfyUI-Music-to-Video - dependency installer (Linux/macOS)
#  Geekatplay Studio - Vladimir Chopine
#
#  Run from inside the node folder, using the same Python that runs
#  ComfyUI (activate your venv first if you use one).
# ===================================================================
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "Installing with: $PY"
"$PY" -m pip install -r requirements.txt

echo
echo "-------------------------------------------------------------------"
echo " ffmpeg check"
echo "-------------------------------------------------------------------"
echo " ffmpeg joins the clips and muxes your song into the final video."
for tool in ffmpeg ffprobe; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo " $tool : FOUND"
    else
        echo " $tool : NOT FOUND - install ffmpeg (apt install ffmpeg / brew install ffmpeg)"
    fi
done

echo
echo "Done. Restart ComfyUI, then open example_workflows/music_video_ALL_IN_ONE.json"
echo
echo "Thank you for your support! Star the project on GitHub and subscribe"
echo "on YouTube: @geekatplay  @geekatplay-ru  @v-code-studio"
