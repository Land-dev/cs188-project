#!/bin/bash
# Record a mission video of the drone's CV feed

echo "🎥 Starting drone mission recording..."
source ~/miniforge3/bin/activate drones
cd "$(dirname "$0")/.."
python sim.py --record --headless

if [ -f "drone_cv_mission.avi" ]; then
    echo "✅ Recording complete: 'drone_cv_mission.avi'"
    if command -v ffmpeg &> /dev/null; then
        echo "🔄 Converting and moving to assets/drone_cv_mission.mp4..."
        ffmpeg -y -i drone_cv_mission.avi -c:v libx264 -preset slow -crf 22 -c:a aac -b:a 128k assets/drone_cv_mission.mp4
        echo "✨ Final video: 'assets/drone_cv_mission.mp4'"
        rm drone_cv_mission.avi
    fi
else
    echo "❌ Error: Video file not found."
fi
