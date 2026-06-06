#!/bin/bash
# Launch the full demo stack: Isaac Sim + Nav2 + noise node + RViz
# Usage: ./scripts/benchmark/launch_demo.sh [SCENE] [ROBOT] [OBSTACLES]
#
# Requires: tmux, Isaac Sim, ROS 2 Jazzy, Nav2, SLAM Toolbox

SCENE="${1:-warehouse}"
ROBOT="${2:-turtlebot}"
OBSTACLES="${3:-6}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

SESSION="nav_demo"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null

# Clean DDS cache
rm -rf /dev/shm/fastrtps_* /tmp/fastrtps_*

# Window 0: Isaac Sim
tmux new-session -d -s "$SESSION" -n "isaac"
tmux send-keys -t "$SESSION:isaac" \
    "~/isaacsim/python.sh $PROJECT_DIR/scripts/benchmark/run_benchmark.py --scene $SCENE --robot $ROBOT --obstacles $OBSTACLES" Enter

# Window 1: Nav2 + SLAM
tmux new-window -t "$SESSION" -n "nav2"
tmux send-keys -t "$SESSION:nav2" \
    "source /opt/ros/jazzy/setup.bash && ros2 launch $PROJECT_DIR/launch/nav2_slam_launch.py" Enter

# Window 2: Noise node
tmux new-window -t "$SESSION" -n "noise"
tmux send-keys -t "$SESSION:noise" \
    "conda deactivate 2>/dev/null; source /opt/ros/jazzy/setup.bash && python3 $PROJECT_DIR/scripts/tools/add_sensor_noise.py --lidar-noise-std 0.15 --lidar-dropout 0.20 --lidar-range-bias 0.05" Enter

# Window 3: RViz
tmux new-window -t "$SESSION" -n "rviz"
tmux send-keys -t "$SESSION:rviz" \
    "source /opt/ros/jazzy/setup.bash && rviz2 -d $PROJECT_DIR/config/rviz.rviz" Enter

# Go back to first window and attach
tmux select-window -t "$SESSION:isaac"
tmux attach -t "$SESSION"
