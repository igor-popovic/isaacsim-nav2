#!/usr/bin/env python3
"""
Launch Nav2 + SLAM Toolbox for Isaac Sim (no pre-built map needed).

Usage:
    ros2 launch ~/isaacsim-nav2/launch/nav2_slam_launch.py \
        params_file:=/path/to/nav2.yaml \
        slam_params_file:=/path/to/slam.yaml
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


TESTBED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(TESTBED_ROOT, "config")
DEFAULT_NAV2_PARAMS = os.path.join(CONFIG_DIR, "nav2_navfn.yaml")
DEFAULT_SLAM_PARAMS = os.path.join(CONFIG_DIR, "slam.yaml")


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    slam_toolbox_dir = get_package_share_directory("slam_toolbox")

    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")
    slam_params_file = LaunchConfiguration("slam_params_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="True",
            description="Use simulation clock",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=DEFAULT_NAV2_PARAMS,
            description="Path to Nav2 parameters file",
        ),
        DeclareLaunchArgument(
            "slam_params_file",
            default_value=DEFAULT_SLAM_PARAMS,
            description="Path to SLAM Toolbox parameters file",
        ),

        # SLAM Toolbox
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_toolbox_dir, "launch", "online_async_launch.py")
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "slam_params_file": slam_params_file,
            }.items(),
        ),

        # Nav2 navigation stack
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "params_file": params_file,
            }.items(),
        ),
    ])
