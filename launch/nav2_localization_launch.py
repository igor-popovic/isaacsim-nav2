#!/usr/bin/env python3
"""
Launch Nav2 + AMCL localization with a pre-built map.

Usage:
    ros2 launch ~/isaac_nav_diploma/launch/nav2_localization_launch.py \
        map:=/path/to/map.yaml \
        params_file:=/path/to/nav2.yaml
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


TESTBED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(TESTBED_ROOT, "config")
DEFAULT_MAP = os.path.join(TESTBED_ROOT, "maps", "warehouse.yaml")
DEFAULT_PARAMS = os.path.join(CONFIG_DIR, "nav2.yaml")


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="True",
            description="Use simulation clock",
        ),
        DeclareLaunchArgument(
            "map",
            default_value=DEFAULT_MAP,
            description="Path to map YAML file",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=DEFAULT_PARAMS,
            description="Path to Nav2 parameters file",
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, "launch", "bringup_launch.py")
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "map": map_yaml,
                "params_file": params_file,
            }.items(),
        ),
    ])
