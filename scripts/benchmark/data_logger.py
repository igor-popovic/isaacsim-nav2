"""
Log simulation data (trajectories, sensor readings, metrics) to files.

Usage:
    ~/isaacsim/python.sh data_logger.py --output-dir ../data/logs
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class NavMetrics:
    """Navigation benchmark metrics for a single run."""
    scene_name: str = ""
    robot_name: str = ""
    goal_reached: bool = False
    travel_time: float = 0.0
    path_length: float = 0.0
    collisions: int = 0
    min_obstacle_distance: float = float("inf")
    timestamps: List[float] = field(default_factory=list)
    positions: List[tuple] = field(default_factory=list)


class DataLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = NavMetrics()
        self._start_time = None

    def start(self, scene_name, robot_name):
        self.metrics = NavMetrics(scene_name=scene_name, robot_name=robot_name)
        self._start_time = time.time()

    def log_pose(self, position, orientation=None):
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        self.metrics.timestamps.append(elapsed)
        self.metrics.positions.append(tuple(position))

    def log_collision(self):
        self.metrics.collisions += 1

    def log_obstacle_distance(self, distance):
        self.metrics.min_obstacle_distance = min(self.metrics.min_obstacle_distance, distance)

    def finish(self, goal_reached):
        self.metrics.goal_reached = goal_reached
        if self._start_time:
            self.metrics.travel_time = time.time() - self._start_time

        if len(self.metrics.positions) > 1:
            import numpy as np
            pts = np.array(self.metrics.positions)
            self.metrics.path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def save_summary(self, filename="summary.csv"):
        filepath = os.path.join(self.output_dir, filename)
        file_exists = os.path.exists(filepath)

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "scene", "robot", "goal_reached", "travel_time_s",
                    "path_length_m", "collisions", "min_obstacle_dist_m"
                ])
            writer.writerow([
                self.metrics.scene_name,
                self.metrics.robot_name,
                self.metrics.goal_reached,
                f"{self.metrics.travel_time:.2f}",
                f"{self.metrics.path_length:.3f}",
                self.metrics.collisions,
                f"{self.metrics.min_obstacle_distance:.3f}",
            ])

    def save_trajectory(self, filename="trajectory.csv"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "x", "y", "z"])
            for t, pos in zip(self.metrics.timestamps, self.metrics.positions):
                writer.writerow([f"{t:.4f}", *[f"{v:.4f}" for v in pos]])


def main():
    parser = argparse.ArgumentParser(description="Data Logger")
    parser.add_argument("--output-dir", type=str, default="../data/logs", help="Output directory")
    args = parser.parse_args()

    # Demo usage
    logger = DataLogger(args.output_dir)
    logger.start("warehouse", "carter_v1")
    logger.log_pose([0.0, 0.0, 0.0])
    logger.log_pose([1.0, 0.5, 0.0])
    logger.log_pose([2.0, 1.0, 0.0])
    logger.log_collision()
    logger.log_obstacle_distance(0.3)
    logger.finish(goal_reached=True)
    logger.save_summary()
    logger.save_trajectory()
    print(f"Logs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
