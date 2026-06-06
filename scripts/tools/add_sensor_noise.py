"""
ROS 2 node that injects configurable noise into sensor data.

Subscribes to clean Isaac Sim topics, applies noise, and republishes.
Point Nav2 at the noisy topics to benchmark under realistic conditions.

Runs inside Isaac Sim's Python (uses bundled rclpy) or as a standalone
ROS 2 node.

Usage (inside Isaac Sim Python — for use alongside run_benchmark.py):
    ~/isaacsim/python.sh ~/etf_isaac_nav_testbed/scripts/tools/add_sensor_noise.py \
        --lidar-noise-std 0.02 --lidar-dropout 0.05

Usage (standalone ROS 2 node):
    python3 ~/etf_isaac_nav_testbed/scripts/tools/add_sensor_noise.py \
        --lidar-noise-std 0.02 --imu-accel-noise 0.05

Topic mapping:
    /scan       -> /scan_noisy
    /imu        -> /imu_noisy
    /depth      -> /depth_noisy
"""

import os
import sys
import argparse
parser = argparse.ArgumentParser(description="Sensor Noise Injection Node")

# LiDAR noise
parser.add_argument("--lidar-noise-std", type=float, default=0.01,
                    help="Gaussian noise std on LiDAR ranges (meters)")
parser.add_argument("--lidar-dropout", type=float, default=0.03,
                    help="Fraction of LiDAR rays randomly set to max range")
parser.add_argument("--lidar-range-bias", type=float, default=0.0,
                    help="Constant bias added to all ranges (meters)")
parser.add_argument("--lidar-angular-noise", type=float, default=0.0,
                    help="Gaussian noise std on ray angles (radians)")
parser.add_argument("--lidar-input", type=str, default="/scan",
                    help="Input LiDAR topic")
parser.add_argument("--lidar-output", type=str, default="/scan_noisy",
                    help="Output noisy LiDAR topic")

# IMU noise
parser.add_argument("--imu-accel-noise", type=float, default=0.01,
                    help="Gaussian noise std on accelerometer (m/s^2)")
parser.add_argument("--imu-gyro-noise", type=float, default=0.001,
                    help="Gaussian noise std on gyroscope (rad/s)")
parser.add_argument("--imu-accel-bias", type=float, default=0.0,
                    help="Constant accelerometer bias (m/s^2)")
parser.add_argument("--imu-gyro-bias", type=float, default=0.0,
                    help="Constant gyroscope bias (rad/s)")
parser.add_argument("--imu-input", type=str, default="/imu",
                    help="Input IMU topic")
parser.add_argument("--imu-output", type=str, default="/imu_noisy",
                    help="Output noisy IMU topic")

# Depth camera noise
parser.add_argument("--depth-noise-std", type=float, default=0.005,
                    help="Gaussian noise std on depth pixels (meters)")
parser.add_argument("--depth-invalid-rate", type=float, default=0.02,
                    help="Fraction of depth pixels randomly set to 0 (invalid)")
parser.add_argument("--depth-input", type=str, default="/depth",
                    help="Input depth topic")
parser.add_argument("--depth-output", type=str, default="/depth_noisy",
                    help="Output noisy depth topic")

# General
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed (None = non-deterministic)")
parser.add_argument("--disable-lidar", action="store_true",
                    help="Don't process LiDAR")
parser.add_argument("--disable-imu", action="store_true",
                    help="Don't process IMU")
parser.add_argument("--disable-depth", action="store_true",
                    help="Don't process depth camera")

args = parser.parse_args()

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from rclpy.qos import qos_profile_sensor_data


# ---------------------------------------------------------------------------
# Noise node
# ---------------------------------------------------------------------------
class SensorNoiseNode(Node):

    def __init__(self):
        super().__init__("sensor_noise")
        self.rng = np.random.default_rng(args.seed)

        active = []

        # --- LiDAR ---
        if not args.disable_lidar:
            self._scan_pub = self.create_publisher(LaserScan, args.lidar_output, qos_profile_sensor_data)
            self.create_subscription(LaserScan, args.lidar_input, self._scan_cb, qos_profile_sensor_data)
            active.append(f"LiDAR: {args.lidar_input} -> {args.lidar_output} "
                          f"(std={args.lidar_noise_std}, dropout={args.lidar_dropout})")

        # --- IMU ---
        if not args.disable_imu:
            self._imu_pub = self.create_publisher(Imu, args.imu_output, qos_profile_sensor_data)
            self.create_subscription(Imu, args.imu_input, self._imu_cb, qos_profile_sensor_data)
            active.append(f"IMU:   {args.imu_input} -> {args.imu_output} "
                          f"(accel_std={args.imu_accel_noise}, gyro_std={args.imu_gyro_noise})")

        # --- Depth ---
        if not args.disable_depth:
            self._depth_pub = self.create_publisher(Image, args.depth_output, qos_profile_sensor_data)
            self.create_subscription(Image, args.depth_input, self._depth_cb, qos_profile_sensor_data)
            active.append(f"Depth: {args.depth_input} -> {args.depth_output} "
                          f"(std={args.depth_noise_std}, invalid={args.depth_invalid_rate})")

        self.get_logger().info("Sensor Noise Node started")
        for line in active:
            self.get_logger().info(f"  {line}")
        if not active:
            self.get_logger().warn("No sensors enabled. Remove --disable-* flags to process sensor data.")

    # ----- LiDAR callback -----
    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Gaussian range noise
        if args.lidar_noise_std > 0:
            ranges += self.rng.normal(0, args.lidar_noise_std, ranges.shape).astype(np.float32)

        # Constant bias
        if args.lidar_range_bias != 0:
            ranges += args.lidar_range_bias

        # Random dropouts (set to max range — Nav2 treats these as "no obstacle")
        if args.lidar_dropout > 0:
            dropout_mask = self.rng.random(len(ranges)) < args.lidar_dropout
            ranges[dropout_mask] = msg.range_max

        # Angular noise: shift angle_min slightly each frame
        angle_min = msg.angle_min
        if args.lidar_angular_noise > 0:
            angle_min += self.rng.normal(0, args.lidar_angular_noise)

        # Clamp to valid range
        ranges = np.clip(ranges, msg.range_min, msg.range_max)

        # Publish
        noisy = LaserScan()
        noisy.header = msg.header
        noisy.angle_min = angle_min
        noisy.angle_max = msg.angle_max + (angle_min - msg.angle_min)
        noisy.angle_increment = msg.angle_increment
        noisy.time_increment = msg.time_increment
        noisy.scan_time = msg.scan_time
        noisy.range_min = msg.range_min
        noisy.range_max = msg.range_max
        noisy.ranges = ranges.tolist()
        noisy.intensities = list(msg.intensities) if msg.intensities else []

        self._scan_pub.publish(noisy)

    # ----- IMU callback -----
    def _imu_cb(self, msg: Imu):
        noisy = Imu()
        noisy.header = msg.header
        noisy.orientation = msg.orientation
        noisy.orientation_covariance = msg.orientation_covariance

        # Accelerometer noise + bias
        noisy.linear_acceleration.x = (msg.linear_acceleration.x
            + args.imu_accel_bias + self.rng.normal(0, args.imu_accel_noise))
        noisy.linear_acceleration.y = (msg.linear_acceleration.y
            + args.imu_accel_bias + self.rng.normal(0, args.imu_accel_noise))
        noisy.linear_acceleration.z = (msg.linear_acceleration.z
            + args.imu_accel_bias + self.rng.normal(0, args.imu_accel_noise))
        noisy.linear_acceleration_covariance = msg.linear_acceleration_covariance

        # Gyroscope noise + bias
        noisy.angular_velocity.x = (msg.angular_velocity.x
            + args.imu_gyro_bias + self.rng.normal(0, args.imu_gyro_noise))
        noisy.angular_velocity.y = (msg.angular_velocity.y
            + args.imu_gyro_bias + self.rng.normal(0, args.imu_gyro_noise))
        noisy.angular_velocity.z = (msg.angular_velocity.z
            + args.imu_gyro_bias + self.rng.normal(0, args.imu_gyro_noise))
        noisy.angular_velocity_covariance = msg.angular_velocity_covariance

        self._imu_pub.publish(noisy)

    # ----- Depth callback -----
    def _depth_cb(self, msg: Image):
        # Decode depth image (assumes 32FC1 encoding)
        if msg.encoding != "32FC1":
            self._depth_pub.publish(msg)  # pass through unsupported encodings
            return

        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()

        # Gaussian noise
        if args.depth_noise_std > 0:
            depth += self.rng.normal(0, args.depth_noise_std, depth.shape).astype(np.float32)

        # Random invalid pixels
        if args.depth_invalid_rate > 0:
            invalid = self.rng.random(depth.shape) < args.depth_invalid_rate
            depth[invalid] = 0.0

        depth = np.maximum(depth, 0.0)

        noisy = Image()
        noisy.header = msg.header
        noisy.height = msg.height
        noisy.width = msg.width
        noisy.encoding = msg.encoding
        noisy.is_bigendian = msg.is_bigendian
        noisy.step = msg.step
        noisy.data = depth.tobytes()

        self._depth_pub.publish(noisy)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    node = SensorNoiseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
