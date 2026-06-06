"""
Run navigation benchmark using Nav2 via the Isaac Sim ROS 2 bridge.

Isaac Sim side: loads the scene, spawns the robot, creates OmniGraph action
graphs (differential drive, odometry, TF, LiDAR) and runs the physics loop.

Nav2 side (launched separately by the user):
    ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True

Usage:
    ~/isaacsim/python.sh ~/etf_isaac_nav_testbed/scripts/benchmark/run_benchmark.py \
        --scene warehouse --robot turtlebot
    ~/isaacsim/python.sh ~/etf_isaac_nav_testbed/scripts/benchmark/run_benchmark.py \
        --scene warehouse --robot carter --headless --timeout 120
"""

import os
import sys
import argparse
import json
import time
import math

# ---------------------------------------------------------------------------
# Parse CLI args before SimulationApp so --help works without GPU
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Nav2 Navigation Benchmark")
parser.add_argument(
    "--scene", type=str, default="warehouse",
    choices=["warehouse", "warehouse_digital_twin", "warehouse_full",
             "warehouse_forklifts", "simple_room", "hospital", "office"],
    help="Scene to load",
)
parser.add_argument(
    "--robot", type=str, default="carter",
    choices=["carter", "carter_lidar", "turtlebot"],
    help="Robot to spawn",
)
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--timeout", type=float, default=180.0,
                    help="Max seconds per goal before timeout")
parser.add_argument("--goal-tolerance", type=float, default=0.3,
                    help="Distance (m) to consider goal reached (fallback check)")
parser.add_argument("--goals-file", type=str, default=None,
                    help="JSON file with custom goal waypoints")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Directory for benchmark logs")
parser.add_argument("--log-interval", type=int, default=30,
                    help="Log pose every N simulation steps")
parser.add_argument("--nav2-action", type=str, default="navigate_to_pose",
                    help="Nav2 action server name")
parser.add_argument("--no-lidar", action="store_true",
                    help="Skip LiDAR sensor setup (if robot already has one)")
parser.add_argument("--no-imu", action="store_true",
                    help="Skip IMU sensor setup")
parser.add_argument("--no-depth", action="store_true",
                    help="Skip depth camera setup")
parser.add_argument("--obstacles", type=int, default=0,
                    help="Number of static obstacle capsules to spawn in the scene")
parser.add_argument("--save-map", action="store_true",
                    help="Save the SLAM map to maps/ after the benchmark completes")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Launch SimulationApp & enable extensions
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import carb
import numpy as np
import omni
import omni.usd
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf, Usd

# Enable the ROS 2 bridge
enable_extension("isaacsim.ros2.bridge")
omni.kit.app.get_app().update()

# Use Isaac Sim's bundled rclpy (Python 3.11 compatible) instead of
# the system /opt/ros/humble rclpy (Python 3.10).
_ISAAC_SIM_ROOT = os.environ.get("ISAAC_PATH", os.path.expanduser("~/isaacsim"))
_ISAACSIM_RCLPY = os.path.join(
    _ISAAC_SIM_ROOT, "exts", "isaacsim.ros2.bridge", "humble", "rclpy",
)

# Remove system ROS Python paths and prepend Isaac Sim's bundled one
sys.path = [p for p in sys.path if "/ros/humble/" not in p and "/ros/jazzy/" not in p]
sys.path.insert(0, _ISAACSIM_RCLPY)

# Clear ALL ROS-related modules cached from the bridge extension's failed
# import attempt (it tried the system Python 3.10 packages first).
_ros_pkgs = [
    "rclpy", "rcl_interfaces", "rosidl_generator_py", "rosidl_parser",
    "rosidl_runtime_py", "rpyutils", "action_msgs", "builtin_interfaces",
    "geometry_msgs", "nav_msgs", "nav2_msgs", "std_msgs", "sensor_msgs",
    "unique_identifier_msgs", "rosgraph_msgs", "rcutils", "lifecycle_msgs",
    "composition_interfaces", "tf2_msgs",
]
for mod_name in list(sys.modules.keys()):
    if any(mod_name == pkg or mod_name.startswith(pkg + ".") for pkg in _ros_pkgs):
        del sys.modules[mod_name]

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatus
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_logger import DataLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths relative to the Isaac Sim asset root (Nucleus / S3 CDN).
# Resolved at runtime via get_assets_root_path().
NUCLEUS_SCENES = {
    "warehouse":              "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_digital_twin": "/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
    "warehouse_full":         "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "warehouse_forklifts":    "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
    "simple_room":            "/Isaac/Environments/Simple_Room/simple_room.usd",
    "hospital":               "/Isaac/Environments/Hospital/hospital.usd",
    "office":                 "/Isaac/Environments/Office/office.usd",
}

NUCLEUS_ROBOTS = {
    "carter":       "/Isaac/Robots/NVIDIA/Carter/carter_v1.usd",
    "carter_lidar": "/Isaac/Robots/NVIDIA/Carter/carter_v1_physx_lidar.usd",
    "turtlebot":    "/Isaac/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd",
}

ROBOT_PARAMS = {
    "carter": {
        "wheel_radius": 0.04295,
        "wheel_base":   0.4132,
        "wheel_dof_names": ["left_wheel", "right_wheel"],
        "chassis_link": "base_link",        # Nav2 expects base_link
        "lidar_parent": "chassis_link",
        "lidar_offset": (0.0, 0.0, 0.45),
        "imu_parent":   "chassis_link",
        "imu_offset":   (0.0, 0.0, 0.1),
        "camera_parent": "chassis_link",
        "camera_offset": (0.15, 0.0, 0.3),
        "camera_orient": (0.0, 0.0, 0.0),     # euler XYZ degrees
    },
    "carter_lidar": {
        "wheel_radius": 0.04295,
        "wheel_base":   0.4132,
        "wheel_dof_names": ["left_wheel", "right_wheel"],
        "chassis_link": "base_link",        # Nav2 expects base_link
        "lidar_parent": "chassis_link",
        "lidar_offset": (0.0, 0.0, 0.25),
        "imu_parent":   "chassis_link",
        "imu_offset":   (0.0, 0.0, 0.1),
        "camera_parent": "chassis_link",
        "camera_offset": (0.15, 0.0, 0.3),
        "camera_orient": (0.0, 0.0, 0.0),
    },
    "turtlebot": {
        "wheel_radius": 0.033,
        "wheel_base":   0.16,
        "wheel_dof_names": ["wheel_left_joint", "wheel_right_joint"],
        "chassis_link": "base_link",           # Nav2/SLAM expect base_link
        "lidar_parent": "base_footprint",
        "lidar_offset": (0.0, 0.0, 0.25),
        "imu_parent":   "base_link",
        "imu_offset":   (0.0, 0.0, 0.05),
        "camera_parent": "base_footprint",
        "camera_offset": (0.05, 0.0, 0.25),
        "camera_orient": (0.0, 15.0, 0.0),    # slight downward tilt
    },
}

# Per-scene spawn position (x, y, z) — keeps the robot above ground and clear
# of obstacles. Scenes with origin on the ground plane need a small z offset.
SPAWN_POSITIONS = {
    "warehouse":              (0.0, 0.0, 0.0),
    "warehouse_digital_twin": (0.0, 0.0, 0.0),
    "warehouse_full":         (0.0, 0.0, 0.0),
    "warehouse_forklifts":    (0.0, 0.0, 0.0),
    "simple_room":            (2.0, 2.0, 0.5),
    "hospital":               (0.0, 0.0, 0.0),
    "office":                 (0.0, 0.0, 0.0),
}

DEFAULT_GOALS = {
    "warehouse":              [(3.0, 0.0), (5.0, 3.0), (0.0, 5.0), (0.0, 0.0)],
    "warehouse_digital_twin": [(2.0, 1.0), (4.0, 2.0), (2.0, 4.0), (0.0, 0.0)],
    "warehouse_full":         [(3.0, 0.0), (6.0, 3.0), (3.0, 6.0), (0.0, 0.0)],
    "warehouse_forklifts":    [(3.0, 0.0), (5.0, 3.0), (0.0, 5.0), (0.0, 0.0)],
    "simple_room":            [(2.0, 0.0), (2.0, 2.0), (-2.0, 2.0), (0.0, 0.0)],
    "hospital":               [(3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)],
    "office":                 [(2.0, 0.0), (4.0, 2.0), (2.0, 4.0), (0.0, 0.0)],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_goals(goals_file, scene_name):
    """Load goal waypoints from JSON file or use scene defaults."""
    if goals_file:
        with open(goals_file) as f:
            data = json.load(f)
        if scene_name in data:
            return [tuple(g) for g in data[scene_name]]
        return [tuple(g) for g in data.get("goals", data)]
    return DEFAULT_GOALS.get(scene_name, [(2.0, 0.0), (0.0, 0.0)])


def create_ros2_action_graph(robot_prim_path, params):
    """Build an OmniGraph that wires ROS 2 ↔ Isaac Sim for a diff-drive robot.

    Creates nodes for:
      - cmd_vel subscriber  → differential controller → articulation controller
      - chassis odometry    → /odom publisher + /tf (odom→base_link)
    """
    graph_path = robot_prim_path + "/ActionGraph"

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick",  "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime",     "isaacsim.core.nodes.IsaacReadSimulationTime"),
                # Sim clock (required for use_sim_time nodes)
                ("PublishClock",    "isaacsim.ros2.bridge.ROS2PublishClock"),
                # Twist subscriber + differential drive
                ("SubscribeTwist",  "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("BreakLinVel",     "omni.graph.nodes.BreakVector3"),
                ("BreakAngVel",     "omni.graph.nodes.BreakVector3"),
                ("DiffController",  "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArtController",   "isaacsim.core.nodes.IsaacArticulationController"),
                # Odometry
                ("ComputeOdom",     "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishOdom",     "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                # TF: odom → base_link
                ("PublishTF",       "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ],
            keys.CONNECT: [
                # Tick drives everything
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ComputeOdom.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishOdom.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArtController.inputs:execIn"),
                # Twist → diff controller → articulation
                ("SubscribeTwist.outputs:execOut",          "DiffController.inputs:execIn"),
                ("SubscribeTwist.outputs:linearVelocity",   "BreakLinVel.inputs:tuple"),
                ("SubscribeTwist.outputs:angularVelocity",  "BreakAngVel.inputs:tuple"),
                ("BreakLinVel.outputs:x",                   "DiffController.inputs:linearVelocity"),
                ("BreakAngVel.outputs:z",                   "DiffController.inputs:angularVelocity"),
                ("DiffController.outputs:velocityCommand",  "ArtController.inputs:velocityCommand"),
                # Odometry → publish
                ("ComputeOdom.outputs:angularVelocity",     "PublishOdom.inputs:angularVelocity"),
                ("ComputeOdom.outputs:linearVelocity",      "PublishOdom.inputs:linearVelocity"),
                ("ComputeOdom.outputs:orientation",          "PublishOdom.inputs:orientation"),
                ("ComputeOdom.outputs:position",             "PublishOdom.inputs:position"),
                # Odometry → TF (odom → base_link)
                ("ComputeOdom.outputs:orientation",          "PublishTF.inputs:rotation"),
                ("ComputeOdom.outputs:position",             "PublishTF.inputs:translation"),
                # Timestamps
                ("ReadSimTime.outputs:simulationTime",       "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime",       "PublishOdom.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime",       "PublishTF.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                # Reset sim time on stop so clock/TF/scan timestamps stay in sync
                ("ReadSimTime.inputs:resetOnStop", True),
                # Differential controller params
                ("DiffController.inputs:wheelRadius",   params["wheel_radius"]),
                ("DiffController.inputs:wheelDistance",  params["wheel_base"]),
                # Articulation controller — target robot, joint names
                ("ArtController.inputs:targetPrim",     [usdrt.Sdf.Path(robot_prim_path)]),
                ("ArtController.inputs:jointNames",     params["wheel_dof_names"]),
                # Odometry — chassis prim
                ("ComputeOdom.inputs:chassisPrim",      [usdrt.Sdf.Path(robot_prim_path)]),
                # Topics
                ("SubscribeTwist.inputs:topicName",     "cmd_vel"),
                ("PublishOdom.inputs:topicName",        "odom"),
                ("PublishTF.inputs:topicName",          "tf"),
                ("PublishTF.inputs:parentFrameId",      "odom"),
                ("PublishTF.inputs:childFrameId",       params["chassis_link"]),
            ],
        },
    )
    print(f"  OmniGraph created at {graph_path}")
    return graph_path


def create_lidar_sensor(robot_prim_path, params):
    """Create a PhysX LiDAR sensor on the robot and publish /scan via OmniGraph."""
    import omni.kit.commands

    enable_extension("isaacsim.sensors.physx")
    omni.kit.app.get_app().update()

    lidar_parent = params["lidar_parent"]
    lidar_offset = params["lidar_offset"]
    lidar_parent_path = f"{robot_prim_path}/{lidar_parent}"
    lidar_prim_path = f"{lidar_parent_path}/lidar_sensor"

    omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path="lidar_sensor",
        parent=lidar_parent_path,
        min_range=0.1,
        max_range=25.0,
        draw_points=True,
        draw_lines=True,
        horizontal_fov=360.0,
        vertical_fov=1.0,
        horizontal_resolution=0.4,    # 360/0.4 = 900 rays
        vertical_resolution=1.0,
        rotation_rate=0.0,            # all rays each frame
        translation=Gf.Vec3d(*lidar_offset),
    )
    print(f"  PhysX LiDAR created at {lidar_prim_path}")

    for _ in range(5):
        omni.kit.app.get_app().update()

    # OmniGraph: ReadLidarBeams → ROS2PublishLaserScan
    graph_path = robot_prim_path + "/LidarGraph"
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime",    "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ReadLidar",      "isaacsim.sensors.physx.IsaacReadLidarBeams"),
                ("PublishScan",    "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick",               "ReadLidar.inputs:execIn"),
                ("ReadLidar.outputs:execOut",                  "PublishScan.inputs:execIn"),
                ("ReadLidar.outputs:azimuthRange",             "PublishScan.inputs:azimuthRange"),
                ("ReadLidar.outputs:depthRange",               "PublishScan.inputs:depthRange"),
                ("ReadLidar.outputs:horizontalFov",            "PublishScan.inputs:horizontalFov"),
                ("ReadLidar.outputs:horizontalResolution",     "PublishScan.inputs:horizontalResolution"),
                ("ReadLidar.outputs:intensitiesData",          "PublishScan.inputs:intensitiesData"),
                ("ReadLidar.outputs:linearDepthData",          "PublishScan.inputs:linearDepthData"),
                ("ReadLidar.outputs:numCols",                  "PublishScan.inputs:numCols"),
                ("ReadLidar.outputs:numRows",                  "PublishScan.inputs:numRows"),
                ("ReadLidar.outputs:rotationRate",             "PublishScan.inputs:rotationRate"),
                ("ReadSimTime.outputs:simulationTime",         "PublishScan.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                ("ReadSimTime.inputs:resetOnStop", True),
                ("ReadLidar.inputs:lidarPrim",     [usdrt.Sdf.Path(lidar_prim_path)]),
                ("PublishScan.inputs:topicName",    "scan"),
                ("PublishScan.inputs:frameId",      "base_link"),
            ],
        },
    )
    print("  LaserScan publisher attached -> /scan")

    return lidar_prim_path


def create_imu_sensor(robot_prim_path, params):
    """Create an IMU sensor on the robot and publish /imu via OmniGraph."""
    import omni.kit.commands

    imu_parent = params["imu_parent"]
    imu_offset = params["imu_offset"]
    imu_prim_path = f"{robot_prim_path}/{imu_parent}/imu_sensor"

    # Create IMU sensor prim
    omni.kit.commands.execute(
        "IsaacSensorCreateImuSensor",
        path=imu_prim_path,
        parent=None,
        sensor_period=1.0 / 60.0,
        translation=Gf.Vec3d(*imu_offset),
        linear_acceleration_filter_size=1,
        angular_velocity_filter_size=1,
        orientation_filter_size=1,
    )
    print(f"  IMU sensor created at {imu_prim_path}")

    # OmniGraph: ReadIMU -> PublishImu
    graph_path = robot_prim_path + "/IMUGraph"
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime",    "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ReadIMU",        "isaacsim.sensors.physics.IsaacReadIMU"),
                ("PublishIMU",     "isaacsim.ros2.bridge.ROS2PublishImu"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick",         "ReadIMU.inputs:execIn"),
                ("ReadIMU.outputs:execOut",             "PublishIMU.inputs:execIn"),
                ("ReadIMU.outputs:linAcc",              "PublishIMU.inputs:linearAcceleration"),
                ("ReadIMU.outputs:angVel",              "PublishIMU.inputs:angularVelocity"),
                ("ReadIMU.outputs:orientation",          "PublishIMU.inputs:orientation"),
                ("ReadSimTime.outputs:simulationTime",   "PublishIMU.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                ("ReadIMU.inputs:imuPrim",              [usdrt.Sdf.Path(imu_prim_path)]),
                ("ReadIMU.inputs:readGravity",          True),
                ("PublishIMU.inputs:topicName",          "imu"),
                ("PublishIMU.inputs:frameId",            "imu_link"),
                ("PublishIMU.inputs:publishLinearAcceleration", True),
                ("PublishIMU.inputs:publishAngularVelocity",   True),
                ("PublishIMU.inputs:publishOrientation",        True),
            ],
        },
    )
    print(f"  IMU publisher -> /imu")
    return imu_prim_path


def create_depth_camera(robot_prim_path, params):
    """Create a depth camera on the robot and publish /depth + /camera_info."""
    camera_parent = params["camera_parent"]
    camera_offset = params["camera_offset"]
    camera_orient = params["camera_orient"]
    camera_prim_path = f"{robot_prim_path}/{camera_parent}/depth_camera"

    stage = omni.usd.get_context().get_stage()

    # Create camera prim
    camera_prim = UsdGeom.Camera.Define(stage, camera_prim_path)
    camera_prim.GetFocalLengthAttr().Set(18.0)       # mm — wide angle
    camera_prim.GetHorizontalApertureAttr().Set(20.955)
    camera_prim.GetVerticalApertureAttr().Set(15.2908)
    camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 20.0))

    # Position and orient the camera on the robot
    xform = UsdGeom.XformCommonAPI(camera_prim)
    xform.SetTranslate(Gf.Vec3d(*camera_offset))
    xform.SetRotate(
        Gf.Vec3f(*camera_orient),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    print(f"  Depth camera created at {camera_prim_path}")
    omni.kit.app.get_app().update()

    # OmniGraph: viewport render product -> ROS2CameraHelper (depth) + CameraInfo
    graph_path = robot_prim_path + "/DepthCameraGraph"
    keys = og.Controller.Keys
    (graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick",           "omni.graph.action.OnTick"),
                ("CreateViewport",   "isaacsim.core.nodes.IsaacCreateViewport"),
                ("GetRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
                ("SetCamera",        "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
                ("DepthHelper",      "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("InfoHelper",       "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick",                             "CreateViewport.inputs:execIn"),
                ("CreateViewport.outputs:execOut",                  "GetRenderProduct.inputs:execIn"),
                ("CreateViewport.outputs:viewport",                "GetRenderProduct.inputs:viewport"),
                ("GetRenderProduct.outputs:execOut",               "SetCamera.inputs:execIn"),
                ("GetRenderProduct.outputs:renderProductPath",     "SetCamera.inputs:renderProductPath"),
                ("SetCamera.outputs:execOut",                      "DepthHelper.inputs:execIn"),
                ("SetCamera.outputs:execOut",                      "InfoHelper.inputs:execIn"),
                ("GetRenderProduct.outputs:renderProductPath",     "DepthHelper.inputs:renderProductPath"),
                ("GetRenderProduct.outputs:renderProductPath",     "InfoHelper.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                ("CreateViewport.inputs:viewportId",  1),
                ("SetCamera.inputs:cameraPrim",       [usdrt.Sdf.Path(camera_prim_path)]),
                ("DepthHelper.inputs:type",           "depth"),
                ("DepthHelper.inputs:topicName",      "depth"),
                ("DepthHelper.inputs:frameId",        "camera_depth"),
                ("InfoHelper.inputs:topicName",       "camera_info"),
                ("InfoHelper.inputs:frameId",         "camera_depth"),
            ],
        },
    )
    # Evaluate once to set up the render pipeline
    og.Controller.evaluate_sync(graph)
    print(f"  Depth camera publisher -> /depth + /camera_info")
    return camera_prim_path


# ---------------------------------------------------------------------------
# Nav2 benchmark node
# ---------------------------------------------------------------------------
class Nav2BenchmarkNode(Node):
    """ROS 2 node that sends goals to Nav2 and collects results."""

    def __init__(self, action_name, timeout):
        super().__init__("nav2_benchmark")
        self._action_client = ActionClient(self, NavigateToPose, action_name)
        self._timeout = timeout

        # State
        self._goal_handle = None
        self._result_future = None
        self._status = None          # None | "active" | "succeeded" | "failed" | "timeout"
        self._feedback_distance = float("inf")
        self._goal_start_time = None

        # Odom subscriber for logging robot pose
        self._latest_position = None
        self.create_subscription(Odometry, "odom", self._odom_cb, 10)

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self._latest_position = (p.x, p.y, p.z)

    # ----- public API -----

    def wait_for_nav2(self, timeout_sec=30.0):
        """Block until the Nav2 action server is available."""
        self.get_logger().info(f"Waiting for Nav2 action server '{args.nav2_action}'...")
        if not self._action_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error("Nav2 action server not available!")
            return False
        self.get_logger().info("Nav2 action server ready.")
        return True

    def send_goal(self, x, y, yaw=0.0):
        """Send a NavigateToPose goal and return immediately."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._status = "active"
        self._feedback_distance = float("inf")
        self._goal_start_time = time.time()

        future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self._feedback_cb,
        )
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().warn("Goal REJECTED by Nav2")
            self._status = "failed"
            return
        self._result_future = self._goal_handle.get_result_async()
        self._result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self._feedback_distance = fb.distance_remaining

    def _result_cb(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._status = "succeeded"
        else:
            self._status = "failed"

    @property
    def is_done(self):
        if self._status in ("succeeded", "failed", "timeout"):
            return True
        # Check timeout
        if self._goal_start_time and (time.time() - self._goal_start_time) > self._timeout:
            self._status = "timeout"
            if self._goal_handle:
                self._goal_handle.cancel_goal_async()
            return True
        return False

    @property
    def status(self):
        return self._status

    @property
    def elapsed(self):
        if self._goal_start_time:
            return time.time() - self._goal_start_time
        return 0.0

    def spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.0)


def spawn_obstacles(stage, scene_root, num_obstacles, seed=42):
    """Spawn static capsule obstacles in the scene for obstacle avoidance testing."""
    import random
    rng = random.Random(seed)

    obstacle_root = f"{scene_root}/Obstacles"
    UsdGeom.Xform.Define(stage, obstacle_root)

    for i in range(num_obstacles):
        x = rng.uniform(-3.0, 3.0)
        y = rng.uniform(-3.0, 3.0)
        path = f"{obstacle_root}/obstacle_{i}"

        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))

        capsule = UsdGeom.Capsule.Define(stage, f"{path}/body")
        capsule.GetHeightAttr().Set(0.8)
        capsule.GetRadiusAttr().Set(0.2)
        capsule.GetAxisAttr().Set("Z")
        capsule_xform = UsdGeom.Xformable(capsule.GetPrim())
        capsule_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.6))

        colors = [(0.9, 0.2, 0.2), (0.2, 0.6, 0.9), (0.9, 0.7, 0.1), (0.7, 0.3, 0.8)]
        capsule.GetDisplayColorAttr().Set([Gf.Vec3f(*colors[i % len(colors)])])

        from pxr import UsdPhysics
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())

        print(f"  Obstacle {i} at ({x:.1f}, {y:.1f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Resolve scene & robot paths from Nucleus/CDN
    from isaacsim.storage.native import get_assets_root_path

    try:
        assets_root = get_assets_root_path()
    except RuntimeError:
        assets_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

    scene_path = assets_root + NUCLEUS_SCENES[args.scene]
    robot_usd = assets_root + NUCLEUS_ROBOTS[args.robot]

    params = ROBOT_PARAMS[args.robot]

    print(f"{'='*60}")
    print(f"ETF Isaac Nav Testbed — Nav2 Benchmark")
    print(f"{'='*60}")
    print(f"  Scene:   {args.scene} -> {scene_path}")
    print(f"  Robot:   {args.robot} -> {robot_usd}")
    print(f"  Timeout: {args.timeout}s per goal")
    print()

    # --- Load scene ---
    omni.usd.get_context().open_stage(scene_path)
    # Wait for remote USD to fully download and load
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        omni.kit.app.get_app().update()
    omni.kit.app.get_app().update()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print(f"ERROR: Failed to load scene: {scene_path}")
        simulation_app.close()
        return

    # --- Determine scene root prim ---
    # Scenes use different root prims (/Root, /World, etc.)
    root_children = [p for p in stage.GetPseudoRoot().GetChildren()]
    if root_children:
        scene_root = str(root_children[0].GetPath())
    else:
        scene_root = "/World"
    print(f"  Scene root prim: {scene_root}")

    # --- Spawn robot under the scene root ---
    robot_prim_path = f"{scene_root}/Robot"
    add_reference_to_stage(usd_path=robot_usd, prim_path=robot_prim_path)
    # Wait for robot USD to fully load
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        omni.kit.app.get_app().update()
    omni.kit.app.get_app().update()

    # Set spawn position (some scenes need an offset to avoid floor collision)
    spawn_pos = SPAWN_POSITIONS.get(args.scene, (0.0, 0.0, 0.0))
    if spawn_pos != (0.0, 0.0, 0.0):
        prim = stage.GetPrimAtPath(robot_prim_path)
        translate_attr = prim.GetAttribute("xformOp:translate")
        if translate_attr:
            translate_attr.Set(Gf.Vec3d(*spawn_pos))
        else:
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*spawn_pos))
        omni.kit.app.get_app().update()
        print(f"  Robot spawn offset: {spawn_pos}")

    # Debug: print robot prim children to verify structure
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    print(f"  Robot prim valid: {robot_prim.IsValid()}")
    print(f"  Robot prim type: {robot_prim.GetTypeName()}")
    print(f"  Robot children: {[str(c.GetPath()) for c in robot_prim.GetChildren()[:10]]}")

    # Spawn obstacles if requested
    if args.obstacles > 0:
        spawn_obstacles(stage, scene_root, args.obstacles)

    print("Setting up ROS 2 bridge...")

    # --- Build OmniGraph (cmd_vel, odom, TF, clock) ---
    create_ros2_action_graph(robot_prim_path, params)

    # Always publish even without subscribers (needed for startup)
    carb.settings.get_settings().set_bool(
        "/exts/isaacsim.ros2.bridge/publish_without_verification", True
    )

    # Increase PhysX GPU buffers for scenes with many collision pairs
    settings = carb.settings.get_settings()
    settings.set_int("/physics/gpu/foundLostAggregatePairsCapacity", 4096)
    settings.set_int("/physics/gpu/totalAggregatePairsCapacity", 4096)
    settings.set_int("/physics/gpu/foundLostPairsCapacity", 4096)
    settings.set_int("/physics/gpu/collisionStackSize", 67108864)

    # --- Start simulation BEFORE creating sensors ---
    # RTX LiDAR render products require the timeline to be playing
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        omni.kit.app.get_app().update()
    print("  Simulation playing")

    # --- Add sensors (after timeline.play) ---
    if not args.no_lidar:
        create_lidar_sensor(robot_prim_path, params)
    if not args.no_imu:
        create_imu_sensor(robot_prim_path, params)
    if not args.no_depth:
        create_depth_camera(robot_prim_path, params)

    # Let sensors initialize for a few frames
    for _ in range(30):
        omni.kit.app.get_app().update()

    # --- Set up logging ---
    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "data", "logs", f"{args.scene}_{args.robot}"
    )
    logger = DataLogger(output_dir)

    # --- Load goals ---
    goals = load_goals(args.goals_file, args.scene)
    print(f"  Goals: {goals}")

    # --- Initialize ROS 2 node ---
    rclpy.init()
    nav_node = Nav2BenchmarkNode(args.nav2_action, args.timeout)

    if not nav_node.wait_for_nav2(timeout_sec=60.0):
        print("\nERROR: Nav2 is not running. Please launch it first:")
        print("  ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True")
        rclpy.shutdown()
        timeline.stop()
        simulation_app.close()
        return

    # --- Wait for Nav2 to be fully active ---
    # The action server registers before the lifecycle nodes are active.
    # We need to wait for SLAM to publish the map->odom transform and
    # for the costmaps to be populated before Nav2 will accept goals.
    print("  Waiting for Nav2 to become fully active...")
    print("    (SLAM needs LiDAR data to publish the map->odom transform)")
    warmup_start = time.time()
    max_warmup = 60.0  # seconds
    odom_received = False
    while time.time() - warmup_start < max_warmup and simulation_app.is_running():
        omni.kit.app.get_app().update()
        nav_node.spin_once()
        if nav_node._latest_position is not None and not odom_received:
            odom_received = True
            print(f"    Odometry active (robot at {nav_node._latest_position})")
            break
    elapsed_warmup = time.time() - warmup_start
    if not odom_received:
        print("  WARNING: No odometry received during warm-up!")
    else:
        # Give SLAM a few more seconds to build the initial map->odom transform
        for _ in range(120):
            omni.kit.app.get_app().update()
            nav_node.spin_once()
    print(f"  Warm-up complete ({elapsed_warmup:.1f}s)")

    # --- Benchmark loop ---
    logger.start(args.scene, args.robot)
    goals_succeeded = 0
    goals_failed = 0
    step_count = 0

    max_retries = 5
    retry_warmup = 3.0  # seconds to wait between retries

    for goal_idx, goal in enumerate(goals):
        gx, gy = goal[0], goal[1]
        print(f"\n--- Goal {goal_idx + 1}/{len(goals)}: ({gx}, {gy}) ---")

        # Retry loop: Nav2 may reject goals if costmap isn't ready yet
        for attempt in range(max_retries):
            nav_node.send_goal(gx, gy)

            while simulation_app.is_running() and not nav_node.is_done:
                omni.kit.app.get_app().update()
                nav_node.spin_once()
                step_count += 1

                # Log pose at intervals
                if step_count % args.log_interval == 0 and nav_node._latest_position:
                    logger.log_pose(list(nav_node._latest_position))

            if nav_node.status != "failed" or nav_node.elapsed > 1.0:
                # Goal was accepted (succeeded, timed out, or genuinely failed
                # during execution) — don't retry
                break

            # Goal was instantly rejected — Nav2 not ready yet
            if attempt < max_retries - 1:
                print(f"  Goal rejected, retrying in {retry_warmup}s "
                      f"(attempt {attempt + 2}/{max_retries})...")
                retry_start = time.time()
                while time.time() - retry_start < retry_warmup and simulation_app.is_running():
                    omni.kit.app.get_app().update()
                    nav_node.spin_once()

        # Record result
        status = nav_node.status
        elapsed = nav_node.elapsed
        dist = nav_node._feedback_distance

        if status == "succeeded":
            print(f"  REACHED in {elapsed:.1f}s (remaining: {dist:.3f}m)")
            goals_succeeded += 1
        elif status == "timeout":
            print(f"  TIMED OUT after {elapsed:.1f}s (remaining: {dist:.3f}m)")
            goals_failed += 1
        else:
            print(f"  FAILED after {elapsed:.1f}s (status: {status})")
            goals_failed += 1

        if nav_node._latest_position:
            logger.log_pose(list(nav_node._latest_position))

        if not simulation_app.is_running():
            break

    # --- Save results ---
    all_reached = (goals_failed == 0)
    logger.finish(goal_reached=all_reached)
    logger.save_summary()
    logger.save_trajectory()

    print(f"\n{'='*60}")
    print(f"Benchmark complete: {args.scene} / {args.robot}")
    print(f"  Goals succeeded: {goals_succeeded}/{len(goals)}")
    print(f"  Goals failed:    {goals_failed}/{len(goals)}")
    print(f"  Total time:      {logger.metrics.travel_time:.2f}s")
    print(f"  Path length:     {logger.metrics.path_length:.3f}m")
    print(f"  Logs saved to:   {output_dir}")
    print(f"{'='*60}")

    # --- Save SLAM map if requested ---
    if args.save_map:
        import subprocess
        maps_dir = os.path.join(PROJECT_ROOT, "maps")
        os.makedirs(maps_dir, exist_ok=True)
        map_path = os.path.join(maps_dir, f"{args.scene}")
        print(f"  Saving map to {map_path}...")
        result = subprocess.run(
            ["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", map_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"  Map saved: {map_path}.pgm + {map_path}.yaml")
        else:
            print(f"  Map save failed: {result.stderr.strip()}")

    # --- Cleanup ---
    nav_node.destroy_node()
    rclpy.shutdown()
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
