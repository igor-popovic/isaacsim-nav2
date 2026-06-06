"""
Multi-robot test: spawn N robots in a scene, each with namespaced ROS 2 topics.

Each robot gets its own:
  - /robot_N/cmd_vel  (subscriber)
  - /robot_N/odom     (publisher)
  - /robot_N/scan     (publisher)
  - /robot_N/tf       (publisher)
  - /robot_N/imu      (publisher)

Robots can be driven independently via teleop or Nav2 with namespaces.

Usage:
    ~/isaacsim/python.sh ~/etf_isaac_nav_testbed/scripts/tools/multi_robot_test.py \
        --scene warehouse --robot turtlebot --num-robots 3

    ~/isaacsim/python.sh ~/etf_isaac_nav_testbed/scripts/tools/multi_robot_test.py \
        --scene simple_room --robot carter --num-robots 2 --headless

Teleop a specific robot:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard \
        --ros-args -r /cmd_vel:=/robot_0/cmd_vel

Launch Nav2 per robot (each in its own terminal):
    ros2 launch nav2_bringup navigation_launch.py \
        use_sim_time:=True namespace:=robot_0
"""

import os
import sys
import argparse
import math

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Multi-Robot Test")
parser.add_argument(
    "--scene", type=str, default="warehouse",
    choices=["warehouse", "warehouse_digital_twin", "warehouse_full",
             "warehouse_forklifts", "simple_room", "hospital", "office"],
)
parser.add_argument(
    "--robot", type=str, default="turtlebot",
    choices=["carter", "carter_lidar", "turtlebot"],
)
parser.add_argument("--num-robots", type=int, default=2, help="Number of robots (max 8)")
parser.add_argument("--spacing", type=float, default=2.0, help="Spacing between robots (m)")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--no-lidar", action="store_true", help="Skip LiDAR (saves VRAM)")
parser.add_argument("--no-imu", action="store_true", help="Skip IMU")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Isaac Sim
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import carb
import numpy as np
import omni
import omni.usd
import omni.graph.core as og
import omni.kit.commands
import usdrt.Sdf
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf, Usd

enable_extension("isaacsim.ros2.bridge")
omni.kit.app.get_app().update()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        "wheel_radius": 0.04295, "wheel_base": 0.4132,
        "wheel_dof_names": ["left_wheel", "right_wheel"],
        "chassis_link": "base_link",
        "lidar_parent": "chassis_link", "lidar_offset": (0.0, 0.0, 0.25),
        "imu_parent": "chassis_link", "imu_offset": (0.0, 0.0, 0.1),
    },
    "carter_lidar": {
        "wheel_radius": 0.04295, "wheel_base": 0.4132,
        "wheel_dof_names": ["left_wheel", "right_wheel"],
        "chassis_link": "base_link",
        "lidar_parent": "chassis_link", "lidar_offset": (0.0, 0.0, 0.25),
        "imu_parent": "chassis_link", "imu_offset": (0.0, 0.0, 0.1),
    },
    "turtlebot": {
        "wheel_radius": 0.033, "wheel_base": 0.16,
        "wheel_dof_names": ["wheel_left_joint", "wheel_right_joint"],
        "chassis_link": "base_footprint",
        "lidar_parent": "base_footprint", "lidar_offset": (0.0, 0.0, 0.15),
        "imu_parent": "base_footprint", "imu_offset": (0.0, 0.0, 0.05),
    },
}

# Spawn positions: arrange in a line or grid
def get_spawn_positions(n, spacing):
    cols = min(n, 4)
    positions = []
    for i in range(n):
        x = (i % cols) * spacing
        y = (i // cols) * spacing
        positions.append((x, y, 0.0))
    return positions


# ---------------------------------------------------------------------------
# Per-robot OmniGraph setup (namespaced)
# ---------------------------------------------------------------------------
def setup_robot_graph(robot_prim_path, params, namespace, publish_clock=False):
    """Create the differential drive + odom + TF + clock OmniGraph with a ROS namespace."""
    graph_path = robot_prim_path + "/ActionGraph"

    keys = og.Controller.Keys

    nodes = [
        ("OnPlaybackTick",  "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime",     "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("SubscribeTwist",  "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
        ("BreakLinVel",     "omni.graph.nodes.BreakVector3"),
        ("BreakAngVel",     "omni.graph.nodes.BreakVector3"),
        ("DiffController",  "isaacsim.robot.wheeled_robots.DifferentialController"),
        ("ArtController",   "isaacsim.core.nodes.IsaacArticulationController"),
        ("ComputeOdom",     "isaacsim.core.nodes.IsaacComputeOdometry"),
        ("PublishOdom",     "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ("PublishTF",       "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
    ]

    connections = [
        ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ComputeOdom.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "PublishOdom.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ArtController.inputs:execIn"),
        ("SubscribeTwist.outputs:execOut",         "DiffController.inputs:execIn"),
        ("SubscribeTwist.outputs:linearVelocity",  "BreakLinVel.inputs:tuple"),
        ("SubscribeTwist.outputs:angularVelocity", "BreakAngVel.inputs:tuple"),
        ("BreakLinVel.outputs:x",                  "DiffController.inputs:linearVelocity"),
        ("BreakAngVel.outputs:z",                  "DiffController.inputs:angularVelocity"),
        ("DiffController.outputs:velocityCommand", "ArtController.inputs:velocityCommand"),
        ("ComputeOdom.outputs:angularVelocity",    "PublishOdom.inputs:angularVelocity"),
        ("ComputeOdom.outputs:linearVelocity",     "PublishOdom.inputs:linearVelocity"),
        ("ComputeOdom.outputs:orientation",         "PublishOdom.inputs:orientation"),
        ("ComputeOdom.outputs:position",            "PublishOdom.inputs:position"),
        ("ComputeOdom.outputs:orientation",         "PublishTF.inputs:rotation"),
        ("ComputeOdom.outputs:position",            "PublishTF.inputs:translation"),
        ("ReadSimTime.outputs:simulationTime",      "PublishOdom.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime",      "PublishTF.inputs:timeStamp"),
    ]

    values = [
        ("ReadSimTime.inputs:resetOnStop", True),
        ("DiffController.inputs:wheelRadius",  params["wheel_radius"]),
        ("DiffController.inputs:wheelDistance", params["wheel_base"]),
        ("ArtController.inputs:targetPrim",    [usdrt.Sdf.Path(robot_prim_path)]),
        ("ArtController.inputs:jointNames",    params["wheel_dof_names"]),
        ("ComputeOdom.inputs:chassisPrim",     [usdrt.Sdf.Path(robot_prim_path)]),
        ("SubscribeTwist.inputs:topicName",    "cmd_vel"),
        ("SubscribeTwist.inputs:nodeNamespace", namespace),
        ("PublishOdom.inputs:topicName",       "odom"),
        ("PublishOdom.inputs:nodeNamespace",    namespace),
        ("PublishTF.inputs:topicName",         "tf"),
        ("PublishTF.inputs:nodeNamespace",      namespace),
        ("PublishTF.inputs:parentFrameId",     f"{namespace}/odom"),
        ("PublishTF.inputs:childFrameId",      f"{namespace}/{params['chassis_link']}"),
    ]

    # Only first robot publishes /clock (shared, not namespaced)
    if publish_clock:
        nodes.append(("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"))
        connections.append(("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"))
        connections.append(("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"))

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: nodes,
            keys.CONNECT: connections,
            keys.SET_VALUES: values,
        },
    )


def setup_robot_lidar(robot_prim_path, params, namespace):
    """Add a PhysX 2D LiDAR with namespaced /scan topic."""
    enable_extension("isaacsim.sensors.physx")
    omni.kit.app.get_app().update()

    lidar_parent_path = f"{robot_prim_path}/{params['lidar_parent']}"
    lidar_prim_path = f"{lidar_parent_path}/lidar_sensor"

    omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path="lidar_sensor",
        parent=lidar_parent_path,
        min_range=0.1,
        max_range=25.0,
        draw_points=False,
        draw_lines=False,
        horizontal_fov=360.0,
        vertical_fov=1.0,
        horizontal_resolution=0.4,
        vertical_resolution=1.0,
        rotation_rate=0.0,
        translation=Gf.Vec3d(*params["lidar_offset"]),
    )

    for _ in range(5):
        omni.kit.app.get_app().update()

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
                ("PublishScan.inputs:nodeNamespace", namespace),
                ("PublishScan.inputs:frameId",      f"{namespace}/{params['chassis_link']}"),
            ],
        },
    )


def setup_robot_imu(robot_prim_path, params, namespace):
    """Add an IMU sensor with namespaced /imu topic."""
    imu_path = f"{robot_prim_path}/{params['imu_parent']}/imu_sensor"

    omni.kit.commands.execute(
        "IsaacSensorCreateImuSensor",
        path=imu_path,
        parent=None,
        sensor_period=1.0 / 60.0,
        translation=Gf.Vec3d(*params["imu_offset"]),
        linear_acceleration_filter_size=1,
        angular_velocity_filter_size=1,
        orientation_filter_size=1,
    )

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
                ("OnPlaybackTick.outputs:tick",        "ReadIMU.inputs:execIn"),
                ("ReadIMU.outputs:execOut",            "PublishIMU.inputs:execIn"),
                ("ReadIMU.outputs:linAcc",             "PublishIMU.inputs:linearAcceleration"),
                ("ReadIMU.outputs:angVel",             "PublishIMU.inputs:angularVelocity"),
                ("ReadIMU.outputs:orientation",         "PublishIMU.inputs:orientation"),
                ("ReadSimTime.outputs:simulationTime",  "PublishIMU.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                ("ReadSimTime.inputs:resetOnStop", True),
                ("ReadIMU.inputs:imuPrim",             [usdrt.Sdf.Path(imu_path)]),
                ("ReadIMU.inputs:readGravity",         True),
                ("PublishIMU.inputs:topicName",         "imu"),
                ("PublishIMU.inputs:nodeNamespace",     namespace),
                ("PublishIMU.inputs:frameId",           f"{namespace}/imu_link"),
            ],
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    num = min(args.num_robots, 8)
    params = ROBOT_PARAMS[args.robot]

    # Resolve paths from Nucleus/CDN
    try:
        from isaacsim.storage.native import get_assets_root_path
        assets_root = get_assets_root_path()
    except RuntimeError:
        assets_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

    scene_path = assets_root + NUCLEUS_SCENES[args.scene]
    robot_usd = assets_root + NUCLEUS_ROBOTS[args.robot]
    positions = get_spawn_positions(num, args.spacing)

    print(f"{'='*60}")
    print(f"ETF Isaac Nav Testbed — Multi-Robot Test")
    print(f"{'='*60}")
    print(f"  Scene:  {args.scene}")
    print(f"  Robot:  {args.robot} x{num}")
    print(f"  Spacing: {args.spacing}m")
    print()

    # Load scene
    omni.usd.get_context().open_stage(scene_path)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        omni.kit.app.get_app().update()
    omni.kit.app.get_app().update()
    stage = omni.usd.get_context().get_stage()

    # Spawn each robot
    for i in range(num):
        namespace = f"robot_{i}"
        prim_path = f"/World/{namespace}"
        pos = positions[i]

        print(f"  [{namespace}] Spawning at ({pos[0]:.1f}, {pos[1]:.1f})")
        add_reference_to_stage(usd_path=robot_usd, prim_path=prim_path)
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            omni.kit.app.get_app().update()

        # Set position
        prim = stage.GetPrimAtPath(prim_path)
        translate_attr = prim.GetAttribute("xformOp:translate")
        if translate_attr:
            translate_attr.Set(Gf.Vec3d(*pos))
        else:
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pos))
        omni.kit.app.get_app().update()

        # OmniGraph: drive + odom + TF (first robot also publishes /clock)
        setup_robot_graph(prim_path, params, namespace, publish_clock=(i == 0))

        # Sensors
        if not args.no_lidar:
            setup_robot_lidar(prim_path, params, namespace)
        if not args.no_imu:
            setup_robot_imu(prim_path, params, namespace)

        print(f"  [{namespace}] Topics: /{namespace}/cmd_vel, /{namespace}/odom, "
              f"/{namespace}/scan, /{namespace}/tf")

    carb.settings.get_settings().set_bool(
        "/exts/isaacsim.ros2.bridge/publish_without_verification", True
    )

    # Start simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        omni.kit.app.get_app().update()

    print(f"\n{'='*60}")
    print(f"All {num} robots spawned and publishing.")
    print()
    print(f"Drive a robot with teleop:")
    print(f"  ros2 run teleop_twist_keyboard teleop_twist_keyboard \\")
    print(f"      --ros-args -r /cmd_vel:=/robot_0/cmd_vel")
    print()
    print(f"Launch Nav2 for a robot:")
    print(f"  ros2 launch nav2_bringup navigation_launch.py \\")
    print(f"      use_sim_time:=True namespace:=robot_0")
    print()
    print(f"List all topics:")
    print(f"  ros2 topic list | grep robot_")
    print(f"{'='*60}")
    print(f"\nSimulation running... (close window or Ctrl+C to stop)")

    while simulation_app.is_running():
        omni.kit.app.get_app().update()

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
