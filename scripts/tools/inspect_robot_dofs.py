"""Inspect DOF names for a robot USD. Run with:
   ~/isaacsim/python.sh ~/etf_isaac_nav_testbed/scripts/tools/inspect_robot_dofs.py --robot carter
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="carter", choices=["turtlebot", "carter", "carter_lidar"])
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage

NUCLEUS_ROBOTS = {
    "carter":       "/Isaac/Robots/NVIDIA/Carter/carter_v1.usd",
    "carter_lidar": "/Isaac/Robots/NVIDIA/Carter/carter_v1_physx_lidar.usd",
    "turtlebot":    "/Isaac/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd",
}

try:
    from isaacsim.storage.native import get_assets_root_path
    assets_root = get_assets_root_path()
except RuntimeError:
    assets_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
usd_path = assets_root + NUCLEUS_ROBOTS[args.robot]

print(f"Robot: {args.robot}")
print(f"USD:   {usd_path}")

add_reference_to_stage(usd_path=usd_path, prim_path="/World/Robot")
simulation_app.update()

world = World(stage_units_in_meters=1.0)
robot = world.scene.add(Robot(prim_path="/World/Robot", name=args.robot))
world.reset()

print(f"\nDOF names: {robot.dof_names}")
print(f"Num DOFs:  {robot.num_dof}")

# Also print all joint paths from the stage for reference
stage = omni.usd.get_context().get_stage()
from pxr import UsdPhysics
print(f"\nAll joint prims:")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint) or "Joint" in prim.GetTypeName():
        print(f"  {prim.GetPath()} ({prim.GetTypeName()})")

simulation_app.close()
