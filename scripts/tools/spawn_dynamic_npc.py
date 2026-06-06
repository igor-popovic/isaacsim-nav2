"""
Spawn dynamic NPC obstacles with movement patterns in a scene.

Uses character USD models from characters/ directory (police, medical,
construction, business). Falls back to capsule shapes if --no-characters
is passed. NPCs have invisible collision capsules for LiDAR + physics.

Movement modes: random_walk, patrol, circular.

Usage:
    ~/isaacsim/python.sh ~/etf_isaac_nav_testbed/scripts/tools/spawn_dynamic_npc.py \
        --scene warehouse --num-npcs 5 --pattern random_walk
    ~/isaacsim/python.sh ~/etf_isaac_nav_testbed/scripts/tools/spawn_dynamic_npc.py \
        --scene simple_room --num-npcs 3 --pattern patrol --speed 0.3
    ~/isaacsim/python.sh ~/etf_isaac_nav_testbed/scripts/tools/spawn_dynamic_npc.py \
        --scene warehouse --num-npcs 4 --no-characters
"""

import os
import argparse
import math

parser = argparse.ArgumentParser(description="Dynamic NPC Spawner")
parser.add_argument(
    "--scene", type=str, default="warehouse",
    choices=["warehouse", "warehouse_digital_twin", "warehouse_forklifts",
             "simple_room", "hospital", "office"],
    help="Scene to load",
)
parser.add_argument("--num-npcs", type=int, default=3, help="Number of NPCs to spawn")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument(
    "--pattern", type=str, default="random_walk",
    choices=["random_walk", "patrol", "circular"],
    help="NPC movement pattern",
)
parser.add_argument("--speed", type=float, default=0.5, help="NPC movement speed (m/s)")
parser.add_argument("--area", type=float, default=5.0,
                    help="Half-size of spawn area (NPCs spawn in [-area, area])")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--characters", action="store_true",
                    help="Use character USD models instead of capsule shapes (requires more VRAM)")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUCLEUS_SCENES = {
    "warehouse":              "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_digital_twin": "/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
    "warehouse_forklifts":    "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
    "simple_room":            "/Isaac/Environments/Simple_Room/simple_room.usd",
    "hospital":               "/Isaac/Environments/Hospital/hospital.usd",
    "office":                 "/Isaac/Environments/Office/office.usd",
}

# Character models — use the "new" variants where available (better quality),
# skip the biped_demo and originals.
CHARACTER_MODELS = [
    ("female_adult_police_01_new",      "female_adult_police_01_new.usd"),
    ("female_adult_police_02",          "female_adult_police_02.usd"),
    ("female_adult_police_03_new",      "female_adult_police_03_new.usd"),
    ("male_adult_police_04",            "male_adult_police_04.usd"),
    ("male_adult_construction_01_new",  "male_adult_construction_01_new.usd"),
    ("male_adult_construction_03",      "male_adult_construction_03.usd"),
    ("male_adult_construction_05_new",  "male_adult_construction_05_new.usd"),
    ("F_Business_02",                   "F_Business_02.usd"),
    ("F_Medical_01",                    "F_Medical_01.usd"),
    ("M_Medical_01",                    "M_Medical_01.usd"),
]

# Nucleus CDN path for character models
NUCLEUS_CHARACTERS_ROOT = "/Isaac/People/Characters"


def _discover_characters():
    """Return list of (label, usd_path) for available character models from Nucleus CDN."""
    try:
        from isaacsim.storage.native import get_assets_root_path
        assets_root = get_assets_root_path()
    except RuntimeError:
        assets_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    available = []
    for folder_name, usd_name in CHARACTER_MODELS:
        usd_path = f"{assets_root}{NUCLEUS_CHARACTERS_ROOT}/{folder_name}/{usd_name}"
        available.append((folder_name, usd_path))
    return available


# ---------------------------------------------------------------------------
# NPC body creation
# ---------------------------------------------------------------------------
def spawn_npc_character(stage, name, position, character_usd, yaw=0.0):
    """Spawn a character USD model with an invisible collision capsule.

    The character USD is referenced directly as the NPC prim (not nested)
    so that the animation graph is at the expected level for ag.get_character().
    A collision capsule is added as a child for LiDAR/physics detection.
    """
    prim_path = f"/World/NPCs/{name}"

    # Reference the character USD directly as the NPC prim
    char_prim = stage.DefinePrim(prim_path)
    char_prim.GetReferences().AddReference(character_usd)

    # Set initial position and rotation (character USDs already have xform ops)
    translate_attr = char_prim.GetAttribute("xformOp:translate")
    if translate_attr:
        translate_attr.Set(Gf.Vec3d(position[0], position[1], position[2]))
    else:
        UsdGeom.Xformable(char_prim).AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))

    rotate_attr = char_prim.GetAttribute("xformOp:rotateXYZ")
    if rotate_attr:
        rotate_attr.Set(Gf.Vec3d(0, 0, yaw))
    else:
        UsdGeom.Xformable(char_prim).AddRotateZOp().Set(yaw)

    # Invisible collision capsule (for LiDAR + robot collision)
    collider_path = f"{prim_path}/collider"
    capsule = UsdGeom.Capsule.Define(stage, collider_path)
    capsule.GetHeightAttr().Set(1.0)
    capsule.GetRadiusAttr().Set(0.3)
    capsule.GetAxisAttr().Set("Z")
    capsule.GetVisibilityAttr().Set("invisible")

    # Position collider at character center of mass
    capsule_xform = UsdGeom.Xformable(capsule.GetPrim())
    capsule_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.85))

    # Kinematic rigid body on collider
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
    rigid_body_api.GetKinematicEnabledAttr().Set(True)
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())

    return prim_path


def spawn_npc_capsule(stage, name, position, color_idx=0):
    """Spawn a colored capsule NPC (fallback when characters are unavailable)."""
    colors = [
        (0.9, 0.2, 0.2), (0.2, 0.6, 0.9), (0.2, 0.8, 0.3), (0.9, 0.7, 0.1),
        (0.7, 0.3, 0.8), (0.9, 0.5, 0.2), (0.3, 0.8, 0.8), (0.8, 0.4, 0.5),
    ]
    prim_path = f"/World/NPCs/{name}"

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))

    capsule_path = f"{prim_path}/body"
    capsule = UsdGeom.Capsule.Define(stage, capsule_path)
    capsule.GetHeightAttr().Set(1.0)
    capsule.GetRadiusAttr().Set(0.3)
    capsule.GetAxisAttr().Set("Z")

    capsule_xform = UsdGeom.Xformable(capsule.GetPrim())
    capsule_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.85))

    capsule.GetDisplayColorAttr().Set([Gf.Vec3f(*colors[color_idx % len(colors)])])

    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
    rigid_body_api.GetKinematicEnabledAttr().Set(True)
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())

    return prim_path


# ---------------------------------------------------------------------------
# Movement controllers
# ---------------------------------------------------------------------------
class RandomWalkController:
    """NPC walks in a random direction, turns at random intervals."""

    def __init__(self, start_pos, speed, area, rng):
        self.pos = np.array(start_pos, dtype=float)
        self.speed = speed
        self.area = area
        self.rng = rng
        self._pick_new_heading()

    def _pick_new_heading(self):
        angle = self.rng.uniform(0, 2 * math.pi)
        self.direction = np.array([math.cos(angle), math.sin(angle)])
        self.walk_time = self.rng.uniform(2.0, 6.0)
        self.elapsed = 0.0

    def step(self, dt):
        self.elapsed += dt
        if self.elapsed >= self.walk_time:
            self._pick_new_heading()

        new_pos = self.pos[:2] + self.direction * self.speed * dt

        # Bounce off area boundaries
        for axis in range(2):
            if abs(new_pos[axis]) > self.area:
                new_pos[axis] = np.clip(new_pos[axis], -self.area, self.area)
                self.direction[axis] *= -1

        self.pos[:2] = new_pos
        return self.pos.copy()


class PatrolController:
    """NPC walks back and forth between two waypoints."""

    def __init__(self, start_pos, speed, area, rng):
        self.speed = speed
        # Generate two patrol endpoints
        offset = rng.uniform(1.5, area)
        angle = rng.uniform(0, 2 * math.pi)
        dx, dy = offset * math.cos(angle), offset * math.sin(angle)
        self.waypoints = [
            np.array([start_pos[0] - dx, start_pos[1] - dy, start_pos[2]]),
            np.array([start_pos[0] + dx, start_pos[1] + dy, start_pos[2]]),
        ]
        self.current_wp = 0
        self.pos = np.array(start_pos, dtype=float)

    def step(self, dt):
        target = self.waypoints[self.current_wp]
        to_target = target[:2] - self.pos[:2]
        dist = np.linalg.norm(to_target)

        if dist < 0.1:
            # Reached waypoint, switch direction
            self.current_wp = 1 - self.current_wp
        else:
            direction = to_target / dist
            step_size = min(self.speed * dt, dist)
            self.pos[:2] += direction * step_size

        return self.pos.copy()


class CircularController:
    """NPC walks in a circle around its spawn point."""

    def __init__(self, start_pos, speed, area, rng):
        self.center = np.array(start_pos[:2], dtype=float)
        self.radius = rng.uniform(1.0, min(3.0, area * 0.6))
        self.z = start_pos[2]
        self.angle = rng.uniform(0, 2 * math.pi)
        # Angular speed = linear speed / radius
        self.angular_speed = speed / self.radius
        # Random direction (CW or CCW)
        if rng.random() < 0.5:
            self.angular_speed *= -1

    def step(self, dt):
        self.angle += self.angular_speed * dt
        x = self.center[0] + self.radius * math.cos(self.angle)
        y = self.center[1] + self.radius * math.sin(self.angle)
        return np.array([x, y, self.z])


CONTROLLERS = {
    "random_walk": RandomWalkController,
    "patrol":      PatrolController,
    "circular":    CircularController,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(args.seed)

    # Resolve scene path from Nucleus/CDN
    try:
        from isaacsim.storage.native import get_assets_root_path
        assets_root = get_assets_root_path()
    except RuntimeError:
        assets_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    scene_path = assets_root + NUCLEUS_SCENES[args.scene]

    print(f"Scene:   {args.scene} -> {scene_path}")
    print(f"NPCs:    {args.num_npcs}")
    print(f"Pattern: {args.pattern} at {args.speed} m/s")

    # Load scene
    omni.usd.get_context().open_stage(scene_path)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    simulation_app.update()

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    stage = omni.usd.get_context().get_stage()

    # Discover available character models
    characters = _discover_characters() if args.characters else []
    use_characters = len(characters) > 0
    if use_characters:
        print(f"  Found {len(characters)} character models")
    else:
        reason = "not enabled (use --characters)" if not args.characters else "not found"
        print(f"  Characters {reason}, using capsule shapes")

    # Spawn NPCs with controllers
    npcs = []
    controller_cls = CONTROLLERS[args.pattern]

    for i in range(args.num_npcs):
        pos = [
            float(rng.uniform(-args.area, args.area)),
            float(rng.uniform(-args.area, args.area)),
            0.0,
        ]
        yaw = float(rng.uniform(0, 360))

        if use_characters:
            char_label, char_usd = characters[i % len(characters)]
            prim_path = spawn_npc_character(stage, f"npc_{i}", pos, char_usd, yaw=yaw)
            label = char_label
        else:
            prim_path = spawn_npc_capsule(stage, f"npc_{i}", pos, color_idx=i)
            label = "capsule"

        controller = controller_cls(pos, args.speed, args.area, rng)
        npcs.append({"prim_path": prim_path, "controller": controller})
        print(f"  Spawned {prim_path} ({label}) at ({pos[0]:.1f}, {pos[1]:.1f})")

    world.reset()

    # Let the animation graph initialize for a few frames
    for _ in range(30):
        world.step(render=True)

    # Simulation loop
    dt = 1.0 / 60.0
    step_count = 0

    print(f"\nSimulation running... (close window or Ctrl+C to stop)")

    while simulation_app.is_running():
        world.step(render=True)
        step_count += 1

        if not world.is_playing():
            continue

        # Update NPC positions
        for npc in npcs:
            new_pos = npc["controller"].step(dt)
            prim = stage.GetPrimAtPath(npc["prim_path"])
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                ops = xformable.GetOrderedXformOps()
                if ops:
                    ops[0].Set(Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))

    simulation_app.close()


if __name__ == "__main__":
    main()
