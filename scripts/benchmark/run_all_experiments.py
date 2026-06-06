#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class NoiseConfig:
    enabled: bool = False
    lidar_noise_std: float = 0.0
    lidar_dropout: float = 0.0
    lidar_range_bias: float = 0.0
    lidar_angular_noise: float = 0.0
    imu_accel_noise: float = 0.0
    imu_gyro_noise: float = 0.0
    imu_accel_bias: float = 0.0
    imu_gyro_bias: float = 0.0
    depth_noise_std: float = 0.0
    depth_invalid_rate: float = 0.0


@dataclass
class RunConfig:
    experiment_id: str
    scenario_name: str
    scene: str
    robot: str
    mode: str
    planner: str
    controller: str
    nav2_params_file: str
    slam_params_file: Optional[str] = None
    map_yaml: Optional[str] = None
    goals_file: Optional[str] = None
    num_obstacles: int = 0
    randomized_obstacles: bool = False
    obstacle_seed: Optional[int] = None
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    monte_carlo_index: int = 0
    global_seed: int = 0
    timeout_s: float = 180.0
    save_slam_snapshots: bool = False
    slam_snapshot_interval_s: float = 10.0
    slam_snapshot_timeout_s: float = 20.0
    notes: str = ""


@dataclass
class RunResult:
    experiment_id: str
    scenario_name: str
    monte_carlo_index: int
    run_dir: str
    success: bool
    return_code: int
    started_at: str
    ended_at: str
    duration_wall_s: float
    summary_csv: Optional[str] = None
    trajectory_csv: Optional[str] = None
    metadata_json: Optional[str] = None
    error: str = ""


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def mkdir_clean(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_csv(path: Path, header: List[str], row: List[Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()


def wait_or_kill(proc: subprocess.Popen[Any], timeout: float) -> int:
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                proc.kill()
            return proc.wait(timeout=5)


def build_noise_args(noise: NoiseConfig, seed: int) -> List[str]:
    if not noise.enabled:
        return []
    return [
        "--seed", str(seed),
        "--lidar-noise-std", str(noise.lidar_noise_std),
        "--lidar-dropout", str(noise.lidar_dropout),
        "--lidar-range-bias", str(noise.lidar_range_bias),
        "--lidar-angular-noise", str(noise.lidar_angular_noise),
        "--imu-accel-noise", str(noise.imu_accel_noise),
        "--imu-gyro-noise", str(noise.imu_gyro_noise),
        "--imu-accel-bias", str(noise.imu_accel_bias),
        "--imu-gyro-bias", str(noise.imu_gyro_bias),
        "--depth-noise-std", str(noise.depth_noise_std),
        "--depth-invalid-rate", str(noise.depth_invalid_rate),
    ]


class CommandBuilder:
    def __init__(self, project_root: Path, ros_setup: str, isaac_python: str):
        self.project_root = project_root
        self.ros_setup = ros_setup
        self.isaac_python = isaac_python

    def nav2_command(self, cfg: RunConfig) -> List[str]:
        if cfg.mode == "localization":
            if not cfg.map_yaml:
                raise ValueError(f"{cfg.experiment_id}: localization mode requires map_yaml")
    
            launch_file = self.project_root / "launch" / "nav2_localization_launch.py"
            map_path = (self.project_root / cfg.map_yaml).resolve()
            params_path = (self.project_root / cfg.nav2_params_file).resolve()
    
            return [
                "bash", "-lc",
                (
                    f"source {self.ros_setup} && ros2 launch {launch_file} "
                    f"map:={map_path} params_file:={params_path}"
                )
            ]
    
        if cfg.mode == "slam":
            launch_file = self.project_root / "launch" / "nav2_slam_launch.py"
            if not cfg.slam_params_file:
                raise ValueError(f"{cfg.experiment_id}: slam mode requires slam_params_file")
    
            params_path = (self.project_root / cfg.nav2_params_file).resolve()
            slam_params_path = (self.project_root / cfg.slam_params_file).resolve()
    
            return [
                "bash", "-lc",
                (
                    f"source {self.ros_setup} && ros2 launch {launch_file} "
                    f"params_file:={params_path} slam_params_file:={slam_params_path}"
                )
            ]
    
        raise ValueError(f"Unsupported mode: {cfg.mode}")

    def noise_command(self, cfg: RunConfig) -> Optional[List[str]]:
        if not cfg.noise.enabled:
            return None
        script = self.project_root / "scripts" / "tools" / "add_sensor_noise.py"
        seed = cfg.global_seed + cfg.monte_carlo_index
        args = build_noise_args(cfg.noise, seed)
        joined = " ".join(map(str, args))
        return [
            "bash", "-lc",
            f"source {self.ros_setup} && python3 {script} {joined}"
        ]

    def snapshot_saver_command(self, cfg: RunConfig, output_dir: Path) -> Optional[List[str]]:
        if not (cfg.mode == "slam" and cfg.save_slam_snapshots):
            return None
        script = self.project_root / "scripts" / "tools" / "slam_snapshot_saver.py"
        maps_dir = output_dir / "maps"
        duration = max(cfg.timeout_s * 1.2, cfg.slam_snapshot_interval_s * 2.0)
        return [
            "bash", "-lc",
            (
                f"source {self.ros_setup} && python3 {script} "
                f"--output-dir {maps_dir} "
                f"--base-name map "
                f"--interval {cfg.slam_snapshot_interval_s} "
                f"--duration {duration} "
                f"--timeout {cfg.slam_snapshot_timeout_s} "
                f"--ros-setup {self.ros_setup}"
            )
        ]

    def benchmark_command(self, cfg: RunConfig, output_dir: Path) -> List[str]:
        script = self.project_root / "scripts" / "benchmark" / "run_benchmark.py"
        args = [
            self.isaac_python,
            str(script),
            "--scene", cfg.scene,
            "--robot", cfg.robot,
            "--timeout", str(cfg.timeout_s),
            "--output-dir", str(output_dir),
            "--no-depth",
        ]
        if cfg.goals_file:
            args += ["--goals-file", cfg.goals_file]
        if cfg.num_obstacles > 0:
            args += ["--obstacles", str(cfg.num_obstacles)]
        return args


class ExperimentFactory:
    def __init__(self, base_seed: int):
        self.base_seed = base_seed

    def build(self) -> Dict[str, List[RunConfig]]:
        return {
            "exp1": self._exp1_baseline(),
            "exp2": self._exp2_sensor_noise_mc(),
            "exp3": self._exp3_obstacles_mc(),
            "exp4": self._exp4_slam_mc(),
        }

    def _exp1_baseline(self) -> List[RunConfig]:
        runs: List[RunConfig] = []
        planner_to_params = {
            "navfn": "config/nav2_navfn.yaml",
            "smac": "config/nav2_smac.yaml",
        }
        for planner, params_file in planner_to_params.items():
            for repeat in range(5):
                runs.append(RunConfig(
                    experiment_id="exp1",
                    scenario_name=f"known_map_{planner}",
                    scene="warehouse",
                    robot="turtlebot",
                    mode="localization",
                    planner=planner,
                    controller="dwb",
                    nav2_params_file=params_file,
                    map_yaml="maps/warehouse.yaml",
                    monte_carlo_index=repeat,
                    global_seed=self.base_seed,
                    notes="Deterministic baseline; repeated for reproducibility check.",
                ))
        return runs

    def _exp2_sensor_noise_mc(self) -> List[RunConfig]:
        runs: List[RunConfig] = []
        noise_levels = {
            "low": NoiseConfig(enabled=True, lidar_noise_std=0.01, lidar_dropout=0.02, lidar_range_bias=0.00),
            "mid": NoiseConfig(enabled=True, lidar_noise_std=0.03, lidar_dropout=0.08, lidar_range_bias=0.02),
            "high": NoiseConfig(enabled=True, lidar_noise_std=0.06, lidar_dropout=0.15, lidar_range_bias=0.05),
        }
        for name, noise in noise_levels.items():
            for mc in range(10):
                runs.append(RunConfig(
                    experiment_id="exp2",
                    scenario_name=f"known_map_noise_{name}",
                    scene="warehouse",
                    robot="turtlebot",
                    mode="localization",
                    planner="navfn",
                    controller="dwb",
                    nav2_params_file="config/nav2_navfn.yaml",
                    map_yaml="maps/warehouse.yaml",
                    noise=noise,
                    monte_carlo_index=mc,
                    global_seed=self.base_seed + 100,
                    notes="Monte Carlo over sensor noise realizations.",
                ))
        return runs

    def _exp3_obstacles_mc(self) -> List[RunConfig]:
        runs: List[RunConfig] = []
        for obstacle_count in (3, 6, 9):
            for mc in range(10):
                runs.append(RunConfig(
                    experiment_id="exp3",
                    scenario_name=f"known_map_obstacles_{obstacle_count}",
                    scene="warehouse",
                    robot="turtlebot",
                    mode="localization",
                    planner="navfn",
                    controller="dwb",
                    nav2_params_file="config/nav2_navfn.yaml",
                    map_yaml="maps/warehouse.yaml",
                    num_obstacles=obstacle_count,
                    randomized_obstacles=True,
                    obstacle_seed=self.base_seed + 1000 + mc,
                    monte_carlo_index=mc,
                    global_seed=self.base_seed + 200,
                    notes="Monte Carlo over randomized extra obstacle placement.",
                ))
        return runs

    def _exp4_slam_mc(self) -> List[RunConfig]:
        runs: List[RunConfig] = []
        noise = NoiseConfig(enabled=True, lidar_noise_std=0.02, lidar_dropout=0.05, lidar_range_bias=0.01)
        for mc in range(10):
            runs.append(RunConfig(
                experiment_id="exp4",
                scenario_name="unknown_map_slam_noise",
                scene="warehouse",
                robot="turtlebot",
                mode="slam",
                planner="navfn",
                controller="dwb",
                nav2_params_file="config/nav2_navfn.yaml",
                slam_params_file="config/slam.yaml",
                noise=noise,
                monte_carlo_index=mc,
                global_seed=self.base_seed + 300,
                save_slam_snapshots=True,
                slam_snapshot_interval_s=10.0,
                slam_snapshot_timeout_s=20.0,
                notes="SLAM scenario with Monte Carlo noise realizations and periodic map snapshots.",
            ))
        return runs


class ExperimentRunner:
    def __init__(self, builder: CommandBuilder, output_root: Path, dry_run: bool = False):
        self.builder = builder
        self.output_root = output_root
        self.dry_run = dry_run
        mkdir_clean(output_root)
        self.master_csv = output_root / "all_runs.csv"

    def run_one(self, cfg: RunConfig) -> RunResult:
        run_name = f"{cfg.experiment_id}__{cfg.scenario_name}__mc{cfg.monte_carlo_index:02d}"
        run_dir = self.output_root / run_name
        mkdir_clean(run_dir)

        write_json(run_dir / "run_config.json", asdict(cfg))
        started_at = now_iso()
        wall_start = time.time()

        env = dict(os.environ)
        env["ETF_EXPERIMENT_SEED"] = str(cfg.global_seed + cfg.monte_carlo_index)
        if cfg.obstacle_seed is not None:
            env["ETF_OBSTACLE_SEED"] = str(cfg.obstacle_seed)

        nav2_cmd = self.builder.nav2_command(cfg)
        noise_cmd = self.builder.noise_command(cfg)
        snapshot_cmd = self.builder.snapshot_saver_command(cfg, run_dir)
        bench_cmd = self.builder.benchmark_command(cfg, run_dir)

        write_json(run_dir / "commands.json", {
            "nav2_cmd": nav2_cmd,
            "noise_cmd": noise_cmd,
            "snapshot_cmd": snapshot_cmd,
            "benchmark_cmd": bench_cmd,
        })

        if self.dry_run:
            ended_at = now_iso()
            result = RunResult(
                experiment_id=cfg.experiment_id,
                scenario_name=cfg.scenario_name,
                monte_carlo_index=cfg.monte_carlo_index,
                run_dir=str(run_dir),
                success=True,
                return_code=0,
                started_at=started_at,
                ended_at=ended_at,
                duration_wall_s=0.0,
                metadata_json=str(run_dir / "run_config.json"),
            )
            write_json(run_dir / "run_result.json", asdict(result))
            return result

        nav2_log = (run_dir / "nav2.log").open("w", encoding="utf-8")
        noise_log = (run_dir / "noise.log").open("w", encoding="utf-8")
        snapshot_log = (run_dir / "snapshots.log").open("w", encoding="utf-8")
        bench_log = (run_dir / "benchmark.log").open("w", encoding="utf-8")

        nav2_proc = None
        noise_proc = None
        snapshot_proc = None
        bench_proc = None
        error_msg = ""
        return_code = -1

        try:
            nav2_proc = subprocess.Popen(
                nav2_cmd,
                stdout=nav2_log,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if os.name == "posix" else None,
            )
            time.sleep(8.0)

            if noise_cmd is not None:
                noise_proc = subprocess.Popen(
                    noise_cmd,
                    stdout=noise_log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=os.setsid if os.name == "posix" else None,
                )
                time.sleep(2.0)

            if snapshot_cmd is not None:
                snapshot_proc = subprocess.Popen(
                    snapshot_cmd,
                    stdout=snapshot_log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=os.setsid if os.name == "posix" else None,
                )
                time.sleep(1.0)

            bench_proc = subprocess.Popen(
                bench_cmd,
                stdout=bench_log,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if os.name == "posix" else None,
            )
            return_code = wait_or_kill(bench_proc, timeout=cfg.timeout_s * 10 + 180)

        except Exception as exc:
            error_msg = str(exc)

        finally:
            if bench_proc is not None:
                kill_process_tree(bench_proc)
            if snapshot_proc is not None:
                kill_process_tree(snapshot_proc)
            if noise_proc is not None:
                kill_process_tree(noise_proc)
            if nav2_proc is not None:
                kill_process_tree(nav2_proc)

            nav2_log.close()
            noise_log.close()
            snapshot_log.close()
            bench_log.close()

        duration = time.time() - wall_start
        ended_at = now_iso()

        summary_csv = run_dir / "summary.csv"
        trajectory_csv = run_dir / "trajectory.csv"
        success = (return_code == 0) and summary_csv.exists()

        result = RunResult(
            experiment_id=cfg.experiment_id,
            scenario_name=cfg.scenario_name,
            monte_carlo_index=cfg.monte_carlo_index,
            run_dir=str(run_dir),
            success=success,
            return_code=return_code,
            started_at=started_at,
            ended_at=ended_at,
            duration_wall_s=duration,
            summary_csv=str(summary_csv) if summary_csv.exists() else None,
            trajectory_csv=str(trajectory_csv) if trajectory_csv.exists() else None,
            metadata_json=str(run_dir / "run_config.json"),
            error=error_msg,
        )

        write_json(run_dir / "run_result.json", asdict(result))
        self._append_master(result)
        return result

    def _append_master(self, result: RunResult) -> None:
        append_csv(
            self.master_csv,
            [
                "experiment_id", "scenario_name", "mc_index", "run_dir",
                "success", "return_code", "started_at", "ended_at",
                "duration_wall_s", "summary_csv", "trajectory_csv", "error",
            ],
            [
                result.experiment_id,
                result.scenario_name,
                result.monte_carlo_index,
                result.run_dir,
                result.success,
                result.return_code,
                result.started_at,
                result.ended_at,
                f"{result.duration_wall_s:.2f}",
                result.summary_csv or "",
                result.trajectory_csv or "",
                result.error,
            ],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diploma experiment batches")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--ros-setup", type=str, default="/opt/ros/jazzy/setup.bash")
    parser.add_argument("--isaac-python", type=str, default=str(Path.home() / "isaacsim" / "python.sh"))
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", choices=["exp1", "exp2", "exp3", "exp4"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()

    factory = ExperimentFactory(base_seed=args.base_seed)
    all_runs = factory.build()
    selected = args.only if args.only else ["exp1", "exp2", "exp3", "exp4"]

    builder = CommandBuilder(project_root, args.ros_setup, args.isaac_python)
    runner = ExperimentRunner(builder, output_root, args.dry_run)

    total = 0
    ok = 0
    failed = 0

    for exp_id in selected:
        runs = all_runs[exp_id]
        print(f"\n=== {exp_id}: {len(runs)} run(s) ===")
        for cfg in runs:
            total += 1
            print(f"[{total}] {cfg.scenario_name} | mc={cfg.monte_carlo_index}")
            result = runner.run_one(cfg)
            if result.success:
                ok += 1
                print(f"    OK   -> {result.run_dir}")
            else:
                failed += 1
                print(f"    FAIL -> {result.run_dir} (rc={result.return_code})")
                if result.error:
                    print(f"           {result.error}")

    print("\n" + "=" * 72)
    print("Experiment batch complete")
    print(f"  Total runs : {total}")
    print(f"  Successful : {ok}")
    print(f"  Failed     : {failed}")
    print(f"  Output dir : {output_root}")
    print(f"  Master CSV : {output_root / 'all_runs.csv'}")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
