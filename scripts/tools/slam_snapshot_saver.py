#!/usr/bin/env python3
"""
Post-process experiment results produced by run_all_experiments.py and compute
Monte Carlo aggregate statistics.

What it does:
  1) Scans all run folders under an output root.
  2) Reads run_result.json, run_config.json, summary.csv, trajectory.csv.
  3) Builds a flat per-run table.
  4) Computes Monte Carlo aggregates (mean/std/min/max/success-rate) grouped by
     experiment/scenario and key config fields.
  5) Saves CSV outputs.
  6) Generates plots for travel time, path length, and success rate.
  7) For SLAM runs, generates map visualizations when map files are available.

Important note for SLAM map visualization:
  - If a run folder contains only a final map (.pgm + .yaml), this script will
    create a final map plot and optionally overlay the robot trajectory.
  - If a run folder contains multiple map snapshots (for example map_0001.pgm,
    map_0002.pgm, ... with matching .yaml files), this script will create a
    snapshot gallery and an animation-ready ordered frame set.
  - If you want maps \"during search\" and your current pipeline only saves the
    final map, then you will need to add periodic map saving during the run.
    This script already supports such snapshots if they exist.

Example usage:
  python3 postprocess_results.py --results-root ./results
  python3 postprocess_results.py --results-root ./results --only exp2 exp4
  python3 postprocess_results.py --results-root ./results --slam-maps-root ./maps
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary_csv(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def read_trajectory_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_map_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key == "origin":
            value = value.strip("[]")
            parts = [p.strip() for p in value.split(",")]
            data[key] = [float(p) for p in parts]
        elif key in {"resolution", "occupied_thresh", "free_thresh"}:
            data[key] = float(value)
        elif key in {"negate"}:
            data[key] = int(value)
        else:
            data[key] = value.strip('"').strip("'")
    return data


def load_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic not in {b"P2", b"P5"}:
            raise ValueError(f"Unsupported PGM format in {path}: {magic!r}")

        def next_token() -> bytes:
            while True:
                line = f.readline()
                if not line:
                    raise EOFError("Unexpected EOF while reading PGM")
                line = line.strip()
                if not line or line.startswith(b"#"):
                    continue
                return line

        dims = next_token().split()
        while len(dims) < 2:
            dims += next_token().split()
        width, height = map(int, dims[:2])
        maxval = int(next_token())
        if maxval > 255:
            raise ValueError("Only 8-bit PGM files are supported")

        if magic == b"P5":
            data = np.frombuffer(f.read(width * height), dtype=np.uint8)
        else:
            tokens: List[int] = []
            while len(tokens) < width * height:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line or line.startswith(b"#"):
                    continue
                tokens.extend(int(x) for x in line.split())
            data = np.array(tokens[: width * height], dtype=np.uint8)

        return data.reshape((height, width))


def world_to_map_pixel(
    x: np.ndarray,
    y: np.ndarray,
    resolution: float,
    origin: List[float],
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    px = (x - origin[0]) / resolution
    py = height - (y - origin[1]) / resolution
    return px, py


def find_map_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for yaml_path in sorted(folder.glob("*.yaml")):
        meta = parse_map_yaml(yaml_path)
        image_name = meta.get("image")
        if not image_name:
            continue
        pgm_path = (yaml_path.parent / image_name).resolve()
        if pgm_path.exists():
            pairs.append((yaml_path, pgm_path))
    return pairs


def choose_map_pairs_for_run(run_dir: Path, slam_maps_root: Optional[Path]) -> List[Tuple[Path, Path]]:
    pairs = find_map_pairs(run_dir)
    if pairs:
        return pairs

    maps_dir = run_dir / "maps"
    if maps_dir.exists():
        pairs = find_map_pairs(maps_dir)
        if pairs:
            return pairs

    if slam_maps_root is None:
        return []

    candidate_dir = slam_maps_root / run_dir.name
    if candidate_dir.exists():
        pairs = find_map_pairs(candidate_dir)
        if pairs:
            return pairs

    return []


def collect_runs(results_root: Path, only: Optional[List[str]] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for run_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        run_result_path = run_dir / "run_result.json"
        run_config_path = run_dir / "run_config.json"

        if not run_result_path.exists() or not run_config_path.exists():
            continue

        run_result = read_json(run_result_path)
        run_config = read_json(run_config_path)

        if only and run_result.get("experiment_id") not in only:
            continue

        row: Dict[str, Any] = {}
        row.update({f"cfg_{k}": v for k, v in run_config.items()})
        row.update({f"res_{k}": v for k, v in run_result.items()})
        row["run_dir"] = str(run_dir)

        summary_path = run_dir / "summary.csv"
        trajectory_path = run_dir / "trajectory.csv"

        row["summary_exists"] = summary_path.exists()
        row["trajectory_exists"] = trajectory_path.exists()

        if summary_path.exists():
            summary = read_summary_csv(summary_path)
            for k, v in summary.items():
                row[f"summary_{k}"] = v
        else:
            row["summary_goal_reached"] = np.nan
            row["summary_travel_time_s"] = np.nan
            row["summary_path_length_m"] = np.nan
            row["summary_collisions"] = np.nan
            row["summary_min_obstacle_dist_m"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in [
        "summary_travel_time_s",
        "summary_path_length_m",
        "summary_collisions",
        "summary_min_obstacle_dist_m",
        "res_duration_wall_s",
        "cfg_monte_carlo_index",
        "cfg_num_obstacles",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "summary_goal_reached" in df.columns:
        df["summary_goal_reached"] = df["summary_goal_reached"].astype(str).str.lower().map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })

    return df


def aggregate_monte_carlo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    group_cols = [
        "cfg_experiment_id",
        "cfg_scenario_name",
        "cfg_mode",
        "cfg_scene",
        "cfg_robot",
        "cfg_planner",
        "cfg_controller",
        "cfg_num_obstacles",
    ]

    work = df.copy()
    work["success_numeric"] = work["summary_goal_reached"].astype(float)

    grouped = work.groupby(group_cols, dropna=False)
    agg = grouped.agg(
        runs=("run_dir", "count"),
        success_rate=("success_numeric", "mean"),
        travel_time_mean=("summary_travel_time_s", "mean"),
        travel_time_std=("summary_travel_time_s", "std"),
        travel_time_min=("summary_travel_time_s", "min"),
        travel_time_max=("summary_travel_time_s", "max"),
        path_length_mean=("summary_path_length_m", "mean"),
        path_length_std=("summary_path_length_m", "std"),
        path_length_min=("summary_path_length_m", "min"),
        path_length_max=("summary_path_length_m", "max"),
        collisions_mean=("summary_collisions", "mean"),
        collisions_std=("summary_collisions", "std"),
        min_obstacle_distance_mean=("summary_min_obstacle_dist_m", "mean"),
        min_obstacle_distance_std=("summary_min_obstacle_dist_m", "std"),
        wall_time_mean=("res_duration_wall_s", "mean"),
        wall_time_std=("res_duration_wall_s", "std"),
    ).reset_index()

    agg["success_rate"] = 100.0 * agg["success_rate"]
    return agg


def plot_metric_bars(
    agg_df: pd.DataFrame,
    value_col: str,
    error_col: Optional[str],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    if agg_df.empty or value_col not in agg_df.columns:
        return

    labels = [
        f"{row['cfg_experiment_id']}\n{row['cfg_scenario_name']}\n{row['cfg_planner']}"
        for _, row in agg_df.iterrows()
    ]
    values = agg_df[value_col].to_numpy(dtype=float)
    errors = agg_df[error_col].fillna(0.0).to_numpy(dtype=float) if error_col and error_col in agg_df.columns else None

    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    plt.bar(x, values, yerr=errors, capsize=4)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def overlay_trajectory_on_map(ax: Any, map_meta: Dict[str, Any], image: np.ndarray, trajectory_df: Optional[pd.DataFrame]) -> None:
    if trajectory_df is None or trajectory_df.empty:
        return

    resolution = float(map_meta["resolution"])
    origin = map_meta["origin"]
    height = image.shape[0]

    px, py = world_to_map_pixel(
        trajectory_df["x"].to_numpy(dtype=float),
        trajectory_df["y"].to_numpy(dtype=float),
        resolution,
        origin,
        height,
    )
    ax.plot(px, py, linewidth=1.5)
    ax.scatter(px[:1], py[:1], s=20, marker="o")
    ax.scatter(px[-1:], py[-1:], s=20, marker="x")


def make_final_map_plot(run_dir: Path, yaml_path: Path, pgm_path: Path, out_path: Path) -> None:
    meta = parse_map_yaml(yaml_path)
    image = load_pgm(pgm_path)
    traj_path = run_dir / "trajectory.csv"
    traj_df = read_trajectory_csv(traj_path) if traj_path.exists() else None

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray", origin="upper")
    ax = plt.gca()
    overlay_trajectory_on_map(ax, meta, image, traj_df)
    plt.title(f"SLAM Final Map: {run_dir.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_snapshot_gallery(run_dir: Path, map_pairs: List[Tuple[Path, Path]], out_dir: Path) -> None:
    if not map_pairs:
        return

    n = len(map_pairs)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(4 * cols, 4 * rows))
    traj_path = run_dir / "trajectory.csv"
    traj_df = read_trajectory_csv(traj_path) if traj_path.exists() else None

    for idx, (yaml_path, pgm_path) in enumerate(map_pairs, start=1):
        meta = parse_map_yaml(yaml_path)
        image = load_pgm(pgm_path)
        ax = plt.subplot(rows, cols, idx)
        ax.imshow(image, cmap="gray", origin="upper")
        overlay_trajectory_on_map(ax, meta, image, traj_df)
        ax.set_title(yaml_path.stem)
        ax.axis("off")

    plt.suptitle(f"SLAM map snapshots: {run_dir.name}")
    plt.tight_layout()
    gallery_path = out_dir / f"{run_dir.name}__slam_snapshot_gallery.png"
    plt.savefig(gallery_path, dpi=180)
    plt.close()

    frames_dir = out_dir / f"{run_dir.name}__slam_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, (yaml_path, pgm_path) in enumerate(map_pairs):
        meta = parse_map_yaml(yaml_path)
        image = load_pgm(pgm_path)
        traj_df = read_trajectory_csv(traj_path) if traj_path.exists() else None
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"

        plt.figure(figsize=(7, 7))
        plt.imshow(image, cmap="gray", origin="upper")
        ax = plt.gca()
        overlay_trajectory_on_map(ax, meta, image, traj_df)
        plt.title(yaml_path.stem)
        plt.tight_layout()
        plt.savefig(frame_path, dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process diploma experiment results")
    parser.add_argument("--results-root", type=Path, required=True, help="Root folder created by run_all_experiments.py")
    parser.add_argument("--only", nargs="*", choices=["exp1", "exp2", "exp3", "exp4"], help="Process only selected experiment groups")
    parser.add_argument("--slam-maps-root", type=Path, default=None, help="Optional extra folder containing SLAM map outputs or snapshots")
    parser.add_argument("--plots-dirname", type=str, default="plots", help="Subfolder name for generated plots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    slam_maps_root = args.slam_maps_root.resolve() if args.slam_maps_root else None
    plots_dir = results_root / args.plots_dirname
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = collect_runs(results_root, only=args.only)
    if df.empty:
        print(f"No run folders found under {results_root}")
        return 1

    per_run_csv = results_root / "per_run_results.csv"
    df.to_csv(per_run_csv, index=False)

    agg_df = aggregate_monte_carlo(df)
    agg_csv = results_root / "aggregated_results.csv"
    agg_df.to_csv(agg_csv, index=False)

    plot_metric_bars(
        agg_df,
        value_col="success_rate",
        error_col=None,
        out_path=plots_dir / "success_rate.png",
        title="Monte Carlo Success Rate by Scenario",
        ylabel="Success rate [%]",
    )

    plot_metric_bars(
        agg_df,
        value_col="travel_time_mean",
        error_col="travel_time_std",
        out_path=plots_dir / "travel_time.png",
        title="Monte Carlo Mean Travel Time",
        ylabel="Travel time [s]",
    )

    plot_metric_bars(
        agg_df,
        value_col="path_length_mean",
        error_col="path_length_std",
        out_path=plots_dir / "path_length.png",
        title="Monte Carlo Mean Path Length",
        ylabel="Path length [m]",
    )

    slam_df = df[df.get("cfg_mode", pd.Series(dtype=str)) == "slam"].copy()
    slam_index_rows: List[Dict[str, Any]] = []

    for _, row in slam_df.iterrows():
        run_dir = Path(row["run_dir"])
        map_pairs = choose_map_pairs_for_run(run_dir, slam_maps_root)

        if not map_pairs:
            slam_index_rows.append({
                "run_dir": str(run_dir),
                "status": "no_maps_found",
                "maps_detected": 0,
            })
            continue

        final_yaml, final_pgm = map_pairs[-1]
        final_map_plot = plots_dir / f"{run_dir.name}__slam_final_map.png"
        make_final_map_plot(run_dir, final_yaml, final_pgm, final_map_plot)

        if len(map_pairs) > 1:
            make_snapshot_gallery(run_dir, map_pairs, plots_dir)
            status = "snapshots_and_final"
        else:
            status = "final_only"

        slam_index_rows.append({
            "run_dir": str(run_dir),
            "status": status,
            "maps_detected": len(map_pairs),
            "final_yaml": str(final_yaml),
            "final_pgm": str(final_pgm),
            "final_map_plot": str(final_map_plot),
        })

    if slam_index_rows:
        pd.DataFrame(slam_index_rows).to_csv(results_root / "slam_map_index.csv", index=False)

    print("=" * 72)
    print("Post-processing complete")
    print(f"  Per-run CSV      : {per_run_csv}")
    print(f"  Aggregated CSV   : {agg_csv}")
    print(f"  Plots directory  : {plots_dir}")
    if slam_index_rows:
        print(f"  SLAM map index   : {results_root / 'slam_map_index.csv'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
