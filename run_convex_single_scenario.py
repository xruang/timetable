"""Run the CO_V2-based model optimizer for one real station section."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from timetable.bidirectional_model import build_bidirectional_model
from timetable.convex_lower import ConvexLowerOptions, ConvexLowerSolver
from timetable.cvxpy_lower import CvxpyLowerOptions, CvxpyLowerSolver
from timetable.train_model import create_train_config_from_mass
from timetable.timetable_0601_bidirectional_online import build_parser as build_model_parser
from timetable.visualization import write_csv


PROJECT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CO_V2凸优化单站间下层驾驶策略")
    parser.add_argument("--direction", choices=("up", "down"), default="up")
    parser.add_argument("--section", default="HC-DGNL")
    parser.add_argument("--mass-t", type=float, default=255.0)
    parser.add_argument("--target-time-s", type=float)
    parser.add_argument("--segments", type=int, default=600)
    parser.add_argument("--speed-limit-ratio", type=float, default=0.8)
    parser.add_argument("--solver-log", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("gurobi", "cvxpy"),
        default="gurobi",
        help="cvxpy uses the licence-free CLARABEL solver",
    )
    parser.add_argument("--list-sections", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "lower_convex_single_scenario",
    )
    return parser


def _summary(result) -> dict:
    return {
        "status": result.status,
        "section_name": result.section_name,
        "target_time_s": result.target_time_s,
        "solver_status_code": result.solver_status_code,
        "solver_runtime_s": result.solver_runtime_s,
        "optimized_objective_kwh": result.optimized_objective_kwh,
        "actual_time_s": result.actual_time_s,
        "timetable_error_s": result.timetable_error_s,
        "actual_traction_kwh": result.actual_traction_kwh,
        "actual_regenerative_kwh": result.actual_regenerative_kwh,
        "actual_net_energy_kwh": result.actual_net_energy_kwh,
        "max_alpha_relative_error": result.max_alpha_relative_error,
        "max_beta_normalized_error": result.max_beta_normalized_error,
        "max_force_gap_kn": result.max_force_gap_kn,
        "feasible": result.feasible,
        "validation_messages": result.validation_messages,
        "numerical_terminal_speed_mps": result.numerical_terminal_speed_mps,
    }


def _trajectory_rows(result) -> list[dict]:
    rows = []
    for node, position in enumerate(result.position_m):
        row = {
            "node": node,
            "position_m": float(position),
            "speed_kmh": float(result.speed_kmh[node]),
            "beta_absolute_error_m2ps2": float(result.beta_absolute_error_m2ps2[node]),
        }
        if node > 0:
            cell = node - 1
            row.update(
                segment_time_s=float(result.segment_time_s[cell]),
                optimized_force_kn=float(result.optimized_force_kn[cell]),
                recomputed_force_kn=float(result.recomputed_force_kn[cell]),
                recomputed_resistance_kn=float(result.recomputed_resistance_kn[cell]),
                alpha_relative_error=float(result.alpha_relative_error[cell]),
            )
        rows.append(row)
    return rows


def _plot(result, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    distance_km = (result.position_m - result.position_m[0]) / 1000.0
    cell_km = 0.5 * (distance_km[:-1] + distance_km[1:])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(distance_km, result.speed_kmh, color="tab:blue", label="optimized speed")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(cell_km, result.optimized_force_kn, label="optimized force", alpha=0.8)
    axes[1].plot(cell_km, result.recomputed_force_kn, "--", label="recomputed force")
    axes[1].set_xlabel("Distance (km)")
    axes[1].set_ylabel("Force (kN)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(
        f"{result.section_name}: target={result.target_time_s:.1f}s, "
        f"actual={result.actual_time_s:.3f}s, net={result.actual_net_energy_kwh:.3f}kWh"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "convex_speed_force_trajectory.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.mass_t) or args.mass_t <= 0.0:
        raise ValueError("mass-t must be finite and positive")
    model_args = build_model_parser().parse_args([])
    model = build_bidirectional_model(model_args)
    if args.list_sections:
        for direction, train_idx in (("up", 0), ("down", model_args.up_trains)):
            for section_idx, environment in enumerate(model[3][train_idx]):
                print(
                    f"{direction:4s} {environment.section.name:12s} "
                    f"{float(model[4][train_idx, section_idx]):.1f}s"
                )
        return None

    train_idx = 0 if args.direction == "up" else model_args.up_trains
    environments = model[3][train_idx]
    section_idx = next(
        (idx for idx, env in enumerate(environments) if env.section.name == args.section),
        None,
    )
    if section_idx is None:
        raise ValueError(f"Unknown {args.direction} section {args.section}; use --list-sections")
    section = environments[section_idx].section
    target = (
        float(args.target_time_s)
        if args.target_time_s is not None
        else float(model[4][train_idx, section_idx])
    )
    config = replace(
        create_train_config_from_mass("Convex single scenario", args.mass_t),
        dt_s=float(model_args.simulation_dt_s),
    )
    if args.backend == "cvxpy":
        options = CvxpyLowerOptions(
            spatial_segments=args.segments,
            speed_limit_ratio=args.speed_limit_ratio,
            output_flag=args.solver_log,
        )
        solver = CvxpyLowerSolver(options)
    else:
        options = ConvexLowerOptions(
            spatial_segments=args.segments,
            speed_limit_ratio=args.speed_limit_ratio,
            output_flag=args.solver_log,
        )
        solver = ConvexLowerSolver(options)
    result = solver.solve(section, config, target)

    output_dir = args.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = _summary(result)
    summary["direction"] = args.direction
    summary["mass_t"] = args.mass_t
    summary["backend"] = args.backend
    summary["options"] = asdict(options)
    (output_dir / "convex_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result.has_solution:
        write_csv(output_dir / "convex_trajectory.csv", _trajectory_rows(result))
        _plot(result, output_dir)
        lower_result = solver.to_lower_result(result, section, config)
        write_csv(
            output_dir / "lower_result_trajectory.csv",
            [asdict(point) for point in lower_result.trajectory],
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_dir}")
    return result


if __name__ == "__main__":
    main()

