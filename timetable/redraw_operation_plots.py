"""Redraw operation plots from saved rollout CSV files without retraining."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .config import (
        DEFAULT_DOWN_CURVATURE_PATH,
        DEFAULT_DOWN_SLOPE_PATH,
        DEFAULT_LINE_DIR,
    )
    from .line_profile import LineProfile
    from .lower_control import LowerResult, TrajectoryPoint
    from .visualization import plot_section_operation_overview, plot_train_operation
except ImportError:
    from config import (
        DEFAULT_DOWN_CURVATURE_PATH,
        DEFAULT_DOWN_SLOPE_PATH,
        DEFAULT_LINE_DIR,
    )
    from line_profile import LineProfile
    from lower_control import LowerResult, TrajectoryPoint
    from visualization import plot_section_operation_overview, plot_train_operation


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Redraw full-line and per-section operation plots from saved "
            "*_operation.csv files. This does not train or run the lower policy."
        )
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Existing experiment result directory containing *_operation.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for redrawn figures. Defaults to <result-dir>/redrawn_plots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write redrawn figures back into --result-dir instead of redrawn_plots.",
    )
    parser.add_argument(
        "--line-dir",
        type=Path,
        default=None,
        help="Directory containing slope.xlsx and curvature.xlsx for the up direction.",
    )
    parser.add_argument(
        "--down-curvature",
        type=Path,
        default=None,
        help="Down-direction curvature workbook.",
    )
    parser.add_argument(
        "--down-slope",
        type=Path,
        default=None,
        help="Down-direction slope workbook.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_online_operation.csv",
        help="CSV glob pattern to redraw, e.g. '*_operation.csv'.",
    )
    return parser


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def consecutive_section_groups(rows: Sequence[dict[str, str]]) -> Iterable[list[dict[str, str]]]:
    group: list[dict[str, str]] = []
    current_name: str | None = None
    for row in rows:
        section_name = row.get("section_name", "")
        if group and section_name != current_name:
            yield group
            group = []
        group.append(row)
        current_name = section_name
    if group:
        yield group


def transitions(values: Sequence[str]) -> int:
    return sum(1 for previous, current in zip(values, values[1:]) if previous != current)


def force_jump_index(forces_kn: Sequence[float]) -> float:
    if len(forces_kn) < 2:
        return 0.0
    scale = max(max(abs(value) for value in forces_kn), 1.0)
    return sum(abs(current - previous) / scale for previous, current in zip(forces_kn, forces_kn[1:]))


def result_from_rows(section_rows: Sequence[dict[str, str]]) -> LowerResult:
    first = section_rows[0]
    section_name = first["section_name"]
    trajectory: list[TrajectoryPoint] = []
    for row in section_rows:
        radius = row.get("radius_m", "")
        trajectory.append(
            TrajectoryPoint(
                section_name=section_name,
                elapsed_s=parse_float(row.get("section_time_s")),
                position_m=parse_float(row.get("position_m")),
                speed_kmh=parse_float(row.get("speed_kmh")),
                force_kn=parse_float(row.get("force_kn")),
                resistance_kn=parse_float(row.get("resistance_kn")),
                speed_limit_kmh=parse_float(row.get("speed_limit_kmh")),
                gradient_per_mille=parse_float(row.get("gradient_per_mille")),
                radius_m=None if radius in ("", None) else parse_float(radius),
                requested_action=parse_int(row.get("requested_action")),
                applied_action=parse_int(row.get("applied_action")),
                regime=row.get("regime", "unknown"),
                traction_kwh=parse_float(row.get("traction_kwh")),
                regenerative_kwh=parse_float(row.get("regenerative_kwh")),
            )
        )

    actual_time_s = max((point.elapsed_s for point in trajectory), default=0.0)
    traction_kwh = max(0.0, trajectory[-1].traction_kwh - trajectory[0].traction_kwh)
    regenerative_kwh = max(0.0, trajectory[-1].regenerative_kwh - trajectory[0].regenerative_kwh)
    regimes = [point.regime for point in trajectory]
    forces = [point.force_kn for point in trajectory]
    traction_coast = sum(
        1
        for previous, current in zip(forces, forces[1:])
        if (previous > 1.0 and abs(current) <= 1.0)
        or (abs(previous) <= 1.0 and current > 1.0)
    )
    traction_brake = sum(
        1
        for previous, current in zip(forces, forces[1:])
        if (previous > 1.0 and current < -1.0)
        or (previous < -1.0 and current > 1.0)
    )
    max_violation = max(
        (max(0.0, point.speed_kmh - point.speed_limit_kmh) for point in trajectory),
        default=0.0,
    )
    return LowerResult(
        section_name=section_name,
        target_time_s=actual_time_s,
        actual_time_s=actual_time_s,
        traction_kwh=traction_kwh,
        regenerative_kwh=regenerative_kwh,
        net_energy_kwh=traction_kwh - regenerative_kwh,
        arrival_speed_mps=trajectory[-1].speed_kmh / 3.6 if trajectory else 0.0,
        feasible=True,
        timed_out=False,
        regime_switches=transitions(regimes),
        traction_brake_transitions=traction_brake,
        safety_brake_overrides=0,
        traction_coast_transitions=traction_coast,
        short_dwell_transitions=0,
        force_jump_index=force_jump_index(forces),
        smoothness_penalty=0.0,
        jerk_integral_mps3_s=0.0,
        comfort_penalty=0.0,
        traction_reward_component=0.0,
        punctuality_reward_component=0.0,
        comfort_reward_component=0.0,
        max_speed_limit_violation_kmh=max_violation,
        speed_regulation_steps=0,
        speed_limit_action_shield_overrides=0,
        strict_timetable_feasible=True,
        accepted_arrival_tolerance_s=None,
        episode_reward=0.0,
        trajectory=trajectory,
    )


def results_from_operation_csv(path: Path) -> list[LowerResult]:
    rows = read_csv_rows(path)
    if not rows:
        return []
    return [result_from_rows(group) for group in consecutive_section_groups(rows)]


def build_profiles(args: argparse.Namespace) -> tuple[LineProfile, LineProfile]:
    # Prefer project-local data over paths embedded in historical metadata;
    # older result folders commonly point to files on the original desktop.
    line_dir = args.line_dir or DEFAULT_LINE_DIR
    down_curvature = args.down_curvature or DEFAULT_DOWN_CURVATURE_PATH
    down_slope = args.down_slope or DEFAULT_DOWN_SLOPE_PATH
    up_profile = LineProfile.from_excel(line_dir / "slope.xlsx", line_dir / "curvature.xlsx")
    down_profile = LineProfile.reversed_with_slope_and_curvature(
        up_profile,
        down_slope,
        down_curvature,
    )
    return up_profile, down_profile


def operation_csv_files(result_dir: Path, pattern: str) -> list[Path]:
    files = sorted(result_dir.glob(pattern))
    return [
        path
        for path in files
        if not path.name.endswith("_time_speed_with_dwell.csv")
        and "section_plot_reference" not in path.name
    ]


def redraw_file(path: Path, up_profile: LineProfile, down_profile: LineProfile, output_dir: Path) -> None:
    results = results_from_operation_csv(path)
    if not results:
        print(f"Skipped empty operation CSV: {path.name}")
        return
    profile = down_profile if path.name.startswith("down_") else up_profile
    stem = path.stem
    plot_train_operation(profile, results, output_dir, stem)
    plot_section_operation_overview(profile, results, output_dir, stem.replace("_operation", ""))
    print(f"Redrawn: {path.name}")


def main() -> None:
    args = build_parser().parse_args()
    result_dir = args.result_dir.resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")
    output_dir = result_dir if args.overwrite else (args.output_dir or result_dir / "redrawn_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    up_profile, down_profile = build_profiles(args)
    files = operation_csv_files(result_dir, args.pattern)
    if not files:
        raise FileNotFoundError(
            f"No operation CSV files matching {args.pattern!r} in {result_dir}"
        )
    for path in files:
        redraw_file(path, up_profile, down_profile, output_dir)
    print(f"Saved redrawn operation plots to: {output_dir}")


if __name__ == "__main__":
    main()
