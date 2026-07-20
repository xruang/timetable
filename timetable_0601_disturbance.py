"""Timetable optimization with random dwell-overrun disturbances.

This entrypoint keeps the original timetable scripts unchanged. It injects fixed
station dwell overruns into selected trains, then lets the upper MAPPO planner
adjust departures under the same safety headway constraint.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    from .config import (
        DEFAULT_STATION_DWELL_S,
        DEFAULT_LINE_DIR,
        INITIAL_TIMETABLE,
        REAL_SUPPLY_ZONE_NAMES,
        REAL_SUPPLY_ZONE_SECTION_GROUPS,
        SECTION_NAMES,
    )
    from .line_profile import LineProfile
    from .lower_control import (
        DQNAgent,
        DRIVING_SMOOTHNESS,
        LowerSectionEnvironment,
        PRIMARY_ARRIVAL_TOLERANCE_S,
        RELAXED_ARRIVAL_TOLERANCE_S,
        build_response_tables,
        train_lower_agents,
        validate_initial_timetable,
    )
    from .mappo import (
        IndependentActorMAPPOAgent,
        MultiTrainTimetableEnvironment,
        save_multi_train_checkpoint,
        train_multi_train_mappo,
    )
    from .train_model import TrainConfig, TrainDynamics, configured_train_configs
    from .visualization import export_mappo_visualizations, export_validation_visualizations, write_csv
except ImportError:
    from config import (
        DEFAULT_STATION_DWELL_S,
        DEFAULT_LINE_DIR,
        INITIAL_TIMETABLE,
        REAL_SUPPLY_ZONE_NAMES,
        REAL_SUPPLY_ZONE_SECTION_GROUPS,
        SECTION_NAMES,
    )
    from line_profile import LineProfile
    from lower_control import (
        DQNAgent,
        DRIVING_SMOOTHNESS,
        LowerSectionEnvironment,
        PRIMARY_ARRIVAL_TOLERANCE_S,
        RELAXED_ARRIVAL_TOLERANCE_S,
        build_response_tables,
        train_lower_agents,
        validate_initial_timetable,
    )
    from mappo import (
        IndependentActorMAPPOAgent,
        MultiTrainTimetableEnvironment,
        save_multi_train_checkpoint,
        train_multi_train_mappo,
    )
    from train_model import TrainConfig, TrainDynamics, configured_train_configs
    from visualization import export_mappo_visualizations, export_validation_visualizations, write_csv


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Four-train timetable optimization with dwell-overrun disturbances."
    )
    parser.add_argument("--mode", choices=("validate", "train"), default="train")
    parser.add_argument("--line-dir", type=Path, default=DEFAULT_LINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_disturbance")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the per-run result folder. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to the current run result folder.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lower-episodes", type=int, default=300)
    parser.add_argument("--upper-episodes", type=int, default=1000)
    parser.add_argument("--upper-steps", type=int, default=24)
    parser.add_argument("--response-span", type=int, default=10)
    parser.add_argument("--response-step", type=int, default=2)
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--initial-headway", type=int, default=480)
    parser.add_argument("--minimum-headway", type=int, default=120)
    parser.add_argument("--station-dwell-s", type=float, default=DEFAULT_STATION_DWELL_S)
    parser.add_argument("--departure-span", type=int, default=90)
    parser.add_argument("--departure-quantum", type=int, default=5)
    parser.add_argument("--traction-energy-weight", type=float, default=0.5)
    parser.add_argument("--regenerative-match-weight", type=float, default=0.5)
    parser.add_argument(
        "--upper-control-mode",
        choices=("departure", "joint"),
        default="departure",
    )
    parser.add_argument(
        "--disturbance-trains",
        type=str,
        default="2,3",
        help="1-based train indices that receive random dwell overruns.",
    )
    parser.add_argument(
        "--disturbance-count-per-train",
        type=int,
        default=4,
        help="Number of disturbed intermediate station stops per selected train.",
    )
    parser.add_argument(
        "--dwell-overrun-s",
        type=float,
        default=10.0,
        help="Extra dwell time added after the selected interstation section.",
    )
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_run_outputs(args: argparse.Namespace) -> None:
    base_run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_root = args.output_dir
    run_name = base_run_name
    run_dir = args.output_dir / run_name
    suffix = 2
    while run_dir.exists():
        run_name = f"{base_run_name}_{suffix:02d}"
        run_dir = args.output_dir / run_name
        suffix += 1
    args.run_name = run_name
    args.output_dir = run_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is None:
        args.checkpoint = args.output_dir / "hierarchical_mappo_disturbance_checkpoint.pt"


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_run_documentation(
    output_dir: Path,
    args: argparse.Namespace,
    train_configs: Sequence[TrainConfig],
    disturbance_rows: Sequence[dict[str, object]],
    station_delays: np.ndarray,
    phase: str,
) -> None:
    section_index_by_name = {name: idx for idx, name in enumerate(SECTION_NAMES)}

    def normalise_zone_group(group: Sequence[Any]) -> tuple[list[int], list[str]]:
        section_indices: list[int] = []
        section_names: list[str] = []
        for item in group:
            if isinstance(item, str):
                section_names.append(item)
                section_indices.append(section_index_by_name[item] + 1)
            else:
                section_idx = int(item)
                section_indices.append(section_idx + 1)
                section_names.append(SECTION_NAMES[section_idx])
        return section_indices, section_names

    supply_zone_rows = []
    for zone_name, group in zip(REAL_SUPPLY_ZONE_NAMES, REAL_SUPPLY_ZONE_SECTION_GROUPS):
        section_indices, section_names = normalise_zone_group(group)
        supply_zone_rows.append(
            {
                "zone_name": zone_name,
                "section_indices_1based": section_indices,
                "section_names": section_names,
            }
        )

    station_delay_total_s = float(np.sum(station_delays))
    metadata = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "script": Path(__file__).name,
        "arguments": json_ready(vars(args)),
        "initial_timetable_s": list(INITIAL_TIMETABLE),
        "train_configs": [asdict(config) for config in train_configs],
        "driving_smoothness": asdict(DRIVING_SMOOTHNESS),
        "arrival_tolerance_s": {
            "primary": PRIMARY_ARRIVAL_TOLERANCE_S,
            "relaxed": RELAXED_ARRIVAL_TOLERANCE_S,
        },
        "supply_zones": supply_zone_rows,
        "disturbances": list(disturbance_rows),
        "station_delay_after_section_s": station_delays.tolist(),
        "station_delay_total_s": station_delay_total_s,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(metadata), handle, ensure_ascii=False, indent=2)

    lines = [
        "# Disturbance Timetable Simulation Run",
        "",
        f"- Run name: `{args.run_name}`",
        f"- Created at: `{metadata['created_at']}`",
        f"- Mode/phase: `{args.mode}` / `{phase}`",
        f"- Seed: `{args.seed}`",
        f"- Output directory: `{output_dir.resolve()}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Initial headway: `{args.initial_headway}s`; minimum safe headway: `{args.minimum_headway}s`",
        f"- Scheduled station dwell: `{args.station_dwell_s:.1f}s` at every intermediate station",
        f"- Upper control mode: `{args.upper_control_mode}`",
        f"- Lower episodes: `{args.lower_episodes}`; upper episodes: `{args.upper_episodes}`",
        f"- Response table sampling: +/-`{args.response_span}s`, step `{args.response_step}s`",
        f"- Total injected dwell overrun: `{station_delay_total_s:.1f}s`",
        "",
        "## Train Configuration",
        "",
        "| Train | Mass (kg) | Max traction force (N) | Max traction power (W) | Davis A | Davis B | Davis C |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in train_configs:
        lines.append(
            f"| {config.name} | {config.mass_kg:.1f} | {config.max_traction_force_n:.1f} | "
            f"{config.max_traction_power_w:.1f} | {config.davis_a:.6g} | "
            f"{config.davis_b:.6g} | {config.davis_c:.6g} |"
        )
    lines.extend(["", "## Dwell-Overrun Disturbances", ""])
    if disturbance_rows:
        lines.extend(
            [
                "| Train | After section | Station | Overrun (s) |",
                "|---:|---|---|---:|",
            ]
        )
        for row in disturbance_rows:
            lines.append(
                f"| {row['train_index']} | {row['after_section_name']} | "
                f"{row['station_name']} | {float(row['dwell_overrun_s']):.1f} |"
            )
    else:
        lines.append("No dwell-overrun disturbances were injected in this run.")
    lines.extend(["", "## Supply Zones", ""])
    lines.extend(["| Zone | Sections |", "|---|---|"])
    for row in supply_zone_rows:
        section_names = ", ".join(row["section_names"])
        lines.append(f"| {row['zone_name']} | {section_names} |")
    lines.append("")
    lines.append("The full machine-readable configuration is saved in `run_metadata.json`.")
    (output_dir / "run_description.md").write_text("\n".join(lines), encoding="utf-8")


def parse_train_indices(value: str, train_count: int) -> list[int]:
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item) - 1
        if idx < 0 or idx >= train_count:
            raise ValueError(f"Invalid train index: {item}")
        indices.append(idx)
    return sorted(set(indices))


def build_station_delay_matrix(
    train_count: int,
    section_count: int,
    disturbed_train_indices: Sequence[int],
    count_per_train: int,
    dwell_overrun_s: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    delays = np.zeros((train_count, section_count), dtype=np.float32)
    candidate_sections = np.arange(0, section_count - 1, dtype=int)
    count = min(max(0, int(count_per_train)), len(candidate_sections))
    for train_idx in disturbed_train_indices:
        selected = rng.choice(candidate_sections, size=count, replace=False)
        delays[train_idx, selected] = float(dwell_overrun_s)
    return delays


def station_delay_rows(delays: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for train_idx in range(delays.shape[0]):
        for section_idx in range(delays.shape[1]):
            delay_s = float(delays[train_idx, section_idx])
            if delay_s <= 0.0:
                continue
            rows.append(
                {
                    "train_index": train_idx + 1,
                    "after_section_index": section_idx + 1,
                    "after_section_name": SECTION_NAMES[section_idx],
                    "station_name": SECTION_NAMES[section_idx].split("-")[-1],
                    "dwell_overrun_s": delay_s,
                }
            )
    return rows


def disturbed_station_schedule_rows(
    timetables_s: Sequence[Sequence[float]],
    departure_times_s: Sequence[float],
    station_delays_s: np.ndarray,
    station_dwell_s: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    timetables = np.asarray(timetables_s, dtype=float)
    departures = np.asarray(departure_times_s, dtype=float)
    for train_idx in range(timetables.shape[0]):
        current = float(departures[train_idx])
        rows.append(
            {
                "train_index": train_idx + 1,
                "station_index": 1,
                "station_name": SECTION_NAMES[0].split("-")[0],
                "arrival_time_s": current,
                "departure_time_s": current,
                "scheduled_dwell_s": 0.0,
                "dwell_overrun_s": 0.0,
                "total_dwell_s": 0.0,
            }
        )
        for section_idx in range(timetables.shape[1]):
            current += float(timetables[train_idx, section_idx])
            delay_s = float(station_delays_s[train_idx, section_idx])
            base_dwell_s = float(station_dwell_s) if section_idx < timetables.shape[1] - 1 else 0.0
            total_dwell_s = base_dwell_s + (delay_s if section_idx < timetables.shape[1] - 1 else 0.0)
            rows.append(
                {
                    "train_index": train_idx + 1,
                    "station_index": section_idx + 2,
                    "station_name": SECTION_NAMES[section_idx].split("-")[-1],
                    "arrival_time_s": current,
                    "departure_time_s": current + total_dwell_s if section_idx < timetables.shape[1] - 1 else "",
                    "scheduled_dwell_s": base_dwell_s,
                    "dwell_overrun_s": delay_s if section_idx < timetables.shape[1] - 1 else 0.0,
                    "total_dwell_s": total_dwell_s,
                }
            )
            current += total_dwell_s
    return rows


def build_route_and_lower_environments(
    line_dir: Path,
    train_count: int,
) -> tuple[LineProfile, list[TrainConfig], list[list[LowerSectionEnvironment]]]:
    profile = LineProfile.from_excel(line_dir / "slope.xlsx", line_dir / "curvature.xlsx")
    profile.print_summary()
    train_configs = configured_train_configs(train_count)
    print("Configured heterogeneous trains")
    for config in train_configs:
        print(
            f"  {config.name}: mass={config.mass_kg / 1000.0:.1f}t "
            f"max_force={config.max_traction_force_n / 1000.0:.1f}kN "
            f"max_power={config.max_traction_power_w / 1000.0:.1f}kW"
        )
    print(
        f"Arrival tolerance: prefer +/-{PRIMARY_ARRIVAL_TOLERANCE_S:.1f}s, "
        f"relax to +/-{RELAXED_ARRIVAL_TOLERANCE_S:.1f}s only if necessary"
    )
    environments_by_train = [
        [
            LowerSectionEnvironment(section, TrainDynamics(config))
            for section in profile.sections
        ]
        for config in train_configs
    ]
    return profile, train_configs, environments_by_train


def run_validation(
    profile: LineProfile,
    train_configs: list[TrainConfig],
    environments_by_train: list[list[LowerSectionEnvironment]],
    output_dir: Path,
) -> None:
    for train_idx, (config, environments) in enumerate(zip(train_configs, environments_by_train)):
        print(f"\nValidating {config.name}")
        results = validate_initial_timetable(environments, INITIAL_TIMETABLE)
        if train_idx == 0:
            export_validation_visualizations(profile, results, output_dir)


def run_training(
    args: argparse.Namespace,
    profile: LineProfile,
    train_configs: list[TrainConfig],
    environments_by_train: list[list[LowerSectionEnvironment]],
) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    disturbed_train_indices = parse_train_indices(args.disturbance_trains, args.train_count)
    station_delays = build_station_delay_matrix(
        args.train_count,
        len(INITIAL_TIMETABLE),
        disturbed_train_indices,
        args.disturbance_count_per_train,
        args.dwell_overrun_s,
        args.seed,
    )
    rows = station_delay_rows(station_delays)
    write_csv(output_dir / "station_dwell_overrun_disturbances.csv", rows)
    write_run_documentation(
        output_dir=output_dir,
        args=args,
        train_configs=train_configs,
        disturbance_rows=rows,
        station_delays=station_delays,
        phase="train",
    )
    print("Injected station dwell-overrun disturbances:")
    for row in rows:
        print(
            f"  Train {row['train_index']}: after {row['after_section_name']} "
            f"at {row['station_name']} +{row['dwell_overrun_s']:.1f}s"
        )
    print("Using real power-supply zones: " + ", ".join(REAL_SUPPLY_ZONE_NAMES))

    lower_agents: list[DQNAgent] = []
    lower_history = []
    response_tables_by_train = []
    for config, environments in zip(train_configs, environments_by_train):
        lower_agent = DQNAgent(
            state_dim=LowerSectionEnvironment.STATE_DIM,
            action_dim=LowerSectionEnvironment.ACTION_DIM,
        )
        lower_agents.append(lower_agent)
        lower_history.extend(
            train_lower_agents(
                environments,
                lower_agent,
                INITIAL_TIMETABLE,
                episodes=args.lower_episodes,
                target_span_s=args.response_span,
                train_name=config.name,
            )
        )
        response_tables_by_train.append(
            build_response_tables(
                environments,
                INITIAL_TIMETABLE,
                time_span_s=args.response_span,
                time_step_s=args.response_step,
                learned_controller=lower_agent,
                train_name=config.name,
            )
        )

    upper_environment = MultiTrainTimetableEnvironment(
        initial_timetable=INITIAL_TIMETABLE,
        response_tables_by_train=response_tables_by_train,
        train_count=args.train_count,
        initial_headway_s=args.initial_headway,
        minimum_headway_s=args.minimum_headway,
        station_dwell_s=args.station_dwell_s,
        time_span_s=args.response_span,
        time_quantum_s=args.response_step,
        departure_span_s=args.departure_span,
        departure_quantum_s=args.departure_quantum,
        traction_energy_weight=args.traction_energy_weight,
        regenerative_match_weight=args.regenerative_match_weight,
        control_mode=args.upper_control_mode,
        max_steps=args.upper_steps,
        supply_zone_section_groups=REAL_SUPPLY_ZONE_SECTION_GROUPS,
        supply_zone_names=REAL_SUPPLY_ZONE_NAMES,
        station_delay_after_section_s=station_delays,
    )
    upper_agent = IndependentActorMAPPOAgent(
        train_count=upper_environment.train_count,
        observation_dim=upper_environment.local_observation_dim,
        action_dim=upper_environment.action_dim,
        central_state_dim=upper_environment.central_state_dim,
        active_agent_mask=upper_environment.active_agent_mask,
    )
    training_result, upper_history = train_multi_train_mappo(
        upper_environment,
        upper_agent,
        episodes=args.upper_episodes,
    )
    save_multi_train_checkpoint(
        args.checkpoint,
        lower_agents,
        upper_agent,
        training_result,
        [asdict(config) for config in train_configs],
        asdict(DRIVING_SMOOTHNESS),
    )
    export_mappo_visualizations(
        profile=profile,
        environments_by_train=environments_by_train,
        lower_controllers=lower_agents,
        tables_by_train=response_tables_by_train,
        upper_environment=upper_environment,
        result=training_result,
        lower_history=lower_history,
        upper_history=upper_history,
        output_dir=output_dir,
    )

    write_csv(
        output_dir / "disturbed_station_schedule_times.csv",
        disturbed_station_schedule_rows(
            training_result.timetables_s,
            training_result.departure_times_s,
            station_delays,
            args.station_dwell_s,
        ),
    )

    print("\nBest disturbance-resilient timetable:")
    for train_index, timetable in enumerate(training_result.timetables_s, start=1):
        values = ", ".join(str(int(round(value))) for value in timetable)
        print(f"  Train {train_index}: [{values}]")
    print(
        "Stable departure times: "
        + ", ".join(f"{value / 60.0:.2f} min" for value in training_result.departure_times_s)
    )
    print(
        f"Grid energy: {training_result.grid_energy_kwh:.2f} kWh, "
        f"matched regenerative energy: {training_result.matched_regenerative_kwh:.2f} kWh, "
        f"regenerative utilization: {training_result.regenerative_utilization * 100:.2f}%."
    )
    print(f"Training figures and CSV files written to: {output_dir.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    configure_run_outputs(args)
    seed_everything(args.seed)
    profile, train_configs, environments_by_train = build_route_and_lower_environments(
        args.line_dir,
        args.train_count,
    )
    if args.mode == "validate":
        write_run_documentation(
            output_dir=args.output_dir,
            args=args,
            train_configs=train_configs,
            disturbance_rows=[],
            station_delays=np.zeros((args.train_count, len(INITIAL_TIMETABLE)), dtype=np.float32),
            phase="validate",
        )
        run_validation(profile, train_configs, environments_by_train, args.output_dir)
        return
    run_training(args, profile, train_configs, environments_by_train)


if __name__ == "__main__":
    main()
