"""Custom power-zone timetable optimization experiment entrypoint.

The detailed responsibilities live in focused modules:
line_profile.py, train_model.py, lower_control.py, mappo.py and visualization.py.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

try:
    from .config import (
        DEFAULT_STATION_DWELL_S,
        DEFAULT_LINE_DIR,
        INITIAL_TIMETABLE,
        REAL_SUPPLY_ZONE_NAMES,
        REAL_SUPPLY_ZONE_SECTION_GROUPS,
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
    from .visualization import (
        export_mappo_visualizations,
        export_validation_visualizations,
    )
except ImportError:
    from config import (
        DEFAULT_STATION_DWELL_S,
        DEFAULT_LINE_DIR,
        INITIAL_TIMETABLE,
        REAL_SUPPLY_ZONE_NAMES,
        REAL_SUPPLY_ZONE_SECTION_GROUPS,
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
    from visualization import (
        export_mappo_visualizations,
        export_validation_visualizations,
    )


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical MARL timetable optimization with custom power-supply "
            "zones. The default uses the real THP-DGNL, DGNL-SY, SY-ZL, "
            "ZL-ST and ST-ZCP power-supply zones."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("validate", "train"),
        default="validate",
        help="Validate the route model or run lower-level DQN plus upper-level MAPPO.",
    )
    parser.add_argument("--line-dir", type=Path, default=DEFAULT_LINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_power_zone_real")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR / "hierarchical_mappo_power_zone_real_checkpoint.pt",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--lower-episodes",
        type=int,
        default=300,
        help="Lower-level DQN training episodes per heterogeneous train.",
    )
    parser.add_argument("--upper-episodes", type=int, default=1000)
    parser.add_argument("--upper-steps", type=int, default=24)
    parser.add_argument("--response-span", type=int, default=10)
    parser.add_argument("--response-step", type=int, default=2)
    parser.add_argument(
        "--supply-zone-size",
        type=int,
        default=0,
        help="Fallback number of interstation sections per power-supply zone.",
    )
    parser.add_argument(
        "--supply-zone-section-counts",
        type=str,
        default="",
        help=(
            "Comma-separated custom power-supply zone section counts, e.g. 4,5,5,5. "
            "When omitted and --supply-zone-size is 0, real named zones are used."
        ),
    )
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument(
        "--initial-headway",
        type=int,
        default=480,
        help="Initial train departure headway in seconds (default: 8 minutes).",
    )
    parser.add_argument("--minimum-headway", type=int, default=120)
    parser.add_argument("--station-dwell-s", type=float, default=DEFAULT_STATION_DWELL_S)
    parser.add_argument("--departure-span", type=int, default=60)
    parser.add_argument("--departure-quantum", type=int, default=5)
    parser.add_argument("--traction-energy-weight", type=float, default=0.5)
    parser.add_argument("--regenerative-match-weight", type=float, default=0.5)
    parser.add_argument(
        "--upper-control-mode",
        choices=("departure", "joint"),
        default="departure",
        help="Optimize departure times only by default; use joint for departure and interstation-time optimization.",
    )
    parser.add_argument(
        "--upper-algorithm",
        choices=("ppo",),
        default="ppo",
        help="MAPPO uses independent PPO actors and a centralized critic.",
    )
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_supply_zone_section_counts(value: str) -> tuple[int, ...]:
    value = value.strip()
    if not value:
        return ()
    counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if any(item <= 0 for item in counts):
        raise ValueError("--supply-zone-section-counts must contain positive integers")
    return counts


def build_route_and_lower_environments(
    line_dir: Path,
    train_count: int,
) -> tuple[
    LineProfile,
    list[TrainConfig],
    list[list[LowerSectionEnvironment]],
]:
    profile = LineProfile.from_excel(
        line_dir / "slope.xlsx",
        line_dir / "curvature.xlsx",
    )
    profile.print_summary()
    train_configs = configured_train_configs(train_count)
    print("Configured heterogeneous trains")
    for config in train_configs:
        print(
            f"  {config.name}: mass={config.mass_kg / 1000.0:.1f}t "
            f"max_force={config.max_traction_force_n / 1000.0:.1f}kN "
            f"max_power={config.max_traction_power_w / 1000.0:.1f}kW"
        )
    print("Configured driving-smoothness soft constraints")
    print(
        f"  preferred traction/coast hold: "
        f"{DRIVING_SMOOTHNESS.preferred_traction_hold_s:.0f}/"
        f"{DRIVING_SMOOTHNESS.preferred_coast_hold_s:.0f}s"
    )
    print(
        f"  heuristic release/re-entry margin: "
        f"{DRIVING_SMOOTHNESS.heuristic_traction_release_margin_mps:.1f}/"
        f"{DRIVING_SMOOTHNESS.heuristic_coast_reentry_margin_mps:.1f}m/s"
    )
    print(
        f"  arrival tolerance: prefer +/-{PRIMARY_ARRIVAL_TOLERANCE_S:.1f}s, "
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
    for train_idx, (config, environments) in enumerate(
        zip(train_configs, environments_by_train)
    ):
        print(f"\nValidating {config.name}")
        results = validate_initial_timetable(environments, INITIAL_TIMETABLE)
        feasible_count = sum(result.feasible for result in results)
        strict_count = sum(result.strict_timetable_feasible for result in results)
        relaxed_only_count = sum(
            result.relaxed_timetable_feasible and not result.strict_timetable_feasible
            for result in results
        )
        net_energy = sum(result.net_energy_kwh for result in results)
        print(
            f"{config.name} validation: {feasible_count}/{len(results)} "
            f"sections feasible ({strict_count} strict, {relaxed_only_count} relaxed), "
            f"net energy {net_energy:.2f} kWh."
        )
        if train_idx == 0:
            export_validation_visualizations(profile, results, output_dir)
    print(f"Validation figures written to: {output_dir.resolve()}")


def run_training(
    args: argparse.Namespace,
    profile: LineProfile,
    train_configs: list[TrainConfig],
    environments_by_train: list[list[LowerSectionEnvironment]],
) -> None:
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

    supply_zone_counts = parse_supply_zone_section_counts(args.supply_zone_section_counts)
    use_real_supply_zones = not supply_zone_counts and args.supply_zone_size <= 0
    if use_real_supply_zones:
        print("Using real power-supply zones:")
        for name, group in zip(REAL_SUPPLY_ZONE_NAMES, REAL_SUPPLY_ZONE_SECTION_GROUPS):
            print(f"  {name}: {', '.join(group)}")
        print("  unassigned connector sections do not absorb regenerative energy.")

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
        supply_zone_size=args.supply_zone_size,
        supply_zone_section_counts=supply_zone_counts,
        supply_zone_section_groups=REAL_SUPPLY_ZONE_SECTION_GROUPS if use_real_supply_zones else None,
        supply_zone_names=REAL_SUPPLY_ZONE_NAMES if use_real_supply_zones else None,
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
        output_dir=args.output_dir,
    )

    print("\nBest four-train coordinated timetable:")
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
        f"regenerative utilization: {training_result.regenerative_utilization * 100:.2f}%, "
        f"supply zones: {training_result.supply_zone_count}."
    )
    print(f"Checkpoint written to: {args.checkpoint.resolve()}")
    print(f"Training figures and CSV files written to: {args.output_dir.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    profile, train_configs, environments_by_train = build_route_and_lower_environments(
        args.line_dir,
        args.train_count,
    )
    if args.mode == "validate":
        run_validation(profile, train_configs, environments_by_train, args.output_dir)
        return
    run_training(args, profile, train_configs, environments_by_train)


if __name__ == "__main__":
    main()
