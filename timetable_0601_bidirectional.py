"""Bidirectional 6-up/6-down timetable and regenerative-energy optimization."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config import (
        DEFAULT_LINE_DIR,
        DEFAULT_STATION_DWELL_S,
        INITIAL_TIMETABLE,
        REAL_SUPPLY_ZONE_NAMES,
        REAL_SUPPLY_ZONE_SECTION_GROUPS,
        SECTION_NAMES,
    )
    from .line_profile import LineProfile
    from .lower_control import (
        DQNAgent,
        DRIVING_SMOOTHNESS,
        LOWER_REWARD,
        LowerSectionEnvironment,
        build_response_tables,
        evaluate_timetable,
        train_lower_agents,
    )
    from .mappo import (
        IndependentActorMAPPOAgent,
        MultiTrainTimetableEnvironment,
        save_multi_train_checkpoint,
        train_multi_train_mappo,
    )
    from .train_model import (
        TrainConfig,
        TrainDynamics,
        configured_bidirectional_train_configs,
    )
    from .visualization import (
        lower_result_summary_rows,
        plot_response_tables,
        plot_section_operation_overview,
        plot_train_operation,
        plot_training_process,
        write_csv,
    )
except ImportError:
    from config import (
        DEFAULT_LINE_DIR,
        DEFAULT_STATION_DWELL_S,
        INITIAL_TIMETABLE,
        REAL_SUPPLY_ZONE_NAMES,
        REAL_SUPPLY_ZONE_SECTION_GROUPS,
        SECTION_NAMES,
    )
    from line_profile import LineProfile
    from lower_control import (
        DQNAgent,
        DRIVING_SMOOTHNESS,
        LOWER_REWARD,
        LowerSectionEnvironment,
        build_response_tables,
        evaluate_timetable,
        train_lower_agents,
    )
    from mappo import (
        IndependentActorMAPPOAgent,
        MultiTrainTimetableEnvironment,
        save_multi_train_checkpoint,
        train_multi_train_mappo,
    )
    from train_model import (
        TrainConfig,
        TrainDynamics,
        configured_bidirectional_train_configs,
    )
    from visualization import (
        lower_result_summary_rows,
        plot_response_tables,
        plot_section_operation_overview,
        plot_train_operation,
        plot_training_process,
        write_csv,
    )


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bidirectional timetable optimization with 6 up and 6 down trains."
    )
    parser.add_argument("--line-dir", type=Path, default=DEFAULT_LINE_DIR)
    parser.add_argument(
        "--down-curvature",
        type=Path,
        default=Path.home() / "Desktop" / "curvature.xlsx",
    )
    parser.add_argument(
        "--down-slope",
        type=Path,
        default=Path.home() / "Desktop" / "slpoe.xlsx",
        help=(
            "Down-direction slope workbook. Its mileage is interpreted as the "
            "same physical mileage axis as the up direction, then converted to "
            "down travel coordinates with the gradient sign reversed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results_bidirectional",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--up-trains", type=int, default=6)
    parser.add_argument("--down-trains", type=int, default=6)
    parser.add_argument("--lower-episodes", type=int, default=3000)
    parser.add_argument(
        "--lower-policy-mode",
        choices=("fresh", "continue", "reuse"),
        default="fresh",
        help=(
            "fresh: train lower DQN from scratch; continue: load lower DQN "
            "weights then keep training; reuse: load lower DQN weights and skip "
            "lower training."
        ),
    )
    parser.add_argument(
        "--lower-policy-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint containing lower_policies, usually a previous "
            "hierarchical_mappo_bidirectional_checkpoint.pt."
        ),
    )
    parser.add_argument("--lower-target-span", type=int, default=10)
    parser.add_argument("--upper-episodes", type=int, default=1000)
    parser.add_argument("--upper-steps", type=int, default=24)
    parser.add_argument("--response-span", type=float, default=0.5)
    parser.add_argument("--response-step", type=float, default=0.1)
    parser.add_argument("--initial-headway", type=float, default=480.0)
    parser.add_argument("--minimum-headway", type=float, default=120.0)
    parser.add_argument(
        "--station-dwell-s",
        type=float,
        default=DEFAULT_STATION_DWELL_S,
    )
    parser.add_argument("--departure-span", type=float, default=90.0)
    parser.add_argument("--departure-quantum", type=float, default=5.0)
    parser.add_argument("--traction-energy-weight", type=float, default=0.5)
    parser.add_argument("--regenerative-match-weight", type=float, default=0.5)
    parser.add_argument(
        "--upper-control-mode",
        choices=("departure", "joint"),
        default="joint",
    )
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_run_outputs(args: argparse.Namespace) -> None:
    base_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = base_name
    run_dir = args.output_dir / run_name
    suffix = 2
    while run_dir.exists():
        run_name = f"{base_name}_{suffix:02d}"
        run_dir = args.output_dir / run_name
        suffix += 1
    args.run_name = run_name
    args.output_dir = run_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is None:
        args.checkpoint = run_dir / "hierarchical_mappo_bidirectional_checkpoint.pt"


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


def build_bidirectional_model(
    args: argparse.Namespace,
) -> tuple[
    LineProfile,
    LineProfile,
    list[TrainConfig],
    list[list[LowerSectionEnvironment]],
    np.ndarray,
    np.ndarray,
    list[list[int]],
    list[list[str]],
    list[str],
]:
    up_profile = LineProfile.from_excel(
        args.line_dir / "slope.xlsx",
        args.line_dir / "curvature.xlsx",
    )
    down_profile = LineProfile.reversed_with_slope_and_curvature(
        up_profile,
        args.down_slope,
        args.down_curvature,
    )
    train_configs = configured_bidirectional_train_configs(
        args.up_trains,
        args.down_trains,
    )
    total_trains = args.up_trains + args.down_trains
    direction_labels = ["up"] * args.up_trains + ["down"] * args.down_trains
    environments_by_train: list[list[LowerSectionEnvironment]] = []
    for train_idx, config in enumerate(train_configs):
        profile = up_profile if train_idx < args.up_trains else down_profile
        environments_by_train.append(
            [
                LowerSectionEnvironment(section, TrainDynamics(config))
                for section in profile.sections
            ]
        )

    up_timetable = np.asarray(INITIAL_TIMETABLE, dtype=np.float32)
    down_timetable = up_timetable[::-1].copy()
    initial_timetables = np.vstack(
        [
            np.tile(up_timetable, (args.up_trains, 1)),
            np.tile(down_timetable, (args.down_trains, 1)),
        ]
    )
    base_departures = np.concatenate(
        [
            np.arange(args.up_trains, dtype=np.float32) * args.initial_headway,
            np.arange(args.down_trains, dtype=np.float32) * args.initial_headway,
        ]
    )
    direction_groups = [
        list(range(args.up_trains)),
        list(range(args.up_trains, total_trains)),
    ]
    physical_sections = [
        list(SECTION_NAMES) for _ in range(args.up_trains)
    ] + [
        list(reversed(SECTION_NAMES)) for _ in range(args.down_trains)
    ]
    return (
        up_profile,
        down_profile,
        train_configs,
        environments_by_train,
        initial_timetables,
        base_departures,
        direction_groups,
        physical_sections,
        direction_labels,
    )


def station_names(profile: LineProfile) -> list[str]:
    return [profile.sections[0].name.split("-")[0]] + [
        section.name.split("-")[-1] for section in profile.sections
    ]


def down_line_profile_rows(
    up_profile: LineProfile,
    down_profile: LineProfile,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in down_profile.sections:
        for segment_idx, segment in enumerate(section.segments, start=1):
            physical_start_m = segment.start_m if segment.physical_start_m is None else segment.physical_start_m
            physical_end_m = segment.end_m if segment.physical_end_m is None else segment.physical_end_m
            rows.append(
                {
                    "down_section_name": section.name,
                    "segment_index": segment_idx,
                    "travel_start_m": segment.start_m,
                    "travel_end_m": segment.end_m,
                    "physical_mileage_from_m": physical_start_m,
                    "physical_mileage_to_m": physical_end_m,
                    "gradient_per_mille": segment.gradient_per_mille,
                    "radius_m": segment.radius_m,
                    "speed_limit_kmh": segment.speed_limit_kmh,
                }
            )
    return rows


def timetable_rows(
    environment: MultiTrainTimetableEnvironment,
    train_configs: Sequence[TrainConfig],
    direction_labels: Sequence[str],
    up_profile: LineProfile,
    down_profile: LineProfile,
) -> list[dict[str, Any]]:
    arrivals = environment.station_times()
    departures = environment.station_departure_times()
    rows: list[dict[str, Any]] = []
    for train_idx, config in enumerate(train_configs):
        profile = up_profile if direction_labels[train_idx] == "up" else down_profile
        names = station_names(profile)
        for station_idx, name in enumerate(names):
            is_terminal = station_idx == len(names) - 1
            rows.append(
                {
                    "train_index": train_idx + 1,
                    "train_name": config.name,
                    "direction": direction_labels[train_idx],
                    "station_index": station_idx + 1,
                    "station_name": name,
                    "arrival_time_s": arrivals[train_idx, station_idx],
                    "departure_time_s": (
                        ""
                        if is_terminal
                        else departures[
                            train_idx,
                            min(station_idx, departures.shape[1] - 1),
                        ]
                    ),
                    "scheduled_dwell_s": (
                        environment.station_dwell_s
                        if 0 < station_idx < len(names) - 1
                        else 0.0
                    ),
                }
            )
    return rows


def plot_bidirectional_timetable(
    environment: MultiTrainTimetableEnvironment,
    train_configs: Sequence[TrainConfig],
    direction_labels: Sequence[str],
    up_profile: LineProfile,
    down_profile: LineProfile,
    output_dir: Path,
) -> None:
    arrival_times = environment.station_times() / 60.0
    departure_times = environment.station_departure_times() / 60.0
    up_positions = [up_profile.sections[0].display_start_m] + [
        section.display_end_m for section in up_profile.sections
    ]
    down_positions = [down_profile.sections[0].display_start_m] + [
        section.display_end_m for section in down_profile.sections
    ]

    fig, axes = plt.subplots(2, 1, figsize=(15, 11))
    for train_idx, config in enumerate(train_configs):
        is_up = direction_labels[train_idx] == "up"
        color = "#1565C0" if is_up else "#C62828"
        positions = up_positions if is_up else down_positions
        plot_times = [arrival_times[train_idx, 0]]
        plot_positions = [positions[0]]
        for station_idx in range(1, len(positions)):
            plot_times.append(arrival_times[train_idx, station_idx])
            plot_positions.append(positions[station_idx])
            if station_idx < len(positions) - 1:
                plot_times.append(departure_times[train_idx, station_idx])
                plot_positions.append(positions[station_idx])
        axes[0].plot(
            plot_times,
            np.asarray(plot_positions) / 1000.0,
            color=color,
            linewidth=1.1,
            alpha=0.80,
            label=(
                "Up trains"
                if is_up and train_idx == 0
                else "Down trains"
                if not is_up and train_idx == direction_labels.index("down")
                else None
            ),
        )
        for station_idx in range(1, len(positions) - 1):
            axes[0].plot(
                [
                    arrival_times[train_idx, station_idx],
                    departure_times[train_idx, station_idx],
                ],
                [positions[station_idx] / 1000.0] * 2,
                color="#F9A825",
                linewidth=2.0,
                alpha=0.75,
            )
        axes[1].scatter(
            environment.departure_times_s[train_idx] / 60.0,
            train_idx + 1,
            color=color,
            s=35,
        )
        axes[1].text(
            environment.departure_times_s[train_idx] / 60.0,
            train_idx + 1,
            f"  {config.name}",
            va="center",
            fontsize=8,
        )

    axes[0].plot([], [], color="#F9A825", linewidth=2.0, label="60 s station dwell")
    axes[0].set_title("Bidirectional train timetable with station dwell")
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("Physical mileage (km)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Optimized departures for 6 up and 6 down trains")
    axes[1].set_xlabel("Departure time (min)")
    axes[1].set_ylabel("Train index")
    axes[1].set_yticks(range(1, len(train_configs) + 1))
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "bidirectional_timetable_optimization.png", dpi=180)
    plt.close(fig)


def export_bidirectional_operation_trajectories(
    up_profile: LineProfile,
    down_profile: LineProfile,
    environments_by_train: Sequence[Sequence[LowerSectionEnvironment]],
    lower_agents: Sequence[DQNAgent],
    train_configs: Sequence[TrainConfig],
    direction_labels: Sequence[str],
    timetables_s: Sequence[Sequence[float]],
    departure_times_s: Sequence[float],
    station_dwell_s: float,
    output_dir: Path,
) -> None:
    summary_rows: list[dict[str, Any]] = []
    direction_counts = {"up": 0, "down": 0}
    for train_idx, timetable in enumerate(timetables_s):
        profile = up_profile if direction_labels[train_idx] == "up" else down_profile
        direction = direction_labels[train_idx]
        direction_counts[direction] += 1
        direction_train_idx = direction_counts[direction]
        results = evaluate_timetable(
            environments_by_train[train_idx],
            timetable,
            lower_agents[train_idx],
        )
        for row in lower_result_summary_rows(results, train_configs[train_idx].name):
            row["direction"] = direction
            summary_rows.append(row)
        stem = f"{direction}_train_{direction_train_idx:02d}_operation"
        plot_train_operation(profile, results, output_dir, stem)
        plot_section_operation_overview(
            profile,
            results,
            output_dir,
            f"{direction}_train_{direction_train_idx:02d}",
        )
        plot_time_speed_with_station_dwell(
            results,
            float(departure_times_s[train_idx]),
            station_dwell_s,
            output_dir,
            f"{direction}_train_{direction_train_idx:02d}",
        )
    write_csv(
        output_dir / "bidirectional_train_operation_summary.csv",
        summary_rows,
    )


def plot_time_speed_with_station_dwell(
    results,
    departure_time_s: float,
    station_dwell_s: float,
    output_dir: Path,
    stem: str,
) -> None:
    rows: list[dict[str, Any]] = []
    dwell_intervals: list[tuple[float, float]] = []
    section_start_s = float(departure_time_s)
    for section_idx, result in enumerate(results):
        for point in result.trajectory:
            rows.append(
                {
                    "section_name": result.section_name,
                    "global_time_s": section_start_s + point.elapsed_s,
                    "global_time_min": (section_start_s + point.elapsed_s) / 60.0,
                    "speed_kmh": point.speed_kmh,
                    "regime": point.regime,
                }
            )
        arrival_s = section_start_s + result.actual_time_s
        if section_idx < len(results) - 1:
            dwell_end_s = arrival_s + station_dwell_s
            dwell_intervals.append((arrival_s, dwell_end_s))
            rows.extend(
                [
                    {
                        "section_name": result.section_name,
                        "global_time_s": arrival_s,
                        "global_time_min": arrival_s / 60.0,
                        "speed_kmh": 0.0,
                        "regime": "station_dwell",
                    },
                    {
                        "section_name": result.section_name,
                        "global_time_s": dwell_end_s,
                        "global_time_min": dwell_end_s / 60.0,
                        "speed_kmh": 0.0,
                        "regime": "station_dwell",
                    },
                ]
            )
            section_start_s = dwell_end_s
        else:
            section_start_s = arrival_s
    rows.sort(key=lambda row: row["global_time_s"])
    write_csv(output_dir / f"{stem}_time_speed_with_dwell.csv", rows)

    fig, axis = plt.subplots(figsize=(16, 5))
    axis.plot(
        [row["global_time_min"] for row in rows],
        [row["speed_kmh"] for row in rows],
        color="#1565C0",
        linewidth=0.9,
        label="Train speed",
    )
    for interval_idx, (start_s, end_s) in enumerate(dwell_intervals):
        axis.axvspan(
            start_s / 60.0,
            end_s / 60.0,
            color="#F9A825",
            alpha=0.20,
            label="60 s station dwell" if interval_idx == 0 else None,
        )
    axis.set_title(f"{stem.replace('_', ' ').title()}: time-speed trajectory")
    axis.set_xlabel("Global time (min)")
    axis.set_ylabel("Speed (km/h)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_time_speed_with_dwell.png", dpi=180)
    plt.close(fig)


def write_run_metadata(
    args: argparse.Namespace,
    train_configs: Sequence[TrainConfig],
    initial_timetables: np.ndarray,
    direction_labels: Sequence[str],
    down_profile: LineProfile,
) -> None:
    metadata = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "arguments": vars(args),
        "direction_labels": list(direction_labels),
        "initial_timetables_s": initial_timetables,
        "train_configs": [asdict(config) for config in train_configs],
        "driving_smoothness": asdict(DRIVING_SMOOTHNESS),
        "lower_reward": asdict(LOWER_REWARD),
        "supply_zone_names": REAL_SUPPLY_ZONE_NAMES,
        "supply_zone_section_groups": REAL_SUPPLY_ZONE_SECTION_GROUPS,
        "down_direction_assumptions": {
            "section_order": "reverse of up direction",
            "gradient": (
                f"read from {args.down_slope}; mileage follows up physical "
                "axis and gradient sign is reversed for down travel direction"
            ),
            "curve_radius": f"read from {args.down_curvature}",
            "speed_limit": (
                "read from down curvature file and inherited in travel direction"
            ),
        },
        "down_curvature_corrections": list(
            getattr(down_profile, "source_corrections", ())
        ),
    }
    with (args.output_dir / "run_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(json_ready(metadata), handle, ensure_ascii=False, indent=2)


def restore_best_environment_state(
    environment: MultiTrainTimetableEnvironment,
    result,
) -> None:
    environment.timetables = np.asarray(result.timetables_s, dtype=np.float32)
    environment.departure_offsets_s = (
        np.asarray(result.departure_times_s, dtype=np.float32)
        - environment.base_departures_s
    )
    environment.points = environment._lookup_points()
    environment.metrics = environment._calculate_metrics()
    environment.objective = environment.metrics["objective"]


def load_lower_policies_from_checkpoint(
    checkpoint_path: Path,
    lower_agents: Sequence[DQNAgent],
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Lower-policy checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policies = checkpoint.get("lower_policies")
    if policies is None and checkpoint.get("lower_controller_states") is not None:
        policies = [
            item["policy"]
            for item in checkpoint["lower_controller_states"]
        ]
    if policies is None:
        raise ValueError(
            f"Checkpoint does not contain lower_policies: {checkpoint_path}"
        )
    if len(policies) != len(lower_agents):
        raise ValueError(
            "Lower-policy count does not match current train count: "
            f"checkpoint has {len(policies)}, current experiment needs "
            f"{len(lower_agents)}"
        )
    adapted_count = 0
    for train_idx, (agent, state_dict) in enumerate(zip(lower_agents, policies)):
        if isinstance(state_dict, dict) and "policy" in state_dict:
            state_dict = state_dict["policy"]
        try:
            agent.policy.load_state_dict(state_dict)
            if hasattr(agent, "target_actor"):
                agent.target_actor.load_state_dict(state_dict)
        except RuntimeError as exc:
            current_state = agent.policy.state_dict()
            saved_input = state_dict.get("layers.0.weight")
            current_input = current_state.get("layers.0.weight")
            can_adapt_input = (
                saved_input is not None
                and current_input is not None
                and saved_input.ndim == current_input.ndim == 2
                and saved_input.shape[0] == current_input.shape[0]
                and saved_input.shape[1] < current_input.shape[1]
            )
            if not can_adapt_input:
                raise RuntimeError(
                    "Failed to load lower DQN policy for train "
                    f"{train_idx + 1}. The checkpoint may come from an "
                    "incompatible state dimension or action-space version."
                ) from exc
            adapted_state = dict(state_dict)
            adapted_input = current_input.clone()
            adapted_input.zero_()
            adapted_input[:, : saved_input.shape[1]] = saved_input
            adapted_state["layers.0.weight"] = adapted_input
            try:
                agent.policy.load_state_dict(adapted_state)
            except RuntimeError as adapted_exc:
                raise RuntimeError(
                    "Failed to adapt lower DQN policy for train "
                    f"{train_idx + 1}. Only a smaller first-layer input "
                    "dimension can be migrated automatically."
                ) from adapted_exc
            adapted_count += 1
        if hasattr(agent, "target"):
            agent.target.load_state_dict(agent.policy.state_dict())
            agent.target.eval()
        if hasattr(agent, "target_actor"):
            agent.target_actor.load_state_dict(agent.policy.state_dict())
            agent.target_actor.eval()
    print(f"Loaded lower DQN policies from: {checkpoint_path.resolve()}")
    if adapted_count:
        print(
            "Adapted lower DQN input layer for "
            f"{adapted_count} trains; new state-feature weights were initialized "
            "to zero."
        )


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.response_span <= 0.5:
        raise ValueError("--response-span must be within (0, 0.5] seconds")
    if not 0.0 < args.response_step <= args.response_span:
        raise ValueError("--response-step must be positive and no larger than response span")
    if args.lower_policy_mode != "fresh" and args.lower_policy_checkpoint is None:
        raise ValueError(
            "--lower-policy-checkpoint is required when --lower-policy-mode is "
            "'continue' or 'reuse'"
        )
    configure_run_outputs(args)
    seed_everything(args.seed)
    (
        up_profile,
        down_profile,
        train_configs,
        environments_by_train,
        initial_timetables,
        base_departures,
        direction_groups,
        physical_sections,
        direction_labels,
    ) = build_bidirectional_model(args)
    write_run_metadata(
        args,
        train_configs,
        initial_timetables,
        direction_labels,
        down_profile,
    )
    write_csv(
        args.output_dir / "down_line_profile.csv",
        down_line_profile_rows(up_profile, down_profile),
    )

    print(
        f"Bidirectional experiment: {args.up_trains} up trains + "
        f"{args.down_trains} down trains"
    )
    print("Down profile uses reversed section order in the down travel direction.")
    print(
        "Down slope loaded from physical-mileage workbook and sign-reversed: "
        f"{args.down_slope}"
    )
    print(
        f"Down curvature and speed limits loaded from: {args.down_curvature}"
    )
    for correction in getattr(down_profile, "source_corrections", ()):
        print(f"  corrected down mileage: {correction}")
    print(
        "Regenerative matching combines both directions inside the same physical "
        "power-supply zone."
    )

    lower_agents: list[DQNAgent] = []
    lower_history: list[dict[str, Any]] = []
    response_tables_by_train = []
    for _config in train_configs:
        lower_agent = DQNAgent(
            state_dim=LowerSectionEnvironment.STATE_DIM,
            action_dim=LowerSectionEnvironment.ACTION_DIM,
        )
        lower_agents.append(lower_agent)

    if args.lower_policy_mode in {"continue", "reuse"}:
        load_lower_policies_from_checkpoint(
            args.lower_policy_checkpoint,
            lower_agents,
        )

    for train_idx, (config, environments) in enumerate(
        zip(train_configs, environments_by_train)
    ):
        train_timetable = initial_timetables[train_idx]
        lower_agent = lower_agents[train_idx]
        if args.lower_policy_mode == "reuse":
            print(
                f"Reusing lower DQN policy for {config.name}; "
                "lower training skipped."
            )
        elif args.lower_episodes > 0:
            lower_history.extend(
                train_lower_agents(
                    environments,
                    lower_agent,
                    train_timetable,
                    episodes=args.lower_episodes,
                    target_span_s=args.lower_target_span,
                    train_name=config.name,
                )
            )
        else:
            print(f"Lower training skipped for {config.name}: lower_episodes=0")
        response_tables_by_train.append(
            build_response_tables(
                environments,
                train_timetable,
                time_span_s=args.response_span,
                time_step_s=args.response_step,
                learned_controller=lower_agent,
                train_name=config.name,
            )
        )

    total_trains = args.up_trains + args.down_trains
    upper_environment = MultiTrainTimetableEnvironment(
        initial_timetable=np.asarray(INITIAL_TIMETABLE, dtype=np.float32),
        initial_timetables_by_train=initial_timetables,
        response_tables_by_train=response_tables_by_train,
        train_count=total_trains,
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
        base_departures_s=base_departures,
        direction_groups=direction_groups,
        physical_section_names_by_train=physical_sections,
    )
    upper_agent = IndependentActorMAPPOAgent(
        train_count=total_trains,
        observation_dim=upper_environment.local_observation_dim,
        action_dim=upper_environment.action_dim,
        central_state_dim=upper_environment.central_state_dim,
        active_agent_mask=upper_environment.active_agent_mask,
    )
    result, upper_history = train_multi_train_mappo(
        upper_environment,
        upper_agent,
        episodes=args.upper_episodes,
    )
    save_multi_train_checkpoint(
        args.checkpoint,
        lower_agents,
        upper_agent,
        result,
        [asdict(config) for config in train_configs],
        asdict(DRIVING_SMOOTHNESS),
    )
    restore_best_environment_state(upper_environment, result)

    write_csv(
        args.output_dir / "bidirectional_station_schedule.csv",
        timetable_rows(
            upper_environment,
            train_configs,
            direction_labels,
            up_profile,
            down_profile,
        ),
    )
    write_csv(
        args.output_dir / "train_parameters.csv",
        [
            {
                **asdict(config),
                "direction": direction_labels[idx],
            }
            for idx, config in enumerate(train_configs)
        ],
    )
    plot_bidirectional_timetable(
        upper_environment,
        train_configs,
        direction_labels,
        up_profile,
        down_profile,
        args.output_dir,
    )
    export_bidirectional_operation_trajectories(
        up_profile,
        down_profile,
        environments_by_train,
        lower_agents,
        train_configs,
        direction_labels,
        result.timetables_s,
        result.departure_times_s,
        args.station_dwell_s,
        args.output_dir,
    )
    plot_training_process(lower_history, upper_history, args.output_dir)
    plot_response_tables(response_tables_by_train, args.output_dir)

    summary = {
        "up_train_count": args.up_trains,
        "down_train_count": args.down_trains,
        "station_dwell_s": args.station_dwell_s,
        "initial_headway_s_each_direction": args.initial_headway,
        "minimum_headway_s": args.minimum_headway,
        "grid_energy_kwh": result.grid_energy_kwh,
        "traction_kwh": result.traction_kwh,
        "regenerative_kwh": result.regenerative_kwh,
        "matched_regenerative_kwh": result.matched_regenerative_kwh,
        "regenerative_utilization": result.regenerative_utilization,
        "normalized_weighted_objective": result.objective,
        "traction_energy_weight": args.traction_energy_weight,
        "regenerative_utilization_weight": args.regenerative_match_weight,
        "running_time_adjustment_bound_s": args.response_span,
        "running_time_adjustment_step_s": args.response_step,
    }
    write_csv(args.output_dir / "bidirectional_experiment_summary.csv", [summary])
    print(f"Saved bidirectional experiment results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
