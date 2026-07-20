"""CSV and figure exporters for train control and MAPPO coordination."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

try:
    from .config import EPS, INITIAL_TIMETABLE, SECTION_NAMES
    from .line_profile import LineProfile
    from .lower_control import DRIVING_SMOOTHNESS, DQNAgent, LowerResult, SectionResponseTable, evaluate_timetable
    from .mappo import MultiTrainTimetableEnvironment, MultiTrainTrainingResult
except ImportError:
    from config import EPS, INITIAL_TIMETABLE, SECTION_NAMES
    from line_profile import LineProfile
    from lower_control import DRIVING_SMOOTHNESS, DQNAgent, LowerResult, SectionResponseTable, evaluate_timetable
    from mappo import MultiTrainTimetableEnvironment, MultiTrainTrainingResult


plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def write_csv(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: Sequence[float], window: int = 20) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return array
    if np.any(~np.isfinite(array)):
        finite = np.isfinite(array)
        if not np.any(finite):
            return np.full(len(array), np.nan)
        indices = np.arange(len(array))
        array = np.interp(indices, indices[finite], array[finite])
    width = min(window, len(array))
    averaged = np.convolve(array, np.ones(width) / width, mode="valid")
    return np.concatenate((np.full(width - 1, averaged[0]), averaged))


def history_series(
    rows: Sequence[Dict[str, Any]],
    key: str,
    default: float = np.nan,
    transform=None,
) -> List[float]:
    values: List[float] = []
    for item in rows:
        value = item.get(key, default)
        if value in ("", None):
            value = default
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        values.append(transform(numeric) if transform is not None else numeric)
    return values


def combined_trajectory_rows(
    results: Sequence[LowerResult],
    profile: Optional[LineProfile] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    time_offset = traction_offset = regenerative_offset = 0.0
    for result in results:
        section = None if profile is None else profile.section_by_name.get(result.section_name)
        for point in result.trajectory:
            physical_position_m = (
                point.position_m
                if section is None
                else section.physical_position_m(point.position_m)
            )
            rows.append(
                {
                    "section_name": point.section_name,
                    "global_time_s": time_offset + point.elapsed_s,
                    "section_time_s": point.elapsed_s,
                    "position_m": point.position_m,
                    "physical_position_m": physical_position_m,
                    "speed_kmh": point.speed_kmh,
                    "speed_limit_kmh": point.speed_limit_kmh,
                    "force_kn": point.force_kn,
                    "resistance_kn": point.resistance_kn,
                    "gradient_per_mille": point.gradient_per_mille,
                    "radius_m": point.radius_m,
                    "requested_action": point.requested_action,
                    "applied_action": point.applied_action,
                    "regime": point.regime,
                    "traction_kwh": traction_offset + point.traction_kwh,
                    "regenerative_kwh": regenerative_offset + point.regenerative_kwh,
                }
            )
        time_offset += result.actual_time_s
        traction_offset += result.traction_kwh
        regenerative_offset += result.regenerative_kwh
    return rows


def lower_result_summary_rows(
    results: Sequence[LowerResult],
    train_name: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "train_name": train_name,
            "section_name": result.section_name,
            "target_time_s": result.target_time_s,
            "actual_time_s": result.actual_time_s,
            "timetable_error_s": result.timetable_error_s,
            "net_energy_kwh": result.net_energy_kwh,
            "feasible": result.feasible,
            "strict_timetable_feasible": result.strict_timetable_feasible,
            "relaxed_timetable_feasible": result.relaxed_timetable_feasible,
            "accepted_arrival_tolerance_s": result.accepted_arrival_tolerance_s,
            "regime_switches": result.regime_switches,
            "traction_coast_transitions": result.traction_coast_transitions,
            "traction_brake_transitions": result.traction_brake_transitions,
            "short_dwell_transitions": result.short_dwell_transitions,
            "force_jump_index": result.force_jump_index,
            "smoothness_penalty": result.smoothness_penalty,
            "jerk_integral_mps3_s": result.jerk_integral_mps3_s,
            "comfort_penalty": result.comfort_penalty,
            "traction_reward_component": result.traction_reward_component,
            "punctuality_reward_component": result.punctuality_reward_component,
            "comfort_reward_component": result.comfort_reward_component,
            "max_speed_limit_violation_kmh": result.max_speed_limit_violation_kmh,
            "speed_regulation_steps": result.speed_regulation_steps,
            "expert_interventions": getattr(result, "expert_interventions", 0),
            "episode_reward": result.episode_reward,
        }
        for result in results
    ]


def station_ticks(profile: LineProfile, use_physical_axis: bool = True) -> Tuple[List[float], List[str]]:
    first_position_m = (
        profile.sections[0].display_start_m
        if use_physical_axis
        else profile.sections[0].start_m
    )
    positions = [first_position_m / 1000.0]
    labels = [profile.sections[0].name.split("-")[0]]
    for section in profile.sections:
        end_m = section.display_end_m if use_physical_axis else section.end_m
        positions.append(end_m / 1000.0)
        labels.append(section.name.split("-")[-1])
    return positions, labels


def speed_limit_steps(
    profile: LineProfile,
    use_physical_axis: bool = True,
) -> Tuple[List[float], List[float]]:
    positions: List[float] = []
    limits: List[float] = []
    for section in profile.sections:
        for segment in section.segments:
            if use_physical_axis:
                start_m = segment.start_m if segment.physical_start_m is None else segment.physical_start_m
                end_m = segment.end_m if segment.physical_end_m is None else segment.physical_end_m
            else:
                start_m = segment.start_m
                end_m = segment.end_m
            positions.extend((start_m / 1000.0, end_m / 1000.0))
            limits.extend((segment.speed_limit_kmh, segment.speed_limit_kmh))
    return positions, limits


UP_ELEVATION_AXIS_LIMITS_M = (-50.0, 250.0)
DOWN_ELEVATION_AXIS_LIMITS_M = (-250.0, 50.0)


def is_down_direction_profile(profile: LineProfile) -> bool:
    if not profile.sections:
        return False
    return profile.sections[0].name != SECTION_NAMES[0]


def elevation_axis_limits(profile: LineProfile) -> Tuple[float, float]:
    return (
        DOWN_ELEVATION_AXIS_LIMITS_M
        if is_down_direction_profile(profile)
        else UP_ELEVATION_AXIS_LIMITS_M
    )


def elevation_profile(
    profile: LineProfile,
    use_physical_axis: bool = True,
) -> Tuple[List[float], List[float]]:
    positions: List[float] = []
    elevations: List[float] = []
    current_elevation_m = 0.0
    for section in profile.sections:
        for segment in section.segments:
            if use_physical_axis:
                start_m = segment.start_m if segment.physical_start_m is None else segment.physical_start_m
                end_m = segment.end_m if segment.physical_end_m is None else segment.physical_end_m
            else:
                start_m = segment.start_m
                end_m = segment.end_m
            if not positions:
                positions.append(start_m / 1000.0)
                elevations.append(current_elevation_m)
            elif not np.isclose(positions[-1], start_m / 1000.0):
                positions.append(start_m / 1000.0)
                elevations.append(current_elevation_m)
            current_elevation_m += (
                segment.gradient_per_mille / 1000.0 * (segment.end_m - segment.start_m)
            )
            positions.append(end_m / 1000.0)
            elevations.append(current_elevation_m)
    return positions, elevations


def elevation_at_positions(profile: LineProfile, positions_m: Sequence[float]) -> np.ndarray:
    elevation_positions_km, elevations_m = elevation_profile(profile)
    query_km = np.asarray(positions_m, dtype=float) / 1000.0
    profile_positions = np.asarray(elevation_positions_km, dtype=float)
    profile_elevations = np.asarray(elevations_m, dtype=float)
    order = np.argsort(profile_positions)
    return np.interp(query_km, profile_positions[order], profile_elevations[order])


def result_physical_positions(profile: LineProfile, result: LowerResult) -> np.ndarray:
    section = profile.section_by_name.get(result.section_name)
    if section is None:
        return np.asarray([item.position_m for item in result.trajectory], dtype=float)
    return np.asarray(
        [section.physical_position_m(item.position_m) for item in result.trajectory],
        dtype=float,
    )


def result_axis_positions(
    profile: LineProfile,
    result: LowerResult,
    use_physical_axis: bool = True,
) -> np.ndarray:
    if use_physical_axis:
        return result_physical_positions(profile, result)
    return np.asarray([item.position_m for item in result.trajectory], dtype=float)


def set_elevation_axis_style(axis, profile: LineProfile) -> None:
    axis.tick_params(axis="y", colors="#757575", labelsize=8)
    axis.spines["right"].set_color("#9E9E9E")
    axis.yaxis.label.set_color("#757575")
    axis.set_ylim(*elevation_axis_limits(profile))


def plot_train_operation(
    profile: LineProfile,
    results: Sequence[LowerResult],
    output_dir: Path,
    stem: str,
) -> None:
    rows = combined_trajectory_rows(results, profile)
    write_csv(output_dir / f"{stem}.csv", rows)
    use_physical_axis = not is_down_direction_profile(profile)
    limit_positions, limits = speed_limit_steps(profile, use_physical_axis)
    elevation_positions, elevations = elevation_profile(profile, use_physical_axis)
    ticks, labels = station_ticks(profile, use_physical_axis)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    speed_line = []
    force_line = []
    resistance_line = []
    for result_idx, result in enumerate(results):
        positions = result_axis_positions(profile, result, use_physical_axis) / 1000.0
        speeds = np.asarray([item.speed_kmh for item in result.trajectory])
        forces = np.asarray([item.force_kn for item in result.trajectory])
        resistances = np.asarray([item.resistance_kn for item in result.trajectory])
        control_positions = positions
        control_forces = forces
        control_resistances = resistances
        if len(positions) > 1 and result.trajectory[0].elapsed_s <= 1e-9:
            control_positions = positions[1:]
            control_forces = forces[1:]
            control_resistances = resistances[1:]
        speed_line.extend(
            axes[0].plot(
                positions,
                speeds,
                color="#1565C0",
                linewidth=1.0,
                label="Train speed" if result_idx == 0 else None,
            )
        )
        force_line.extend(
            axes[1].plot(
                control_positions,
                control_forces,
                color="#2E7D32",
                linewidth=0.8,
                label="Control force" if result_idx == 0 else None,
            )
        )
        resistance_line.extend(
            axes[1].plot(
                control_positions,
                control_resistances,
                color="#6A1B9A",
                linewidth=0.7,
                label="Resistance" if result_idx == 0 else None,
            )
        )
        axes[1].fill_between(
            control_positions,
            control_forces,
            0.0,
            where=control_forces >= 0.0,
            color="#81C784",
            alpha=0.25,
        )
        axes[1].fill_between(
            control_positions,
            control_forces,
            0.0,
            where=control_forces < 0.0,
            color="#EF9A9A",
            alpha=0.30,
        )
    limit_line = axes[0].plot(limit_positions, limits, color="#C62828", linestyle="--", linewidth=0.8, label="Speed limit")
    elevation_axis = axes[0].twinx()
    elevation_line = elevation_axis.plot(
        elevation_positions,
        elevations,
        color="#757575",
        linewidth=1.0,
        alpha=0.70,
        label="Line elevation",
    )
    elevation_axis.axhline(0.0, color="#BDBDBD", linewidth=0.5, alpha=0.65)
    elevation_axis.set_ylabel("Relative elevation (m)")
    set_elevation_axis_style(elevation_axis, profile)
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].set_title(f"{stem.replace('_', ' ').title()}: speed trajectory")
    legend_handles = []
    if speed_line:
        legend_handles.append(speed_line[0])
    legend_handles.extend(limit_line + elevation_line)
    axes[0].legend(
        legend_handles,
        [item.get_label() for item in legend_handles],
        loc="upper right",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Force (kN)")
    axes[1].set_xlabel(
        "Physical mileage (km)"
        if use_physical_axis
        else "Route mileage along travel direction (km)"
    )
    axes[1].legend(handles=[force_line[0], resistance_line[0]] if force_line and resistance_line else None)
    for axis in axes:
        axis.grid(alpha=0.25)
        for position in ticks:
            axis.axvline(position, color="gray", linewidth=0.35, alpha=0.30)
    axes[1].set_xticks(ticks, labels, rotation=55, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=180)
    plt.close(fig)


def plot_section_operation_overview(
    profile: LineProfile,
    results: Sequence[LowerResult],
    output_dir: Path,
    stem: str,
) -> None:
    fig_speed, speed_axes = plt.subplots(5, 4, figsize=(16, 15))
    fig_force, force_axes = plt.subplots(5, 4, figsize=(16, 15))
    reference_rows: List[Dict[str, Any]] = []
    for panel_idx, (result, speed_axis, force_axis) in enumerate(
        zip(results, speed_axes.flat, force_axes.flat),
        start=1,
    ):
        section = profile.section_by_name.get(result.section_name)
        start_m = section.start_m if section is not None else result.trajectory[0].position_m
        end_m = section.end_m if section is not None else result.trajectory[-1].position_m
        physical_start_m = start_m if section is None else section.display_start_m
        physical_end_m = end_m if section is None else section.display_end_m
        section_length_km = max((end_m - start_m) / 1000.0, EPS)
        positions = np.asarray([(item.position_m - start_m) / 1000.0 for item in result.trajectory])
        speeds = np.asarray([item.speed_kmh for item in result.trajectory])
        limits = np.asarray([item.speed_limit_kmh for item in result.trajectory])
        forces = np.asarray([item.force_kn for item in result.trajectory])
        elevations = elevation_at_positions(
            profile,
            result_physical_positions(profile, result),
        )
        speed_axis.plot(positions, speeds, color="#1565C0", linewidth=0.8)
        speed_axis.plot(positions, limits, color="#C62828", linestyle="--", linewidth=0.7)
        elevation_axis = speed_axis.twinx()
        elevation_axis.plot(positions, elevations, color="#757575", linewidth=0.65, alpha=0.55)
        elevation_axis.axhline(0.0, color="#BDBDBD", linewidth=0.35, alpha=0.50)
        elevation_axis.set_ylim(*elevation_axis_limits(profile))
        elevation_axis.set_yticks([])
        elevation_axis.spines["right"].set_color("#BDBDBD")
        speed_axis.set_xlim(0.0, section_length_km)
        speed_axis.set_title(
            f"{result.section_name} ({physical_start_m / 1000.0:.2f}-{physical_end_m / 1000.0:.2f} km)"
        )
        speed_axis.grid(alpha=0.25)
        force_axis.plot(positions, forces, color="#2E7D32", linewidth=0.8)
        force_axis.fill_between(positions, forces, 0.0, where=forces >= 0.0, color="#81C784", alpha=0.25)
        force_axis.fill_between(positions, forces, 0.0, where=forces < 0.0, color="#EF9A9A", alpha=0.30)
        force_axis.axhline(0.0, color="black", linewidth=0.5)
        force_axis.set_xlim(0.0, section_length_km)
        force_axis.set_title(
            f"{result.section_name} | switches={result.regime_switches}, "
            f"T/C={result.traction_coast_transitions}, short={result.short_dwell_transitions}"
        )
        force_axis.grid(alpha=0.25)
        reference_rows.append(
            {
                "panel_index": panel_idx,
                "section_name": result.section_name,
                "travel_start_km": start_m / 1000.0,
                "travel_end_km": end_m / 1000.0,
                "physical_start_km": physical_start_m / 1000.0,
                "physical_end_km": physical_end_m / 1000.0,
                "section_length_km": section_length_km,
                "first_trajectory_travel_km": result.trajectory[0].position_m / 1000.0,
                "last_trajectory_travel_km": result.trajectory[-1].position_m / 1000.0,
                "first_trajectory_physical_km": result_physical_positions(profile, result)[0] / 1000.0,
                "last_trajectory_physical_km": result_physical_positions(profile, result)[-1] / 1000.0,
                "first_speed_kmh": result.trajectory[0].speed_kmh,
                "last_speed_kmh": result.trajectory[-1].speed_kmh,
            }
        )
    for axis in speed_axes[-1]:
        axis.set_xlabel("Section mileage (km)")
    for axis in force_axes[-1]:
        axis.set_xlabel("Section mileage (km)")
    speed_axes.flat[-1].set_visible(False)
    force_axes.flat[-1].set_visible(False)
    fig_speed.suptitle("Interstation speed trajectories", fontsize=15)
    fig_force.suptitle("Interstation traction and braking force curves", fontsize=15)
    fig_speed.tight_layout(rect=(0, 0, 1, 0.98))
    fig_force.tight_layout(rect=(0, 0, 1, 0.98))
    fig_speed.savefig(output_dir / f"{stem}_section_speed_trajectories.png", dpi=180)
    fig_force.savefig(output_dir / f"{stem}_section_force_curves.png", dpi=180)
    plt.close(fig_speed)
    plt.close(fig_force)
    write_csv(output_dir / f"{stem}_section_plot_reference.csv", reference_rows)


def station_schedule_times(
    timetables: Sequence[Sequence[float]],
    departure_times_s: Sequence[float],
    station_dwell_after_section_s: Optional[Sequence[Sequence[float]]] = None,
    station_delay_after_section_s: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    timetables_array = np.asarray(timetables, dtype=float)
    schedules = np.zeros((len(timetables_array), timetables_array.shape[1] + 1), dtype=float)
    station_delays = np.zeros_like(timetables_array)
    if station_dwell_after_section_s is not None:
        station_delays += np.asarray(station_dwell_after_section_s, dtype=float)
    if station_delay_after_section_s is not None:
        station_delays += np.asarray(station_delay_after_section_s, dtype=float)
    if station_delays.shape != timetables_array.shape:
        raise ValueError(
            "station dwell/delay arrays must have the same shape as timetables"
        )
    if station_dwell_after_section_s is not None or station_delay_after_section_s is not None:
        if np.asarray(station_delays).shape != timetables_array.shape:
            raise ValueError(
                "station dwell/delay arrays must have the same shape as timetables"
            )
    for train_idx, departure_s in enumerate(departure_times_s):
        current_s = float(departure_s)
        schedules[train_idx, 0] = current_s
        for section_idx in range(timetables_array.shape[1]):
            current_s += float(timetables_array[train_idx, section_idx])
            schedules[train_idx, section_idx + 1] = current_s
            current_s += float(station_delays[train_idx, section_idx])
    return schedules


CONTROL_STATUS_COLORS = {
    "traction": "#2E7D32",
    "brake": "#C62828",
    "coast": "#757575",
    "dwell": "#F9A825",
}


def control_status(force_kn: float, regime: str) -> str:
    if force_kn > 1.0:
        return "traction"
    if force_kn < -1.0:
        return "brake"
    if regime == "station_dwell":
        return "dwell"
    return "coast"


def supply_zone_section_spans(
    profile: LineProfile,
    environment: MultiTrainTimetableEnvironment,
) -> List[Dict[str, Any]]:
    zone_to_sections: Dict[int, List[int]] = {}
    for section_idx in range(len(profile.sections)):
        zone_idx = environment.supply_zone_id(section_idx)
        if zone_idx < 0:
            continue
        zone_to_sections.setdefault(zone_idx, []).append(section_idx)
    spans = []
    for zone_idx, section_indices in sorted(zone_to_sections.items()):
        first_idx = min(section_indices)
        last_idx = max(section_indices)
        spans.append(
            {
                "zone_index": zone_idx + 1,
                "zone_name": environment.supply_zone_name(zone_idx),
                "start_m": profile.sections[first_idx].start_m,
                "end_m": profile.sections[last_idx].end_m,
                "section_indices": section_indices,
            }
        )
    return spans


def multi_train_operation_synchronization_rows(
    results_by_train: Sequence[Sequence[LowerResult]],
    train_configs: Sequence[Any],
    environment: MultiTrainTimetableEnvironment,
    result: MultiTrainTrainingResult,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    station_delays = getattr(environment, "station_delay_after_section_s", None)
    station_dwell = getattr(environment, "base_station_dwell_after_section_s", None)
    if station_delays is None:
        station_delays_array = np.zeros_like(np.asarray(result.timetables_s, dtype=float))
    else:
        station_delays_array = np.asarray(station_delays, dtype=float)
    if station_dwell is None:
        station_dwell_array = np.zeros_like(np.asarray(result.timetables_s, dtype=float))
    else:
        station_dwell_array = np.asarray(station_dwell, dtype=float)
    section_index_by_name = {name: idx for idx, name in enumerate(SECTION_NAMES)}
    for train_idx, train_results in enumerate(results_by_train):
        train_name = getattr(train_configs[train_idx], "name", f"Train {train_idx + 1}")
        section_start_s = float(result.departure_times_s[train_idx])
        for fallback_section_idx, lower_result in enumerate(train_results):
            section_idx = section_index_by_name.get(lower_result.section_name, fallback_section_idx)
            zone_idx = environment.supply_zone_id(section_idx)
            zone_name = environment.supply_zone_name(zone_idx) if zone_idx >= 0 else "unassigned"
            for point in lower_result.trajectory:
                status = control_status(point.force_kn, point.regime)
                rows.append(
                    {
                        "train_id": train_idx + 1,
                        "train_name": train_name,
                        "section_index": section_idx + 1,
                        "section_name": lower_result.section_name,
                        "supply_zone_index": zone_idx + 1 if zone_idx >= 0 else "",
                        "supply_zone": zone_name,
                        "global_time_s": section_start_s + point.elapsed_s,
                        "global_time_min": (section_start_s + point.elapsed_s) / 60.0,
                        "position_m": point.position_m,
                        "position_km": point.position_m / 1000.0,
                        "speed_kmh": point.speed_kmh,
                        "force_kn": point.force_kn,
                        "regime": point.regime,
                        "control_status": status,
                    }
                )
            section_start_s += lower_result.actual_time_s
            if section_idx < station_delays_array.shape[1]:
                section_start_s += float(
                    station_dwell_array[train_idx, section_idx]
                    + station_delays_array[train_idx, section_idx]
                )
    rows.sort(key=lambda item: (item["train_id"], item["global_time_s"]))
    return rows


def train_state_segments(train_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if len(train_rows) < 2:
        return segments
    for current, following in zip(train_rows[:-1], train_rows[1:]):
        start_s = float(current["global_time_s"])
        end_s = float(following["global_time_s"])
        if end_s <= start_s:
            continue
        status = str(current["control_status"])
        if (
            end_s - start_s > 2.0
            and abs(float(following["position_m"]) - float(current["position_m"])) < 5.0
        ):
            status = "dwell"
        segments.append(
            {
                "train_id": current["train_id"],
                "start_s": start_s,
                "end_s": end_s,
                "start_min": start_s / 60.0,
                "duration_min": (end_s - start_s) / 60.0,
                "start_position_km": current["position_km"],
                "end_position_km": following["position_km"],
                "supply_zone_index": current["supply_zone_index"],
                "supply_zone": current["supply_zone"],
                "control_status": status,
            }
        )
    return segments


def plot_multi_train_operation_synchronization(
    profile: LineProfile,
    results_by_train: Sequence[Sequence[LowerResult]],
    train_configs: Sequence[Any],
    environment: MultiTrainTimetableEnvironment,
    result: MultiTrainTrainingResult,
    output_dir: Path,
) -> None:
    rows = multi_train_operation_synchronization_rows(
        results_by_train,
        train_configs,
        environment,
        result,
    )
    write_csv(output_dir / "multi_train_operation_synchronization.csv", rows)
    if not rows:
        return

    zone_spans = supply_zone_section_spans(profile, environment)
    train_count = len(results_by_train)
    train_colors = plt.cm.tab10(np.linspace(0, 1, train_count))
    time_values_min = np.asarray([float(row["global_time_min"]) for row in rows])
    x_min = max(0.0, float(np.min(time_values_min)) - 0.5)
    x_max = float(np.max(time_values_min)) + 0.5
    ticks, station_labels = station_ticks(profile)
    status_handles = [
        Line2D([0], [0], color=color, linewidth=3.0, label=label)
        for label, color in (
            ("Traction", CONTROL_STATUS_COLORS["traction"]),
            ("Brake / regenerative", CONTROL_STATUS_COLORS["brake"]),
            ("Coast / zero force", CONTROL_STATUS_COLORS["coast"]),
            ("Station dwell", CONTROL_STATUS_COLORS["dwell"]),
        )
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2, 1.5]},
    )
    position_axis, state_axis, zone_axis = axes

    for span_idx, span in enumerate(zone_spans):
        color = "#F5F5F5" if span_idx % 2 == 0 else "#ECEFF1"
        position_axis.axhspan(
            span["start_m"] / 1000.0,
            span["end_m"] / 1000.0,
            color=color,
            zorder=0,
        )
        position_axis.text(
            x_min + 0.05,
            (span["start_m"] + span["end_m"]) / 2000.0,
            span["zone_name"],
            color="#616161",
            fontsize=8,
            va="center",
        )

    all_segments: List[Dict[str, Any]] = []
    for train_idx in range(train_count):
        train_rows = [row for row in rows if int(row["train_id"]) == train_idx + 1]
        if not train_rows:
            continue
        segments = train_state_segments(train_rows)
        all_segments.extend(segments)
        for segment in segments:
            position_axis.plot(
                [
                    segment["start_min"],
                    segment["start_min"] + segment["duration_min"],
                ],
                [
                    segment["start_position_km"],
                    segment["end_position_km"],
                ],
                color=CONTROL_STATUS_COLORS[str(segment["control_status"])],
                linewidth=1.15,
                alpha=0.90,
            )
            state_axis.broken_barh(
                [(segment["start_min"], segment["duration_min"])],
                (train_idx + 0.15, 0.70),
                facecolors=CONTROL_STATUS_COLORS[str(segment["control_status"])],
                edgecolors="none",
                alpha=0.90,
            )
        position_axis.scatter(
            [float(train_rows[0]["global_time_min"])],
            [float(train_rows[0]["position_km"])],
            color=train_colors[train_idx],
            marker="o",
            s=28,
            zorder=5,
        )
        position_axis.text(
            float(train_rows[0]["global_time_min"]) + 0.04,
            float(train_rows[0]["position_km"]) + 0.10,
            f"T{train_idx + 1}",
            color=train_colors[train_idx],
            fontsize=9,
            weight="bold",
        )

    position_axis.set_title("Multi-train time-position synchronization with traction/braking states")
    position_axis.set_ylabel("Absolute mileage (km)")
    position_axis.set_yticks(ticks, station_labels)
    position_axis.grid(alpha=0.22)
    position_axis.legend(handles=status_handles, loc="lower right", ncol=4, fontsize=8)

    state_axis.set_title("Per-train control-state timeline")
    state_axis.set_yticks(
        np.arange(train_count) + 0.5,
        [f"Train {idx + 1}" for idx in range(train_count)],
    )
    state_axis.set_ylim(0.0, train_count)
    state_axis.grid(axis="x", alpha=0.22)

    bin_s = float(getattr(environment, "power_bin_s", 5.0))
    horizon_s = max(float(row["global_time_s"]) for row in rows)
    bin_count = max(1, int(np.ceil(horizon_s / bin_s)) + 1)
    zone_count = max(1, int(environment.supply_zone_count))
    traction_flags = np.zeros((zone_count, bin_count, train_count), dtype=bool)
    brake_flags = np.zeros((zone_count, bin_count, train_count), dtype=bool)
    for segment in all_segments:
        zone_value = segment["supply_zone_index"]
        if zone_value == "" or str(segment["control_status"]) not in {"traction", "brake"}:
            continue
        zone_idx = int(zone_value) - 1
        train_idx = int(segment["train_id"]) - 1
        first_bin = max(0, int(np.floor(float(segment["start_s"]) / bin_s)))
        final_bin = min(bin_count, int(np.ceil(float(segment["end_s"]) / bin_s)))
        if str(segment["control_status"]) == "traction":
            traction_flags[zone_idx, first_bin:final_bin, train_idx] = True
        else:
            brake_flags[zone_idx, first_bin:final_bin, train_idx] = True
    traction_counts = np.sum(traction_flags, axis=2)
    brake_counts = np.sum(brake_flags, axis=2)
    zone_state = np.zeros((zone_count, bin_count), dtype=int)
    zone_state[(traction_counts > 0) & (brake_counts == 0)] = 1
    zone_state[(traction_counts == 0) & (brake_counts > 0)] = 2
    zone_state[(traction_counts > 0) & (brake_counts > 0)] = 3
    bin_rows: List[Dict[str, Any]] = []
    for zone_idx in range(zone_count):
        zone_name = environment.supply_zone_name(zone_idx)
        for bin_idx in range(bin_count):
            state_code = int(zone_state[zone_idx, bin_idx])
            bin_rows.append(
                {
                    "time_start_s": bin_idx * bin_s,
                    "time_end_s": (bin_idx + 1) * bin_s,
                    "supply_zone_index": zone_idx + 1,
                    "supply_zone": zone_name,
                    "traction_train_count": int(traction_counts[zone_idx, bin_idx]),
                    "braking_train_count": int(brake_counts[zone_idx, bin_idx]),
                    "has_same_zone_traction_and_braking": state_code == 3,
                }
            )
    write_csv(output_dir / "same_supply_zone_train_state_bins.csv", bin_rows)

    zone_cmap = ListedColormap(["#FFFFFF", "#A5D6A7", "#EF9A9A", "#7E57C2"])
    zone_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], zone_cmap.N)
    zone_axis.imshow(
        zone_state,
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, bin_count * bin_s / 60.0, zone_count, 0.0),
        cmap=zone_cmap,
        norm=zone_norm,
    )
    zone_axis.set_yticks(
        np.arange(zone_count) + 0.5,
        [environment.supply_zone_name(idx) for idx in range(zone_count)],
    )
    zone_axis.set_title("Same-zone traction/braking availability by time bin")
    zone_axis.set_xlabel("Time from first departure (min)")
    zone_axis.set_xlim(x_min, x_max)
    zone_axis.grid(axis="x", alpha=0.22)
    zone_axis.legend(
        handles=[
            Line2D([0], [0], color="#A5D6A7", linewidth=6, label="Traction only"),
            Line2D([0], [0], color="#EF9A9A", linewidth=6, label="Braking only"),
            Line2D([0], [0], color="#7E57C2", linewidth=6, label="Traction + braking"),
        ],
        loc="upper right",
        ncol=3,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "multi_train_operation_synchronization.png", dpi=180)
    plt.close(fig)


def plot_multi_train_timetable_result(
    environment: MultiTrainTimetableEnvironment,
    result: MultiTrainTrainingResult,
    output_dir: Path,
) -> None:
    initial = np.tile(INITIAL_TIMETABLE, (environment.train_count, 1))
    optimized = np.asarray(result.timetables_s)
    changes = optimized - initial
    raw_station_delays = getattr(environment, "station_delay_after_section_s", None)
    raw_station_dwell = getattr(environment, "base_station_dwell_after_section_s", None)
    station_delays = (
        np.zeros_like(optimized, dtype=float)
        if raw_station_delays is None
        else np.asarray(raw_station_delays, dtype=float)
    )
    station_dwell = (
        np.zeros_like(optimized, dtype=float)
        if raw_station_dwell is None
        else np.asarray(raw_station_dwell, dtype=float)
    )
    rows: List[Dict[str, Any]] = []
    for train_idx, timetable in enumerate(optimized):
        train_config = environment.response_tables_by_train[train_idx][0]
        for section_idx, section_name in enumerate(SECTION_NAMES):
            rows.append(
                {
                    "train_id": train_idx + 1,
                    "train_name": train_config.train_name,
                    "section_name": section_name,
                    "control_mode": environment.control_mode,
                    "initial_departure_time_s": environment.base_departures_s[train_idx],
                    "departure_time_s": result.departure_times_s[train_idx],
                    "departure_offset_s": result.departure_times_s[train_idx] - environment.base_departures_s[train_idx],
                    "initial_running_time_s": INITIAL_TIMETABLE[section_idx],
                    "optimized_running_time_s": timetable[section_idx],
                    "running_time_change_s": changes[train_idx, section_idx],
                    "scheduled_dwell_after_section_s": station_dwell[train_idx, section_idx],
                    "dwell_overrun_after_section_s": station_delays[train_idx, section_idx],
                    "total_dwell_after_section_s": station_dwell[train_idx, section_idx] + station_delays[train_idx, section_idx],
                }
            )
    write_csv(output_dir / "multi_train_timetable_optimization.csv", rows)

    initial_schedules = station_schedule_times(
        initial,
        environment.base_departures_s,
        station_dwell,
        station_delays,
    )
    optimized_schedules = station_schedule_times(
        optimized,
        result.departure_times_s,
        station_dwell,
        station_delays,
    )
    station_labels = [SECTION_NAMES[0].split("-")[0]] + [item.split("-")[-1] for item in SECTION_NAMES]
    station_axis = np.arange(len(station_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, environment.train_count))
    fig, axis = plt.subplots(figsize=(16, 9))
    train_handles = []
    for train_idx, color in enumerate(colors):
        initial_departure_min = environment.base_departures_s[train_idx] / 60.0
        optimized_departure_min = result.departure_times_s[train_idx] / 60.0
        axis.plot(initial_schedules[train_idx] / 60.0, station_axis, color=color, linestyle="--", alpha=0.45)
        axis.plot(optimized_schedules[train_idx] / 60.0, station_axis, color=color, linewidth=1.7)
        axis.scatter(initial_departure_min, station_axis[0], color=color, marker="x", s=44, zorder=3)
        axis.scatter(optimized_departure_min, station_axis[0], color=color, marker="o", s=30, zorder=3)
        train_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.7,
                label=f"Train {train_idx + 1}: {initial_departure_min:.2f} -> {optimized_departure_min:.2f} min",
            )
        )
    has_dwell_overrun = False
    for train_idx in range(environment.train_count):
        for section_idx in range(min(station_delays.shape[1], len(station_axis) - 1)):
            delay_s = float(station_delays[train_idx, section_idx])
            if delay_s <= 0.0:
                continue
            station_idx = section_idx + 1
            arrival_min = optimized_schedules[train_idx, station_idx] / 60.0
            departure_min = (optimized_schedules[train_idx, station_idx] + delay_s) / 60.0
            axis.plot(
                [arrival_min, departure_min],
                [station_axis[station_idx], station_axis[station_idx]],
                color="#D32F2F",
                linewidth=3.0,
                solid_capstyle="round",
                zorder=5,
            )
            axis.scatter(
                arrival_min,
                station_axis[station_idx],
                color="#D32F2F",
                edgecolor="white",
                linewidth=0.5,
                marker="*",
                s=130,
                zorder=6,
            )
            axis.text(
                departure_min + 0.03,
                station_axis[station_idx] + 0.08,
                f"T{train_idx + 1}+{delay_s:.0f}s",
                color="#B71C1C",
                fontsize=8,
                ha="left",
                va="bottom",
            )
            has_dwell_overrun = True
    style_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Initial timetable"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.7, label="Stable MAPPO timetable"),
        Line2D([0], [0], color="black", marker="x", linestyle="None", label="Initial departure"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", label="Stable departure"),
    ]
    if has_dwell_overrun:
        style_handles.append(
            Line2D(
                [0],
                [0],
                color="#D32F2F",
                marker="*",
                linestyle="-",
                linewidth=2.0,
                label="Dwell-overrun point",
            )
        )
    axis.set_title("Four-train diagram: initial and stable MAPPO departure timetables")
    axis.set_xlabel("Time from first departure (min)")
    axis.set_yticks(station_axis, station_labels)
    axis.grid(alpha=0.25)
    style_legend = axis.legend(handles=style_handles, loc="upper left", ncol=2)
    axis.add_artist(style_legend)
    axis.legend(handles=train_handles, loc="lower right", title="Departure time: initial -> stable", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "multi_train_timetable_optimization.png", dpi=180)
    plt.close(fig)


def plot_regenerative_energy_matching(
    tables_by_train: Sequence[Sequence[SectionResponseTable]],
    result: MultiTrainTrainingResult,
    power_bin_s: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """Mark time windows where regenerative braking can be absorbed by traction."""

    timetables = np.asarray(result.timetables_s, dtype=float)
    departures = np.asarray(result.departure_times_s, dtype=float)
    train_count = len(timetables)
    horizon_s = float(np.max(departures + np.sum(timetables, axis=1))) + power_bin_s
    bin_count = max(1, int(np.ceil(horizon_s / power_bin_s)))

    traction_energy = np.zeros(bin_count, dtype=float)
    regenerative_energy = np.zeros(bin_count, dtype=float)
    traction_by_train = np.zeros((train_count, bin_count), dtype=float)
    regenerative_by_train = np.zeros((train_count, bin_count), dtype=float)
    traction_labels = [set() for _ in range(bin_count)]
    regenerative_labels = [set() for _ in range(bin_count)]

    def distribute_profile(
        train_idx: int,
        section_name: str,
        target: np.ndarray,
        target_by_train: np.ndarray,
        target_labels: List[set],
        profile: Sequence[float],
        source_bin_s: float,
        source_duration_s: float,
        scheduled_start_s: float,
        scheduled_duration_s: float,
    ) -> None:
        time_scale = scheduled_duration_s / max(source_duration_s, EPS)
        label = f"Train {train_idx + 1}:{section_name}"
        for profile_idx, energy_kwh in enumerate(profile):
            if energy_kwh <= EPS:
                continue
            source_start_s = profile_idx * source_bin_s
            source_end_s = min(source_duration_s, source_start_s + source_bin_s)
            local_duration_s = max(EPS, (source_end_s - source_start_s) * time_scale)
            event_start_s = scheduled_start_s + source_start_s * time_scale
            event_end_s = event_start_s + local_duration_s
            first_bin = max(0, int(np.floor(event_start_s / power_bin_s)))
            final_bin = min(bin_count, int(np.ceil(event_end_s / power_bin_s)))
            for bin_idx in range(first_bin, final_bin):
                bin_start_s = bin_idx * power_bin_s
                bin_end_s = bin_start_s + power_bin_s
                overlap_s = max(
                    0.0,
                    min(event_end_s, bin_end_s) - max(event_start_s, bin_start_s),
                )
                if overlap_s <= 0.0:
                    continue
                contribution_kwh = energy_kwh * overlap_s / local_duration_s
                target[bin_idx] += contribution_kwh
                target_by_train[train_idx, bin_idx] += contribution_kwh
                target_labels[bin_idx].add(label)

    for train_idx, train_tables in enumerate(tables_by_train):
        section_start_s = float(departures[train_idx])
        for section_idx, table in enumerate(train_tables):
            scheduled_duration_s = float(timetables[train_idx, section_idx])
            point = table.lookup(scheduled_duration_s)
            distribute_profile(
                train_idx,
                table.section_name,
                traction_energy,
                traction_by_train,
                traction_labels,
                point.traction_profile_kwh,
                point.profile_bin_s,
                point.actual_time_s,
                section_start_s,
                scheduled_duration_s,
            )
            distribute_profile(
                train_idx,
                table.section_name,
                regenerative_energy,
                regenerative_by_train,
                regenerative_labels,
                point.regenerative_profile_kwh,
                point.profile_bin_s,
                point.actual_time_s,
                section_start_s,
                scheduled_duration_s,
            )
            section_start_s += scheduled_duration_s

    global_matched_regenerative = np.minimum(traction_energy, regenerative_energy)
    cross_train_matched_regenerative = np.zeros(bin_count, dtype=float)
    for bin_idx in range(bin_count):
        traction_total = traction_energy[bin_idx]
        regenerative_total = regenerative_energy[bin_idx]
        largest_same_train_pair = max(
            (
                traction_by_train[train_idx, bin_idx]
                + regenerative_by_train[train_idx, bin_idx]
                for train_idx in range(train_count)
            ),
            default=0.0,
        )
        cross_train_matched_regenerative[bin_idx] = max(
            0.0,
            min(
                traction_total,
                regenerative_total,
                traction_total + regenerative_total - largest_same_train_pair,
            ),
        )
    available_mask = cross_train_matched_regenerative > EPS
    bin_rows: List[Dict[str, Any]] = []
    for bin_idx in range(bin_count):
        row: Dict[str, Any] = {
            "time_bin_index": bin_idx,
            "time_start_s": bin_idx * power_bin_s,
            "time_end_s": (bin_idx + 1) * power_bin_s,
            "time_start_min": bin_idx * power_bin_s / 60.0,
            "time_end_min": (bin_idx + 1) * power_bin_s / 60.0,
            "supply_zone": "global",
            "traction_kwh": traction_energy[bin_idx],
            "regenerative_kwh": regenerative_energy[bin_idx],
            "matched_regenerative_kwh": cross_train_matched_regenerative[bin_idx],
            "global_matched_regenerative_kwh": global_matched_regenerative[bin_idx],
            "same_train_or_same_bin_matched_kwh": max(
                0.0,
                global_matched_regenerative[bin_idx] - cross_train_matched_regenerative[bin_idx],
            ),
            "unmatched_regenerative_kwh": max(
                0.0,
                regenerative_energy[bin_idx] - cross_train_matched_regenerative[bin_idx],
            ),
            "regenerative_absorption_ratio": cross_train_matched_regenerative[bin_idx]
            / max(regenerative_energy[bin_idx], EPS),
            "is_regenerative_available": bool(available_mask[bin_idx]),
        }
        for train_idx in range(train_count):
            row[f"train_{train_idx + 1}_traction_kwh"] = traction_by_train[train_idx, bin_idx]
            row[f"train_{train_idx + 1}_regenerative_kwh"] = regenerative_by_train[train_idx, bin_idx]
        row["traction_train_ids"] = ";".join(
            str(train_idx + 1)
            for train_idx in range(train_count)
            if traction_by_train[train_idx, bin_idx] > EPS
        )
        row["regenerative_train_ids"] = ";".join(
            str(train_idx + 1)
            for train_idx in range(train_count)
            if regenerative_by_train[train_idx, bin_idx] > EPS
        )
        row["traction_sections"] = ";".join(sorted(traction_labels[bin_idx]))
        row["regenerative_sections"] = ";".join(sorted(regenerative_labels[bin_idx]))
        bin_rows.append(row)

    interval_rows: List[Dict[str, Any]] = []
    interval_idx = 0
    bin_idx = 0
    while bin_idx < bin_count:
        if not available_mask[bin_idx]:
            bin_idx += 1
            continue
        start_idx = bin_idx
        traction_sections = set()
        regenerative_sections = set()
        traction_train_ids = set()
        regenerative_train_ids = set()
        while bin_idx < bin_count and available_mask[bin_idx]:
            traction_sections.update(traction_labels[bin_idx])
            regenerative_sections.update(regenerative_labels[bin_idx])
            traction_train_ids.update(
                str(train_idx + 1)
                for train_idx in range(train_count)
                if traction_by_train[train_idx, bin_idx] > EPS
            )
            regenerative_train_ids.update(
                str(train_idx + 1)
                for train_idx in range(train_count)
                if regenerative_by_train[train_idx, bin_idx] > EPS
            )
            bin_idx += 1
        end_idx = bin_idx
        traction_sum = float(np.sum(traction_energy[start_idx:end_idx]))
        regenerative_sum = float(np.sum(regenerative_energy[start_idx:end_idx]))
        cross_matched_sum = float(np.sum(cross_train_matched_regenerative[start_idx:end_idx]))
        global_matched_sum = float(np.sum(global_matched_regenerative[start_idx:end_idx]))
        interval_rows.append(
            {
                "interval_id": interval_idx + 1,
                "time_start_s": start_idx * power_bin_s,
                "time_end_s": end_idx * power_bin_s,
                "duration_s": (end_idx - start_idx) * power_bin_s,
                "time_start_min": start_idx * power_bin_s / 60.0,
                "time_end_min": end_idx * power_bin_s / 60.0,
                "supply_zone": "global",
                "traction_kwh": traction_sum,
                "regenerative_kwh": regenerative_sum,
                "matched_regenerative_kwh": cross_matched_sum,
                "global_matched_regenerative_kwh": global_matched_sum,
                "same_train_or_same_bin_matched_kwh": max(0.0, global_matched_sum - cross_matched_sum),
                "unmatched_regenerative_kwh": max(0.0, regenerative_sum - cross_matched_sum),
                "regenerative_absorption_ratio": cross_matched_sum / max(regenerative_sum, EPS),
                "traction_train_ids": ";".join(sorted(traction_train_ids)),
                "regenerative_train_ids": ";".join(sorted(regenerative_train_ids)),
                "traction_sections": ";".join(sorted(traction_sections)),
                "regenerative_sections": ";".join(sorted(regenerative_sections)),
            }
        )
        interval_idx += 1

    write_csv(output_dir / "regenerative_matching_time_bins.csv", bin_rows)
    write_csv(output_dir / "available_regenerative_intervals.csv", interval_rows)

    time_min = (np.arange(bin_count) + 0.5) * power_bin_s / 60.0
    colors = plt.cm.tab10(np.linspace(0, 1, train_count))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.35]},
    )
    for row in interval_rows:
        start_min = row["time_start_min"]
        end_min = row["time_end_min"]
        axes[0].axvspan(start_min, end_min, color="#FFE082", alpha=0.20)
        axes[1].axvspan(start_min, end_min, color="#FFE082", alpha=0.20)
    axes[0].step(time_min, traction_energy, where="mid", color="#2E7D32", label="Traction demand")
    axes[0].step(time_min, regenerative_energy, where="mid", color="#C62828", label="Regenerative braking")
    axes[0].step(
        time_min,
        global_matched_regenerative,
        where="mid",
        color="#616161",
        linestyle=":",
        linewidth=1.0,
        label="Global matched regen used in objective",
    )
    axes[0].fill_between(
        time_min,
        cross_train_matched_regenerative,
        step="mid",
        color="#FB8C00",
        alpha=0.35,
        label="Cross-train available regen",
    )
    axes[0].set_title("Cross-train available regenerative-braking intervals under current global zone model")
    axes[0].set_ylabel(f"Energy per {power_bin_s:.0f}s bin (kWh)")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.25)

    max_traction = max(float(np.max(traction_by_train)), EPS)
    max_regenerative = max(float(np.max(regenerative_by_train)), EPS)
    heat = np.zeros((train_count * 2, bin_count), dtype=float)
    labels = []
    for train_idx in range(train_count):
        heat[train_idx * 2] = traction_by_train[train_idx] / max_traction
        heat[train_idx * 2 + 1] = regenerative_by_train[train_idx] / max_regenerative
        labels.extend((f"Train {train_idx + 1} traction", f"Train {train_idx + 1} regen"))
    axes[1].imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, bin_count * power_bin_s / 60.0, train_count * 2, 0.0),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    for train_idx, color in enumerate(colors):
        axes[1].axhline(train_idx * 2, color=color, linewidth=1.0, alpha=0.55)
    axes[1].set_yticks(np.arange(train_count * 2) + 0.5, labels)
    axes[1].set_xlabel("Time from first departure (min)")
    axes[1].set_title("Per-train traction and regenerative-braking activity")
    fig.tight_layout()
    fig.savefig(output_dir / "regenerative_energy_matching.png", dpi=180)
    plt.close(fig)

    return {
        "available_regenerative_interval_count": len(interval_rows),
        "available_regenerative_time_s": float(np.sum(available_mask) * power_bin_s),
        "marked_cross_train_matched_regenerative_kwh": float(np.sum(cross_train_matched_regenerative)),
        "marked_global_matched_regenerative_kwh": float(np.sum(global_matched_regenerative)),
        "same_train_or_same_bin_matched_regenerative_kwh": float(
            np.sum(np.maximum(0.0, global_matched_regenerative - cross_train_matched_regenerative))
        ),
    }


def plot_regenerative_energy_matching_by_zone(
    tables_by_train: Sequence[Sequence[SectionResponseTable]],
    result: MultiTrainTrainingResult,
    power_bin_s: float,
    supply_zone_size: int,
    supply_zone_section_counts: Optional[Sequence[int]],
    supply_zone_section_groups: Optional[Sequence[Sequence[str]]],
    supply_zone_names: Optional[Sequence[str]],
    station_dwell_after_section_s: Optional[Sequence[Sequence[float]]],
    station_delay_after_section_s: Optional[Sequence[Sequence[float]]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Mark regenerative-energy availability by time bin and power-supply zone."""

    timetables = np.asarray(result.timetables_s, dtype=float)
    departures = np.asarray(result.departure_times_s, dtype=float)
    station_delays = np.zeros_like(timetables, dtype=float)
    if station_dwell_after_section_s is not None:
        station_delays += np.asarray(station_dwell_after_section_s, dtype=float)
    if station_delay_after_section_s is not None:
        station_delays += np.asarray(station_delay_after_section_s, dtype=float)
    if station_delays.shape != timetables.shape:
        raise ValueError(
            "station dwell/delay arrays must match timetables shape, "
            f"got {station_delays.shape} vs {timetables.shape}"
        )
    train_count = len(timetables)
    section_count = len(SECTION_NAMES)
    supply_zone_size = max(0, int(supply_zone_size))
    zone_counts = tuple(
        int(value)
        for value in (supply_zone_section_counts or ())
        if int(value) > 0
    )
    if zone_counts and sum(zone_counts) != section_count:
        raise ValueError(
            "supply_zone_section_counts must sum to the number of interstation "
            f"sections ({section_count}), got {sum(zone_counts)} from {zone_counts}"
        )
    section_index = {name: idx for idx, name in enumerate(SECTION_NAMES)}
    section_to_zone = np.full(section_count, -1, dtype=int)
    zone_groups: Tuple[Tuple[int, ...], ...] = ()
    zone_group_names: Tuple[str, ...] = ()
    if supply_zone_section_groups:
        groups: List[Tuple[int, ...]] = []
        for zone_idx, group in enumerate(supply_zone_section_groups):
            indices: List[int] = []
            for section_name in group:
                if section_name not in section_index:
                    raise ValueError(f"Unknown section in supply zone: {section_name}")
                section_idx = section_index[section_name]
                if section_to_zone[section_idx] >= 0:
                    raise ValueError(f"Section appears in multiple supply zones: {section_name}")
                section_to_zone[section_idx] = zone_idx
                indices.append(section_idx)
            if not indices:
                raise ValueError("Supply-zone section groups cannot contain empty groups")
            groups.append(tuple(indices))
        zone_groups = tuple(groups)
        if supply_zone_names:
            zone_group_names = tuple(str(name) for name in supply_zone_names)
            if len(zone_group_names) != len(zone_groups):
                raise ValueError("supply_zone_names must match supply_zone_section_groups")
        else:
            zone_group_names = tuple(
                f"zone_{idx + 1:02d}_{SECTION_NAMES[group[0]]}__{SECTION_NAMES[group[-1]]}"
                for idx, group in enumerate(zone_groups)
            )
    zone_count = (
        len(zone_groups)
        if zone_groups
        else len(zone_counts)
        if zone_counts
        else 1
        if supply_zone_size <= 0
        else int(np.ceil(section_count / supply_zone_size))
    )
    section_finish_times = departures[:, None] + np.cumsum(timetables + station_delays, axis=1)
    horizon_s = float(np.max(section_finish_times)) + power_bin_s
    bin_count = max(1, int(np.ceil(horizon_s / power_bin_s)))

    def zone_id(section_idx: int) -> int:
        if zone_groups:
            return int(section_to_zone[int(section_idx)])
        if zone_counts:
            boundary = 0
            for idx, count in enumerate(zone_counts):
                boundary += count
                if section_idx < boundary:
                    return idx
            return len(zone_counts) - 1
        return 0 if supply_zone_size <= 0 else section_idx // supply_zone_size

    def zone_name(zone_idx: int) -> str:
        if zone_groups:
            return zone_group_names[zone_idx]
        if zone_counts:
            start_idx = sum(zone_counts[:zone_idx])
            end_idx = start_idx + zone_counts[zone_idx]
        elif supply_zone_size <= 0:
            return "global"
        else:
            start_idx = zone_idx * supply_zone_size
            end_idx = min(section_count, start_idx + supply_zone_size)
        return f"zone_{zone_idx + 1:02d}_{SECTION_NAMES[start_idx]}__{SECTION_NAMES[end_idx - 1]}"

    traction = np.zeros((zone_count, bin_count), dtype=float)
    regen = np.zeros((zone_count, bin_count), dtype=float)
    traction_by_train = np.zeros((zone_count, train_count, bin_count), dtype=float)
    regen_by_train = np.zeros((zone_count, train_count, bin_count), dtype=float)
    traction_labels = [[set() for _ in range(bin_count)] for _ in range(zone_count)]
    regen_labels = [[set() for _ in range(bin_count)] for _ in range(zone_count)]

    def distribute(
        zone_idx: int,
        train_idx: int,
        section_name: str,
        target: np.ndarray,
        target_by_train: np.ndarray,
        labels: List[List[set]],
        profile: Sequence[float],
        source_bin_s: float,
        source_duration_s: float,
        scheduled_start_s: float,
        scheduled_duration_s: float,
    ) -> None:
        time_scale = scheduled_duration_s / max(source_duration_s, EPS)
        label = f"Train {train_idx + 1}:{section_name}"
        for profile_idx, energy_kwh in enumerate(profile):
            if energy_kwh <= EPS:
                continue
            source_start_s = profile_idx * source_bin_s
            source_end_s = min(source_duration_s, source_start_s + source_bin_s)
            local_duration_s = max(EPS, (source_end_s - source_start_s) * time_scale)
            event_start_s = scheduled_start_s + source_start_s * time_scale
            event_end_s = event_start_s + local_duration_s
            first_bin = max(0, int(np.floor(event_start_s / power_bin_s)))
            final_bin = min(bin_count, int(np.ceil(event_end_s / power_bin_s)))
            for bin_idx in range(first_bin, final_bin):
                bin_start_s = bin_idx * power_bin_s
                bin_end_s = bin_start_s + power_bin_s
                overlap_s = max(0.0, min(event_end_s, bin_end_s) - max(event_start_s, bin_start_s))
                if overlap_s <= 0.0:
                    continue
                contribution_kwh = energy_kwh * overlap_s / local_duration_s
                target[zone_idx, bin_idx] += contribution_kwh
                target_by_train[zone_idx, train_idx, bin_idx] += contribution_kwh
                labels[zone_idx][bin_idx].add(label)

    for train_idx, train_tables in enumerate(tables_by_train):
        section_start_s = float(departures[train_idx])
        for section_idx, table in enumerate(train_tables):
            scheduled_duration_s = float(timetables[train_idx, section_idx])
            point = table.lookup(scheduled_duration_s)
            current_zone = zone_id(section_idx)
            if current_zone < 0:
                section_start_s += scheduled_duration_s + float(station_delays[train_idx, section_idx])
                continue
            distribute(
                current_zone,
                train_idx,
                table.section_name,
                traction,
                traction_by_train,
                traction_labels,
                point.traction_profile_kwh,
                point.profile_bin_s,
                point.actual_time_s,
                section_start_s,
                scheduled_duration_s,
            )
            distribute(
                current_zone,
                train_idx,
                table.section_name,
                regen,
                regen_by_train,
                regen_labels,
                point.regenerative_profile_kwh,
                point.profile_bin_s,
                point.actual_time_s,
                section_start_s,
                scheduled_duration_s,
            )
            section_start_s += scheduled_duration_s + float(station_delays[train_idx, section_idx])

    zone_matched = np.minimum(traction, regen)
    cross_train_matched = np.zeros((zone_count, bin_count), dtype=float)
    for zone_idx in range(zone_count):
        for bin_idx in range(bin_count):
            largest_same_train_pair = max(
                traction_by_train[zone_idx, train_idx, bin_idx]
                + regen_by_train[zone_idx, train_idx, bin_idx]
                for train_idx in range(train_count)
            )
            cross_train_matched[zone_idx, bin_idx] = max(
                0.0,
                min(
                    traction[zone_idx, bin_idx],
                    regen[zone_idx, bin_idx],
                    traction[zone_idx, bin_idx] + regen[zone_idx, bin_idx] - largest_same_train_pair,
                ),
            )
    available = zone_matched > EPS

    bin_rows: List[Dict[str, Any]] = []
    for zone_idx in range(zone_count):
        for bin_idx in range(bin_count):
            row: Dict[str, Any] = {
                "time_bin_index": bin_idx,
                "time_start_s": bin_idx * power_bin_s,
                "time_end_s": (bin_idx + 1) * power_bin_s,
                "time_start_min": bin_idx * power_bin_s / 60.0,
                "time_end_min": (bin_idx + 1) * power_bin_s / 60.0,
                "supply_zone": zone_name(zone_idx),
                "supply_zone_index": zone_idx + 1,
                "traction_kwh": traction[zone_idx, bin_idx],
                "regenerative_kwh": regen[zone_idx, bin_idx],
                "matched_regenerative_kwh": zone_matched[zone_idx, bin_idx],
                "cross_train_matched_regenerative_kwh": cross_train_matched[zone_idx, bin_idx],
                "same_train_or_same_bin_matched_kwh": max(
                    0.0,
                    zone_matched[zone_idx, bin_idx] - cross_train_matched[zone_idx, bin_idx],
                ),
                "unmatched_regenerative_kwh": max(0.0, regen[zone_idx, bin_idx] - zone_matched[zone_idx, bin_idx]),
                "regenerative_absorption_ratio": zone_matched[zone_idx, bin_idx] / max(regen[zone_idx, bin_idx], EPS),
                "is_regenerative_available": bool(available[zone_idx, bin_idx]),
            }
            for train_idx in range(train_count):
                row[f"train_{train_idx + 1}_traction_kwh"] = traction_by_train[zone_idx, train_idx, bin_idx]
                row[f"train_{train_idx + 1}_regenerative_kwh"] = regen_by_train[zone_idx, train_idx, bin_idx]
            row["traction_train_ids"] = ";".join(
                str(train_idx + 1)
                for train_idx in range(train_count)
                if traction_by_train[zone_idx, train_idx, bin_idx] > EPS
            )
            row["regenerative_train_ids"] = ";".join(
                str(train_idx + 1)
                for train_idx in range(train_count)
                if regen_by_train[zone_idx, train_idx, bin_idx] > EPS
            )
            row["traction_sections"] = ";".join(sorted(traction_labels[zone_idx][bin_idx]))
            row["regenerative_sections"] = ";".join(sorted(regen_labels[zone_idx][bin_idx]))
            bin_rows.append(row)

    interval_rows: List[Dict[str, Any]] = []
    interval_id = 1
    for zone_idx in range(zone_count):
        bin_idx = 0
        while bin_idx < bin_count:
            if not available[zone_idx, bin_idx]:
                bin_idx += 1
                continue
            start_idx = bin_idx
            traction_sections = set()
            regen_sections = set()
            traction_train_ids = set()
            regen_train_ids = set()
            while bin_idx < bin_count and available[zone_idx, bin_idx]:
                traction_sections.update(traction_labels[zone_idx][bin_idx])
                regen_sections.update(regen_labels[zone_idx][bin_idx])
                traction_train_ids.update(
                    str(train_idx + 1)
                    for train_idx in range(train_count)
                    if traction_by_train[zone_idx, train_idx, bin_idx] > EPS
                )
                regen_train_ids.update(
                    str(train_idx + 1)
                    for train_idx in range(train_count)
                    if regen_by_train[zone_idx, train_idx, bin_idx] > EPS
                )
                bin_idx += 1
            end_idx = bin_idx
            traction_sum = float(np.sum(traction[zone_idx, start_idx:end_idx]))
            regen_sum = float(np.sum(regen[zone_idx, start_idx:end_idx]))
            cross_sum = float(np.sum(cross_train_matched[zone_idx, start_idx:end_idx]))
            zone_sum = float(np.sum(zone_matched[zone_idx, start_idx:end_idx]))
            interval_rows.append(
                {
                    "interval_id": interval_id,
                    "time_start_s": start_idx * power_bin_s,
                    "time_end_s": end_idx * power_bin_s,
                    "duration_s": (end_idx - start_idx) * power_bin_s,
                    "time_start_min": start_idx * power_bin_s / 60.0,
                    "time_end_min": end_idx * power_bin_s / 60.0,
                    "supply_zone": zone_name(zone_idx),
                    "supply_zone_index": zone_idx + 1,
                    "traction_kwh": traction_sum,
                    "regenerative_kwh": regen_sum,
                    "matched_regenerative_kwh": zone_sum,
                    "cross_train_matched_regenerative_kwh": cross_sum,
                    "same_train_or_same_bin_matched_kwh": max(0.0, zone_sum - cross_sum),
                    "unmatched_regenerative_kwh": max(0.0, regen_sum - zone_sum),
                    "regenerative_absorption_ratio": zone_sum / max(regen_sum, EPS),
                    "traction_train_ids": ";".join(sorted(traction_train_ids)),
                    "regenerative_train_ids": ";".join(sorted(regen_train_ids)),
                    "traction_sections": ";".join(sorted(traction_sections)),
                    "regenerative_sections": ";".join(sorted(regen_sections)),
                }
            )
            interval_id += 1

    write_csv(output_dir / "regenerative_matching_time_bins.csv", bin_rows)
    write_csv(output_dir / "available_regenerative_intervals.csv", interval_rows)

    time_min = (np.arange(bin_count) + 0.5) * power_bin_s / 60.0
    traction_total = np.sum(traction, axis=0)
    regen_total = np.sum(regen, axis=0)
    zone_matched_total = np.sum(zone_matched, axis=0)
    cross_total = np.sum(cross_train_matched, axis=0)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.1, 1.35]},
    )
    for row in interval_rows:
        for axis in axes:
            axis.axvspan(row["time_start_min"], row["time_end_min"], color="#FFE082", alpha=0.16)
    axes[0].step(time_min, traction_total, where="mid", color="#2E7D32", label="Traction demand")
    axes[0].step(time_min, regen_total, where="mid", color="#C62828", label="Regenerative braking")
    axes[0].step(time_min, zone_matched_total, where="mid", color="#616161", linestyle=":", linewidth=1.0, label="Zone matched regen used in objective")
    axes[0].fill_between(time_min, zone_matched_total, step="mid", color="#FB8C00", alpha=0.35, label="Zone available regen")
    axes[0].step(time_min, cross_total, where="mid", color="#8E24AA", linestyle="--", linewidth=0.9, label="Cross-train matched regen")
    axes[0].set_title(
        "Available regenerative-braking intervals by power-supply zone"
        if zone_groups or zone_counts or supply_zone_size > 0
        else "Available regenerative-braking intervals under global zone model"
    )
    axes[0].set_ylabel(f"Energy per {power_bin_s:.0f}s bin (kWh)")
    axes[0].legend(ncol=4)
    axes[0].grid(alpha=0.25)

    zone_heat = np.where(available, zone_matched, 0.0)
    zone_scale = max(float(np.max(zone_heat)), EPS)
    axes[1].imshow(
        zone_heat / zone_scale,
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, bin_count * power_bin_s / 60.0, zone_count, 0.0),
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_yticks(np.arange(zone_count) + 0.5, [zone_name(idx) for idx in range(zone_count)])
    axes[1].set_title("Matched regenerative energy by power-supply zone")

    traction_by_train_total = np.sum(traction_by_train, axis=0)
    regen_by_train_total = np.sum(regen_by_train, axis=0)
    max_traction = max(float(np.max(traction_by_train_total)), EPS)
    max_regen = max(float(np.max(regen_by_train_total)), EPS)
    heat = np.zeros((train_count * 2, bin_count), dtype=float)
    labels = []
    colors = plt.cm.tab10(np.linspace(0, 1, train_count))
    for train_idx in range(train_count):
        heat[train_idx * 2] = traction_by_train_total[train_idx] / max_traction
        heat[train_idx * 2 + 1] = regen_by_train_total[train_idx] / max_regen
        labels.extend((f"Train {train_idx + 1} traction", f"Train {train_idx + 1} regen"))
    axes[2].imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, bin_count * power_bin_s / 60.0, train_count * 2, 0.0),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    for train_idx, color in enumerate(colors):
        axes[2].axhline(train_idx * 2, color=color, linewidth=1.0, alpha=0.55)
    axes[2].set_yticks(np.arange(train_count * 2) + 0.5, labels)
    axes[2].set_xlabel("Time from first departure (min)")
    axes[2].set_title("Per-train traction and regenerative-braking activity")
    fig.tight_layout()
    fig.savefig(output_dir / "regenerative_energy_matching.png", dpi=180)
    plt.close(fig)

    return {
        "supply_zone_size": supply_zone_size,
        "supply_zone_count": zone_count,
        "supply_zone_section_counts": ",".join(str(value) for value in zone_counts),
        "supply_zone_names": ";".join(zone_group_names),
        "supply_zone_section_groups": ";".join(
            ",".join(SECTION_NAMES[idx] for idx in group)
            for group in zone_groups
        ),
        "available_regenerative_interval_count": len(interval_rows),
        "available_regenerative_time_s": float(np.count_nonzero(np.any(available, axis=0)) * power_bin_s),
        "available_regenerative_zone_time_s": float(np.sum(available) * power_bin_s),
        "marked_cross_train_matched_regenerative_kwh": float(np.sum(cross_train_matched)),
        "marked_zone_matched_regenerative_kwh": float(np.sum(zone_matched)),
        "same_train_or_same_bin_matched_regenerative_kwh": float(
            np.sum(np.maximum(0.0, zone_matched - cross_train_matched))
        ),
    }


def export_mappo_iteration_history(
    history: Sequence[Dict[str, Any]],
    environment: MultiTrainTimetableEnvironment,
    output_dir: Path,
) -> None:
    if not history:
        return
    scalar_rows: List[Dict[str, Any]] = []
    timetable_rows: List[Dict[str, Any]] = []
    departure_rows: List[Dict[str, Any]] = []
    for item in history:
        scalar_rows.append({key: value for key, value in item.items() if not isinstance(value, list)})
        for train_idx in range(environment.train_count):
            current_departure = item["current_departure_times_s"][train_idx]
            best_departure = item["best_departure_times_s"][train_idx]
            departure_rows.append(
                {
                    "episode": item["episode"],
                    "train_id": train_idx + 1,
                    "initial_departure_time_s": environment.base_departures_s[train_idx],
                    "current_departure_time_s": current_departure,
                    "best_departure_time_s": best_departure,
                    "current_departure_offset_s": current_departure - environment.base_departures_s[train_idx],
                    "best_departure_offset_s": best_departure - environment.base_departures_s[train_idx],
                    "current_previous_train_headway_s": (
                        ""
                        if train_idx == 0
                        else current_departure - item["current_departure_times_s"][train_idx - 1]
                    ),
                    "best_previous_train_headway_s": (
                        ""
                        if train_idx == 0
                        else best_departure - item["best_departure_times_s"][train_idx - 1]
                    ),
                }
            )
            for section_idx, section_name in enumerate(SECTION_NAMES):
                timetable_rows.append(
                    {
                        "episode": item["episode"],
                        "train_id": train_idx + 1,
                        "section_name": section_name,
                        "current_running_time_s": item["current_timetables_s"][train_idx][section_idx],
                        "current_change_s": item["current_timetables_s"][train_idx][section_idx] - INITIAL_TIMETABLE[section_idx],
                        "best_running_time_s": item["best_timetables_s"][train_idx][section_idx],
                        "best_change_s": item["best_timetables_s"][train_idx][section_idx] - INITIAL_TIMETABLE[section_idx],
                        "current_departure_time_s": item["current_departure_times_s"][train_idx],
                        "best_departure_time_s": item["best_departure_times_s"][train_idx],
                    }
                )
    write_csv(output_dir / "mappo_training_history.csv", scalar_rows)
    write_csv(output_dir / "mappo_timetable_iteration_history.csv", timetable_rows)
    write_csv(output_dir / "mappo_departure_adjustment_history.csv", departure_rows)

    episodes = np.asarray([item["episode"] for item in history])
    current = np.asarray([item["current_timetables_s"] for item in history])
    best = np.asarray([item["best_timetables_s"] for item in history])
    current_departures = np.asarray([item["current_departure_times_s"] for item in history])
    best_departures = np.asarray([item["best_departure_times_s"] for item in history])
    current_departure_offsets = current_departures - environment.base_departures_s[None, :]
    best_departure_offsets = best_departures - environment.base_departures_s[None, :]
    current_headways = np.diff(current_departures, axis=1)
    best_headways = np.diff(best_departures, axis=1)
    final_best_running_time_changes = best[-1] - INITIAL_TIMETABLE[None, :]
    station_delays = getattr(environment, "station_delay_after_section_s", None)
    station_dwell = getattr(environment, "base_station_dwell_after_section_s", None)
    initial_schedules = station_schedule_times(
        np.tile(INITIAL_TIMETABLE, (environment.train_count, 1)),
        environment.base_departures_s,
        station_dwell,
        station_delays,
    )
    final_best_schedules = station_schedule_times(
        best[-1],
        best_departures[-1],
        station_dwell,
        station_delays,
    )
    final_best_station_time_changes = final_best_schedules - initial_schedules
    final_best_changes = (
        final_best_station_time_changes
        if environment.control_mode == "departure"
        else final_best_running_time_changes
    )
    final_change_labels = (
        [SECTION_NAMES[0].split("-")[0]] + [item.split("-")[-1] for item in SECTION_NAMES]
        if environment.control_mode == "departure"
        else SECTION_NAMES
    )
    limit = max(1.0, float(np.max(np.abs(final_best_changes))))
    colors = plt.cm.tab10(np.linspace(0, 1, environment.train_count))

    fig, axes = plt.subplots(3, 1, figsize=(16, 13))
    for train_idx, color in enumerate(colors):
        axes[0].plot(episodes, current_departure_offsets[:, train_idx], color=color, alpha=0.20, linewidth=0.8)
        axes[0].plot(episodes, best_departure_offsets[:, train_idx], color=color, linewidth=1.5, label=f"Train {train_idx + 1}")
    axes[0].axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    axes[0].set_title("Departure-time offsets: solid best, faint current")
    axes[0].set_ylabel("Offset from initial departure (s)")
    axes[0].legend(ncol=4)
    for headway_idx in range(environment.train_count - 1):
        color = colors[headway_idx + 1]
        label = f"Train {headway_idx + 1} -> {headway_idx + 2}"
        axes[1].plot(episodes, current_headways[:, headway_idx], color=color, alpha=0.20, linewidth=0.8)
        axes[1].plot(episodes, best_headways[:, headway_idx], color=color, linewidth=1.5, label=label)
    axes[1].axhline(environment.initial_headway_s, color="black", linestyle=":", linewidth=0.9, label="Initial headway")
    axes[1].axhline(environment.minimum_headway_s, color="#C62828", linestyle="--", linewidth=0.9, label="Minimum headway")
    axes[1].set_title("Adjacent departure headways: solid best, faint current")
    axes[1].set_ylabel("Headway (s)")
    axes[1].legend(ncol=5)
    image = axes[2].imshow(final_best_changes, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[2].set_title(
        "Final best station-time adjustments from departure optimization"
        if environment.control_mode == "departure"
        else "Final best interstation running-time adjustments"
    )
    axes[2].set_xlabel("Station" if environment.control_mode == "departure" else "Interstation section")
    axes[2].set_ylabel("Train")
    axes[2].set_yticks(np.arange(environment.train_count), [f"Train {idx + 1}" for idx in range(environment.train_count)])
    axes[2].set_xticks(np.arange(len(final_change_labels)), final_change_labels, rotation=55, ha="right")
    for train_idx in range(environment.train_count):
        for section_idx in range(len(final_change_labels)):
            axes[2].text(section_idx, train_idx, f"{final_best_changes[train_idx, section_idx]:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axes[2], label="Schedule change (s)")
    for axis in axes[:2]:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "mappo_timetable_iteration_process.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for train_idx, color in enumerate(colors):
        axes[0].plot(episodes, best_departures[:, train_idx] / 60.0, color=color, linewidth=1.5, label=f"Train {train_idx + 1}")
        axes[1].plot(episodes, best_departure_offsets[:, train_idx], color=color, linewidth=1.5, label=f"Train {train_idx + 1}")
    axes[0].set_title("Best departure times during MAPPO coordination")
    axes[0].set_ylabel("Departure time (min)")
    axes[0].legend(ncol=4)
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    axes[1].set_title("Best departure offsets from the 8-minute initial schedule")
    axes[1].set_ylabel("Offset (s)")
    for headway_idx in range(environment.train_count - 1):
        axes[2].plot(episodes, best_headways[:, headway_idx], color=colors[headway_idx + 1], linewidth=1.5, label=f"Train {headway_idx + 1} -> {headway_idx + 2}")
    axes[2].axhline(environment.initial_headway_s, color="black", linestyle=":", linewidth=0.9, label="Initial headway")
    axes[2].axhline(environment.minimum_headway_s, color="#C62828", linestyle="--", linewidth=0.9, label="Minimum headway")
    axes[2].set_title("Best adjacent departure headways")
    axes[2].set_ylabel("Headway (s)")
    axes[2].set_xlabel("Episode")
    axes[2].legend(ncol=5)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "mappo_departure_time_adjustments.png", dpi=180)
    plt.close(fig)


def plot_training_process(
    lower_history: Sequence[Dict[str, Any]],
    upper_history: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    write_csv(output_dir / "lower_training_history.csv", lower_history)
    if lower_history:
        def _row_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes"}

        lower_algorithms = list(
            dict.fromkeys(str(item.get("lower_algorithm", "dqn")).upper() for item in lower_history)
        )
        lower_algorithm = "+".join(lower_algorithms)
        train_names = list(dict.fromkeys(item["train_name"] for item in lower_history))
        colors = plt.cm.tab10(np.linspace(0, 1, len(train_names)))
        reward_history = [
            item
            for item in lower_history
            if str(item.get("phase", "")) != "behavior_clone_pretrain"
        ] or list(lower_history)
        reward_train_names = list(dict.fromkeys(item["train_name"] for item in reward_history))
        reward_colors = plt.cm.tab10(np.linspace(0, 1, len(reward_train_names)))

        def _training_steps(rows: Sequence[Dict[str, Any]]) -> List[int]:
            return list(range(1, len(rows) + 1))

        def _phase_boundaries(rows: Sequence[Dict[str, Any]]) -> List[int]:
            boundaries: List[int] = []
            previous_phase = rows[0].get("phase") if rows else None
            for idx, row in enumerate(rows[1:], start=2):
                phase = row.get("phase")
                if phase != previous_phase:
                    boundaries.append(idx - 1)
                previous_phase = phase
            return boundaries

        def _draw_phase_boundaries(axes_to_mark, rows: Sequence[Dict[str, Any]]) -> None:
            for boundary in _phase_boundaries(rows):
                for axis in axes_to_mark:
                    axis.axvline(
                        boundary + 0.5,
                        color="#757575",
                        linestyle=":",
                        linewidth=0.9,
                        alpha=0.75,
                    )

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        convergence_rows: List[Dict[str, Any]] = []
        for train_name, color in zip(reward_train_names, reward_colors):
            rows = [item for item in reward_history if item["train_name"] == train_name]
            episodes = _training_steps(rows)
            reward = history_series(rows, "episode_reward_per_km")
            traction_energy = history_series(rows, "traction_kwh")
            punctuality_error = history_series(
                rows,
                "timetable_error_s",
                transform=abs,
            )
            jerk = history_series(rows, "jerk_integral_mps3_s")
            axes[0, 0].plot(episodes, moving_average(reward), color=color, label=train_name)
            axes[0, 1].plot(episodes, moving_average(traction_energy), color=color, label=train_name)
            axes[1, 0].plot(episodes, moving_average(punctuality_error), color=color, label=train_name)
            axes[1, 1].plot(episodes, moving_average(jerk), color=color, label=train_name)
            reward_average = moving_average(reward)
            finite_reward_average = reward_average[np.isfinite(reward_average)]
            convergence_window = min(50, len(finite_reward_average))
            final_window = (
                finite_reward_average[-convergence_window:]
                if convergence_window > 0
                else np.asarray([np.nan])
            )
            final_slope = (
                float(np.polyfit(np.arange(convergence_window), final_window, 1)[0])
                if convergence_window >= 2
                else 0.0
            )
            convergence_rows.append(
                {
                    "train_name": train_name,
                    "episodes": len(rows),
                    "phase_filter": "exclude_behavior_clone_pretrain",
                    "moving_average_window": min(20, len(rows)),
                    "convergence_window": convergence_window,
                    "initial_reward_per_km_moving_average": (
                        finite_reward_average[0]
                        if len(finite_reward_average)
                        else ""
                    ),
                    "final_reward_per_km_moving_average": (
                        finite_reward_average[-1]
                        if len(finite_reward_average)
                        else ""
                    ),
                    "final_window_std": (
                        float(np.std(final_window))
                        if np.all(np.isfinite(final_window))
                        else ""
                    ),
                    "final_window_slope": final_slope,
                }
            )
        axes[0, 0].set_title(f"Lower {lower_algorithm} RL normalized-reward moving average")
        axes[0, 0].set_ylabel("Reward per km")
        axes[0, 1].set_title(f"Lower {lower_algorithm} RL traction-energy moving average")
        axes[0, 1].set_ylabel("Traction energy (kWh)")
        axes[1, 0].set_title(f"Lower {lower_algorithm} RL punctuality-error moving average")
        axes[1, 0].set_ylabel("Absolute time error (s)")
        axes[1, 1].set_title(f"Lower {lower_algorithm} RL jerk moving average")
        axes[1, 1].set_ylabel("Integral |jerk| dt (m/s^2)")
        for axis in axes.flat:
            axis.set_xlabel("RL fine-tuning step per policy")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "lower_training_process.png", dpi=180)
        plt.close(fig)
        write_csv(output_dir / "lower_reward_convergence_summary.csv", convergence_rows)

        section_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for item in reward_history:
            key = (
                str(item.get("train_name", "")),
                str(item.get("section_name", "")),
            )
            section_groups.setdefault(key, []).append(item)
        section_summary_rows: List[Dict[str, Any]] = []
        for (train_name, section_name), rows in section_groups.items():
            rewards = np.asarray(history_series(rows, "episode_reward_per_km"), dtype=float)
            masses = np.asarray(history_series(rows, "sampled_total_mass_t"), dtype=float)
            finite_rewards = rewards[np.isfinite(rewards)]
            finite_masses = masses[np.isfinite(masses)]
            section_summary_rows.append(
                {
                    "train_name": train_name,
                    "section_name": section_name,
                    "sample_count": len(rows),
                    "mean_reward_per_km": (
                        float(np.mean(finite_rewards)) if len(finite_rewards) else ""
                    ),
                    "std_reward_per_km": (
                        float(np.std(finite_rewards)) if len(finite_rewards) else ""
                    ),
                    "best_reward_per_km": (
                        float(np.max(finite_rewards)) if len(finite_rewards) else ""
                    ),
                    "worst_reward_per_km": (
                        float(np.min(finite_rewards)) if len(finite_rewards) else ""
                    ),
                    "mean_sampled_mass_t": (
                        float(np.mean(finite_masses)) if len(finite_masses) else ""
                    ),
                    "feasible_rate": float(
                        np.mean([1.0 if _row_bool(row.get("feasible")) else 0.0 for row in rows])
                    ),
                }
            )
        write_csv(output_dir / "lower_reward_by_section_summary.csv", section_summary_rows)

        def _mass_bin_label(value: Any, width_t: float = 5.0) -> str:
            try:
                mass_t = float(value)
            except (TypeError, ValueError):
                return "unknown"
            if not np.isfinite(mass_t):
                return "unknown"
            lower = np.floor(mass_t / width_t) * width_t
            upper = lower + width_t
            return f"{lower:.0f}-{upper:.0f}t"

        coverage_groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
        direction_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for item in lower_history:
            phase = str(item.get("phase", "rl_finetune"))
            train_name = str(item.get("train_name", ""))
            direction = str(item.get("direction", "")) or "unknown"
            section_name = str(item.get("section_name", ""))
            mass_bin = _mass_bin_label(item.get("sampled_total_mass_t"))
            coverage_groups.setdefault(
                (train_name, phase, direction, section_name, mass_bin),
                [],
            ).append(item)
            direction_groups.setdefault((train_name, phase, direction), []).append(item)

        coverage_rows: List[Dict[str, Any]] = []
        for (train_name, phase, direction, section_name, mass_bin), rows in sorted(
            coverage_groups.items()
        ):
            rewards = np.asarray(history_series(rows, "episode_reward_per_km"), dtype=float)
            masses = np.asarray(history_series(rows, "sampled_total_mass_t"), dtype=float)
            finite_rewards = rewards[np.isfinite(rewards)]
            finite_masses = masses[np.isfinite(masses)]
            coverage_rows.append(
                {
                    "train_name": train_name,
                    "phase": phase,
                    "direction": direction,
                    "section_name": section_name,
                    "mass_bin": mass_bin,
                    "sample_count": len(rows),
                    "feasible_count": sum(1 for row in rows if _row_bool(row.get("feasible"))),
                    "feasible_rate": float(
                        np.mean([1.0 if _row_bool(row.get("feasible")) else 0.0 for row in rows])
                    ),
                    "mean_sampled_mass_t": (
                        float(np.mean(finite_masses)) if len(finite_masses) else ""
                    ),
                    "mean_reward_per_km": (
                        float(np.mean(finite_rewards)) if len(finite_rewards) else ""
                    ),
                    "std_reward_per_km": (
                        float(np.std(finite_rewards)) if len(finite_rewards) else ""
                    ),
                }
            )
        write_csv(output_dir / "lower_training_coverage_by_mass_bin.csv", coverage_rows)

        direction_rows: List[Dict[str, Any]] = []
        for (train_name, phase, direction), rows in sorted(direction_groups.items()):
            sampled_sections = sorted(
                {
                    str(row.get("section_name", ""))
                    for row in rows
                    if str(row.get("section_name", ""))
                }
            )
            sampled_mass_bins = sorted(
                {
                    _mass_bin_label(row.get("sampled_total_mass_t"))
                    for row in rows
                }
            )
            direction_rows.append(
                {
                    "train_name": train_name,
                    "phase": phase,
                    "direction": direction,
                    "sample_count": len(rows),
                    "unique_section_count": len(sampled_sections),
                    "sampled_sections": ";".join(sampled_sections),
                    "sampled_mass_bins": ";".join(sampled_mass_bins),
                    "feasible_rate": float(
                        np.mean([1.0 if _row_bool(row.get("feasible")) else 0.0 for row in rows])
                    ),
                }
            )
        write_csv(output_dir / "lower_training_coverage_by_direction.csv", direction_rows)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for train_name, color in zip(reward_train_names, reward_colors):
            rows = [item for item in reward_history if item["train_name"] == train_name]
            episodes = _training_steps(rows)
            reward_per_km = np.asarray(history_series(rows, "episode_reward_per_km"), dtype=float)
            total_reward = np.asarray(history_series(rows, "episode_reward"), dtype=float)
            feasible = np.asarray(
                [1.0 if _row_bool(row.get("feasible")) else 0.0 for row in rows],
                dtype=float,
            )
            axes[0, 0].plot(
                episodes,
                reward_per_km,
                color=color,
                alpha=0.18,
                linewidth=0.6,
            )
            axes[0, 0].plot(
                episodes,
                moving_average(reward_per_km),
                color=color,
                linewidth=1.5,
                label=f"{train_name} reward/km",
            )
            axes[0, 1].plot(
                episodes,
                moving_average(total_reward),
                color=color,
                linewidth=1.2,
                label=train_name,
            )
            axes[1, 0].plot(
                episodes,
                moving_average(feasible, window=min(50, max(1, len(feasible)))),
                color=color,
                linewidth=1.5,
                label=train_name,
            )
        axes[0, 0].set_title("Lower-policy RL episode reward per km")
        axes[0, 0].set_ylabel("Reward / km")
        axes[0, 1].set_title("Lower-policy RL total episode reward")
        axes[0, 1].set_ylabel("Episode reward")
        axes[1, 0].set_title("Rolling feasible rate")
        axes[1, 0].set_ylabel("Feasible ratio")
        axes[1, 0].set_ylim(-0.05, 1.05)

        component_rows = list(reward_history)
        component_x = np.arange(1, len(component_rows) + 1)
        component_specs = [
            ("traction_reward_component", "traction component", "#1565C0"),
            ("punctuality_reward_component", "punctuality component", "#2E7D32"),
            ("comfort_reward_component", "comfort component", "#EF6C00"),
            ("smoothness_penalty", "smoothness penalty", "#6A1B9A"),
        ]
        plotted_component = False
        for key, label, color in component_specs:
            values = np.asarray(history_series(component_rows, key), dtype=float)
            if np.any(np.isfinite(values)):
                axes[1, 1].plot(
                    component_x,
                    moving_average(values),
                    color=color,
                    linewidth=1.3,
                    label=label,
                )
                plotted_component = True
        if not plotted_component and section_summary_rows:
            labels = [row["section_name"] for row in section_summary_rows]
            values = [
                float(row["mean_reward_per_km"])
                for row in section_summary_rows
                if row["mean_reward_per_km"] != ""
            ]
            labels = labels[: len(values)]
            axes[1, 1].bar(np.arange(len(values)), values, color="#78909C")
            axes[1, 1].set_xticks(np.arange(len(values)), labels, rotation=55, ha="right")
        axes[1, 1].set_title("Reward-component moving averages")
        axes[1, 1].set_ylabel("Component value")
        for axis in axes.flat:
            axis.set_xlabel("RL fine-tuning step per policy")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "lower_reward_training_process.png", dpi=180)
        plt.close(fig)

        masses = np.asarray(history_series(reward_history, "sampled_total_mass_t"), dtype=float)
        rewards = np.asarray(history_series(reward_history, "episode_reward_per_km"), dtype=float)
        finite = np.isfinite(masses) & np.isfinite(rewards)
        if np.any(finite):
            fig, axis = plt.subplots(figsize=(10, 6))
            axis.scatter(masses[finite], rewards[finite], s=10, alpha=0.25, color="#1565C0")
            axis.set_title("Lower reward sensitivity to sampled train mass")
            axis.set_xlabel("Sampled total mass (t)")
            axis.set_ylabel("Reward / km")
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / "lower_reward_mass_sensitivity.png", dpi=180)
            plt.close(fig)
    if not upper_history:
        return
    write_csv(output_dir / "mappo_training_history.csv", upper_history)
    episodes = np.asarray([item["episode"] for item in upper_history])
    rewards = [item["reward"] for item in upper_history]
    objectives = [item["objective"] for item in upper_history]
    traction_energy = [item["traction_kwh"] for item in upper_history]
    matched_regenerative = [item["matched_regenerative_kwh"] for item in upper_history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(episodes, objectives, alpha=0.25, label="Current")
    axes[0, 0].plot(episodes, moving_average(objectives), label="Current moving average")
    axes[0, 0].plot(episodes, [item["best_objective"] for item in upper_history], label="Best")
    axes[0, 0].set_title("Normalized weighted objective (lower is better)")
    axes[0, 0].legend()
    axes[0, 1].plot(episodes, rewards, alpha=0.25, label="Episode reward")
    axes[0, 1].plot(episodes, moving_average(rewards), label="Reward moving average")
    axes[0, 1].axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    axes[0, 1].set_title("MAPPO episode reward (higher is better)")
    axes[0, 1].legend()
    axes[1, 0].plot(episodes, traction_energy, alpha=0.25, label="Traction energy")
    axes[1, 0].plot(episodes, moving_average(traction_energy), label="Traction-energy moving average")
    axes[1, 0].set_title("Total traction-energy convergence (lower is better)")
    axes[1, 0].set_ylabel("Traction energy (kWh)")
    axes[1, 0].legend()
    axes[1, 1].plot(episodes, matched_regenerative, alpha=0.25, color="#EF6C00", label="Matched regenerative energy")
    axes[1, 1].plot(episodes, moving_average(matched_regenerative), color="#EF6C00", label="Matched-energy moving average")
    axes[1, 1].set_title("Utilized regenerative-energy convergence (higher is better)")
    axes[1, 1].set_ylabel("Matched regenerative energy (kWh)")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "mappo_training_process.png", dpi=180)
    plt.close(fig)

    if "mean_abs_runtime_adjustment_s" in upper_history[0]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes[0, 0].plot(
            episodes,
            [item.get("mean_abs_runtime_adjustment_s", 0.0) for item in upper_history],
            alpha=0.35,
            label="Mean |runtime adjustment|",
        )
        axes[0, 0].plot(
            episodes,
            moving_average([item.get("mean_abs_runtime_adjustment_s", 0.0) for item in upper_history]),
            label="Moving average",
        )
        axes[0, 0].set_title("Online upper action: section runtime adjustment")
        axes[0, 0].set_ylabel("Seconds")
        axes[0, 0].legend()
        axes[0, 1].plot(
            episodes,
            [item.get("mean_abs_dwell_adjustment_s", 0.0) for item in upper_history],
            alpha=0.35,
            label="Mean |dwell/departure adjustment|",
        )
        axes[0, 1].plot(
            episodes,
            moving_average([item.get("mean_abs_dwell_adjustment_s", 0.0) for item in upper_history]),
            label="Moving average",
        )
        axes[0, 1].set_title("Online upper action: station departure adjustment")
        axes[0, 1].set_ylabel("Seconds")
        axes[0, 1].legend()
        axes[1, 0].plot(
            episodes,
            [item.get("reward_traction_energy_component", 0.0) for item in upper_history],
            alpha=0.35,
            label="Traction-energy reward component",
        )
        axes[1, 0].plot(
            episodes,
            moving_average([item.get("reward_traction_energy_component", 0.0) for item in upper_history]),
            label="Moving average",
        )
        axes[1, 0].set_title("Reward component: traction energy")
        axes[1, 0].legend()
        axes[1, 1].plot(
            episodes,
            [item.get("reward_regenerative_energy_component", 0.0) for item in upper_history],
            alpha=0.35,
            label="Regenerative-match reward component",
        )
        axes[1, 1].plot(
            episodes,
            moving_average([item.get("reward_regenerative_energy_component", 0.0) for item in upper_history]),
            label="Moving average",
        )
        axes[1, 1].set_title("Reward component: matched regenerative energy")
        axes[1, 1].legend()
        for axis in axes.flat:
            axis.set_xlabel("Episode")
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "mappo_online_action_diagnostics.png", dpi=180)
        plt.close(fig)

    policy_kl = [item.get("policy_kl", float("nan")) for item in upper_history]
    clip_fraction = [100.0 * item.get("clip_fraction", float("nan")) for item in upper_history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(episodes, [item["actor_loss"] for item in upper_history], alpha=0.35, label="Actor loss")
    axes[0, 0].plot(episodes, moving_average([item["actor_loss"] for item in upper_history]), label="Actor moving average")
    axes[0, 0].set_title("Actor loss diagnostic")
    axes[0, 0].legend()
    axes[0, 1].plot(episodes, [item["critic_loss"] for item in upper_history], alpha=0.35, label="Critic loss")
    axes[0, 1].plot(episodes, moving_average([item["critic_loss"] for item in upper_history]), label="Critic moving average")
    axes[0, 1].set_title("Critic loss diagnostic")
    axes[0, 1].legend()
    axes[1, 0].plot(episodes, policy_kl, color="#6A1B9A", label="Approximate KL")
    clip_axis = axes[1, 0].twinx()
    clip_axis.plot(episodes, clip_fraction, color="#EF6C00", alpha=0.75, label="Clip fraction")
    axes[1, 0].set_title("PPO policy-update diagnostics")
    axes[1, 0].set_ylabel("Approximate KL")
    clip_axis.set_ylabel("Clipped samples (%)")
    axes[1, 0].legend(loc="upper left")
    clip_axis.legend(loc="upper right")
    axes[1, 1].plot(episodes, [item["entropy"] for item in upper_history], label="Entropy")
    value_axis = axes[1, 1].twinx()
    value_axis.plot(episodes, [item.get("value_explained_variance", float("nan")) for item in upper_history], color="#2E7D32", label="Explained variance")
    axes[1, 1].set_title("Policy exploration and critic fit")
    axes[1, 1].set_ylabel("Entropy")
    value_axis.set_ylabel("Critic explained variance")
    axes[1, 1].legend(loc="upper left")
    value_axis.legend(loc="upper right")
    for axis in axes.flat:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "mappo_ppo_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_response_tables(
    tables_by_train: Sequence[Sequence[SectionResponseTable]],
    output_dir: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    fig, axis = plt.subplots(figsize=(14, 8))
    for tables in tables_by_train:
        for table in tables:
            axis.plot(
                [point.requested_time_s for point in table.points],
                [point.traction_kwh for point in table.points],
                marker="o",
                markersize=2.5,
                linewidth=0.7,
                alpha=0.75,
                label=f"{table.train_name} / {table.section_name}",
            )
            for point in table.points:
                rows.append(
                    {
                        "train_name": table.train_name,
                        "section_name": table.section_name,
                        "requested_time_s": point.requested_time_s,
                        "actual_time_s": point.actual_time_s,
                        "traction_kwh": point.traction_kwh,
                        "regenerative_kwh": point.regenerative_kwh,
                        "net_energy_kwh": point.net_energy_kwh,
                        "timetable_error_s": point.timetable_error_s,
                        "feasible": point.feasible,
                        "strict_timetable_feasible": point.strict_timetable_feasible,
                        "accepted_arrival_tolerance_s": point.accepted_arrival_tolerance_s,
                    }
                )
    write_csv(output_dir / "lower_response_tables.csv", rows)
    axis.set_title("Lower-agent response tables: minimum traction energy at T")
    axis.set_xlabel("Requested interstation running time (s)")
    axis.set_ylabel("Traction energy (kWh)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=4, fontsize=5)
    fig.tight_layout()
    fig.savefig(output_dir / "lower_response_curves.png", dpi=180)
    plt.close(fig)


def export_validation_visualizations(
    profile: LineProfile,
    results: Sequence[LowerResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "initial_train_operation_summary.csv",
        lower_result_summary_rows(results, "Train 1"),
    )
    plot_train_operation(profile, results, output_dir, "initial_train_operation")
    plot_section_operation_overview(profile, results, output_dir, "initial")
    print(f"Saved validation visual reports and CSV data: {output_dir}")


def export_mappo_visualizations(
    profile: LineProfile,
    environments_by_train,
    lower_controllers: Sequence[DQNAgent],
    tables_by_train: Sequence[Sequence[SectionResponseTable]],
    upper_environment: MultiTrainTimetableEnvironment,
    result: MultiTrainTrainingResult,
    lower_history: Sequence[Dict[str, Any]],
    upper_history: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_configs = [
        train_environments[0].dynamics.config
        for train_environments in environments_by_train
    ]
    write_csv(
        output_dir / "train_parameters.csv",
        [asdict(config) for config in train_configs],
    )
    write_csv(
        output_dir / "driving_smoothness_parameters.csv",
        [asdict(DRIVING_SMOOTHNESS)],
    )
    operation_summary_rows: List[Dict[str, Any]] = []
    operation_results_by_train: List[Sequence[LowerResult]] = []
    for train_idx, timetable in enumerate(result.timetables_s):
        train_results = evaluate_timetable(
            environments_by_train[train_idx],
            timetable,
            lower_controllers[train_idx],
        )
        operation_results_by_train.append(train_results)
        operation_summary_rows.extend(
            lower_result_summary_rows(train_results, train_configs[train_idx].name)
        )
        stem = f"optimized_train_{train_idx + 1:02d}_operation"
        plot_train_operation(profile, train_results, output_dir, stem)
        plot_section_operation_overview(profile, train_results, output_dir, f"optimized_train_{train_idx + 1:02d}")
    write_csv(output_dir / "optimized_train_operation_summary.csv", operation_summary_rows)
    plot_multi_train_timetable_result(upper_environment, result, output_dir)
    plot_multi_train_operation_synchronization(
        profile,
        operation_results_by_train,
        train_configs,
        upper_environment,
        result,
        output_dir,
    )
    export_mappo_iteration_history(upper_history, upper_environment, output_dir)
    plot_training_process(lower_history, upper_history, output_dir)
    plot_response_tables(tables_by_train, output_dir)
    matching_summary = plot_regenerative_energy_matching_by_zone(
        tables_by_train,
        result,
        upper_environment.power_bin_s,
        upper_environment.supply_zone_size,
        upper_environment.supply_zone_section_counts,
        [
            [SECTION_NAMES[idx] for idx in group]
            for group in upper_environment.supply_zone_section_groups
        ],
        upper_environment.supply_zone_names,
        upper_environment.base_station_dwell_after_section_s,
        upper_environment.station_delay_after_section_s,
        output_dir,
    )
    write_csv(
        output_dir / "multi_train_experiment_summary.csv",
        [
            {
                "train_count": len(result.timetables_s),
                "upper_control_mode": upper_environment.control_mode,
                "initial_headway_s": upper_environment.initial_headway_s,
                "station_dwell_s": upper_environment.station_dwell_s,
                "initial_grid_energy_kwh": result.initial_grid_energy_kwh,
                "optimized_grid_energy_kwh": result.grid_energy_kwh,
                "grid_energy_saving_kwh": result.initial_grid_energy_kwh - result.grid_energy_kwh,
                "initial_matched_regenerative_kwh": result.initial_matched_regenerative_kwh,
                "optimized_matched_regenerative_kwh": result.matched_regenerative_kwh,
                "optimized_regenerative_utilization": result.regenerative_utilization,
                "traction_energy_weight": upper_environment.traction_energy_weight,
                "regenerative_match_weight": upper_environment.regenerative_match_weight,
                "supply_zone_size": upper_environment.supply_zone_size,
                "supply_zone_count": upper_environment.supply_zone_count,
                "supply_zone_section_counts": ",".join(
                    str(value) for value in upper_environment.supply_zone_section_counts
                ),
                "supply_zone_names": ";".join(upper_environment.supply_zone_names),
                "supply_zone_section_groups": ";".join(
                    ",".join(SECTION_NAMES[idx] for idx in group)
                    for group in upper_environment.supply_zone_section_groups
                ),
                **matching_summary,
                "initial_objective": result.initial_objective,
                "optimized_objective": result.objective,
            }
        ],
    )
    print(f"Saved four-train MAPPO visual reports and CSV data: {output_dir}")
