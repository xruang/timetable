"""Four-train MAPPO timetable coordination environment and trainer."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from .config import DEFAULT_STATION_DWELL_S, EPS, SECTION_NAMES
    from .lower_control import DQNAgent, SectionResponseTable
except ImportError:
    from config import DEFAULT_STATION_DWELL_S, EPS, SECTION_NAMES
    from lower_control import DQNAgent, SectionResponseTable


class MultiTrainTimetableEnvironment:
    """Upper environment: each train is one timetable-planning agent."""

    def __init__(
        self,
        initial_timetable: np.ndarray,
        response_tables_by_train: Sequence[Sequence[SectionResponseTable]],
        train_count: int = 4,
        initial_headway_s: float = 480.0,
        minimum_headway_s: float = 120.0,
        time_span_s: float = 10.0,
        time_quantum_s: float = 2.0,
        departure_span_s: float = 60.0,
        departure_quantum_s: float = 5.0,
        traction_energy_weight: float = 0.5,
        regenerative_match_weight: float = 0.5,
        control_mode: str = "departure",
        max_steps: int = 24,
        power_bin_s: float = 5.0,
        supply_zone_size: int = 0,
        supply_zone_section_counts: Optional[Sequence[int]] = None,
        supply_zone_section_groups: Optional[Sequence[Sequence[str]]] = None,
        supply_zone_names: Optional[Sequence[str]] = None,
        station_dwell_s: float = DEFAULT_STATION_DWELL_S,
        station_delay_after_section_s: Optional[Sequence[Sequence[float]]] = None,
        initial_timetables_by_train: Optional[Sequence[Sequence[float]]] = None,
        base_departures_s: Optional[Sequence[float]] = None,
        direction_groups: Optional[Sequence[Sequence[int]]] = None,
        physical_section_names_by_train: Optional[Sequence[Sequence[str]]] = None,
    ) -> None:
        if control_mode not in {"departure", "joint"}:
            raise ValueError(f"Unsupported upper control mode: {control_mode}")
        self.initial_timetable = initial_timetable.astype(np.float32)
        self.train_count = max(2, int(train_count))
        self.section_count = len(self.initial_timetable)
        self.initial_timetables = self._normalise_initial_timetables(
            initial_timetables_by_train
        )
        self.response_tables_by_train = [list(tables) for tables in response_tables_by_train]
        if len(self.response_tables_by_train) != self.train_count:
            raise ValueError("response_tables_by_train must contain one table group per train")
        if any(len(tables) != self.section_count for tables in self.response_tables_by_train):
            raise ValueError("Each train must have one response table per interstation section")
        self.initial_headway_s = float(initial_headway_s)
        self.minimum_headway_s = float(minimum_headway_s)
        self.time_span_s = float(time_span_s)
        self.time_quantum_s = float(time_quantum_s)
        self.departure_span_s = float(departure_span_s)
        self.departure_quantum_s = float(departure_quantum_s)
        self.traction_energy_weight = float(traction_energy_weight)
        self.regenerative_match_weight = float(regenerative_match_weight)
        self.reference_traction_kwh: Optional[float] = None
        self.control_mode = control_mode
        self.optimize_running_times = control_mode == "joint"
        self.max_steps = int(max_steps)
        self.power_bin_s = float(power_bin_s)
        self.supply_zone_size = max(0, int(supply_zone_size))
        self.supply_zone_section_counts = self._normalise_supply_zone_section_counts(
            supply_zone_section_counts
        )
        (
            self.supply_zone_section_groups,
            self.supply_zone_names,
            self._section_to_supply_zone,
        ) = self._normalise_supply_zone_section_groups(
            supply_zone_section_groups,
            supply_zone_names,
        )
        self.station_delay_after_section_s = self._normalise_station_delays(
            station_delay_after_section_s
        )
        self.station_dwell_s = max(0.0, float(station_dwell_s))
        self.base_station_dwell_after_section_s = self._base_station_dwell_after_sections()
        self.direction_groups = self._normalise_direction_groups(direction_groups)
        self._direction_membership = {
            train_idx: tuple(group)
            for group in self.direction_groups
            for train_idx in group
        }
        self.base_departures_s = self._normalise_base_departures(base_departures_s)
        self.physical_section_names_by_train = self._normalise_physical_section_names(
            physical_section_names_by_train
        )
        self.min_times = self.initial_timetables - self.time_span_s
        self.max_times = self.initial_timetables + self.time_span_s
        self.action_dim = self.section_count + 1 if self.optimize_running_times else 1
        self.local_observation_dim = self.section_count * 3 + 4
        self.central_state_dim = self.train_count * self.local_observation_dim
        self.reset()

    def _normalise_initial_timetables(
        self,
        timetables: Optional[Sequence[Sequence[float]]],
    ) -> np.ndarray:
        if timetables is None:
            return np.tile(self.initial_timetable, (self.train_count, 1)).astype(np.float32)
        array = np.asarray(timetables, dtype=np.float32)
        expected = (self.train_count, self.section_count)
        if array.shape != expected:
            raise ValueError(
                f"initial_timetables_by_train must have shape {expected}, got {array.shape}"
            )
        return array

    def _normalise_direction_groups(
        self,
        groups: Optional[Sequence[Sequence[int]]],
    ) -> Tuple[Tuple[int, ...], ...]:
        if groups is None:
            return (tuple(range(self.train_count)),)
        normalised = tuple(tuple(int(idx) for idx in group) for group in groups)
        flattened = [idx for group in normalised for idx in group]
        if (
            any(not group for group in normalised)
            or sorted(flattened) != list(range(self.train_count))
            or len(set(flattened)) != self.train_count
        ):
            raise ValueError(
                "direction_groups must partition all train indices exactly once"
            )
        return normalised

    def _normalise_base_departures(
        self,
        departures: Optional[Sequence[float]],
    ) -> np.ndarray:
        if departures is None:
            return (
                np.arange(self.train_count, dtype=np.float32)
                * self.initial_headway_s
            )
        array = np.asarray(departures, dtype=np.float32)
        if array.shape != (self.train_count,):
            raise ValueError(
                "base_departures_s must contain one departure time per train"
            )
        return array

    def _normalise_physical_section_names(
        self,
        names: Optional[Sequence[Sequence[str]]],
    ) -> Tuple[Tuple[str, ...], ...]:
        if names is None:
            default = tuple(SECTION_NAMES[: self.section_count])
            return tuple(default for _ in range(self.train_count))
        normalised = tuple(tuple(str(name) for name in row) for row in names)
        expected = (self.train_count, self.section_count)
        if (
            len(normalised) != expected[0]
            or any(len(row) != expected[1] for row in normalised)
        ):
            raise ValueError(
                f"physical_section_names_by_train must have shape {expected}"
            )
        unknown = sorted(
            {
                name
                for row in normalised
                for name in row
                if name not in SECTION_NAMES
            }
        )
        if unknown:
            raise ValueError(f"Unknown physical section names: {unknown}")
        return normalised

    def _base_station_dwell_after_sections(self) -> np.ndarray:
        dwell = np.full((self.train_count, self.section_count), self.station_dwell_s, dtype=np.float32)
        if self.section_count > 0:
            dwell[:, -1] = 0.0
        return dwell

    def total_station_dwell_after_section_s(self) -> np.ndarray:
        return self.base_station_dwell_after_section_s + self.station_delay_after_section_s

    def _normalise_station_delays(
        self,
        delays: Optional[Sequence[Sequence[float]]],
    ) -> np.ndarray:
        if delays is None:
            return np.zeros((self.train_count, self.section_count), dtype=np.float32)
        array = np.asarray(delays, dtype=np.float32)
        if array.shape != (self.train_count, self.section_count):
            raise ValueError(
                "station_delay_after_section_s must have shape "
                f"{(self.train_count, self.section_count)}, got {array.shape}"
            )
        if np.any(array < 0.0):
            raise ValueError("station_delay_after_section_s cannot contain negative values")
        return array

    def _normalise_supply_zone_section_counts(
        self,
        counts: Optional[Sequence[int]],
    ) -> Tuple[int, ...]:
        if counts is None:
            return ()
        normalised = tuple(int(value) for value in counts if int(value) > 0)
        if not normalised:
            return ()
        if sum(normalised) != self.section_count:
            raise ValueError(
                "supply_zone_section_counts must sum to the number of interstation "
                f"sections ({self.section_count}), got {sum(normalised)} from {normalised}"
            )
        return normalised

    def _normalise_supply_zone_section_groups(
        self,
        groups: Optional[Sequence[Sequence[str]]],
        names: Optional[Sequence[str]],
    ) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[str, ...], np.ndarray]:
        section_to_zone = np.full(self.section_count, -1, dtype=int)
        if not groups:
            return (), (), section_to_zone

        section_index = {name: idx for idx, name in enumerate(SECTION_NAMES)}
        normalised_groups: List[Tuple[int, ...]] = []
        for zone_idx, group in enumerate(groups):
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
            normalised_groups.append(tuple(indices))

        if names is None:
            normalised_names = tuple(
                f"zone_{idx + 1:02d}_{SECTION_NAMES[group[0]]}__{SECTION_NAMES[group[-1]]}"
                for idx, group in enumerate(normalised_groups)
            )
        else:
            normalised_names = tuple(str(name) for name in names)
            if len(normalised_names) != len(normalised_groups):
                raise ValueError("supply_zone_names must match supply_zone_section_groups")
        return tuple(normalised_groups), normalised_names, section_to_zone

    @property
    def supply_zone_count(self) -> int:
        if self.supply_zone_section_groups:
            return len(self.supply_zone_section_groups)
        if self.supply_zone_section_counts:
            return len(self.supply_zone_section_counts)
        if self.supply_zone_size <= 0:
            return 1
        return int(math.ceil(self.section_count / self.supply_zone_size))

    def supply_zone_id(self, section_idx: int, train_idx: Optional[int] = None) -> int:
        if train_idx is not None and self.supply_zone_section_groups:
            physical_name = self.physical_section_names_by_train[int(train_idx)][
                int(section_idx)
            ]
            physical_idx = SECTION_NAMES.index(physical_name)
            return int(self._section_to_supply_zone[physical_idx])
        if self.supply_zone_section_groups:
            return int(self._section_to_supply_zone[int(section_idx)])
        if self.supply_zone_section_counts:
            boundary = 0
            for zone_idx, count in enumerate(self.supply_zone_section_counts):
                boundary += count
                if section_idx < boundary:
                    return zone_idx
            return len(self.supply_zone_section_counts) - 1
        if self.supply_zone_size <= 0:
            return 0
        return int(section_idx) // self.supply_zone_size

    def supply_zone_name(self, zone_idx: int) -> str:
        if self.supply_zone_section_groups:
            return self.supply_zone_names[zone_idx]
        if self.supply_zone_section_counts:
            start_idx = sum(self.supply_zone_section_counts[:zone_idx])
            end_idx = start_idx + self.supply_zone_section_counts[zone_idx]
        elif self.supply_zone_size <= 0:
            return "global"
        else:
            start_idx = zone_idx * self.supply_zone_size
            end_idx = min(self.section_count, start_idx + self.supply_zone_size)
        return f"zone_{zone_idx + 1:02d}_{SECTION_NAMES[start_idx]}__{SECTION_NAMES[end_idx - 1]}"

    @property
    def active_agent_mask(self) -> np.ndarray:
        if self.optimize_running_times:
            return np.ones(self.train_count, dtype=np.float32)
        mask = np.ones(self.train_count, dtype=np.float32)
        for group in self.direction_groups:
            mask[group[0]] = 0.0
        return mask

    @property
    def departure_agent_mask(self) -> np.ndarray:
        mask = np.ones(self.train_count, dtype=np.float32)
        for group in self.direction_groups:
            mask[group[0]] = 0.0
        return mask

    @property
    def departure_times_s(self) -> np.ndarray:
        return self.base_departures_s + self.departure_offsets_s

    def reset(self) -> np.ndarray:
        self.timetables = self.initial_timetables.copy()
        self.departure_offsets_s = np.zeros(self.train_count, dtype=np.float32)
        self.steps = 0
        self.points = self._lookup_points()
        self.metrics = self._calculate_metrics()
        if self.reference_traction_kwh is None:
            self.reference_traction_kwh = max(
                self.metrics["traction_kwh"],
                EPS,
            )
            self.metrics = self._calculate_metrics()
        self.objective = self.metrics["objective"]
        self.episode_initial_objective = self.objective
        return self.local_observations()

    def _lookup_points(self):
        return [
            [
                table.lookup(float(target))
                for table, target in zip(self.response_tables_by_train[train_idx], timetable)
            ]
            for train_idx, timetable in enumerate(self.timetables)
        ]

    def _project_timetable(self, train_idx: int, timetable: np.ndarray) -> np.ndarray:
        desired = np.clip(
            timetable.astype(float),
            self.min_times[train_idx],
            self.max_times[train_idx],
        )
        candidate_times: List[List[float]] = []
        for section_idx, table in enumerate(self.response_tables_by_train[train_idx]):
            feasible_times = sorted(
                {
                    float(point.requested_time_s)
                    for point in table.points
                    if point.feasible
                    and self.min_times[train_idx, section_idx]
                    <= point.requested_time_s
                    <= self.max_times[train_idx, section_idx]
                }
            )
            if not feasible_times:
                raise ValueError(
                    f"No feasible lower-level response for train {train_idx + 1}, "
                    f"section {table.section_name}"
                )
            candidate_times.append(feasible_times)

        # Response tables are discrete. Dynamic programming finds the nearest
        # feasible combination while preserving route running time whenever possible.
        target_total_ms = int(
            round(float(np.sum(self.initial_timetables[train_idx])) * 1000.0)
        )
        states: Dict[int, Tuple[float, List[float]]] = {0: (0.0, [])}
        for section_idx, section_candidates in enumerate(candidate_times):
            next_states: Dict[int, Tuple[float, List[float]]] = {}
            for total_ms, (cost, selected) in states.items():
                for candidate in section_candidates:
                    candidate_ms = int(round(candidate * 1000.0))
                    candidate_cost = cost + (candidate - desired[section_idx]) ** 2
                    next_total_ms = total_ms + candidate_ms
                    current = next_states.get(next_total_ms)
                    if current is None or candidate_cost < current[0]:
                        next_states[next_total_ms] = (
                            candidate_cost,
                            selected + [candidate],
                        )
            states = next_states
        _, (_, projected) = min(
            states.items(),
            key=lambda item: (abs(item[0] - target_total_ms), item[1][0]),
        )
        return np.asarray(projected, dtype=np.float32)

    def station_times(self) -> np.ndarray:
        station_times = np.zeros((self.train_count, self.section_count + 1), dtype=float)
        station_times[:, 0] = self.departure_times_s
        total_dwell = self.total_station_dwell_after_section_s()
        for train_idx in range(self.train_count):
            current = float(self.departure_times_s[train_idx])
            for section_idx in range(self.section_count):
                current += float(self.timetables[train_idx, section_idx])
                station_times[train_idx, section_idx + 1] = current
                current += float(total_dwell[train_idx, section_idx])
        return station_times

    def section_start_times(self) -> np.ndarray:
        section_starts = np.zeros((self.train_count, self.section_count), dtype=float)
        total_dwell = self.total_station_dwell_after_section_s()
        for train_idx in range(self.train_count):
            current = float(self.departure_times_s[train_idx])
            for section_idx in range(self.section_count):
                section_starts[train_idx, section_idx] = current
                current += (
                    float(self.timetables[train_idx, section_idx])
                    + float(total_dwell[train_idx, section_idx])
                )
        return section_starts

    def station_departure_times(self) -> np.ndarray:
        departures = np.zeros((self.train_count, self.section_count), dtype=float)
        departures[:, 0] = self.departure_times_s
        arrivals = self.station_times()
        total_dwell = self.total_station_dwell_after_section_s()
        if self.section_count > 1:
            departures[:, 1:] = (
                arrivals[:, 1:-1]
                + total_dwell[:, :-1]
            )
        return departures

    def _matched_regenerative_energy_kwh(self) -> float:
        section_starts = self.section_start_times()
        horizon_s = float(np.max(section_starts[:, -1] + self.timetables[:, -1])) + self.power_bin_s
        bin_count = max(1, int(math.ceil(horizon_s / self.power_bin_s)))
        traction_energy = np.zeros((self.supply_zone_count, bin_count), dtype=float)
        regenerative_energy = np.zeros((self.supply_zone_count, bin_count), dtype=float)

        def distribute_profile(
            target: np.ndarray,
            supply_zone_idx: int,
            profile: Sequence[float],
            source_bin_s: float,
            source_duration_s: float,
            scheduled_start_s: float,
            scheduled_duration_s: float,
        ) -> None:
            time_scale = scheduled_duration_s / max(source_duration_s, EPS)
            for profile_idx, energy_kwh in enumerate(profile):
                if energy_kwh <= 0.0:
                    continue
                source_start = profile_idx * source_bin_s
                source_end = min(source_duration_s, source_start + source_bin_s)
                local_duration = max(EPS, (source_end - source_start) * time_scale)
                event_start = scheduled_start_s + source_start * time_scale
                event_end = event_start + local_duration
                first_bin = max(0, int(math.floor(event_start / self.power_bin_s)))
                final_bin = min(bin_count, int(math.ceil(event_end / self.power_bin_s)))
                for bin_idx in range(first_bin, final_bin):
                    bin_start = bin_idx * self.power_bin_s
                    bin_end = bin_start + self.power_bin_s
                    overlap = max(0.0, min(event_end, bin_end) - max(event_start, bin_start))
                    if overlap > 0.0:
                        target[supply_zone_idx, bin_idx] += energy_kwh * overlap / local_duration

        for train_idx in range(self.train_count):
            for section_idx, point in enumerate(self.points[train_idx]):
                start_s = float(section_starts[train_idx, section_idx])
                duration_s = max(float(self.timetables[train_idx, section_idx]), self.power_bin_s)
                supply_zone_idx = self.supply_zone_id(section_idx, train_idx)
                if supply_zone_idx < 0:
                    continue
                distribute_profile(
                    traction_energy,
                    supply_zone_idx,
                    point.traction_profile_kwh,
                    point.profile_bin_s,
                    point.actual_time_s,
                    start_s,
                    duration_s,
                )
                distribute_profile(
                    regenerative_energy,
                    supply_zone_idx,
                    point.regenerative_profile_kwh,
                    point.profile_bin_s,
                    point.actual_time_s,
                    start_s,
                    duration_s,
                )
        return float(np.sum(np.minimum(traction_energy, regenerative_energy)))

    def _calculate_metrics(self) -> Dict[str, float]:
        points = [point for train_points in self.points for point in train_points]
        traction_kwh = sum(point.traction_kwh for point in points)
        regenerative_kwh = sum(point.regenerative_kwh for point in points)
        matched_regenerative_kwh = self._matched_regenerative_energy_kwh()
        grid_energy_kwh = traction_kwh - 0.95 * matched_regenerative_kwh
        traction_energy_ratio = traction_kwh / max(
            self.reference_traction_kwh or traction_kwh,
            EPS,
        )
        regenerative_utilization = matched_regenerative_kwh / max(
            regenerative_kwh,
            EPS,
        )
        normalized_objective = (
            self.traction_energy_weight * traction_energy_ratio
            - self.regenerative_match_weight * regenerative_utilization
        )
        timetable_error_s = sum(abs(point.timetable_error_s) for point in points)
        infeasible_sections = sum(not point.feasible for point in points)
        station_times = self.station_times()
        station_departures = self.station_departure_times()
        headway_violation_s = 0.0
        for group in self.direction_groups:
            if len(group) < 2:
                continue
            indices = np.asarray(group, dtype=int)
            headways = np.diff(station_times[indices], axis=0)
            departure_headways = np.diff(station_departures[indices], axis=0)
            headway_violation_s += float(
                np.sum(np.maximum(0.0, self.minimum_headway_s - headways))
                + np.sum(
                    np.maximum(
                        0.0,
                        self.minimum_headway_s - departure_headways,
                    )
                )
            )
        total_extra_station_delay_s = float(np.sum(self.station_delay_after_section_s))
        total_base_station_dwell_s = float(np.sum(self.base_station_dwell_after_section_s))
        total_station_dwell_s = float(np.sum(self.total_station_dwell_after_section_s()))
        return {
            "objective": float(
                normalized_objective
            ),
            "objective_energy_kwh": float(normalized_objective),
            "normalized_objective": float(normalized_objective),
            "traction_energy_ratio": float(traction_energy_ratio),
            "grid_energy_kwh": float(grid_energy_kwh),
            "traction_kwh": float(traction_kwh),
            "regenerative_kwh": float(regenerative_kwh),
            "matched_regenerative_kwh": float(matched_regenerative_kwh),
            "regenerative_utilization": float(regenerative_utilization),
            "traction_energy_weight": float(self.traction_energy_weight),
            "regenerative_match_weight": float(self.regenerative_match_weight),
            "supply_zone_size": float(self.supply_zone_size),
            "supply_zone_count": float(self.supply_zone_count),
            "supply_zone_section_counts": ",".join(str(value) for value in self.supply_zone_section_counts),
            "supply_zone_names": ";".join(self.supply_zone_names),
            "supply_zone_section_groups": ";".join(
                ",".join(SECTION_NAMES[idx] for idx in group)
                for group in self.supply_zone_section_groups
            ),
            "timetable_error_s": float(timetable_error_s),
            "infeasible_sections": float(infeasible_sections),
            "headway_violation_s": float(headway_violation_s),
            "station_dwell_s": float(self.station_dwell_s),
            "total_base_station_dwell_s": total_base_station_dwell_s,
            "total_extra_station_delay_s": total_extra_station_delay_s,
            "total_station_dwell_s": total_station_dwell_s,
        }

    def local_observations(self) -> np.ndarray:
        observations: List[np.ndarray] = []
        departures = self.departure_times_s
        for train_idx in range(self.train_count):
            points = self.points[train_idx]
            offset = (
                self.timetables[train_idx] - self.initial_timetables[train_idx]
            ) / max(self.time_span_s, 1)
            energy = np.asarray([point.net_energy_kwh for point in points], dtype=np.float32) / 100.0
            error = np.asarray([point.timetable_error_s for point in points], dtype=np.float32) / 20.0
            group = self._direction_membership[train_idx]
            group_position = group.index(train_idx)
            previous_headway = (
                self.initial_headway_s
                if group_position == 0
                else departures[train_idx] - departures[group[group_position - 1]]
            )
            next_headway = (
                self.initial_headway_s
                if group_position == len(group) - 1
                else departures[group[group_position + 1]] - departures[train_idx]
            )
            interaction = np.asarray(
                [
                    self.departure_offsets_s[train_idx] / max(self.departure_span_s, 1.0),
                    previous_headway / self.initial_headway_s,
                    next_headway / self.initial_headway_s,
                    self.metrics["regenerative_utilization"],
                ],
                dtype=np.float32,
            )
            observations.append(np.concatenate((offset, energy, error, interaction)))
        return np.asarray(observations, dtype=np.float32)

    def central_state(self) -> np.ndarray:
        return self.local_observations().reshape(-1)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.train_count, self.action_dim):
            raise ValueError(f"Expected upper actions {(self.train_count, self.action_dim)}, got {actions.shape}")

        previous_timetables = self.timetables.copy()
        previous_departure_offsets = self.departure_offsets_s.copy()
        previous_points = self.points
        previous_metrics = dict(self.metrics)
        previous_objective = self.objective
        for train_idx in range(self.train_count):
            if self.optimize_running_times:
                section_delta = np.tanh(actions[train_idx, : self.section_count])
                section_delta -= float(np.mean(section_delta))
                self.timetables[train_idx] = self._project_timetable(
                    train_idx,
                    self.timetables[train_idx] + self.time_quantum_s * section_delta
                )
            if self.departure_agent_mask[train_idx] > 0.0:
                offset_delta = self.departure_quantum_s * math.tanh(float(actions[train_idx, -1]))
                self.departure_offsets_s[train_idx] = float(
                    np.clip(
                        self.departure_offsets_s[train_idx] + offset_delta,
                        -self.departure_span_s,
                        self.departure_span_s,
                    )
                )

        self.points = self._lookup_points()
        self.metrics = self._calculate_metrics()
        self.objective = self.metrics["objective"]
        action_rejected = bool(
            self.metrics["infeasible_sections"] > 0.0
            or self.metrics["headway_violation_s"] > 0.0
        )
        if action_rejected:
            self.timetables = previous_timetables
            self.departure_offsets_s = previous_departure_offsets
            self.points = previous_points
            self.metrics = previous_metrics
            self.objective = previous_objective
        self.steps += 1
        done = self.steps >= self.max_steps
        traction_reward_component = (
            self.traction_energy_weight
            * (
                previous_metrics["traction_kwh"]
                - self.metrics["traction_kwh"]
            )
            / max(self.reference_traction_kwh or 0.0, EPS)
        )
        regenerative_reward_component = (
            self.regenerative_match_weight
            * (
                self.metrics["regenerative_utilization"]
                - previous_metrics["regenerative_utilization"]
            )
        )
        reward = traction_reward_component + regenerative_reward_component
        info = dict(self.metrics)
        info.update(
            {
                "reward_traction_energy_component": float(
                    traction_reward_component
                ),
                "reward_regenerative_energy_component": float(
                    regenerative_reward_component
                ),
                "action_rejected": float(action_rejected),
            }
        )
        return (
            self.local_observations(),
            float(reward),
            done,
            info,
        )


class PPOActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.00))

    def distribution(self, observation: torch.Tensor):
        mean = self.layers(observation)
        return torch.distributions.Normal(mean, self.log_std.clamp(-3.0, 1.0).exp().expand_as(mean))


class PPOCentralCritic(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class IndependentActorMAPPOAgent:
    """Independent train actors for decentralized execution with one centralized critic."""

    def __init__(
        self,
        train_count: int,
        observation_dim: int,
        action_dim: int,
        central_state_dim: int,
        active_agent_mask: Sequence[float] | None = None,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.20,
        update_epochs: int = 8,
        entropy_coefficient: float = 0.0002,
        device_name: Optional[str] = None,
    ) -> None:
        self.train_count = train_count
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.entropy_coefficient = entropy_coefficient
        if device_name is None or device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        if active_agent_mask is None:
            active_agent_mask = [1.0] * train_count
        if len(active_agent_mask) != train_count:
            raise ValueError("active_agent_mask must match train_count")
        self.active_agent_mask = torch.as_tensor(
            active_agent_mask,
            dtype=torch.float32,
            device=self.device,
        )
        self.actors = nn.ModuleList(
            [PPOActor(observation_dim, action_dim) for _ in range(train_count)]
        ).to(self.device)
        self.critic = PPOCentralCritic(central_state_dim).to(self.device)
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=actor_learning_rate)
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def select_actions(
        self,
        local_observations: np.ndarray,
        central_state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        with torch.no_grad():
            observations = torch.as_tensor(local_observations, dtype=torch.float32, device=self.device)
            state = torch.as_tensor(central_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions: List[np.ndarray] = []
            log_probabilities: List[float] = []
            for train_idx, actor in enumerate(self.actors):
                distribution = actor.distribution(observations[train_idx])
                action = distribution.sample()
                actions.append(action.cpu().numpy())
                log_probabilities.append(float(distribution.log_prob(action).sum().item()))
            return (
                np.asarray(actions, dtype=np.float32),
                np.asarray(log_probabilities, dtype=np.float32),
                float(self.critic(state).squeeze().item()),
            )

    def update(self, trajectory: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        observations = torch.as_tensor(np.asarray([item["observations"] for item in trajectory]), dtype=torch.float32, device=self.device)
        states = torch.as_tensor(np.asarray([item["state"] for item in trajectory]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray([item["actions"] for item in trajectory]), dtype=torch.float32, device=self.device)
        old_log_probabilities = torch.as_tensor(np.asarray([item["log_probabilities"] for item in trajectory]), dtype=torch.float32, device=self.device)
        if trajectory and "active_agent_mask" in trajectory[0]:
            active_masks = torch.as_tensor(
                np.asarray([item["active_agent_mask"] for item in trajectory]),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            active_masks = self.active_agent_mask.unsqueeze(0).expand(
                len(trajectory),
                -1,
            )
        rewards = [float(item["reward"]) for item in trajectory]
        values = [float(item["value"]) for item in trajectory]
        dones = [bool(item["done"]) for item in trajectory]

        advantages: List[float] = []
        gae = 0.0
        next_value = 0.0
        for reward, value, done in reversed(list(zip(rewards, values, dones))):
            mask = 0.0 if done else 1.0
            gae = reward + self.gamma * next_value * mask - value + self.gamma * self.gae_lambda * mask * gae
            advantages.append(gae)
            next_value = value
        advantages.reverse()
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages_tensor + torch.as_tensor(values, dtype=torch.float32, device=self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std(unbiased=False) + 1e-6)

        losses = {}
        for _ in range(self.update_epochs):
            actor_losses: List[float] = []
            entropies: List[float] = []
            policy_kls: List[float] = []
            clip_fractions: List[float] = []
            for train_idx, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
                if self.active_agent_mask[train_idx].item() <= 0.0:
                    continue
                sample_mask = active_masks[:, train_idx] * self.active_agent_mask[train_idx]
                selected = sample_mask > 0.0
                if not bool(torch.any(selected).item()):
                    continue
                distribution = actor.distribution(observations[:, train_idx, :])
                log_probabilities = distribution.log_prob(actions[:, train_idx, :]).sum(dim=1)
                ratio = torch.exp(log_probabilities - old_log_probabilities[:, train_idx])
                entropy_by_sample = distribution.entropy().sum(dim=1)
                clipped_objective = torch.min(
                    ratio * advantages_tensor,
                    torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_tensor,
                )
                actor_loss = (
                    -clipped_objective[selected].mean()
                    - self.entropy_coefficient * entropy_by_sample[selected].mean()
                )

                optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
                actor_losses.append(float(actor_loss.item()))
                entropies.append(float(entropy_by_sample[selected].mean().item()))
                policy_kls.append(
                    float(
                        ((ratio - 1.0) - torch.log(ratio + EPS))[selected]
                        .mean()
                        .item()
                    )
                )
                clip_fractions.append(
                    float(
                        ((ratio - 1.0).abs() > self.clip_ratio)
                        .float()[selected]
                        .mean()
                        .item()
                    )
                )

            critic_values = self.critic(states).squeeze(1)
            critic_loss = F.mse_loss(critic_values, returns)
            self.critic_optimizer.zero_grad()
            (0.50 * critic_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
            with torch.no_grad():
                return_variance = torch.var(returns, unbiased=False)
                explained_variance = 1.0 - torch.var(
                    returns - critic_values,
                    unbiased=False,
                ) / (return_variance + 1e-8)
            losses = {
                "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
                "critic_loss": float(critic_loss.item()),
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                "policy_kl": float(np.mean(policy_kls)) if policy_kls else 0.0,
                "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                "value_explained_variance": float(explained_variance.item()),
            }
        return losses


# Backward-compatible import name. The implementation now uses independent actors.
SharedMAPPOAgent = IndependentActorMAPPOAgent


@dataclass
class MultiTrainTrainingResult:
    control_mode: str
    initial_headway_s: float
    timetables_s: List[List[float]]
    departure_times_s: List[float]
    objective: float
    grid_energy_kwh: float
    traction_kwh: float
    regenerative_kwh: float
    matched_regenerative_kwh: float
    regenerative_utilization: float
    supply_zone_size: int
    supply_zone_count: int
    supply_zone_section_counts: List[int]
    supply_zone_names: List[str]
    supply_zone_section_groups: List[List[str]]
    timetable_error_s: float
    infeasible_sections: float
    initial_objective: float
    initial_grid_energy_kwh: float
    initial_matched_regenerative_kwh: float


def _capture_result(environment: MultiTrainTimetableEnvironment, initial_metrics: Dict[str, float]) -> MultiTrainTrainingResult:
    return MultiTrainTrainingResult(
        control_mode=environment.control_mode,
        initial_headway_s=environment.initial_headway_s,
        timetables_s=environment.timetables.tolist(),
        departure_times_s=environment.departure_times_s.tolist(),
        objective=environment.objective,
        grid_energy_kwh=environment.metrics["grid_energy_kwh"],
        traction_kwh=environment.metrics["traction_kwh"],
        regenerative_kwh=environment.metrics["regenerative_kwh"],
        matched_regenerative_kwh=environment.metrics["matched_regenerative_kwh"],
        regenerative_utilization=environment.metrics["regenerative_utilization"],
        supply_zone_size=environment.supply_zone_size,
        supply_zone_count=environment.supply_zone_count,
        supply_zone_section_counts=list(environment.supply_zone_section_counts),
        supply_zone_names=list(environment.supply_zone_names),
        supply_zone_section_groups=[
            [SECTION_NAMES[idx] for idx in group]
            for group in environment.supply_zone_section_groups
        ],
        timetable_error_s=environment.metrics["timetable_error_s"],
        infeasible_sections=environment.metrics["infeasible_sections"],
        initial_objective=initial_metrics["objective"],
        initial_grid_energy_kwh=initial_metrics["grid_energy_kwh"],
        initial_matched_regenerative_kwh=initial_metrics["matched_regenerative_kwh"],
    )


def train_multi_train_mappo(
    environment: MultiTrainTimetableEnvironment,
    controller: IndependentActorMAPPOAgent,
    episodes: int,
) -> Tuple[MultiTrainTrainingResult, List[Dict[str, Any]]]:
    environment.reset()
    initial_metrics = dict(environment.metrics)
    best = _capture_result(environment, initial_metrics)
    history: List[Dict[str, Any]] = []
    print(
        f"Training {environment.train_count}-train upper MAPPO policy for {episodes} episodes "
        f"(mode={environment.control_mode}, "
        f"initial grid={best.grid_energy_kwh:.2f}kWh, "
        f"matched regen={best.matched_regenerative_kwh:.2f}kWh)"
    )

    for episode in range(episodes):
        observations = environment.reset()
        trajectory: List[Dict[str, Any]] = []
        episode_reward = 0.0
        episode_traction_component = 0.0
        episode_regenerative_component = 0.0
        rejected_action_count = 0.0
        done = False
        while not done:
            state = environment.central_state()
            actions, log_probabilities, value = controller.select_actions(observations, state)
            next_observations, reward, done, info = environment.step(actions)
            trajectory.append(
                {
                    "observations": observations,
                    "state": state,
                    "actions": actions,
                    "log_probabilities": log_probabilities,
                    "value": value,
                    "reward": reward,
                    "done": done,
                }
            )
            observations = next_observations
            episode_reward += reward
            episode_traction_component += info.get(
                "reward_traction_energy_component",
                0.0,
            )
            episode_regenerative_component += info.get(
                "reward_regenerative_energy_component",
                0.0,
            )
            rejected_action_count += info.get("action_rejected", 0.0)
            if environment.objective < best.objective:
                best = _capture_result(environment, initial_metrics)

        losses = controller.update(trajectory)
        current_timetables = environment.timetables.tolist()
        current_departures = environment.departure_times_s.tolist()
        current_timetable_array = np.asarray(current_timetables, dtype=float)
        best_timetable_array = np.asarray(best.timetables_s, dtype=float)
        current_departure_array = np.asarray(current_departures, dtype=float)
        best_departure_array = np.asarray(best.departure_times_s, dtype=float)
        history.append(
            {
                "episode": episode + 1,
                "control_mode": environment.control_mode,
                "reward": episode_reward,
                "objective": environment.objective,
                "normalized_objective": environment.metrics["normalized_objective"],
                "best_objective": best.objective,
                "objective_improvement": initial_metrics["objective"] - environment.objective,
                "best_objective_improvement": initial_metrics["objective"] - best.objective,
                "grid_energy_kwh": environment.metrics["grid_energy_kwh"],
                "objective_energy_kwh": environment.metrics["objective_energy_kwh"],
                "traction_kwh": environment.metrics["traction_kwh"],
                "traction_energy_ratio": environment.metrics["traction_energy_ratio"],
                "best_grid_energy_kwh": best.grid_energy_kwh,
                "matched_regenerative_kwh": environment.metrics["matched_regenerative_kwh"],
                "best_matched_regenerative_kwh": best.matched_regenerative_kwh,
                "regenerative_utilization": environment.metrics["regenerative_utilization"],
                "traction_energy_weight": environment.metrics["traction_energy_weight"],
                "regenerative_match_weight": environment.metrics["regenerative_match_weight"],
                "reward_traction_energy_component": episode_traction_component,
                "reward_regenerative_energy_component": episode_regenerative_component,
                "rejected_action_count": rejected_action_count,
                "supply_zone_size": environment.metrics["supply_zone_size"],
                "supply_zone_count": environment.metrics["supply_zone_count"],
                "supply_zone_section_counts": environment.metrics["supply_zone_section_counts"],
                "supply_zone_names": environment.metrics["supply_zone_names"],
                "supply_zone_section_groups": environment.metrics["supply_zone_section_groups"],
                "timetable_error_s": environment.metrics["timetable_error_s"],
                "infeasible_sections": environment.metrics["infeasible_sections"],
                "headway_violation_s": environment.metrics["headway_violation_s"],
                "mean_abs_timetable_change_s": float(
                    np.mean(
                        np.abs(
                            current_timetable_array
                            - environment.initial_timetables
                        )
                    )
                ),
                "best_mean_abs_timetable_change_s": float(
                    np.mean(
                        np.abs(
                            best_timetable_array
                            - environment.initial_timetables
                        )
                    )
                ),
                "mean_abs_departure_offset_s": float(
                    np.mean(np.abs(current_departure_array - environment.base_departures_s))
                ),
                "best_mean_abs_departure_offset_s": float(
                    np.mean(np.abs(best_departure_array - environment.base_departures_s))
                ),
                "actor_loss": losses["actor_loss"],
                "critic_loss": losses["critic_loss"],
                "entropy": losses["entropy"],
                "policy_kl": losses["policy_kl"],
                "clip_fraction": losses["clip_fraction"],
                "value_explained_variance": losses["value_explained_variance"],
                "current_timetables_s": current_timetables,
                "best_timetables_s": [list(row) for row in best.timetables_s],
                "current_departure_times_s": current_departures,
                "best_departure_times_s": list(best.departure_times_s),
            }
        )
        if episode == 0 or (episode + 1) % max(1, episodes // 10) == 0:
            print(
                f"  MAPPO episode {episode + 1:5d}/{episodes}: "
                f"reward={episode_reward:+10.6f} "
                f"objective={environment.objective:+10.6f} "
                f"best={best.objective:+10.6f} "
                f"grid={environment.metrics['grid_energy_kwh']:7.2f}kWh "
                f"regen_use={100.0 * environment.metrics['regenerative_utilization']:5.1f}% "
                f"headway_violation={environment.metrics['headway_violation_s']:6.1f}s"
            )
    return best, history


def save_multi_train_checkpoint(
    output_path: Path,
    lower_controllers: Sequence[DQNAgent],
    upper_controller: IndependentActorMAPPOAgent,
    result: MultiTrainTrainingResult,
    train_parameters: Sequence[Dict[str, Any]],
    driving_smoothness_parameters: Dict[str, Any],
    lower_algorithm: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lower_controller_states = []
    for controller in lower_controllers:
        state: Dict[str, Any] = {
            "policy": controller.policy.state_dict(),
        }
        for attribute in (
            "target",
            "actor",
            "target_actor",
            "critic",
            "target_critic",
            "critic1",
            "critic2",
            "target_critic1",
            "target_critic2",
            "optimizer",
            "actor_optimizer",
            "critic_optimizer",
            "critic1_optimizer",
            "critic2_optimizer",
            "alpha_optimizer",
        ):
            item = getattr(controller, attribute, None)
            if item is not None and hasattr(item, "state_dict"):
                state[attribute] = item.state_dict()
        log_alpha = getattr(controller, "log_alpha", None)
        if log_alpha is not None:
            state["log_alpha"] = log_alpha.detach().cpu()
        if hasattr(controller, "update_count"):
            state["update_count"] = int(controller.update_count)
        lower_controller_states.append(state)
    torch.save(
        {
            "algorithm": "independent_actor_mappo_ctde",
            "lower_algorithm": lower_algorithm or type(lower_controllers[0]).__name__,
            "lower_policies": [controller.policy.state_dict() for controller in lower_controllers],
            "lower_controller_states": lower_controller_states,
            "upper_actors": [actor.state_dict() for actor in upper_controller.actors],
            "upper_critic": upper_controller.critic.state_dict(),
            "upper_actor_optimizers": [
                optimizer.state_dict()
                for optimizer in upper_controller.actor_optimizers
            ],
            "upper_critic_optimizer": upper_controller.critic_optimizer.state_dict(),
            "upper_result": asdict(result),
            "train_parameters": list(train_parameters),
            "driving_smoothness_parameters": dict(driving_smoothness_parameters),
        },
        output_path,
    )
    print(f"Saved four-train MAPPO checkpoint: {output_path}")
