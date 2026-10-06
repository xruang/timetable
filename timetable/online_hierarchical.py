"""Online hierarchical timetable environment.

The lower DQN is called at every station-to-station decision instead of being
compressed into an offline E_min(T) response table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .config import EPS, SECTION_NAMES
    from .lower_control import DQNAgent, LowerResult, LowerSectionEnvironment
    from .mappo import IndependentActorMAPPOAgent, MultiTrainTrainingResult
except ImportError:
    from config import EPS, SECTION_NAMES
    from lower_control import DQNAgent, LowerResult, LowerSectionEnvironment
    from mappo import IndependentActorMAPPOAgent, MultiTrainTrainingResult


@dataclass
class OnlineStepRecord:
    step: int
    train_index: int
    section_index: int
    section_name: str
    physical_section_name: str
    departure_time_s: float
    arrival_time_s: float
    requested_target_runtime_s: float
    target_runtime_s: float
    actual_runtime_s: float
    dwell_adjustment_s: float
    runtime_adjustment_s: float
    traction_kwh: float
    regenerative_kwh: float
    feasible: bool
    safety_filtered: bool
    speed_limit_action_shield_overrides: int


class RollingProgressPrinter:
    def __init__(self, interval: int = 10, keep_lines: int = 10) -> None:
        self.interval = max(0, int(interval))
        self.keep_lines = max(1, int(keep_lines))
        self.buffered_lines = 0

    def maybe_print(
        self,
        episode: int,
        episodes: int,
        event_step: int,
        total_event_steps: int,
        record: OnlineStepRecord,
        metrics: Dict[str, float],
    ) -> None:
        if self.interval <= 0:
            return
        if event_step % self.interval != 0 and event_step != total_event_steps:
            return
        if self.buffered_lines >= self.keep_lines:
            print(f"\033[{self.buffered_lines}A", end="")
            for _ in range(self.buffered_lines):
                print("\033[2K\033[1B", end="")
            print(f"\033[{self.buffered_lines}A", end="")
            self.buffered_lines = 0
        print(
            f"  ep {episode:4d}/{episodes:<4d} "
            f"event {event_step:4d}/{total_event_steps:<4d} "
            f"train {record.train_index + 1:02d} "
            f"sec {record.section_index + 1:02d} {record.section_name:<10s} "
            f"T={record.target_runtime_s:6.1f}/{record.actual_runtime_s:6.1f}s "
            f"Etr={metrics['traction_kwh']:8.1f}kWh "
            f"regen={100.0 * metrics['regenerative_utilization']:5.1f}% "
            f"filtered={int(record.safety_filtered)} "
            f"speed_shield={record.speed_limit_action_shield_overrides}"
        )
        self.buffered_lines += 1


class LowerRolloutCache:
    """Cache policy rollouts or model-based lower optimization results."""

    def __init__(self, time_quantum_s: float = 0.1) -> None:
        self.time_quantum_s = max(float(time_quantum_s), 1e-6)
        self.items: Dict[Tuple[Any, str, float], LowerResult] = {}
        self.failures: Dict[Tuple[Any, str, float], str] = {}
        self.hits = 0
        self.misses = 0

    def _quantize_time(self, target_time_s: float) -> float:
        return round(round(float(target_time_s) / self.time_quantum_s) * self.time_quantum_s, 6)

    @staticmethod
    def _train_signature(section_env: LowerSectionEnvironment) -> Tuple[float, ...]:
        cfg = section_env.dynamics.config
        return (
            float(cfg.mass_kg),
            float(cfg.max_traction_force_n),
            float(cfg.max_braking_force_n),
            float(cfg.max_traction_power_w),
            float(cfg.davis_a),
            float(cfg.davis_b),
            float(cfg.davis_c),
            float(cfg.traction_efficiency),
            float(cfg.regenerative_efficiency),
        )

    def get(
        self,
        train_idx: int,
        section_idx: int,
        section_env: LowerSectionEnvironment,
        controller: Optional[DQNAgent],
        target_time_s: float,
        model_solver: Optional[Any] = None,
    ) -> Tuple[LowerResult, float, bool]:
        quantized_time = self._quantize_time(target_time_s)
        key = (
            self._train_signature(section_env),
            section_env.section.name,
            quantized_time,
        )
        cached = self.items.get(key)
        if cached is not None:
            self.hits += 1
            return cached, quantized_time, True
        if key in self.failures:
            self.hits += 1
            raise RuntimeError(self.failures[key])
        self.misses += 1
        if model_solver is None:
            if controller is None:
                raise ValueError("A lower policy controller or model solver is required")
            result = section_env.rollout(
                controller,
                quantized_time,
                training=False,
            )
        else:
            optimized = model_solver.solve_environment(section_env, quantized_time)
            if not optimized.has_solution:
                message = (
                    f"Model-based lower solve failed for {section_env.section.name} "
                    f"at {quantized_time:.3f}s: {optimized.status}; "
                    + "; ".join(optimized.validation_messages)
                )
                self.failures[key] = message
                raise RuntimeError(message)
            result = model_solver.to_lower_result(
                optimized,
                section_env.section,
                section_env.dynamics.config,
            )
        self.items[key] = result
        return result, quantized_time, False


class OnlineHierarchicalTimetableEnvironment:
    """Upper environment that invokes lower DQN rollouts during every step."""

    def __init__(
        self,
        initial_timetables_by_train: Sequence[Sequence[float]],
        environments_by_train: Sequence[Sequence[LowerSectionEnvironment]],
        lower_agents: Optional[Sequence[DQNAgent]],
        base_departures_s: Sequence[float],
        direction_groups: Sequence[Sequence[int]],
        physical_section_names_by_train: Sequence[Sequence[str]],
        supply_zone_section_groups: Sequence[Sequence[str]],
        supply_zone_names: Sequence[str],
        station_dwell_s: float,
        minimum_headway_s: float,
        runtime_adjustment_span_s: float,
        dwell_adjustment_span_s: float,
        traction_energy_weight: float,
        regenerative_match_weight: float,
        power_bin_s: float = 5.0,
        rollout_cache_quantum_s: float = 0.1,
        lower_model_solver: Optional[Any] = None,
        model_runtime_search_step_s: float = 1.0,
        model_runtime_search_max_extra_s: float = 30.0,
    ) -> None:
        self.initial_timetables = np.asarray(initial_timetables_by_train, dtype=np.float32)
        self.requested_initial_timetables = self.initial_timetables.copy()
        self.environments_by_train = [list(items) for items in environments_by_train]
        self.lower_agents = list(lower_agents or [])
        self.lower_model_solver = lower_model_solver
        self.base_departures_s = np.asarray(base_departures_s, dtype=np.float32)
        self.direction_groups = tuple(tuple(int(idx) for idx in group) for group in direction_groups)
        self.physical_section_names_by_train = tuple(
            tuple(str(name) for name in row) for row in physical_section_names_by_train
        )
        self.supply_zone_names = tuple(str(name) for name in supply_zone_names)
        self.station_dwell_s = float(station_dwell_s)
        self.minimum_headway_s = float(minimum_headway_s)
        self.runtime_adjustment_span_s = float(runtime_adjustment_span_s)
        self.dwell_adjustment_span_s = float(dwell_adjustment_span_s)
        self.traction_energy_weight = float(traction_energy_weight)
        self.regenerative_match_weight = float(regenerative_match_weight)
        self.power_bin_s = float(power_bin_s)
        self.rollout_cache = LowerRolloutCache(rollout_cache_quantum_s)
        self.model_runtime_search_step_s = max(
            float(model_runtime_search_step_s),
            self.rollout_cache.time_quantum_s,
        )
        self.model_runtime_search_max_extra_s = max(
            0.0,
            float(model_runtime_search_max_extra_s),
        )
        self.model_runtime_adjustments: List[Dict[str, Any]] = []
        self.train_count, self.section_count = self.initial_timetables.shape
        if self.lower_model_solver is None and len(self.lower_agents) != self.train_count:
            raise ValueError(
                "lower_agents must contain one controller per train when no "
                "model-based lower solver is configured"
            )
        self.action_dim = 2
        self.local_observation_dim = 12
        self.central_state_dim = self.train_count * self.local_observation_dim
        self.active_agent_mask = np.ones(self.train_count, dtype=np.float32)
        self.control_mode = (
            "online_model_based_lower_optimization"
            if self.lower_model_solver is not None
            else "online_lower_rollout"
        )
        self.supply_zone_size = 0
        self.supply_zone_section_counts: Tuple[int, ...] = ()
        self.supply_zone_section_groups = self._normalise_supply_zone_groups(
            supply_zone_section_groups
        )
        self._section_to_supply_zone = self._section_to_zone_map()
        if self.lower_model_solver is not None:
            self._calibrate_model_based_initial_runtimes()
        self.reference_traction_kwh = self._baseline_traction_kwh()
        self.reset()

    @property
    def supply_zone_count(self) -> int:
        return len(self.supply_zone_section_groups)

    @property
    def departure_times_s(self) -> np.ndarray:
        return self.departure_start_times[:, 0]

    def _normalise_supply_zone_groups(
        self,
        groups: Sequence[Sequence[str]],
    ) -> Tuple[Tuple[int, ...], ...]:
        section_index = {name: idx for idx, name in enumerate(SECTION_NAMES)}
        return tuple(
            tuple(section_index[name] for name in group)
            for group in groups
        )

    def _section_to_zone_map(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for zone_idx, group in enumerate(self.supply_zone_section_groups):
            for section_idx in group:
                mapping[SECTION_NAMES[section_idx]] = zone_idx
        return mapping

    def supply_zone_id(self, section_idx: int, train_idx: int | None = None) -> int:
        if train_idx is None:
            physical_name = SECTION_NAMES[int(section_idx)]
        else:
            physical_name = self.physical_section_names_by_train[int(train_idx)][
                int(section_idx)
            ]
        return int(self._section_to_supply_zone.get(physical_name, -1))

    def supply_zone_name(self, zone_idx: int) -> str:
        if 0 <= zone_idx < len(self.supply_zone_names):
            return self.supply_zone_names[zone_idx]
        return f"zone_{zone_idx + 1}"

    def _baseline_traction_kwh(self) -> float:
        total = 0.0
        for train_idx in range(self.train_count):
            for section_idx, section_env in enumerate(self.environments_by_train[train_idx]):
                if self.lower_model_solver is None:
                    result = section_env.rollout(
                        self.lower_agents[train_idx],
                        float(self.initial_timetables[train_idx, section_idx]),
                        training=False,
                    )
                else:
                    result, _, _ = self.rollout_cache.get(
                        train_idx,
                        section_idx,
                        section_env,
                        None,
                        float(self.initial_timetables[train_idx, section_idx]),
                        model_solver=self.lower_model_solver,
                    )
                total += result.traction_kwh
        return max(total, EPS)

    def _calibrate_model_based_initial_runtimes(self) -> None:
        """Raise only physically infeasible baseline runtimes.

        A reduced speed limit can make a timetable value that was feasible at
        the original line limit impossible.  Search upward in explicit steps
        so MAPPO starts from a lower-level-feasible timetable instead of
        aborting during reference-energy construction.
        """
        for train_idx in range(self.train_count):
            for section_idx, section_env in enumerate(
                self.environments_by_train[train_idx]
            ):
                requested = float(self.initial_timetables[train_idx, section_idx])
                attempts = int(
                    math.floor(
                        self.model_runtime_search_max_extra_s
                        / self.model_runtime_search_step_s
                    )
                )
                failures: List[str] = []
                accepted_result: Optional[LowerResult] = None
                accepted_runtime = requested
                for attempt in range(attempts + 1):
                    candidate = requested + attempt * self.model_runtime_search_step_s
                    try:
                        result, quantized_runtime, _ = self.rollout_cache.get(
                            train_idx,
                            section_idx,
                            section_env,
                            None,
                            candidate,
                            model_solver=self.lower_model_solver,
                        )
                    except RuntimeError as error:
                        failures.append(str(error))
                        continue
                    if result.strict_timetable_feasible:
                        accepted_result = result
                        accepted_runtime = quantized_runtime
                        break
                    failures.append(
                        f"{section_env.section.name} at {quantized_runtime:.3f}s "
                        "returned a trajectory that failed physical validation"
                    )
                if accepted_result is None:
                    details = " | ".join(dict.fromkeys(failures[-3:]))
                    raise RuntimeError(
                        "Could not find a feasible model-based baseline runtime "
                        f"for train {train_idx + 1}, section "
                        f"{section_env.section.name}, starting at {requested:.3f}s "
                        f"within +{self.model_runtime_search_max_extra_s:.1f}s. "
                        f"Last diagnostics: {details}"
                    )
                self.initial_timetables[train_idx, section_idx] = accepted_runtime
                if abs(accepted_runtime - requested) > 1e-9:
                    adjustment = {
                        "train_index": train_idx,
                        "section_index": section_idx,
                        "section_name": section_env.section.name,
                        "requested_runtime_s": requested,
                        "feasible_runtime_s": accepted_runtime,
                        "adjustment_s": accepted_runtime - requested,
                    }
                    self.model_runtime_adjustments.append(adjustment)
                    print(
                        "Adjusted infeasible convex baseline: "
                        f"train {train_idx + 1}, {section_env.section.name}, "
                        f"{requested:.1f}s -> {accepted_runtime:.1f}s"
                    )

    def reset(self) -> np.ndarray:
        self.current_section = np.zeros(self.train_count, dtype=np.int32)
        self.current_time_s = self.base_departures_s.astype(float).copy()
        self.timetables = self.initial_timetables.copy()
        self.departure_start_times = np.full(
            (self.train_count, self.section_count),
            np.nan,
            dtype=float,
        )
        self.arrival_times = np.full(
            (self.train_count, self.section_count + 1),
            np.nan,
            dtype=float,
        )
        self.arrival_times[:, 0] = self.base_departures_s
        self.results_by_train: List[List[LowerResult | None]] = [
            [None for _ in range(self.section_count)]
            for _ in range(self.train_count)
        ]
        self.step_records: List[OnlineStepRecord] = []
        self.traction_energy_bins = np.zeros((self.supply_zone_count, 1), dtype=float)
        self.regenerative_energy_bins = np.zeros((self.supply_zone_count, 1), dtype=float)
        self.matched_regenerative_kwh = 0.0
        self.total_traction_kwh = 0.0
        self.total_regenerative_kwh = 0.0
        self.steps = 0
        self.done = False
        self.metrics = self._calculate_metrics()
        self.objective = self.metrics["objective"]
        return self.local_observations()

    def next_event_train_idx(self) -> int:
        candidates = [
            train_idx
            for train_idx in range(self.train_count)
            if self.current_section[train_idx] < self.section_count
        ]
        if not candidates:
            return -1
        return min(candidates, key=lambda idx: (self.current_time_s[idx], idx))

    def current_active_agent_mask(self) -> np.ndarray:
        mask = np.zeros(self.train_count, dtype=np.float32)
        train_idx = self.next_event_train_idx()
        if train_idx >= 0:
            mask[train_idx] = 1.0
        return mask

    def _valid_results(self) -> List[Tuple[int, int, LowerResult]]:
        results: List[Tuple[int, int, LowerResult]] = []
        for train_idx, row in enumerate(self.results_by_train):
            for section_idx, result in enumerate(row):
                if result is not None:
                    results.append((train_idx, section_idx, result))
        return results

    def _ensure_energy_bin_capacity(self, bin_idx: int) -> None:
        if bin_idx < self.traction_energy_bins.shape[1]:
            return
        extra = bin_idx + 1 - self.traction_energy_bins.shape[1]
        self.traction_energy_bins = np.pad(
            self.traction_energy_bins,
            ((0, 0), (0, extra)),
        )
        self.regenerative_energy_bins = np.pad(
            self.regenerative_energy_bins,
            ((0, 0), (0, extra)),
        )

    def _add_result_to_energy_ledger(
        self,
        train_idx: int,
        section_idx: int,
        result: LowerResult,
        scheduled_start_s: float,
    ) -> None:
        zone_idx = self.supply_zone_id(section_idx, train_idx)
        if zone_idx < 0:
            return
        self.total_traction_kwh += result.traction_kwh
        self.total_regenerative_kwh += result.regenerative_kwh
        previous_traction = 0.0
        previous_regen = 0.0
        for point in result.trajectory:
            traction_increment = max(0.0, point.traction_kwh - previous_traction)
            regen_increment = max(0.0, point.regenerative_kwh - previous_regen)
            previous_traction = point.traction_kwh
            previous_regen = point.regenerative_kwh
            if traction_increment <= 0.0 and regen_increment <= 0.0:
                continue
            bin_idx = int((scheduled_start_s + point.elapsed_s) // self.power_bin_s)
            if bin_idx < 0:
                continue
            self._ensure_energy_bin_capacity(bin_idx)
            old_matched = min(
                self.traction_energy_bins[zone_idx, bin_idx],
                self.regenerative_energy_bins[zone_idx, bin_idx],
            )
            self.traction_energy_bins[zone_idx, bin_idx] += traction_increment
            self.regenerative_energy_bins[zone_idx, bin_idx] += regen_increment
            new_matched = min(
                self.traction_energy_bins[zone_idx, bin_idx],
                self.regenerative_energy_bins[zone_idx, bin_idx],
            )
            self.matched_regenerative_kwh += new_matched - old_matched

    def _matched_regenerative_energy_kwh(self) -> float:
        return float(self.matched_regenerative_kwh)

    def _headway_violation_s(self) -> float:
        violation = 0.0
        for group in self.direction_groups:
            if len(group) < 2:
                continue
            indices = np.asarray(group, dtype=int)
            station_times = self.arrival_times[indices]
            for station_idx in range(self.section_count + 1):
                values = station_times[:, station_idx]
                if np.any(~np.isfinite(values)):
                    continue
                headways = np.diff(values)
                violation += float(
                    np.sum(np.maximum(0.0, self.minimum_headway_s - headways))
                )
        return violation

    def _calculate_metrics(self) -> Dict[str, float]:
        valid = self._valid_results()
        traction_kwh = self.total_traction_kwh
        regenerative_kwh = self.total_regenerative_kwh
        matched_kwh = self._matched_regenerative_energy_kwh()
        utilization = matched_kwh / max(regenerative_kwh, EPS)
        traction_ratio = traction_kwh / max(self.reference_traction_kwh, EPS)
        objective = (
            self.traction_energy_weight * traction_ratio
            - self.regenerative_match_weight * utilization
        )
        # Upper-layer feasibility keeps the original punctuality semantics.
        # LowerResult.feasible now reports physical/safety feasibility only.
        infeasible = sum(not result.strict_timetable_feasible for _, _, result in valid)
        timetable_error = sum(abs(result.timetable_error_s) for _, _, result in valid)
        return {
            "objective": float(objective),
            "objective_energy_kwh": float(objective),
            "normalized_objective": float(objective),
            "traction_energy_ratio": float(traction_ratio),
            "grid_energy_kwh": float(traction_kwh - 0.95 * matched_kwh),
            "traction_kwh": float(traction_kwh),
            "regenerative_kwh": float(regenerative_kwh),
            "matched_regenerative_kwh": float(matched_kwh),
            "regenerative_utilization": float(utilization),
            "traction_energy_weight": self.traction_energy_weight,
            "regenerative_match_weight": self.regenerative_match_weight,
            "supply_zone_size": 0.0,
            "supply_zone_count": float(self.supply_zone_count),
            "supply_zone_section_counts": "",
            "supply_zone_names": ";".join(self.supply_zone_names),
            "supply_zone_section_groups": ";".join(
                ",".join(SECTION_NAMES[idx] for idx in group)
                for group in self.supply_zone_section_groups
            ),
            "timetable_error_s": float(timetable_error),
            "infeasible_sections": float(infeasible),
            "headway_violation_s": float(self._headway_violation_s()),
            "rollout_cache_hits": float(self.rollout_cache.hits),
            "rollout_cache_misses": float(self.rollout_cache.misses),
            "rollout_cache_size": float(len(self.rollout_cache.items)),
        }

    def local_observations(self) -> np.ndarray:
        observations: List[np.ndarray] = []
        completed_fraction = self.current_section / max(self.section_count, 1)
        for train_idx in range(self.train_count):
            section_idx = min(int(self.current_section[train_idx]), self.section_count - 1)
            zone_idx = self.supply_zone_id(section_idx, train_idx)
            last_result = None
            if section_idx > 0:
                last_result = self.results_by_train[train_idx][section_idx - 1]
            group = next(group for group in self.direction_groups if train_idx in group)
            group_position = group.index(train_idx)
            previous_headway = self.minimum_headway_s
            next_headway = self.minimum_headway_s
            station_idx = int(self.current_section[train_idx])
            if station_idx <= self.section_count:
                if group_position > 0:
                    prev_train = group[group_position - 1]
                    previous_headway = (
                        self.arrival_times[train_idx, station_idx]
                        - self.arrival_times[prev_train, station_idx]
                        if np.isfinite(self.arrival_times[prev_train, station_idx])
                        and np.isfinite(self.arrival_times[train_idx, station_idx])
                        else self.minimum_headway_s
                    )
                if group_position < len(group) - 1:
                    next_train = group[group_position + 1]
                    next_headway = (
                        self.arrival_times[next_train, station_idx]
                        - self.arrival_times[train_idx, station_idx]
                        if np.isfinite(self.arrival_times[next_train, station_idx])
                        and np.isfinite(self.arrival_times[train_idx, station_idx])
                        else self.minimum_headway_s
                    )
            observations.append(
                np.asarray(
                    [
                        completed_fraction[train_idx],
                        section_idx / max(self.section_count - 1, 1),
                        self.current_time_s[train_idx] / 7200.0,
                        self.initial_timetables[train_idx, section_idx] / 300.0,
                        previous_headway / max(self.minimum_headway_s, EPS),
                        next_headway / max(self.minimum_headway_s, EPS),
                        max(zone_idx, 0) / max(self.supply_zone_count - 1, 1),
                        self.metrics["regenerative_utilization"],
                        self.metrics["traction_energy_ratio"],
                        0.0 if last_result is None else last_result.timetable_error_s / 10.0,
                        0.0 if last_result is None else last_result.traction_kwh / 50.0,
                        0.0 if last_result is None else last_result.regenerative_kwh / 50.0,
                    ],
                    dtype=np.float32,
                )
            )
        return np.asarray(observations, dtype=np.float32)

    def central_state(self) -> np.ndarray:
        return self.local_observations().reshape(-1)

    def _candidate_target_runtimes_s(
        self,
        initial_runtime_s: float,
        requested_runtime_s: float,
    ) -> List[float]:
        lower_bound = max(1.0, initial_runtime_s - self.runtime_adjustment_span_s)
        upper_bound = max(lower_bound, initial_runtime_s + self.runtime_adjustment_span_s)
        quantum = max(self.rollout_cache.time_quantum_s, EPS)
        candidate_count = int(round((upper_bound - lower_bound) / quantum)) + 1
        candidates = [
            lower_bound + idx * quantum
            for idx in range(max(candidate_count, 1))
        ]
        candidates.extend([requested_runtime_s, initial_runtime_s])
        quantized_unique = {
            self.rollout_cache._quantize_time(
                min(upper_bound, max(lower_bound, float(value)))
            )
            for value in candidates
        }
        return sorted(
            quantized_unique,
            key=lambda value: (
                abs(value - requested_runtime_s),
                abs(value - initial_runtime_s),
            ),
        )

    def _safe_lower_rollout(
        self,
        train_idx: int,
        section_idx: int,
        requested_runtime_s: float,
    ) -> Tuple[LowerResult, float, bool, bool, int]:
        initial_runtime_s = float(self.initial_timetables[train_idx, section_idx])
        candidates = self._candidate_target_runtimes_s(
            initial_runtime_s,
            requested_runtime_s,
        )
        attempted: List[Tuple[LowerResult, float, bool]] = []
        failures: List[str] = []
        requested_quantized = self.rollout_cache._quantize_time(requested_runtime_s)
        for candidate_runtime_s in candidates:
            try:
                result, quantized_runtime_s, cache_hit = self.rollout_cache.get(
                    train_idx,
                    section_idx,
                    self.environments_by_train[train_idx][section_idx],
                    (
                        self.lower_agents[train_idx]
                        if self.lower_model_solver is None
                        else None
                    ),
                    candidate_runtime_s,
                    model_solver=self.lower_model_solver,
                )
            except RuntimeError as error:
                failures.append(str(error))
                continue
            attempted.append((result, quantized_runtime_s, cache_hit))
            if result.strict_timetable_feasible:
                return (
                    result,
                    quantized_runtime_s,
                    cache_hit,
                    abs(quantized_runtime_s - requested_quantized) > 1e-6,
                    len(attempted),
                )
        try:
            initial_result, initial_quantized_runtime_s, initial_cache_hit = (
                self.rollout_cache.get(
                    train_idx,
                    section_idx,
                    self.environments_by_train[train_idx][section_idx],
                    (
                        self.lower_agents[train_idx]
                        if self.lower_model_solver is None
                        else None
                    ),
                    initial_runtime_s,
                    model_solver=self.lower_model_solver,
                )
            )
        except RuntimeError as error:
            failures.append(str(error))
            details = " | ".join(dict.fromkeys(failures))
            raise RuntimeError(
                f"No lower-level solution for train {train_idx + 1}, "
                f"section {section_idx + 1}: {details}"
            ) from error
        return (
            initial_result,
            initial_quantized_runtime_s,
            initial_cache_hit,
            True,
            len(attempted),
        )

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self.done:
            raise RuntimeError("step() called after online environment finished")
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.train_count, self.action_dim):
            raise ValueError(
                f"Expected upper actions {(self.train_count, self.action_dim)}, "
                f"got {actions.shape}"
            )
        previous_metrics = dict(self.metrics)
        train_idx = self.next_event_train_idx()
        if train_idx < 0:
            self.done = True
            return self.local_observations(), 0.0, True, dict(self.metrics)

        section_idx = int(self.current_section[train_idx])
        # actions[..., 0]：停站时间调整；影响当前站发车及后续时刻。
        dwell_adjustment = (
            self.dwell_adjustment_span_s * math.tanh(float(actions[train_idx, 0]))
        )
        # actions[..., 1]：站间运行时间调整；不等同于下层训练目标扰动。
        runtime_adjustment = (
            self.runtime_adjustment_span_s * math.tanh(float(actions[train_idx, 1]))
        )
        if section_idx > 0:
            self.current_time_s[train_idx] += self.station_dwell_s
        self.current_time_s[train_idx] += dwell_adjustment
        requested_target_runtime = max(
            1.0,
            float(self.initial_timetables[train_idx, section_idx])
            + runtime_adjustment,
        )
        section_start_s = float(self.current_time_s[train_idx])
        (
            lower_result,
            target_runtime,
            cache_hit,
            safety_filtered,
            safety_candidate_count,
        ) = self._safe_lower_rollout(
            train_idx,
            section_idx,
            requested_target_runtime,
        )
        runtime_adjustment = target_runtime - float(
            self.initial_timetables[train_idx, section_idx]
        )
        arrival_s = section_start_s + lower_result.actual_time_s
        self.departure_start_times[train_idx, section_idx] = section_start_s
        self.arrival_times[train_idx, section_idx + 1] = arrival_s
        self.results_by_train[train_idx][section_idx] = lower_result
        self.timetables[train_idx, section_idx] = target_runtime
        self.current_time_s[train_idx] = arrival_s
        self.current_section[train_idx] += 1
        self._add_result_to_energy_ledger(
            train_idx,
            section_idx,
            lower_result,
            section_start_s,
        )
        self.step_records.append(
            OnlineStepRecord(
                step=self.steps + 1,
                train_index=train_idx,
                section_index=section_idx,
                section_name=lower_result.section_name,
                physical_section_name=self.physical_section_names_by_train[train_idx][section_idx],
                departure_time_s=section_start_s,
                arrival_time_s=arrival_s,
                requested_target_runtime_s=requested_target_runtime,
                target_runtime_s=target_runtime,
                actual_runtime_s=lower_result.actual_time_s,
                dwell_adjustment_s=dwell_adjustment,
                runtime_adjustment_s=runtime_adjustment,
                traction_kwh=lower_result.traction_kwh,
                regenerative_kwh=lower_result.regenerative_kwh,
                feasible=lower_result.strict_timetable_feasible,
                safety_filtered=safety_filtered,
                speed_limit_action_shield_overrides=lower_result.speed_limit_action_shield_overrides,
            )
        )

        self.steps += 1
        self.done = bool(np.all(self.current_section >= self.section_count))
        self.metrics = self._calculate_metrics()
        self.objective = self.metrics["objective"]
        traction_reward_component = (
            self.traction_energy_weight
            * (previous_metrics["traction_kwh"] - self.metrics["traction_kwh"])
            / max(self.reference_traction_kwh, EPS)
        )
        regenerative_reward_component = (
            self.regenerative_match_weight
            * (
                self.metrics["regenerative_utilization"]
                - previous_metrics["regenerative_utilization"]
            )
        )
        feasibility_penalty = -0.02 * self.metrics["infeasible_sections"]
        safety_filter_penalty = -0.002 if safety_filtered else 0.0
        headway_penalty = -0.001 * self.metrics["headway_violation_s"]
        reward = (
            traction_reward_component
            + regenerative_reward_component
            + feasibility_penalty
            + safety_filter_penalty
            + headway_penalty
        )
        info = dict(self.metrics)
        info.update(
            {
                "active_train_index": float(train_idx),
                "rollout_cache_hit": float(cache_hit),
                "reward_traction_energy_component": float(traction_reward_component),
                "reward_regenerative_energy_component": float(regenerative_reward_component),
                "reward_feasibility_component": float(feasibility_penalty),
                "reward_safety_filter_component": float(safety_filter_penalty),
                "reward_headway_component": float(headway_penalty),
                "action_rejected": float(safety_filtered),
                "safety_filtered": float(safety_filtered),
                "safety_candidate_count": float(safety_candidate_count),
                "requested_target_runtime_s": float(requested_target_runtime),
            }
        )
        return self.local_observations(), float(reward), self.done, info

    def result(self, initial_metrics: Dict[str, float]) -> MultiTrainTrainingResult:
        return MultiTrainTrainingResult(
            control_mode=self.control_mode,
            initial_headway_s=0.0,
            timetables_s=self.timetables.tolist(),
            departure_times_s=self.departure_times_s.tolist(),
            objective=self.objective,
            grid_energy_kwh=self.metrics["grid_energy_kwh"],
            traction_kwh=self.metrics["traction_kwh"],
            regenerative_kwh=self.metrics["regenerative_kwh"],
            matched_regenerative_kwh=self.metrics["matched_regenerative_kwh"],
            regenerative_utilization=self.metrics["regenerative_utilization"],
            supply_zone_size=0,
            supply_zone_count=self.supply_zone_count,
            supply_zone_section_counts=[],
            supply_zone_names=list(self.supply_zone_names),
            supply_zone_section_groups=[
                [SECTION_NAMES[idx] for idx in group]
                for group in self.supply_zone_section_groups
            ],
            timetable_error_s=self.metrics["timetable_error_s"],
            infeasible_sections=self.metrics["infeasible_sections"],
            initial_objective=initial_metrics["objective"],
            initial_grid_energy_kwh=initial_metrics["grid_energy_kwh"],
            initial_matched_regenerative_kwh=initial_metrics["matched_regenerative_kwh"],
        )

    def step_record_rows(self) -> List[Dict[str, Any]]:
        return [record.__dict__.copy() for record in self.step_records]

    def operation_results(self) -> List[List[LowerResult]]:
        rows: List[List[LowerResult]] = []
        for train_results in self.results_by_train:
            rows.append([result for result in train_results if result is not None])
        return rows


def train_online_mappo(
    environment: OnlineHierarchicalTimetableEnvironment,
    controller: IndependentActorMAPPOAgent,
    episodes: int,
    progress_interval: int = 10,
) -> Tuple[MultiTrainTrainingResult, List[Dict[str, Any]], List[List[LowerResult]], List[Dict[str, Any]]]:
    environment.reset()
    initial_metrics = dict(environment.metrics)
    best: MultiTrainTrainingResult | None = None
    best_results: List[List[LowerResult]] = []
    best_step_rows: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    print(
        f"Training online hierarchical MAPPO for {episodes} episodes "
        f"with lower-policy rollout calls"
    )
    total_event_steps = environment.train_count * environment.section_count
    progress = RollingProgressPrinter(progress_interval)
    try:
        for episode in range(episodes):
            observations = environment.reset()
            trajectory: List[Dict[str, Any]] = []
            episode_reward = 0.0
            episode_traction_component = 0.0
            episode_regenerative_component = 0.0
            episode_safety_filter_component = 0.0
            rejected_action_count = 0.0
            done = False
            while not done:
                state = environment.central_state()
                active_agent_mask = environment.current_active_agent_mask()
                actions, log_probabilities, value = controller.select_actions(
                    observations,
                    state,
                )
                next_observations, reward, done, info = environment.step(actions)
                if environment.step_records:
                    progress.maybe_print(
                        episode + 1,
                        episodes,
                        environment.steps,
                        total_event_steps,
                        environment.step_records[-1],
                        environment.metrics,
                    )
                trajectory.append(
                    {
                        "observations": observations,
                        "state": state,
                        "actions": actions,
                        "log_probabilities": log_probabilities,
                        "value": value,
                        "reward": reward,
                        "done": done,
                        "active_agent_mask": active_agent_mask,
                    }
                )
                observations = next_observations
                episode_reward += reward
                episode_traction_component += info.get("reward_traction_energy_component", 0.0)
                episode_regenerative_component += info.get("reward_regenerative_energy_component", 0.0)
                episode_safety_filter_component += info.get("reward_safety_filter_component", 0.0)
                rejected_action_count += info.get("action_rejected", 0.0)
            losses = controller.update(trajectory)
            current = environment.result(initial_metrics)
            step_records = environment.step_records
            runtime_adjustments = np.asarray(
                [record.runtime_adjustment_s for record in step_records],
                dtype=float,
            )
            dwell_adjustments = np.asarray(
                [record.dwell_adjustment_s for record in step_records],
                dtype=float,
            )
            if best is None or current.objective < best.objective:
                best = current
                best_results = environment.operation_results()
                best_step_rows = environment.step_record_rows()
            assert best is not None
            history.append(
                {
                    "episode": episode + 1,
                    "control_mode": environment.control_mode,
                    "reward": episode_reward,
                    "objective": current.objective,
                    "normalized_objective": current.objective,
                    "best_objective": best.objective,
                    "objective_improvement": initial_metrics["objective"] - current.objective,
                    "best_objective_improvement": initial_metrics["objective"] - best.objective,
                    "grid_energy_kwh": current.grid_energy_kwh,
                    "objective_energy_kwh": current.objective,
                    "traction_kwh": current.traction_kwh,
                    "traction_energy_ratio": environment.metrics["traction_energy_ratio"],
                    "best_grid_energy_kwh": best.grid_energy_kwh,
                    "matched_regenerative_kwh": current.matched_regenerative_kwh,
                    "best_matched_regenerative_kwh": best.matched_regenerative_kwh,
                    "regenerative_utilization": current.regenerative_utilization,
                    "traction_energy_weight": environment.traction_energy_weight,
                    "regenerative_match_weight": environment.regenerative_match_weight,
                    "reward_traction_energy_component": episode_traction_component,
                    "reward_regenerative_energy_component": episode_regenerative_component,
                    "reward_safety_filter_component": episode_safety_filter_component,
                    "rejected_action_count": rejected_action_count,
                    "mean_abs_runtime_adjustment_s": float(
                        np.mean(np.abs(runtime_adjustments))
                        if len(runtime_adjustments)
                        else 0.0
                    ),
                    "max_abs_runtime_adjustment_s": float(
                        np.max(np.abs(runtime_adjustments))
                        if len(runtime_adjustments)
                        else 0.0
                    ),
                    "mean_abs_dwell_adjustment_s": float(
                        np.mean(np.abs(dwell_adjustments))
                        if len(dwell_adjustments)
                        else 0.0
                    ),
                    "max_abs_dwell_adjustment_s": float(
                        np.max(np.abs(dwell_adjustments))
                        if len(dwell_adjustments)
                        else 0.0
                    ),
                    "supply_zone_size": 0.0,
                    "supply_zone_count": environment.supply_zone_count,
                    "supply_zone_section_counts": "",
                    "supply_zone_names": ";".join(environment.supply_zone_names),
                    "supply_zone_section_groups": ";".join(
                        ",".join(SECTION_NAMES[idx] for idx in group)
                        for group in environment.supply_zone_section_groups
                    ),
                    "timetable_error_s": current.timetable_error_s,
                    "infeasible_sections": current.infeasible_sections,
                    "headway_violation_s": environment.metrics["headway_violation_s"],
                    "rollout_cache_hits": environment.metrics["rollout_cache_hits"],
                    "rollout_cache_misses": environment.metrics["rollout_cache_misses"],
                    "rollout_cache_size": environment.metrics["rollout_cache_size"],
                    "mean_abs_timetable_change_s": float(
                        np.mean(np.abs(np.asarray(current.timetables_s) - environment.initial_timetables))
                    ),
                    "best_mean_abs_timetable_change_s": float(
                        np.mean(np.abs(np.asarray(best.timetables_s) - environment.initial_timetables))
                    ),
                    "mean_abs_departure_offset_s": float(
                        np.mean(np.abs(np.asarray(current.departure_times_s) - environment.base_departures_s))
                    ),
                    "best_mean_abs_departure_offset_s": float(
                        np.mean(np.abs(np.asarray(best.departure_times_s) - environment.base_departures_s))
                    ),
                    "actor_loss": losses["actor_loss"],
                    "critic_loss": losses["critic_loss"],
                    "entropy": losses["entropy"],
                    "policy_kl": losses["policy_kl"],
                    "clip_fraction": losses["clip_fraction"],
                    "value_explained_variance": losses["value_explained_variance"],
                    "current_timetables_s": current.timetables_s,
                    "best_timetables_s": best.timetables_s,
                    "current_departure_times_s": current.departure_times_s,
                    "best_departure_times_s": best.departure_times_s,
                }
            )
            if episode == 0 or (episode + 1) % max(1, episodes // 10) == 0:
                print(
                    f"  online MAPPO episode {episode + 1:5d}/{episodes}: "
                    f"reward={episode_reward:+10.6f} "
                    f"objective={current.objective:+10.6f} "
                    f"best={best.objective:+10.6f} "
                    f"traction={current.traction_kwh:7.2f}kWh "
                    f"regen_use={100.0 * current.regenerative_utilization:5.1f}%"
                )
    except KeyboardInterrupt:
        if best is None:
            raise
        print(
            "\nTraining interrupted; exporting the best complete episode "
            "available so far."
        )
    assert best is not None
    return best, history, best_results, best_step_rows


def write_online_operation_rows(
    output_path: Path,
    results_by_train: Sequence[Sequence[LowerResult]],
    train_names: Sequence[str],
) -> None:
    import csv

    rows: List[Dict[str, Any]] = []
    for train_idx, results in enumerate(results_by_train):
        for section_idx, result in enumerate(results):
            for point in result.trajectory:
                rows.append(
                    {
                        "train_name": train_names[train_idx],
                        "section_index": section_idx + 1,
                        "section_name": result.section_name,
                        "elapsed_s": point.elapsed_s,
                        "position_m": point.position_m,
                        "speed_kmh": point.speed_kmh,
                        "force_kn": point.force_kn,
                        "speed_limit_kmh": point.speed_limit_kmh,
                        "gradient_per_mille": point.gradient_per_mille,
                        "regime": point.regime,
                        "traction_kwh": point.traction_kwh,
                        "regenerative_kwh": point.regenerative_kwh,
                    }
                )
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
