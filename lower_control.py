"""Lower-level train eco-driving agents and response-table construction."""

from __future__ import annotations

import bisect
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from .config import EPS, KWH_IN_JOULES
    from .line_profile import StationSection, TrackSegment
    from .train_model import TRAIN_CONFIGS, TrainDynamics, create_train_config_from_mass
except ImportError:
    from config import EPS, KWH_IN_JOULES
    from line_profile import StationSection, TrackSegment
    from train_model import TRAIN_CONFIGS, TrainDynamics, create_train_config_from_mass


PRIMARY_ARRIVAL_TOLERANCE_S = 1.0
RELAXED_ARRIVAL_TOLERANCE_S = 1.2


@dataclass(frozen=True)
class TrajectoryPoint:
    section_name: str
    elapsed_s: float
    position_m: float
    speed_kmh: float
    force_kn: float
    resistance_kn: float
    speed_limit_kmh: float
    gradient_per_mille: float
    radius_m: Optional[float]
    requested_action: int
    applied_action: int
    regime: str
    traction_kwh: float
    regenerative_kwh: float


@dataclass
class LowerResult:
    section_name: str
    target_time_s: float
    actual_time_s: float
    traction_kwh: float
    regenerative_kwh: float
    net_energy_kwh: float
    arrival_speed_mps: float
    feasible: bool
    timed_out: bool
    regime_switches: int
    traction_brake_transitions: int
    safety_brake_overrides: int
    traction_coast_transitions: int
    short_dwell_transitions: int
    force_jump_index: float
    smoothness_penalty: float
    jerk_integral_mps3_s: float
    comfort_penalty: float
    traction_reward_component: float
    punctuality_reward_component: float
    comfort_reward_component: float
    max_speed_limit_violation_kmh: float
    speed_regulation_steps: int
    strict_timetable_feasible: bool
    relaxed_timetable_feasible: bool
    accepted_arrival_tolerance_s: Optional[float]
    episode_reward: float
    expert_interventions: int = 0
    trajectory: List[TrajectoryPoint] = field(default_factory=list, repr=False)

    @property
    def timetable_error_s(self) -> float:
        return self.actual_time_s - self.target_time_s


@dataclass(frozen=True)
class DrivingSmoothnessConfig:
    """Soft constraints for realistic traction, coasting and braking profiles."""

    preferred_traction_hold_s: float = 40.0
    preferred_coast_hold_s: float = 40.0
    preferred_brake_hold_s: float = 6.0
    heuristic_traction_release_margin_mps: float = 5.0
    heuristic_coast_reentry_margin_mps: float = 5.0
    reward_regime_switch: float = 0.08
    reward_traction_coast_switch: float = 0.14
    reward_short_dwell: float = 0.18
    reward_force_jump: float = 0.025
    score_regime_switch: float = 0.20
    score_traction_coast_switch: float = 0.65
    score_short_dwell: float = 0.90
    score_force_jump: float = 0.08
    speed_hold_activation_margin_mps: float = 1.50
    speed_hold_target_buffer_mps: float = 0.80
    speed_hold_response_time_s: float = 4.00
    speed_hold_force_ramp_n_per_s: float = 100000.0
    speed_hold_max_traction_ratio: float = 0.35
    speed_hold_max_braking_ratio: float = 0.45
    safety_latch_tolerance_mps: float = 0.50
    safety_release_margin_mps: float = 1.00
    schedule_tracking_activation_distance_m: float = 1600.0
    schedule_tracking_activation_time_s: float = 100.0
    schedule_tracking_time_buffer_s: float = 2.50
    schedule_tracking_response_time_s: float = 2.0
    schedule_tracking_max_traction_ratio: float = 1.0
    schedule_tracking_max_braking_ratio: float = 1.0
    speed_limit_buffer_mps: float = 0.50
    speed_limit_reaction_time_s: float = 2.0
    speed_limit_distance_margin_m: float = 15.0
    conservative_braking_factor: float = 0.80
    minimum_effective_braking_deceleration_mps2: float = 0.35
    predictive_braking_trigger_margin_mps: float = 0.25
    predictive_braking_response_time_s: float = 1.50
    force_command_quantum_ratio: float = 0.08
    force_command_deadband_ratio: float = 0.05
    force_command_min_hold_s: float = 2.0
    force_command_coast_deadband_ratio: float = 0.03


# Edit this single object to tune driving smoothness without changing dynamics.
DRIVING_SMOOTHNESS = DrivingSmoothnessConfig()


@dataclass(frozen=True)
class LowerRewardConfig:
    """Three-term lower reward: traction energy, punctuality and jerk comfort."""

    traction_energy_weight: float = 0.12
    traction_energy_reference_km: float = 1.0
    schedule_progress_weight: float = 0.30
    punctuality_weight: float = 0.80
    jerk_weight: float = 0.02
    force_change_weight: float = 0.01
    speed_limit_violation_weight: float = 0.05
    timeout_penalty: float = 120.0
    premature_stop_penalty: float = 200.0


LOWER_REWARD = LowerRewardConfig()


@dataclass(frozen=True)
class ExpertKnowledgeConfig:
    """Minimal expert rules for train-operation feasibility."""

    enabled: bool = True
    no_stop_before_station_distance_m: float = 35.0
    low_speed_guard_mps: float = 0.50
    premature_stop_position_tolerance_m: float = 1.0
    speed_limit_traction_margin_mps: float = 0.80


# Edit this single object to tune expert safety/action-shield rules.
EXPERT_KNOWLEDGE = ExpertKnowledgeConfig()


class LowerSectionEnvironment:
    """Eco-driving environment for one station-to-station section."""

    ACTION_BRAKE = 0
    ACTION_COAST = 1
    ACTION_HALF_TRACTION = 2
    ACTION_FULL_TRACTION = 3
    ACTION_DIM = 4
    STATE_DIM = 17
    COMPACT_STATE_DIM = 13
    MIN_TRACTION_HOLD_S = 4.0
    MIN_COAST_HOLD_S = 3.0
    MIN_BRAKE_HOLD_S = 4.0
    POST_BRAKE_COAST_S = 3.0

    def __init__(
        self,
        section: StationSection,
        dynamics: TrainDynamics,
        speed_limit_penalty_ratio: float = 1.0,
    ) -> None:
        self.section = section
        self.dynamics = dynamics
        self.speed_limit_penalty_ratio = float(speed_limit_penalty_ratio)
        self.reset(1.0)

    def reset(self, target_time_s: float) -> np.ndarray:
        self.target_time_s = max(float(target_time_s), 1.0)
        self.position_m = self.section.start_m
        self.speed_mps = 0.0
        self.elapsed_s = 0.0
        self.traction_kwh = 0.0
        self.regenerative_kwh = 0.0
        self.done = False
        self.timed_out = False
        self.step_count = 0
        self.max_steps = int(max(1200, self.target_time_s * 6.0) / self.dynamics.config.dt_s)
        self.current_action = self.ACTION_COAST
        self.current_regime = "coast"
        self.regime_elapsed_s = self.MIN_COAST_HOLD_S
        self.post_brake_coast_remaining_s = 0.0
        self.safety_brake_latched = False
        self.terminal_braking_latched = False
        self.regime_switches = 0
        self.traction_brake_transitions = 0
        self.safety_brake_overrides = 0
        self.expert_interventions = 0
        self.traction_coast_transitions = 0
        self.short_dwell_transitions = 0
        self.force_jump_index = 0.0
        self.smoothness_penalty = 0.0
        self.speed_regulation_steps = 0
        self.speed_regulation_active = False
        self.speed_regulation_mode: Optional[str] = None
        self.requested_speed_hold_target_mps: Optional[float] = None
        self.pending_transition_penalty = 0.0
        self.previous_force_n = 0.0
        self.force_command_elapsed_s = DRIVING_SMOOTHNESS.force_command_min_hold_s
        self.previous_acceleration_mps2 = 0.0
        self.jerk_integral_mps3_s = 0.0
        self.comfort_penalty = 0.0
        self.traction_reward_component = 0.0
        self.punctuality_reward_component = 0.0
        self.comfort_reward_component = 0.0
        self.episode_reward = 0.0
        self.last_non_coast_regime: Optional[str] = None
        self.trajectory: List[TrajectoryPoint] = []
        self._append_trajectory_point(self.ACTION_COAST, self.ACTION_COAST, 0.0, 0.0)
        return self.observation()

    @property
    def remaining_m(self) -> float:
        return max(0.0, self.section.end_m - self.position_m)

    def _conservative_braking_deceleration_mps2(
        self,
        target_position_m: float,
    ) -> float:
        cfg = self.dynamics.config
        braking_force_n = cfg.max_braking_force_n * cfg.service_braking_ratio
        path_segments = [
            self.section.segment_at(self.position_m),
            *[
                segment
                for segment in self.section.future_segments(self.position_m)
                if segment.start_m < target_position_m
            ],
        ]
        effective_decelerations = []
        for segment in path_segments:
            resistance_n = self.dynamics.total_resistance_n(
                max(self.speed_mps, segment.speed_limit_kmh / 3.6),
                segment,
            )
            effective_deceleration = max(
                DRIVING_SMOOTHNESS.minimum_effective_braking_deceleration_mps2,
                (braking_force_n + resistance_n) / cfg.mass_kg,
            )
            effective_decelerations.append(
                min(cfg.max_deceleration_mps2, effective_deceleration)
            )
        return (
            min(effective_decelerations)
            * DRIVING_SMOOTHNESS.conservative_braking_factor
        )

    def _track_speed_limit_preview(self) -> Tuple[float, float, float]:
        """Return safe speed, governing future limit and its distance."""
        current = self.section.segment_at(self.position_m)
        current_limit_mps = (
            current.speed_limit_kmh * self.speed_limit_penalty_ratio / 3.6
        )
        safe_speed = max(
            0.0,
            current_limit_mps - DRIVING_SMOOTHNESS.speed_limit_buffer_mps,
        )
        governing_limit_mps = current_limit_mps
        governing_distance_m = self.remaining_m
        for future in self.section.future_segments(self.position_m):
            future_limit = (
                future.speed_limit_kmh * self.speed_limit_penalty_ratio / 3.6
            )
            if future_limit >= current_limit_mps - 1e-6:
                continue
            distance = max(0.0, future.start_m - self.position_m)
            braking_deceleration = self._conservative_braking_deceleration_mps2(
                future.start_m
            )
            usable_distance = max(
                0.0,
                distance
                - self.speed_mps
                * DRIVING_SMOOTHNESS.speed_limit_reaction_time_s
                - DRIVING_SMOOTHNESS.speed_limit_distance_margin_m,
            )
            buffered_limit = max(
                0.0,
                future_limit - DRIVING_SMOOTHNESS.speed_limit_buffer_mps,
            )
            candidate_speed = math.sqrt(
                max(
                    0.0,
                    buffered_limit**2
                    + 2.0 * braking_deceleration * usable_distance,
                )
            )
            if candidate_speed < safe_speed:
                safe_speed = candidate_speed
                governing_limit_mps = future_limit
                governing_distance_m = distance
        return safe_speed, governing_limit_mps, governing_distance_m

    def _safe_speed_mps(self) -> float:
        track_safe_speed, _, _ = self._track_speed_limit_preview()
        terminal_deceleration = self._conservative_braking_deceleration_mps2(
            self.section.end_m
        )
        terminal_usable_distance = self.remaining_m
        terminal_safe_speed = math.sqrt(
            max(0.0, 2.0 * terminal_deceleration * terminal_usable_distance)
        )
        return min(track_safe_speed, terminal_safe_speed)

    def observation(self) -> np.ndarray:
        segment = self.section.segment_at(self.position_m)
        safe_speed = self._safe_speed_mps()
        _, next_limit_mps, next_limit_distance_m = self._track_speed_limit_preview()
        remaining_time = self.target_time_s - self.elapsed_s
        required_average_speed = self.remaining_m / max(remaining_time, 1.0)
        radius_feature = 0.0 if segment.radius_m is None else min(1.0, 600.0 / segment.radius_m)
        return np.asarray(
            [
                self.remaining_m / max(self.section.length_m, 1.0),
                self.speed_mps / 40.0,
                safe_speed / 40.0,
                segment.speed_limit_kmh * self.speed_limit_penalty_ratio / 125.0,
                segment.gradient_per_mille / 30.0,
                radius_feature,
                self.elapsed_s / self.target_time_s,
                remaining_time / self.target_time_s,
                required_average_speed / 40.0,
                self.section.length_m / 6500.0,
                self.dynamics.config.effective_motor_car_mass_kg / 200000.0,
                self.dynamics.config.effective_trailer_car_mass_kg / 150000.0,
                {"brake": -1.0, "coast": 0.0, "traction": 1.0}[self.current_regime],
                min(1.0, self.regime_elapsed_s / 10.0),
                min(1.0, self.post_brake_coast_remaining_s / self.POST_BRAKE_COAST_S),
                next_limit_mps / 40.0,
                min(1.0, next_limit_distance_m / max(self.section.length_m, 1.0)),
            ],
            dtype=np.float32,
        )

    def compact_observation(self) -> np.ndarray:
        """Compact state for continuous lower-control agents.

        The policy sees train motion, schedule progress, mass and a concise
        preview of line conditions.  This keeps DDPG smaller than the discrete
        DQN state while still giving it explicit awareness of gradient, curve
        resistance and speed-limit changes.
        """
        segment = self.section.segment_at(self.position_m)
        _, next_limit_mps, next_limit_distance_m = self._track_speed_limit_preview()
        travelled_ratio = 1.0 - self.remaining_m / max(self.section.length_m, 1.0)
        elapsed_ratio = self.elapsed_s / max(self.target_time_s, EPS)
        target_average_speed = self.section.length_m / max(self.target_time_s, EPS)
        radius_feature = 0.0 if segment.radius_m is None else min(1.0, 600.0 / segment.radius_m)
        max_force = max(
            self.dynamics.config.max_traction_force_n,
            self.dynamics.config.max_braking_force_n,
            EPS,
        )
        cfg = self.dynamics.config
        acceleration_scale = max(
            cfg.max_acceleration_mps2,
            cfg.max_deceleration_mps2,
            EPS,
        )
        return np.asarray(
            [
                travelled_ratio,
                self.speed_mps / 40.0,
                self.previous_acceleration_mps2 / acceleration_scale,
                elapsed_ratio,
                target_average_speed / 40.0,
                cfg.effective_motor_car_mass_kg / 200000.0,
                cfg.effective_trailer_car_mass_kg / 150000.0,
                self.previous_force_n / max_force,
                segment.speed_limit_kmh * self.speed_limit_penalty_ratio / 125.0,
                segment.gradient_per_mille / 30.0,
                radius_feature,
                next_limit_mps / 40.0,
                min(1.0, next_limit_distance_m / max(self.section.length_m, 1.0)),
            ],
            dtype=np.float32,
        )

    def heuristic_action(self, aggressiveness: float = 1.0) -> int:
        self.requested_speed_hold_target_mps = None
        safe_speed = self._safe_speed_mps()
        remaining_time = max(self.target_time_s - self.elapsed_s, 1.0)
        required_average_speed = self.remaining_m / remaining_time
        segment = self.section.segment_at(self.position_m)
        traction_release_margin = 0.0
        coast_reentry_margin = 0.0
        if self.current_regime == "traction":
            traction_release_margin = (
                DRIVING_SMOOTHNESS.heuristic_traction_release_margin_mps
                * max(
                    0.0,
                    DRIVING_SMOOTHNESS.preferred_traction_hold_s - self.regime_elapsed_s,
                )
                / DRIVING_SMOOTHNESS.preferred_traction_hold_s
            )
        elif self.current_regime == "coast":
            coast_reentry_margin = (
                DRIVING_SMOOTHNESS.heuristic_coast_reentry_margin_mps
                * max(
                    0.0,
                    DRIVING_SMOOTHNESS.preferred_coast_hold_s - self.regime_elapsed_s,
                )
                / DRIVING_SMOOTHNESS.preferred_coast_hold_s
            )
        if self.speed_mps > safe_speed + 0.15:
            return self.ACTION_BRAKE
        if self.elapsed_s >= self.target_time_s:
            return self.ACTION_FULL_TRACTION
        if (
            segment.gradient_per_mille <= -8.0
            and self.speed_mps > max(6.0, required_average_speed * 1.45 * aggressiveness)
        ):
            self.requested_speed_hold_target_mps = max(
                6.0,
                required_average_speed * 1.45 * aggressiveness,
            )
            return self.ACTION_BRAKE
        if self.speed_mps > max(4.0, required_average_speed * 1.18 * aggressiveness) + traction_release_margin:
            return self.ACTION_COAST
        if self.speed_mps < min(
            safe_speed - 0.6,
            required_average_speed * aggressiveness - coast_reentry_margin,
        ):
            return self.ACTION_FULL_TRACTION
        if self.speed_mps < min(
            safe_speed - 0.25,
            required_average_speed * 1.15 * aggressiveness - coast_reentry_margin,
        ):
            return self.ACTION_HALF_TRACTION
        return self.ACTION_COAST

    def _requested_force_n(self, action: int) -> float:
        cfg = self.dynamics.config
        if action == self.ACTION_BRAKE:
            return -cfg.max_braking_force_n * cfg.service_braking_ratio
        if action == self.ACTION_COAST:
            return 0.0
        if action == self.ACTION_HALF_TRACTION:
            return 0.5 * self.dynamics.max_traction_force_n(self.speed_mps)
        if action == self.ACTION_FULL_TRACTION:
            return self.dynamics.max_traction_force_n(self.speed_mps)
        raise ValueError(f"Unknown lower action: {action}")

    def action_for_force_n(self, force_n: float) -> int:
        cfg = self.dynamics.config
        traction_threshold = 0.05 * cfg.max_traction_force_n
        brake_threshold = -0.05 * cfg.max_braking_force_n
        if force_n <= brake_threshold:
            return self.ACTION_BRAKE
        if force_n >= 0.55 * self.dynamics.max_traction_force_n(self.speed_mps):
            return self.ACTION_FULL_TRACTION
        if force_n >= traction_threshold:
            return self.ACTION_HALF_TRACTION
        return self.ACTION_COAST

    def _clip_continuous_force_n(self, force_n: float, applied_action: int) -> float:
        cfg = self.dynamics.config
        if applied_action == self.ACTION_BRAKE:
            return float(np.clip(force_n, -cfg.max_braking_force_n, 0.0))
        if applied_action in (self.ACTION_HALF_TRACTION, self.ACTION_FULL_TRACTION):
            return float(np.clip(force_n, 0.0, self.dynamics.max_traction_force_n(self.speed_mps)))
        return 0.0

    def acceleration_allowed_force_bounds_n(
        self,
        resistance_n: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Dynamic feasible force interval implied by acceleration limits."""
        cfg = self.dynamics.config
        if resistance_n is None:
            segment = self.section.segment_at(self.position_m)
            resistance_n = self.dynamics.total_resistance_n(self.speed_mps, segment)
        lower_force_n = resistance_n - cfg.mass_kg * cfg.max_deceleration_mps2
        upper_force_n = resistance_n + cfg.mass_kg * cfg.max_acceleration_mps2
        lower_force_n = max(lower_force_n, -cfg.max_braking_force_n)
        upper_force_n = min(
            upper_force_n,
            self.dynamics.max_traction_force_n(self.speed_mps),
        )
        if lower_force_n > upper_force_n:
            midpoint = 0.5 * (lower_force_n + upper_force_n)
            feasible_force_n = self.dynamics.acceleration_limited_force_n(
                midpoint,
                resistance_n,
                self.speed_mps,
            )
            return feasible_force_n, feasible_force_n
        return lower_force_n, upper_force_n

    def clip_force_to_action_space_n(
        self,
        force_n: float,
        resistance_n: Optional[float] = None,
    ) -> float:
        lower_force_n, upper_force_n = self.acceleration_allowed_force_bounds_n(
            resistance_n,
        )
        return float(np.clip(force_n, lower_force_n, upper_force_n))

    def force_ratio_for_force_n(self, force_n: float) -> float:
        positive_force_scale = max(
            self.dynamics.max_traction_force_n(self.speed_mps),
            EPS,
        )
        negative_force_scale = max(
            self.dynamics.config.max_braking_force_n,
            EPS,
        )
        return (
            force_n / positive_force_scale
            if force_n >= 0.0
            else force_n / negative_force_scale
        )

    def continuous_force_ratio_bounds(
        self,
        resistance_n: Optional[float] = None,
    ) -> Tuple[float, float]:
        lower_force_n, upper_force_n = self.acceleration_allowed_force_bounds_n(
            resistance_n,
        )
        return (
            self.force_ratio_for_force_n(lower_force_n),
            self.force_ratio_for_force_n(upper_force_n),
        )

    def clip_force_ratio(
        self,
        ratio: float,
        resistance_n: Optional[float] = None,
    ) -> float:
        lower_ratio, upper_ratio = self.continuous_force_ratio_bounds(resistance_n)
        if lower_ratio > upper_ratio:
            lower_ratio, upper_ratio = upper_ratio, lower_ratio
        return float(np.clip(ratio, lower_ratio, upper_ratio))

    def valid_discrete_actions(
        self,
        resistance_n: Optional[float] = None,
    ) -> List[int]:
        lower_force_n, upper_force_n = self.acceleration_allowed_force_bounds_n(
            resistance_n,
        )
        actions = [
            action
            for action in range(self.ACTION_DIM)
            if lower_force_n - EPS <= self._requested_force_n(action) <= upper_force_n + EPS
        ]
        if actions:
            return actions
        midpoint_force_n = 0.5 * (lower_force_n + upper_force_n)
        closest_action = min(
            range(self.ACTION_DIM),
            key=lambda candidate: abs(self._requested_force_n(candidate) - midpoint_force_n),
        )
        return [closest_action]

    def mask_invalid_discrete_action(
        self,
        action: int,
        resistance_n: Optional[float] = None,
    ) -> int:
        valid_actions = self.valid_discrete_actions(resistance_n)
        if action in valid_actions:
            return action
        clipped_force_n = self.clip_force_to_action_space_n(
            self._requested_force_n(action),
            resistance_n,
        )
        return min(
            valid_actions,
            key=lambda candidate: abs(self._requested_force_n(candidate) - clipped_force_n),
        )

    def _force_regime(self, force_n: float) -> str:
        cfg = self.dynamics.config
        traction_threshold = (
            DRIVING_SMOOTHNESS.force_command_coast_deadband_ratio
            * max(self.dynamics.max_traction_force_n(self.speed_mps), EPS)
        )
        brake_threshold = (
            DRIVING_SMOOTHNESS.force_command_coast_deadband_ratio
            * max(cfg.max_braking_force_n, EPS)
        )
        if force_n >= traction_threshold:
            return "traction"
        if force_n <= -brake_threshold:
            return "brake"
        return "coast"

    def _quantize_force_command_n(self, force_n: float) -> float:
        cfg = self.dynamics.config
        quantum = max(DRIVING_SMOOTHNESS.force_command_quantum_ratio, EPS)
        if force_n > 0.0:
            max_force = max(self.dynamics.max_traction_force_n(self.speed_mps), EPS)
            ratio = force_n / max_force
            if ratio < DRIVING_SMOOTHNESS.force_command_coast_deadband_ratio:
                return 0.0
            return float(np.clip(round(ratio / quantum) * quantum, 0.0, 1.0) * max_force)
        if force_n < 0.0:
            max_force = max(cfg.max_braking_force_n, EPS)
            ratio = -force_n / max_force
            if ratio < DRIVING_SMOOTHNESS.force_command_coast_deadband_ratio:
                return 0.0
            return -float(np.clip(round(ratio / quantum) * quantum, 0.0, 1.0) * max_force)
        return 0.0

    def _filter_force_command_n(
        self,
        force_n: float,
        allow_immediate: bool = False,
    ) -> Tuple[float, bool]:
        if allow_immediate:
            return force_n, abs(force_n - self.previous_force_n) > EPS
        candidate = self._quantize_force_command_n(force_n)
        scale = max(
            self.dynamics.max_traction_force_n(self.speed_mps),
            self.dynamics.config.max_braking_force_n,
            EPS,
        )
        if (
            abs(candidate - self.previous_force_n) / scale
            < DRIVING_SMOOTHNESS.force_command_deadband_ratio
        ):
            candidate = self.previous_force_n
        same_regime = self._force_regime(candidate) == self._force_regime(
            self.previous_force_n
        )
        if (
            same_regime
            and abs(candidate - self.previous_force_n) > EPS
            and self.force_command_elapsed_s
            < DRIVING_SMOOTHNESS.force_command_min_hold_s
        ):
            candidate = self.previous_force_n
        if candidate > 0.0:
            candidate = min(candidate, self.dynamics.max_traction_force_n(self.speed_mps))
        elif candidate < 0.0:
            candidate = max(candidate, -self.dynamics.config.max_braking_force_n)
        return candidate, abs(candidate - self.previous_force_n) > EPS

    @classmethod
    def _regime_for_action(cls, action: int) -> str:
        if action == cls.ACTION_BRAKE:
            return "brake"
        if action == cls.ACTION_COAST:
            return "coast"
        return "traction"

    def _set_applied_action(self, action: int) -> int:
        new_regime = self._regime_for_action(action)
        if new_regime != self.current_regime:
            previous_regime = self.current_regime
            previous_elapsed_s = self.regime_elapsed_s
            self.regime_switches += 1
            self.current_regime = new_regime
            self.regime_elapsed_s = 0.0
            transition_penalty = DRIVING_SMOOTHNESS.reward_regime_switch
            if {previous_regime, new_regime} == {"traction", "coast"}:
                self.traction_coast_transitions += 1
                transition_penalty += DRIVING_SMOOTHNESS.reward_traction_coast_switch
            preferred_hold_s = {
                "traction": DRIVING_SMOOTHNESS.preferred_traction_hold_s,
                "coast": DRIVING_SMOOTHNESS.preferred_coast_hold_s,
                "brake": DRIVING_SMOOTHNESS.preferred_brake_hold_s,
            }[previous_regime]
            dwell_deficit_ratio = max(0.0, preferred_hold_s - previous_elapsed_s) / preferred_hold_s
            if dwell_deficit_ratio > 0.0:
                self.short_dwell_transitions += 1
                transition_penalty += DRIVING_SMOOTHNESS.reward_short_dwell * dwell_deficit_ratio
            self.pending_transition_penalty += transition_penalty
            if previous_regime == "brake" and new_regime == "coast":
                self.post_brake_coast_remaining_s = self.POST_BRAKE_COAST_S
            if new_regime != "coast":
                if self.last_non_coast_regime not in (None, new_regime):
                    self.traction_brake_transitions += 1
                self.last_non_coast_regime = new_regime
        self.current_action = action
        return action

    def _apply_expert_knowledge(
        self,
        requested_action: int,
        track_safe_speed_mps: float,
    ) -> Tuple[int, bool]:
        if not EXPERT_KNOWLEDGE.enabled:
            return requested_action, False
        if self.terminal_braking_latched or self.safety_brake_latched:
            return requested_action, False

        requested_regime = self._regime_for_action(requested_action)
        low_speed_before_station = (
            self.speed_mps < EXPERT_KNOWLEDGE.low_speed_guard_mps
            and self.remaining_m
            > EXPERT_KNOWLEDGE.premature_stop_position_tolerance_m
            and requested_regime in {"brake", "coast"}
        )
        if low_speed_before_station:
            return self.ACTION_HALF_TRACTION, True

        unsafe_traction_near_limit = (
            requested_regime == "traction"
            and self.speed_mps
            >= track_safe_speed_mps
            - EXPERT_KNOWLEDGE.speed_limit_traction_margin_mps
        )
        if unsafe_traction_near_limit:
            return self.ACTION_COAST, True

        return requested_action, False

    def _apply_operational_constraints(self, requested_action: int, safe_speed_mps: float) -> int:
        cfg = self.dynamics.config
        service_deceleration = min(
            cfg.max_deceleration_mps2,
            cfg.max_braking_force_n * cfg.service_braking_ratio / cfg.mass_kg,
        )
        terminal_braking_distance = self.speed_mps**2 / max(2.0 * service_deceleration, EPS)
        terminal_braking_distance += self.speed_mps * cfg.terminal_braking_margin_s
        terminal_resistance_n = self.dynamics.total_resistance_n(
            self.speed_mps,
            self.section.segment_at(self.position_m),
        )
        terminal_deceleration = self.speed_mps**2 / max(2.0 * self.remaining_m, EPS)
        terminal_force_n = terminal_resistance_n - cfg.mass_kg * terminal_deceleration
        if (
            self.speed_mps > 0.5
            and self.remaining_m <= terminal_braking_distance
            and terminal_force_n <= 0.0
        ):
            self.terminal_braking_latched = True
        if self.terminal_braking_latched:
            return self._set_applied_action(self.ACTION_BRAKE)

        current_limit_mps = (
            self.section.segment_at(self.position_m).speed_limit_kmh
            * self.speed_limit_penalty_ratio
            / 3.6
        )
        track_safe_speed_mps, _, _ = self._track_speed_limit_preview()
        predictive_limit_risk = track_safe_speed_mps < (
            current_limit_mps - DRIVING_SMOOTHNESS.speed_limit_buffer_mps
        )
        if (
            self.speed_mps > current_limit_mps
            or (
                predictive_limit_risk
                and self.speed_mps
                >= track_safe_speed_mps
                - DRIVING_SMOOTHNESS.predictive_braking_trigger_margin_mps
            )
        ):
            self.safety_brake_latched = True
        elif self.safety_brake_latched and self.speed_mps <= max(
            0.0,
            track_safe_speed_mps - DRIVING_SMOOTHNESS.safety_release_margin_mps,
        ):
            self.safety_brake_latched = False
        if self.safety_brake_latched:
            if requested_action != self.ACTION_BRAKE:
                self.safety_brake_overrides += 1
            return self._set_applied_action(self.ACTION_BRAKE)

        requested_regime = self._regime_for_action(requested_action)
        if self.current_regime == "brake":
            if self.regime_elapsed_s < self.MIN_BRAKE_HOLD_S or requested_regime == "brake":
                return self._set_applied_action(self.ACTION_BRAKE)
            return self._set_applied_action(self.ACTION_COAST)
        if self.current_regime == "traction":
            if requested_regime == "traction":
                return self._set_applied_action(requested_action)
            if self.regime_elapsed_s < self.MIN_TRACTION_HOLD_S:
                return self._set_applied_action(self.current_action)
            return self._set_applied_action(self.ACTION_COAST)
        if self.post_brake_coast_remaining_s > 0.0:
            return self._set_applied_action(self.ACTION_COAST)
        if self.regime_elapsed_s < self.MIN_COAST_HOLD_S and requested_regime != "coast":
            return self._set_applied_action(self.ACTION_COAST)
        return self._set_applied_action(requested_action)

    def _terminal_braking_force_n(self, resistance_n: float) -> float:
        cfg = self.dynamics.config
        if self.remaining_m <= EPS:
            return -cfg.max_braking_force_n
        required_deceleration = min(
            cfg.max_deceleration_mps2,
            self.speed_mps**2 / (2.0 * self.remaining_m),
        )
        return min(0.0, max(-cfg.max_braking_force_n, resistance_n - cfg.mass_kg * required_deceleration))

    def _speed_limit_braking_force_n(self, resistance_n: float, safe_speed_mps: float) -> float:
        cfg = self.dynamics.config
        target_speed_mps = max(0.0, safe_speed_mps - DRIVING_SMOOTHNESS.speed_hold_target_buffer_mps)
        desired_acceleration = min(
            0.0,
            (target_speed_mps - self.speed_mps)
            / DRIVING_SMOOTHNESS.predictive_braking_response_time_s,
        )
        desired_acceleration = max(
            -cfg.max_deceleration_mps2,
            desired_acceleration,
        )
        return min(
            0.0,
            max(
                -cfg.max_braking_force_n * cfg.service_braking_ratio,
                resistance_n + cfg.mass_kg * desired_acceleration,
            ),
        )

    def _speed_hold_force_n(
        self,
        resistance_n: float,
        target_speed_mps: float,
        response_time_s: Optional[float] = None,
        max_traction_ratio: Optional[float] = None,
        max_braking_ratio: Optional[float] = None,
    ) -> float:
        cfg = self.dynamics.config
        max_traction_ratio = (
            DRIVING_SMOOTHNESS.speed_hold_max_traction_ratio
            if max_traction_ratio is None
            else max_traction_ratio
        )
        max_braking_ratio = (
            DRIVING_SMOOTHNESS.speed_hold_max_braking_ratio
            if max_braking_ratio is None
            else max_braking_ratio
        )
        desired_acceleration = (
            target_speed_mps - self.speed_mps
        ) / max(
            response_time_s or DRIVING_SMOOTHNESS.speed_hold_response_time_s,
            EPS,
        )
        desired_acceleration = max(
            -cfg.max_deceleration_mps2,
            min(cfg.max_acceleration_mps2, desired_acceleration),
        )
        target_force_n = resistance_n + cfg.mass_kg * desired_acceleration
        target_force_n = float(
            np.clip(
                target_force_n,
                -cfg.max_braking_force_n * max_braking_ratio,
                self.dynamics.max_traction_force_n(self.speed_mps)
                * max_traction_ratio,
            )
        )
        max_force_change_n = (
            DRIVING_SMOOTHNESS.speed_hold_force_ramp_n_per_s * cfg.dt_s
        )
        return float(
            np.clip(
                target_force_n,
                self.previous_force_n - max_force_change_n,
                self.previous_force_n + max_force_change_n,
            )
        )

    def _predict_step_motion(
        self,
        force_n: float,
        resistance_n: float,
    ) -> Tuple[float, float, float, float]:
        cfg = self.dynamics.config
        acceleration = (force_n - resistance_n) / cfg.mass_kg
        unconstrained_speed = self.speed_mps + acceleration * cfg.dt_s
        if acceleration < 0.0 and unconstrained_speed < 0.0:
            moving_time = -self.speed_mps / acceleration
            new_speed = 0.0
            distance = max(
                0.0,
                self.speed_mps * moving_time
                + 0.5 * acceleration * moving_time**2,
            )
        else:
            moving_time = cfg.dt_s
            new_speed = max(0.0, unconstrained_speed)
            distance = max(
                0.0,
                (self.speed_mps + new_speed) * 0.5 * cfg.dt_s,
            )
        return acceleration, moving_time, new_speed, distance

    def _is_premature_stop(self) -> bool:
        return (
            EXPERT_KNOWLEDGE.enabled
            and self.speed_mps <= EPS
            and self.position_m
            < self.section.end_m - EXPERT_KNOWLEDGE.premature_stop_position_tolerance_m
        )

    def _schedule_tracking_target_speed_mps(
        self,
        safe_speed_mps: float,
    ) -> Optional[float]:
        remaining_time_s = self.target_time_s - self.elapsed_s
        if remaining_time_s <= 0.0:
            return None
        if (
            self.remaining_m > DRIVING_SMOOTHNESS.schedule_tracking_activation_distance_m
            and remaining_time_s > DRIVING_SMOOTHNESS.schedule_tracking_activation_time_s
        ):
            return None
        target_speed_mps = 2.0 * self.remaining_m / (
            remaining_time_s + DRIVING_SMOOTHNESS.schedule_tracking_time_buffer_s
        )
        return min(
            max(0.0, safe_speed_mps - DRIVING_SMOOTHNESS.speed_hold_target_buffer_mps),
            target_speed_mps,
        )

    def _append_trajectory_point(
        self,
        requested_action: int,
        applied_action: int,
        force_n: float,
        resistance_n: float,
        segment: Optional[TrackSegment] = None,
    ) -> None:
        segment = segment or self.section.segment_at(self.position_m)
        self.trajectory.append(
            TrajectoryPoint(
                self.section.name,
                self.elapsed_s,
                self.position_m,
                self.speed_mps * 3.6,
                force_n / 1000.0,
                resistance_n / 1000.0,
                segment.speed_limit_kmh,
                segment.gradient_per_mille,
                segment.radius_m,
                requested_action,
                applied_action,
                self.speed_regulation_mode if self.speed_regulation_active else self.current_regime,
                self.traction_kwh,
                self.regenerative_kwh,
            )
        )

    def step(
        self,
        action: int,
        continuous_force_n: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self.done:
            raise RuntimeError("step() called after lower environment terminated")
        cfg = self.dynamics.config
        segment = self.section.segment_at(self.position_m)
        safe_speed = self._safe_speed_mps()
        track_safe_speed, _, _ = self._track_speed_limit_preview()
        requested_action = int(action)
        requested_speed_hold_target_mps = self.requested_speed_hold_target_mps
        self.requested_speed_hold_target_mps = None
        self.speed_regulation_active = False
        self.speed_regulation_mode = None
        schedule_tracking_target_mps = self._schedule_tracking_target_speed_mps(
            safe_speed,
        )
        speed_limit_hold_target_mps: Optional[float] = None
        requested_action, expert_intervened = self._apply_expert_knowledge(
            requested_action,
            track_safe_speed,
        )
        if expert_intervened:
            self.expert_interventions += 1
            continuous_force_n = None
        if self.speed_mps >= max(
            0.0,
            track_safe_speed
            - DRIVING_SMOOTHNESS.speed_hold_activation_margin_mps,
        ):
            speed_limit_hold_target_mps = max(
                0.0,
                track_safe_speed
                - DRIVING_SMOOTHNESS.speed_hold_target_buffer_mps,
            )
            if self._regime_for_action(requested_action) == "traction":
                requested_action = self.ACTION_COAST
        self.pending_transition_penalty = 0.0
        applied_action = self._apply_operational_constraints(requested_action, safe_speed)
        resistance_n = self.dynamics.total_resistance_n(self.speed_mps, segment)
        action_space_lower_force_n, action_space_upper_force_n = (
            self.acceleration_allowed_force_bounds_n(resistance_n)
        )
        force_n = self._requested_force_n(applied_action)
        if continuous_force_n is not None and applied_action == requested_action:
            force_n = self._clip_continuous_force_n(float(continuous_force_n), applied_action)
        force_filter_immediate = False
        if self.terminal_braking_latched:
            force_n = self._terminal_braking_force_n(resistance_n)
            force_filter_immediate = True
        elif (
            self.safety_brake_latched
            and self.speed_mps > track_safe_speed + 2.0
        ):
            force_n = -cfg.max_braking_force_n
            force_filter_immediate = True
        elif self.safety_brake_latched:
            force_n = self._speed_limit_braking_force_n(
                resistance_n,
                track_safe_speed,
            )
            force_filter_immediate = True
        elif requested_speed_hold_target_mps is not None:
            self.speed_regulation_active = True
            self.speed_regulation_mode = "downhill_hold"
            force_n = self._speed_hold_force_n(
                resistance_n,
                requested_speed_hold_target_mps,
            )
        elif schedule_tracking_target_mps is not None:
            self.speed_regulation_active = True
            self.speed_regulation_mode = "schedule_hold"
            force_n = self._speed_hold_force_n(
                resistance_n,
                schedule_tracking_target_mps,
                DRIVING_SMOOTHNESS.schedule_tracking_response_time_s,
                DRIVING_SMOOTHNESS.schedule_tracking_max_traction_ratio,
                DRIVING_SMOOTHNESS.schedule_tracking_max_braking_ratio,
            )
        elif speed_limit_hold_target_mps is not None:
            self.speed_regulation_active = True
            self.speed_regulation_mode = "speed_hold"
            force_n = self._speed_hold_force_n(
                resistance_n,
                speed_limit_hold_target_mps,
            )
        if self.speed_regulation_active:
            self.speed_regulation_steps += 1
        force_n, force_command_changed = self._filter_force_command_n(
            force_n,
            force_filter_immediate,
        )
        commanded_force_n = force_n
        force_n = self.clip_force_to_action_space_n(
            force_n,
            resistance_n,
        )
        acceleration_limited = abs(force_n - commanded_force_n) > EPS
        filtered_action = self.action_for_force_n(force_n)
        if filtered_action != applied_action:
            applied_action = self._set_applied_action(filtered_action)
        else:
            applied_action = filtered_action
        acceleration, moving_time, new_speed, distance = self._predict_step_motion(
            force_n,
            resistance_n,
        )
        force_jump_ratio = abs(force_n - self.previous_force_n) / max(
            cfg.max_traction_force_n,
            cfg.max_braking_force_n,
            EPS,
        )
        self.force_jump_index += force_jump_ratio
        force_jump_penalty = LOWER_REWARD.force_change_weight * force_jump_ratio
        step_smoothness_penalty = self.pending_transition_penalty + force_jump_penalty
        self.smoothness_penalty += step_smoothness_penalty
        self.previous_force_n = force_n

        jerk_mps3 = (
            acceleration - self.previous_acceleration_mps2
        ) / max(cfg.dt_s, EPS)
        jerk_increment = abs(jerk_mps3) * cfg.dt_s
        comfort_penalty = LOWER_REWARD.jerk_weight * jerk_increment
        self.previous_acceleration_mps2 = acceleration
        self.jerk_integral_mps3_s += jerk_increment
        self.comfort_penalty += comfort_penalty

        if distance > self.remaining_m:
            ratio = self.remaining_m / max(distance, EPS)
            distance = self.remaining_m
            elapsed_increment = moving_time * ratio
            new_speed = math.sqrt(max(0.0, self.speed_mps**2 + 2.0 * acceleration * distance))
        else:
            elapsed_increment = cfg.dt_s

        work_j = force_n * distance
        traction_increment = max(work_j, 0.0) / cfg.traction_efficiency / KWH_IN_JOULES
        regen_increment = max(-work_j, 0.0) * cfg.regenerative_efficiency / KWH_IN_JOULES
        self.traction_kwh += traction_increment
        self.regenerative_kwh += regen_increment
        self.position_m += distance
        self.speed_mps = new_speed
        if self.position_m >= self.section.end_m - 1e-6:
            self.position_m = self.section.end_m
            self.speed_mps = 0.0
        self.elapsed_s += elapsed_increment
        self.step_count += 1
        self.regime_elapsed_s += elapsed_increment
        if force_command_changed:
            self.force_command_elapsed_s = 0.0
        else:
            self.force_command_elapsed_s += elapsed_increment
        if self.current_regime == "coast":
            self.post_brake_coast_remaining_s = max(
                0.0,
                self.post_brake_coast_remaining_s - elapsed_increment,
            )
        self._append_trajectory_point(requested_action, applied_action, force_n, resistance_n, segment)

        section_length_km = max(
            self.section.length_m / 1000.0,
            LOWER_REWARD.traction_energy_reference_km,
            EPS,
        )
        normalized_traction_increment = traction_increment / section_length_km
        spatial_progress = 1.0 - self.remaining_m / max(self.section.length_m, EPS)
        temporal_progress = min(1.0, self.elapsed_s / max(self.target_time_s, EPS))
        schedule_progress_error = abs(spatial_progress - temporal_progress)
        schedule_progress_component = (
            -LOWER_REWARD.schedule_progress_weight
            * schedule_progress_error
        )
        traction_component = (
            -LOWER_REWARD.traction_energy_weight
            * normalized_traction_increment
        )
        force_change_component = -force_jump_penalty
        comfort_component = -comfort_penalty + force_change_component
        punctuality_component = 0.0
        premature_stop_component = 0.0
        timeout_component = 0.0
        speed_limit_violation_kmh = max(
            0.0,
            self.speed_mps * 3.6
            - segment.speed_limit_kmh * self.speed_limit_penalty_ratio,
        )
        speed_limit_component = (
            -LOWER_REWARD.speed_limit_violation_weight
            * speed_limit_violation_kmh
        )
        reward = (
            traction_component
            + comfort_component
            + schedule_progress_component
            + speed_limit_component
        )
        if self.position_m >= self.section.end_m - 1e-6:
            self.position_m = self.section.end_m
            self.speed_mps = 0.0
            self.done = True
            punctuality_component = (
                -LOWER_REWARD.punctuality_weight
                * abs(self.elapsed_s - self.target_time_s)
            )
            reward += punctuality_component
        elif self._is_premature_stop():
            self.done = True
            self.timed_out = True
            premature_stop_component = -LOWER_REWARD.premature_stop_penalty
            punctuality_component = (
                -LOWER_REWARD.punctuality_weight
                * abs(self.elapsed_s - self.target_time_s)
            )
            reward += premature_stop_component + punctuality_component
        elif self.step_count >= self.max_steps:
            self.done = True
            self.timed_out = True
            timeout_component = -LOWER_REWARD.timeout_penalty
            punctuality_component = (
                -LOWER_REWARD.punctuality_weight
                * abs(self.elapsed_s - self.target_time_s)
            )
            reward += timeout_component + punctuality_component
        self.traction_reward_component += traction_component
        self.punctuality_reward_component += (
            punctuality_component + schedule_progress_component
        )
        self.comfort_reward_component += comfort_component
        self.episode_reward += reward
        positive_force_scale = max(
            self.dynamics.max_traction_force_n(max(self.speed_mps, EPS)),
            EPS,
        )
        negative_force_scale = max(cfg.max_braking_force_n, EPS)
        return self.observation(), reward, self.done, {
            "traction_increment_kwh": traction_increment,
            "normalized_traction_increment": normalized_traction_increment,
            "regen_increment_kwh": regen_increment,
            "force_n": force_n,
            "commanded_force_n": commanded_force_n,
            "acceleration_limited": acceleration_limited,
            "max_acceleration_mps2": cfg.max_acceleration_mps2,
            "max_deceleration_mps2": cfg.max_deceleration_mps2,
            "action_space_lower_force_n": action_space_lower_force_n,
            "action_space_upper_force_n": action_space_upper_force_n,
            "action_space_lower_force_kn": action_space_lower_force_n / 1000.0,
            "action_space_upper_force_kn": action_space_upper_force_n / 1000.0,
            "force_ratio": (
                force_n / positive_force_scale
                if force_n >= 0.0
                else force_n / negative_force_scale
            ),
            "jerk_mps3": jerk_mps3,
            "comfort_penalty": comfort_penalty,
            "force_jump_penalty": force_jump_penalty,
            "traction_reward_component": traction_component,
            "punctuality_reward_component": punctuality_component,
            "schedule_progress_component": schedule_progress_component,
            "schedule_progress_error": schedule_progress_error,
            "speed_limit_violation_kmh": speed_limit_violation_kmh,
            "speed_limit_penalty_ratio": self.speed_limit_penalty_ratio,
            "speed_limit_penalty_threshold_kmh": (
                segment.speed_limit_kmh * self.speed_limit_penalty_ratio
            ),
            "speed_limit_component": speed_limit_component,
            "force_change_component": force_change_component,
            "comfort_reward_component": comfort_component,
            "premature_stop_component": premature_stop_component,
            "timeout_component": timeout_component,
            "premature_stop": self._is_premature_stop(),
        }

    def result(self) -> LowerResult:
        timetable_error_s = self.elapsed_s - self.target_time_s
        max_speed_limit_violation_kmh = max(
            (
                max(
                    0.0,
                    point.speed_kmh
                    - point.speed_limit_kmh * self.speed_limit_penalty_ratio,
                )
                for point in self.trajectory
            ),
            default=0.0,
        )
        operationally_feasible = not self.timed_out and self.speed_mps <= 2.0
        strict_timetable_feasible = (
            operationally_feasible
            and abs(timetable_error_s) <= PRIMARY_ARRIVAL_TOLERANCE_S
        )
        relaxed_timetable_feasible = (
            operationally_feasible
            and abs(timetable_error_s) <= RELAXED_ARRIVAL_TOLERANCE_S
        )
        return LowerResult(
            section_name=self.section.name,
            target_time_s=self.target_time_s,
            actual_time_s=self.elapsed_s,
            traction_kwh=self.traction_kwh,
            regenerative_kwh=self.regenerative_kwh,
            net_energy_kwh=self.traction_kwh - self.regenerative_kwh,
            arrival_speed_mps=self.speed_mps,
            feasible=relaxed_timetable_feasible,
            timed_out=self.timed_out,
            regime_switches=self.regime_switches,
            traction_brake_transitions=self.traction_brake_transitions,
            safety_brake_overrides=self.safety_brake_overrides,
            traction_coast_transitions=self.traction_coast_transitions,
            short_dwell_transitions=self.short_dwell_transitions,
            force_jump_index=self.force_jump_index,
            smoothness_penalty=self.smoothness_penalty,
            jerk_integral_mps3_s=self.jerk_integral_mps3_s,
            comfort_penalty=self.comfort_penalty,
            traction_reward_component=self.traction_reward_component,
            punctuality_reward_component=self.punctuality_reward_component,
            comfort_reward_component=self.comfort_reward_component,
            max_speed_limit_violation_kmh=max_speed_limit_violation_kmh,
            speed_regulation_steps=self.speed_regulation_steps,
            strict_timetable_feasible=strict_timetable_feasible,
            relaxed_timetable_feasible=relaxed_timetable_feasible,
            accepted_arrival_tolerance_s=(
                PRIMARY_ARRIVAL_TOLERANCE_S
                if strict_timetable_feasible
                else RELAXED_ARRIVAL_TOLERANCE_S
                if relaxed_timetable_feasible
                else None
            ),
            episode_reward=self.episode_reward,
            expert_interventions=self.expert_interventions,
            trajectory=list(self.trajectory),
        )

    def rollout(
        self,
        controller: Optional["DQNAgent"] = None,
        target_time_s: float = 1.0,
        training: bool = False,
        epsilon: float = 0.0,
        heuristic_probability: float = 0.0,
        aggressiveness: float = 1.0,
    ) -> LowerResult:
        self.reset(target_time_s)
        state = (
            controller.observation_from_environment(self)
            if controller is not None and hasattr(controller, "observation_from_environment")
            else self.observation()
        )
        while not self.done:
            continuous_force_n = None
            if controller is None or (training and random.random() < heuristic_probability):
                action = self.mask_invalid_discrete_action(
                    self.heuristic_action(aggressiveness)
                )
            else:
                if hasattr(controller, "select_action_and_force"):
                    action, continuous_force_n = controller.select_action_and_force(self)
                else:
                    action = controller.select_action(
                        state,
                        epsilon,
                        valid_actions=self.valid_discrete_actions(),
                    )
            _, reward, done, step_info = self.step(action, continuous_force_n)
            done = self.done
            next_state = (
                controller.observation_from_environment(self)
                if controller is not None and hasattr(controller, "observation_from_environment")
                else self.observation()
            )
            if training and controller is not None:
                if hasattr(controller, "remember"):
                    replay_action = (
                        step_info.get("force_ratio", action)
                        if hasattr(controller, "select_action_and_force")
                        else action
                    )
                    if hasattr(controller, "select_action_and_force"):
                        controller.remember(state, replay_action, reward, next_state, done)
                    else:
                        controller.remember(
                            state,
                            replay_action,
                            reward,
                            next_state,
                            done,
                            self.valid_discrete_actions() if not done else [],
                        )
                    controller.update()
            state = next_state
        return self.result()


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        samples = random.sample(self.buffer, batch_size)
        transition_width = len(samples[0])
        if transition_width == 5:
            state, action, reward, next_state, done = zip(*samples)
            next_valid_action_mask = None
        else:
            state, action, reward, next_state, done, next_valid_actions = zip(*samples)
            if all(
                len(actions) > 0
                and all(isinstance(value, (bool, np.bool_)) for value in actions)
                for actions in next_valid_actions
            ):
                next_valid_action_mask = np.asarray(
                    next_valid_actions,
                    dtype=bool,
                )
            else:
                action_dim = max(
                    (max(actions) if actions else 0)
                    for actions in next_valid_actions
                ) + 1
                action_dim = max(action_dim, 1)
                next_valid_action_mask = np.zeros(
                    (batch_size, action_dim),
                    dtype=bool,
                )
                for row_idx, actions in enumerate(next_valid_actions):
                    if not actions:
                        next_valid_action_mask[row_idx, :] = True
                    else:
                        for action_idx in actions:
                            if 0 <= int(action_idx) < action_dim:
                                next_valid_action_mask[row_idx, int(action_idx)] = True
        result = (
            torch.as_tensor(np.asarray(state), dtype=torch.float32),
            torch.as_tensor(action, dtype=torch.long).unsqueeze(1),
            torch.as_tensor(reward, dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.asarray(next_state), dtype=torch.float32),
            torch.as_tensor(done, dtype=torch.float32).unsqueeze(1),
        )
        if next_valid_action_mask is not None:
            result = result + (
                torch.as_tensor(next_valid_action_mask, dtype=torch.bool),
            )
        return result

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        replay_capacity: int = 100000,
        batch_size: int = 128,
        target_sync_interval: int = 250,
        device_name: Optional[str] = None,
    ) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval
        if device_name is None or device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.policy = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(replay_capacity)
        self.update_count = 0
        self.last_loss: Optional[float] = None

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        valid_actions: Optional[Sequence[int]] = None,
    ) -> int:
        valid_action_list = (
            [int(action) for action in valid_actions]
            if valid_actions is not None and len(valid_actions) > 0
            else list(range(self.action_dim))
        )
        if random.random() < epsilon:
            return int(random.choice(valid_action_list))
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy(tensor).squeeze(0)
            mask = torch.full_like(q_values, -torch.inf)
            mask[torch.as_tensor(valid_action_list, dtype=torch.long, device=self.device)] = 0.0
            return int((q_values + mask).argmax(dim=0).item())

    def remember(self, *transition) -> None:
        if len(transition) == 6:
            state, action, reward, next_state, done, next_valid_actions = transition
            if next_valid_actions:
                mask = [False] * self.action_dim
                for action_idx in next_valid_actions:
                    if 0 <= int(action_idx) < self.action_dim:
                        mask[int(action_idx)] = True
            else:
                mask = [True] * self.action_dim
            transition = (state, action, reward, next_state, done, tuple(mask))
        self.replay.push(*transition)

    def update(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None
        sample = self.replay.sample(self.batch_size)
        if len(sample) == 5:
            state, action, reward, next_state, done = sample
            next_valid_action_mask = None
        else:
            state, action, reward, next_state, done, next_valid_action_mask = sample
        state, action = state.to(self.device), action.to(self.device)
        reward, next_state, done = reward.to(self.device), next_state.to(self.device), done.to(self.device)
        q_value = self.policy(state).gather(1, action)
        with torch.no_grad():
            next_q_values = self.target(next_state)
            if next_valid_action_mask is not None:
                mask = next_valid_action_mask.to(self.device)
                if mask.shape[1] < next_q_values.shape[1]:
                    padded = torch.ones_like(next_q_values, dtype=torch.bool)
                    padded[:, : mask.shape[1]] = mask
                    mask = padded
                elif mask.shape[1] > next_q_values.shape[1]:
                    mask = mask[:, : next_q_values.shape[1]]
                next_q_values = next_q_values.masked_fill(~mask, -1.0e9)
            expected = reward + self.gamma * next_q_values.max(dim=1, keepdim=True).values * (1.0 - done)
        loss = F.smooth_l1_loss(q_value, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_sync_interval == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss


def _sample_train_mass_t(
    mass_min_t: float,
    mass_max_t: float,
    anchor_mass_probability: float,
    boundary_mass_probability: float,
) -> float:
    anchor_masses_t = [config.mass_kg / 1000.0 for config in TRAIN_CONFIGS]
    boundary_probability = max(0.0, min(1.0, float(boundary_mass_probability)))
    anchor_probability = max(0.0, min(1.0, float(anchor_mass_probability)))
    draw = random.random()
    if draw < boundary_probability:
        return random.choice([mass_min_t, mass_max_t])
    if draw < boundary_probability + anchor_probability:
        anchors_in_range = [
            mass for mass in anchor_masses_t if mass_min_t <= mass <= mass_max_t
        ]
        return random.choice(anchors_in_range or anchor_masses_t)
    return random.uniform(mass_min_t, mass_max_t)


def _mass_sampled_environment(
    environment: LowerSectionEnvironment,
    train_name: str,
    random_mass_training: bool,
    mass_min_t: float,
    mass_max_t: float,
    anchor_mass_probability: float,
    boundary_mass_probability: float,
) -> Tuple[LowerSectionEnvironment, float]:
    sampled_mass_t = environment.dynamics.config.mass_kg / 1000.0
    if not random_mass_training:
        return environment, sampled_mass_t
    sampled_mass_t = _sample_train_mass_t(
        mass_min_t,
        mass_max_t,
        anchor_mass_probability,
        boundary_mass_probability,
    )
    sampled_config = create_train_config_from_mass(
        name=f"{train_name} sampled {sampled_mass_t:.1f}t",
        total_mass_t=sampled_mass_t,
        max_braking_force_kn=environment.dynamics.config.max_braking_force_n / 1000.0,
    )
    return (
        LowerSectionEnvironment(
            environment.section,
            TrainDynamics(sampled_config),
            speed_limit_penalty_ratio=environment.speed_limit_penalty_ratio,
        ),
        sampled_mass_t,
    )


def train_lower_agents(
    environments: Sequence[LowerSectionEnvironment],
    controller: DQNAgent,
    initial_timetable: np.ndarray,
    episodes: int,
    target_span_s: int,
    train_name: str = "Train 1",
    random_mass_training: bool = False,
    mass_min_t: float = 235.0,
    mass_max_t: float = 275.0,
    anchor_mass_probability: float = 0.15,
    boundary_mass_probability: float = 0.05,
    log_interval: int = 0,
    environment_tags: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    print(f"Training lower DQN policy for {train_name}: {episodes} episodes")
    history: List[Dict[str, Any]] = []
    for episode in range(episodes):
        idx = random.randrange(len(environments))
        tag = environment_tags[idx] if environment_tags is not None else {}
        environment, sampled_mass_t = _mass_sampled_environment(
            environments[idx],
            train_name,
            random_mass_training,
            float(mass_min_t),
            float(mass_max_t),
            anchor_mass_probability,
            boundary_mass_probability,
        )
        target = float(initial_timetable[idx] + random.randint(-target_span_s, target_span_s))
        epsilon = 0.90 + (0.05 - 0.90) * min(1.0, episode / max(episodes - 1, 1))
        result = environment.rollout(
            controller,
            target,
            training=True,
            epsilon=epsilon,
            heuristic_probability=0.30,
        )
        sampled_config = environment.dynamics.config
        section_length_km = max(environment.section.length_m / 1000.0, EPS)
        history.append(
            {
                "episode": episode + 1,
                "train_name": train_name,
                "lower_algorithm": "dqn",
                "source_train_id": tag.get("train_id", ""),
                "source_train_name": tag.get("train_name", ""),
                "direction": tag.get("direction", ""),
                "speed_limit_penalty_ratio": environment.speed_limit_penalty_ratio,
                "random_mass_training": random_mass_training,
                "sampled_total_mass_t": sampled_mass_t,
                "sampled_motor_car_mass_t": sampled_config.effective_motor_car_mass_kg / 1000.0,
                "sampled_trailer_car_mass_t": sampled_config.effective_trailer_car_mass_kg / 1000.0,
                "sampled_max_traction_force_kn": sampled_config.max_traction_force_n / 1000.0,
                "sampled_max_traction_power_kw": sampled_config.max_traction_power_w / 1000.0,
                "sampled_davis_a": sampled_config.davis_a,
                "sampled_davis_b": sampled_config.davis_b,
                "sampled_davis_c": sampled_config.davis_c,
                "section_name": result.section_name,
                "target_time_s": target,
                "actual_time_s": result.actual_time_s,
                "timetable_error_s": result.timetable_error_s,
                "net_energy_kwh": result.net_energy_kwh,
                "net_energy_kwh_per_km": result.net_energy_kwh / section_length_km,
                "traction_kwh": result.traction_kwh,
                "traction_kwh_per_km": result.traction_kwh / section_length_km,
                "feasible": result.feasible,
                "regime_switches": result.regime_switches,
                "traction_coast_transitions": result.traction_coast_transitions,
                "short_dwell_transitions": result.short_dwell_transitions,
                "force_jump_index": result.force_jump_index,
                "smoothness_penalty": result.smoothness_penalty,
                "jerk_integral_mps3_s": result.jerk_integral_mps3_s,
                "jerk_integral_mps3_s_per_km": (
                    result.jerk_integral_mps3_s / section_length_km
                ),
                "comfort_penalty": result.comfort_penalty,
                "traction_reward_component": result.traction_reward_component,
                "punctuality_reward_component": result.punctuality_reward_component,
                "comfort_reward_component": result.comfort_reward_component,
                "max_speed_limit_violation_kmh": (
                    result.max_speed_limit_violation_kmh
                ),
                "speed_regulation_steps": result.speed_regulation_steps,
                "episode_reward": result.episode_reward,
                "episode_reward_per_km": result.episode_reward / section_length_km,
                "epsilon": epsilon,
                "loss": controller.last_loss,
            }
        )
        effective_log_interval = log_interval if log_interval > 0 else max(1, episodes // 10)
        if episode == 0 or (episode + 1) % effective_log_interval == 0:
            print(
                f"  lower episode {episode + 1:5d}/{episodes}: "
                f"section={result.section_name:9s} target={target:6.1f}s "
                f"mass={sampled_mass_t:5.1f}t "
                f"davis=({sampled_config.davis_a:.1f}, {sampled_config.davis_b:.2f}, {sampled_config.davis_c:.4f}) "
                f"actual={result.actual_time_s:7.1f}s traction={result.traction_kwh:7.2f}kWh "
                f"error={result.timetable_error_s:+5.1f}s "
                f"jerk={result.jerk_integral_mps3_s:7.2f} "
                f"reward/km={result.episode_reward / section_length_km:7.2f}"
            )
    return history


def lower_result_score(result: LowerResult) -> float:
    return (
        LOWER_REWARD.traction_energy_weight * result.traction_kwh
        + LOWER_REWARD.punctuality_weight * abs(result.timetable_error_s)
        + LOWER_REWARD.jerk_weight * result.jerk_integral_mps3_s
    )


def evaluate_lower_agent(
    environment: LowerSectionEnvironment,
    target_time_s: float,
    learned_controller: Optional[DQNAgent] = None,
) -> LowerResult:
    candidates = [
        environment.rollout(None, target_time_s, aggressiveness=value)
        for value in (
            0.90,
            0.95,
            1.00,
            1.02,
            1.04,
            1.06,
            1.08,
            1.10,
            1.20,
            1.30,
            1.40,
        )
    ]
    if learned_controller is not None:
        candidates.append(environment.rollout(learned_controller, target_time_s))
    strict_candidates = [result for result in candidates if result.strict_timetable_feasible]
    if strict_candidates:
        return min(strict_candidates, key=lower_result_score)

    relaxed_candidates = [result for result in candidates if result.relaxed_timetable_feasible]
    if relaxed_candidates:
        return min(relaxed_candidates, key=lower_result_score)

    return min(
        candidates,
        key=lambda result: (
            result.timed_out or result.arrival_speed_mps > 2.0,
            abs(result.timetable_error_s),
            lower_result_score(result),
        ),
    )


def evaluate_timetable(
    environments: Sequence[LowerSectionEnvironment],
    timetable: Sequence[float],
    learned_controller: Optional[DQNAgent] = None,
) -> List[LowerResult]:
    return [
        evaluate_lower_agent(environment, float(target), learned_controller)
        for environment, target in zip(environments, timetable)
    ]


def validate_initial_timetable(
    environments: Sequence[LowerSectionEnvironment],
    timetable: np.ndarray,
) -> List[LowerResult]:
    print("Initial timetable heuristic rollouts")
    results = evaluate_timetable(environments, timetable)
    for result in results:
        print(
            f"  {result.section_name:9s}: target={result.target_time_s:6.1f}s "
            f"actual={result.actual_time_s:7.1f}s error={result.timetable_error_s:+7.1f}s "
            f"net={result.net_energy_kwh:7.2f}kWh switches={result.regime_switches:2d} "
            f"T/C={result.traction_coast_transitions:2d} "
            f"short={result.short_dwell_transitions:2d} "
            f"hold={result.speed_regulation_steps:3d} "
            f"T/B={result.traction_brake_transitions:2d} feasible={result.feasible} "
            f"tolerance={result.accepted_arrival_tolerance_s}"
        )
    print(
        f"  total: timetable={float(np.sum(timetable)):.1f}s "
        f"traction={sum(item.traction_kwh for item in results):.2f}kWh "
        f"regen={sum(item.regenerative_kwh for item in results):.2f}kWh "
        f"net={sum(item.net_energy_kwh for item in results):.2f}kWh"
    )
    return results


@dataclass(frozen=True)
class ResponsePoint:
    requested_time_s: float
    actual_time_s: float
    traction_kwh: float
    regenerative_kwh: float
    net_energy_kwh: float
    timetable_error_s: float
    feasible: bool
    strict_timetable_feasible: bool = False
    accepted_arrival_tolerance_s: Optional[float] = None
    traction_profile_kwh: Tuple[float, ...] = field(default_factory=tuple, repr=False)
    regenerative_profile_kwh: Tuple[float, ...] = field(default_factory=tuple, repr=False)
    profile_bin_s: float = field(default=5.0, repr=False)


def result_energy_profiles(
    result: LowerResult,
    bin_s: float = 5.0,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    bin_count = max(1, int(math.ceil(result.actual_time_s / bin_s)))
    traction = np.zeros(bin_count, dtype=float)
    regenerative = np.zeros(bin_count, dtype=float)
    previous = result.trajectory[0]
    for point in result.trajectory[1:]:
        bin_idx = min(bin_count - 1, int(previous.elapsed_s // bin_s))
        traction[bin_idx] += max(0.0, point.traction_kwh - previous.traction_kwh)
        regenerative[bin_idx] += max(0.0, point.regenerative_kwh - previous.regenerative_kwh)
        previous = point
    return tuple(traction.tolist()), tuple(regenerative.tolist())


class SectionResponseTable:
    """Lower-to-upper message: E_min(T), feasibility and real power profiles."""

    def __init__(
        self,
        section_name: str,
        points: Sequence[ResponsePoint],
        train_name: str = "Train 1",
    ) -> None:
        self.train_name = train_name
        self.section_name = section_name
        self.points = sorted(points, key=lambda item: item.requested_time_s)
        self.times = [item.requested_time_s for item in self.points]

    def lookup(self, requested_time_s: float) -> ResponsePoint:
        idx = bisect.bisect_left(self.times, requested_time_s)
        if idx <= 0:
            return self.points[0]
        if idx >= len(self.points):
            return self.points[-1]
        left, right = self.points[idx - 1], self.points[idx]
        return left if abs(left.requested_time_s - requested_time_s) <= abs(right.requested_time_s - requested_time_s) else right


def build_response_tables(
    environments: Sequence[LowerSectionEnvironment],
    initial_timetable: np.ndarray,
    time_span_s: float,
    time_step_s: float,
    learned_controller: Optional[DQNAgent] = None,
    train_name: str = "Train 1",
) -> List[SectionResponseTable]:
    print(f"Building lower-agent E_min(T) response tables for {train_name}")
    if time_span_s < 0.0 or time_step_s <= 0.0:
        raise ValueError("Response-table span must be nonnegative and step must be positive")
    interval_count = 2.0 * time_span_s / time_step_s
    if not math.isclose(interval_count, round(interval_count), abs_tol=1e-8):
        raise ValueError("Response-table step must divide the full sampling interval")
    tables: List[SectionResponseTable] = []
    for idx, environment in enumerate(environments):
        base_time = float(initial_timetable[idx])
        points: List[ResponsePoint] = []
        sample_count = int(round(interval_count)) + 1
        targets = np.linspace(
            base_time - time_span_s,
            base_time + time_span_s,
            sample_count,
        )
        for target in targets:
            target = float(round(float(target), 6))
            result = evaluate_lower_agent(environment, target, learned_controller)
            traction_profile, regenerative_profile = result_energy_profiles(result)
            points.append(
                ResponsePoint(
                    requested_time_s=float(target),
                    actual_time_s=result.actual_time_s,
                    traction_kwh=result.traction_kwh,
                    regenerative_kwh=result.regenerative_kwh,
                    net_energy_kwh=result.net_energy_kwh,
                    timetable_error_s=result.timetable_error_s,
                    feasible=result.feasible,
                    strict_timetable_feasible=result.strict_timetable_feasible,
                    accepted_arrival_tolerance_s=result.accepted_arrival_tolerance_s,
                    traction_profile_kwh=traction_profile,
                    regenerative_profile_kwh=regenerative_profile,
                )
            )
        table = SectionResponseTable(environment.section.name, points, train_name=train_name)
        tables.append(table)
        base = table.lookup(float(base_time))
        print(
            f"  {environment.section.name:9s}: samples={len(points):2d} "
            f"base={base_time:6.1f}s actual={base.actual_time_s:7.1f}s "
            f"net={base.net_energy_kwh:7.2f}kWh feasible={base.feasible} "
            f"tolerance={base.accepted_arrival_tolerance_s}"
        )
    return tables
