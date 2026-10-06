"""Model-based lower-level eco-driving based on the CO_V2 convex relaxation.

The optimizer works in the spatial domain.  It deliberately keeps the
relaxation variables used by the research-group prototype (alpha >= 1/v and
beta >= v**2), but all values returned to the upper layer are recomputed from
the physical speed trajectory.  In particular, upper-layer energy and running
time never come directly from the relaxed objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import EPS, KWH_IN_JOULES
from .line_profile import StationSection
from .lower_control import LowerResult, LowerSectionEnvironment, TrajectoryPoint
from .train_model import TrainConfig, TrainDynamics


@dataclass(frozen=True)
class ConvexLowerOptions:
    spatial_segments: int = 600
    speed_limit_ratio: float = 0.8
    numerical_terminal_speed_mps: float = 0.1
    arrival_tolerance_s: float = 0.5
    alpha_relative_tolerance: float = 1e-3
    beta_normalized_tolerance: float = 1e-3
    physical_speed_tolerance_mps: float = 0.02
    physical_force_tolerance_n: float = 100.0
    physical_power_tolerance_w: float = 1000.0
    physical_acceleration_tolerance_mps2: float = 1e-4
    output_flag: bool = False
    time_limit_s: Optional[float] = None

    def validate(self) -> None:
        if self.spatial_segments < 2:
            raise ValueError("spatial_segments must be at least 2")
        if not 0.0 < self.speed_limit_ratio <= 1.0:
            raise ValueError("speed_limit_ratio must be in (0, 1]")
        if self.numerical_terminal_speed_mps <= 0.0:
            raise ValueError("numerical_terminal_speed_mps must be positive")
        if self.arrival_tolerance_s < 0.0:
            raise ValueError("arrival_tolerance_s must be nonnegative")


@dataclass
class ConvexLowerResult:
    status: str
    section_name: str
    target_time_s: float
    solver_status_code: Optional[int] = None
    solver_runtime_s: float = 0.0
    optimized_objective_kwh: Optional[float] = None
    actual_time_s: Optional[float] = None
    actual_traction_kwh: Optional[float] = None
    actual_regenerative_kwh: Optional[float] = None
    actual_net_energy_kwh: Optional[float] = None
    position_m: np.ndarray = field(default_factory=lambda: np.empty(0))
    speed_kmh: np.ndarray = field(default_factory=lambda: np.empty(0))
    optimized_force_kn: np.ndarray = field(default_factory=lambda: np.empty(0))
    recomputed_force_kn: np.ndarray = field(default_factory=lambda: np.empty(0))
    recomputed_resistance_kn: np.ndarray = field(default_factory=lambda: np.empty(0))
    segment_time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    alpha_relative_error: np.ndarray = field(default_factory=lambda: np.empty(0))
    beta_absolute_error_m2ps2: np.ndarray = field(default_factory=lambda: np.empty(0))
    max_alpha_relative_error: Optional[float] = None
    max_beta_normalized_error: Optional[float] = None
    max_force_gap_kn: Optional[float] = None
    timetable_error_s: Optional[float] = None
    feasible: bool = False
    validation_messages: List[str] = field(default_factory=list)
    numerical_terminal_speed_mps: float = 0.1
    _internal_speed_mps: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    _delta_height_m: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    _node_speed_limits_mps: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    @property
    def has_solution(self) -> bool:
        return self.position_m.size > 0 and self._internal_speed_mps.size > 0


def _status_name(status_code: int) -> str:
    try:
        import gurobipy as gp
    except ImportError:
        return "solver_unavailable"
    names = {
        gp.GRB.OPTIMAL: "optimal",
        gp.GRB.INFEASIBLE: "infeasible",
        gp.GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        gp.GRB.UNBOUNDED: "unbounded",
        gp.GRB.TIME_LIMIT: "time_limit",
        gp.GRB.SUBOPTIMAL: "suboptimal",
        gp.GRB.NUMERIC: "numerical_failure",
        gp.GRB.INTERRUPTED: "interrupted",
    }
    return names.get(status_code, f"solver_status_{status_code}")


def _node_speed_limits(
    section: StationSection,
    positions_m: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """Use the stricter adjacent limit at every track discontinuity."""
    limits = []
    for position in positions_m:
        probes = [position]
        if position > section.start_m:
            probes.append(max(section.start_m, position - 1e-5))
        if position < section.end_m:
            probes.append(min(section.end_m - 1e-6, position + 1e-5))
        raw_limit = min(section.segment_at(probe).speed_limit_kmh for probe in probes)
        limits.append(raw_limit * ratio / 3.6)
    return np.asarray(limits, dtype=float)


def _integrated_height_changes(
    section: StationSection,
    positions_m: np.ndarray,
) -> np.ndarray:
    """Integrate the piecewise-constant gradient over every spatial cell."""
    changes = np.zeros(len(positions_m) - 1, dtype=float)
    for cell_idx, (left, right) in enumerate(zip(positions_m[:-1], positions_m[1:])):
        delta_h = 0.0
        for segment in section.segments:
            overlap = max(0.0, min(right, segment.end_m) - max(left, segment.start_m))
            if overlap <= 0.0:
                continue
            slope = segment.gradient_per_mille / 1000.0
            delta_h += overlap * math.sin(math.atan(slope))
        changes[cell_idx] = delta_h
    return changes


def _physical_recalculation(
    config: TrainConfig,
    positions_m: np.ndarray,
    speeds_mps: np.ndarray,
    delta_height_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    distances = np.diff(positions_m)
    average_speeds = 0.5 * (speeds_mps[:-1] + speeds_mps[1:])
    if np.any(average_speeds <= EPS):
        raise ValueError("A spatial cell has zero average speed; physical time is undefined")
    segment_times = distances / average_speeds
    speed_kmh = average_speeds * 3.6
    resistances = (
        float(config.davis_a)
        + float(config.davis_b) * speed_kmh
        + float(config.davis_c) * speed_kmh**2
    )
    accelerations = (speeds_mps[1:] ** 2 - speeds_mps[:-1] ** 2) / (2.0 * distances)
    gradient_forces = config.mass_kg * config.gravity_mps2 * delta_height_m / distances
    forces = resistances + config.mass_kg * accelerations + gradient_forces

    mechanical_work_j = forces * distances
    traction_j = np.maximum(mechanical_work_j, 0.0) / config.traction_efficiency
    regenerative_j = np.maximum(-mechanical_work_j, 0.0) * config.regenerative_efficiency
    traction_kwh = float(np.sum(traction_j) / KWH_IN_JOULES)
    regenerative_kwh = float(np.sum(regenerative_j) / KWH_IN_JOULES)
    return (
        segment_times,
        resistances,
        forces,
        traction_kwh,
        regenerative_kwh,
        traction_kwh - regenerative_kwh,
    )


class ConvexLowerSolver:
    """Parameterized Gurobi QCP implementation of the CO_V2 model."""

    def __init__(self, options: Optional[ConvexLowerOptions] = None) -> None:
        self.options = options or ConvexLowerOptions()
        self.options.validate()

    def solve(
        self,
        section: StationSection,
        train_config: TrainConfig,
        target_time_s: float,
    ) -> ConvexLowerResult:
        if not math.isfinite(target_time_s) or target_time_s <= 0.0:
            raise ValueError("target_time_s must be finite and positive")
        try:
            import gurobipy as gp
        except ImportError as error:
            return ConvexLowerResult(
                status="solver_unavailable",
                section_name=section.name,
                target_time_s=float(target_time_s),
                validation_messages=[f"gurobipy is unavailable: {error}"],
                numerical_terminal_speed_mps=self.options.numerical_terminal_speed_mps,
            )

        opts = self.options
        n = opts.spatial_segments
        positions = np.linspace(section.start_m, section.end_m, n + 1, dtype=float)
        distances = np.diff(positions)
        node_limits = _node_speed_limits(section, positions, opts.speed_limit_ratio)
        delta_height = _integrated_height_changes(section, positions)
        terminal_speed = opts.numerical_terminal_speed_mps
        if terminal_speed > min(node_limits[0], node_limits[-1]):
            raise ValueError("numerical terminal speed exceeds the endpoint speed limit")

        try:
            model = gp.Model(f"convex_lower_{section.name}")
            model.Params.OutputFlag = int(opts.output_flag)
            # Avoid an ambiguous INF_OR_UNBD classification on fine spatial
            # grids; the upper layer needs a definite feasibility status.
            model.Params.DualReductions = 0
            if opts.time_limit_s is not None:
                model.Params.TimeLimit = float(opts.time_limit_s)

            nodes = range(n + 1)
            cells = range(1, n + 1)
            speed = model.addVars(nodes, lb=0.0, name="speed_mps")
            beta = model.addVars(nodes, lb=0.0, name="speed_square_relaxation")
            alpha = model.addVars(cells, lb=0.0, ub=1.0, name="reciprocal_speed")
            average_speed = model.addVars(cells, lb=0.0, name="average_speed_mps")
            elapsed = model.addVars(cells, lb=0.0, name="elapsed_time_s")
            resistance = model.addVars(cells, lb=0.0, name="davis_resistance_n")
            force = model.addVars(
                cells,
                lb=-float(train_config.max_braking_force_n) / 1000.0,
                ub=float(train_config.max_traction_force_n) / 1000.0,
                name="optimized_force_kn",
            )
            electrical_energy = model.addVars(
                cells, lb=-gp.GRB.INFINITY, name="electrical_energy_kj"
            )

            for node in range(1, n):
                model.addConstr(
                    speed[node] <= float(node_limits[node]),
                    name=f"speed_limit[{node}]",
                )
                model.addConstr(
                    beta[node] <= float(node_limits[node] ** 2),
                    name=f"beta_speed_limit[{node}]",
                )
            model.addConstr(speed[0] == terminal_speed, name="departure_speed")
            model.addConstr(speed[n] == terminal_speed, name="arrival_speed")
            model.addConstr(beta[0] == terminal_speed**2, name="departure_beta")
            model.addConstr(beta[n] == terminal_speed**2, name="arrival_beta")
            model.addConstr(
                gp.quicksum(elapsed[cell] for cell in cells) == target_time_s,
                name="total_running_time",
            )

            for cell in cells:
                ds = float(distances[cell - 1])
                model.addConstr(
                    speed[cell] + speed[cell - 1] - 2.0 * average_speed[cell] == 0.0,
                    name=f"average_speed[{cell}]",
                )
                model.addConstr(
                    elapsed[cell] == ds * alpha[cell],
                    name=f"cell_time[{cell}]",
                )
                model.addQConstr(
                    alpha[cell] * average_speed[cell] >= 1.0,
                    name=f"reciprocal_speed[{cell}]",
                )
                model.addConstr(
                    electrical_energy[cell]
                    >= force[cell] * ds / train_config.traction_efficiency,
                    name=f"traction_energy[{cell}]",
                )
                model.addConstr(
                    electrical_energy[cell]
                    >= force[cell] * ds * train_config.regenerative_efficiency,
                    name=f"braking_energy[{cell}]",
                )
                max_power_kw = train_config.max_traction_power_w / 1000.0
                model.addConstr(
                    force[cell] >= -max_power_kw * alpha[cell],
                    name=f"braking_power[{cell}]",
                )
                model.addConstr(
                    force[cell] <= max_power_kw * alpha[cell],
                    name=f"traction_power[{cell}]",
                )
                model.addConstr(
                    resistance[cell]
                    == float(train_config.davis_a) / 1000.0
                    + float(train_config.davis_b) / 1000.0 * 3.6 * speed[cell]
                    + float(train_config.davis_c) / 1000.0 * 3.6**2 * beta[cell],
                    name=f"davis_resistance[{cell}]",
                )
                model.addConstr(
                    force[cell] * ds
                    - 0.5 * train_config.mass_kg / 1000.0
                    * (beta[cell] - beta[cell - 1])
                    - ds * resistance[cell]
                    - train_config.mass_kg / 1000.0
                    * train_config.gravity_mps2
                    * float(delta_height[cell - 1])
                    == 0.0,
                    name=f"energy_balance[{cell}]",
                )
                spatial_acceleration = 0.5 * (beta[cell] - beta[cell - 1]) / ds
                model.addConstr(
                    spatial_acceleration >= -train_config.max_deceleration_mps2,
                    name=f"deceleration[{cell}]",
                )
                model.addConstr(
                    spatial_acceleration <= train_config.max_acceleration_mps2,
                    name=f"acceleration[{cell}]",
                )
                model.addConstr(
                    speed[cell] - speed[cell - 1]
                    >= -train_config.max_deceleration_mps2 * elapsed[cell],
                    name=f"time_deceleration[{cell}]",
                )
                model.addConstr(
                    speed[cell] - speed[cell - 1]
                    <= train_config.max_acceleration_mps2 * elapsed[cell],
                    name=f"time_acceleration[{cell}]",
                )
            for node in nodes:
                model.addQConstr(
                    speed[node] * speed[node] <= beta[node],
                    name=f"speed_square[{node}]",
                )

            model.setObjective(
                gp.quicksum(electrical_energy[cell] for cell in cells),
                gp.GRB.MINIMIZE,
            )
            model.optimize()
        except gp.GurobiError as error:
            return ConvexLowerResult(
                status="solver_error",
                section_name=section.name,
                target_time_s=float(target_time_s),
                validation_messages=[f"Gurobi error {error.errno}: {error}"],
                numerical_terminal_speed_mps=terminal_speed,
            )

        status_code = int(model.Status)
        status = _status_name(status_code)
        acceptable_solution = (
            status_code in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL}
            and int(model.SolCount) > 0
        )
        if not acceptable_solution:
            messages = [f"Gurobi did not return an optimal solution ({status})"]
            return ConvexLowerResult(
                status=status,
                section_name=section.name,
                target_time_s=float(target_time_s),
                solver_status_code=status_code,
                solver_runtime_s=float(model.Runtime),
                validation_messages=messages,
                numerical_terminal_speed_mps=terminal_speed,
            )

        speed_values = np.asarray([speed[node].X for node in nodes], dtype=float)
        beta_values = np.asarray([beta[node].X for node in nodes], dtype=float)
        alpha_values = np.asarray([alpha[cell].X for cell in cells], dtype=float)
        optimized_forces_kn = np.asarray([force[cell].X for cell in cells], dtype=float)
        optimized_forces = optimized_forces_kn * 1000.0
        average_speeds = 0.5 * (speed_values[:-1] + speed_values[1:])
        alpha_errors = alpha_values * average_speeds - 1.0
        beta_errors = beta_values - speed_values**2
        beta_scale = np.maximum(node_limits**2, 1.0)
        max_beta_normalized_error = float(np.max(np.maximum(beta_errors, 0.0) / beta_scale))

        try:
            (
                segment_times,
                actual_resistances,
                actual_forces,
                traction_kwh,
                regenerative_kwh,
                net_kwh,
            ) = _physical_recalculation(
                train_config,
                positions,
                speed_values,
                delta_height,
            )
        except ValueError as error:
            return ConvexLowerResult(
                status="optimal_physical_violation",
                section_name=section.name,
                target_time_s=float(target_time_s),
                solver_status_code=status_code,
                solver_runtime_s=float(model.Runtime),
                optimized_objective_kwh=float(model.ObjVal / 3600.0),
                validation_messages=[str(error)],
                numerical_terminal_speed_mps=terminal_speed,
            )

        actual_time = float(np.sum(segment_times))
        time_error = actual_time - float(target_time_s)
        force_gap = actual_forces - optimized_forces
        accelerations = (speed_values[1:] ** 2 - speed_values[:-1] ** 2) / (2.0 * distances)
        actual_power = np.abs(actual_forces) * average_speeds
        validation_messages: List[str] = []
        if abs(time_error) > opts.arrival_tolerance_s:
            validation_messages.append(
                f"actual running-time error {time_error:+.6f}s exceeds tolerance"
            )
        if np.max(speed_values - node_limits) > opts.physical_speed_tolerance_mps:
            validation_messages.append("recomputed speed trajectory violates the reduced speed limit")
        if np.max(actual_forces - train_config.max_traction_force_n) > opts.physical_force_tolerance_n:
            validation_messages.append("recomputed traction force exceeds the force limit")
        if np.max(-train_config.max_braking_force_n - actual_forces) > opts.physical_force_tolerance_n:
            validation_messages.append("recomputed braking force exceeds the force limit")
        if np.max(actual_power - train_config.max_traction_power_w) > opts.physical_power_tolerance_w:
            validation_messages.append("recomputed force violates the symmetric constant-power limit")
        if np.max(accelerations - train_config.max_acceleration_mps2) > opts.physical_acceleration_tolerance_mps2:
            validation_messages.append("recomputed acceleration exceeds the acceleration limit")
        if np.max(-train_config.max_deceleration_mps2 - accelerations) > opts.physical_acceleration_tolerance_mps2:
            validation_messages.append("recomputed acceleration exceeds the deceleration limit")
        max_alpha_error = float(np.max(np.maximum(alpha_errors, 0.0)))
        if max_alpha_error > opts.alpha_relative_tolerance:
            validation_messages.append(
                f"alpha relaxation is loose (max relative error={max_alpha_error:.6g})"
            )
        if max_beta_normalized_error > opts.beta_normalized_tolerance:
            validation_messages.append(
                "beta relaxation is loose "
                f"(max normalized error={max_beta_normalized_error:.6g})"
            )

        if status_code == gp.GRB.SUBOPTIMAL:
            validation_messages.append(
                "Gurobi returned SUBOPTIMAL; candidate retained only for diagnostic validation"
            )
        feasible = not validation_messages
        public_speeds = speed_values.copy()
        public_speeds[0] = 0.0
        public_speeds[-1] = 0.0
        return ConvexLowerResult(
            status="optimal_valid" if feasible else "optimal_physical_violation",
            section_name=section.name,
            target_time_s=float(target_time_s),
            solver_status_code=status_code,
            solver_runtime_s=float(model.Runtime),
            optimized_objective_kwh=float(model.ObjVal / 3600.0),
            actual_time_s=actual_time,
            actual_traction_kwh=traction_kwh,
            actual_regenerative_kwh=regenerative_kwh,
            actual_net_energy_kwh=net_kwh,
            position_m=positions,
            speed_kmh=public_speeds * 3.6,
            optimized_force_kn=optimized_forces_kn,
            recomputed_force_kn=actual_forces / 1000.0,
            recomputed_resistance_kn=actual_resistances / 1000.0,
            segment_time_s=segment_times,
            alpha_relative_error=alpha_errors,
            beta_absolute_error_m2ps2=beta_errors,
            max_alpha_relative_error=max_alpha_error,
            max_beta_normalized_error=max_beta_normalized_error,
            max_force_gap_kn=float(np.max(np.abs(force_gap)) / 1000.0),
            timetable_error_s=time_error,
            feasible=feasible,
            validation_messages=validation_messages,
            numerical_terminal_speed_mps=terminal_speed,
            _internal_speed_mps=speed_values,
            _delta_height_m=delta_height,
            _node_speed_limits_mps=node_limits,
        )

    def solve_environment(
        self,
        environment: LowerSectionEnvironment,
        target_time_s: float,
    ) -> ConvexLowerResult:
        """Upper-layer-facing adapter using the existing environment data."""
        return self.solve(environment.section, environment.dynamics.config, target_time_s)

    def to_lower_result(
        self,
        optimized: ConvexLowerResult,
        section: StationSection,
        train_config: TrainConfig,
    ) -> LowerResult:
        """Convert a validated optimization result to the existing upper-layer type."""
        if not optimized.has_solution:
            raise ValueError(f"Cannot convert result without a solution: {optimized.status}")
        speeds = optimized._internal_speed_mps
        forces_n = optimized.recomputed_force_kn * 1000.0
        resistances_n = optimized.recomputed_resistance_kn * 1000.0
        elapsed = np.concatenate(([0.0], np.cumsum(optimized.segment_time_s)))
        distances = np.diff(optimized.position_m)
        work = forces_n * distances
        traction_steps = np.maximum(work, 0.0) / train_config.traction_efficiency / KWH_IN_JOULES
        regen_steps = np.maximum(-work, 0.0) * train_config.regenerative_efficiency / KWH_IN_JOULES
        cumulative_traction = np.concatenate(([0.0], np.cumsum(traction_steps)))
        cumulative_regen = np.concatenate(([0.0], np.cumsum(regen_steps)))

        actions: List[int] = []
        regimes: List[str] = []
        for force_n in forces_n:
            if force_n < -0.05 * train_config.max_braking_force_n:
                actions.append(LowerSectionEnvironment.ACTION_BRAKE)
                regimes.append("brake")
            elif force_n > 0.55 * train_config.max_traction_force_n:
                actions.append(LowerSectionEnvironment.ACTION_FULL_TRACTION)
                regimes.append("traction")
            elif force_n > 0.05 * train_config.max_traction_force_n:
                actions.append(LowerSectionEnvironment.ACTION_HALF_TRACTION)
                regimes.append("traction")
            else:
                actions.append(LowerSectionEnvironment.ACTION_COAST)
                regimes.append("coast")

        trajectory: List[TrajectoryPoint] = []
        for node, position in enumerate(optimized.position_m):
            segment = section.segment_at(min(position, section.end_m - 1e-6))
            if node == 0:
                force_kn = 0.0
                resistance_kn = 0.0
                action = LowerSectionEnvironment.ACTION_COAST
                regime = "coast"
            else:
                force_kn = optimized.recomputed_force_kn[node - 1]
                resistance_kn = optimized.recomputed_resistance_kn[node - 1]
                action = actions[node - 1]
                regime = regimes[node - 1]
            trajectory.append(
                TrajectoryPoint(
                    section_name=section.name,
                    elapsed_s=float(elapsed[node]),
                    position_m=float(position),
                    speed_kmh=float(optimized.speed_kmh[node]),
                    force_kn=float(force_kn),
                    resistance_kn=float(resistance_kn),
                    speed_limit_kmh=float(segment.speed_limit_kmh * self.options.speed_limit_ratio),
                    gradient_per_mille=float(segment.gradient_per_mille),
                    radius_m=segment.radius_m,
                    requested_action=action,
                    applied_action=action,
                    regime=regime,
                    traction_kwh=float(cumulative_traction[node]),
                    regenerative_kwh=float(cumulative_regen[node]),
                )
            )

        regime_switches = sum(a != b for a, b in zip(regimes, regimes[1:]))
        traction_brake_transitions = sum(
            {a, b} == {"traction", "brake"} for a, b in zip(regimes, regimes[1:])
        )
        traction_coast_transitions = sum(
            {a, b} == {"traction", "coast"} for a, b in zip(regimes, regimes[1:])
        )
        force_scale = max(
            train_config.max_traction_force_n,
            train_config.max_braking_force_n,
            EPS,
        )
        force_jump_index = float(
            np.sum(np.abs(np.diff(np.concatenate(([0.0], forces_n))))) / force_scale
        )
        accelerations = (speeds[1:] ** 2 - speeds[:-1] ** 2) / (2.0 * distances)
        jerk_integral = float(
            np.sum(np.abs(np.diff(np.concatenate(([0.0], accelerations)))))
        )
        time_error = float(optimized.actual_time_s - optimized.target_time_s)
        strict = optimized.feasible and abs(time_error) <= self.options.arrival_tolerance_s
        max_violation = max(
            (max(0.0, point.speed_kmh - point.speed_limit_kmh) for point in trajectory),
            default=0.0,
        )
        return LowerResult(
            section_name=section.name,
            target_time_s=optimized.target_time_s,
            actual_time_s=float(optimized.actual_time_s),
            traction_kwh=float(optimized.actual_traction_kwh),
            regenerative_kwh=float(optimized.actual_regenerative_kwh),
            net_energy_kwh=float(optimized.actual_net_energy_kwh),
            arrival_speed_mps=0.0,
            feasible=optimized.feasible,
            timed_out=False,
            regime_switches=regime_switches,
            traction_brake_transitions=traction_brake_transitions,
            safety_brake_overrides=0,
            traction_coast_transitions=traction_coast_transitions,
            short_dwell_transitions=0,
            force_jump_index=force_jump_index,
            smoothness_penalty=0.0,
            jerk_integral_mps3_s=jerk_integral,
            comfort_penalty=0.0,
            traction_reward_component=0.0,
            punctuality_reward_component=0.0,
            comfort_reward_component=0.0,
            max_speed_limit_violation_kmh=float(max_violation),
            speed_regulation_steps=0,
            speed_limit_action_shield_overrides=0,
            strict_timetable_feasible=strict,
            accepted_arrival_tolerance_s=(self.options.arrival_tolerance_s if strict else None),
            episode_reward=-float(optimized.actual_net_energy_kwh),
            expert_interventions=0,
            trajectory=trajectory,
        )


def solve_convex_lower(
    section: StationSection,
    train_config: TrainConfig,
    target_time_s: float,
    *,
    spatial_segments: int = 600,
    speed_limit_ratio: float = 0.8,
    output_flag: bool = False,
) -> ConvexLowerResult:
    """Convenience interface requested by upper-layer callers."""
    solver = ConvexLowerSolver(
        ConvexLowerOptions(
            spatial_segments=spatial_segments,
            speed_limit_ratio=speed_limit_ratio,
            output_flag=output_flag,
        )
    )
    return solver.solve(section, train_config, target_time_s)

