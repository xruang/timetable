"""License-free CVXPY backend for the CO_V2 lower-level model.

The mathematical relaxation and the public result type match
``timetable.convex_lower``.  CLARABEL is used by default and SCS is retained
as an optional fallback; neither solver requires a commercial licence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import List, Optional

import numpy as np

from .convex_lower import (
    ConvexLowerOptions,
    ConvexLowerResult,
    ConvexLowerSolver,
    _integrated_height_changes,
    _node_speed_limits,
    _physical_recalculation,
)
from .line_profile import StationSection
from .lower_control import LowerSectionEnvironment
from .train_model import TrainConfig


@dataclass(frozen=True)
class CvxpyLowerOptions(ConvexLowerOptions):
    """Options specific to the open-source CVXPY solver backend."""

    solver: str = "CLARABEL"
    fallback_solver: Optional[str] = "SCS"

    def validate(self) -> None:
        super().validate()
        if not self.solver.strip():
            raise ValueError("solver must not be empty")
        if self.fallback_solver is not None and not self.fallback_solver.strip():
            raise ValueError("fallback_solver must be None or a non-empty name")


def _empty_result(
    status: str,
    section: StationSection,
    target_time_s: float,
    terminal_speed_mps: float,
    message: str,
    runtime_s: float = 0.0,
) -> ConvexLowerResult:
    return ConvexLowerResult(
        status=status,
        section_name=section.name,
        target_time_s=float(target_time_s),
        solver_runtime_s=float(runtime_s),
        validation_messages=[message],
        numerical_terminal_speed_mps=terminal_speed_mps,
    )


class CvxpyLowerSolver(ConvexLowerSolver):
    """CO_V2-compatible convex lower solver without a Gurobi dependency."""

    options: CvxpyLowerOptions

    def __init__(self, options: Optional[CvxpyLowerOptions] = None) -> None:
        self.options = options or CvxpyLowerOptions()
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
            import cvxpy as cp
        except ImportError as error:
            return _empty_result(
                "solver_unavailable",
                section,
                target_time_s,
                self.options.numerical_terminal_speed_mps,
                f"cvxpy is unavailable: {error}",
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

        speed = cp.Variable(n + 1, nonneg=True, name="speed_mps")
        beta = cp.Variable(n + 1, nonneg=True, name="speed_square_relaxation")
        alpha = cp.Variable(n, nonneg=True, name="reciprocal_speed")
        average_speed = cp.Variable(n, nonneg=True, name="average_speed_mps")
        elapsed = cp.Variable(n, nonneg=True, name="elapsed_time_s")
        resistance = cp.Variable(n, nonneg=True, name="davis_resistance_kn")
        force = cp.Variable(n, name="optimized_force_kn")
        electrical_energy = cp.Variable(n, name="electrical_energy_kj")

        mass_t = train_config.mass_kg / 1000.0
        max_power_kw = train_config.max_traction_power_w / 1000.0
        constraints = [
            speed[1:n] <= node_limits[1:n],
            beta[1:n] <= np.square(node_limits[1:n]),
            speed[0] == terminal_speed,
            speed[n] == terminal_speed,
            beta[0] == terminal_speed**2,
            beta[n] == terminal_speed**2,
            alpha <= 1.0,
            cp.sum(elapsed) == target_time_s,
            force >= -train_config.max_braking_force_n / 1000.0,
            force <= train_config.max_traction_force_n / 1000.0,
        ]

        for cell in range(n):
            ds = float(distances[cell])
            constraints.extend(
                [
                    average_speed[cell]
                    == 0.5 * (speed[cell] + speed[cell + 1]),
                    elapsed[cell] == ds * alpha[cell],
                    # inv_pos(v_bar) <= alpha is exactly alpha*v_bar >= 1
                    # on the positive domain and is recognized as convex DCP.
                    cp.inv_pos(average_speed[cell]) <= alpha[cell],
                    electrical_energy[cell]
                    >= force[cell] * ds / train_config.traction_efficiency,
                    electrical_energy[cell]
                    >= force[cell] * ds * train_config.regenerative_efficiency,
                    force[cell] >= -max_power_kw * alpha[cell],
                    force[cell] <= max_power_kw * alpha[cell],
                    resistance[cell]
                    == float(train_config.davis_a) / 1000.0
                    + float(train_config.davis_b) / 1000.0 * 3.6 * speed[cell + 1]
                    + float(train_config.davis_c) / 1000.0 * 3.6**2 * beta[cell + 1],
                    force[cell] * ds
                    - 0.5 * mass_t * (beta[cell + 1] - beta[cell])
                    - ds * resistance[cell]
                    - mass_t
                    * train_config.gravity_mps2
                    * float(delta_height[cell])
                    == 0.0,
                    0.5 * (beta[cell + 1] - beta[cell]) / ds
                    >= -train_config.max_deceleration_mps2,
                    0.5 * (beta[cell + 1] - beta[cell]) / ds
                    <= train_config.max_acceleration_mps2,
                    speed[cell + 1] - speed[cell]
                    >= -train_config.max_deceleration_mps2 * elapsed[cell],
                    speed[cell + 1] - speed[cell]
                    <= train_config.max_acceleration_mps2 * elapsed[cell],
                ]
            )
        constraints.append(cp.square(speed) <= beta)

        problem = cp.Problem(cp.Minimize(cp.sum(electrical_energy)), constraints)
        if not problem.is_dcp():
            return _empty_result(
                "model_error",
                section,
                target_time_s,
                terminal_speed,
                "CVXPY rejected the lower-level model as non-DCP",
            )

        installed = {name.upper(): name for name in cp.installed_solvers()}
        candidates = [opts.solver]
        if opts.fallback_solver and opts.fallback_solver.upper() != opts.solver.upper():
            candidates.append(opts.fallback_solver)
        candidates = [name for name in candidates if name.upper() in installed]
        if not candidates:
            return _empty_result(
                "solver_unavailable",
                section,
                target_time_s,
                terminal_speed,
                "No requested open-source CVXPY solver is installed; "
                f"requested={[opts.solver, opts.fallback_solver]}, "
                f"installed={cp.installed_solvers()}",
            )

        started = time.perf_counter()
        solve_errors: List[str] = []
        used_solver: Optional[str] = None
        for candidate in candidates:
            solver_name = installed[candidate.upper()]
            solve_kwargs = {"solver": solver_name, "verbose": opts.output_flag}
            if opts.time_limit_s is not None:
                if candidate.upper() == "CLARABEL":
                    solve_kwargs["time_limit"] = float(opts.time_limit_s)
                elif candidate.upper() == "SCS":
                    solve_kwargs["time_limit_secs"] = float(opts.time_limit_s)
            try:
                problem.solve(**solve_kwargs)
                used_solver = solver_name
            except cp.error.SolverError as error:
                solve_errors.append(f"{solver_name}: {error}")
                continue
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                break
        runtime = time.perf_counter() - started

        if used_solver is None:
            return _empty_result(
                "solver_error",
                section,
                target_time_s,
                terminal_speed,
                "; ".join(solve_errors) or "CVXPY solver failed",
                runtime,
            )

        status_map = {
            cp.OPTIMAL: "optimal",
            cp.OPTIMAL_INACCURATE: "optimal_inaccurate",
            cp.INFEASIBLE: "infeasible",
            cp.INFEASIBLE_INACCURATE: "infeasible_inaccurate",
            cp.UNBOUNDED: "unbounded",
            cp.UNBOUNDED_INACCURATE: "unbounded_inaccurate",
            cp.USER_LIMIT: "time_limit",
        }
        solver_status = status_map.get(problem.status, str(problem.status))
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            details = f"{used_solver} did not return a solution ({solver_status})"
            if solve_errors:
                details += "; earlier attempts: " + "; ".join(solve_errors)
            return _empty_result(
                solver_status,
                section,
                target_time_s,
                terminal_speed,
                details,
                runtime,
            )

        values = (speed.value, beta.value, alpha.value, force.value)
        if any(value is None for value in values):
            return _empty_result(
                "solver_error",
                section,
                target_time_s,
                terminal_speed,
                f"{used_solver} reported {solver_status} without primal values",
                runtime,
            )
        speed_values = np.asarray(speed.value, dtype=float).reshape(-1)
        beta_values = np.asarray(beta.value, dtype=float).reshape(-1)
        alpha_values = np.asarray(alpha.value, dtype=float).reshape(-1)
        optimized_forces_kn = np.asarray(force.value, dtype=float).reshape(-1)
        average_speeds = 0.5 * (speed_values[:-1] + speed_values[1:])
        alpha_errors = alpha_values * average_speeds - 1.0
        beta_errors = beta_values - speed_values**2
        beta_scale = np.maximum(node_limits**2, 1.0)
        max_beta_normalized_error = float(
            np.max(np.maximum(beta_errors, 0.0) / beta_scale)
        )

        try:
            (
                segment_times,
                actual_resistances,
                actual_forces,
                traction_kwh,
                regenerative_kwh,
                net_kwh,
            ) = _physical_recalculation(
                train_config, positions, speed_values, delta_height
            )
        except ValueError as error:
            return _empty_result(
                "optimal_physical_violation",
                section,
                target_time_s,
                terminal_speed,
                str(error),
                runtime,
            )

        actual_time = float(np.sum(segment_times))
        time_error = actual_time - float(target_time_s)
        optimized_forces = optimized_forces_kn * 1000.0
        force_gap = actual_forces - optimized_forces
        accelerations = (
            speed_values[1:] ** 2 - speed_values[:-1] ** 2
        ) / (2.0 * distances)
        actual_power = np.abs(actual_forces) * average_speeds
        validation_messages: List[str] = []
        if abs(time_error) > opts.arrival_tolerance_s:
            validation_messages.append(
                f"actual running-time error {time_error:+.6f}s exceeds tolerance"
            )
        if np.max(speed_values - node_limits) > opts.physical_speed_tolerance_mps:
            validation_messages.append(
                "recomputed speed trajectory violates the reduced speed limit"
            )
        if (
            np.max(actual_forces - train_config.max_traction_force_n)
            > opts.physical_force_tolerance_n
        ):
            validation_messages.append(
                "recomputed traction force exceeds the force limit"
            )
        if (
            np.max(-train_config.max_braking_force_n - actual_forces)
            > opts.physical_force_tolerance_n
        ):
            validation_messages.append(
                "recomputed braking force exceeds the force limit"
            )
        if (
            np.max(actual_power - train_config.max_traction_power_w)
            > opts.physical_power_tolerance_w
        ):
            validation_messages.append(
                "recomputed force violates the symmetric constant-power limit"
            )
        if (
            np.max(accelerations - train_config.max_acceleration_mps2)
            > opts.physical_acceleration_tolerance_mps2
        ):
            validation_messages.append(
                "recomputed acceleration exceeds the acceleration limit"
            )
        if (
            np.max(-train_config.max_deceleration_mps2 - accelerations)
            > opts.physical_acceleration_tolerance_mps2
        ):
            validation_messages.append(
                "recomputed acceleration exceeds the deceleration limit"
            )
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
        if problem.status == cp.OPTIMAL_INACCURATE:
            validation_messages.append(
                f"{used_solver} returned OPTIMAL_INACCURATE; candidate kept for diagnostics"
            )

        feasible = not validation_messages
        public_speeds = speed_values.copy()
        public_speeds[0] = 0.0
        public_speeds[-1] = 0.0
        objective_kwh = (
            None if problem.value is None else float(problem.value / 3600.0)
        )
        return ConvexLowerResult(
            status="optimal_valid" if feasible else "optimal_physical_violation",
            section_name=section.name,
            target_time_s=float(target_time_s),
            solver_runtime_s=float(runtime),
            optimized_objective_kwh=objective_kwh,
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
        return self.solve(environment.section, environment.dynamics.config, target_time_s)


def solve_cvxpy_lower(
    section: StationSection,
    train_config: TrainConfig,
    target_time_s: float,
    *,
    spatial_segments: int = 600,
    speed_limit_ratio: float = 0.8,
    output_flag: bool = False,
    solver: str = "CLARABEL",
) -> ConvexLowerResult:
    """Upper-layer convenience interface using only open-source solvers."""
    backend = CvxpyLowerSolver(
        CvxpyLowerOptions(
            spatial_segments=spatial_segments,
            speed_limit_ratio=speed_limit_ratio,
            output_flag=output_flag,
            solver=solver,
        )
    )
    return backend.solve(section, train_config, target_time_s)
