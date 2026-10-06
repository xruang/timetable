"""Train parameters and resistance model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

try:
    from .line_profile import TrackSegment
except ImportError:
    from line_profile import TrackSegment


@dataclass(frozen=True)
class TrainConfig:
    name: str = "Train 1"
    mass_kg: float = 260000.0
    motor_car_mass_kg: Optional[float] = None
    trailer_car_mass_kg: Optional[float] = None
    max_traction_force_n: float = 290000.0
    max_traction_power_w: float = 3100.0 * 1000.0
    max_braking_force_n: float = 280000.0
    traction_efficiency: float = 0.90
    regenerative_efficiency: float = 0.60
    gravity_mps2: float = 9.81
    davis_a: Optional[float] = None
    davis_b: Optional[float] = None
    davis_c: Optional[float] = None
    curve_resistance_coefficient: float = 600.0
    dt_s: float = 0.5
    safety_braking_ratio: float = 0.65
    service_braking_ratio: float = 0.65
    terminal_braking_margin_s: float = 1.5
    max_acceleration_mps2: float = 1.0
    max_deceleration_mps2: float = 1.0

    def __post_init__(self) -> None:
        motor_t = self.effective_motor_car_mass_kg / 1000.0
        trailer_t = self.effective_trailer_car_mass_kg / 1000.0
        if self.davis_a is None:
            object.__setattr__(
                self,
                "davis_a",
                (1.65 * motor_t + 0.78 * trailer_t) * 9.8,
            )
        if self.davis_b is None:
            object.__setattr__(
                self,
                "davis_b",
                (0.0247 * motor_t + 0.0028 * trailer_t) * 9.8,
            )
        if self.davis_c is None:
            object.__setattr__(self, "davis_c", 0.6566)

    @property
    def effective_motor_car_mass_kg(self) -> float:
        return (
            self.motor_car_mass_kg
            if self.motor_car_mass_kg is not None
            else self.mass_kg * 0.70
        )

    @property
    def effective_trailer_car_mass_kg(self) -> float:
        return (    
            self.trailer_car_mass_kg
            if self.trailer_car_mass_kg is not None
            else self.mass_kg - self.effective_motor_car_mass_kg
        )


def make_train_config(
    name: str,
    total_mass_t: float,
    max_traction_force_kn: float,
    max_traction_power_kw: float,
    motor_car_mass_t: Optional[float] = None,
    trailer_car_mass_t: Optional[float] = None,
    max_braking_force_kn: float = 280.0,
) -> TrainConfig:
    """Create one train template using engineering units.

    Mass inputs are tonnes, force inputs are kN and power inputs are kW.
    If motor/trailer masses are not provided, TrainConfig uses the default
    split from total mass.
    """
    return TrainConfig(
        name=name,
        mass_kg=total_mass_t * 1000.0,
        motor_car_mass_kg=(
            None if motor_car_mass_t is None else motor_car_mass_t * 1000.0
        ),
        trailer_car_mass_kg=(
            None if trailer_car_mass_t is None else trailer_car_mass_t * 1000.0
        ),
        max_traction_force_n=max_traction_force_kn * 1000.0,
        max_traction_power_w=max_traction_power_kw * 1000.0,
        max_braking_force_n=max_braking_force_kn * 1000.0,
    )


# Edit this single table for heterogeneous-train experiments.
TRAIN_CONFIGS = [
    make_train_config("Train 1", 235.0, 260.0, 2770.0),
    make_train_config("Train 2", 240.0, 265.0, 2850.0),
    make_train_config("Train 3", 245.0, 270.0, 2900.0),
    make_train_config("Train 4", 250.0, 275.0, 2970.0),
    make_train_config("Train 5", 255.0, 280.0, 3020.0),
    make_train_config("Train 6", 260.0, 285.0, 3070.0),
    make_train_config("Train 7", 265.0, 290.0, 3140.0),
    make_train_config("Train 8", 270.0, 295.0, 3180.0),
    make_train_config("Train 9", 275.0, 300.0, 3230.0),
]


def interpolate_train_constraints(
    total_mass_t: float,
) -> tuple[float, float]:
    """Map a train mass to traction-force and power limits.

    The nine configured train templates are treated as calibration anchors.
    Intermediate masses use linear interpolation. Values outside the table are
    conservatively clipped to the nearest anchor instead of extrapolated.
    """
    anchors = sorted(
        (
            config.mass_kg / 1000.0,
            config.max_traction_force_n / 1000.0,
            config.max_traction_power_w / 1000.0,
        )
        for config in TRAIN_CONFIGS
    )
    mass_t = float(total_mass_t)
    if mass_t <= anchors[0][0]:
        return anchors[0][1], anchors[0][2]
    if mass_t >= anchors[-1][0]:
        return anchors[-1][1], anchors[-1][2]
    for left, right in zip(anchors[:-1], anchors[1:]):
        left_mass, left_force, left_power = left
        right_mass, right_force, right_power = right
        if left_mass <= mass_t <= right_mass:
            ratio = (mass_t - left_mass) / max(right_mass - left_mass, 1e-9)
            force_kn = left_force + ratio * (right_force - left_force)
            power_kw = left_power + ratio * (right_power - left_power)
            return force_kn, power_kw
    return anchors[-1][1], anchors[-1][2]


def create_train_config_from_mass(
    name: str,
    total_mass_t: float,
    motor_car_mass_t: Optional[float] = None,
    trailer_car_mass_t: Optional[float] = None,
    max_braking_force_kn: float = 280.0,
) -> TrainConfig:
    """Create a train config from a total mass using the anchor table.

    If motor/trailer masses are omitted, the motor-car mass is set to 70% of
    total mass and the trailer-car mass takes the remainder.
    """
    total_mass_t = float(total_mass_t)
    if motor_car_mass_t is None and trailer_car_mass_t is None:
        motor_car_mass_t = total_mass_t * 0.70
        trailer_car_mass_t = total_mass_t - motor_car_mass_t
    elif motor_car_mass_t is None:
        trailer_car_mass_t = float(trailer_car_mass_t)
        motor_car_mass_t = total_mass_t - trailer_car_mass_t
    elif trailer_car_mass_t is None:
        motor_car_mass_t = float(motor_car_mass_t)
        trailer_car_mass_t = total_mass_t - motor_car_mass_t
    if motor_car_mass_t < 0.0 or trailer_car_mass_t < 0.0:
        raise ValueError("Motor/trailer masses must be non-negative")
    force_kn, power_kw = interpolate_train_constraints(total_mass_t)
    return make_train_config(
        name=name,
        total_mass_t=total_mass_t,
        max_traction_force_kn=force_kn,
        max_traction_power_kw=power_kw,
        motor_car_mass_t=motor_car_mass_t,
        trailer_car_mass_t=trailer_car_mass_t,
        max_braking_force_kn=max_braking_force_kn,
    )


def configured_train_configs(train_count: int) -> list[TrainConfig]:
    if train_count < 2:
        raise ValueError("At least two trains are required for coordinated optimization")
    if train_count > len(TRAIN_CONFIGS):
        raise ValueError(
            f"Requested {train_count} trains, but only {len(TRAIN_CONFIGS)} entries "
            "exist in TRAIN_CONFIGS"
        )
    return TRAIN_CONFIGS[:train_count]


def configured_bidirectional_train_configs(
    up_train_count: int = 6,
    down_train_count: int = 6,
) -> list[TrainConfig]:
    if up_train_count < 1 or down_train_count < 1:
        raise ValueError("Both directions require at least one train")

    configs: list[TrainConfig] = []
    for direction, count in (("Up", up_train_count), ("Down", down_train_count)):
        for train_idx in range(count):
            template = TRAIN_CONFIGS[len(configs) % len(TRAIN_CONFIGS)]
            configs.append(
                replace(
                    template,
                    name=f"{direction} Train {train_idx + 1}",
                )
            )
    return configs


class TrainDynamics:
    def __init__(self, config: Optional[TrainConfig] = None) -> None:
        self.config = config or TrainConfig()

    def davis_resistance_n(self, speed_mps: float) -> float:
        speed_kmh = speed_mps * 3.6
        cfg = self.config
        return cfg.davis_a + cfg.davis_b * speed_kmh + cfg.davis_c * speed_kmh**2

    def gradient_resistance_n(self, gradient_per_mille: float) -> float:
        cfg = self.config
        return cfg.mass_kg * cfg.gravity_mps2 * gradient_per_mille / 1000.0

    def curve_resistance_n(self, radius_m: Optional[float]) -> float:
        if radius_m is None:
            return 0.0
        cfg = self.config
        specific_resistance = cfg.curve_resistance_coefficient / radius_m
        return cfg.mass_kg * cfg.gravity_mps2 * specific_resistance / 1000.0

    def total_resistance_n(self, speed_mps: float, segment: TrackSegment) -> float:
        return (
            self.davis_resistance_n(speed_mps)
            + self.gradient_resistance_n(segment.gradient_per_mille)
            + self.curve_resistance_n(segment.radius_m)
        )

    def max_traction_force_n(self, speed_mps: float) -> float:
        cfg = self.config
        if speed_mps <= 1e-6:
            return cfg.max_traction_force_n
        return min(cfg.max_traction_force_n, cfg.max_traction_power_w / speed_mps)

    def acceleration_limited_force_n(
        self,
        force_n: float,
        resistance_n: float,
        speed_mps: float = 0.0,
    ) -> float:
        """Return the effective force after +/- acceleration limits."""
        cfg = self.config
        raw_acceleration = (force_n - resistance_n) / cfg.mass_kg
        limited_acceleration = max(
            -cfg.max_deceleration_mps2,
            min(cfg.max_acceleration_mps2, raw_acceleration),
        )
        if limited_acceleration == raw_acceleration:
            return force_n
        limited_force_n = resistance_n + cfg.mass_kg * limited_acceleration
        return max(
            -cfg.max_braking_force_n,
            min(self.max_traction_force_n(speed_mps), limited_force_n),
        )
