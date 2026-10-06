"""Shared construction helpers for bidirectional timetable experiments."""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from typing import Any

import numpy as np
import torch

try:
    from .config import INITIAL_TIMETABLE, SECTION_NAMES
    from .line_profile import LineProfile
    from .lower_control import LowerSectionEnvironment
    from .train_model import (
        TrainConfig,
        TrainDynamics,
        configured_bidirectional_train_configs,
    )
except ImportError:
    from config import INITIAL_TIMETABLE, SECTION_NAMES
    from line_profile import LineProfile
    from lower_control import LowerSectionEnvironment
    from train_model import (
        TrainConfig,
        TrainDynamics,
        configured_bidirectional_train_configs,
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch consistently."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    """Build up/down profiles, trains, lower environments and base schedules."""
    up_profile = LineProfile.from_excel(
        args.line_dir / "slope.xlsx",
        args.line_dir / "curvature.xlsx",
    )
    # 当前下行坡度源表的全线积分与上行端点高差不闭合。最小模型使用
    # 同一物理纵断面：反转上行坡度并改变符号；下行曲线和限速仍用
    # 下行专用数据。这样两方向的累计净高差必然互为相反数。
    down_profile = LineProfile.reversed_with_curvature(
        up_profile,
        args.down_curvature,
    )
    train_configs = configured_bidirectional_train_configs(
        args.up_trains,
        args.down_trains,
    )
    train_configs = [
        replace(config, dt_s=float(args.simulation_dt_s))
        for config in train_configs
    ]
    total_trains = args.up_trains + args.down_trains
    direction_labels = ["up"] * args.up_trains + ["down"] * args.down_trains
    environments_by_train: list[list[LowerSectionEnvironment]] = []
    for train_idx, config in enumerate(train_configs):
        profile = up_profile if train_idx < args.up_trains else down_profile
        direction_sign = 1.0 if train_idx < args.up_trains else -1.0
        section_count = len(profile.sections)
        environments_by_train.append(
            [
                LowerSectionEnvironment(
                    section,
                    TrainDynamics(config),
                    direction_sign=direction_sign,
                    section_index_normalized=(
                        0.0
                        if section_count <= 1
                        else 2.0 * section_idx / (section_count - 1) - 1.0
                    ),
                )
                for section_idx, section in enumerate(profile.sections)
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
    """Return station names in the profile's travel direction."""
    return [profile.sections[0].name.split("-")[0]] + [
        section.name.split("-")[-1] for section in profile.sections
    ]


def down_line_profile_rows(
    up_profile: LineProfile,
    down_profile: LineProfile,
) -> list[dict[str, Any]]:
    """Serialize the down profile with travel and physical mileage axes."""
    del up_profile  # Retained in the signature for compatibility with callers.
    rows: list[dict[str, Any]] = []
    for section in down_profile.sections:
        for segment_idx, segment in enumerate(section.segments, start=1):
            physical_start_m = (
                segment.start_m
                if segment.physical_start_m is None
                else segment.physical_start_m
            )
            physical_end_m = (
                segment.end_m
                if segment.physical_end_m is None
                else segment.physical_end_m
            )
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
