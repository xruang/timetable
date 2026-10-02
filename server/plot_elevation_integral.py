"""Plot cumulative elevation reconstructed from the active line segments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from timetable.line_profile import LineProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "current_elevation_integral_comparison.png",
        help="输出图片路径。",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=PROJECT_DIR / "timetable" / "current_line_segments.csv",
        help="同步导出当前上下行模型使用的合并线路片段。",
    )
    return parser.parse_args()


def profile_segments(profile: LineProfile) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for section in profile.sections:
        for segment in section.segments:
            rows.append(
                {
                    "internal_start_m": segment.start_m,
                    "internal_end_m": segment.end_m,
                    "physical_start_m": (
                        segment.start_m
                        if segment.physical_start_m is None
                        else segment.physical_start_m
                    ),
                    "physical_end_m": (
                        segment.end_m
                        if segment.physical_end_m is None
                        else segment.physical_end_m
                    ),
                    "length_m": segment.end_m - segment.start_m,
                    "gradient_per_mille": segment.gradient_per_mille,
                }
            )
    return sorted(rows, key=lambda row: row["internal_start_m"])


def integrate_in_travel_direction(
    segments: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance_m = [0.0]
    elevation_m = [0.0]
    physical_m = [segments[0]["physical_start_m"]]
    for row in segments:
        distance_m.append(distance_m[-1] + row["length_m"])
        elevation_m.append(
            elevation_m[-1]
            + row["gradient_per_mille"] * row["length_m"] / 1000.0
        )
        physical_m.append(row["physical_end_m"])
    return (
        np.asarray(distance_m),
        np.asarray(elevation_m),
        np.asarray(physical_m),
    )


def export_current_segments(
    path: Path,
    up_profile: LineProfile,
    down_profile: LineProfile,
) -> None:
    fieldnames = [
        "direction",
        "section_index",
        "section_name",
        "segment_index",
        "physical_start_m",
        "physical_end_m",
        "internal_start_m",
        "internal_end_m",
        "length_m",
        "gradient_per_mille",
        "radius_m",
        "speed_limit_kmh",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for direction, profile in (("up", up_profile), ("down", down_profile)):
            for section_index, section in enumerate(profile.sections, start=1):
                for segment_index, segment in enumerate(section.segments, start=1):
                    writer.writerow(
                        {
                            "direction": direction,
                            "section_index": section_index,
                            "section_name": section.name,
                            "segment_index": segment_index,
                            "physical_start_m": (
                                segment.start_m
                                if segment.physical_start_m is None
                                else segment.physical_start_m
                            ),
                            "physical_end_m": (
                                segment.end_m
                                if segment.physical_end_m is None
                                else segment.physical_end_m
                            ),
                            "internal_start_m": segment.start_m,
                            "internal_end_m": segment.end_m,
                            "length_m": segment.end_m - segment.start_m,
                            "gradient_per_mille": segment.gradient_per_mille,
                            "radius_m": segment.radius_m,
                            "speed_limit_kmh": segment.speed_limit_kmh,
                        }
                    )


def main() -> None:
    args = parse_args()
    data_dir = PROJECT_DIR / "timetable" / "data"
    up_profile = LineProfile.from_excel(
        data_dir / "up" / "slope.xlsx",
        data_dir / "up" / "curvature.xlsx",
    )
    # Match the active model: down gradients are reversed from the up profile,
    # while down curvature and speed limits retain their dedicated source.
    down_profile = LineProfile.reversed_with_curvature(
        up_profile,
        data_dir / "down" / "curvature.xlsx",
    )
    export_current_segments(args.export_csv, up_profile, down_profile)
    up = integrate_in_travel_direction(profile_segments(up_profile))
    down = integrate_in_travel_direction(profile_segments(down_profile))
    up_distance_m, up_elevation_m, up_physical_m = up
    down_distance_m, down_elevation_m, down_physical_m = down

    # Both physical profiles use THP as the zero-elevation reference.
    up_physical_elevation_m = up_elevation_m
    down_physical_elevation_m = down_elevation_m - down_elevation_m[-1]
    down_order = np.argsort(down_physical_m)
    down_physical_m = down_physical_m[down_order]
    down_physical_elevation_m = down_physical_elevation_m[down_order]

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)

    axes[0].plot(
        up_distance_m / 1000.0,
        up_elevation_m,
        color="#1769aa",
        linewidth=2.0,
        label=f"上行：终点 {up_elevation_m[-1]:+.2f} m",
    )
    axes[0].plot(
        down_distance_m / 1000.0,
        down_elevation_m,
        color="#d1495b",
        linewidth=2.0,
        label=f"下行：终点 {down_elevation_m[-1]:+.2f} m",
    )
    axes[0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0].scatter(
        [up_distance_m[-1] / 1000.0, down_distance_m[-1] / 1000.0],
        [up_elevation_m[-1], down_elevation_m[-1]],
        color=["#1769aa", "#d1495b"],
        zorder=3,
    )
    axes[0].set_title("沿各自运行方向积分的累计高差")
    axes[0].set_xlabel("累计运行距离（km）")
    axes[0].set_ylabel("相对起点高差（m）")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(
        up_physical_m / 1000.0,
        up_physical_elevation_m,
        color="#1769aa",
        linewidth=2.0,
        label="上行坡度重建",
    )
    axes[1].plot(
        down_physical_m / 1000.0,
        down_physical_elevation_m,
        color="#d1495b",
        linewidth=2.0,
        label="下行坡度重建（反向对齐）",
    )
    common_physical_m = np.linspace(
        max(up_physical_m.min(), down_physical_m.min()),
        min(up_physical_m.max(), down_physical_m.max()),
        2000,
    )
    up_common_m = np.interp(common_physical_m, up_physical_m, up_physical_elevation_m)
    down_common_m = np.interp(
        common_physical_m,
        down_physical_m,
        down_physical_elevation_m,
    )
    axes[1].fill_between(
        common_physical_m / 1000.0,
        up_common_m,
        down_common_m,
        color="#8c8c8c",
        alpha=0.15,
        label="两套坡度重建差异",
    )
    endpoint_gap_m = up_elevation_m[-1] + down_elevation_m[-1]
    axes[1].annotate(
        f"终点高程差 {endpoint_gap_m:+.2f} m",
        xy=(up_physical_m[-1] / 1000.0, up_physical_elevation_m[-1]),
        xytext=(-160, -35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#333333"},
    )
    axes[1].set_title("统一到上行物理里程后的高程重建（THP设为0 m）")
    axes[1].set_xlabel("物理里程（km）")
    axes[1].set_ylabel("相对THP高程（m）")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("当前上下行坡度积分净高差核对", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"上行累计高差: {up_elevation_m[-1]:+.3f} m")
    print(f"下行累计高差: {down_elevation_m[-1]:+.3f} m")
    print(f"统一端点后的高程差: {endpoint_gap_m:+.3f} m")
    print(f"图片: {args.output}")


if __name__ == "__main__":
    main()
