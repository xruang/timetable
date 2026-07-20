"""Compare the original lower driving strategy with a real-data policy.

This script does not replace the original DQN/MAPPO workflow. It evaluates the
same initial timetable twice:
1. the existing lower strategy without real-data training;
2. the continuous force policy trained from real operation records.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config import DEFAULT_LINE_DIR, INITIAL_TIMETABLE
    from .line_profile import LineProfile
    from .lower_control import LowerResult, LowerSectionEnvironment, evaluate_timetable
    from .real_strategy import (
        RealStrategyConfig,
        RealStrategyController,
        train_real_strategy_policy,
    )
    from .train_model import TrainDynamics, configured_train_configs
    from .visualization import (
        lower_result_summary_rows,
        plot_section_operation_overview,
        plot_train_operation,
        write_csv,
    )
except ImportError:
    from config import DEFAULT_LINE_DIR, INITIAL_TIMETABLE
    from line_profile import LineProfile
    from lower_control import LowerResult, LowerSectionEnvironment, evaluate_timetable
    from real_strategy import (
        RealStrategyConfig,
        RealStrategyController,
        train_real_strategy_policy,
    )
    from train_model import TrainDynamics, configured_train_configs
    from visualization import (
        lower_result_summary_rows,
        plot_section_operation_overview,
        plot_train_operation,
        write_csv,
    )


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare original lower driving strategy with a real-data policy."
    )
    parser.add_argument("--line-dir", type=Path, default=DEFAULT_LINE_DIR)
    parser.add_argument(
        "--real-data",
        type=Path,
        nargs="*",
        default=[DEFAULT_LINE_DIR],
        help="Real-operation xlsx/csv files or folders used when training the real policy.",
    )
    parser.add_argument(
        "--real-policy",
        type=Path,
        default=SCRIPT_DIR / "real_strategy_policy.pt",
        help="Checkpoint path for the real-data driving policy.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain the real-data policy even if --real-policy already exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "real_strategy_comparison",
    )
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history-len", type=int, default=12)
    parser.add_argument("--max-abs-force-n", type=float, default=300000.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.08)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--max-gradient-per-mille", type=float, default=200.0)
    parser.add_argument("--max-speed-kmh", type=float, default=130.0)
    parser.add_argument("--max-filter-force-n", type=float, default=400000.0)
    parser.add_argument("--min-train-mass-kg", type=float, default=180000.0)
    parser.add_argument("--max-train-mass-kg", type=float, default=360000.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser


def build_train1_environments(profile: LineProfile) -> List[LowerSectionEnvironment]:
    config = configured_train_configs(4)[0]
    return [
        LowerSectionEnvironment(section, TrainDynamics(config))
        for section in profile.sections
    ]


def ensure_real_policy(args: argparse.Namespace) -> RealStrategyController:
    if args.real_policy.exists() and not args.force_retrain:
        print(f"Loading real-data policy: {args.real_policy.resolve()}")
        return RealStrategyController.from_checkpoint(args.real_policy, device_name=args.device)
    if not args.real_data:
        raise ValueError("--real-data is required when --real-policy does not exist")
    print("Training real-data lower driving policy for comparison")
    config = RealStrategyConfig(
        history_len=args.history_len,
        max_abs_force_n=args.max_abs_force_n,
        force_smoothness_weight=args.smoothness_weight,
        sample_interval_s=args.sample_interval,
        max_abs_gradient_per_mille=args.max_gradient_per_mille,
        max_speed_kmh=args.max_speed_kmh,
        max_abs_force_n_for_filter=args.max_filter_force_n,
        min_train_mass_kg=args.min_train_mass_kg,
        max_train_mass_kg=args.max_train_mass_kg,
    )
    stats = train_real_strategy_policy(
        data_paths=args.real_data,
        output_path=args.real_policy,
        config=config,
        epochs=args.train_epochs,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(
        f"Real policy trained: windows={stats.sequence_count}, "
        f"validation_loss={stats.validation_loss:.6f}"
    )
    print(
        f"Real-data filtering: samples={stats.sample_count}, "
        f"accepted={stats.accepted_sample_count}, filtered={stats.filtered_sample_count}"
    )
    if stats.filter_reasons:
        print("Filter reasons: " + ", ".join(
            f"{key}={value}" for key, value in sorted(stats.filter_reasons.items())
        ))
    return RealStrategyController.from_checkpoint(args.real_policy, device_name=args.device)


def strategy_totals(strategy_name: str, results: Sequence[LowerResult]) -> Dict[str, Any]:
    return {
        "strategy": strategy_name,
        "section_count": len(results),
        "strict_feasible_count": sum(item.strict_timetable_feasible for item in results),
        "relaxed_feasible_count": sum(item.relaxed_timetable_feasible for item in results),
        "total_target_time_s": float(sum(item.target_time_s for item in results)),
        "total_actual_time_s": float(sum(item.actual_time_s for item in results)),
        "total_abs_timetable_error_s": float(sum(abs(item.timetable_error_s) for item in results)),
        "traction_kwh": float(sum(item.traction_kwh for item in results)),
        "regenerative_kwh": float(sum(item.regenerative_kwh for item in results)),
        "net_energy_kwh": float(sum(item.net_energy_kwh for item in results)),
        "regime_switches": float(sum(item.regime_switches for item in results)),
        "traction_coast_transitions": float(sum(item.traction_coast_transitions for item in results)),
        "traction_brake_transitions": float(sum(item.traction_brake_transitions for item in results)),
        "short_dwell_transitions": float(sum(item.short_dwell_transitions for item in results)),
        "force_jump_index": float(sum(item.force_jump_index for item in results)),
        "speed_regulation_steps": float(sum(item.speed_regulation_steps for item in results)),
    }


def export_comparison_plot(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    metrics = [
        ("net_energy_kwh", "Net energy (kWh)"),
        ("total_abs_timetable_error_s", "Abs timetable error (s)"),
        ("force_jump_index", "Force jump index"),
        ("traction_coast_transitions", "Traction/coast transitions"),
    ]
    labels = [str(row["strategy"]) for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (key, title) in zip(axes.flat, metrics):
        values = [float(row[key]) for row in rows]
        axis.bar(x, values, color=["#1565C0", "#EF6C00"][: len(labels)])
        axis.set_xticks(x, labels, rotation=15)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            axis.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Lower Driving Strategy Comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "lower_strategy_comparison.png", dpi=180)
    plt.close(fig)


def export_strategy_outputs(
    profile: LineProfile,
    strategy_name: str,
    results: Sequence[LowerResult],
    output_dir: Path,
) -> None:
    strategy_dir = output_dir / strategy_name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        strategy_dir / f"{strategy_name}_summary.csv",
        lower_result_summary_rows(results, strategy_name),
    )
    plot_train_operation(profile, results, strategy_dir, f"{strategy_name}_operation")
    plot_section_operation_overview(profile, results, strategy_dir, strategy_name)


def rollout_timetable_with_controller(
    environments: Sequence[LowerSectionEnvironment],
    timetable: Sequence[float],
    controller,
) -> List[LowerResult]:
    return [
        environment.rollout(controller, float(target))
        for environment, target in zip(environments, timetable)
    ]


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = LineProfile.from_excel(
        args.line_dir / "slope.xlsx",
        args.line_dir / "curvature.xlsx",
    )
    real_controller = ensure_real_policy(args)

    baseline_environments = build_train1_environments(profile)
    real_environments = build_train1_environments(profile)

    print("\nEvaluating original lower strategy")
    baseline_results = evaluate_timetable(baseline_environments, INITIAL_TIMETABLE)
    print("Evaluating real-data lower strategy as direct controller")
    real_results = rollout_timetable_with_controller(
        real_environments,
        INITIAL_TIMETABLE,
        real_controller,
    )

    export_strategy_outputs(profile, "baseline", baseline_results, args.output_dir)
    export_strategy_outputs(profile, "real_data_policy", real_results, args.output_dir)

    summary_rows = [
        strategy_totals("baseline", baseline_results),
        strategy_totals("real_data_policy", real_results),
    ]
    write_csv(args.output_dir / "lower_strategy_comparison_summary.csv", summary_rows)
    detailed_rows = []
    for strategy_name, results in (
        ("baseline", baseline_results),
        ("real_data_policy", real_results),
    ):
        for row in lower_result_summary_rows(results, strategy_name):
            detailed_rows.append(row)
    write_csv(args.output_dir / "lower_strategy_comparison_by_section.csv", detailed_rows)
    export_comparison_plot(summary_rows, args.output_dir)

    print("\nLower driving strategy comparison:")
    for row in summary_rows:
        print(
            f"  {row['strategy']}: net={float(row['net_energy_kwh']):.2f}kWh, "
            f"abs_error={float(row['total_abs_timetable_error_s']):.2f}s, "
            f"force_jump={float(row['force_jump_index']):.2f}, "
            f"T/C={float(row['traction_coast_transitions']):.0f}"
        )
    print(f"Comparison reports written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
