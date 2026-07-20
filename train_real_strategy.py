"""Train an independent real-operation traction strategy network.

Example:
    python timetable/train_real_strategy.py --data real_data.csv
    python timetable/train_real_strategy.py --data real_data_dir --group-cols train_id section_name
    python timetable/train_real_strategy.py --data "C:\\Users\\DELL\\Desktop\\21号线\\3-THP-TD.xlsx"
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .real_strategy import RealStrategyConfig, train_real_strategy_policy
except ImportError:
    from real_strategy import RealStrategyConfig, train_real_strategy_policy


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an independent behavior-cloning network from measured "
            "Guangzhou Metro traction, speed and position data."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="CSV/XLSX file(s) or directory/directories containing real-operation data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "real_strategy_policy.pt",
        help="Output checkpoint path for the learned real-traction strategy.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--history-len", type=int, default=12)
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="Fixed sampling interval in seconds. Guangzhou Metro files use 0.1 s.",
    )
    parser.add_argument(
        "--max-abs-force-n",
        type=float,
        default=300000.0,
        help="Force used to normalize traction/braking force labels.",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=0.08,
        help="Weight for penalizing abrupt changes from previous real force ratio.",
    )
    parser.add_argument("--max-gradient-per-mille", type=float, default=200.0)
    parser.add_argument("--max-speed-kmh", type=float, default=130.0)
    parser.add_argument("--max-filter-force-n", type=float, default=400000.0)
    parser.add_argument("--min-train-mass-kg", type=float, default=180000.0)
    parser.add_argument("--max-train-mass-kg", type=float, default=360000.0)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Override automatic device selection.",
    )
    parser.add_argument(
        "--group-cols",
        nargs="*",
        default=(),
        help=(
            "Columns used to split independent trips/sections, e.g. "
            "--group-cols train_id section_name. If omitted, source file and "
            "common id columns are used."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        data_paths=args.data,
        output_path=args.output,
        config=config,
        group_columns=args.group_cols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        device_name=args.device,
    )
    print("\nSaved real-traction strategy checkpoint:")
    print(f"  {args.output.resolve()}")
    print(
        f"  windows={stats.sequence_count}, train_loss={stats.train_loss:.6f}, "
        f"validation_loss={stats.validation_loss:.6f}"
    )
    print(
        f"  samples={stats.sample_count}, accepted={stats.accepted_sample_count}, "
        f"filtered={stats.filtered_sample_count}"
    )
    if stats.filter_reasons:
        print("  filter reasons: " + ", ".join(
            f"{key}={value}" for key, value in sorted(stats.filter_reasons.items())
        ))
    print("  feature order: " + ", ".join(stats.feature_names))


if __name__ == "__main__":
    main()
