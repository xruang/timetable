"""Regenerate lower-policy plots from an existing result directory."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate Chinese training plots without rerunning training."
    )
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir.expanduser().resolve()
    lower_path = result_dir / "lower_training_history.csv"
    if not lower_path.is_file():
        raise FileNotFoundError(f"Missing training history: {lower_path}")

    from timetable.visualization import plot_training_process

    upper_path = result_dir / "mappo_training_history.csv"
    lower_history = read_csv(lower_path)
    upper_history = read_csv(upper_path) if upper_path.is_file() else []
    obsolete_mass_plot = result_dir / "lower_reward_mass_sensitivity.png"
    if obsolete_mass_plot.is_file():
        obsolete_mass_plot.unlink()
    plot_training_process(
        lower_history,
        upper_history,
        result_dir,
        write_history=False,
    )
    print(f"Regenerated plots in: {result_dir}")


if __name__ == "__main__":
    main()
