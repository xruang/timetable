"""Select the lowest-energy strictly punctual policy from repeated single-scenario runs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--profile", default="average_speed_v1")
    parser.add_argument("--max-terminal-shield-ratio", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidates = []
    for evaluation_path in sorted((args.run_root / args.profile).glob("seed*/best_feasible_evaluation.json")):
        row = json.loads(evaluation_path.read_text(encoding="utf-8"))
        run_dir = evaluation_path.parent
        config = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
        error = row.get("arrival_error_s")
        valid = (
            bool(row.get("strict"))
            and error is not None
            and abs(float(error)) <= 0.5
            and float(row.get("max_speed_violation_kmh", 0.0)) <= 1e-9
            and (run_dir / "best_feasible_policy.pt").is_file()
        )
        candidates.append({
            "seed": int(config["seed"]),
            "valid": valid,
            "arrival_error_s": error,
            "traction_kwh": float(row["traction_kwh"]),
            "max_speed_violation_kmh": float(row.get("max_speed_violation_kmh", 0.0)),
            "terminal_shields": int(row.get("terminal_shields", 0)),
            "control_steps": int(row.get("control_steps", 0)),
            "terminal_shield_ratio": float(row.get("terminal_shield_ratio", 1.0)),
            "meets_recommended_shield_ratio": (
                float(row.get("terminal_shield_ratio", 1.0))
                <= args.max_terminal_shield_ratio
            ),
            "checkpoint": str(run_dir / "best_feasible_policy.pt"),
            "run_dir": str(run_dir),
        })

    valid = [row for row in candidates if row["valid"]]
    if not valid:
        raise RuntimeError("No strictly punctual, speed-safe checkpoint was found")
    selected = min(valid, key=lambda row: (
        row["terminal_shield_ratio"], row["traction_kwh"], abs(row["arrival_error_s"])
    ))
    source = Path(selected["run_dir"])
    output = args.output or (args.run_root / "selected_single_policy")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "best_feasible_policy.pt", output / "best_feasible_policy.pt")
    shutil.copy2(source / "experiment_config.json", output / "experiment_config.json")
    shutil.copy2(source / "best_feasible_evaluation.json", output / "best_feasible_evaluation.json")

    with (output / "candidate_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    summary = {
        "selection_rule": ("require abs(arrival_error_s)<=0.5 and no speed violation; "
                           "then minimize terminal_shield_ratio, traction_kwh, and time error"),
        "max_terminal_shield_ratio": args.max_terminal_shield_ratio,
        "scenario": {key: json.loads((source / "experiment_config.json").read_text(encoding="utf-8"))[key]
                     for key in ("direction", "section", "mass_t", "target_time_s", "dt_s", "reward_profile")},
        "valid_seed_count": len(valid),
        "candidate_seed_count": len(candidates),
        "selected": selected,
    }
    (output / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
