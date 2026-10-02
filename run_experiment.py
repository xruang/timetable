"""Unified launcher for reproducible timetable experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from timetable.timetable_0601_bidirectional_online import (
    build_parser as build_training_parser,
)
from timetable.timetable_0601_bidirectional_online import main as run_training


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "experiments"
RESULTS_DIR = PROJECT_DIR / "results"
RESERVED_PARAMETERS = {"output_dir", "run_name", "checkpoint", "experiment_config"}


def load_config(name_or_path: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(name_or_path)
    if not candidate.exists():
        candidate = CONFIG_DIR / (
            name_or_path if name_or_path.endswith(".json") else f"{name_or_path}.json"
        )
    if not candidate.exists():
        raise FileNotFoundError(f"Experiment config not found: {name_or_path}")
    with candidate.resolve().open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config.get("parameters"), dict):
        raise ValueError("Experiment config must contain an object named 'parameters'.")
    return candidate.resolve(), config


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"Override must use key=value syntax: {item}")
    key, raw_value = item.split("=", 1)
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key.replace("-", "_"), value


def parameters_to_argv(parameters: dict[str, Any]) -> list[str]:
    parser = build_training_parser()
    actions = {action.dest: action for action in parser._actions}
    argv: list[str] = []
    for key, value in parameters.items():
        if key in RESERVED_PARAMETERS:
            raise ValueError(f"'{key}' is managed by run_experiment.py")
        action = actions.get(key)
        if action is None:
            raise ValueError(f"Unknown training parameter: {key}")
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                argv.append(option)
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not bool(value):
                argv.append(option)
            continue
        if value is not None:
            argv.extend((option, str(value)))
    parser.parse_args(argv)
    return argv


def list_configs() -> None:
    for path in sorted(CONFIG_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        print(f"{path.stem:24s} {config.get('description', '')}")


def run_config(
    name_or_path: str,
    overrides: Sequence[str],
    dry_run: bool,
) -> Path | None:
    config_path, config = load_config(name_or_path)
    parameters = dict(config["parameters"])
    for item in overrides:
        key, value = parse_override(item)
        parameters[key] = value

    experiment_name = str(config.get("experiment_name") or config_path.stem)
    seed = int(parameters.get("seed", 7))
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{seed}"
    output_base = RESULTS_DIR / experiment_name
    while (output_base / run_name).exists():
        run_name += "_2"
    run_dir = output_base / run_name

    resolved_config = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "description": config.get("description", ""),
        "source_config": str(config_path),
        "run_name": run_name,
        "parameters": parameters,
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    training_argv = parameters_to_argv(parameters)
    training_argv.extend(
        (
            "--output-dir",
            str(output_base),
            "--run-name",
            run_name,
        )
    )
    print(f"Experiment : {experiment_name}")
    print(f"Stage      : {parameters.get('experiment_stage', 'hierarchical')}")
    print(f"Algorithm  : {parameters.get('lower_algorithm', 'dqn')}")
    print(f"Result dir : {run_dir}")
    if dry_run:
        print("Dry run only; no training was started.")
        return None

    status = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "run_name": run_name,
    }
    temporary_config: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(resolved_config, handle, ensure_ascii=False, indent=2)
            temporary_config = Path(handle.name)
        training_argv.extend(("--experiment-config", str(temporary_config)))
        completed_dir = run_training(training_argv)
        status["status"] = "completed"
        status["completed_at"] = datetime.now().isoformat(timespec="seconds")
        with (completed_dir / "run_status.json").open("w", encoding="utf-8") as handle:
            json.dump(status, handle, ensure_ascii=False, indent=2)
        return completed_dir
    except Exception:
        status["status"] = "failed"
        status["failed_at"] = datetime.now().isoformat(timespec="seconds")
        status["traceback"] = traceback.format_exc()
        if run_dir.exists():
            with (run_dir / "run_status.json").open("w", encoding="utf-8") as handle:
                json.dump(status, handle, ensure_ascii=False, indent=2)
        raise
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a named timetable experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List available experiment configurations.")
    run_parser = subparsers.add_parser("run", help="Run one experiment configuration.")
    run_parser.add_argument("config", help="Config name or JSON path.")
    run_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one config value for this run; may be repeated.",
    )
    run_parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list":
        list_configs()
    else:
        run_config(args.config, args.set, args.dry_run)


if __name__ == "__main__":
    main()
