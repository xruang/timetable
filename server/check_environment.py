"""Validate a Linux training environment before starting a long run."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    import matplotlib
    import numpy as np
    import openpyxl
    import torch

    print(f"Project     : {PROJECT_DIR}")
    print(f"Platform    : {platform.platform()}")
    print(f"Python      : {platform.python_version()}")
    print(f"NumPy       : {np.__version__}")
    print(f"Matplotlib  : {matplotlib.__version__}")
    print(f"openpyxl    : {openpyxl.__version__}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"Torch CUDA  : {torch.version.cuda}")
    print(f"CUDA ready  : {torch.cuda.is_available()}")

    if sys.version_info < (3, 9):
        raise RuntimeError("Python 3.9 or newer is required.")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a GPU. Check the "
            "NVIDIA driver and the installed PyTorch wheel."
        )
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / (1024 ** 3)
            print(f"GPU {index}       : {properties.name} ({memory_gib:.1f} GiB)")
        torch.zeros(1, device="cuda")

    required_data = (
        PROJECT_DIR / "timetable" / "data" / "up" / "slope.xlsx",
        PROJECT_DIR / "timetable" / "data" / "up" / "curvature.xlsx",
        PROJECT_DIR / "timetable" / "data" / "down" / "slope.xlsx",
        PROJECT_DIR / "timetable" / "data" / "down" / "curvature.xlsx",
    )
    missing = [str(path) for path in required_data if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing line data: " + ", ".join(missing))

    from timetable.timetable_0601_bidirectional_online import (
        build_bidirectional_model,
        build_parser as build_training_parser,
    )

    model_args = build_training_parser().parse_args(
        ["--up-trains", "1", "--down-trains", "1"]
    )
    model = build_bidirectional_model(model_args)
    if len(model[0].sections) != 19 or len(model[1].sections) != 19:
        raise RuntimeError("Expected 19 interstation sections in each direction.")
    print("Line data   : OK (19 up + 19 down sections)")
    print("Environment : READY")


if __name__ == "__main__":
    main()
