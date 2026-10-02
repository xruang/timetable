"""Create a small server bundle without historical results or caches."""

from __future__ import annotations

import zipfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_DIR / "SAC_server_bundle.zip"


def selected_files() -> list[Path]:
    files = [
        PROJECT_DIR / "run_experiment.py",
        PROJECT_DIR / "run_single_scenario.py",
        PROJECT_DIR / "run_sac_diagnostics.py",
        PROJECT_DIR / "requirements.txt",
        PROJECT_DIR / "README.md",
        PROJECT_DIR / "MODELING_OVERVIEW.md",
        PROJECT_DIR / "PROJECT_SUMMARY.md",
        PROJECT_DIR / "CURRENT_TRAIN_AND_LINE_MODEL.md",
        PROJECT_DIR / "SERVER_TRAINING.md",
        PROJECT_DIR / "LOWER_FIXES_20260917.md",
        PROJECT_DIR / ".gitignore",
    ]
    files.extend((PROJECT_DIR / "timetable").glob("*.py"))
    files.extend((PROJECT_DIR / "timetable" / "data").rglob("*.xlsx"))
    files.extend((PROJECT_DIR / "experiments").glob("*.json"))
    files.extend((PROJECT_DIR / "experiments").glob("*.md"))
    files.extend((PROJECT_DIR / "server").glob("*.py"))
    files.extend((PROJECT_DIR / "server").glob("*.sh"))
    return sorted({path for path in files if path.is_file()})


def main() -> None:
    with zipfile.ZipFile(
        OUTPUT_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in selected_files():
            archive.write(path, Path("SAC") / path.relative_to(PROJECT_DIR))
    size_mib = OUTPUT_PATH.stat().st_size / (1024 ** 2)
    print(f"Created {OUTPUT_PATH} ({size_mib:.2f} MiB)")


if __name__ == "__main__":
    main()
