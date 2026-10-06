"""Shared experiment constants."""

from pathlib import Path

import numpy as np


SECTION_NAMES = [
    "THP-TD",
    "TD-HC",
    "HC-DGNL",
    "DGNL-THSC",
    "THSC-SZL",
    "SZL-SC",
    "SC-SY",
    "SY-SX",
    "SX-CP",
    "CP-JK",
    "JK-ZLX",
    "ZLX-ZL",
    "ZL-ZX",
    "ZX-KB",
    "KB-FG",
    "FG-ZC",
    "ZC-ST",
    "ST-ZG",
    "ZG-ZCP",
]

REAL_SUPPLY_ZONE_NAMES = [
    "THP-DGNL",
    "DGNL-SY",
    "SY-ZL",
    "ZL-ST",
    "ST-ZCP",
]

REAL_SUPPLY_ZONE_SECTION_GROUPS = [
    ("THP-TD", "TD-HC", "HC-DGNL"),
    ("DGNL-THSC", "THSC-SZL", "SZL-SC", "SC-SY"),
    ("SY-SX", "SX-CP", "CP-JK", "JK-ZLX", "ZLX-ZL"),
    ("ZL-ZX", "ZX-KB", "KB-FG", "FG-ZC", "ZC-ST"),
    ("ST-ZG", "ZG-ZCP"),
]

INITIAL_TIMETABLE = np.asarray(
    [
        184,
        131,
        87,
        121,
        185,
        122,
        142,
        148,
        160,
        251,
        260,
        120,
        130,
        164,
        154,
        176,
        221,
        236,
        144,
    ],
    dtype=np.float32,
)

INITIAL_SPEED_LIMIT_KMH = 40.0
DEFAULT_STATION_DWELL_S = 60.0
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_LINE_DIR = DATA_DIR / "up"
DEFAULT_DOWN_SLOPE_PATH = DATA_DIR / "down" / "slope.xlsx"
DEFAULT_DOWN_CURVATURE_PATH = DATA_DIR / "down" / "curvature.xlsx"
KWH_IN_JOULES = 3.6e6
EPS = 1e-9
