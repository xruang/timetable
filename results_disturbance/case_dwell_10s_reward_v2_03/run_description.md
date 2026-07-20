# Disturbance Timetable Simulation Run

- Run name: `case_dwell_10s_reward_v2_03`
- Created at: `2026-06-18T10:35:05`
- Mode/phase: `train` / `train`
- Seed: `7`
- Output directory: `D:\Codd\SAC\timetable\results_disturbance\case_dwell_10s_reward_v2_03`
- Checkpoint: `D:\Codd\SAC\timetable\results_disturbance\case_dwell_10s_reward_v2_03\hierarchical_mappo_disturbance_checkpoint.pt`
- Initial headway: `480s`; minimum safe headway: `120s`
- Upper control mode: `departure`
- Lower episodes: `300`; upper episodes: `1000`
- Response table sampling: +/-`10s`, step `2s`
- Total injected dwell overrun: `80.0s`

## Train Configuration

| Train | Mass (kg) | Max traction force (N) | Max traction power (W) | Davis A | Davis B | Davis C |
|---|---:|---:|---:|---:|---:|---:|
| Train 1 | 260000.0 | 290000.0 | 3100000.0 | 0.6566 | 45.7463 | 3511.51 |
| Train 2 | 245000.0 | 270000.0 | 2900000.0 | 0.6566 | 43.5219 | 3332.7 |
| Train 3 | 250000.0 | 275000.0 | 2970000.0 | 0.6566 | 44.1445 | 3388.23 |
| Train 4 | 270000.0 | 295000.0 | 3180000.0 | 0.6566 | 47.0646 | 3632.69 |

## Dwell-Overrun Disturbances

| Train | After section | Station | Overrun (s) |
|---:|---|---|---:|
| 2 | JK-ZLX | ZLX | 10.0 |
| 2 | ZLX-ZL | ZL | 10.0 |
| 2 | KB-FG | FG | 10.0 |
| 2 | ZC-ST | ST | 10.0 |
| 3 | THP-TD | TD | 10.0 |
| 3 | DGNL-THSC | THSC | 10.0 |
| 3 | SZL-SC | SC | 10.0 |
| 3 | ST-ZG | ZG | 10.0 |

## Supply Zones

| Zone | Sections |
|---|---|
| THP-DGNL | THP-TD, TD-HC, HC-DGNL |
| DGNL-SY | DGNL-THSC, THSC-SZL, SZL-SC, SC-SY |
| SY-ZL | SY-SX, SX-CP, CP-JK, JK-ZLX, ZLX-ZL |
| ZL-ST | ZL-ZX, ZX-KB, KB-FG, FG-ZC, ZC-ST |
| ST-ZCP | ST-ZG, ZG-ZCP |

The full machine-readable configuration is saved in `run_metadata.json`.