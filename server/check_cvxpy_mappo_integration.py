"""Interface check for MAPPO environment + model-based CVXPY lower layer."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timetable.cvxpy_lower import CvxpyLowerOptions, CvxpyLowerSolver
from timetable.line_profile import StationSection, TrackSegment
from timetable.lower_control import LowerSectionEnvironment
from timetable.online_hierarchical import OnlineHierarchicalTimetableEnvironment
from timetable.train_model import TRAIN_CONFIGS, TrainDynamics


class CvxpyMappoIntegrationTests(unittest.TestCase):
    def test_upper_step_receives_recomputed_lower_result(self):
        section = StationSection(
            "THP-TD",
            0.0,
            1000.0,
            [TrackSegment(0.0, 1000.0, 0.0, None, 100.0)],
        )
        config = replace(
            TRAIN_CONFIGS[4],
            max_acceleration_mps2=1.0,
            max_deceleration_mps2=1.0,
        )
        section_environment = LowerSectionEnvironment(
            section,
            TrainDynamics(config),
            arrival_tolerance_s=0.5,
        )
        lower_solver = CvxpyLowerSolver(
            CvxpyLowerOptions(
                spatial_segments=40,
                speed_limit_ratio=0.8,
                arrival_tolerance_s=0.5,
            )
        )
        environment = OnlineHierarchicalTimetableEnvironment(
            initial_timetables_by_train=[[75.0]],
            environments_by_train=[[section_environment]],
            lower_agents=[],
            base_departures_s=[0.0],
            direction_groups=[[0]],
            physical_section_names_by_train=[["THP-TD"]],
            supply_zone_section_groups=[["THP-TD"]],
            supply_zone_names=["test-zone"],
            station_dwell_s=60.0,
            minimum_headway_s=120.0,
            runtime_adjustment_span_s=0.0,
            dwell_adjustment_span_s=0.0,
            traction_energy_weight=0.5,
            regenerative_match_weight=0.5,
            rollout_cache_quantum_s=0.1,
            lower_model_solver=lower_solver,
        )

        _, _, done, _ = environment.step(np.zeros((1, 2), dtype=np.float32))
        result = environment.results_by_train[0][0]
        self.assertTrue(done)
        self.assertIsNotNone(result)
        self.assertTrue(result.strict_timetable_feasible)
        self.assertAlmostEqual(result.actual_time_s, 75.0, places=2)
        self.assertEqual(len(result.trajectory), 41)
        self.assertEqual(environment.rollout_cache.misses, 1)
        self.assertGreaterEqual(environment.rollout_cache.hits, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
