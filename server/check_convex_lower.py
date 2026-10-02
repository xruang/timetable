"""Small deterministic checks for the CO_V2-compatible lower solver."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timetable.convex_lower import ConvexLowerOptions, ConvexLowerSolver
from timetable.line_profile import StationSection, TrackSegment
from timetable.train_model import TRAIN_CONFIGS


class ConvexLowerTests(unittest.TestCase):
    def test_flat_section_returns_recomputed_trajectories(self):
        section = StationSection(
            "flat-test",
            0.0,
            1000.0,
            [TrackSegment(0.0, 1000.0, 0.0, None, 100.0)],
        )
        config = replace(TRAIN_CONFIGS[4], max_acceleration_mps2=1.0, max_deceleration_mps2=1.0)
        solver = ConvexLowerSolver(
            ConvexLowerOptions(
                spatial_segments=40,
                speed_limit_ratio=0.8,
                arrival_tolerance_s=0.5,
                output_flag=False,
            )
        )
        result = solver.solve(section, config, 75.0)
        self.assertTrue(result.has_solution, result.validation_messages)
        self.assertEqual(result.position_m.size, 41)
        self.assertEqual(result.speed_kmh.size, 41)
        self.assertEqual(result.optimized_force_kn.size, 40)
        self.assertEqual(result.recomputed_force_kn.size, 40)
        self.assertIsNotNone(result.actual_time_s)
        self.assertIsNotNone(result.actual_net_energy_kwh)
        lower_result = solver.to_lower_result(result, section, config)
        self.assertEqual(len(lower_result.trajectory), 41)
        self.assertAlmostEqual(lower_result.actual_time_s, result.actual_time_s)


if __name__ == "__main__":
    unittest.main(verbosity=2)

