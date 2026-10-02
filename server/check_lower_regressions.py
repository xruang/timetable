"""Lower-controller regression tests. Run: python server/check_lower_regressions.py."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timetable.line_profile import StationSection, TrackSegment
from timetable.train_model import TRAIN_CONFIGS, TrainDynamics
from timetable.lower_control import LOWER_REWARD, LowerSectionEnvironment
from timetable.lower_continuous import (
    SACAgent, DDPGAgent, SACActor, force_from_ratio, _training_environment,
    pretrain_continuous_actor_with_heuristic,
)
from timetable.timetable_0601_bidirectional_online import _validate_lower_action_semantics

PHYSICAL_GRID = "--physical-grid" in sys.argv
if PHYSICAL_GRID:
    sys.argv.remove("--physical-grid")


def environment(length=1200.0):
    section = StationSection("test", 0.0, length,
        [TrackSegment(0.0, length, 0.0, None, 125.0)])
    return LowerSectionEnvironment(section,
        TrainDynamics(replace(TRAIN_CONFIGS[0], dt_s=0.2)))


class ShortEnvironment(LowerSectionEnvironment):
    def reset(self, target_time_s):
        obs = super().reset(target_time_s)
        self.max_steps = 3
        self.infos = []
        return obs

    def step(self, *args):
        output = super().step(*args)
        self.infos.append(output[3])
        return output


class LowerRegressionTests(unittest.TestCase):
    @unittest.skipUnless(PHYSICAL_GRID, "add --physical-grid to run the 76 actual-line rollouts")
    def test_physical_grid_safety_and_parking(self):
        from timetable.bidirectional_model import build_bidirectional_model
        from timetable.timetable_0601_bidirectional_online import build_parser
        args = build_parser().parse_args(["--simulation-dt-s", "0.2"])
        model = build_bidirectional_model(args)
        count = 0
        for train_idx in (0, 6):
            for section_idx, base in enumerate(model[3][train_idx]):
                for template in (TRAIN_CONFIGS[0], TRAIN_CONFIGS[-1]):
                    env = LowerSectionEnvironment(base.section,
                        TrainDynamics(replace(template, dt_s=0.2)))
                    env.reset(float(model[4][train_idx, section_idx]))
                    while not env.done:
                        env.step(env.ACTION_FULL_TRACTION,
                            env.dynamics.max_traction_force_n(env.speed_mps))
                    result = env.result()
                    with self.subTest(train=train_idx, section=base.section.name,
                                      mass=template.mass_kg):
                        self.assertFalse(result.timed_out)
                        self.assertLessEqual(result.max_speed_limit_violation_kmh, 1e-6)
                        self.assertLessEqual(result.arrival_speed_mps, 1e-3)
                        self.assertLess(result.actual_time_s, result.target_time_s)
                    count += 1
        self.assertEqual(count, 76)
        print("Physical grid: 76 safe parked rollouts; NOT a punctual-policy success test.")

    def test_raw_proposals_are_replayed_for_sac_and_ddpg(self):
        base = environment()
        base.dynamics = TrainDynamics(replace(base.dynamics.config, max_acceleration_mps2=0.1))
        for agent_type in (SACAgent, DDPGAgent):
            env = ShortEnvironment(base.section, base.dynamics)
            agent = agent_type(16, batch_size=8, device_name="cpu")
            with torch.no_grad():
                for parameter in agent.actor.parameters():
                    parameter.zero_()
                if agent_type is SACAgent:
                    agent.actor.mean.bias.fill_(float(np.arctanh(0.9)))
                    agent.set_training_mode(False)
                else:
                    agent.actor.layers[-2].bias.fill_(float(np.arctanh(0.9)))
                    agent.set_exploration_noise(0.0)
            env.rollout(agent, 100.0, training=True)
            self.assertEqual(len(agent.replay), 3)
            for transition, info in zip(agent.replay.buffer, env.infos):
                self.assertAlmostEqual(transition[1], 0.9, places=6)
                self.assertAlmostEqual(info["proposed_force_ratio"], 0.9, places=6)
                self.assertLess(info["force_ratio"], 0.9)  # smoothing altered execution
            agent.batch_size = 3
            self.assertTrue(np.isfinite(agent.update()))
            self.assertTrue(all(torch.isfinite(p).all() for p in agent.actor.parameters()))

    def test_no_dynamic_clipping_before_environment(self):
        env = environment()
        env.dynamics = TrainDynamics(replace(env.dynamics.config, max_acceleration_mps2=0.1))
        env.reset(100.0)
        env.speed_mps = 20.0
        self.assertLess(env.clip_force_ratio(1.0), 1.0)
        self.assertAlmostEqual(force_from_ratio(env, 1.0),
            env.dynamics.max_traction_force_n(20.0))

    def test_continuous_small_force_is_not_zeroed_by_label(self):
        env = environment()
        for force in (1000.0, -1000.0):
            self.assertEqual(env._clip_continuous_force_n(force, env.ACTION_COAST), force)
            env.reset(100.0)
            env.speed_mps = 10.0
            env.position_m = 250.0
            env.previous_force_n = force
            _, _, _, info = env.step(env.ACTION_COAST, force)
            self.assertAlmostEqual(info["force_n"], force)

    def test_execution_normalization_uses_pre_step_velocity(self):
        env = environment()
        env.reset(260.0)
        env.position_m = 250.0
        env.speed_mps = 20.0
        maximum = env.dynamics.max_traction_force_n(env.speed_mps)
        env.previous_force_n = maximum
        _, _, _, info = env.step(env.ACTION_FULL_TRACTION, maximum)
        self.assertAlmostEqual(info["force_ratio"], info["force_n"] / maximum)
        self.assertLessEqual(info["force_ratio"], 1.0)

    def test_terminal_projection_is_event_triggered_not_latched(self):
        env = environment()
        env.reset(100.0)
        env.position_m = env.section.end_m - 50.0
        env.speed_mps = 2.0
        resistance = env.dynamics.total_resistance_n(2.0, env.section.segment_at(env.position_m))
        safe_force = 20000.0
        self.assertEqual(env._terminal_action_shield_force_n(safe_force, resistance), safe_force)
        env.speed_mps = 8.0
        unsafe = env.dynamics.max_traction_force_n(8.0)
        self.assertLess(env._terminal_action_shield_force_n(unsafe, resistance), unsafe)
        env.speed_mps = 2.0
        self.assertEqual(env._terminal_action_shield_force_n(safe_force, resistance), safe_force)
        self.assertFalse(hasattr(env, "terminal_braking_latched"))
        self.assertFalse(hasattr(env, "safety_brake_latched"))

    def test_saturated_gaussian_has_finite_correct_entropy_gradient(self):
        actor = SACActor(16)
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            actor.mean.bias.fill_(20.0)
            actor.log_std.bias.fill_(-5.0)
        _, log_prob = actor.sample(torch.zeros(4, 16))
        self.assertTrue(torch.isfinite(log_prob).all())
        log_prob.mean().backward()
        self.assertAlmostEqual(actor.mean.bias.grad.item(), 2.0, places=3)

    def test_random_mass_preserves_dt_and_constraints(self):
        base = environment()
        base.dynamics = TrainDynamics(replace(base.dynamics.config,
            max_acceleration_mps2=0.7, traction_efficiency=0.85))
        sampled, _ = _training_environment(base, "test", True, 235.0, 275.0, 0.15, 0.05)
        self.assertEqual(sampled.dynamics.config.dt_s, 0.2)
        self.assertEqual(sampled.dynamics.config.max_acceleration_mps2, 0.7)
        self.assertEqual(sampled.dynamics.config.traction_efficiency, 0.85)

    def test_pretrain_replay_uses_heuristic_proposal_not_execution(self):
        base = environment()
        env = ShortEnvironment(base.section, base.dynamics)
        agent = SACAgent(16, batch_size=8, device_name="cpu")
        pretrain_continuous_actor_with_heuristic([env], agent, np.array([100.0]),
            1, 0, "test", "sac", log_interval=10)
        for transition, info in zip(agent.replay.buffer, env.infos):
            self.assertAlmostEqual(transition[1], info["proposed_force_ratio"])

    def test_unsafe_endpoint_cannot_be_reported_feasible(self):
        env = environment()
        env.reset(100.0)
        env.position_m = env.section.end_m - 0.01
        env.speed_mps = 20.0
        env.elapsed_s = 99.8
        env.step(env.ACTION_BRAKE, -env.dynamics.config.max_braking_force_n)
        result = env.result()
        self.assertGreater(result.arrival_speed_mps, 10.0)
        self.assertFalse(result.feasible)

    def test_physical_feasibility_is_separate_from_strict_punctuality(self):
        env = environment()
        env.arrival_tolerance_s = 0.5
        env.reset(100.0)
        env.position_m = env.section.end_m
        env.speed_mps = 0.0
        env.arrival_speed_before_stop_mps = 0.0
        env.elapsed_s = 101.0
        env.done = True
        result = env.result()
        self.assertTrue(result.feasible)
        self.assertFalse(result.strict_timetable_feasible)
        self.assertTrue(result.task_success_5s)

    def test_checkpoint_action_semantics_gate(self):
        agent = SACAgent(16, device_name="cpu")
        with self.assertRaises(ValueError):
            _validate_lower_action_semantics({}, [agent])
        _validate_lower_action_semantics({"lower_action_semantics": agent.action_semantics}, [agent])

    def test_collection_warmup_freezes_networks_until_threshold(self):
        agent = SACAgent(16, device_name='cpu', batch_size=4,
                         learning_starts=8, replay_capacity=32)
        networks = (agent.actor, agent.critic1, agent.critic2,
                    agent.target_critic1, agent.target_critic2)
        before = [[p.detach().clone() for p in n.parameters()] for n in networks]
        for _ in range(7):
            agent.remember(np.zeros(16, dtype=np.float32), 0.3, -4.0,
                           np.ones(16, dtype=np.float32), True)
            self.assertIsNone(agent.update())
        for net, values in zip(networks, before):
            self.assertTrue(all(torch.equal(p, v) for p, v in zip(net.parameters(), values)))
        agent.remember(np.zeros(16, dtype=np.float32), 0.3, -4.0,
                       np.ones(16, dtype=np.float32), True)
        self.assertTrue(np.isfinite(agent.update()))
        for net, values in zip(networks, before):
            self.assertTrue(any(not torch.equal(p, v) for p, v in zip(net.parameters(), values)))
        with self.assertRaises(ValueError):
            SACAgent(16, device_name='cpu', learning_starts=33, replay_capacity=32)

    def test_failure_deadline_is_target_plus_five_seconds(self):
        env = environment()
        env.reset(100.0)
        self.assertEqual(env.max_steps, 525)
        env.step_count = env.max_steps - 1
        env.elapsed_s = 104.8
        env.position_m = 200.0
        env.speed_mps = 10.0
        _, _, done, _ = env.step(env.ACTION_COAST, 0.0)
        self.assertTrue(done)
        self.assertTrue(env.result().timed_out)
        self.assertFalse(env.result().task_success_5s)

    def test_bounded_reward_at_both_time_steps(self):
        for dt in (0.2, 0.5):
            env = environment()
            env.dynamics = TrainDynamics(replace(env.dynamics.config, dt_s=dt))
            env.reset(100.0)
            for _ in range(12):
                _, reward, _, info = env.step(env.ACTION_FULL_TRACTION,
                                             env.dynamics.config.max_traction_force_n)
                self.assertTrue(np.isfinite(reward))
                self.assertEqual(info['force_change_component'], 0.0)
                self.assertEqual(info['force_jump_penalty'], 0.0)
                self.assertEqual(info['punctuality_reward_component'], 0.0)
                self.assertGreaterEqual(info['traction_reward_component'],
                                        -LOWER_REWARD.bounded_energy_weight)
                self.assertLessEqual(info['traction_reward_component'], 0.0)
                self.assertLessEqual(abs(info['schedule_progress_component']),
                                     2 * LOWER_REWARD.schedule_progress_weight)
                self.assertAlmostEqual(reward, sum(info[k] for k in (
                    'traction_reward_component', 'comfort_reward_component',
                    'schedule_progress_component', 'speed_limit_component',
                    'speed_limit_action_shield_component', 'punctuality_reward_component',
                    'average_speed_component', 'premature_stop_component',
                    'timeout_component')))
            env.step_count = env.max_steps - 1
            env.elapsed_s = 105.0 - dt
            env.position_m = 200.0
            env.speed_mps = 10.0
            _, reward, done, info = env.step(env.ACTION_COAST, 0.0)
            self.assertTrue(done)
            self.assertEqual(info['timeout_component'], -LOWER_REWARD.bounded_failure_weight)
            self.assertEqual(info['punctuality_reward_component'], 0.0)
            self.assertGreater(reward, -10.0)

    def test_average_speed_reward_is_dense_and_has_no_arrival_bonus(self):
        env = environment()
        env.reward_profile = 'average_speed_v1'
        env.reset(100.0)
        _, _, _, info = env.step(env.ACTION_FULL_TRACTION,
                                 env.dynamics.config.max_traction_force_n)
        self.assertLess(info['average_speed_component'], 0.0)
        self.assertEqual(info['punctuality_reward_component'], 0.0)
        env.position_m = env.section.end_m - 0.01
        env.speed_mps = 0.01
        env.elapsed_s = 99.8
        env._predict_step_motion = lambda *args: (0.0, 0.2, 0.0, env.remaining_m)
        _, _, done, info = env.step(env.ACTION_FULL_TRACTION,
                                   env.dynamics.config.max_traction_force_n)
        self.assertTrue(done)
        self.assertLessEqual(info['punctuality_reward_component'], 0.0)

    def test_early_stop_penalty_exceeds_observed_full_failure_return(self):
        env = environment()
        env.reset(100.0)
        env.position_m = 200.0
        env.speed_mps = 0.0
        env.elapsed_s = 10.0
        # Isolate reward/termination from the low-speed action guard.
        env._predict_step_motion = lambda *args: (0.0, 0.2, 0.0, 0.0)
        _, reward, done, info = env.step(env.ACTION_BRAKE,
                                      -env.dynamics.config.max_braking_force_n)
        self.assertTrue(done)
        self.assertEqual(info['premature_stop_component'], -7.0)
        self.assertEqual(info['punctuality_reward_component'], 0.0)
        self.assertLess(reward, -5.9176)

    def test_arrival_has_no_terminal_time_error_cost(self):
        for elapsed in (50.0, 99.0):
            env = environment()
            env.reset(100.0)
            env.position_m = env.section.end_m - 0.01
            env.speed_mps = 0.01
            env.elapsed_s = elapsed
            env._predict_step_motion = lambda *args: (0.0, 0.2, 0.0, env.remaining_m)
            _, _, done, info = env.step(env.ACTION_FULL_TRACTION,
                                       env.dynamics.config.max_traction_force_n)
            self.assertTrue(done)
            self.assertEqual(info['punctuality_reward_component'], 0.0)
            self.assertEqual(info['force_change_component'], 0.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
