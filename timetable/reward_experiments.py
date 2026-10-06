"""Opt-in reward ablations; no changes to dynamics, policy, or replay.

Continuous terminal weights are experimental: retain the arrival_v1 slope
0.4 / s and distance slope 8 / section length; add 1 per 10 m/s for the
physical arrival speed BEFORE the simulator resets speed at the endpoint.
This is a soft objective, not an exact feasibility constraint.
"""

REWARD_PROFILES = ('legacy', 'arrival_v1', 'arrival_p16', 'average_speed_v1',
                   'continuous_p8', 'continuous_p16')


def shaping_weight(profile):
    return 16.0 if profile in ('arrival_p16', 'continuous_p16') else 8.0


def terminal_cost(elapsed_s, target_s, remaining_m, length_m, speed_mps):
    return (0.4 * abs(elapsed_s - target_s)
            + 8.0 * abs(remaining_m) / length_m
            + 0.1 * abs(speed_mps))


def reward_spec(profile):
    return dict(profile=profile,
                shaping_weight=0.0 if profile == 'average_speed_v1' else shaping_weight(profile),
                terminal='continuous_absolute_errors' if profile.startswith('continuous')
                else 'continuous_arrival_error_no_bonus' if profile == 'average_speed_v1'
                else 'original', time_cost_per_s=0.4,
                distance_cost_per_section=8.0, speed_cost_per_mps=0.1,
                terminal_potential=0.0,
                average_speed_weight=4.0 if profile == 'average_speed_v1' else 0.0,
                average_speed_normalization='local_safe_speed',
                failure_cost=8.0 if profile == 'average_speed_v1' else None)
