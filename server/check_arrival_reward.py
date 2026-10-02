"""Actual-line reward ordering check; no expert data enters SAC training."""
from pathlib import Path
import argparse
import csv
import json
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_single_scenario import make_environment
from timetable.bidirectional_model import build_bidirectional_model
from timetable.lower_continuous import force_from_ratio
from timetable.train_model import create_train_config_from_mass
from timetable.timetable_0601_bidirectional_online import build_parser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/arrival_reward_validation'))
    args = parser.parse_args()
    model_args = build_parser().parse_args(['--simulation-dt-s', '0.5'])
    model = build_bidirectional_model(model_args)
    base = next(e for e in model[3][0] if e.section.name == 'HC-DGNL')
    cfg = replace(create_train_config_from_mass('Reward validation', 255), dt_s=0.5)
    gamma, target = 0.9998, 87.0
    rows = []
    for profile in ('legacy', 'arrival_v1'):
        for ratio in [-0.1, 0.0, 0.1, 0.2, *np.linspace(0.4, 1.0, 61)]:
            env = make_environment(base.section, cfg, base.direction_sign,
                base.section_index_normalized, gamma, profile)
            env.reset(target)
            rewards = []
            shaping = []
            while not env.done:
                force = force_from_ratio(env, ratio)
                action = env.ACTION_FULL_TRACTION if ratio > 0 else (
                    env.ACTION_COAST if ratio == 0 else env.ACTION_BRAKE)
                _, reward, _, info = env.step(action, force)
                rewards.append(reward)
                shaping.append(info['schedule_progress_component'])
            if profile == 'arrival_v1':
                assert abs(sum(gamma**i*r for i,r in enumerate(shaping))) < 1e-8, (
                    'Terminal potential leaked into the discounted task objective')
            result = env.result()
            rows.append(dict(profile=profile, ratio=float(ratio), strict=result.strict_timetable_feasible,
                within5s=result.task_success_5s, timed_out=result.timed_out,
                elapsed_s=result.actual_time_s,
                arrival_error_s=None if result.timed_out else result.timetable_error_s,
                progress=(env.position_m-base.section.start_m)/base.section.length_m,
                traction_kwh=result.traction_kwh, reward=result.episode_reward,
                discounted_return=sum(gamma**i*r for i,r in enumerate(rewards)),
                max_speed_violation_kmh=result.max_speed_limit_violation_kmh,
                arrival_speed_mps=result.arrival_speed_mps))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'reward_ordering.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    new = [r for r in rows if r['profile'] == 'arrival_v1']
    old = [r for r in rows if r['profile'] == 'legacy']
    for before, after in zip(old, new):
        for key in ('elapsed_s', 'progress', 'traction_kwh', 'within5s'):
            assert before[key] == after[key], f'Reward profile changed dynamics: {key}'
    failure_probes = [next(r for r in new if r['ratio'] == ratio) for ratio in (0.0, 0.1, 0.2, 0.5)]
    assert all(r['timed_out'] for r in failure_probes)
    assert all(a['discounted_return'] < b['discounted_return']
        for a, b in zip(failure_probes, failure_probes[1:])), 'Failure progress ordering failed'
    good = [r for r in new if r['within5s']]
    bad = [r for r in new if not r['within5s']]
    assert good and bad, 'Need both feasible and failed trajectories; do not start training.'
    assert min(r['discounted_return'] for r in good) > max(r['discounted_return'] for r in bad), (
        'Reward ordering failed: a sampled failed trajectory outranks feasible driving.')
    best = min(good, key=lambda r: abs(r['arrival_error_s']))
    summary = dict(checked_rollouts=len(rows), feasible_baselines=len(good), best_baseline=best,
        minimum_feasible_return=min(r['discounted_return'] for r in good),
        maximum_failed_return=max(r['discounted_return'] for r in bad),
        note='Fixed-action feasibility/reward check, NOT trained SAC success or a global guarantee.')
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
