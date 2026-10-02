"""Fixed-transition-budget reward ablation using the unchanged production SAC.

No pretraining, imitation, custom replay, curriculum, or checkpoint resumption.
Evaluation uses separate environments and does not consume training RNG.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_single_scenario as single
from timetable.lower_continuous import force_from_ratio
from timetable.reward_experiments import reward_spec, terminal_cost

PROFILES = ('arrival_v1', 'arrival_p16', 'average_speed_v1',
            'continuous_p8', 'continuous_p16')


def write_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def setup():
    args = single.build_model_parser().parse_args(['--simulation-dt-s', '0.5'])
    model = single.build_bidirectional_model(args)
    base = next(e for e in model[3][0] if e.section.name == 'HC-DGNL')
    config = replace(single.create_train_config_from_mass('Single scenario', 255), dt_s=.5)
    def factory(profile):
        return single.make_environment(base.section, config, base.direction_sign,
            base.section_index_normalized, .9998, profile)
    return factory, config


def preflight(out):
    factory, _ = setup()
    rows = []
    for ratio in [-.1, 0., .1, .2, *np.linspace(.4, 1., 61)]:
        paired = []
        for profile in PROFILES:
            env = factory(profile)
            env.reset(87.)
            total = shaping = 0.
            steps = 0
            while not env.done:
                f = force_from_ratio(env, ratio)
                _, r, _, info = env.step(env.action_for_force_n(f), f)
                total += .9998**steps * r
                shaping += .9998**steps * info['schedule_progress_component']
                steps += 1
            result = env.result()
            assert abs(shaping) < 1e-8, 'Potential leaked into full return'
            row = dict(profile=profile, ratio=float(ratio), elapsed=result.actual_time_s,
                remaining=env.remaining_m, speed=result.arrival_speed_mps,
                within5=result.task_success_5s, strict=result.strict_timetable_feasible,
                timed_out=result.timed_out, energy=result.traction_kwh,
                return_=total, terminal=info['punctuality_reward_component'],
                shaping_sum=shaping, steps=steps)
            rows.append(row)
            paired.append(row)
        for row in paired[1:]:
            for key in ('elapsed', 'remaining', 'speed', 'energy', 'steps', 'within5', 'strict'):
                assert row[key] == paired[0][key], f'Dynamics changed: {key}'
        assert abs(paired[0]['return_']-paired[1]['return_']) < 1e-8
        assert abs(paired[3]['return_']-paired[4]['return_']) < 1e-8
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out/'fixed_action_reward_check.csv', rows)
    # Actual safe trajectory; changing target here is solely a boundary probe,
    # not a change to the target used for training.
    boundary = []
    arrival = next(r for r in rows if r['profile']=='arrival_v1' and abs(r['ratio']-.75)<1e-8)['elapsed']
    for profile in ('arrival_v1', 'continuous_p8'):
        for signed_error in (-5.0001, -4.9999, -.5001, -.4999, .4999, .5001, 4.9999, 5.0001):
            env = factory(profile)
            env.reset(arrival-signed_error)
            while not env.done:
                f = force_from_ratio(env, .75)
                _, _, _, info = env.step(env.action_for_force_n(f), f)
            boundary.append(dict(profile=profile, requested_error=signed_error,
                actual_error=env.elapsed_s-env.target_time_s, remaining=env.remaining_m,
                speed=env.result().arrival_speed_mps,
                terminal=info['punctuality_reward_component']))
    write_rows(out/'terminal_boundary_check.csv', boundary)
    continuous = [r for r in boundary if r['profile']=='continuous_p8']
    for a,b in zip(continuous[::2], continuous[1::2]):
        assert abs(a['terminal']-b['terminal']) < .001
    assert terminal_cost(87,87,0,1000,1) > terminal_cost(87,87,0,1000,0)
    summary = dict(rollouts=len(rows), profiles={},
        note='Offline screening only; no global constraint or convergence guarantee.')
    for profile in PROFILES:
        subset = [r for r in rows if r['profile']==profile]
        best = max(subset, key=lambda r:r['return_'])
        assert best['strict'], 'Highest-reward sampled trajectory is not strictly feasible'
        failures = [next(r for r in subset if r['ratio']==x) for x in (0., .1, .2, .5)]
        if profile != 'average_speed_v1':
            assert all(a['return_'] < b['return_'] for a,b in zip(failures, failures[1:]))
        summary['profiles'][profile] = dict(best_sampled=best,
            minimum_within5_return=min(r['return_'] for r in subset if r['within5']),
            maximum_failed_return=max(r['return_'] for r in subset if not r['within5']))
    (out/'preflight.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


def train(args):
    single.seed_everything(args.seed)
    torch.set_num_threads(1)
    factory, cfg = setup()
    env = factory(args.profile)
    agent = single.SACAgent(16, device_name=args.device, gamma=.9998,
        initial_alpha=.02, min_alpha=.02, replay_capacity=1000000, learning_starts=10000,
        include_schedule_deviation_state=True, schedule_deviation_scale=5.,
        include_shared_context_state=True)
    agent.set_training_mode(True)
    agent.set_auto_alpha_enabled(False)
    out = args.output/args.profile/f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    resolved = dict(direction='up',section='HC-DGNL',mass_t=255.,target_time_s=87.,
        dt_s=.5,gamma=.9998,alpha=.02,seed=args.seed,reward_profile=args.profile,
        reward_spec=reward_spec(args.profile),transitions=args.steps,eval_steps=args.eval_steps,
        device=str(agent.device),replay_capacity=1000000,learning_starts=10000,
        action_semantics=agent.action_semantics,alpha_auto_tune_enabled=False,
        train_parameters=asdict(cfg),lower_reward=asdict(single.LOWER_REWARD),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(),ROOT/'run_single_scenario.py',*sorted((ROOT/'timetable').glob('*.py'))]})
    (out/'experiment_config.json').write_text(json.dumps(resolved,indent=2),encoding='utf-8')
    evaluations, history, diagnostics = [], [], []
    updates = 0
    def audit(batch):
        nonlocal updates
        updates += 1
        if updates != 1 and updates % 1000:
            return
        for key in ('q1','q2','target','reward'):
            assert torch.isfinite(batch[key]).all(), key
        terminal = batch['done'].bool()
        assert torch.equal(batch['target'][terminal],batch['reward'][terminal])
        diagnostics.append(dict(update=updates,transitions=step,
            critic_loss=agent.last_loss,actor_loss=agent.last_actor_loss,
            q_mean=batch['q1'].mean().item(),target_mean=batch['target'].mean().item(),
            terminal_n=int(terminal.sum()),
            terminal_td_abs=(batch['target']-torch.minimum(batch['q1'],batch['q2']))[terminal].abs().mean().item()
                if terminal.any() else None))
    agent.update_diagnostics_callback = audit
    best = None
    episode = step = 0
    started = time.time()
    def assess():
        nonlocal best
        # Guard against accidental evaluation effects on RNG or training diagnostics.
        py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        attrs = {k:v for k,v in vars(agent).items() if k.startswith('_diagnostic') or k.endswith('_sum') or k=='last_action_ratio'}
        row = single.evaluate(agent, factory(args.profile), 87., episode)
        random.setstate(py_rng); np.random.set_state(np_rng); torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        for k,v in attrs.items():
            setattr(agent,k,v)
        row.update(transitions=step,network_updates=updates,wall_seconds=time.time()-started)
        evaluations.append(row)
        write_rows(out/'deterministic_evaluation.csv',evaluations)
        write_rows(out/'training_history.csv',history)
        write_rows(out/'update_diagnostics.csv',diagnostics)
        rank = (bool(row['strict']), -row['terminal_shield_ratio'],
                -row['traction_kwh'], -abs(row['arrival_error_s'])) if row['within5s'] else None
        if rank is not None and (best is None or rank > best):
            best = rank
            single.save_lower_only_checkpoint(out/'best_feasible_policy.pt',[agent],[cfg],'sac')
            (out/'best_feasible_evaluation.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        single.save_lower_only_checkpoint(out/'lower_policy_checkpoint.pt',[agent],[cfg],'sac')
        print(json.dumps(dict(profile=args.profile,seed=args.seed,**row)),flush=True)
    status = dict(status='running')
    (out/'run_status.json').write_text(json.dumps(status),encoding='utf-8')
    try:
        assess()
        env.reset(87.)
        for step in range(1,args.steps+1):
            state = agent.observation_from_environment(env)
            action, force = agent.select_action_and_force(env)
            proposal = agent.last_action_ratio
            _, reward, done, info = env.step(action,force)
            assert abs(proposal-info['proposed_force_ratio']) < 1e-5
            agent.remember(state,proposal,reward,agent.observation_from_environment(env),done)
            agent.update()
            if done:
                episode += 1
                r = env.result()
                history.append(dict(episode=episode,transitions=step,within5=r.task_success_5s,
                    strict=r.strict_timetable_feasible,timed_out=r.timed_out,
                    elapsed=r.actual_time_s,error=None if r.timed_out else r.timetable_error_s,
                    energy=r.traction_kwh,reward=r.episode_reward,comfort=r.comfort_penalty,
                    position_ratio=(env.position_m-env.section.start_m)/env.section.length_m))
                env.reset(87.)
                agent.reset_policy_diagnostics()
            if step % args.eval_steps == 0 or step == args.steps:
                assess()
        status = dict(status='completed',transitions=step,network_updates=updates,
            completed_episodes=episode,unfinished_episode_steps=env.step_count,
            note='Budget cutoff is not a terminal transition. Last checkpoint is evaluation-only, not resumable.')
    except BaseException as exc:
        status = dict(status='failed',error=repr(exc),transitions=step)
        raise
    finally:
        (out/'run_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--profile',choices=PROFILES,default='arrival_v1')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--steps',type=int,default=40000)
    p.add_argument('--eval-steps',type=int,default=2000)
    p.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    p.add_argument('--output',type=Path,default=ROOT/'results/reward_ablation/local')
    args = p.parse_args()
    if args.steps < 1 or args.eval_steps < 1:
        p.error('step budgets must be positive')
    if args.preflight:
        preflight(args.output)
    else:
        train(args)


if __name__ == '__main__':
    main()
