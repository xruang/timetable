"""Single-scenario SAC diagnostic entry point, sharing the production controller."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from timetable.bidirectional_model import build_bidirectional_model, seed_everything
from timetable.lower_continuous import SACAgent, _train_lower_continuous_agents
from timetable.lower_control import LOWER_REWARD, LowerSectionEnvironment
from timetable.reward_experiments import REWARD_PROFILES
from timetable.train_model import TrainDynamics, create_train_config_from_mass
from timetable.timetable_0601_bidirectional_online import (
    build_parser as build_model_parser, save_lower_only_checkpoint,
)
from timetable.visualization import write_csv

PROJECT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单情景SAC：固定站间、质量、目标时间和alpha；不训练上层。")
    parser.add_argument('--direction', choices=('up', 'down'), default='up', help='运行方向')
    parser.add_argument('--section', default='HC-DGNL', help='按该方向的站间名称，例如HC-DGNL')
    parser.add_argument('--mass-t', type=float, default=255.0, help='固定总质量，单位吨，235—275')
    parser.add_argument('--target-time-s', type=float, help='固定目标秒数；不填则使用真实运营时间')
    parser.add_argument('--alpha', type=float, default=0.02, help='固定熵温度，始终关闭自动调节')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--eval-interval', type=int, default=100, help='确定性评估及自动保存间隔')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--plot-interval', type=int, default=100)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gamma', type=float, default=0.9998)
    parser.add_argument('--dt-s', type=float, default=0.5)
    parser.add_argument('--replay-capacity', type=int, default=1000000)
    parser.add_argument('--learning-starts', type=int, default=10000,
        help='采集多少条转移后才开始更新actor和critic；0恢复满batch即更新')
    parser.add_argument('--reward-profile', choices=REWARD_PROFILES, default='legacy',
        help='arrival_v1: linear potential and explicit safe arrival objective')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_DIR / 'results' / 'lower_sac_single_scenario')
    parser.add_argument('--list-sections', action='store_true', help='列出上下行站间及目标时间后退出')
    parser.add_argument('--dry-run', action='store_true', help='验证情景，不启动训练')
    return parser


def make_environment(section, config, direction_sign, section_index, gamma, reward_profile='legacy'):
    return LowerSectionEnvironment(section, TrainDynamics(config),
        speed_limit_penalty_ratio=1.0, arrival_tolerance_s=0.5,
        direction_sign=direction_sign, section_index_normalized=section_index,
        schedule_shaping_gamma=gamma, reward_profile=reward_profile)


def export_trajectory(result, output_dir, step_diagnostics=None):
    """Write the deterministic rollout and its speed/force curves."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(point) for point in result.trajectory]
    write_csv(output_dir/'selected_policy_trajectory.csv', rows)
    if not rows:
        return
    distance_km = np.asarray([row['position_m']-rows[0]['position_m'] for row in rows])/1000.0
    elapsed_s = np.asarray([row['elapsed_s'] for row in rows])
    speed_kmh = np.asarray([row['speed_kmh'] for row in rows])
    speed_limit_kmh = np.asarray([row['speed_limit_kmh'] for row in rows])
    force_kn = np.asarray([row['force_kn'] for row in rows])
    diagnostics = list(step_diagnostics or [])
    if diagnostics:
        write_csv(output_dir/'selected_policy_shield_diagnostics.csv', diagnostics)
    diagnostic_mask = [bool(row.get('terminal_action_shielded', False)) for row in diagnostics]
    if len(diagnostics) == len(rows) - 1:
        shield_mask = np.asarray([False, *diagnostic_mask], dtype=bool)
    elif len(diagnostics) == len(rows):
        shield_mask = np.asarray(diagnostic_mask, dtype=bool)
    else:
        shield_mask = np.zeros(len(rows), dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    axes[0].plot(distance_km, speed_kmh, linewidth=1.8, label='Train speed')
    axes[0].step(distance_km, speed_limit_kmh, where='post', linestyle='--',
        linewidth=1.2, color='tab:red', label='Speed limit')
    if shield_mask.any():
        axes[0].scatter(distance_km[shield_mask], speed_kmh[shield_mask], s=9,
            color='tab:orange', label='Terminal shield active', zorder=3)
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].legend()
    axes[1].plot(elapsed_s, speed_kmh, linewidth=1.8, color='tab:blue')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Speed (km/h)')
    axes[2].plot(distance_km, force_kn, linewidth=1.3, color='tab:green')
    if shield_mask.any():
        proposed_values = [float(row['raw_requested_force_n'])/1000.0 for row in diagnostics]
        if len(proposed_values) == len(rows) - 1:
            proposed_values = [np.nan, *proposed_values]
        proposed_force_kn = np.asarray(proposed_values)
        axes[2].plot(distance_km, proposed_force_kn, linewidth=0.9, alpha=0.7,
            linestyle='--', color='tab:purple', label='Policy proposal')
        axes[2].scatter(distance_km[shield_mask], force_kn[shield_mask], s=9,
            color='tab:orange', label='Shielded force', zorder=3)
    axes[2].axhline(0.0, color='black', linewidth=0.7)
    axes[2].set_xlabel('Section distance (km)')
    axes[2].set_ylabel('Force (kN)')
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f'{result.section_name}: target {result.target_time_s:.1f} s, '
        f'actual {result.actual_time_s:.3f} s, traction {result.traction_kwh:.3f} kWh')
    fig.tight_layout()
    fig.savefig(output_dir/'selected_policy_speed_curve.png', dpi=180)
    plt.close(fig)


def plot_reward_curve(history, evaluations, output_dir, moving_window=20):
    """Plot total training and deterministic evaluation rewards."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    rows = [row for row in history
            if str(row.get('phase', '')) != 'behavior_clone_pretrain']
    if not rows:
        return
    episodes = np.asarray([int(row['episode']) for row in rows], dtype=int)
    rewards = np.asarray(
        [float(row['episode_reward']) for row in rows], dtype=float)
    window = max(1, min(int(moving_window), len(rewards)))
    cumulative = np.cumsum(np.insert(rewards, 0, 0.0))
    moving = (cumulative[window:] - cumulative[:-window]) / window
    moving_episodes = episodes[window - 1:]

    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(episodes, rewards, color='tab:blue', alpha=0.18,
              linewidth=0.7, label='Training reward')
    axis.plot(moving_episodes, moving, color='tab:blue', linewidth=1.8,
              label=f'{window}-episode moving average')
    completed_evaluations = [row for row in evaluations if int(row['episode']) > 0]
    if completed_evaluations:
        eval_episodes = [int(row['episode']) for row in completed_evaluations]
        eval_rewards = [float(row['reward']) for row in completed_evaluations]
        axis.scatter(eval_episodes, eval_rewards, color='tab:orange', s=28,
                     label='Deterministic evaluation reward', zorder=3)
    axis.set_title('Single-scenario SAC reward curve')
    axis.set_xlabel('Training episode')
    axis.set_ylabel('Episode reward (higher is better)')
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'single_scenario_reward_curve.png', dpi=180)
    plt.close(fig)


def evaluate(agent, env, target, episode, trajectory_output_dir=None):
    """Evaluate the mean policy and value scale without changing training/RNG state."""
    was_training = agent.training_mode
    agent.set_training_mode(False)
    rewards, q_values, proposals, shield_deltas, step_diagnostics = [], [], [], [], []
    try:
        env.reset(target)
        while not env.done:
            state = torch.as_tensor(agent.observation_from_environment(env),
                dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action_tensor = agent.actor.deterministic_action(state)
                q_values.append(torch.minimum(agent.critic1(state, action_tensor),
                    agent.critic2(state, action_tensor)).item())
            action, force = agent.select_action_and_force(env)
            _, reward, _, info = env.step(action, force)
            rewards.append(reward)
            proposals.append(agent.last_action_ratio)
            shield_deltas.append(abs(info['terminal_intervention_force_delta_n']))
            if trajectory_output_dir is not None:
                step_diagnostics.append(dict(
                    step=env.step_count,
                    elapsed_s=env.elapsed_s,
                    position_m=env.position_m,
                    speed_kmh=env.speed_mps*3.6,
                    proposed_force_ratio=info['proposed_force_ratio'],
                    raw_requested_force_n=info['raw_requested_force_n'],
                    executed_force_n=info['force_n'],
                    terminal_action_shielded=info['terminal_action_shielded'],
                    terminal_intervention_force_delta_n=info['terminal_intervention_force_delta_n'],
                    speed_limit_action_shielded=info['speed_limit_action_shielded'],
                ))
        result = env.result()
        if trajectory_output_dir is not None:
            export_trajectory(result, trajectory_output_dir, step_diagnostics)
        return dict(episode=episode, feasible=result.feasible,
            strict=result.strict_timetable_feasible,
            within5s=result.task_success_5s, timed_out=result.timed_out,
            failure_reason='premature_stop' if info['premature_stop'] else
                'deadline' if result.timed_out else 'arrival',
            target_time_s=target, elapsed_s=result.actual_time_s,
            arrival_error_s=None if result.timed_out else result.timetable_error_s,
            position_ratio=(env.position_m-env.section.start_m)/env.section.length_m,
            reward=result.episode_reward,
            traction_kwh=result.traction_kwh,
            discounted_return=sum(agent.gamma**i*r for i, r in enumerate(rewards)),
            initial_q=q_values[0], last_q=q_values[-1], terminal_reward=rewards[-1],
            terminal_td_error=rewards[-1]-q_values[-1],
            mean_proposal=float(np.mean(proposals)), alpha=agent.alpha_value,
            terminal_shields=env.terminal_action_shield_overrides,
            control_steps=env.step_count,
            terminal_shield_ratio=env.terminal_action_shield_overrides/max(1,env.step_count),
            mean_terminal_force_correction_kn=float(np.mean(shield_deltas))/1000.0,
            max_terminal_force_correction_kn=float(np.max(shield_deltas))/1000.0,
            max_speed_violation_kmh=result.max_speed_limit_violation_kmh)
    finally:
        agent.set_training_mode(was_training)


def main(argv=None):
    args = build_parser().parse_args(argv)
    for key in ('mass_t', 'alpha', 'gamma', 'dt_s'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 235 <= args.mass_t <= 275 or not 0 < args.gamma < 1:
        raise ValueError('mass must be 235—275t; gamma must be between 0 and 1')
    if min(args.episodes, args.eval_interval, args.log_interval, args.replay_capacity) <= 0 or args.plot_interval < 0:
        raise ValueError('episodes/intervals/replay capacity must be positive; plot interval may be zero')
    if args.learning_starts < 0 or args.replay_capacity < max(128, args.learning_starts):
        raise ValueError('learning starts must be nonnegative; replay capacity must cover warm-up and batch')
    if args.target_time_s is not None and (not math.isfinite(args.target_time_s) or args.target_time_s <= 0):
        raise ValueError('target time must be finite and positive')
    seed_everything(args.seed)
    model_args = build_model_parser().parse_args(['--simulation-dt-s', str(args.dt_s)])
    model = build_bidirectional_model(model_args)
    if args.list_sections:
        for direction, index in (('up', 0), ('down', model_args.up_trains)):
            for section_index, env in enumerate(model[3][index]):
                print(f'{direction:4s} {env.section.name:12s} {float(model[4][index,section_index]):.1f}s')
        return
    train_index = 0 if args.direction == 'up' else model_args.up_trains
    candidates = model[3][train_index]
    section_index = next((i for i,e in enumerate(candidates) if e.section.name == args.section), None)
    if section_index is None:
        raise ValueError(f'Unknown {args.direction} section {args.section}; use --list-sections')
    base = candidates[section_index]
    target = args.target_time_s if args.target_time_s is not None else float(model[4][train_index,section_index])
    config = replace(create_train_config_from_mass('Single scenario',args.mass_t),dt_s=args.dt_s)
    def new_env():
        return make_environment(base.section, config, base.direction_sign,
            base.section_index_normalized, args.gamma, args.reward_profile)
    print(f'Single scenario: {args.direction} {args.section}, mass={args.mass_t}t, '
          f'target={target}s, alpha={args.alpha}(fixed), dt={args.dt_s}s')
    if args.dry_run:
        return
    torch.set_num_threads(1)
    agent = SACAgent(16, device_name=args.device, gamma=args.gamma,
        initial_alpha=args.alpha, min_alpha=args.alpha, replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        include_schedule_deviation_state=True, schedule_deviation_scale=5.0,
        include_shared_context_state=True)
    out = args.output_dir.resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S_%f')+f'_seed{args.seed}')
    out.mkdir(parents=True)
    print(f'Collection warm-up: {agent.learning_starts} transitions before network updates', flush=True)
    resolved = vars(args).copy()
    resolved.update(output_dir=str(out),target_time_s=target,alpha_auto_tune_enabled=False,
        action_semantics=agent.action_semantics,train_parameters=asdict(config),
        lower_reward=asdict(LOWER_REWARD),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted((PROJECT_DIR/'timetable').glob('*.py'))]})
    (out/'experiment_config.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2),encoding='utf-8')
    evaluations = []
    best_rank = None
    def assess(episode):
        nonlocal best_rank
        row = evaluate(agent,new_env(),target,episode)
        evaluations.append(row)
        write_csv(out/'deterministic_evaluation.csv',evaluations)
        rank = (bool(row['strict']), bool(row['within5s']),
            -abs(row['arrival_error_s']) if row['arrival_error_s'] is not None else -float('inf'),
            row['position_ratio'])
        if row['within5s'] and (best_rank is None or rank > best_rank):
            best_rank = rank
            save_lower_only_checkpoint(out/'best_feasible_policy.pt',[agent],[config],'sac')
            (out/'best_feasible_evaluation.json').write_text(
                json.dumps(row, indent=2), encoding='utf-8')
        print(f"EVAL {episode}: feasible={row['feasible']} strict={row['strict']} "
              f"within5s={row['within5s']} "
              f"reason={row['failure_reason']} arrival_error={row['arrival_error_s']} "
              f"Q={row['initial_q']:.2f} return={row['discounted_return']:.2f} "
              f"terminal_TD={row['terminal_td_error']:.2f}")
    def autosave(phase,episode,history):
        save_lower_only_checkpoint(out/'lower_policy_autosave.pt',[agent],[config],'sac')
        write_csv(out/'lower_training_autosave_history.csv',history)
        assess(episode)
        plot_reward_curve(history, evaluations, out)
    status = {'status':'running'}
    (out/'run_status.json').write_text(json.dumps(status),encoding='utf-8')
    print(f'Results: {out}')
    try:
        assess(0)
        history = _train_lower_continuous_agents([new_env()], agent,
            np.asarray([target],dtype=np.float64),args.episodes,0,'Single scenario','sac',
            random_mass_training=False,log_interval=args.log_interval,
            plot_interval=args.plot_interval,output_dir=out,
            environment_tags=[{'direction':args.direction,'train_name':'Single scenario'}],
            alpha_auto_tune_start_episode=args.episodes,
            checkpoint_interval=args.eval_interval,checkpoint_callback=autosave)
        write_csv(out/'lower_training_history.csv',history)
        save_lower_only_checkpoint(out/'lower_policy_checkpoint.pt',[agent],[config],'sac')
        if evaluations[-1]['episode'] != args.episodes:
            assess(args.episodes)
        plot_reward_curve(history, evaluations, out)
        status = {'status':'completed','completed_episodes':args.episodes}
    except BaseException as error:
        status = {'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed','error':str(error)}
        raise
    finally:
        (out/'run_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    return out


if __name__ == '__main__':
    main()
