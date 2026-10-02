"""Single-scenario SAC data audit and terminal-replay controlled experiment."""
from __future__ import annotations

import csv
from collections import deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

import run_single_scenario as single
from timetable.lower_continuous import ContinuousReplayBuffer, SACAgent, _train_lower_continuous_agents
from timetable.lower_control import LowerSectionEnvironment
from timetable.visualization import write_csv


class DiagnosticReplay(ContinuousReplayBuffer):
    def __init__(self, capacity, terminal_fraction):
        super().__init__(capacity)
        self.terminal_fraction = terminal_fraction
        self.terminals, self.ordinary = deque(), deque()

    def push(self, *transition):
        if len(self.buffer) == self.buffer.maxlen:
            old = self.buffer[0]
            group = self.terminals if old[-1] else self.ordinary
            assert group.popleft() is old
        super().push(*transition)
        (self.terminals if transition[-1] else self.ordinary).append(self.buffer[-1])

    def sample(self, batch_size):
        if self.terminal_fraction == 0:
            return super().sample(batch_size)
        terminal_count = min(len(self.terminals), round(batch_size*self.terminal_fraction))
        terminal_count = max(terminal_count, batch_size-len(self.ordinary))
        selected = random.sample(self.terminals, terminal_count)
        selected += random.sample(self.ordinary, batch_size-terminal_count)
        random.shuffle(selected)
        state, action, reward, next_state, done = zip(*selected)
        return (torch.tensor(np.asarray(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32).unsqueeze(1),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.asarray(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1))


class AuditedEnvironment(LowerSectionEnvironment):
    def step(self, *args, **kwargs):
        self.before_observation = self.compact_observation(True, 5.0, True).copy()
        output = super().step(*args, **kwargs)
        self.last_output = output
        return output


class DiagnosticAgent(SACAgent):
    def remember(self, state, action, reward, next_state, done):
        env = self.audit_environment
        _, expected_reward, expected_done, info = env.last_output
        assert np.allclose(state, env.before_observation), 'replay state differs from executed state'
        assert np.allclose(next_state, self.observation_from_environment(env)), 'next state mismatch'
        assert abs(action-info['proposed_force_ratio']) < 1e-5, 'replay does not contain raw proposal'
        assert abs(reward-expected_reward) < 1e-7 and bool(done) == bool(expected_done) == env.done
        assert len(state) == 16 and np.isfinite(state).all() and np.isfinite(next_state).all()
        assert np.isfinite(reward) and np.isfinite(action) and -1.000001 <= action <= 1.000001
        self.audited_transitions += 1
        self.training_episode += int(bool(done))
        for fraction in (0.0, 0.25, 0.5, 0.75):
            if fraction not in self.probe_states and state[0] >= fraction:
                self.probe_states[fraction] = np.asarray(state).copy()
        if self.audited_transitions <= 3 or done:
            row = dict(transition=self.audited_transitions, episode=self.training_episode + int(not done),
                state=np.asarray(state).tolist(), raw_action=float(action),
                executed_action=info['force_ratio'], reward=float(reward),
                next_state=np.asarray(next_state).tolist(), done=bool(done),
                reason='premature_stop' if info['premature_stop'] else 'deadline' if env.timed_out
                    else 'arrival' if done else 'ordinary')
            self.audit_file.write(json.dumps(row)+'\n'); self.audit_file.flush()
        super().remember(state, action, reward, next_state, done)


def append_csv(path, row):
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main(argv=None):
    parser = single.build_parser()
    parser.description = '固定单情景SAC诊断：检查实际转移、Bellman目标和critic误差'
    parser.set_defaults(output_dir=single.PROJECT_DIR/'results'/'sac_diagnostics')
    parser.add_argument('--terminal-fraction', type=float, default=0.0,
        help='0为原始均匀回放；0.25为终止样本占批次25%%，不足时不重复采样')
    parser.add_argument('--diagnostic-interval', type=int, default=1000, help='每多少次网络更新输出批次诊断')
    args = parser.parse_args(argv)
    if not 0 <= args.terminal_fraction <= 1 or args.diagnostic_interval < 1:
        parser.error('terminal fraction must be 0—1; diagnostic interval must be positive')
    # Reuse the production single-scenario validation, including section lookup.
    checked = [*(sys.argv[1:] if argv is None else argv)]
    for flag in ('--terminal-fraction', '--diagnostic-interval'):
        for index in range(len(checked)-1, -1, -1):
            if checked[index] == flag:
                del checked[index:index+2]
            elif checked[index].startswith(flag+'='):
                del checked[index]
    if '--dry-run' not in checked:
        checked.append('--dry-run')
    single.main(checked)
    if args.dry_run or args.list_sections:
        return
    if args.replay_capacity < 128:
        parser.error('replay capacity must be at least batch size 128')
    single.seed_everything(args.seed)
    torch.set_num_threads(1)
    model_args = single.build_model_parser().parse_args(['--simulation-dt-s',str(args.dt_s)])
    model = single.build_bidirectional_model(model_args)
    train_index = 0 if args.direction == 'up' else model_args.up_trains
    section_index = next(i for i,e in enumerate(model[3][train_index]) if e.section.name == args.section)
    base = model[3][train_index][section_index]
    target = args.target_time_s if args.target_time_s is not None else float(model[4][train_index,section_index])
    config = single.replace(single.create_train_config_from_mass('Single scenario',args.mass_t),dt_s=args.dt_s)
    def new_env(audited=False):
        cls = AuditedEnvironment if audited else LowerSectionEnvironment
        return cls(base.section,single.TrainDynamics(config),arrival_tolerance_s=0.5,
            direction_sign=base.direction_sign, section_index_normalized=base.section_index_normalized,
            schedule_shaping_gamma=args.gamma, reward_profile=args.reward_profile)
    env = new_env(True)
    agent = DiagnosticAgent(16,device_name=args.device,gamma=args.gamma,
        initial_alpha=args.alpha,min_alpha=args.alpha,replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        include_schedule_deviation_state=True,schedule_deviation_scale=5.0,include_shared_context_state=True)
    agent.replay = DiagnosticReplay(args.replay_capacity,args.terminal_fraction)
    agent.audit_environment = env
    agent.audited_transitions = agent.training_episode = 0
    agent.probe_states = {}
    label = 'uniform' if args.terminal_fraction == 0 else f'terminal_{args.terminal_fraction:g}'
    out = args.output_dir.resolve()/label/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+f'_seed{args.seed}')
    out.mkdir(parents=True)
    agent.audit_file = (out/'transition_audit.jsonl').open('w',encoding='utf-8')
    resolved = vars(args).copy()
    resolved.update(output_dir=str(out), target_time_s=target,alpha_auto_tune_enabled=False,
        train_parameters=single.asdict(config),action_semantics=agent.action_semantics,
        lower_reward=single.asdict(single.LOWER_REWARD),
        source_sha256={str(p.relative_to(single.PROJECT_DIR)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(),single.PROJECT_DIR/'run_single_scenario.py',
                *sorted((single.PROJECT_DIR/'timetable').glob('*.py'))]})
    (out/'experiment_config.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2),encoding='utf-8')
    updates = 0
    def diagnostics(batch):
        nonlocal updates
        updates += 1
        for key in ('state','action','reward','next_state','done','q1','q2','target'):
            assert torch.isfinite(batch[key]).all(), f'nonfinite {key}'
        mask = batch['done'].squeeze(1).bool()
        assert torch.equal(batch['target'][mask],batch['reward'][mask]), 'terminal target is not reward'
        assert ((batch['done']==0)|(batch['done']==1)).all(), 'invalid done'
        if updates != 1 and updates % args.diagnostic_interval:
            return
        row = dict(update=updates,episode=agent.training_episode,
            replay_size=len(agent.replay),terminal_pool=len(agent.replay.terminals),
            terminal_pool_fraction=len(agent.replay.terminals)/len(agent.replay),
            batch_terminal_count=int(mask.sum().item()),alpha=agent.alpha_value,
            critic_loss=agent.last_loss,actor_loss=agent.last_actor_loss)
        for group, select in (('ordinary',~mask),('terminal',mask)):
            row[group+'_n'] = int(select.sum().item())
            for key in ('reward','q1','q2','target'):
                values = batch[key][select]
                row[group+'_'+key+'_mean'] = values.mean().item() if values.numel() else None
                row[group+'_'+key+'_min'] = values.min().item() if values.numel() else None
                row[group+'_'+key+'_max'] = values.max().item() if values.numel() else None
            for key in ('q1','q2'):
                errors = (batch['target']-batch[key])[select]
                row[group+'_'+key+'_td_mean'] = errors.mean().item() if errors.numel() else None
                row[group+'_'+key+'_td_abs_mean'] = errors.abs().mean().item() if errors.numel() else None
        for key in ('critic1_grad_norm','critic2_grad_norm','actor_grad_norm'):
            row[key+'_before_clip'] = batch[key].item()
        append_csv(out/'update_diagnostics.csv',row)
    agent.update_diagnostics_callback = diagnostics
    evaluations, scans = [], []
    best_rank = None
    def assess(episode):
        nonlocal best_rank
        row = single.evaluate(agent,new_env(),target,episode)
        rank = (bool(row['strict']), -abs(row['arrival_error_s'])
            if row['arrival_error_s'] is not None else -float('inf'))
        if row['within5s'] and (best_rank is None or rank > best_rank):
            best_rank = rank
            single.save_lower_only_checkpoint(out/'best_feasible_policy.pt',[agent],[config],'sac')
            (out/'best_feasible_evaluation.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        row.update(network_updates=updates,audited_transitions=agent.audited_transitions)
        # All stored terminal transitions have exact Bellman targets r, without sampling RNG.
        terminal = list(agent.replay.terminals)
        if terminal:
            states = torch.tensor(np.asarray([t[0] for t in terminal]),dtype=torch.float32,device=agent.device)
            actions = torch.tensor([t[1] for t in terminal],dtype=torch.float32,device=agent.device).unsqueeze(1)
            rewards = torch.tensor([t[2] for t in terminal],dtype=torch.float32,device=agent.device).unsqueeze(1)
            with torch.no_grad():
                for name in ('critic1','critic2'):
                    td = rewards-getattr(agent,name)(states,actions)
                    row[name+'_terminal_pool_td_abs_mean'] = td.abs().mean().item()
                    row[name+'_terminal_pool_td_mean'] = td.mean().item()
        evaluations.append(row)
        write_csv(out/'deterministic_evaluation.csv',evaluations)
        with torch.no_grad():
            for progress,state in agent.probe_states.items():
                actions = torch.linspace(-1,1,21,device=agent.device).unsqueeze(1)
                states = torch.tensor(state,dtype=torch.float32,device=agent.device).repeat(21,1)
                q1,q2 = agent.critic1(states,actions).flatten().tolist(),agent.critic2(states,actions).flatten().tolist()
                for action,v1,v2 in zip(actions.flatten().tolist(),q1,q2):
                    scans.append(dict(episode=episode,probe_progress=progress,
                        state=json.dumps(state.tolist()),proposed_action=action,q1=v1,q2=v2))
        write_csv(out/'fixed_state_action_scan.csv',scans)
        print(f"EVAL {episode}: strict={row['strict']} within5={row['within5s']} "
            f"reason={row['failure_reason']} position={row['position_ratio']:.3f} "
            f"Q={row['initial_q']:.2f} return={row['discounted_return']:.2f} "
            f"terminal_TD={row['terminal_td_error']:.2f}",flush=True)
    def autosave(phase,episode,history):
        single.save_lower_only_checkpoint(out/'lower_policy_autosave.pt',[agent],[config],'sac')
        write_csv(out/'lower_training_autosave_history.csv',history)
        assess(episode)
    status = {'status':'running'}
    (out/'run_status.json').write_text(json.dumps(status),encoding='utf-8')
    print(f'Diagnostic results: {out}',flush=True)
    try:
        assess(0)
        history = _train_lower_continuous_agents([env],agent,np.asarray([target]),args.episodes,0,
            'Single scenario','sac',log_interval=args.log_interval,plot_interval=args.plot_interval,
            output_dir=out,environment_tags=[{'direction':args.direction,'train_name':'Single scenario'}],
            alpha_auto_tune_start_episode=args.episodes,checkpoint_interval=args.eval_interval,
            checkpoint_callback=autosave)
        write_csv(out/'lower_training_history.csv',history)
        single.save_lower_only_checkpoint(out/'lower_policy_checkpoint.pt',[agent],[config],'sac')
        if evaluations[-1]['episode'] != args.episodes:
            assess(args.episodes)
        status = dict(status='completed',completed_episodes=args.episodes,network_updates=updates,
            audited_transitions=agent.audited_transitions)
    except BaseException as error:
        status = dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise
    finally:
        agent.audit_file.close()
        (out/'run_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    return out


if __name__ == '__main__':
    main()
