"""Read-only checkpoint diagnosis; writes analysis artifacts, never trains."""
from pathlib import Path
import sys
import csv
import json
from dataclasses import replace

sys.path.append(r'C:/Users/DELL/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from timetable.bidirectional_model import build_bidirectional_model
from timetable.timetable_0601_bidirectional_online import build_parser
from timetable.lower_continuous import SACAgent
from timetable.lower_control import LowerSectionEnvironment
from timetable.train_model import TrainDynamics, TRAIN_CONFIGS

torch.set_num_threads(1)
root = Path(sys.argv[1]).resolve()
out = root / 'analysis'
out.mkdir(exist_ok=True)
checkpoint = torch.load(root / 'lower_policy_autosave.pt', map_location='cpu', weights_only=True)
agent = SACAgent(16, device_name='cpu', include_schedule_deviation_state=True,
    schedule_deviation_scale=5.0, include_shared_context_state=True, gamma=0.9998)
entry = checkpoint['lower_controller_states'][0]
for name in ('actor', 'critic1', 'critic2', 'target_critic1', 'target_critic2'):
    getattr(agent, name).load_state_dict(entry[name])
agent.log_alpha.data.copy_(entry['log_alpha'])
agent.set_training_mode(False)
args = build_parser().parse_args(['--simulation-dt-s','0.2'])
model = build_bidirectional_model(args)
results = []
for train_index in (0, 6):
    for section_index, base in enumerate(model[3][train_index]):
        for template in (TRAIN_CONFIGS[0], TRAIN_CONFIGS[-1]):
            env = LowerSectionEnvironment(base.section, TrainDynamics(replace(template, dt_s=0.2)),
                arrival_tolerance_s=0.5, direction_sign=base.direction_sign,
                section_index_normalized=base.section_index_normalized, schedule_shaping_gamma=0.9998)
            target = float(model[4][train_index, section_index])
            env.reset(target)
            states, acts, qs, entropy, rewards = [], [], [], [], []
            while not env.done:
                state = agent.observation_from_environment(env)
                tensor = torch.tensor(state).unsqueeze(0)
                with torch.no_grad():
                    a = agent.actor.deterministic_action(tensor)
                    q = torch.minimum(agent.critic1(tensor,a),agent.critic2(tensor,a)).item()
                action, force = agent.select_action_and_force(env)
                _, reward, _, info = env.step(action, force)
                states.append(state.tolist()); acts.append(agent.last_action_ratio); qs.append(q); rewards.append(reward)
            r = env.result()
            discounted = sum((agent.gamma**i)*v for i,v in enumerate(rewards))
            results.append(dict(direction='up' if train_index==0 else 'down', section=base.section.name,
                mass_t=template.mass_kg/1000, target_s=target, elapsed_s=r.actual_time_s,
                timed_out=r.timed_out, strict=r.strict_timetable_feasible,
                within5=not r.timed_out and abs(r.actual_time_s-target)<=5 and r.arrival_speed_mps<=2 and r.max_speed_limit_violation_kmh<=1e-6,
                stop_position_ratio=(env.position_m-base.section.start_m)/base.section.length_m,
                arrival_speed=r.arrival_speed_mps, reward=r.episode_reward,
                discounted_return=discounted, initial_q=qs[0], q_mean=float(np.mean(qs)),
                mean_proposal=float(np.mean(acts)), min_proposal=min(acts),max_proposal=max(acts),
                terminal_shields=env.terminal_action_shield_overrides,
                failure_reason='premature_stop' if info['premature_stop'] else 'deadline' if r.timed_out else 'arrival',
                max_speed_violation=r.max_speed_limit_violation_kmh))
            if train_index==0 and section_index==0 and template==TRAIN_CONFIGS[0]:
                (out/'representative_trajectory.json').write_text(json.dumps(dict(states=states,proposals=acts,q=qs,rewards=rewards)),encoding='utf-8')
with (out/'deterministic_evaluation_76.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader();w.writerows(results)
summary = dict(n=len(results),strict=sum(r['strict'] for r in results),within5=sum(r['within5'] for r in results),
    failure_reasons={k:sum(r['failure_reason']==k for r in results) for k in ('premature_stop','deadline','arrival')},
    q_initial_range=[min(r['initial_q'] for r in results),max(r['initial_q'] for r in results)],
    discounted_return_range=[min(r['discounted_return'] for r in results),max(r['discounted_return'] for r in results)],
    alpha=agent.alpha_value,max_speed_violation=max(r['max_speed_violation'] for r in results))
(out/'checkpoint_diagnosis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print('Representative results:',json.dumps(results[:4],indent=2))
