"""Re-run a saved single-scenario policy without training."""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict, replace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_single_scenario as single


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--checkpoint', default='best_feasible_policy.pt')
    args = parser.parse_args()
    config = json.loads((args.run_dir/'experiment_config.json').read_text(encoding='utf-8'))
    if config['lower_reward'] != asdict(single.LOWER_REWARD):
        raise ValueError('Saved reward settings differ from current code; use the matching source version.')
    torch.set_num_threads(1)
    model_args = single.build_model_parser().parse_args(['--simulation-dt-s',str(config['dt_s'])])
    model = single.build_bidirectional_model(model_args)
    index = 0 if config['direction'] == 'up' else model_args.up_trains
    base = next(e for e in model[3][index] if e.section.name == config['section'])
    train_config = replace(single.create_train_config_from_mass('Single scenario',config['mass_t']),
        dt_s=config['dt_s'])
    env = single.make_environment(base.section,train_config,base.direction_sign,
        base.section_index_normalized,config['gamma'],config.get('reward_profile','legacy'))
    agent = single.SACAgent(16,device_name='cpu',gamma=config['gamma'],
        initial_alpha=config['alpha'],min_alpha=config['alpha'],
        include_schedule_deviation_state=True,schedule_deviation_scale=5.0,
        include_shared_context_state=True)
    checkpoint = torch.load(args.run_dir/args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint['lower_action_semantics'] != agent.action_semantics:
        raise ValueError('Checkpoint action semantics mismatch')
    state = checkpoint['lower_controller_states'][0]
    for name in ('actor','critic1','critic2'):
        getattr(agent,name).load_state_dict(state[name])
    with torch.no_grad():
        agent.log_alpha.copy_(state['log_alpha'])
    row = single.evaluate(agent,env,config['target_time_s'],-1,
        trajectory_output_dir=args.run_dir)
    output = args.run_dir/(Path(args.checkpoint).stem+'_reevaluation.json')
    output.write_text(json.dumps(row,indent=2),encoding='utf-8')
    print(json.dumps(row,indent=2))


if __name__ == '__main__':
    main()
