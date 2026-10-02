"""Frozen final policies: Q versus Monte Carlo soft returns on common physical states."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_single_scenario as single
from server.reward_ablation import setup, write_rows
from timetable.lower_continuous import force_from_ratio


def load(path):
    a=single.SACAgent(16,device_name='cpu',gamma=.9998,initial_alpha=.02,min_alpha=.02,
        include_schedule_deviation_state=True,schedule_deviation_scale=5.,include_shared_context_state=True)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['lower_action_semantics']==a.action_semantics
    state=saved['lower_controller_states'][0]
    for key in ('actor','critic1','critic2'):
        getattr(a,key).load_state_dict(state[key])
    with torch.no_grad():
        a.log_alpha.copy_(state['log_alpha'])
    return a


def tensor(a,e):
    return torch.as_tensor(a.observation_from_environment(e),dtype=torch.float32).unsqueeze(0)


def step(e,u):
    f=force_from_ratio(e,u)
    return e.step(e.action_for_force_n(f),f)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--samples',type=int,default=16)
    args=p.parse_args()
    if args.samples<2:
        p.error('At least two samples required for standard errors')
    torch.set_num_threads(1)
    factory,_=setup()
    rows=[]
    with torch.no_grad():
        for cfgpath in sorted(args.root.glob('*/seed*/experiment_config.json')):
            d=cfgpath.parent
            status=json.loads((d/'run_status.json').read_text())
            if status['status']!='completed':
                continue
            cfg=json.loads(cfgpath.read_text(encoding='utf-8'))
            profile=cfg['reward_profile']
            a=load(d/'lower_policy_checkpoint.pt')
            replay_eval=single.evaluate(a,factory(profile),87.,-1)
            (d/'final_reevaluation.json').write_text(json.dumps(replay_eval,indent=2),encoding='utf-8')
            e=factory(profile);e.reset(87.)
            snapshots={}
            while not e.done:
                if e.elapsed_s in (0.,10.,30.,60.,80.):
                    snapshots[e.elapsed_s]=copy.deepcopy(e)
                step(e,.75)
            assert e.result().strict_timetable_feasible
            for t,s in snapshots.items():
                x=tensor(a,s);u=a.actor.deterministic_action(x)
                q=min(a.critic1(x,u).item(),a.critic2(x,u).item())
                samples=[]
                for n in range(args.samples):
                    torch.manual_seed(22000+n)
                    env=copy.deepcopy(s)
                    _,r,_,_=step(env,u.item());total=r;k=1
                    while not env.done:
                        action,logp=a.actor.sample(tensor(a,env))
                        _,r,_,_=step(env,action.item())
                        total+=a.gamma**k*(r-a.alpha_value*logp.item());k+=1
                    samples.append((total,float(env.result().task_success_5s)))
                values=np.asarray(samples)
                rows.append(dict(profile=profile,seed=cfg['seed'],time=t,q=q,
                    soft_mc=float(values[:,0].mean()),mc_se=float(values[:,0].std(ddof=1)/np.sqrt(args.samples)),
                    q_minus_mc=float(q-values[:,0].mean()),success_rate=float(values[:,1].mean()),samples=args.samples))
            write_rows(args.root/'fixed_state_soft_value_audit.csv',rows)
            print(json.dumps(dict(profile=profile,seed=cfg['seed'],states=len(snapshots))),flush=True)


if __name__=='__main__':
    main()
