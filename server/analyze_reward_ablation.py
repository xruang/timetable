"""Summarize completed and incomplete ablations without selecting best-only results."""
from pathlib import Path
import argparse
import csv
import json
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server.reward_ablation import write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    rows = []
    curves = []
    for path in sorted(args.root.glob('*/seed*/experiment_config.json')):
        directory = path.parent
        cfg = json.loads(path.read_text(encoding='utf-8'))
        status = json.loads((directory/'run_status.json').read_text(encoding='utf-8'))
        eval_path = directory/'deterministic_evaluation.csv'
        if not eval_path.exists():
            continue
        ev = list(csv.DictReader(eval_path.open(encoding='utf-8-sig')))
        active = [r for r in ev if int(r['transitions']) >= 10000]
        tail = [r for r in ev if int(r['transitions']) >= .8*cfg['transitions']]
        good = [r for r in active if r['within5s']=='True']
        first = int(good[0]['transitions']) if good else None
        after = [r for r in active if first is not None and int(r['transitions']) >= first]
        history_path = directory/'training_history.csv'
        hist = list(csv.DictReader(history_path.open(encoding='utf-8-sig'))) if history_path.exists() else []
        final = ev[-1]
        rows.append(dict(profile=cfg['reward_profile'],seed=cfg['seed'],status=status['status'],
            transitions=int(final['transitions']),updates=int(final['network_updates']),
            first_within5_step=first,eval_within5_count=len(good),eval_count=len(active),
            tail_eval_count=len(tail),
            tail_within5_rate=sum(r['within5s']=='True' for r in tail)/len(tail) if tail else None,
            tail_strict_rate=sum(r['strict']=='True' for r in tail)/len(tail) if tail else None,
            retention_after_first=sum(r['within5s']=='True' for r in after)/len(after) if after else None,
            final_within5=final['within5s']=='True',final_strict=final['strict']=='True',
            final_error_s=final['arrival_error_s'] or None,final_position=float(final['position_ratio']),
            final_energy=float(final['traction_kwh']),
            best_abs_error_s=min(abs(float(r['arrival_error_s'])) for r in good) if good else None,
            training_episodes=len(hist),training_within5_count=sum(r['within5']=='True' for r in hist),
            training_strict_count=sum(r['strict']=='True' for r in hist)))
        curves.append((cfg['reward_profile'],cfg['seed'],ev))
    write_rows(args.root/'summary.csv',rows)
    groups=[]
    for profile in sorted({r['profile'] for r in rows}):
        subset=[r for r in rows if r['profile']==profile and r['status']=='completed']
        if not subset:
            continue
        groups.append(dict(profile=profile,completed_seeds=len(subset),
            final_within5_seed_fraction=statistics.mean(r['final_within5'] for r in subset),
            final_strict_seed_fraction=statistics.mean(r['final_strict'] for r in subset),
            tail_within5_mean=statistics.mean(r['tail_within5_rate'] for r in subset),
            tail_within5_sd=statistics.stdev(r['tail_within5_rate'] for r in subset) if len(subset)>1 else None))
    (args.root/'summary.json').write_text(json.dumps(dict(runs=rows,groups=groups,
        note='Tail rate is over checkpoints on ONE fixed scenario, not independent scenario trials. '
             'Only completed seeds enter group summaries. Best checkpoint is not convergence evidence.'),indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        for profile,seed,ev in curves:
            xs=[int(r['transitions']) for r in ev]
            axes[0].plot(xs,[float(r['position_ratio']) for r in ev],label=f'{profile}, seed {seed}')
            axes[1].plot(xs,[float(r['arrival_error_s']) if r['arrival_error_s'] else float('nan') for r in ev],marker='.',label=f'{profile}, seed {seed}')
        axes[0].set_ylabel('Final position / section length')
        axes[0].legend(fontsize=8)
        axes[1].axhspan(-5,5,color='green',alpha=.08)
        axes[1].axhspan(-.5,.5,color='green',alpha=.18)
        axes[1].set_ylabel('Arrival time error (s); failures omitted')
        axes[1].set_xlabel('Training environment transitions')
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle('Deterministic policy evaluation; fixed HC-DGNL, 255 t, target 87 s')
        fig.tight_layout()
        fig.savefig(args.root/'evaluation_curves.png',dpi=160)


if __name__=='__main__':
    main()
