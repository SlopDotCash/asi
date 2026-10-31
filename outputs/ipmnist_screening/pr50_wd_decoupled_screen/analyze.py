"""Decomposition of the pr50 gain on seeds 4-13 (60 tasks), all paired per seed vs the #3580 parent prefix [:60].
Components: pr50 (coupled, stored), steponly (pr50 step, parent decay), decayonly (parent step, pr50 decay);
also each split arm minus pr50, and interaction = pr50 - steponly - decayonly (each vs parent). 95% t CI df=9 (t=2.262). Blocks of 15 tasks.
Preregistered rule on S = steponly gain, C = pr50 gain (both means vs parent):
  STEP-REAL (recommend f=0.5 at 200 tasks): S >= 0.5*C and CI_lo(S) > 0
  DECAY-DRIVEN (do not recommend): CI_lo(S) <= 0 or S < 0.25*C
  otherwise MIXED (no 200-task recommendation without a new prereg)."""
import json, statistics as st
D='outputs/ipmnist_screening/pr50_wd_decoupled_screen/'; P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000'
S=range(4,14)
def acc(p): a=json.load(open(p))['per_task_accuracy'][:60]; assert len(a)==60, p; return a
par={s:acc(f'{D}parent_reused/{P}_seed{s}.json') for s in S}
arm={'pr50':{s:acc(f'{D}pr50_reused/{P}_pr50_seed{s}.json') for s in S}}
for a in ('steponly','decayonly'): arm[a]={s:acc(f'{D}confirm/{P}_pr50_{a}_seed{s}.json') for s in S}
def d(x,y): return [i-j for i,j in zip(x,y)]
def summ(name, per_seed_task_diffs):
    g=[sum(v)/60 for v in per_seed_task_diffs]; m=st.mean(g); se=st.stdev(g)/10**.5
    bl=[st.mean(sum(v[k:k+15])/15 for v in per_seed_task_diffs) for k in range(0,60,15)]
    print(f'{name}: ' + ' '.join(f's{s} {x:+.6f}' for s,x in zip(S,g)))
    print(f'  mean {m:+.6f} SE {se:.6f} CI95 [{m-2.262*se:+.6f}, {m+2.262*se:+.6f}] improved {sum(x>0 for x in g)}/10 blocks ' + ' / '.join(f'{b:+.6f}' for b in bl))
    return m, m-2.262*se
r={}
for a in ('pr50','steponly','decayonly'): r[a]=summ(f'{a} vs parent', [d(arm[a][s],par[s]) for s in S])
for a in ('steponly','decayonly'): summ(f'{a} minus pr50', [d(arm[a][s],arm['pr50'][s]) for s in S])
summ('interaction (pr50 - steponly - decayonly + parent)', [[p-q-u+w for p,q,u,w in zip(arm['pr50'][s],arm['steponly'][s],arm['decayonly'][s],par[s])] for s in S])
(Sm,Slo),(C,_)=r['steponly'],r['pr50']
v='STEP-REAL' if Sm>=0.5*C and Slo>0 else ('DECAY-DRIVEN' if Slo<=0 or Sm<0.25*C else 'MIXED')
print(f'DECISION {v} (S {Sm:+.6f}, C {C:+.6f}, S/C {Sm/C:.3f}, CI_lo(S) {Slo:+.6f})')
