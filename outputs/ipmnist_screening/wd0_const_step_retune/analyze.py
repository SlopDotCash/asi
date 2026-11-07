"""Tune gate and screen verdict per PREREGISTRATION.md (60 tasks, paired vs the constant-step-0.005625 incumbent).
usage: analyze.py tune   -> prints the decision, writes TUNE_DECISION (cell name or STOP), exit 0
       analyze.py screen -> per-seed gains, 10-task blocks, mean/SE/CI (t=2.262), bars"""
import json, math, statistics as st, sys
D='outputs/ipmnist_screening/wd0_const_step_retune/'; B='rls_head_resid_identmap50_r_idet003_s'
INC=B+'005625_anneal_f1t1000_wd0'; S0=0.005625
CELLS=[(B+'004_anneal_f1t1000_wd0',0.004),(B+'0075_anneal_f1t1000_wd0',0.0075),(B+'01_anneal_f1t1000_wd0',0.01)]
def acc(sub,c,s):
    v=json.load(open(f'{D}{sub}/{c}_seed{s}.json'))['per_task_accuracy']; assert len(v)==60, (c,s,len(v)); return v
def gain(sub,c,s,k=0,n=60): return st.mean(x-y for x,y in zip(acc(sub,c,s)[k:k+n],acc(sub,INC,s)[k:k+n]))
if sys.argv[1]=='tune':
    g=[(gain('tune',c,101),c,f) for c,f in CELLS]
    for x,c,f in g:
        print(f'tune seed 101 {c}: {x:+.6f}  blocks(10t): '+' '.join(f'{gain("tune",c,101,k,10):+.6f}' for k in range(0,60,10)))
    best=max(x for x,_,_ in g); tied=[(abs(math.log2(f/S0)),f,c) for x,c,f in g if best-x<=1e-9]
    pick=sorted(tied)[0][2]
    dec=pick if best>0.0005 else 'STOP'
    print(f'best {best:+.6f} -> {"winner "+pick if dec!="STOP" else "STOP: no step beats 0.005625 by > +0.0005 (ties to the current step)"}')
    open(D+'TUNE_DECISION','w').write(dec+'\n')
else:
    c=open(D+'TUNE_DECISION').read().strip(); S=range(102,112)
    g=[gain('screen',c,s) for s in S]
    for s,x in zip(S,g): print(f'seed {s} {x:+.6f}')
    b=[st.mean(gain('screen',c,s,k,10) for s in S) for k in range(0,60,10)]
    mu=st.mean(g); se=st.stdev(g)/10**.5; lo,hi=mu-2.262*se,mu+2.262*se; n=sum(x>0 for x in g)
    print(f'{c} vs {INC}'); print('blocks (10 tasks, seed mean): '+' / '.join(f'{x:+.6f}' for x in b))
    print(f'mean {mu:+.6f} SE {se:.6f} CI95 [{lo:+.6f}, {hi:+.6f}] improved {n}/10')
    ok=mu>=0.002 and n>=7 and lo>0
    print(f'BARS: mean>=+0.002 {mu>=0.002}, >=7/10 {n>=7}, CI_lo>0 {lo>0} -> {"PASS" if ok else "FAIL"}')
