"""Tune gate and screen verdict per PREREGISTRATION.md (60 tasks, paired vs the wd0 parent).
usage: analyze.py tune   -> prints the decision, writes TUNE_DECISION (cell name or STOP), exit 0
       analyze.py screen -> per-seed gains, halves, mean/SE/CI (t=2.262), bars"""
import json, math, statistics as st, sys
D='outputs/ipmnist_screening/wd0_floor_retune/'; B='rls_head_resid_identmap50_r_idet003_s005625_anneal_'
PARENT=B+'f2t1000_wd0'; F0=0.0028125
CELLS=[(B+'f1t1000_wd0',0.005625),(B+'f4t1000_wd0',0.00140625),(B+'f8t1000_wd0',0.000703125)]
def acc(sub,c,s):
    v=json.load(open(f'{D}{sub}/{c}_seed{s}.json'))['per_task_accuracy']; assert len(v)==60, (c,s,len(v)); return v
def gain(sub,c,s): return st.mean(x-y for x,y in zip(acc(sub,c,s),acc(sub,PARENT,s)))
if sys.argv[1]=='tune':
    g=[(gain('tune',c,70),c,f) for c,f in CELLS]
    for x,c,f in g: print(f'tune seed 70 {c}: {x:+.6f}')
    best=max(x for x,_,_ in g); tied=[(abs(math.log2(f/F0)),-f,c) for x,c,f in g if best-x<=1e-9]
    pick=sorted(tied)[0][2]
    dec=pick if best>0.0005 else 'STOP'
    print(f'best {best:+.6f} -> {"winner "+pick if dec!="STOP" else "STOP: no cell beats the current floor by > +0.0005 (ties to f2)"}')
    open(D+'TUNE_DECISION','w').write(dec+'\n')
else:
    c=open(D+'TUNE_DECISION').read().strip(); S=range(71,81)
    g=[gain('screen',c,s) for s in S]
    for s,x in zip(S,g): print(f'seed {s} {x:+.6f}')
    h=[st.mean(st.mean(x-y for x,y in zip(acc('screen',c,s)[k:k+30],acc('screen',PARENT,s)[k:k+30])) for s in S) for k in (0,30)]
    mu=st.mean(g); se=st.stdev(g)/10**.5; lo,hi=mu-2.262*se,mu+2.262*se; n=sum(x>0 for x in g)
    print(f'{c} vs {PARENT}'); print(f'halves (30 tasks): {h[0]:+.6f} / {h[1]:+.6f}')
    print(f'mean {mu:+.6f} SE {se:.6f} CI95 [{lo:+.6f}, {hi:+.6f}] improved {n}/10')
    ok=mu>=0.002 and n>=7 and lo>0
    print(f'BARS: mean>=+0.002 {mu>=0.002}, >=7/10 {n>=7}, CI_lo>0 {lo>0} -> {"PASS" if ok else "FAIL"}')
