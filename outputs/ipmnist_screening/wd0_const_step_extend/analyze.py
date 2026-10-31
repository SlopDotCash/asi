"""Tune gate and screen verdict per PREREGISTRATION.md (60 tasks, paired vs the constant-step-0.01 incumbent).
usage: analyze.py tune   -> prints the decision, writes TUNE_DECISION (cell name or STOP), exit 0
       analyze.py screen -> per-seed gains, 10-task blocks, mean/SE/CI (t=2.262), bars, edge flag
Divergence rule: a candidate shard that is missing (crashed twice) or holds any non-finite or
out-of-[0,1] accuracy is a loss: ineligible at the tune gate, and gain -1.0 (not improved) in the
screen. Any task with accuracy < 0.5 is printed as a divergence warning."""
import json, math, os, statistics as st, sys
D='outputs/ipmnist_screening/wd0_const_step_extend/'; B='rls_head_resid_identmap50_r_idet003_s'
INC=B+'01_anneal_f1t1000_wd0'
CELLS=[(B+'0125_anneal_f1t1000_wd0',0.0125),(B+'015_anneal_f1t1000_wd0',0.015),(B+'02_anneal_f1t1000_wd0',0.02)]
EDGE=B+'02_anneal_f1t1000_wd0'
def acc(sub,c,s):
    p=f'{D}{sub}/{c}_seed{s}.json'
    if not os.path.exists(p): print(f'LOSS: {sub} {c} seed {s} missing (crash)'); return None
    v=json.load(open(p))['per_task_accuracy']; assert len(v)==60, (c,s,len(v))
    if not all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in v): print(f'LOSS: {sub} {c} seed {s} non-finite accuracy'); return None
    low=[i for i,x in enumerate(v) if x<0.5]
    if low: print(f'WARN divergence: {sub} {c} seed {s} tasks with acc<0.5: {low}')
    return v
def diff(sub,c,s):
    a,b=acc(sub,c,s),acc(sub,INC,s); assert b is not None, 'incumbent shard invalid -> VOID'
    return None if a is None else [x-y for x,y in zip(a,b)]
blk=lambda d:' '.join(f'{st.mean(d[k:k+10]):+.6f}' for k in range(0,60,10))
if sys.argv[1]=='tune':
    g=[]
    for c,f in CELLS:
        d=diff('tune',c,112)
        if d is None: print(f'tune seed 112 {c}: LOSS (ineligible)'); continue
        print(f'tune seed 112 {c}: {st.mean(d):+.6f}  blocks(10t): {blk(d)}'); g.append((st.mean(d),c,f))
    best=max([x for x,_,_ in g],default=-1.0); pick=sorted((f,c) for x,c,f in g if best-x<=1e-9)[0][1] if g else None
    dec=pick if best>0.0005 else 'STOP'
    print(f'best {best:+.6f} -> {"winner "+pick if dec!="STOP" else "STOP: no step beats 0.01 by > +0.0005 (ties to the smaller step)"}')
    open(D+'TUNE_DECISION','w').write(dec+'\n')
else:
    c=open(D+'TUNE_DECISION').read().strip(); S=range(113,123)
    ds={s:diff('screen',c,s) for s in S}; g=[-1.0 if ds[s] is None else st.mean(ds[s]) for s in S]
    for s,x in zip(S,g): print(f'seed {s} {x:+.6f}'+(' (LOSS: crash/non-finite)' if ds[s] is None else ''))
    ok_s=[s for s in S if ds[s] is not None]
    b=[st.mean(st.mean(ds[s][k:k+10]) for s in ok_s) for k in range(0,60,10)] if ok_s else []
    mu=st.mean(g); se=st.stdev(g)/10**.5; lo,hi=mu-2.262*se,mu+2.262*se; n=sum(x>0 for x in g)
    print(f'{c} vs {INC}'); print(f'blocks (10 tasks, mean over {len(ok_s)} valid seeds): '+' / '.join(f'{x:+.6f}' for x in b))
    print(f'mean {mu:+.6f} SE {se:.6f} CI95 [{lo:+.6f}, {hi:+.6f}] improved {n}/10')
    ok=mu>=0.002 and n>=7 and lo>0
    print(f'BARS: mean>=+0.002 {mu>=0.002}, >=7/10 {n>=7}, CI_lo>0 {lo>0} -> {"PASS" if ok else "FAIL"}')
    if c==EDGE: print('EDGE FLAG: the winner is the largest step tested (0.02); the optimum may be higher. Verdict unchanged.')
