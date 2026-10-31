"""wd_sweep_screen decisions (preregistered). Usage: analyze.py tune|tiebreak|screen|final
Paired per seed vs the anneal_f2t1000 parent, gain = mean over 60 tasks of (arm - parent).
tune:     seed 30 over wd {0.005, 0.0025, 0.001, 0}. No gain > 0 -> STOP.
          Values within TIE=0.0005 of the best seed-30 gain are tied. One -> FROZEN; more -> TIEBREAK.
tiebreak: tied values on seeds 30-39 (parent and wd0005 31-39 read from the stored 11a5a7ff shards);
          highest mean gain wins; exact equality -> larger weight_decay.
screen:   frozen value on seeds 40-42: mean gain > 0 -> PASS, else STOP.
final:    seeds 40-49 (10): PASS iff mean >= +0.002 AND >= 7/10 improved AND CI95 lower bound > 0 (t=2.262, df=9)."""
import json, os, subprocess, sys, statistics as st
R='outputs/ipmnist_screening/wd_sweep_screen/'; P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000'
STORED='11a5a7ff:outputs/ipmnist_screening/decayonly_fresh_screen/fresh/'
WD={'0005':0.005,'00025':0.0025,'0001':0.001,'0':0.0}; TIE=0.0005
def cfg(t): return P if t is None else f'{P}_wd{t}'
def acc(t,s):
    f=f'{cfg(t)}_seed{s}.json'
    for sub in ('tune','tiebreak','screen','confirm','envcheck'):
        if os.path.exists(R+sub+'/'+f): a=json.load(open(R+sub+'/'+f))['per_task_accuracy']; break
    else: a=json.loads(subprocess.check_output(['git','show',STORED+f]))['per_task_accuracy']
    assert len(a)==60; return a
def gain(t,s): return sum(x-y for x,y in zip(acc(t,s),acc(None,s)))/60
def w(name,txt): open(R+name,'w').write(txt+'\n'); print(txt)
def frozen(): return open(R+'FROZEN_WD').read().split()[1]
m=sys.argv[1]
if m=='tune':
    g={t:gain(t,30) for t in WD}
    for t,v in g.items(): print(f'seed30 wd={WD[t]} gain {v:+.6f}')
    b=max(g.values())
    if b<=0: w('TUNE_DECISION','STOP no value beats the parent on seed 30'); sys.exit(1)
    tied=[t for t in WD if g[t]>=b-TIE]
    if len(tied)==1: w('TUNE_DECISION',f'FROZEN {tied[0]}'); w('FROZEN_WD',f'FROZEN {tied[0]} {WD[tied[0]]}')
    else: w('TUNE_DECISION','TIEBREAK '+' '.join(tied))
elif m=='tiebreak':
    tied=open(R+'TUNE_DECISION').read().split()[1:]
    mg={t:st.mean(gain(t,s) for s in range(30,40)) for t in tied}
    for t,v in mg.items(): print(f'seeds30-39 wd={WD[t]} mean gain {v:+.6f}')
    best=max(tied,key=lambda t:(mg[t],WD[t])); w('FROZEN_WD',f'FROZEN {best} {WD[best]}')
elif m=='screen':
    t=frozen(); g=[gain(t,s) for s in (40,41,42)]; print('screen', ' '.join(f'{x:+.6f}' for x in g))
    w('SCREEN_DECISION', ('PASS' if st.mean(g)>0 else 'STOP')+f' mean {st.mean(g):+.6f}')
    sys.exit(0 if st.mean(g)>0 else 1)
elif m=='final':
    t=frozen(); S=range(40,50)
    d=[[x-y for x,y in zip(acc(t,s),acc(None,s))] for s in S]; g=[sum(v)/60 for v in d]
    mu=st.mean(g); se=st.stdev(g)/10**.5; lo,hi=mu-2.262*se,mu+2.262*se; n=sum(x>0 for x in g)
    print(f'frozen wd={WD[t]}'); print('per seed: '+' '.join(f's{s} {x:+.6f}' for s,x in zip(S,g)))
    print('blocks (15 tasks): '+' / '.join(f'{st.mean(sum(v[k:k+15])/15 for v in d):+.6f}' for k in range(0,60,15)))
    print(f'mean {mu:+.6f} SE {se:.6f} CI95 [{lo:+.6f}, {hi:+.6f}] improved {n}/10')
    ok=mu>=0.002 and n>=7 and lo>0
    print(f'BARS: mean>=+0.002 {mu>=0.002}, >=7/10 {n>=7}, CI_lo>0 {lo>0} -> {"PASS" if ok else "FAIL"}')
