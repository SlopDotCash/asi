"""Stage gates and confirmation bars vs parent prefix [:60]. Usage: analyze.py tune|screen|confirm [cfg]
tune: prints CHOSEN <cfg> (best seed-0 gain, must be >0) or STOP. screen: seeds 1-3 mean>0 -> GO else STOP.
confirm: seeds 4-13 bars: mean>=+0.002, >=7/10 improved, 95% t CI lo>0 (t=2.262). Prints every seed."""
import json, sys, statistics as st
D='outputs/ipmnist_screening/anneal_peak_screen/'; P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000'
def acc(path, n=60):
    a=json.load(open(path))['per_task_accuracy'][:n]; assert len(a)==n, path; return sum(a)/n
def gain(cfg, sub, s): return acc(f'{D}{sub}/{cfg}_seed{s}.json')-acc(f'{D}parent_reused/{P}_seed{s}.json')
stage=sys.argv[1]
if stage=='tune':
    g={t: gain(f'{P}_{t}','tune',0) for t in ('p75','p10','p15')}
    for t,v in g.items(): print(f'seed0 {t} {v:+.6f}')
    best=max(g,key=g.get); print(f'CHOSEN {P}_{best}' if g[best]>0 else 'STOP no peak beats parent on seed 0'); sys.exit(0 if g[best]>0 else 1)
cfg=sys.argv[2]
if stage=='screen':
    g=[gain(cfg,'screen',s) for s in (1,2,3)]
    for s,v in zip((1,2,3),g): print(f'seed{s} {v:+.6f}')
    m=st.mean(g); print(f'screen mean {m:+.6f}', 'GO' if m>0 else 'STOP'); sys.exit(0 if m>0 else 1)
seeds=range(4,14); g=[gain(cfg,'confirm',s) for s in seeds]
for s,v in zip(seeds,g): print(f'seed{s} {v:+.6f}')
m=st.mean(g); se=st.stdev(g)/len(g)**.5; lo,hi=m-2.262*se,m+2.262*se; k=sum(v>0 for v in g)
print(f'mean {m:+.6f} SE {se:.6f} CI95 [{lo:+.6f}, {hi:+.6f}] improved {k}/10')
ok=m>=0.002 and k>=7 and lo>0; print('VERDICT', 'PASS' if ok else 'FAIL')
