"""Final verdict per PREREGISTRATION.md: seeds 50-69, 200 tasks, wd0 minus parent."""
import json, statistics as st
D='outputs/ipmnist_screening/wd0_confirm200/shards/'; P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000'
def a(c,s):
    v=json.load(open(f'{D}{c}_seed{s}.json'))['per_task_accuracy']; assert len(v)==200; return v
S=range(50,70); d=[[x-y for x,y in zip(a(P+'_wd0',s),a(P,s))] for s in S]; g=[sum(v)/200 for v in d]
mu=st.mean(g); se=st.stdev(g)/20**.5; lo,hi=mu-2.093*se,mu+2.093*se; n=sum(x>0 for x in g)
for s,x in zip(S,g): print(f'seed {s} {x:+.6f}')
print('blocks (50 tasks): '+' / '.join(f'{st.mean(sum(v[k:k+50])/50 for v in d):+.6f}' for k in range(0,200,50)))
print(f'mean {mu:+.6f} SE {se:.6f} CI95 [{lo:+.6f}, {hi:+.6f}] improved {n}/20')
ok=mu>=0.002 and n>=14 and lo>0
print(f'BARS: mean>=+0.002 {mu>=0.002}, >=14/20 {n>=14}, CI_lo>0 {lo>0} -> {"PASS" if ok else "FAIL"}')
