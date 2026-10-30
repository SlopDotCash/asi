"""Anneal sweep analysis. Run from outputs/ipmnist_screening/."""
import json, math
W='wave_identmap50_r_idet003_s005625_anneal_sweep/shards/'; R='wave_identmap50_r_idet003_s005625_anneal/shards/'
B='rls_head_resid_identmap50_r_idet003_s005625_anneal_'
S=range(10); T95,TB=2.2621571627409915,2.685011
L=lambda f:json.load(open(f))['per_task_accuracy']
m=lambda a,i=0,j=60:sum(a[i:j])/(j-i)
par={s:L(R+f'{B}f2t1000_seed{s}.json') for s in S}
arm={t:{s:L(W+f'{B}{t}_seed{s}.json') for s in S} for t in ('f2t2000','f4t1000')}
def stats(d):
    n=len(d); mu=sum(d)/n; se=math.sqrt(sum((x-mu)**2 for x in d)/(n-1)/n)
    return mu,mu-T95*se,mu+T95*se,mu-TB*se,mu+TB*se,sum(x>0 for x in d)
for t,A in arm.items():
    print(f'\n### {t} vs parent f2t1000\n| seed | parent | {t} | paired Δ |\n|---|---|---|---|')
    d=[]
    for s in S:
        d.append(m(A[s])-m(par[s])); print(f'| {s} | {m(par[s]):.6f} | {m(A[s]):.6f} | {d[-1]:+.6f} |')
    mu,l,h,lb,hb,k=stats(d)
    ok=mu>=0.002 and k>=7 and l>0 and lb>0
    print(f'\nmean {mu:+.6f}; 95% CI [{l:+.6f}, {h:+.6f}]; Bonferroni 97.5% CI [{lb:+.6f}, {hb:+.6f}]; improved {k}/10 -> {"CLEARS" if ok else "FAILS"}')
    print('horizon: '+'; '.join(f'{i+1}-{j} {sum(m(A[s],i,j)-m(par[s],i,j) for s in S)/10:+.6f}' for i,j in ((0,15),(15,30),(30,45),(45,60))))
d=[m(arm['f4t1000'][s])-m(arm['f2t2000'][s]) for s in S]; mu,l,h,_,_,k=stats(d)
print(f'\n### Secondary f4t1000 − f2t2000: mean {mu:+.6f}; 95% CI [{l:+.6f}, {h:+.6f}]; f4t1000 better on {k}/10')
print('horizon: '+'; '.join(f'{i+1}-{j} {sum(m(arm["f4t1000"][s],i,j)-m(arm["f2t2000"][s],i,j) for s in S)/10:+.6f}' for i,j in ((0,15),(15,30),(30,45),(45,60))))
