"""Holdout seeds 20-29 analysis. Run from outputs/ipmnist_screening/."""
import json, math
D='holdout_anneal_f2t1000_seeds20_29/shards/'; P='rls_head_resid_identmap50_r_idet003_s005625'
S=range(20,30); T95=2.2621571627409915
L=lambda f:json.load(open(f))['per_task_accuracy']
m=lambda a,i=0,j=60:sum(a[i:j])/(j-i)
par={s:L(D+f'{P}_seed{s}.json') for s in S}; arm={s:L(D+f'{P}_anneal_f2t1000_seed{s}.json') for s in S}
print('| seed | parent s005625 | anneal_f2t1000 | paired Δ |\n|---|---|---|---|')
d=[]
for s in S:
    d.append(m(arm[s])-m(par[s])); print(f'| {s} | {m(par[s]):.6f} | {m(arm[s]):.6f} | {d[-1]:+.6f} |')
n=len(d); mu=sum(d)/n; se=math.sqrt(sum((x-mu)**2 for x in d)/(n-1)/n); k=sum(x>0 for x in d)
ok=mu>=0.002 and k>=7 and mu-T95*se>0
print(f'\nmean {mu:+.6f}; 95% CI [{mu-T95*se:+.6f}, {mu+T95*se:+.6f}]; improved {k}/10 -> {"PASSES" if ok else "FAILS"}')
print('horizon: '+'; '.join(f'{i+1}-{j} {sum(m(arm[s],i,j)-m(par[s],i,j) for s in S)/10:+.6f}' for i,j in ((0,15),(15,30),(30,45),(45,60))))
