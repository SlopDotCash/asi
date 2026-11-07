"""f4t1000 vs anneal_f2t1000 at 200 tasks, seeds 0-19. Run from outputs/ipmnist_screening/anneal_f4t1000_confirm200/."""
import json, math
P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000'; A='rls_head_resid_identmap50_r_idet003_s005625_anneal_f4t1000'
T=2.093024; S=range(20)
L=lambda f:json.load(open(f))['per_task_accuracy']
m=lambda a,i=0,j=200:sum(a[i:j])/(j-i)
par={s:L(f'parent_reused/{P}_seed{s}.json') for s in S}; arm={s:L(f'shards/{A}_seed{s}.json') for s in S}
assert all(len(par[s])==200==len(arm[s]) for s in S)
print('| seed | parent f2t1000 | arm f4t1000 | paired Δ |\n|---|---|---|---|')
d=[]
for s in S:
    d.append(m(arm[s])-m(par[s])); print(f'| {s} | {m(par[s]):.6f} | {m(arm[s]):.6f} | {d[-1]:+.6f} |')
n=len(d); mu=sum(d)/n; se=math.sqrt(sum((x-mu)**2 for x in d)/(n-1)/n); k=sum(x>0 for x in d)
b=[mu>=0.002, k>=14, mu-T*se>0]
print(f'\nmean {mu:+.6f}; SE {se:.6f}; 95% CI [{mu-T*se:+.6f}, {mu+T*se:+.6f}]; improved {k}/20')
print(f'bars: mean>=+0.002 {b[0]}; >=14/20 {b[1]}; CI lo>0 {b[2]} -> {"PASSES" if all(b) else "FAILS"}')
print('blocks: '+'; '.join(f'{i+1}-{j} {sum(m(arm[s],i,j)-m(par[s],i,j) for s in S)/20:+.6f}' for i,j in ((0,50),(50,100),(100,150),(150,200))))
print('seeds 0-9 mean', f'{sum(d[:10])/10:+.6f}', 'seeds 10-19 mean', f'{sum(d[10:])/10:+.6f}')
