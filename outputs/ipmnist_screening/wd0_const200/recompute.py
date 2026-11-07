import json,glob,numpy as np
from scipy import stats
D='outputs/ipmnist_screening/wd0_const200/shards/'
A='rls_head_resid_identmap50_r_idet003_s005625_anneal_f1t1000_wd0';P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0'
print('shards',len(glob.glob(D+'*.json')))
a=np.array([json.load(open(f'{D}{A}_seed{s}.json'))['per_task_accuracy'] for s in range(81,101)],dtype=np.float64)
p=np.array([json.load(open(f'{D}{P}_seed{s}.json'))['per_task_accuracy'] for s in range(81,101)],dtype=np.float64)
d=a-p;g=d.mean(1);t=stats.t.ppf(.975,19);se=g.std(ddof=1)/np.sqrt(20)
print(f't={t:.4f} mean {g.mean():+.6f} SD {g.std(ddof=1):.6f} SE {se:.6f} CI [{g.mean()-t*se:+.6f},{g.mean()+t*se:+.6f}] improved {(g>0).sum()}/20 min {g.min():+.6f} max {g.max():+.6f}')
print('paired t-test p', stats.ttest_rel(a.mean(1),p.mean(1)).pvalue, 'tasks>0',(d>0).sum(),'tasks<0',(d<0).sum())
for k in range(0,200,50):
  b=d[:,k:k+50].mean(1);s=b.std(ddof=1)/np.sqrt(20);print(f'block {k+1}-{k+50} {b.mean():+.6f} [{b.mean()-t*s:+.6f},{b.mean()+t*s:+.6f}] {(b>0).sum()}/20')
print('per-seed',' '.join(f'{x:+.6f}' for x in g))
