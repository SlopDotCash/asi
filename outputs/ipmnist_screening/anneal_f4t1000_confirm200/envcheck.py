"""Bit-for-bit environment check. Exit 0 only if all three match. Run from the worktree root."""
import json, subprocess, sys
D='outputs/ipmnist_screening/anneal_f4t1000_confirm200/envcheck/'; S='outputs/ipmnist_screening/'
refs=[('upgd_w_control_seed0.json','3c38b025',S+'wave_identmap50_r_idet003_s005625_anneal/shards/upgd_w_control_seed0.json'),
 ('rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_seed0.json','612491db',S+'anneal_f2t1000_confirm_r1/shards/rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_seed0.json'),
 ('rls_head_resid_identmap50_r_idet003_s005625_anneal_f4t1000_seed0.json','0240ff25',S+'wave_identmap50_r_idet003_s005625_anneal_sweep/shards/rls_head_resid_identmap50_r_idet003_s005625_anneal_f4t1000_seed0.json')]
ok=True
for new,sha,path in refs:
    a=json.load(open(D+new))['per_task_accuracy']
    b=json.loads(subprocess.check_output(['git','show',f'{sha}:{path}']))['per_task_accuracy']
    m=len(a)==len(b) and all(x==y for x,y in zip(a,b)); ok&=m
    print(('MATCH' if m else 'MISMATCH'),new,len(a),'vs',sha,len(b))
print('ENVCHECK', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
