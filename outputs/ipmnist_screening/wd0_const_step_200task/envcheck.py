"""Bit-for-bit env check per DEVELOPMENT_PLAN.md. Exit 0 only if all three match."""
import json, subprocess, sys
D='outputs/ipmnist_screening/wd0_const_step_200task/envcheck/'
B='rls_head_resid_identmap50_r_idet003_s'
refs=[('upgd_w_control_seed0.json','3c38b025','outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/'),
      (B+'02_anneal_f1t1000_wd0_seed113.json','ef85fff8','outputs/ipmnist_screening/wd0_const_step_extend/screen/'),
      (B+'005625_anneal_f2t1000_wd0_seed40.json','ef85fff8','outputs/ipmnist_screening/wd0_confirm200/envcheck/')]
ok=True
for f,c,w in refs:
    a=json.load(open(D+f))['per_task_accuracy']; b=json.loads(subprocess.check_output(['git','show',f'{c}:{w}{f}']))['per_task_accuracy']
    m=len(a)==len(b)==60 and all(x==y for x,y in zip(a,b)); ok&=m; print('MATCH' if m else 'MISMATCH', f, 'vs', c)
print('ENVCHECK', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
