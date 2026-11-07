"""Bit-for-bit env check per PREREGISTRATION.md. Exit 0 only if both match."""
import json, subprocess, sys
D='outputs/ipmnist_screening/wd0_const_step_extend/envcheck/'
W='outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/'; S='outputs/ipmnist_screening/wd0_const_step_retune/screen/'
P='rls_head_resid_identmap50_r_idet003_s01_anneal_f1t1000_wd0'
refs=[('upgd_w_control_seed0.json','3c38b025',W),(P+'_seed102.json','f3c7d4a2',S)]
ok=True
for f,c,w in refs:
    a=json.load(open(D+f))['per_task_accuracy']; b=json.loads(subprocess.check_output(['git','show',f'{c}:{w}{f}']))['per_task_accuracy']
    m=len(a)==len(b)==60 and all(x==y for x,y in zip(a,b)); ok&=m; print('MATCH' if m else 'MISMATCH', f, 'vs', c)
print('ENVCHECK', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
