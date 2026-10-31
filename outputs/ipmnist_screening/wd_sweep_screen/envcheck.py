"""Bit-for-bit env check. control seed 0 + parent seed 0 vs 3c38b025 (anneal wave);
wd0005 seed 30 vs 11a5a7ff (decayonly_fresh_screen). Exit 0 only if all match."""
import json, subprocess, sys
D='outputs/ipmnist_screening/wd_sweep_screen/envcheck/'; P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000'
W='outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/'
F='outputs/ipmnist_screening/decayonly_fresh_screen/fresh/'
refs=[('upgd_w_control_seed0.json','3c38b025',W),(P+'_seed0.json','3c38b025',W),(P+'_wd0005_seed30.json','11a5a7ff',F)]
ok=True
for f,c,w in refs:
    a=json.load(open(D+f))['per_task_accuracy']; b=json.loads(subprocess.check_output(['git','show',f'{c}:{w}{f}']))['per_task_accuracy']
    m=len(a)==len(b)==60 and all(x==y for x,y in zip(a,b)); ok&=m; print('MATCH' if m else 'MISMATCH', f, 'vs', c)
print('ENVCHECK', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
