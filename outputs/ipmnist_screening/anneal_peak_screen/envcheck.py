"""Bit-for-bit environment check (control seed 0 and parent seed 0, 60 tasks vs 3c38b025). Exit 0 only if both match."""
import json, subprocess, sys
D='outputs/ipmnist_screening/anneal_peak_screen/envcheck/'; W='outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/'
ok=True
for f in ('upgd_w_control_seed0.json','rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_seed0.json'):
    a=json.load(open(D+f))['per_task_accuracy']
    b=json.loads(subprocess.check_output(['git','show','3c38b025:'+W+f]))['per_task_accuracy']
    m=len(a)==len(b) and all(x==y for x,y in zip(a,b)); ok&=m
    print('MATCH' if m else 'MISMATCH', f, len(a), 'vs 3c38b025', len(b))
print('ENVCHECK', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
