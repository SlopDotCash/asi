"""Bit-for-bit env check: control seed 0 and parent seed 0 vs 3c38b025; pr50 seed 4 (the arm-under-test's parent) vs partial-reset e0612611. Exit 0 only if all match."""
import json, subprocess, sys
D='outputs/ipmnist_screening/pr50_wd_decoupled_screen/envcheck/'; P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000'
refs=[('upgd_w_control_seed0.json','3c38b025','outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/'),
      (P+'_seed0.json','3c38b025','outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/'),
      (P+'_pr50_seed4.json','e0612611','outputs/ipmnist_screening/anneal_partial_reset_screen/confirm/')]
ok=True
for f,c,w in refs:
    a=json.load(open(D+f))['per_task_accuracy']; b=json.loads(subprocess.check_output(['git','show',f'{c}:{w}{f}']))['per_task_accuracy']
    m=len(a)==len(b)==60 and all(x==y for x,y in zip(a,b)); ok&=m; print('MATCH' if m else 'MISMATCH', f, 'vs', c)
print('ENVCHECK', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
