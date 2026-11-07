"""Bit-for-bit env check per PREREGISTRATION.md. Exit 0 only if all three match.
A reference that cannot be read (missing local shard, commit absent from the clone, e.g. a
shallow clone) is reported as UNREADABLE and the check FAILs (exit 1 -> launch.sh VOID)."""
import json, subprocess, sys
D='outputs/ipmnist_screening/wd0_confirm200/envcheck/'; P='rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000'
W='outputs/ipmnist_screening/wave_identmap50_r_idet003_s005625_anneal/shards/'; S='outputs/ipmnist_screening/wd_sweep_screen/screen/'
refs=[('upgd_w_control_seed0.json','3c38b025',W),(P+'_seed40.json','396d6d3c',S),(P+'_wd0_seed40.json','396d6d3c',S)]
ok=True
for f,c,w in refs:
    try:
        a=json.load(open(D+f))['per_task_accuracy']
        b=json.loads(subprocess.check_output(['git','show',f'{c}:{w}{f}'],stderr=subprocess.PIPE))['per_task_accuracy']
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as e:
        ok=False; print('UNREADABLE', f, 'vs', c, f'({type(e).__name__}; full git history with commit {c} is required)'); continue
    m=len(a)==len(b)==60 and all(x==y for x,y in zip(a,b)); ok&=m; print('MATCH' if m else 'MISMATCH', f, 'vs', c)
print('ENVCHECK', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
