"""Final verdict per PREREGISTRATION.md: seeds 81-100, 200 tasks, f1t1000_wd0 minus anneal_f2t1000_wd0."""
import json, statistics as st
D = 'outputs/ipmnist_screening/wd0_const200/shards/'
A = 'rls_head_resid_identmap50_r_idet003_s005625_anneal_f1t1000_wd0'
P = 'rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000_wd0'
T = 2.093
def acc(c, s):
    v = json.load(open(f'{D}{c}_seed{s}.json'))['per_task_accuracy']; assert len(v) == 200, (c, s, len(v)); return v
def ci(g):
    mu = st.mean(g); sd = st.stdev(g); se = sd / len(g) ** .5; return mu, sd, se, mu - T * se, mu + T * se
S = range(81, 101); arm = {s: acc(A, s) for s in S}; par = {s: acc(P, s) for s in S}
d = {s: [x - y for x, y in zip(arm[s], par[s])] for s in S}; g = [sum(d[s]) / 200 for s in S]
for s, x in zip(S, g): print(f'seed {s} arm {sum(arm[s])/200:.6f} parent {sum(par[s])/200:.6f} delta {x:+.6f}')
mu, sd, se, lo, hi = ci(g); n = sum(x > 0 for x in g)
print(f'mean acc arm {st.mean(sum(arm[s])/200 for s in S):.6f} parent {st.mean(sum(par[s])/200 for s in S):.6f}')
print(f'per-task diffs > 0: {sum(x > 0 for s in S for x in d[s])}/{20*200}')
print(f'mean {mu:+.6f} SD {sd:.6f} SE {se:.6f} CI95 [{lo:+.6f}, {hi:+.6f}] (t={T}) improved {n}/20')
late = None
for k in range(0, 200, 50):
    b = [sum(d[s][k:k+50]) / 50 for s in S]; m, _, _, bl, bh = ci(b)
    print(f'block {k+1}-{k+50}: mean {m:+.6f} CI95 [{bl:+.6f}, {bh:+.6f}] improved {sum(x > 0 for x in b)}/20 (descriptive)'); late = m
if late <= 0: print('FLAG: late-block (151-200) mean <= 0: gain gone or reversed late in life (does not change the verdict)')
else: print('late-block flag: not raised (151-200 mean > 0)')
ok = mu >= 0.002 and n >= 14 and lo > 0
print(f'BARS: mean>=+0.002 {mu>=0.002}, >=14/20 {n>=14}, CI_lo>0 {lo>0} -> {"PASS" if ok else "FAIL"}')
