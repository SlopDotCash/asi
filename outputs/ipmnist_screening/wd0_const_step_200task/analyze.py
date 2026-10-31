"""Confirmation verdict per DEVELOPMENT_PLAN.md (200 tasks, 20 paired seeds 134-153).
usage: analyze.py -> per-seed gains, 50-task blocks, mean/SE/CI (t=2.093), bars, divergence.
The divergence logic is adapted from our own earlier 60-task follow-up screen (unpublished) and rewritten
here self-contained: pure functions classify / screen_verdict, testable on synthetic inputs.
An incumbent shard that diverges raises VoidRun (exit 3)."""
import json, math, os, statistics as st, sys

FLOOR = 0.5          # single-task accuracy floor (justified in DEVELOPMENT_PLAN.md)
T19 = 2.093          # t(0.975, df 19)
NTASK = 200
BLOCK = 50


class VoidRun(Exception):
    pass


def classify(acc, loss, floor=FLOOR):
    """None if the shard is healthy, else the DIVERGED reason. acc None means the shard is missing."""
    if acc is None:
        return 'missing after retry (crash)'
    for name, v in (('accuracy', acc), ('loss', loss)):
        if v is None or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in v):
            return f'non-finite {name}'
    if any(not 0 <= x <= 1 for x in acc):
        return 'accuracy outside [0, 1]'
    low = [i for i, x in enumerate(acc) if x < floor]
    if low:
        return f'single-task accuracy < {floor} on tasks {low}'
    return None


def screen_verdict(gains, t=T19):
    """gains: per-seed mean paired gains, None for a diverged candidate seed (scored -1.0, not improved).
    Any diverged seed makes the verdict FAIL regardless of the bars."""
    g = [-1.0 if x is None else x for x in gains]
    k = len(g); mu = st.mean(g); se = st.stdev(g) / k ** .5; lo = mu - t * se
    n = sum(x is not None and x > 0 for x in gains); nd = sum(x is None for x in gains)
    bars = mu >= 0.002 and n >= 0.7 * k and lo > 0
    return dict(mean=mu, se=se, lo=lo, hi=mu + t * se, improved=n, diverged=nd,
                bars=bars, verdict='PASS' if bars and nd == 0 else 'FAIL')


D = 'outputs/ipmnist_screening/wd0_const_step_200task/'; B = 'rls_head_resid_identmap50_r_idet003_s'
CAND = B + '02_anneal_f1t1000_wd0'
INC = B + '005625_anneal_f2t1000_wd0'
SEEDS = range(134, 154); SUB = 'confirm'


def load(sub, c, s):
    p = f'{D}{sub}/{c}_seed{s}.json'
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p)); a = d['per_task_accuracy']; assert len(a) == NTASK, (c, s, len(a))
    return a, d.get('per_task_loss')


def diff(sub, c, s):
    a, la = load(sub, c, s); b, lb = load(sub, INC, s)
    r = classify(b, lb)
    if r:
        raise VoidRun(f'incumbent {sub} seed {s} DIVERGED ({r}) -> VOID')
    r = classify(a, la)
    if r:
        print(f'DIVERGED: {sub} {c} seed {s}: {r}'); return None
    return [x - y for x, y in zip(a, b)]


def main():
    S = list(SEEDS); ds = {s: diff(SUB, CAND, s) for s in S}
    gains = [None if ds[s] is None else st.mean(ds[s]) for s in S]
    for s, x in zip(S, gains):
        print(f'seed {s} ' + ('DIVERGED (FAIL for this seed, scored -1.0)' if x is None else f'{x:+.6f}'))
    ok_s = [s for s in S if ds[s] is not None]
    b = [st.mean(st.mean(ds[s][k:k + BLOCK]) for s in ok_s) for k in range(0, NTASK, BLOCK)] if ok_s else []
    v = screen_verdict(gains); k = len(S)
    print(f'{CAND} vs {INC}')
    print(f'blocks ({BLOCK} tasks, mean over {len(ok_s)} valid seeds): ' + ' / '.join(f'{x:+.6f}' for x in b))
    print(f"mean {v['mean']:+.6f} SE {v['se']:.6f} CI95 [{v['lo']:+.6f}, {v['hi']:+.6f}] improved {v['improved']}/{k} diverged {v['diverged']}/{k}")
    print(f"BARS: mean>=+0.002 {v['mean'] >= 0.002}, >=14/20 {v['improved'] >= 14}, CI_lo>0 {v['lo'] > 0}; no diverged seed {v['diverged'] == 0} -> {v['verdict']}")
    if b and b[-1] < 0:
        print('LATE-BLOCK FLAG: tasks 151-200 mean gain is negative (disclosure only).')


if __name__ == '__main__':
    try:
        main()
    except VoidRun as e:
        print(f'VOID: {e}'); sys.exit(3)
