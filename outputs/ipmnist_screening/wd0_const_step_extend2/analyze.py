"""Tune gate and screen verdict per PREREGISTRATION.md (60 tasks, paired vs the constant-step-0.02 incumbent).
usage: analyze.py tune   -> prints the decision, writes TUNE_DECISION (cell name, STOP or STOP_ALL_DIVERGED)
       analyze.py screen -> per-seed gains, 10-task blocks, mean/SE/CI (t=2.262), bars, divergence, edge flag
The divergence rules live in the pure functions classify / tune_decision / screen_verdict so they
can be tested on synthetic inputs. An incumbent shard that diverges raises VoidRun."""
import json, math, os, statistics as st, sys

FLOOR = 0.5          # single-task accuracy floor (justified in PREREGISTRATION.md)
GATE = 0.0005        # tune gate margin over the incumbent
T9 = 2.262           # t(0.975, df 9)
NTASK = 60


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


def tune_decision(cells, gate=GATE):
    """cells: list of (name, step, gain or None if diverged). Diverged cells never win.
    Returns the winning name, 'STOP' (no eligible cell beats the incumbent by > gate) or
    'STOP_ALL_DIVERGED'. Ties (within 1e-9) go to the smaller step."""
    ok = [(g, step, n) for n, step, g in cells if g is not None]
    if not ok:
        return 'STOP_ALL_DIVERGED'
    best = max(g for g, _, _ in ok)
    if not best > gate:
        return 'STOP'
    return min((step, n) for g, step, n in ok if best - g <= 1e-9)[1]


def screen_verdict(gains, t=T9):
    """gains: per-seed mean paired gains, None for a diverged winner seed (scored -1.0, not improved).
    Any diverged seed makes the verdict FAIL regardless of the bars."""
    g = [-1.0 if x is None else x for x in gains]
    k = len(g); mu = st.mean(g); se = st.stdev(g) / k ** .5; lo = mu - t * se
    n = sum(x is not None and x > 0 for x in gains); nd = sum(x is None for x in gains)
    bars = mu >= 0.002 and n >= 0.7 * k and lo > 0
    return dict(mean=mu, se=se, lo=lo, hi=mu + t * se, improved=n, diverged=nd,
                bars=bars, verdict='PASS' if bars and nd == 0 else 'FAIL')


D = 'outputs/ipmnist_screening/wd0_const_step_extend2/'; B = 'rls_head_resid_identmap50_r_idet003_s'
INC = B + '02_anneal_f1t1000_wd0'
CELLS = [(B + '03_anneal_f1t1000_wd0', 0.03), (B + '04_anneal_f1t1000_wd0', 0.04), (B + '05_anneal_f1t1000_wd0', 0.05)]
EDGE = B + '05_anneal_f1t1000_wd0'
TUNE_SEED = 123; SCREEN_SEEDS = range(124, 134)


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


blk = lambda d: ' '.join(f'{st.mean(d[k:k + 10]):+.6f}' for k in range(0, NTASK, 10))


def main(mode):
    if mode == 'tune':
        rows = []
        for c, f in CELLS:
            d = diff('tune', c, TUNE_SEED)
            if d is None:
                print(f'tune seed {TUNE_SEED} {c}: DIVERGED (excluded)'); rows.append((c, f, None)); continue
            print(f'tune seed {TUNE_SEED} {c}: {st.mean(d):+.6f}  blocks(10t): {blk(d)}'); rows.append((c, f, st.mean(d)))
        dec = tune_decision(rows)
        print(f'decision: {dec}' + ('' if dec.startswith('STOP') else ' (winner)'))
        open(D + 'TUNE_DECISION', 'w').write(dec + '\n')
    else:
        c = open(D + 'TUNE_DECISION').read().strip(); S = list(SCREEN_SEEDS)
        ds = {s: diff('screen', c, s) for s in S}; gains = [None if ds[s] is None else st.mean(ds[s]) for s in S]
        for s, x in zip(S, gains):
            print(f'seed {s} ' + ('DIVERGED (FAIL for this seed, scored -1.0)' if x is None else f'{x:+.6f}'))
        ok_s = [s for s in S if ds[s] is not None]
        b = [st.mean(st.mean(ds[s][k:k + 10]) for s in ok_s) for k in range(0, NTASK, 10)] if ok_s else []
        v = screen_verdict(gains)
        print(f'{c} vs {INC}')
        print(f'blocks (10 tasks, mean over {len(ok_s)} valid seeds): ' + ' / '.join(f'{x:+.6f}' for x in b))
        print(f"mean {v['mean']:+.6f} SE {v['se']:.6f} CI95 [{v['lo']:+.6f}, {v['hi']:+.6f}] improved {v['improved']}/10 diverged {v['diverged']}/10")
        print(f"BARS: mean>=+0.002 {v['mean'] >= 0.002}, >=7/10 {v['improved'] >= 7}, CI_lo>0 {v['lo'] > 0}; no diverged seed {v['diverged'] == 0} -> {v['verdict']}")
        if c == EDGE:
            print('EDGE FLAG: the winner is the largest step tested (0.05); the optimum may be higher. Verdict unchanged.')


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except VoidRun as e:
        print(f'VOID: {e}'); sys.exit(3)
