# Development plan: full-length RiverSwim calibration of discounted SARSA with Q0 = 10

Fixed and committed before any life in this directory was run. Development-only and
permanently nonpromoting. It does not modify or reinterpret the retained 144-shard
scorecard (`outputs/reference_life_scorecard_development/run2-2026-10-09`), populate
`reference-dev`, qualify timing or create scientific evidence.

## Question

#3308 retained optimistic tabular initialization (`initial_q_value = 1 / (1 - gamma) = 10`) for "a
future full control-calibration protocol" after a 6-seed, 5,000-event screen. This plan is that
protocol. Does discounted SARSA with Q0 = 10 pass the scorecard's own RiverSwim control gate at
the scorecard's full length on fresh seeds?

## Fixed protocol

- **Environment:** the scorecard's six-state RiverSwim (`_river_environment_config(build_development_plan())`).
  Exact dispatch, executor `asi.riverswim.executor`, stationary metrics.
- **Horizon:** 20,000 accepted events per life (`RIVERSWIM_HORIZON`, the scorecard's RiverSwim
  length; README of run2-2026-10-09, lines 9-11).
- **Seeds:** 47310, 47311, ..., 47321 (12 seeds, the scorecard's roster size, matching its
  t critical value 2.201 for 11 degrees of freedom). Freshness: at the base commit no tracked file
  contains any of these numbers as a word, no commit on `origin/main` adds a line matching
  `seed ... <n>`, and they appear in none of the diffs of #3184, #3193, #3214, #3218, #3220,
  #3274 or #3308. Consumed rosters for this environment are 70000-70011 (scorecard),
  81010-81015 (#3308), 100101-100103 and 100201-100203 (#3214).
- **Arms (4 lives per seed, 48 total):**
  - `sarsa`: the scorecard's pinned discounted-SARSA arm, built by the scorecard's own
    `_control_adapter`.
  - `sarsa_q0`: identical, plus `initial_q_value=10.0`. This is the only changed variable.
  - `random` and `privileged_oracle`: the scorecard's pinned normalizers, built by
    `_control_adapter`.
- **Lifecycle ids:** `prototype.<sha256("asi.riverswim_q0_full_calibration.development.v1:riverswim:<arm>:<seed>")[:16]>`.
- **Runtime:** locked project environment, CPU, one thread per process, at most 4 concurrent
  processes, a 3 GB MemAvailable floor (memguard pauses and resumes; this does not change results).

## Decision rule (the scorecard's control gate, unchanged)

`scale = mean_s(oracle_s - random_s)`. For each seed, `score_s = (sarsa_q0_s - random_s) / scale`.
`LCB = mean(score) - 2.201 * sd(score) / sqrt(12)`. Constants are imported from
`reference_life_scorecard.py` (`CONTROL_GATE_T_CRITICAL`, `CONTROL_GATE_LCB_THRESHOLD`).

- **PASS:** LCB > 0.1 (strict), with no VOID condition.
- **FAIL:** LCB <= 0.1, with no VOID condition. This is reported as a clean negative.
- **VOID:** any of the following.
  - The roster is not exactly 48 lives.
  - Any life failed, halted or diverged, or any reward, oracle or regret sum is non-finite.
  - Any parameter-change check fails (the scorecard's `parameter_change_check`; `sarsa_q0` uses
    the `sarsa` expectation).
  - Persistent numeric bytes grew, or differ between `sarsa` and `sarsa_q0`.
  - Lives do not share one source and runtime identity.
  - The scale is not finite and positive.
  - Recorded source digests differ from the analysing checkout.

Descriptive only, not part of the gate: the same statistics for default `sarsa`, the paired
normalized `sarsa_q0 - sarsa` difference with its t LCB, high-end visits and the late 2,000-event
reward. Every seed is kept and reported. Nothing is re-run, retuned or dropped after results are
seen.

## Environment check (runs first; failure is a VOID stop with no automatic re-run)

Four lives must reproduce stored records exactly: accepted events, parameter-change events,
reward, oracle and regret sums, transcript SHA-256 and agent manifest id.

1. `scorecard_sarsa_70000`: the scorecard's `riverswim__sarsa__70000.json` (20,000 events, default path).
2. `scorecard_random_70000`: `riverswim__random__70000.json`.
3. `scorecard_oracle_70000`: `riverswim__privileged_oracle__70000.json`.
4. `pr3308_q0_81010`: #3308's `optimistic-81010.json` (Q0 = 10, 5,000 events, its lifecycle
   id). This exercises the `initial_q_value` path.

## Execution and evidence

`launch.sh` runs the env check, then the 48 lives, then `run.py pack` and `analyze.py`. Runtime
state (raw records, logs, STATUS, MEMGUARD_STOP) lives outside the repository. Each log starts with
the HEAD SHA. No tracked file changes while the run is live. Afterwards, `evidence/seed<n>.json`
(one compact line per seed with its four lives), `analysis.json` and `MANIFEST.sha256` are
committed.

Analysis: `.venv/bin/python <this dir>/analyze.py <this dir>/evidence` (exit 0 PASS, 1 FAIL, 3 VOID).
Tests: `tests/test_riverswim_q0_full_calibration.py`.
