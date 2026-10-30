# RiverSwim discounted-SARSA exploration-decay diagnostic

This directory retains a separate, permanently nonpromoting development diagnostic.
The framework source tree, literal scorecard plan, retained scorecard receipts and qualification gates are unchanged.

The current source's frozen scorecard has `valid_baseline_failure` on RiverSwim. This diagnostic changes only discounted SARSA's `epsilon_decay_steps` from 2,500 to 10,000. Gamma, step size, network, seeds, environment, initialization, horizon and update budget are otherwise matched.

The canonical plan was fixed before seeds 100201, 100202 and 100203 were observed. Each of the six shards executes all 20,000 transitions through `ReferenceLifeRunner`, with at most two independent shards running concurrently. The paired decision rule requires every primary delta to be positive and the mean paired delta to exceed 0.01; a failure is retained rather than retuned.

The primary statistic is mean reward across all transitions. The report includes sample standard deviations, each paired delta, late-window means, high-end visits, complete counters, agent numeric payload sizes and source/runtime identities. Timing is telemetry only. Native peak memory, whole-agent performance, checkpoint safety and scientific promotion are not qualified.

An earlier scan-based plan used separate development seeds 100101–100103. It stopped when seed 100103 did not pass its required 16-step exact learner-state check. Four completed shards from that attempt are kept only as excluded diagnostic records. They are not mixed with the canonical results and do not support a complete three-seed conclusion. The measured v1 shards used 80,000 transitions, with extra short-prefix checks; v1's 120,000-event budget described the planned measured shards and did not include those checks. The v2 protocol removes that shortcut and uses fresh development seeds.

The existing differential-SARSA calibration PR #3184 changes bias and the exploration schedule. This experiment studies a different learner and one variable. The descriptive audit PR #3193 analyses old retained shards; this experiment executes new paired lives. No broad family or published-method claim is made.

## Result

All six full reference lives completed: **120,000 accepted and executed transitions, each passed through the learner update**, with matched environment cursors. The slower decay has a lower whole-life mean reward on all three tested development seeds. The predeclared hypothesis is **not supported**; no setting is selected or promoted.

| Development seed | Decay 2,500 mean reward | Decay 10,000 mean reward | Paired delta | Rightmost visits, base / candidate |
| --- | ---: | ---: | ---: | ---: |
| 100201 | 0.004838250 | 0.004473500 | -0.000364750 | 0 / 1 |
| 100202 | 0.767726750 | 0.613397500 | -0.154329250 | 15554 / 12718 |
| 100203 | 0.768933500 | 0.616889500 | -0.152044000 | 15606 / 12812 |

Across seeds, the baseline is **0.513832833 +/- 0.440802653** and slower decay is **0.411586833 +/- 0.352574812** (mean +/- sample standard deviation). The paired mean difference is **-0.102246000 +/- 0.088239149**. These are descriptive development statistics, not a confidence claim.

Late-window rewards are identical within each pair: approximately 0.0049275, 0.8035 and 0.8255. The longer decay does not establish sustained high-reward learning on the low-return seed, and it reduces whole-life return on the other two. This rejects only the tested one-parameter change under this plan. It does not reject other exploration schedules, remove the state-independent bias, qualify controls for the frozen scorecard, or refute SARSA generally.

The measured control-state numeric payload is **220 bytes, 55 elements and 23 leaves** at both endpoints for both arms. This counts the control-state arrays, not total process memory, environment/wrapper metadata, transient allocations or checkpoint storage. Both arms have equal 20,000-step execution and update budgets; native peak memory is unmeasured.

Runtime: Windows AMD64, CPython 3.12.4, JAX/JAXlib 0.11.0, NumPy 2.5.1 and Chex 0.1.92, CPU backend. The processes set OMP, OpenBLAS and MKL thread environment values to 1 and disable XLA preallocation. At most two shards run concurrently. No cross-machine timing or latency claim is made.

## Reproduction

Use the committed plan, exact source byte manifest and locked project environment. On Windows the interpreter is `.venv/Scripts/python.exe`; on Linux it is `.venv/bin/python`. Run one command for each `(seed, decay)` pair into a new empty output directory. The script refuses existing files.

```text
.venv/Scripts/python.exe outputs/reference_life_control_calibration/discounted-sarsa-decay-2026-10-15/experiment.py --seed 100201 --decay 2500 --out new-results/baseline-seed100201.json
.venv/Scripts/python.exe outputs/reference_life_control_calibration/discounted-sarsa-decay-2026-10-15/experiment.py --seed 100201 --decay 10000 --out new-results/slow-decay-seed100201.json
.venv/Scripts/python.exe outputs/reference_life_control_calibration/discounted-sarsa-decay-2026-10-15/experiment.py --seed 100202 --decay 2500 --out new-results/baseline-seed100202.json
.venv/Scripts/python.exe outputs/reference_life_control_calibration/discounted-sarsa-decay-2026-10-15/experiment.py --seed 100202 --decay 10000 --out new-results/slow-decay-seed100202.json
.venv/Scripts/python.exe outputs/reference_life_control_calibration/discounted-sarsa-decay-2026-10-15/experiment.py --seed 100203 --decay 2500 --out new-results/baseline-seed100203.json
.venv/Scripts/python.exe outputs/reference_life_control_calibration/discounted-sarsa-decay-2026-10-15/experiment.py --seed 100203 --decay 10000 --out new-results/slow-decay-seed100203.json
.venv/Scripts/python.exe outputs/reference_life_control_calibration/discounted-sarsa-decay-2026-10-15/summarize.py new-results new-summary.json
```

Run from the repository root and provide the script's path. The reproduction snapshot is `415e7d225a4338492275be6cafa69d87c158c30a`; run the experiment from that snapshot so new records bind to its bytes. The summary, run from a full clone, rejects any raw record whose recorded plan, script or source digests differ from the retained files or from that commit's tracked blobs. The PR adds the run records and analysis only. Existing framework source bytes are unchanged. `canonical-calibration-plan.json` must be next to `experiment.py`, as required by the original script.

The summary is recomputed from every raw reward, action and resulting state. It validates the RiverSwim reward lattice and deterministic left transitions, complete counters, identical runtime/source identity, equal numeric payload sizes and the exact paired roster. Hashes bind consistency; they are not authenticated execution attestation.

Re-derive the stored summary without rerunning or consuming any seed:

```text
.venv/Scripts/python.exe outputs/reference_life_control_calibration/discounted-sarsa-decay-2026-10-15/summarize.py outputs/reference_life_control_calibration/discounted-sarsa-decay-2026-10-15 new-rederived-summary.json
```

The six input checksums and script/plan digests match the preserved bytes. Independent recomputation validates all 120,000 reward/state/action records.

## Validation limits

- The 34 existing reference-control tests passed.
- Repository Ruff passed before the evidence-only change.
- Repository mypy reports 108 errors in 17 unchanged files on Windows, including POSIX `fcntl`/`os.uname` surfaces. No type gate is weakened.
- The five existing scientific claims already report `invalid` on the baseline due registered-source drift and other retained validation failures. Existing artifacts and validators are preserved.
- The full fast/runtime/scientific suites are not rerun for this evidence-only change. No existing framework source or algorithm implementation is modified.
The preserved experiment snapshots retain the exact source bytes used by the raw shard digests. The standalone summary tool is formatted for inspection; its output was independently re-derived from the stored raw traces.

Raw records preserve their original Windows CRLF bytes because the recorded digests are byte-based. The directory-scoped `.gitattributes` disables text conversion for this new run and recognizes CRLF endings while retaining the default trailing-space, blank-at-EOF and space-before-tab checks. It does not apply to framework source or existing artifacts.
