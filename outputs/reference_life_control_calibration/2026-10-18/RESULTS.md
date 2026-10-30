# RiverSwim optimistic-SARSA calibration result

The frozen six-seed development screen retains optimistic tabular initialization
for a future full control-calibration protocol. Every optimistic arm improved
over its matched default-initialization control.

| Metric | Default initialization | Optimistic `Q_0=10` |
| --- | ---: | ---: |
| Mean 5,000-event reward | 22.365833 | 2,988.749167 |
| Mean final-1,000-event reward | 4.925000 | 814.833333 |
| Seeds reaching state 5 | 1 / 6 | 6 / 6 |
| Persistent numeric bytes, initial/final | 220 / 220 | 220 / 220 |

The six paired reward improvements range from 2,876.075 to 3,107.075,
with a mean of 2,966.383334. Optimistic runs first reached state 5 between
events 143 and 362. All 12 exact-dispatch lives completed, each applied a
parameter change on all 5,000 accepted events, and no arm's persistent numeric
payload grew.

The result supports the mechanism hypothesis: sustained undirected exploration
was ineffective in the preceding diagnostic, while a resource-neutral optimistic
tabular prior reliably crossed the sparse-reward chain. It does not select a new
scorecard control. The screen is shorter than the completed scorecard, uses only
six data-informed development seeds, has no SwitchingTwoState matched panel, and
does not rerun the frozen calibration gate. Timing remains telemetry-only.

`summary.json` owns the decision and exact paired values. The 12 shard records
bind source commit `b132b2693a1305baefe20e98e4dea80d7b252f80`, exact source-file
digests, runtime/dependency identity, resolved agent config and manifest, full
resource counts, transcript digest, and the permanently nonpromoting evidence
policy. The consumed seeds 81010--81015 may never be used for promotion.

The `summarize` command refuses shards whose recorded source-file digests
differ from the summarizing checkout or whose recorded commit is not the
current `HEAD` or one of its ancestors. The commit that retains the shards
cannot be named by the shards themselves, so the recorded commit is the
parent of the retaining commit while every bound source digest is exact.
The rerun at `b132b269` reproduced every shard of the earlier `e9b0760f` run
bit for bit outside the telemetry-only wall-clock field.

Revalidate and reconstruct the summary from the immutable shards:

```bash
.venv/bin/python outputs/reference_life_control_calibration/2026-10-18/run.py \
  summarize \
  --shards outputs/reference_life_control_calibration/2026-10-18/baseline-*.json \
           outputs/reference_life_control_calibration/2026-10-18/optimistic-*.json \
  --out /tmp/reference-life-control-calibration-summary.json
```
