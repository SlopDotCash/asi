# JEPA-WM / V-JEPA 2-AC transfer feasibility

This is the permanently nonpromoting native feasibility lane for issue `#1577`.
It tests one narrow architectural question: can an encoder learned from earlier
ASI transitions be frozen, transferred into a fresh action-conditioned
predictor, and consumed by a live controller under explicit costs?

## Independently verified references

- V-JEPA 2, `arXiv:2506.09985v1`, submitted 2025-06-11; official
  `facebookresearch/vjepa2` commit
  `204698b45b3712590f06245fbfba32d3be539812`. The paper pretrains on more than
  one million hours of internet video, then post-trains V-JEPA 2-AC on under 62
  hours of DROID robot video. The pinned repository exposes 300M–1B parameter
  V-JEPA 2 checkpoints and a ViT-g/16 action-conditioned checkpoint.
- JEPA-WM, `arXiv:2512.24497v3`, revised 2026-05-18 and accepted at TMLR;
  official `facebookresearch/jepa-wms` commit
  `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`. It studies visual
  state-action training, representation-space planning, and simulated and
  real-world navigation/manipulation, reporting gains over DINO-WM and
  V-JEPA-2-AC.

Both exact GitHub commit pages and repository trees were inspected. The pins
are provenance only; external code, checkpoints, and datasets are not imported
or executed by this lane.

## External qualification inventory

Issue #1577's read-only qualification catalog is emitted with:

```bash
.venv/bin/asi-jepa-transfer-feasibility --external-catalog
```

It binds the inspected JEPA-WM source commit and tree, its downloaded source
archive, and its CC BY-NC 4.0 license bytes. It separately binds the V-JEPA 2
source commit and tree, source archive, MIT `LICENSE`, and Apache 2.0
`APACHE-LICENSE`. These are distinct license records; the permissive V-JEPA 2
source license does not override JEPA-WM's noncommercial license.

The catalog also binds Hugging Face repository revision
`9b9c41ef249466630dbf1a20e78391865d07b3b9` and the LFS SHA-256 identities and
sizes of `jepa_wm_droid.pth.tar`, `vjepa2_ac_droid.pth.tar`, and
`vjepa2_ac_oss.pth.tar` (11,678,963,913 bytes in total). It did not download or
deserialize those checkpoint objects. The JEPA-WM configuration subtree is
bound independently; the listed DROID, RoboCasa/RoboSuite, MetaWorld, Push-T,
PointMaze, Wall, and Franka inputs remain names only, not content-bound assets.

All license, dataset-rights, checkpoint-deserialization, external-execution,
physical-execution, paper-parity, and scientific-promotion gates remain closed.
The catalog is a permanently nonpromoting development inventory, not approval
to acquire, redistribute, or execute an external model or dataset.

## Native transfer experiment

The runner stores a bounded ASI-only `SwitchingTwoStateMDP` transition trace,
takes one trainable-encoder pass over it, and transfers only the learned encoder
into a freshly initialized predictor. That predictor then selects actions from
predicted rewards during an A/B recurrence. Frozen seeds, environment horizon,
pretraining trace length, warm-up, and exploration schedule are matched.

The roster separates:

1. encoder-only ASI transfer;
2. no-pretraining with the same frozen random-feature architecture;
3. a permuted transferred-encoder causal control;
4. a full encoder-and-predictor warm-start ceiling;
5. transferred encoder with its decision interface disabled;
6. exact no-model mechanism-off; and
7. the current SARSA agent as a strong live control.

Decision-off must reproduce mechanism-off action and reward hashes exactly.
Every arm reports environment/pretraining steps, pretraining examples and
updates (including applied encoder updates), imported-pretraining bytes (always zero), stored pretraining replay
bytes, online replay bytes (always zero), semantic encoder/model/control query
counts, encoder and total persistent bytes, environment bytes, and scoped
`perf_counter_ns` telemetry for pretraining, decisions, online updates, and
environment/control execution. Timing is process-local telemetry only, never an acceptance metric.
Semantic query counts describe public model operations; they are not FLOP or
kernel-launch estimates. Hashes are consistency bindings, not execution proof.

Run `asi-jepa-transfer-feasibility`. It prints one strictly validated JSON
receipt and never writes `outputs/`. Negative outcomes remain recorded and the
schema forbids scientific promotion or visual/robotics parity claims.

## Explicit gaps

This two-state, one-hot, one-step JAX lane is not a visual JEPA reproduction. It
has no video masking/tokenization, transformer encoder, target-encoder EMA,
image-goal energy planner, multistep latent rollout, DROID/web-video data,
imported checkpoint, camera calibration, or physical robot. It also lacks
paper-exact datasets, preprocessing, model scales, planners, horizons, metrics,
and seeds. External feasibility still requires an isolated PyTorch/CUDA runtime,
asset checksums and licenses, exact checkpoint/data/pretraining receipts,
matched planning budgets, qualified accelerator memory/FLOPs/latency, longer
recurrence and stochastic-retention tests, untouched evaluation seeds, and
robot safety/veto, control-frequency, sim-to-real, and hardware gates.
