# Action-conditioned latent development lane

This is the bounded, permanently nonpromoting comparison lane for backlog issue
`#1575`. It connects ASI's existing `LatentWorldModel` to a real online control
consumer in `SwitchingTwoStateMDP`: after a matched warm-up and recurring
exploration opportunities, the enabled arms choose actions from predicted
immediate rewards. The switching A/B/A reward schedule tests recurrence rather
than a stationary predictive trace.

## Audited sources

The protocol binds exact paper and official-project revisions in every result:

- Dreamer-CDP, `arXiv:2603.07083v2` (revised 2026-04-14), official repository
  commit `a851fa3e3d70b624b094ee1810ad4bb602346092`.
- JEDI, `arXiv:2605.13013v1` (submitted 2026-05-13). No official repository is
  pinned by the current external-qualification catalog.
- JEPA-WM physical planning, `arXiv:2512.24497v3` (revised 2026-05-18, TMLR),
  official repository commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.

The pins establish provenance, not implementation parity. Dreamer-CDP is a
reconstruction-free Dreamer agent with a continuous deterministic JEPA-style
predictor evaluated on Crafter. JEDI is an end-to-end stochastic latent
diffusion world model evaluated on Atari100k and reports materially lower VRAM
and faster sampling/training than its pixel-diffusion control. JEPA-WM trains
from visual state-action datasets and plans in representation space on
simulated and real physical tasks. This lane instead uses fixed random features,
online one-step deterministic prediction, no replay, no imagination/rollout
planner, and a two-state nonvisual stream. Its scores are not comparable to any
paper result.

## Frozen development comparison

Seeds `1575000..1575003`, horizon, phase schedule, initial-state derivation, and
exploration schedule are matched. The roster is:

1. action-conditioned latent prediction with action×latent interactions;
2. the same model without interaction features;
3. an action-masked causal ablation;
4. a trained model behind a disabled decision interface;
5. exact no-model mechanism-off control; and
6. the repository's live SARSA control agent as a stronger non-world-model
comparator.

The existing `SparseFTLWorldModel` is not yet a valid seventh arm. It predicts
only the next observation, while this environment switches its reward matrix
without changing transition dynamics and does not expose the active phase to
the learner. The same current observation, action, and next observation can
therefore carry reward 0 in phase A and reward 1 in phase B. Applying the
environment payoff table to an FTL state prediction would require privileged
phase information and violate the matched information boundary. A valid FTL
arm needs an online learned reward head (with its updates, queries, and bytes
charged) or a separately justified decision objective; next-observation
prediction alone cannot drive the existing immediate-reward selector.

The decision-off and mechanism-off arms must have identical action/reward
transcript hashes. Receipts count real environment steps, model updates,
prequential training queries, decision queries, and exact persistent JAX-array
bytes. Timing is omitted because no qualified timing protocol exists. Compact
hashes are consistency bindings, not authenticated execution proof.

Run the bounded lane with `asi-action-conditioned-latent`. It prints a strict
JSON receipt and never writes `outputs/`. Any negative result remains in the
receipt; no result can promote scientific evidence.

## Gates left open

- reproduce the papers' visual encoders, replay and multi-step imagination or
  diffusion/planning objectives at their pinned revisions;
- add matched Crafter/Atari100k and physical-planning protocols with paper-exact
  preprocessing, action spaces, horizons, seeds, and metrics;
- qualify observation/model queries, accelerator memory, training FLOPs, and
  latency under a separately frozen resource protocol;
- test longer dynamics changes, delayed rewards, stochasticity, and retention;
- freeze untouched evaluation seeds only after development selection; and
- validate robot sensing, control frequency, safety/veto, sim-to-real transfer,
  and hardware resource budgets.
