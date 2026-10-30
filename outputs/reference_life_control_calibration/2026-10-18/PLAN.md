# RiverSwim optimistic-SARSA calibration screen

This is a permanently nonpromoting development screen. It does not modify or
reinterpret the completed reference-life scorecard, populate `reference-dev`,
qualify timing, or create performance or scientific evidence.

## Bottleneck and hypothesis

The completed 144-shard scorecard has a valid baseline failure because its
RiverSwim SARSA-family controls do not pass the frozen calibration gate. The
discounted-SARSA outcomes are bimodal: five seeds reach about 87--88% of the
oracle-normalized scale while seven remain near random. A fresh diagnostic with
constant epsilon 0.5 earned only 66.35 reward and never reached the high end in
20,000 events, rejecting slower epsilon decay as the intervention.

The hypothesis is that zero-information random parameter initialization lets
the sparse left reward lock the policy before it discovers the right-end
reward, while an optimistic tabular `Q_0 = 1 / (1 - gamma) = 10` supplies
directed exploration without adding persistent numeric state.

## Frozen screen

- Environment: the scorecard's six-state RiverSwim configuration.
- Horizon: 5,000 accepted events; early and late windows are 1,000 events.
- Fresh development seeds: 81010--81015.
- Matched arms: the scorecard discounted-SARSA configuration with its default
  initialization, and the same configuration with `initial_q_value=10.0`.
- Primary metric: paired lifetime reward difference, optimistic minus default.
- Mechanism check: post-transition visits to state 5.
- Resource check: exact equality of initial persistent numeric bytes and no
  within-arm persistent-byte growth.

The screen supports retaining the mechanism when at least five of six paired
reward differences are positive, mean paired reward improves, at least five of
six optimistic runs visit state 5, every run completes without a halt, and the
resource check passes. Otherwise it is a useful negative result. These consumed
development seeds may never be used for promotion.

