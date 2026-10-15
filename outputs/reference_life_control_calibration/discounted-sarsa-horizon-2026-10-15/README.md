# RiverSwim discount-horizon preregistration

Pending comparison collection. Permanently nonpromoting development only.

After PR #3220, test one new factor: gamma 0.9 versus 0.99 at fixed lambda 0.8
and learning rate 0.014. The upstream reward is delayed by several stochastic
transitions; a longer discount horizon may help learning. Gamma also changes
trace decay and value scale, so this is a combined gamma intervention, not a
clean mediation test or proof that 0.9 has the wrong optimal policy.

[plan.json](plan.json) freezes all six fresh-process full reference lives:
seeds 101001–101003, 20,000 transitions per arm, total 120,000. Publicly open this
input commit as a draft PR before comparison starts. [run.py](run.py) and the
independent [validator](validate.py) are frozen before collection too.

Primary: paired gamma_099 minus gamma_090 lifetime mean reward. All three
differences must be positive and mean above 0.01; an improvement claim also
requires mean above across-seed sample SD. Otherwise nonpositive mean is
negative, and other nonqualifying positive outcomes are inconclusive. No
secondary metric can rescue failure or select a reference control.

Fresh seeds and all other defaults are paired. No source, previous artifact,
frozen gate or exploration schedule is changed. At most two CPU workers, fixed
thread settings, no retries or seed substitutions. Two excluded one-event CLI
preflights use seed 101099. Failed comparison attempts consume their identity
and stop subsequent batches; preserve failure prefixes and logs.

Resources: record endpoint numeric state and exclude native peak memory and
total process overhead. Timing is telemetry only. Full source/runtime/UTC
identities, raw rewards/actions/positions and fixed diagnostic snapshots are
retained. Three development seeds and two gamma values establish no confidence,
optimality, trace superiority, transfer or scientific evidence. No promotion.
