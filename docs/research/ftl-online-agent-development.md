# FTL Online Agent / Continual Bench development lane

This record pins Liu et al., *Continual Reinforcement Learning by Planning with Online World
Models*, ICML 2025, PMLR 267:38397–38423, and arXiv `2507.09177v1` (12 July 2025). The official
Continual Bench repository is pinned at
`sail-sg/ContinualBench@a4fdb3b94a07a40d76e28d3aeab0f8ca97519dad` (the `main` revision audited
on 17 August 2026). The paper uses a sparse shallow FTL world model, CEM MPC, known changing
reward functions, six 26-dimensional MuJoCo/Meta-World-derived tasks, 600 episodes, and reports
10–15 hours per run on one A100 plus 16 CPUs.

ASI's new `asi.ftl_online_agent_development.v1` lane is deliberately smaller. It connects the
current `SparseFTLWorldModel` to a real observe–plan–act–update loop in a deterministic 2-D,
four-action, three-reward-task analogue. It freezes four development seeds, task order, per-task
steps, MPC horizon, and three arms: online FTL, the identical frozen/no-update mechanism-off arm,
and a live privileged-dynamics MPC control excluded from candidate comparisons. Receipts count
environment steps, accepted model updates, model queries, planner candidates, exact logical compute
units (their sum plus environment transitions), persistent array bytes, and wall time (telemetry
only). Logical units are an implementation-independent call count, not hardware FLOPs. Negative
results must be retained.

The planner maximizes the summed reward over the imagined rollout, matching the paper's Eq. 2
`H`-step finite-horizon return and the per-step cost the lane actually reports. Scoring only the
terminal imagined state optimizes a different objective than the one measured: under it the
privileged-dynamics control degraded monotonically with deeper lookahead (-6, -22, -50, -82 at
horizons 1-4) and lost to the learned online arm on two of four frozen seeds at the frozen default
horizon of two, so it no longer bounded the lane. With the summed return the control holds the
analytic optimum of -6 at every horizon on every frozen seed. Horizon one is a single imagined
step, where the two objectives are identical.

Every arm receives the same current goal for planning and no boundary or task identifier; the
world model itself trains only on observation, action, and next observation. The validator derives
the exact metric sequence length and persistent numeric bytes from the frozen execution contract,
rather than trusting either value from a stored receipt.

This is not Continual Bench parity, a paper-result reproduction, or a scientific result. It does
not use MuJoCo, the paper's 26-D observation contract, continuous actions, CEM, paper LoSSE
hyperparameters, six tasks/reward thresholds, 600-episode budget, evaluation rollouts, deep-model
baselines, task-boundary-dependent baselines, paper regret/AP metrics, hardware, or timing protocol.
Before comparison, pin the official environment dependencies and assets, independently reproduce
the official revision, implement continuous-action CEM with matched planning calls, add Perfect
Memory and deep continual-learning controls, match episode/evaluation schedules, audit information
available to every arm, qualify resource instrumentation, and freeze untouched scientific seeds.

The historical `ftl_world_model_decision_fidelity` artifact remains a separate immutable narrow
open-loop diagnostic. This lane neither reads it nor changes its source inventory, schema, claim,
or outcome; `historical_ftl_claim_reused=false` is validator-enforced.
