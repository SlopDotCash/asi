# CORA continual-RL qualification

This lane pins Powers et al., *CORA: Benchmarks, Baselines, and Metrics as a Platform for
Continual Reinforcement Learning Agents*, CoLLAs 2022, arXiv `2110.10067v2`, and official code
`AGI-Labs/continual_rl@f2754bb282757829765beb4703f24b87efa13ff9` (MIT; audited 17 August 2026).
The code requires Python >=3.7, PyTorch >=1.7 and torchvision. Its environment families add ALE
and Atari ROMs, Procgen, MiniHack plus NLE, or AI2-THOR plus the `crl_alfred` fork and roughly 1 GB
of CHORES trajectories. None is installed or fetched by ASI's qualification CLI.

The pinned metric configuration uses six Atari games for two 50-million-step cycles (600M training
steps); six Procgen games for five 5-million-step cycles (150M), with 200 training levels and the
full level distribution for evaluation; fifteen MiniHack train/eval pairs for two 10-million-step
cycles (300M); and several three-task CHORES sequences at roughly one million steps per task.
CORA continuously evaluates every task, normalizes each task by its largest absolute observed
return, and reports isolated forgetting and zero-shot forward-transfer changes. Its orchestration
knows task boundaries and assigns action-space/task identifiers; whether a particular policy consumes
that identity must be audited per baseline.

ASI's executable `asi.cora_development.v1` slice is a deterministic two-action recurring bandit,
not an environment reproduction. Three reward tasks repeat for two cycles. The runner knows boundaries
to schedule training and continual evaluation, but candidate arms receive no task ID. Replay Q-learning
is paired with an exact-update-budget mechanism-off control that repeats the current transition; a
task-ID Q table is a privileged strong control excluded from candidates, and deterministic uniform
random is retained. Four consumed development seeds bind tie-breaking, task order, updates and replay
selection. Evaluation happens before training and after every block across every task.

Receipts exactly count training and evaluation environment steps, model queries, agent updates,
replay inserts/samples/peak bytes, persistent numeric bytes, logical calls and telemetry-only elapsed
nanoseconds. Validators recompute all counters and all three metric summaries. Results are permanently
nonpromoting, task-information use is explicit, and negative outcomes must be retained.

Protocol gaps before external comparison include environment/ROM/assets checksums and licenses;
the exact historical PyTorch/CUDA/runtime lock; observation preprocessing, frame stacking, action
unification, episode truncation and stochastic seeding; actor/learner concurrency; continual-test
rollout counts and whether they affect learner state; task-ID and boundary exposure per baseline;
CLEAR replay bytes and replay ratio; EWC Fisher computation; P&C capacity; task-specific return
normalization; independent run aggregation; official checkpoint/result parity; full published step
budgets; matched ASI controls; hardware timing; and untouched preregistered scientific seeds. The
native slice's unnormalized binary metrics only test equations and information flow. No CORA result,
performance claim, or SOTA claim exists.

## Procgen compatibility qualification

`cora_procgen_qualification.py` binds the official CORA commit to Git tree
`3c296057e717401053ce0acfe362adeef395aede`, source archive SHA-256
`d634325bd7cc450e68ee55fd5b83118fa4b8d11c0e5e6284daa6bff0a60436db`, and MIT
license SHA-256 `37df918c349040efba06271ed929ffd623506ef2d4a0a7e4ce46e7749ba0cad7`.
It separately binds OpenAI Procgen 0.10.7 at commit
`5e1dbf341d291eff40d1f9e0c0a0d5003643aebf`, tree
`0cb587203bb4d55e001283ad6550f6bc1ef95ad4`, source archive SHA-256
`22940ad0f1fdb4ad1eab3303ce23d3a0ea536700bb1d7c299bee64dbc7c57e9b`, and the
36,395,959-byte CPython 3.10 manylinux2014 x86-64 wheel SHA-256
`4b594d14e42f0f2166e59a9e294477b906eb99c90c3343b570b7124cbc865f53`.
The Procgen MIT license and multi-license asset inventory are independently bound.

This is a proposed isolated Python 3.10 compatibility runtime, not an in-process ASI dependency.
The proposed Torch, torchvision, and Gym artifacts and their transitive dependencies are not yet
content-closed, so dependency locking, runtime qualification, and execution authorization remain
false. Procgen itself documents CPython support only through 3.10 for this release; ASI remains
Python 3.12.

The official CORA constructor normally derives `seed_to_set` from `os.urandom`. A deterministic
compatibility smoke must instead inject an exact uint32 seed and explicitly records that it lacks
official seed parity. The retained receipt schema is create-only: it may report either one initial
observation digest or one bounded failure kind, but always records zero environment steps, rewards,
terminations, policy queries, and model queries. This contract does not authorize a run or create a
CORA result.
