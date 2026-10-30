# Preregistration: anneal schedule sweep on rls_head_resid_identmap50_r_idet003_s005625_anneal_f2t1000

This was committed before any code change or run. Branch `screen/identmap50-r-idet003-s005625-anneal-sweep` is cut from 3c38b025; origin/main is fd25c476 and was fetched 2026-10-30 20:10 GMT, unmoved.
Lane status: development_screening_diagnostic, nonpromoting.

## Parent
The parent is `..._anneal_f2t1000`. Its body step within each detected task is `max(0.005625/2, 0.005625*(1+s/1000)^-1/2)`, where `s = since_shift` in samples and the task length is 5000.
- The floor binds at s = 3000, so the parent runs at the floor 0.0028125 for the last 2000 samples of every task.
- Its 200-task, 20-seed confirmation gave +0.006576, and the gain grew late in life: tasks 181-200 gave +0.00964, against +0.00223 for tasks 1-60.

## Variants (exactly 2; the single changed variable is (floor, tau))
| arm | floor | tau | step at s=5000 | floor binds at |
|---|---|---|---|---|
| parent f2t1000 | 0.0028125 | 1000 | 0.0028125 | s=3000 |
| **gentler f2t2000** | 0.0028125 | 2000 | 0.003007 | never (would be s=6000) |
| **stronger f4t1000** | 0.00140625 | 1000 | 0.002296 | never (would be s=15000) |

- **Stronger, f4t1000** (the suggested arm, already registered): a lower floor keeps the step decaying through the whole task instead of holding flat for the last 40%. The late-growing gain suggests the smaller late-task steps are what help, so a lower floor is the direct test of "more annealing".
- **Gentler, f2t2000** (in place of the suggested f2t500): with tau=1000 the floor is reached at s=3t, so **f2t500 would reach the floor at s=1500. That is a stronger, faster decay, not a gentler one.** A genuinely gentler schedule needs a longer tau. f2t2000 stays above the parent's step at every s > 0, never reaches the floor within a task, and keeps the step closer to 0.005625 early in each task. That is where the anneal screen lost ground (tasks 1-15). It tests whether the parent is over-annealed early.
- f2t2000 is not registered at 3c38b025. It is added as a registry entry only, with no new mechanism. Pin tests cover it: its spec equals the parent's except `anneal_tau`, and it runs through the same anneal code path as the parent's registry entries. A deliberate break must make those tests fail. Then the test file and ruff are run.

## Screen
- Seeds 0-9 at 60 tasks × 5000 for both arms. Each arm is paired per seed against the anneal parent's 60-task shards.
- Parent shards for seeds 0-9 (`wave_identmap50_r_idet003_s005625_anneal/shards/`) are reused only if all three checks pass:
  (1) the traced jaxpr of the parent is identical before and after any change;
  (2) the environment is identical (frozen uv sync, empty freeze diff, equal environment and provenance blocks);
  (3) a fresh seed-0 60-task parent re-run matches the stored shard bit-for-bit.
  Otherwise all of seeds 0-9 are re-run.
- The control `upgd_w_control` is re-measured on seeds 0-2.
- Seeds 10-19 are held out and not run.

## Bar and multiplicity
- **Bar, applied to each arm on its own:** mean paired Δ ≥ +0.002, at least 7/10 seeds improved (the anneal screen's 2-of-3 rule scaled to 10 seeds), and a 95% t-CI lower bound > 0 (df=9, t=2.262157), as in the anneal 10-seed decision.
- **Multiplicity:** there are two comparisons against the same parent. Both arms are reported in full whatever the outcome. An arm also needs its Bonferroni 97.5% t-CI lower bound > 0 (t=2.685011, df=9) to be called "clears". Results are not pooled. No arm is promoted on this 60-task evidence alone: an arm that clears becomes a candidate for its own 200-task confirmation, and if both clear, the better one is still not promoted without that confirmation.
- **Secondary (descriptive, not a decision):** the paired f4t1000 − f2t2000 on seeds 0-9, with mean, 95% CI and the count of seeds where f4t1000 is better. This tells us which direction (more or less annealing) helps.
- **Reported:** per-seed tables, mean, 95% CI and Bonferroni CI, seeds improved, and a horizon breakdown over tasks 1-15, 16-30, 31-45 and 46-60. Any arm missing the bar gets a negative-results entry.
