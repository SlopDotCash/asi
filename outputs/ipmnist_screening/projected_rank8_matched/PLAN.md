# Fixed rank-eight covariance on the IPMNIST screening lane

The [campaign index](../../../docs/research/ipmnist-campaign-index.md#historical-identmap-development-waves) leaves the low-rank question open after an indefinite recurrence invalidated the earlier head-capacity interpretation. This development screen tests one specified subspace. Seven normalized pools aggregate the hidden coordinates by index modulo seven. The eighth coordinate preserves the readout bias. A square-root covariance evolves only in that subspace. The readout remains full width in storage and starts at zero in every arm.

## Comparison fixed before running

Use the public commit carrying this plan, the maintained v2 step-noise CLI and canonical OpenML MNIST. Run **three arms on each of development seeds 191171, 191172 and 191173** for **12 tasks of 5,000 examples** per run. The seeds were absent as whole numbers in the retained IPMNIST screening and local hillclimb outputs at base `9fd70d6765f87637a5f24e08a96493c9e0ede514`. These same-stream measurements are permanently nonpromoting.

| Role | Registered ID |
| --- | --- |
| Matched dense parent | `rls_head_resid_identmap50_r_idet003_s0031640625_r02` |
| Rank-eight candidate | `rls_head_resid_identmap50_r_idet003_s0031640625_r02_projected8` |
| Contemporary dense lead, reported separately | `rls_head_resid_identmap50_r_idet003_s02_anneal_f1t1000_wd0` |

The primary paired contrast is the candidate minus its matched parent in mean pre-update accuracy. The storage tradeoff justifies another development screen only if the mean deficit is at most 0.002 and no individual seed loses more than 0.004. Otherwise stop advancing this fixed basis. These margins are decision rules for these three seeds. The current lead differs in more than covariance and provides deployment context. The mechanistic result concerns this fixed basis and horizon. An adaptive rank-eight sketch would require a new comparison.

Run all nine shards serially in fresh processes, cycling the arm order across seeds. Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` and `JAX_PLATFORMS=cpu`. Compare the exact source, data, runtime and seed schedule. Record the primary metric per seed and its paired mean and sample standard deviation. The 151-feature head keeps 64 float32 covariance entries versus 22,801 in its dense parent. Report actual peak process RSS and wall time for each arm, including all feature, readout and body state. The nine runs total 540,000 online examples with one update per example. They change no reference or scientific claim.
