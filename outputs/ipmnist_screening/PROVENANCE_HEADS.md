# Source-commit provenance for the identifier-campaign output directories

Every shard binds the source commit that ran it, and `merge` fail-closes on
any other source (working as designed). Docs-only commits sit ABOVE the
measurement commits on `feat/rls-resid-precond`, so re-running `merge` at
the branch tip refuses old shards; re-run it at the recorded commit instead.

| output directory | measurement commit | contents |
|---|---|---|
| `precond_r1/` | 3818d2b7 | preconditioned-residual screen (neg. results #19-#20) |
| `precond_r2/` | 12dbc136 | gate x Newton 2x2 + n=10 (60t win) |
| `precond_confirm_r1/` | 12dbc136 | tp_nogate 200t confirmation (FAILED; neg. result #21) |
| `identmap_r1/` | 71a0b186 | identifier 60t screen (WIN x2) |
| `identmap_confirm_r1/` | 626b5605 | identmap200_r 200t confirmation (0.9091) |
| `identmap_star_r1/` | 498d4d36 | match-time star 60t screen (WIN x2) |
| `identmap_star_confirm_r1/` | 498d4d36 | identmap50_r 200t confirmation (0.9166) |
| `identmap_star2_r1/` | 9e0f7dcf | star round 2 (both arms REJECTED; neg. result #22) |

The review-fix commit (pin tests + registry pruning) intentionally changes
source AFTER all measurements; nothing in `outputs/` is rewritten.
