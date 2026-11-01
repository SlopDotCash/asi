# Verification: code at the run commit vs the published head

Checked on 2026-11-01 (GMT). Run commit `fc5d025b`; published head `20e29af5`.

## `git diff --stat fc5d025b 20e29af5` over the library source, runner, launcher and probe test

```
$ git diff --stat fc5d025b 20e29af5 -- <files below> alberta_framework/
(no output: no differences)
```

## sha256 per file at each commit (`git show <commit>:<path> | sha256sum`)

```
fc5d025b 16af65a3f151c04c095c1fbb1f49fd64483402f1dcaf4d8c25dcf738d9a7f9b0  alberta_framework/reference_life_controls.py
fc5d025b 3dea62162c04e44b1c59c88b2ec1aa08bc7efa939bd474b44516820df05b1970  alberta_framework/core/average_reward.py
fc5d025b fc07861de79d433401f79f87d2ca857e34ffb342e3a6eab4702eb2ed0c85c1b1  alberta_framework/core/options.py
fc5d025b c98135eafad74b7dad7ba6491781dabba407a9c4752b2bb5c0d45565b82774e3  alberta_framework/streams/closed_loop.py
fc5d025b 29e63414e933afae66803837113c9b179e8e49d1dc77f0e5fdad19686d78c67c  alberta_framework/benchmarks/reference_life_scorecard.py
fc5d025b 7a0467b7148e4431f2dd7957e19d5d84f2bcc5231adcd20d04a52505a385b486  outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/run.py
fc5d025b bd940afb79ddc7811a3acbd58751b082e9fb1ba9d272bda0c6a53b8ad18dbf00  outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/launch.sh
fc5d025b 52b20c2497d4ce567bcf52c060385db79610373963119e4b18cfbb71d5c1de7c  outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/envcheck.py
fc5d025b 9a83435ad09f5b1c5604cade115af7f8cc7944a3a1dcdce4c01e9577e695cd1b  outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/memguard.sh
fc5d025b d58f48e5720369a595274cd4dfa3af095a2d1bdab2e3e7e933373f970b240874  outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/analyze.py
fc5d025b db4e69915ce29dc58a220e807a7a9e81b29c36edba4bed83479d422d9fe767c4  tests/test_riverswim_diffsarsa_rbar0_calibration.py
20e29af5 16af65a3f151c04c095c1fbb1f49fd64483402f1dcaf4d8c25dcf738d9a7f9b0  alberta_framework/reference_life_controls.py
20e29af5 3dea62162c04e44b1c59c88b2ec1aa08bc7efa939bd474b44516820df05b1970  alberta_framework/core/average_reward.py
20e29af5 fc07861de79d433401f79f87d2ca857e34ffb342e3a6eab4702eb2ed0c85c1b1  alberta_framework/core/options.py
20e29af5 c98135eafad74b7dad7ba6491781dabba407a9c4752b2bb5c0d45565b82774e3  alberta_framework/streams/closed_loop.py
20e29af5 29e63414e933afae66803837113c9b179e8e49d1dc77f0e5fdad19686d78c67c  alberta_framework/benchmarks/reference_life_scorecard.py
20e29af5 7a0467b7148e4431f2dd7957e19d5d84f2bcc5231adcd20d04a52505a385b486  outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/run.py
20e29af5 bd940afb79ddc7811a3acbd58751b082e9fb1ba9d272bda0c6a53b8ad18dbf00  outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/launch.sh
20e29af5 52b20c2497d4ce567bcf52c060385db79610373963119e4b18cfbb71d5c1de7c  outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/envcheck.py
20e29af5 9a83435ad09f5b1c5604cade115af7f8cc7944a3a1dcdce4c01e9577e695cd1b  outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/memguard.sh
20e29af5 d58f48e5720369a595274cd4dfa3af095a2d1bdab2e3e7e933373f970b240874  outputs/reference_life_control_calibration/riverswim-diffsarsa-rbar0-2026-11-01/analyze.py
20e29af5 db4e69915ce29dc58a220e807a7a9e81b29c36edba4bed83479d422d9fe767c4  tests/test_riverswim_diffsarsa_rbar0_calibration.py
```

Every library, runner and launcher file is byte-identical at both commits. Between the two
commits only files inside this probe directory changed: per-seed evidence, env-check lives,
`analysis.json`, `RESULTS.md`, `MANIFEST.sha256`, and a one-line prose fix in
`DEVELOPMENT_PLAN.md` (`git diff --name-status fc5d025b 20e29af5`: 19 added, 1 modified).

After `20e29af5`, `DEVELOPMENT_PLAN.md` received a doc-only adapter path correction
(`alberta_framework/reference_life_controls.py`). `MANIFEST.sha256` is left as written
with the results and lists the plan as `aa69b5e7…0796`; the corrected plan is `e84220b2…2cd7`.

If #3651 merges first, the env check is re-run on top of it before merge.
