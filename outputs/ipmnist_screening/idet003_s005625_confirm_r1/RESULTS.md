# Results — 200-task, 20-seed confirmation of `rls_head_resid_identmap50_r_idet003_s005625`

Bars frozen in `PREREGISTRATION.md` (commit 5d99707c) before any run. Both arms on one runner
(Linux x86_64, Python 3.13.5, jax 0.11.0) at main fd25c476 + this folder; merged `summary.json`.
Development screening diagnostic; permanently nonpromoting; development seeds 0-19 consumed.

## Outcome: CONFIRMED — development best moves

| arm | acc (n=20, mean ± stderr) | paired vs reference | seeds improved |
|---|---|---|---|
| `rls_head_resid_identmap50_r_idet003_s005625` | **0.943181 ± 0.000047** | **+0.026599 ± 0.001105** | **20/20** |
| `rls_head_resid_identmap50_r` (local re-run) | 0.916581 ± 0.001112 | — | — |

- Paired SD 0.004941; 95% t-CI (df=19) [+0.024287, +0.028912]; weakest seed +0.020808 against the +0.002 bar.
- Late window (tasks 181-200) paired +0.032051 (95% CI [+0.029300, +0.034802]); no drift-down.
- Candidate per-seed SD 0.000209 vs the reference's 0.004971 (24x smaller): the equalization signature holds at 200 tasks.

## 60 -> 200 tasks: no shrinkage
The paired difference over the first 60 tasks of these same shards is +0.021238 ± 0.001592 (stderr); over 200 it is
+0.026599. The ratio 60/200 is 0.80, so the effect GROWS by 1.25x, unlike the 3.5-4.7x decays in negative results
#21/#23. On seeds 0-2 the local 60-task prefix (0.93429/0.93457/0.93467) equals this host's Autostep-wave parent re-run
and sits 0.0002 below the stored #3117 shards (cross-host offset, not bitwise).

## Reference reproduction (identmap50_r)
The local re-run gives 0.916581 vs stored 0.916569 (`identmap_star_confirm_r1`, Darwin arm64): offset +0.000013 ±
0.000162 SD per seed (95% CI [−0.000063, +0.000088]), max |per-seed| 0.000287. It is not bit-exact on any seed
(cross-platform XLA). The offset is about 100x smaller than the effect; pairing uses the local reference.

## Per-seed
| seed | candidate | ref (local) | ref (stored) | local-stored | paired diff | diff @60 | late 181-200 |
|---|---|---|---|---|---|---|---|
| 0 | 0.942928 | 0.918463 | 0.918394 | +0.000069 | +0.024465 | +0.018293 | +0.024520 |
| 1 | 0.942792 | 0.904746 | 0.904781 | -0.000035 | +0.038046 | +0.037120 | +0.038760 |
| 2 | 0.943148 | 0.920875 | 0.921044 | -0.000169 | +0.022273 | +0.013467 | +0.029440 |
| 3 | 0.943264 | 0.905821 | 0.905681 | +0.000140 | +0.037443 | +0.035910 | +0.033330 |
| 4 | 0.943187 | 0.922379 | 0.922385 | -0.000006 | +0.020808 | +0.013437 | +0.023800 |
| 5 | 0.943076 | 0.916392 | 0.916308 | +0.000084 | +0.026684 | +0.023450 | +0.032470 |
| 6 | 0.943474 | 0.921260 | 0.921394 | -0.000134 | +0.022214 | +0.016637 | +0.038750 |
| 7 | 0.943295 | 0.915054 | 0.915336 | -0.000282 | +0.028241 | +0.022267 | +0.034420 |
| 8 | 0.943421 | 0.916210 | 0.916274 | -0.000064 | +0.027211 | +0.016423 | +0.028480 |
| 9 | 0.943116 | 0.916841 | 0.916845 | -0.000004 | +0.026275 | +0.024077 | +0.032450 |
| 10 | 0.942966 | 0.916426 | 0.916336 | +0.000090 | +0.026540 | +0.019943 | +0.040870 |
| 11 | 0.942916 | 0.919792 | 0.919935 | -0.000143 | +0.023124 | +0.022543 | +0.024630 |
| 12 | 0.943367 | 0.919177 | 0.919071 | +0.000106 | +0.024190 | +0.016840 | +0.044340 |
| 13 | 0.943119 | 0.916293 | 0.916566 | -0.000273 | +0.026826 | +0.013517 | +0.037250 |
| 14 | 0.943378 | 0.920135 | 0.919955 | +0.000180 | +0.023243 | +0.017443 | +0.025380 |
| 15 | 0.943350 | 0.907301 | 0.907040 | +0.000261 | +0.036049 | +0.035180 | +0.033050 |
| 16 | 0.943424 | 0.919342 | 0.919055 | +0.000287 | +0.024082 | +0.019323 | +0.033270 |
| 17 | 0.943015 | 0.918157 | 0.918216 | -0.000059 | +0.024858 | +0.020587 | +0.025750 |
| 18 | 0.943440 | 0.918033 | 0.917842 | +0.000191 | +0.025407 | +0.020513 | +0.032160 |
| 19 | 0.942938 | 0.918931 | 0.918920 | +0.000011 | +0.024007 | +0.017797 | +0.027900 |

## Operations
- At most 4 one-thread workers; `free -g` and a cargo-build-sbf gate before every shard launch (`screen.log`). No pauses
  were triggered; a separate eliza/thenetwork build shared the host. Wall clock was 528-613 s per shard.
- Interruptions: the orchestrating session was interrupted three times (15:52, about 16:35 and about 16:42 Europe/London). The
  launcher ran detached (nohup) and kept going; every shard ran once to completion (each log reports `done`, 200 tasks).
  No shard was re-run, deleted or edited.
