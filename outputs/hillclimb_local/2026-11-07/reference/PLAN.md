# ASI local reference-life comparison

Permanently nonpromoting. Python 3.12.14, locked JAX 0.11.0 on macOS arm64, CPU only. Use a fresh Python process per life and the retained PR runners at a local composition of #3664 and #3655. Use the exact same explicit life identifier and seed across arms for each environment. Seeds 118331 through 118333 were selected before running and are absent from the retained lanes inspected.

Switching: all four arms on each seed, 4,000 accepted events. Compare Prototype base step 0.20 against 0.10, scorecard 0.05 and differential SARSA. Record paired raw reward, parameter-change events and numeric payload. Hypothesis: 0.20 wins all pairs against 0.10 and exceeds mean differential SARSA reward with unchanged Prototype payload bytes.

RiverSwim: all four arms on each seed, 20,000 accepted events. Compare discounted SARSA with Q0=10 against discounted SARSA at default initialization, and differential SARSA with initial average reward 1.0 against differential SARSA at default initialization. Also report direct paired comparison between the two optimistic controls. Record paired raw reward, high-end visits, parameter-change events and numeric payload. Hypothesis: both optimistic arms improve on their own defaults, with all 3 seeds reaching the high-reward end. The head-to-head ranking is descriptive.

There is no random or oracle normalization in the local comparison. Report all outcomes. Never convert these small local samples or timing telemetry into a qualification gate, scorecard, or promotion.

Attempt 1 stopped at preflight: the trial lifecycle string was rejected by the Prototype codec, and the differential calibration runner does not expose the default discounted SARSA arm. No completed life informed the retry. Attempt 2 uses the same predeclared comparisons and seeds, with a codec-valid lifecycle ID shared within seed and the #3652 runner for the discounted arms.

Before the first RiverSwim life was launched, the local RiverSwim schedule
was bounded to 5,000 accepted events per arm and seed to fit the shared
machine budget. The 20,000-event results come from the independently
recomputed retained campaigns. The local smoke keeps the same seeds,
initialization overrides, runner, control pairings, and raw metrics.
This amendment changes no Switching run.
