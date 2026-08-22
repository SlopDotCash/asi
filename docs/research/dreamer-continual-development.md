# Dreamer-family continual development lane

The family anchor is Hafner et al., *Mastering Diverse Control Tasks Through World Models*,
Nature (2025), DOI `10.1038/s41586-025-08744-2`, originating as arXiv `2301.04104`. The audited
official implementation is `danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe`
(`main`, inspected 17 August 2026). The continual comparator is Kessler et al., *The Effectiveness
of World Models for Continual Reinforcement Learning*, CoLLAs 2023, PMLR 232:184–204, arXiv
`2211.15944`. It studies selective replay and continual exploration on MiniGrid and MiniHack. No
stable official Continual-Dreamer repository revision was identified from the paper landing page,
so source parity remains an explicit blocker rather than an invented pin.

`asi.dreamer_continual_development.v1` is a minimal ASI-native mechanism qualification. A current
action-conditioned world model learns from real transitions; a bounded replay ring supplies anchors;
guarded one-step imagined transitions update persistent action values that choose subsequent real
actions. Frozen development seeds, a recurring three-task sequence, per-task horizon, replay capacity,
and imagination ratio bind online imagination, a causal imagination-off twin, and a privileged live
task control excluded from candidate comparison. Receipts cover real steps, model updates/queries,
replay inserts/samples/peak bytes, imagination proposals/accepts/value updates, exact logical call
units, persistent array bytes, and telemetry-only elapsed nanoseconds. Negative results are retained.

This is not Dreamer, DreamerV3, or Continual-Dreamer reproduction. It lacks an RSSM, categorical
latents, encoder/decoder, KL balancing/free bits, symlog/two-hot objectives, return normalization,
lambda returns, learned actor and critic, backpropagation through multi-step imagination, target
networks, exploration policy, sequence replay, pixels, MiniGrid/MiniHack, paper task schedules,
selective-replay variants, evaluation matrices, published compute budgets, and official checkpoint
parity. Before comparison, locate and pin the Continual-Dreamer source snapshot, implement those
components, match environment/dependency versions and all information boundaries, validate hardware
accounting, reproduce official single-task controls, and freeze untouched scientific seeds.

## External DreamerV3 source qualification

The read-only `dreamerv3_external_qualification` record binds the official DreamerV3 commit and
Git tree, the observed 6,312,430-byte source archive and its SHA-256 digest, the MIT license bytes,
and the official requirements, configuration, and Dockerfile bytes. The prospective first slice is
the official `dmc_proprio` configuration combined with `debug`, which uses proprioceptive state and
the smallest published model overlay.

This closes only the external-source availability and license-review gate. The official Dockerfile
uses mutable apt/PPA inputs and executes a mutable gist; most Python dependencies are not pinned;
the requirements select CUDA; and the dm_control/MuJoCo runtime and assets are not content-closed.
No runtime was built, no environment or workload was executed, and no parity, performance,
execution-attestation, development-result, or scientific claim is made.
