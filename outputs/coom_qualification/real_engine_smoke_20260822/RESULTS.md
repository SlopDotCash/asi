# COOM real-engine qualification smoke — 2026-08-22

This directory is an append-only, permanently nonpromoting development record.
It is not scientific evidence, a COOM performance result, paper parity,
authenticated execution attestation, or a learner comparison.

The pinned COOM/ViZDoom CO8 fixed-action smoke ran from ASI source
`c9aba7b54dedd647f8bd5f5c7bf6780b1413b676` after fixing two execution blockers
found by the retained protocol:

- the documented execute/write-only host output directory could not be traversed
  with `O_RDONLY`; directory traversal now uses Linux `O_PATH`;
- zero-capability publication could not use `linkat(AT_EMPTY_PATH)`, and `fsync`
  cannot flush an `O_PATH` directory descriptor; publication now uses the
  documented `/proc/self/fd` link fallback and `syncfs` on the receipt inode.

ViZDoom also requires a writable working directory for its process-local
`_vizdoom.ini` and `_vizdoom/` files. The run therefore used the already-bounded
`/tmp` tmpfs as its working directory while keeping the source and root
filesystem read-only.

The independently validated receipt records:

- trace SHA-256: `c74968494ccebaaeac4bc1e0c0f1db7546ac5091b831c05a4c0c727266da696f`;
- 8 task resets and 16 environment steps;
- 0 policy queries, learner updates, and model queries;
- UID/GID 65532, zero effective capabilities, and `NoNewPrivs`;
- the pinned 33-asset, 4,153,440-byte source manifest; and
- `scientific_promotion_allowed: false`.

Timing is telemetry-only. The complete machine-validated payload is
`receipt.v1.json`; its file SHA-256 at retention was
`ca19bb23b1bed07b8ad77d7d422c7f8e02aa2194d107b5ed3040d8c3bcaa60ed`.

The broader #1582 acceptance criteria remain open: this smoke does not qualify
the CD/CO/COC/MIXED protocols, TensorFlow/SAC baseline, matched learner/control
campaign, full resource accounting, or performance metrics.
