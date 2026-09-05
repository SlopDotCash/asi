# COOM isolated qualification runtime

This directory builds a source- and dependency-pinned Linux runtime for one
bounded, fixed-action COOM/ViZDoom CO8 smoke. It does not install COOM into the
ASI environment and does not run a learner, compute benchmark metrics, reproduce
the paper, or authorize promotion.

The runtime uses a digest-pinned Python base, fetches the official COOM commit,
verifies the source archive, keeps the upstream MIT license, verifies all 33
WAD/config assets at execution, and installs only a hash-locked environment
subset. The one-line patch replaces an
undeclared legacy `gym.RewardWrapper` import with the declared
`gymnasium.RewardWrapper`; no environment logic or asset is changed.

`qualification-manifest.json` binds the base image, Dockerfile, dependency
lock, patch, and smoke-validator bytes. Before emitting a receipt, the smoke
reconstructs the upstream Git tree (reversing exactly that one import patch),
rehashes the license and all assets, enforces the locked package versions, and
strictly validates the ordered task/step trace, safe empty info subsets,
resources, claims, and independently repeated trace golden.

Build and execute from this directory:

```bash
docker build --tag asi-coom-qualification:development .
receipt_dir="$(mktemp -d /tmp/asi-coom-receipts.XXXXXX)"
chmod 0733 "$receipt_dir"
docker run --rm --network none --read-only --user 65532:65532 \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 2 --memory 2g --pids-limit 64 \
  --workdir /tmp \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=16m \
  -v "$receipt_dir:/output" asi-coom-qualification:development \
  --output /output/receipt.json
chmod 0755 "$receipt_dir"
docker run --rm --network none --read-only --user 65532:65532 \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 2 --memory 2g --pids-limit 64 \
  --workdir /tmp \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=16m \
  -v "$receipt_dir:/input:ro" asi-coom-qualification:development \
  --validate-receipt /input/receipt.json
chmod 0700 "$receipt_dir"
```

The host receipt directory above is explicitly outside the source checkout.
The writable tmpfs working directory is required because ViZDoom creates its
process-local `_vizdoom.ini` and `_vizdoom/` files in the current directory;
the source tree and container root remain read-only.
The verifier refuses to start the engine unless it is UID/GID 65532, has no
effective Linux capabilities, has `NoNewPrivs` set, and sees exactly the eleven
reviewed Python distributions. The command also makes network, root filesystem,
CPU, memory, process, and temporary-filesystem bounds explicit. The receipt does
not claim that these caller-supplied limits are authenticated execution evidence.

Compare `trace_sha256`, not the telemetry-only elapsed time or platform string.
The smoke consumes seed 1582000, all eight official CO8 tasks, and two action-0
steps per task. A matching trace is a deterministic runtime qualification check,
not authenticated execution attestation or a COOM result. Receipts are not
retained automatically; `--output` publishes one read-only file atomically and
refuses replacement, while `--validate-receipt` strictly reloads it inside the
bound image without starting COOM. Failures still exit nonzero without a
structured negative-outcome receipt. Any repository retention belongs in a
separately reviewed, append-only output namespace.
