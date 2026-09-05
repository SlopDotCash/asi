# Prospective Avalanche native-suite runtime

This directory is a reviewed-before-execution plan candidate for issue `#1578`.
It pins the official Avalanche repository at
`eb075be393e1f458b2c352514ff6c17b5a2c0f4e`, its MIT license and relevant
scenario-source bytes, one Linux/x86-64 base image, a CPU-only Torch pair, and
the complete hash-locked resolution of the source revision's declared runtime
dependencies.

The digest-pinned base contributes `pip==23.0.1` and `wheel==0.44.0`; the lock
upgrades `setuptools` to `84.0.0`. GPUtil 1.4.0 is the sole source-build
exception because it publishes no wheel. Its exact 5,545-byte PyPI source
archive is hash-pinned; it declares MIT in package metadata but omits the
license text. All other resolved dependencies must install as wheels. This is
an exact dependency-byte resolution, not a complete transitive license audit.

The image is intentionally data-free. Its entry point only verifies source,
license, lock, package, platform, and scenario-constructor identities. It does
not construct a scenario because Avalanche constructors download MNIST or
CIFAR-100 when the data is absent. It does not execute ASI's native runner.

The plan is explicitly prospective:

- `runtime_build_verified=false`;
- `bit_reproducible_image_claimed=false`;
- `external_execution_authorized=false`;
- `workload_executed=false` and `receipt_created=false`;
- no Avalanche parity, benchmark result, negative outcome, or scientific
  evidence is claimed.

Do not build or invoke this runtime until the plan and dependency surface are
independently reviewed. A later authorized qualification must build the exact
image, capture its immutable image digest, run with networking disabled and a
read-only root, mount separately approved exact dataset bytes, and publish a
strict success or failure receipt at a NEW create-only path outside the
repository. The ten exact blockers in `qualification-plan.json` remain
load-bearing.

That future invocation must also freeze the exact tmpfs path/bytes/mode/owner,
CPU affinity and quota, memory and swap, PID, and host wall-clock caps; drop all
capabilities; set `no-new-privileges`; expose no host devices; mount datasets
read-only; and make the create-only output the sole writable bind mount. The
Dockerfile pins build inputs, but repeated OCI builds are not claimed to be
bit-for-bit identical. A successful review must capture and subsequently
launch the exact built image ID.

The pinned source calls itself `0.6.0a`, although the repository commit is later
than the `0.6.0` PyPI release. Upstream leaves most dependency versions open;
the lock therefore defines a qualification compatibility runtime rather than an
official default environment. Split/rotation membership and transform parity,
dataset archive identities, task-information policy, mechanisms and matched
controls all remain unresolved. This runtime is not completion of #1578.
