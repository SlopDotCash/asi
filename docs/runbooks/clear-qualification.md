# CLEAR qualification lane

This is a setup and accounting lane for CLEAR real-world continual imagery. It
does not download the dataset, train a model, report a score, authorize a run,
or create scientific evidence. Every result is development-only, permanently
nonpromoting, and retained when negative.

## Pinned research surface

- Paper: `arXiv:2201.06289v3` (9 June 2022), which supersedes v1 and v2.
- Official curation code: `linzhiqiu/continual-learning` at
  `620cab4a7d99921fde73b67b53879470533cb39a`.
- Authors' classification reference: `ElvishElvis/CLEAR-Continual_Learning_Benchmark`
  at `75d5d2e7d412a787e0decf0417a4868c56691252`.
- Avalanche adapter and metric implementation: `ContinualAI/avalanche` at
  `eb075be393e1f458b2c352514ff6c17b5a2c0f4e`.
- Official project site source: `clear-benchmark/clear-benchmark.github.io` at
  commit `fa2f22e6bcaa47d4512acbcf4ec1643ad9dc66b2`, tree
  `19c7f8a68d867dab7e6b7236227797773873eec7`.
- Public, ungated provider repository:
  `elvishelvis6/CLEAR-Continual_Learning_Benchmark` at revision
  `b518a845a98f1c913497ab98a19727cb65b74e65`.

CLEAR derives natural temporal buckets from YFCC100M imagery spanning
2004–2014. Bucket 0 is the optional unlabeled/pretraining bucket; the selected
CLEAR-100 supervised lane uses labeled buckets 1–10. The selected protocol is
streaming: the accuracy matrix is used to measure the near future, including
the superdiagonal `next_domain` metric. It is not the alternate within-bucket
70:30 IID protocol.

The project site declares CC BY 4.0, and the provider repository carries the
`license:cc-by-4.0` tag. That does not establish that every underlying
Flickr/YFCC asset remains redistributable or available: privacy, publicity,
moral-rights, takedown, and approved-storage review remain explicit blockers.
The revision above exposes exact Git LFS identities for the selected archives:

- `clear100-train-image-only.zip`: 3,289,951,359 bytes, SHA-256
  `0376b952674e6ef55c3923ee4ce61e5b299fea4e29bbc4780530636e8988fd72`.
- `clear100-test.zip`: 1,640,361,665 bytes, SHA-256
  `c939753be4e62dc7732347e5e636ea599022c82f45443ea9e7166167e467abd0`.

These are provider metadata identities, not local download verification or
proof of archive contents and split semantics. The existing v1 local manifest
field remains `provider_archive_checksums_published: false` because changing
that historical schema would silently change its meaning; a separate
prospective asset-plan schema records the newly reviewed provider identities.

## Frozen development comparison

The adapter records the official Avalanche example's ResNet-18-from-scratch
control: 224-pixel crops, ImageNet normalization, SGD at 0.01 with momentum
0.9 and weight decay 1e-5, batch 256, 100 epochs per bucket, and a step
scheduler every 30 epochs with gamma 0.1. Seeds 0–4 are ASI training RNG roots;
they are not CLEAR IID split seeds. Each control axis is paired with an exact
mechanism-off reduction. There is deliberately no mechanism-on implementation
in this issue.

The plan computes training observations, optimizer updates, model queries,
archive bytes, and zero environment steps from the verified manifest. A future
runner must additionally receipt exact persistent parameter/optimizer/replay
bytes; timing stays telemetry-only. Metrics are the five official matrix
summaries: accuracy, in-domain, next-domain, forward transfer, and backward
transfer.

## Local manifest and CLI

Print the content-closed prospective source/asset freeze without reading a
dataset manifest or archive, accessing the network, or executing a workload:

```bash
.venv/bin/asi-clear-qualification --prospective-assets
```

This output is a plan-only prerequisite record. It is not a download receipt,
runtime qualification, execution authorization, run result, parity result, or
authenticated attestation.

Create a JSON file with this exact shape and locally computed byte identities:

```json
{
  "schema_version": "asi.clear.qualification.v1",
  "dataset": "clear100",
  "protocol": "streaming-near-future",
  "buckets": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "years": [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014],
  "samples_per_bucket": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  "archives": [{
    "role": "locally-acquired-clear100",
    "path": "clear100.zip",
    "size_bytes": 123,
    "sha256": "<64 lowercase hex characters>"
  }],
  "provider_archive_checksums_published": false
}
```

The sample counts above are placeholders and must be replaced from the local
prepared metadata. Run only after the data-use/storage review:

```bash
.venv/bin/asi-clear-qualification manifest.json --dataset-root /approved/clear
```

The command reads and hashes local regular files below the root and prints a
plan to stdout. It rejects symlinks, traversal, extra fields, malformed counts,
hash drift, and oversized manifests. It never extracts or writes data.

## Remaining comparability gates

- Review YFCC/Flickr asset terms, privacy/publicity/moral-rights limits,
  takedown behavior, and approved storage; obtain explicit acquisition
  authorization.
- Download both revision-pinned archives only into approved storage, then
  independently verify their exact local sizes and SHA-256 values.
- Parse the prepared metadata and independently verify class and bucket counts;
  provider metadata identity alone does not prove semantic split parity.
- Qualify the exact runtime and implement a reviewed runner outside the #1578
  native-suite adapter, then add image, label, transform, metric, JIT/parity,
  and end-to-end tests before seeking separate execution authorization.
- Receipt exact model, optimizer, accelerator, preprocessing, and optional
  bucket-0 pretraining costs. The selected control excludes bucket-0
  pretraining; any use needs a separately matched no-pretraining ablation.
- Freeze fresh scientific seeds only after development selection. Nothing in
  this lane can promote a claim or establish SOTA.
