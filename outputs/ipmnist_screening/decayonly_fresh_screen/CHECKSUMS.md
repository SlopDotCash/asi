# Checksums
- `SHA256SUMS` is unchanged and covers the original pretty-printed shards as committed at 5b341326. To check it, restore those bytes with `git show 5b341326:<path>` and run `sha256sum -c SHA256SUMS`.
- The shards at HEAD are re-serialized as compact JSON (one line each) so the PR diff fits GitHub's 20,000-line limit. Each parses to JSON identical to the original. `SHA256SUMS.compact` covers the compacted files.
- `analyze.py` and `envcheck.py` output is byte-identical before and after the re-serialization.

The original manifests also list ignored run logs that were never committed.
A clean checkout can verify all 44 retained files with
`sha256sum -c SHA256SUMS.tracked` from this directory.
