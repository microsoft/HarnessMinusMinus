# Release validation

The repository exposes reusable controllers and adapters for HarnessMM. Its method definitions follow [METHOD.md](METHOD.md); companion configurations retain separate attribution.

Run validation from the repository root:

```bash
python tools/check_release.py --history
python -m pytest -q
python -m build
python tools/check_release.py --distributions dist
```

The current synthetic suite contains 2,020 tests. The source manifest binds 69 extracted files to their source and release hashes. Package initializers, documentation, and publication tooling are additional repository files. The optional native action-bank bridge requires Node.js 22 or later.

Publication checks cover current source, reachable Git blobs, and built distributions. Model weights, datasets, private configuration, learned snapshots, raw receipts, and cluster orchestration are separate assets. Synthetic test results validate controller behavior; the scientific measurements in the README come from the manuscript's matched benchmark evaluations.
