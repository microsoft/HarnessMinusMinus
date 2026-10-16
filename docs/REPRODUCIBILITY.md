# Reproducibility

## This release

This is a source extraction, with an installable controller library and synthetic unit tests. Cluster-specific deployment, raw logs, benchmark data, credentials, model checkpoints, and experiment results were excluded. `SOURCE_MANIFEST.json` identifies the actual source bytes, including working-tree changes; the commit alone is not an exact snapshot of those bytes.

The Python package has no third-party runtime dependencies. The optional native bank boundary requires Node.js 22 or later to run its bundled JavaScript contracts. Test dependencies are pinned in `pyproject.toml`. Original internal schema strings are retained so that controller receipts keep their historical identities. Public Python import paths use `chisel`.

## Running a study

Run benchmarks and experiments only on an authorized server. Prefer an available, suitable H200. Before a GPU run, inspect GPU state, choose or validate a physical device, set `CUDA_VISIBLE_DEVICES`, and save the choice.

Create a new experiment directory. Freeze the configuration, command, Git state, hostname, package versions, seeds, exact model identity and revision, dataset revision, official evaluator revision, and selected GPU. Retain logs, provider receipts, outputs, and failures. Include all generation, verification, refinement, retry, and fallback costs; unknown usage must remain unknown.

Reconstruct each official harness separately using its public setup instructions and applicable licenses. Preserve native tools, splits, metrics, evaluation rules, and original skill/input contracts. This release does not provide the external artifacts needed for a complete historical campaign replay.

Split independent questions, tasks, scenarios, or groups into 20% training, 20% validation, and 60% test before optimization or repetitions. Freeze seed, membership, and deterministic rounding when the population is indivisible. Training supplies revisions; validation selects among training-frozen candidates; freeze the selected configuration before test. If the official benchmark mandates another split, record the conflict and obtain a protocol decision before running a custom partition.

Use matched comparisons and ablations with the same native tools, scoring, model identity, execution count, and limits. Record exact input/output/total tokens and the complete execution denominator, including author, control, fallback, and failed-execution costs. Report consumption as `Tokens/task (k)` with two decimal places: total measured actor input plus output tokens divided by all matched executions and by 1,000. Keep unknown costs explicit. Synthetic tests and transport checks establish implementation behavior; benchmark claims require their frozen evidence.

## Known limits

Typed validity does not establish factual correctness. A generator and verifier using the same model can share errors. Local source windows can omit evidence. Deterministic continuation can only use conditions encoded in the public state. The original reasoning path and native environment remain necessary parts of each integration.
