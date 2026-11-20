# AgentMinusMinus

Code for **ReHO: Optimizing Agent Execution Control Flow through Guarded Decision Reuse**.

ReHO (Reuse-based Harness Optimization) optimizes the execution control flow of an existing agent harness while actor weights remain fixed. A controller at the model-tool boundary reuses resolved decisions to replace later reasoning callbacks. New semantic decisions, unsupported evidence, and selector abstention invoke the original actor again.

## Method

The method has three components, using the terminology of the paper:

| Component | Role |
| --- | --- |
| **Executable Decision Graphs (EDG)** | Store exact native actions, complete literal arguments, causal dependencies, and public guards authored by slow reasoning. |
| **Guarded Decision Reuse (GDR)** | Retain the graph across tool transitions and activate eligible decisions from observed public receipts. A typed selector returns an eligible candidate ID or abstains; a unique eligible node can execute deterministically. |
| **Certified Callback Substitution (CCS)** | Execute an admitted action through the native interface and bind its actual receipt to omission of the registered slow callback. |

Slow reasoning resolves the current action and future decisions. EDG compiles those decisions into executable nodes. GDR checks predecessor completion, public guards, and literal argument closure. CCS executes the admitted node in place of the later slow callback. Each execution receipt updates the same graph; new inference returns control to the slow actor.

A substitution certificate binds the graph hash, decision ID, predecessor receipt, exact arguments, tool receipt, and bypassed callback ID. Certificates establish native execution and callback omission. Native scoring measures task correctness; complete-task totals measure cost. The paper calls the overhead of adding selection without eliminating downstream deliberation the **Substitution Paradox**.

Prompt editing, context compaction, instruction optimization, and finalization rules are companion configurations. They are developed on training data, selected before evaluation, and held fixed during execution. The runtime selector chooses an eligible stored action or abstains.

## Results reported in the paper

The following complete HumanEval comparison uses Qwen3.5-4B on all 98 test tasks:

| Harness | Success (%) | Tokens/task (k) |
| --- | ---: | ---: |
| DeepSeek Harness | 71.43 | 96.28 |
| DeepSeek Harness + ReHO | 85.71 | 11.62 |

This integrated configuration combines decision reuse with fixed context and finalization settings. The separate matched MBPP-sanitized ablation holds authoring and context settings common on all 257 test tasks:

| Condition | Success (%) | Tokens/task (k) |
| --- | ---: | ---: |
| Full ReHO | 93.77 | 13.17 |
| W/o GDR (single-use EDG) | 93.00 | 18.03 |
| W/o CCS (callbacks retained) | 93.39 | 16.37 |
| EDG execution disabled | 95.33 | 16.19 |

Full ReHO reduces complete cost by 26.91% relative to single-use EDGs and by 19.52% relative to retained callbacks. Disabling EDG execution solves four more tasks at higher cost. These ablations test graph persistence and callback removal; the certificate's independent benefit is not ablated.

Token cost includes actor input plus output for authorship, selector calls, added context, remaining slow calls, fallback, recovery, finalization, and failed paid executions. Cached input remains counted. Development costs are reported separately.

## Install

Python 3.10 or later:

```bash
git clone git@github.com:microsoft/HarnessMinusMinus.git
cd HarnessMinusMinus/legacy/reho
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python tools/check_release.py
python -m pytest -q
```

The tests use synthetic inputs and mocked responses. They do not load benchmark datasets, invoke evaluators, contact providers, or require a GPU. Node.js 22 or later is needed for the optional Codex native-bank bridge and its tests. The Python controllers use only the standard library.

## Repository interfaces

The Python namespace remains `chisel`, and the distribution name remains `chisel-jev` for compatibility. Method descriptions use ReHO, EDG, GDR, and CCS as in the paper.

| Area | Entry points | Role in the repository |
| --- | --- | --- |
| Native action-bank execution | `chisel.harness.codex_native_app_bank`, bundled `contracts/cot_jev.mjs` | Slow-authored action banks, public receipts, finite-choice selection, native dispatch, and execution records. |
| Shared commit-or-defer control | `chisel.learning.supported_local_decision` | Candidate support and membership checks, one native commitment or original actor fallback. |
| Public-state adapters | `chisel.harness.gaia2_cot_jev`, `gaia2_typed_continuation_bank`, `terminal_jev` | Native tool contracts, public receipt bindings, and domain-specific continuation checks. |
| Context configurations | `chisel.harness.gaia2_public_context_codec_v14`, `gaia2_fused_bank_v16` | Public-context compaction used as a companion intervention. |
| Instruction and answer-policy studies | `chisel.adapters.qa`, `chisel.learning.qa_typed_recovery`, `alfworld_modular_rsi`, `qa_meta_rsi` | Companion configurations with their own intervention and evaluation scope. |
| Additional continuation adapters | `chisel.learning.conditional_microplans`, `observed_feedback_plans`, `fragment_contracts`, `residual_programs`, `grounded_continuations` | Public-feedback continuation and source-supported retrieval interfaces. |

The release contains reusable controllers and synthetic tests. Frozen benchmark integrations, datasets, learned snapshots, experiment receipts, and result artifacts are separate. The interface map does not substitute for the paper's frozen evaluation configurations. See [method and terminology](docs/METHOD.md), [integration](docs/INTEGRATION.md), and [reproducibility](docs/REPRODUCIBILITY.md).

## Connect an agent

Supply the original actor callback, a public-state adapter, and the native action dispatcher. Record actual tool feedback after each execution. The shared controller can commit a supported literal action or defer once:

```python
from chisel.learning.supported_local_decision import (
    SupportedCandidates, execute_supported_decision,
)

result = execute_supported_decision(
    SupportedCandidates(True, ("make test",), "Observed predecessor return code is zero"),
    commit=native_execute,
    fallback=original_actor,
)
```

Here `native_execute` and `original_actor` are caller-supplied callbacks, and the adapter supplies the actual support evidence. The complete EDG integration also retains graph dependencies, checks public guards, registers the slow boundary, and records receipt-bound substitution certificates. A generic support trace alone is not a CCS certificate.

## Source and license

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records source and release hashes, destination paths, and extraction changes. [The source map](docs/SOURCE_MAP.md) identifies the released modules. Protocol identifiers remain unchanged.

The controller package is licensed under [Apache-2.0](LICENSE). Existing copyright notices are preserved in [LICENSE](LICENSE) and [NOTICE](NOTICE). The original Microsoft repository template license is preserved in [LICENSE-MICROSOFT](LICENSE-MICROSOFT).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
