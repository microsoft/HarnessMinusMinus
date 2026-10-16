# AgentMinusMinus

Jev-inspired typed decisions, bounded recovery, and modular rule evolution for language-model agents.

CHISEL preserves an agent's original prompt and tool contract while controlling when to accept a compact decision, refine it, or return to the original reasoning path. The reusable controllers are provided as a small Python package with no runtime dependencies.

## Install

Python 3.10 or later:

```bash
git clone git@github.com:microsoft/AgentMinusMinus.git
cd AgentMinusMinus
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python tools/check_release.py
python -m pytest -q
```

The tests use synthetic inputs and mocked responses. They do not load benchmark datasets, invoke evaluators, contact providers, or require a GPU. Node.js 22 or later is needed for the optional Codex native-bank bridge and its tests. The Python controllers use only the standard library.

## Modules

| Area | Entry point | Behavior |
| --- | --- | --- |
| QA draft/check loop | `chisel.harness.jev_interleave` | Checks exact source quotations and four typed judgments; permits at most two drafts before fallback. |
| QA call accounting | `chisel.adapters.qa.solve` | Connects the loop to a caller-supplied provider and records tokens for every draft, judgment, refinement, and fallback. |
| QA recovery | `chisel.learning.qa_typed_recovery` | Uses source evidence and explicit keep/restore decisions. |
| Rule evolution | `chisel.learning.qa_meta_rsi`, `qa_single_module_rsi` | Applies bounded edits, preserves module lineage, and checks paired quality/cost evidence. |
| ALFWorld | `chisel.learning.alfworld_jev`, `alfworld_modular_rsi` | Selects admissible actions, retrieves source rules, and builds typed public-progress choices. |
| Spreadsheet tools | `chisel.sb2_protocol.jev_recovery` | Restores or compacts tool history after public stalls and proposes bounded deterministic followups. |
| Terminal tools | `chisel.harness.terminal_jev.TerminalJevSession` | Validates short tool plans and checks public execution conditions. |
| GAIA2 | `chisel.harness.gaia2_jev.PublicContinuation` | Gates planned tool continuations on public observations, errors, and refresh limits. |
| Structured action banks | `chisel.harness.gaia2_structured_bank_v9` | Serializes real reasoning, the current native action, and a later-action bank in one call; retains metered fallback costs. |
| Public receipt bindings | `chisel.harness.gaia2_typed_continuation_bank` | Resolves typed arguments against exact recorded public receipts and defers ambiguous bindings. |
| Context compression | `chisel.harness.gaia2_public_context_codec_v14` | Interns repeated public strings with a verified lossless round trip. |
| Native tool boundary | `chisel.harness.codex_native_app_bank.NativeBankBoundary` | Runs slow-authored candidates through a caller-supplied native executor and independently metered judge. Requires Node.js. |
| Supported candidates | `chisel.learning.supported_local_decision` | Commits a supported selection or calls the original fallback once; callback errors propagate. |
| Guarded continuations | `chisel.learning.conditional_microplans`, `observed_feedback_plans` | Checks actual feedback and native action membership before one continuation. |
| Learned fragment contracts | `chisel.learning.fragment_contracts`, `residual_programs`, `grounded_continuations` | Compiles public transition evidence and retrieves supported continuations with source-group exclusions. |
| Calibration | `chisel.learning.chisel.select_domain_policy` | Selects a predeclared policy from calibration evidence and reports admission failures. |

## Connect an agent

The controllers accept functions and public observations. You supply the model transport and native tool executor. The package does not choose or launch a model.

```python
from chisel.harness.jev_interleave import (
    interleave, generator_messages, verifier_messages,
)

def answer(question, context, generate_text, original_answer):
    return interleave(
        context=context,
        generate=lambda previous, decisions: generate_text(
            generator_messages(question, context, previous, decisions)
        ),
        judge=lambda draft: generate_text(
            verifier_messages(question, context, draft)
        ),
        fallback=original_answer,
        maximum_drafts=2,
    )
```

`generate_text` returns JSON text. `original_answer` is a zero-argument callback that executes the frozen original prompt. Provider failures propagate to the caller; they are not converted into successful decisions. For schemas, output limits, and complete token accounting, use the [QA adapter example](examples/qa_adapter.py) and [integration guide](docs/INTEGRATION.md).

## Scope

“Jev-inspired” refers to typed judgments, abstention, and bounded control flow. These controllers use ordinary model outputs or deterministic rules; they do not reproduce Jev's architecture or calibrated probabilities. No TypeSafe account or SDK is needed. Legacy ALFWorld protocol constructors remain for compatibility but perform no network calls.

Version 0.2.0 updates the reusable implementation through October 2026. Official benchmark harnesses, datasets, checkpoints, learned skill snapshots, cluster launchers, and experimental results are separate artifacts. A passing test suite establishes controller behavior. See [reproducibility requirements](docs/REPRODUCIBILITY.md), the [release notes](CHANGELOG.md), and [publication checks](SECURITY.md).

## Source and license

The controllers use the `chisel` Python namespace and retain their original protocol identifiers. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records the source commit, actual file hashes, destination paths, and extraction changes. See [the source map](docs/SOURCE_MAP.md).

Apache-2.0. Existing copyright notices are preserved in [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Repository license

The controller package is licensed under Apache-2.0. The original Microsoft repository template license is preserved in [LICENSE-MICROSOFT](LICENSE-MICROSOFT).

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
