# Integration

HarnessMM's execution method comprises Executable Decision Graphs (EDG), Guarded Decision Reuse (GDR), and Certified Callback Substitution (CCS). The [method guide](METHOD.md) defines the paper terminology and the distinction between core execution control and companion configurations. The `chisel` namespace and `jev` API names remain stable implementation identifiers.

[controller_defaults.json](../configs/controller_defaults.json) contains explicit controller settings for integration examples. It is not a frozen benchmark configuration.

## Companion answer-policy configuration: QA generation and typed judgments

`interleave()` validates each draft against `DRAFT_SCHEMA` and accepts it only when all four entries in `DECISION_SCHEMA` are `pass`. `fail` and `unknown` permit one refinement. Invalid output, an absent source quotation, or unresolved judgments return to the original callback. A repeated failed draft stops the loop. The verifier sees the quotation and up to 160 characters on either side; it may miss contrary evidence elsewhere in the source.

`chisel.adapters.qa.solve()` preserves the original request-accounting implementation. Its inputs are:

- `item`: a mapping with `id`, `question`, and `context`.
- `arm`: `interleave`, `cot_only`, `original`, `no_skill`, `skillopt`, or `autosaddler`. Baseline arm names select supplied prompts; they do not install or reproduce those complete methods.
- `branch`: the caller's selected original profile, normally `delete` or `modify`.
- `prompts`: a mapping containing that frozen profile and any requested baseline prompts.
- `config`: an `interleave` mapping containing `maximum_drafts`, `draft_max_tokens`, `judge_max_tokens`, and the historical backend identifier `qwen_typed_decision_proxy`.
- `legacy`: an adapter providing `truncate_context(text)` and `build_user(item)` with the original harness's input semantics.
- `call`: a callable accepting messages plus `phase`, `key`, `maximum`, and `response_schema` keyword arguments.

The caller must return `raw_response.choices[0].message.content`, actual `usage` fields (`prompt_tokens`, `completion_tokens`, `total_tokens`), and a `call_file` receipt path. Preserve each call record before returning it. Never substitute estimated token counts for missing provider usage. The package retains the historical backend identifier for compatibility; the caller must separately record the actual model and revision.

The example adapter does not truncate source text. An official benchmark integration must retain that benchmark's own input policy. Use a model endpoint on an authorized server for any benchmark work.

## Public-feedback controllers

ALFWorld helpers take public observations and the environment's admissible action list. `RuleMemory` returns exact spans from the original skill. `progress_choices()` builds bounded choices from public progress; the native environment still executes and validates actions. Legacy Jev wire-format helpers are separate from the local-model selection helpers.

`RecoverySession` receives the original spreadsheet request and returns a request plus a decision receipt through `project()`. Preserve the canonical history outside the controller and pass the actual response to `observe_response()`. Deterministic followups are opt-in. A produced workbook is not proof of task correctness.

`TerminalJevSession` validates native `exec_in_sandbox` plans. Its `condition_target` distinguishes conditions checked on the preceding command from postconditions checked after each followup. Native execution timeouts and tool validation remain the executor's responsibility.

`PublicContinuation` for GAIA2 stores a model-produced plan and checks feedback before emitting the next action. It refuses planned final answers and user-facing messages. `Gaia2Engine` wraps a caller-supplied engine; it does not import or start the official simulator.

## Structured GAIA2 banks

`gaia2_structured_bank_v9.generate()` accepts a caller-owned provider with `config['model']['served_name']`, `endpoint()`, `transport(endpoint, payload, timeout=...)`, and `record(receipt)` attributes. Transport returns the native chat response with its actual model identity and complete token usage. The record callback retains every metered response, including a malformed bank envelope and any subsequent native fallback. Transport errors propagate; the adapter does not silently replay requests.

The response schema carries `reasoning`, `action`, `action_input`, and `jev_bank` together. The parser preserves the actual reasoning and current native action. `gaia2_cot_jev.public_state_candidates()` instantiates later candidates against public observations and optional receipt history. `gaia2_typed_continuation_bank` supports explicit field paths, public record iteration, and exact earlier-receipt references. An unresolved reference, ambiguous receipt, unsupported value, or candidate overflow defers to full reasoning. The caller must still apply the independent finite-choice judge and the native executor. `native_fast_execution_receipt()` checks the actual executed call and returned observation.

`gaia2_fast_public_contract_v12.ground_fast_messages()` adds original native tool descriptions and a returned call/observation pair to the judge input. A returned write call alone does not prove its intended effect. The structured-bank schema permits routine simulator writes; an integration must preserve its own authorization rules and native tool restrictions.

`gaia2_public_context_codec_v14.compact()` interns repeated long strings in a public-state dictionary; `restore()` verifies hashes and reconstructs the original values. `compact_request()` applies the dictionary only when the encoded user-state message is smaller. Include the module's `INSTRUCTION` in the author prompt whenever references are used. This reduces repeated bytes, without a guaranteed model-token reduction.

`gaia2_fused_bank_v16.project()` accepts the exact original messages, tool descriptions, and structured tool records. It compacts tool records only after a round trip and replaces exact duplicate earlier observation bodies by references to a retained copy. The latest message remains byte-for-byte intact. Its return value includes the projected request and an audit receipt. The benchmark-specific prompt installer is excluded.

## Native Responses/MCP boundary

`codex_native_app_bank.NativeBankBoundary` accepts tools, `judge`, `execute`, `generation`, `finished`, and `audit` callbacks. Register a slow response and its public task, then route the original native call through `call(name, arguments)`. The boundary validates the declared bank, sends finite candidates and actual public receipts to the metered judge, and dispatches selected actions through the same `execute` callback. Abstention, invalid decisions, public-state overflow, or an error returns control to the slow path. Record every judge response, including abstentions and invalid outputs.

This optional module invokes the bundled `contracts/codex_bank_contract.mjs` using `node` on `PATH`. Install Node.js 22 or later. It contains no provider client, credentials, SDK installation, or benchmark launcher. Its `fast_request()` helper retains the source's fixed model identifier and one-token contract; integrations using another model require an explicitly frozen, matched configuration.

## One-step public continuations

`supported_local_decision.execute_supported_decision()` receives immutable supported candidates and caller callbacks. An unsupported state, empty candidate set, invalid selection, or explicit deferral calls the original fallback once. Exceptions from selection, execution, or fallback propagate so the owner can reconcile any actual side effect or paid request.

`conditional_microplans` and `observed_feedback_plans` use an explicit decide/install/observe lifecycle. Supply the actual native action and feedback after every execution. A continuation cannot advance on a predicted observation or overwrite an outstanding action. Fragment contracts, residual programs, and grounded continuations accept caller-supplied training evidence with source-group exclusions. Keep evaluation labels and held-out trajectories outside the fitting inputs.

## Companion configuration: rule evolution

The `qa_meta_rsi` and `qa_single_module_rsi` modules accept explicit skill states, proposals, source hashes, and paired evidence. They preserve module boundaries, keep training support separate from held-out evaluation, and retain lineage. Their gate functions operate on supplied measurements; the release includes no learned skill or benchmark result.
