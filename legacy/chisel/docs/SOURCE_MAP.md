# Source map

The manifest records the original source bytes, public destination, and release bytes for every extracted file. Paths are relative to the source repository. Public imports use `chisel`; wire protocol identifiers remain unchanged.

Cluster adapters and campaign launchers are excluded. `gaia2_cot_jev` omits its deployment installer; `gaia2_dual_rsi`, `gaia2_public_literal_table`, and `gaia2_usage` contain the selected pure helpers listed in the manifest. The structured bank uses these helpers directly. The native bank bridge bundles its JavaScript contracts as package data.

| Original module | Public module |
| --- | --- |
| `scripts/jev_searchqa.py` | `chisel/adapters/qa.py` |
| `phiagent/harness/codex_native_app_bank.py` | `chisel/harness/codex_native_app_bank.py` |
| `integrations/deepseek_harness/codex_bank_contract.mjs` | `chisel/harness/contracts/codex_bank_contract.mjs` |
| `integrations/deepseek_harness/cot_jev.mjs` | `chisel/harness/contracts/cot_jev.mjs` |
| `phiagent/harness/gaia2_cot_jev.py` | `chisel/harness/gaia2_cot_jev.py` |
| `phiagent/harness/gaia2_dual_rsi.py` | `chisel/harness/gaia2_dual_rsi.py` |
| `phiagent/harness/gaia2_fast_public_contract_v12.py` | `chisel/harness/gaia2_fast_public_contract_v12.py` |
| `phiagent/harness/gaia2_jev.py` | `chisel/harness/gaia2_jev.py` |
| `phiagent/harness/gaia2_public_context_codec_v14.py` | `chisel/harness/gaia2_public_context_codec_v14.py` |
| `phiagent/harness/gaia2_public_literal_table.py` | `chisel/harness/gaia2_public_literal_table.py` |
| `phiagent/harness/gaia2_public_progress.py` | `chisel/harness/gaia2_public_progress.py` |
| `phiagent/harness/gaia2_state_progress.py` | `chisel/harness/gaia2_state_progress.py` |
| `phiagent/harness/gaia2_structured_bank_v9.py` | `chisel/harness/gaia2_structured_bank_v9.py` |
| `phiagent/harness/gaia2_typed_continuation_bank.py` | `chisel/harness/gaia2_typed_continuation_bank.py` |
| `scripts/run_gaia2_jev.py` | `chisel/harness/gaia2_usage.py` |
| `phiagent/harness/jev_interleave.py` | `chisel/harness/jev_interleave.py` |
| `scripts/terminal_jev.py` | `chisel/harness/terminal_jev.py` |
| `phiagent/harness/typed_decisions.py` | `chisel/harness/typed_decisions.py` |
| `phiagent/learning/alfworld_autosaddler_pfn.py` | `chisel/learning/alfworld_autosaddler_pfn.py` |
| `phiagent/learning/alfworld_jev.py` | `chisel/learning/alfworld_jev.py` |
| `phiagent/learning/alfworld_modular_rsi.py` | `chisel/learning/alfworld_modular_rsi.py` |
| `scripts/alfworld_public_progress.py` | `chisel/learning/alfworld_public_progress.py` |
| `phiagent/learning/chisel.py` | `chisel/learning/chisel.py` |
| `phiagent/learning/compact_qa_contracts.py` | `chisel/learning/compact_qa_contracts.py` |
| `phiagent/learning/conditional_microplans.py` | `chisel/learning/conditional_microplans.py` |
| `phiagent/learning/counterfactual_invocation.py` | `chisel/learning/counterfactual_invocation.py` |
| `phiagent/learning/counterfactual_repair.py` | `chisel/learning/counterfactual_repair.py` |
| `phiagent/learning/entry_guard_refinement.py` | `chisel/learning/entry_guard_refinement.py` |
| `phiagent/learning/fragment_contracts.py` | `chisel/learning/fragment_contracts.py` |
| `phiagent/learning/grounded_continuations.py` | `chisel/learning/grounded_continuations.py` |
| `phiagent/learning/observed_feedback_plans.py` | `chisel/learning/observed_feedback_plans.py` |
| `phiagent/learning/public_evidence_retention.py` | `chisel/learning/public_evidence_retention.py` |
| `phiagent/learning/public_recurrence.py` | `chisel/learning/public_recurrence.py` |
| `phiagent/learning/qa_context_evidence.py` | `chisel/learning/qa_context_evidence.py` |
| `phiagent/learning/qa_meta_rsi.py` | `chisel/learning/qa_meta_rsi.py` |
| `phiagent/learning/qa_rule_compaction.py` | `chisel/learning/qa_rule_compaction.py` |
| `phiagent/learning/qa_rule_module_interventions.py` | `chisel/learning/qa_rule_module_interventions.py` |
| `phiagent/learning/qa_single_module_rsi.py` | `chisel/learning/qa_single_module_rsi.py` |
| `phiagent/learning/qa_typed_recovery.py` | `chisel/learning/qa_typed_recovery.py` |
| `phiagent/learning/residual_programs.py` | `chisel/learning/residual_programs.py` |
| `phiagent/learning/sparse_retrieval.py` | `chisel/learning/sparse_retrieval.py` |
| `phiagent/learning/stage_aware_adm.py` | `chisel/learning/stage_aware_adm.py` |
| `phiagent/learning/supported_local_decision.py` | `chisel/learning/supported_local_decision.py` |
| `phiagent/sb2_protocol/jev_recovery.py` | `chisel/sb2_protocol/jev_recovery.py` |
| `phiagent/sb2_protocol/spec.py` | `chisel/sb2_protocol/spec.py` |
| `phiagent/harness/gaia2_fused_bank_v16.py` | `chisel/harness/gaia2_fused_bank_v16.py` |
