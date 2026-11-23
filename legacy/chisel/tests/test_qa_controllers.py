"""Extracted synthetic controller unit tests; no benchmark or provider calls."""
import copy
import hashlib
import json
import math
from types import SimpleNamespace
import unittest
import unittest.mock
from unittest.mock import patch

from chisel.learning import compact_qa_contracts as contracts

from chisel.learning.qa_context_evidence import (
    ContextSelection,
    pack_duplicate_documents,
    reconstruct_context,
    select_context_for_request_budget,
    verify_duplicate_documents,
)

class QuestionAnchoredRecoveryTests(unittest.TestCase):
    def test_requires_question_clues_in_two_answer_bearing_source_groups(self):
        from chisel.learning.qa_typed_recovery import KEEP, anchored_response_state

        source = "[DOC] Nora Vale wrote Amber Harbor. [DOC] Amber Harbor was written by Nora Vale; Nora Vale wrote it."
        state = anchored_response_state("Who wrote Amber Harbor?", source, "<answer>Nora Vale</answer>")
        self.assertTrue(state["supported"])
        self.assertEqual(state["allowed_actions"][0], KEEP)
        self.assertEqual(len(state["evidence_guard"]["source_groups"]), 2)
        self.assertFalse(state["correctness_certified"])
        self.assertFalse(state["evidence_guard"]["source_groups_are_independent_publishers"])
    def test_rejects_name_mentioned_without_the_requested_relation(self):
        from chisel.learning.qa_typed_recovery import RESTORE, anchored_response_state, response_state

        source = "[DOC] Nora Vale wrote Amber Harbor. Mira Stone painted the cover. [DOC] Mira Stone designed the cover."
        response = "<answer>Mira Stone</answer>"
        self.assertTrue(response_state("Who wrote Amber Harbor?", source, response)["supported"])
        state = anchored_response_state("Who wrote Amber Harbor?", source, response)
        self.assertFalse(state["supported"])
        self.assertEqual(state["allowed_actions"], [RESTORE])
    def test_repeated_mentions_in_one_document_do_not_count_as_corroboration(self):
        from chisel.learning.qa_typed_recovery import anchored_response_state

        source = "[DOC] Nora Vale wrote Amber Harbor. Nora Vale wrote Amber Harbor."
        state = anchored_response_state("Who wrote Amber Harbor?", source, "<answer>Nora Vale</answer>")
        self.assertFalse(state["supported"])
        self.assertEqual(len(state["evidence_guard"]["source_groups"]), 1)
    def test_numeric_qualifier_and_original_invalid_contract_remain_guards(self):
        from chisel.learning.qa_typed_recovery import anchored_response_state

        source = "[DOC] Nora Vale won Amber Award in 2017. [DOC] Nora Vale won Amber Award in 2017."
        state = anchored_response_state("Who won Amber Award in 2018?", source, "<answer>Nora Vale</answer>")
        self.assertFalse(state["supported"])
        invalid = anchored_response_state("Who won Amber Award?", source, "Nora Vale")
        self.assertEqual(invalid["reason"], "invalid_answer_contract")

class TypedRecoveryTests(unittest.TestCase):
    def test_source_rule_retrieval_preserves_normalization_and_source_bytes(self):
        from chisel.learning import qa_meta_rsi as patches
        from chisel.learning.qa_typed_recovery import retrieve_source_rules

        delete, modify = ChiselCompressionTests().sources()
        original = patches.initial_skill({"delete": delete, "modify": modify})
        snapshot = copy.deepcopy(original)
        for profile in ("delete", "modify"):
            text, receipt = retrieve_source_rules(original, profile=profile, question="Who wrote this work?",
                                                   context="[TLE] Synthetic [PAR] Nora wrote the work.")
            self.assertEqual(text.split("## Answer Contract")[1], patches.render_skill(original)[profile].split("## Answer Contract")[1])
            for identifier, rule in original["rules"].items():
                if identifier.startswith(profile + ".normalization."):
                    self.assertIn(identifier, receipt["selected_rule_ids"])
                    self.assertIn(rule, text)
            self.assertTrue(all(rule in patches.render_skill(original)[profile] for rule in receipt["rules"].values()))
            self.assertFalse(receipt["selection_uses_labels"])
            self.assertFalse(receipt["rule_text_rewritten"])
        self.assertEqual(original, snapshot)
    def test_deterministic_public_support_does_not_certify_semantic_correctness(self):
        from chisel.learning.qa_typed_recovery import KEEP, RESTORE, deterministic_recovery, response_state

        state = response_state("Who wrote the book?", "Nora wrote the book. Mira painted its cover.",
                               "<answer>Mira</answer>")
        decision = deterministic_recovery(state)
        self.assertEqual(decision["action"], KEEP)
        self.assertFalse(decision["correctness_certified"])
        self.assertEqual(deterministic_recovery({})["action"], RESTORE)
        self.assertEqual(deterministic_recovery({**state, "allowed_actions": [KEEP, "invent"]})["action"], RESTORE)
    def test_native_two_action_decision_requires_complete_normalized_probabilities(self):
        import math
        from chisel.learning.qa_typed_recovery import KEEP, RESTORE, select_recovery

        raw = {"choices": [{"message": {"content": "A"}, "logprobs": {"content": [{
            "token": "A", "top_logprobs": [{"token": "A", "logprob": math.log(0.8)},
                                               {"token": "R", "logprob": math.log(0.2)}]}]}}]}
        self.assertEqual(select_recovery(raw)["action"], KEEP)
        altered = copy.deepcopy(raw)
        altered["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["token"] = "unknown"
        self.assertEqual(select_recovery(altered)["action"], RESTORE)
        altered = copy.deepcopy(raw)
        altered["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["logprob"] = math.log(0.01)
        self.assertEqual(select_recovery(altered)["action"], RESTORE)
        self.assertEqual(select_recovery({})["action"], RESTORE)
    def test_recovery_messages_do_not_treat_source_match_as_semantic_proof(self):
        from chisel.learning.qa_typed_recovery import recovery_messages, response_state

        context = "Nora wrote the book. Mira painted its cover."
        state = response_state("Who wrote the book?", context, "<answer>Mira</answer>")
        self.assertTrue(state["supported"])
        self.assertFalse(state["correctness_certified"])
        messages = recovery_messages("Who wrote the book?", context, state)
        self.assertIn("matching word alone is insufficient", messages[0]["content"])
        self.assertIn("Nora wrote", messages[1]["content"])
        state["source_start"] = 0
        with self.assertRaises(ValueError):
            recovery_messages("Who wrote the book?", context, state)
    def test_unsupported_or_invalid_drafts_restore_original_without_a_judge(self):
        from chisel.learning.qa_typed_recovery import recover_answer

        for draft in ("<answer>Unknown</answer>", "<answer>Mira</answer>", "Nora", "<answer>Nora</answer> explanation"):
            calls = []

            def forbidden_judge(state):
                raise AssertionError("Unsupported public state must bypass the discriminator")

            result = recover_answer(question="Who wrote Amber Harbor?", context="Nora wrote Amber Harbor.",
                                    draft=draft, decide=forbidden_judge,
                                    original=lambda: calls.append("unchanged_original") or "<answer>Nora</answer>")
            self.assertTrue(result["restored_original"])
            self.assertEqual(calls, ["unchanged_original"])
            self.assertEqual(result["answer"], "<answer>Nora</answer>")
    def test_finite_recovery_choice_cannot_invent_an_answer_or_certify_correctness(self):
        from chisel.learning.qa_typed_recovery import KEEP, RESTORE, recover_answer

        for choice, restored in ((KEEP, False), (RESTORE, True), ("invented_answer", True)):
            calls = []
            result = recover_answer(question="Who wrote Amber Harbor?", context="Nora wrote Amber Harbor.",
                                    draft="<answer>Nora</answer>", decide=lambda state: choice,
                                    original=lambda: calls.append("original") or "<answer>Nora Vale</answer>")
            self.assertEqual(result["restored_original"], restored)
            self.assertEqual(len(calls), int(restored))
            self.assertFalse(result["state"]["correctness_certified"])
            self.assertFalse(result["correctness_guaranteed"])
    def test_recovery_quality_floor_is_inclusive_73_34_and_retains_prior_cost_target(self):
        from chisel.learning.qa_typed_recovery import quality_cost_gate

        arguments = {"tasks": 5000, "total_tokens": 5000000, "parent_correct": 3666,
                     "parent_tokens": 15000000, "no_skill_tokens": 6000000}
        self.assertTrue(quality_cost_gate(correct=3667, **arguments)["all_targets_pass"])
        self.assertFalse(quality_cost_gate(correct=3666, **arguments)["all_targets_pass"])
        arguments["total_tokens"] = 7000000
        result = quality_cost_gate(correct=3667, **arguments)
        self.assertTrue(result["quality_and_large_saving_pass"])
        self.assertFalse(result["all_targets_pass"])

class TypedProbabilityTests(unittest.TestCase):
    def test_reasoning_sentence_stop_preserves_alias_logits_and_full_usage(self):
        from chisel.harness.typed_decisions import alias_choice_logits, reasoned_choices

        groups = {"unknown": [0, 10], "span": [1, 11]}
        raw = {"0": 0.0, "10": -1.0, "1": 1.0, "11": 4.0}
        calls = []

        def advance(tokens, *, final, use_cache):
            reused = bool(calls)
            calls.append(tokens)
            return [{"reused_prefix": reused, "next_token_id": 42,
                     "choice_logits": alias_choice_logits(groups, raw), "token_logits": raw}]

        result = reasoned_choices(input_token_ids=[[12, 13]], option_token_ids=[{"unknown": 0, "span": 1}],
                                  advance=advance, maximum_reasoning_tokens=32, handoff_token_ids=[70, 71],
                                  stop_token_ids=[70], option_token_groups=[groups],
                                  stop_reasoning=lambda tokens: "sentence_end" if len(tokens) == 2 else None)[0]
        self.assertEqual(result["reasoning_stop"], "sentence_end")
        self.assertEqual(result["reasoning_token_ids"], [42, 42])
        self.assertEqual(result["raw_token_logits"], raw)
        self.assertEqual(result["selected"], "span")
        self.assertEqual(result["usage"]["total_tokens"], 6)
        self.assertEqual(len(calls), 3)
    def test_label_alias_readout_includes_space_prefixed_probability_mass(self):
        import math
        from chisel.harness.typed_decisions import alias_choice_logits, choice_probabilities

        groups = {"unknown": [0, 10], "span-a": [1, 11], "span-b": [2, 12]}
        logits = {"0": -1000.0, "10": -1000.0, "1": 0.0, "11": 4.0, "2": 3.0, "12": 1.0}
        combined = alias_choice_logits(groups, logits)
        decision = choice_probabilities(combined)
        self.assertEqual(decision["selected"], "span-a")
        self.assertAlmostEqual(sum(decision["probabilities"].values()), 1.0)
        self.assertAlmostEqual(combined["span-a"], math.log(math.exp(0.0) + math.exp(4.0)))
        self.assertFalse(decision["calibrated"])
    def test_label_alias_readout_rejects_overlaps_missing_logits_and_generated_numbers(self):
        from chisel.harness.typed_decisions import alias_choice_logits

        for groups, logits in [
            ({"unknown": [0], "span": [0]}, {"0": 1.0}),
            ({"unknown": [0], "span": [1]}, {"0": 1.0}),
            ({"unknown": [0], "span": [1]}, {"0": 1.0, "1": "0.9"}),
            ({"unknown": [0], "span": [1]}, {"0": 1.0, "1": float("nan")}),
        ]:
            with self.subTest(groups=groups, logits=logits), self.assertRaises(ValueError):
                alias_choice_logits(groups, logits)
    def test_reasoning_does_not_duplicate_an_observed_close_think_token(self):
        from chisel.harness.typed_decisions import reasoned_choices

        calls = []

        def advance(tokens, *, final, use_cache):
            reused = bool(calls)
            calls.append(tokens)
            return [{"reused_prefix": reused, "next_token_id": 70,
                     "choice_logits": {"unknown": 0.0, "span": 1.0}}]

        result = reasoned_choices(input_token_ids=[[10]], option_token_ids=[{"unknown": 0, "span": 1}],
                                  advance=advance, maximum_reasoning_tokens=96,
                                  handoff_token_ids=[70, 71], stop_token_ids=[70])[0]
        self.assertEqual(calls[-1], [[70, 71]])
        self.assertEqual(result["handoff_token_ids"], [71])
        self.assertEqual(result["usage"]["total_tokens"], 3)
    def test_cached_reasoning_is_bounded_and_charges_every_generated_and_forced_token(self):
        from chisel.harness.typed_decisions import reasoned_choices

        calls = []

        def advance(tokens, *, final, use_cache):
            reused = bool(calls)
            calls.append({"tokens": tokens, "final": final, "use_cache": use_cache})
            if final:
                return [{"reused_prefix": reused, "choice_logits": {"unknown": 0.0, "span": 2.0}}
                        for _ in tokens]
            return [{"reused_prefix": reused, "next_token_id": 99 if index == 0 else 42}
                    for index, _ in enumerate(tokens)]

        results = reasoned_choices(
            input_token_ids=[[10, 11], [12]], option_token_ids=[{"unknown": 0, "span": 1}] * 2,
            advance=advance, maximum_reasoning_tokens=3, handoff_token_ids=[70, 71], stop_token_ids=[99],
        )
        self.assertEqual([result["reasoning_token_ids"] for result in results], [[99], [42, 42, 42]])
        self.assertEqual([result["usage"]["total_tokens"] for result in results], [5, 6])
        self.assertEqual([result["reasoning_stop"] for result in results], ["stop_token", "budget"])
        self.assertTrue(all(result["final_answer_generated"] is False for result in results))
        self.assertEqual(calls[0]["tokens"], [[10, 11], [12]])
        self.assertEqual(calls[-1]["tokens"], [[70, 71], [42, 70, 71]])
        self.assertTrue(all(call["use_cache"] for call in calls))
        self.assertEqual(sum(sum(map(len, call["tokens"])) for call in calls), 11)
    def test_no_reasoning_control_uses_one_forward_and_includes_handoff_cost(self):
        from chisel.harness.typed_decisions import reasoned_choices

        calls = []

        def advance(tokens, *, final, use_cache):
            calls.append(tokens)
            self.assertTrue(final)
            return [{"reused_prefix": False, "choice_logits": {"unknown": 0.0, "span": 1.0}}]

        result = reasoned_choices(
            input_token_ids=[[10, 11]], option_token_ids=[{"unknown": 0, "span": 1}],
            advance=advance, maximum_reasoning_tokens=0, handoff_token_ids=[70], stop_token_ids=[99],
        )[0]
        self.assertEqual(calls, [[[10, 11, 70]]])
        self.assertEqual(result["usage"], {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3,
                                          "source": "native_cached_token_stream"})
        self.assertFalse(result["generated_text"])
    def test_reasoning_rejects_cache_replay_and_oversized_generation_budget(self):
        from chisel.harness.typed_decisions import reasoned_choices

        arguments = {"input_token_ids": [[10]], "option_token_ids": [{"unknown": 0, "span": 1}],
                     "handoff_token_ids": [70], "stop_token_ids": [99]}

        def uncached(tokens, **kwargs):
            return [{"reused_prefix": False, "next_token_id": 42, "choice_logits": {"unknown": 0.0, "span": 1.0}}]

        with self.assertRaisesRegex(ValueError, "prefix cache"):
            reasoned_choices(**arguments, advance=uncached, maximum_reasoning_tokens=2)
        with self.assertRaisesRegex(ValueError, "96 tokens"):
            reasoned_choices(**arguments, advance=uncached, maximum_reasoning_tokens=97)
    def tokenizer(self):
        import json

        class Tokenizer:
            def encode(self, text, **arguments):
                return list(text.encode())

            def apply_chat_template(self, messages, **arguments):
                return list(json.dumps(messages, separators=(",", ":")).encode())

        return Tokenizer()
    def test_probabilities_come_from_logits_and_do_not_claim_calibration(self):
        from chisel.harness.typed_decisions import choice_probabilities

        result = choice_probabilities({"unknown": -10000.0, "source-span": 10000.0})
        self.assertEqual(result["type"], "choice")
        self.assertEqual(result["selected"], "source-span")
        self.assertEqual(result["probabilities"], {"unknown": 0.0, "source-span": 1.0})
        self.assertFalse(result["calibrated"])
        self.assertFalse(result["abstained"])
        self.assertEqual(choice_probabilities({"source-span": 0.0, "unknown": 0.0})["selected"], "unknown")
    def test_invalid_or_self_reported_probabilities_are_rejected(self):
        from chisel.harness.typed_decisions import choice_probabilities

        for logits in ({"answer": 0.5}, {"unknown": 0.0, "answer": "0.99"},
                       {"unknown": 0.0, "answer": float("nan")},
                       {"unknown": 0.0, "answer": float("inf")},
                       {"unknown": 0.0, "answer": True}):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                choice_probabilities(logits)
    def test_single_forward_has_zero_decoding_and_charges_all_input(self):
        from types import SimpleNamespace
        from chisel.harness.typed_decisions import forward_choices

        class Vector:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class Logits:
            def __getitem__(self, position):
                batch_index, final_position, token_ids = position
                self_values = [[-1.0, 2.0, 0.0], [3.0, 1.0, 2.0]][batch_index]
                if final_position != -1:
                    raise AssertionError("Only final input-position logits are read")
                return Vector([self_values[token_id] for token_id in token_ids])

        calls = []

        def model(**arguments):
            calls.append(arguments)
            return SimpleNamespace(logits=Logits())

        results = forward_choices(
            model, input_ids=[[10, 11, 12], [0, 13, 14]],
            attention_mask=[[1, 1, 1], [0, 1, 1]],
            option_token_ids=[{"unknown": 0, "span-1": 1}, {"unknown": 0, "span-2": 2}],
        )
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["use_cache"])
        self.assertEqual(calls[0]["logits_to_keep"], 1)
        self.assertEqual([result["selected"] for result in results], ["span-1", "unknown"])
        self.assertEqual(sum(result["usage"]["total_tokens"] for result in results), 5)
        self.assertTrue(all(result["usage"]["completion_tokens"] == 0 for result in results))
        self.assertTrue(all(result["autoregressive_steps"] == 0 and not result["generated_text"] for result in results))
    def test_invalid_mask_and_duplicate_choice_tokens_block_before_forward(self):
        from chisel.harness.typed_decisions import forward_choices

        def forbidden_model(**arguments):
            raise AssertionError("Invalid input must not execute the model")

        for mask, options in [([1, 0], {"unknown": 0, "span": 1}),
                              ([1, 1], {"unknown": 0, "span": 0}),
                              ([1, 1], {"unknown": 0, "span": True})]:
            with self.subTest(mask=mask, options=options), self.assertRaises(ValueError):
                forward_choices(forbidden_model, input_ids=[[10, 11]], attention_mask=[mask],
                                option_token_ids=[options])

class ExactDocumentPackTests(unittest.TestCase):
    def test_budgeted_evidence_checks_complete_request_and_exact_spans(self):
        context = "[DOC] Unrelated detail. " * 20 + "[DOC] The synthetic requested marker is amber."
        selected = select_context_for_request_budget(
            "What is the requested marker?", context, method="bm25_closure",
            count_tokens=len, count_request=lambda text: len(text) + 40,
            maximum_request_tokens=160,
        )
        self.assertLessEqual(len(selected.context) + 40, 160)
        self.assertIn("amber", selected.context)
        self.assertEqual(reconstruct_context(context, selected.receipt), selected.context)
        self.assertTrue(selected.receipt["request_budget"]["unique_evidence_may_be_omitted"])
        self.assertFalse(selected.receipt["request_budget"]["quality_equivalence_assumed"])
    def test_infeasible_request_budget_falls_back_without_empty_evidence(self):
        context = "[DOC] Synthetic source evidence."
        selected = select_context_for_request_budget(
            "What evidence?", context, method="bm25_closure",
            count_tokens=len, count_request=lambda text: 200 + len(text),
            maximum_request_tokens=150,
        )
        self.assertEqual(selected.context, context)
        self.assertEqual(selected.receipt["request_budget"]["fallback"], "system_and_question_exceed_request_budget")
    def test_only_exact_repeated_documents_are_packed(self):
        first = "[DOC]" + "First exact document with dates and names. " * 10
        second = "[DOC]" + "Different exact document with quantities. " * 10
        original = first + second + first
        packed = pack_duplicate_documents(original)
        self.assertEqual(packed.context, first + second + "[DOC][duplicate of document 1]")
        self.assertEqual(packed.receipt["duplicate_documents"], 1)
        self.assertFalse(packed.receipt["unique_evidence_removed"])
        self.assertEqual(packed.receipt["spans"][2]["repeat_of"], 1)
        verify_duplicate_documents(original, packed)
    def test_unique_documents_and_short_duplicates_are_unchanged(self):
        for original in ("", "A source without markers.", "[DOC]short[DOC]short", "[DOC]Name[DOC]name"):
            with self.subTest(original=original):
                packed = pack_duplicate_documents(original)
                self.assertEqual(packed.context, original)
                self.assertEqual(packed.receipt["duplicate_documents"], 0)
    def test_source_offsets_preserve_unicode_newlines_and_prefix(self):
        document = "[DOC]" + "Name \u4e2d\u6587 1997\r\n" * 20
        original = "Header\n" + document + document
        packed = pack_duplicate_documents(original)
        self.assertEqual(packed.receipt["spans"][2]["repeat_of"], 2)
        spans = packed.receipt["spans"]
        self.assertEqual("".join(original[row["start_char"]:row["end_char"]] for row in spans), original)
        verify_duplicate_documents(original, packed)
    def test_tampered_content_and_receipts_are_rejected(self):
        original = "[DOC]" + "synthetic source " * 20
        packed = pack_duplicate_documents(original)
        with self.assertRaises(ValueError):
            verify_duplicate_documents(original, ContextSelection("modified", packed.receipt))
        receipt = copy.deepcopy(packed.receipt)
        receipt["spans"][0]["start_char"] = 1
        with self.assertRaises(ValueError):
            verify_duplicate_documents(original, ContextSelection(packed.context, receipt))

class ChiselCompressionTests(unittest.TestCase):
    def sources(self):
        delete_sections, modify_sections = [], []
        for section_index, (section, rules) in enumerate(contracts.DELETE_COMPACT_RULES):
            lines = [f"- Synthetic {section} source clause {index}." for index in range(len(rules))]
            delete_sections.append(f"## {section}\n" + "\n".join(lines))
            if section_index == 0:
                lines = lines + [f"- Synthetic extra source clause {index}." for index in range(5)]
            modify_sections.append(f"## {section}\n" + "\n".join(lines))
        prefix = "Synthetic prefix.\n## Skill\n"
        ending = "\n## Answer Contract\nReturn only <answer>...</answer>.\n"
        return prefix + "\n".join(delete_sections) + ending, prefix + "\n".join(modify_sections) + ending
    def test_candidates_keep_route_specific_rules_and_exact_output_contract(self):
        delete, modify = self.sources()
        with patch.object(contracts, "DELETE_SOURCE_SHA256", hashlib.sha256(delete.encode()).hexdigest()), patch.object(
            contracts, "MODIFY_SOURCE_SHA256", hashlib.sha256(modify.encode()).hexdigest(),
        ):
            candidates, receipt = contracts.chisel_compression_contracts(delete, modify)
        self.assertEqual(len(receipt["modify_clauses"]), 38)
        self.assertEqual(len(receipt["modify_only_source_clauses"]), 5)
        self.assertFalse(receipt["semantic_equivalence_established"])
        self.assertNotEqual(candidates["delete_capsule"], candidates["modify_capsule"])
        for candidate in candidates.values():
            self.assertEqual(candidate.split("## Answer Contract\n")[1], delete.split("## Answer Contract\n")[1])
        self.assertTrue(candidates["modify_capsule"].startswith(candidates["delete_capsule"].split("\n\n## Answer Contract")[0]))
    def test_unbound_modify_source_is_rejected(self):
        delete, modify = self.sources()
        with patch.object(contracts, "DELETE_SOURCE_SHA256", hashlib.sha256(delete.encode()).hexdigest()):
            with self.assertRaisesRegex(ValueError, "exact frozen"):
                contracts.chisel_compression_contracts(delete, modify)

class JevInterleaveTests(unittest.TestCase):
    def test_interleaving_has_no_typesafe_client_or_network_dependency(self):
        import ast
        import inspect
        from chisel.harness import jev_interleave

        tree = ast.parse(inspect.getsource(jev_interleave))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any(name.startswith(("urllib", "requests", "httpx", "typesafe")) for name in imports))
        self.assertFalse(hasattr(jev_interleave, "JevClient"))
        self.assertFalse(hasattr(jev_interleave, "jev_payload"))
        self.assertFalse(hasattr(jev_interleave, "jev_decisions"))
    def test_verification_state_is_bound_to_exact_source_window(self):
        from chisel.harness.jev_interleave import Draft, verifier_state

        source = "prefix " * 100 + "Exact quote." + " suffix" * 100
        state = verifier_state("Question", source, Draft("target", "answer", "Exact quote."))
        self.assertEqual(state["source_window"], source[state["source_window_start"]:state["source_window_end"]])
        self.assertFalse(state["full_source_visible"])
    def test_typed_checks_can_stop_after_one_bounded_draft(self):
        import json
        from chisel.harness.jev_interleave import CHECKS, interleave

        generate = unittest.mock.Mock(return_value=json.dumps({
            "target": "Requested label", "answer": "amber", "evidence": "The marker is amber.",
        }))
        judge = unittest.mock.Mock(return_value=dict.fromkeys(CHECKS, "pass"))
        fallback = unittest.mock.Mock(return_value="original answer")
        result = interleave(context="The marker is amber.", generate=generate, judge=judge, fallback=fallback)
        self.assertEqual(result["answer"], "<answer>amber</answer>")
        self.assertFalse(result["fallback"])
        self.assertFalse(result["quality_guarantee"])
        self.assertEqual(generate.call_count, 1)
        fallback.assert_not_called()
    def test_unknown_judgment_refines_once_then_falls_back(self):
        import json
        from chisel.harness.jev_interleave import CHECKS, interleave

        generate = unittest.mock.Mock(return_value=json.dumps({
            "target": "Requested label", "answer": "amber", "evidence": "The marker is amber.",
        }))
        decisions = dict.fromkeys(CHECKS, "pass")
        decisions["target_match"] = "unknown"
        judge = unittest.mock.Mock(return_value=decisions)
        fallback = unittest.mock.Mock(return_value="<answer>original</answer>")
        result = interleave(context="The marker is amber.", generate=generate, judge=judge, fallback=fallback)
        self.assertTrue(result["fallback"])
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(judge.call_count, 2)
        self.assertEqual(generate.call_args.args[1], decisions)
        fallback.assert_called_once()
    def test_fabricated_evidence_or_partial_decisions_cannot_terminate(self):
        import json
        from chisel.harness.jev_interleave import interleave

        for quote, decision in (("Invented evidence", {}), ("Source", {"target_match": "pass"})):
            generate = unittest.mock.Mock(return_value=json.dumps({"target": "label", "answer": "amber", "evidence": quote}))
            judge = unittest.mock.Mock(return_value=decision)
            fallback = unittest.mock.Mock(return_value="original")
            result = interleave(context="Source", generate=generate, judge=judge, fallback=fallback)
            self.assertTrue(result["fallback"])
            fallback.assert_called_once()

class QAAdapterAccountingTests(unittest.TestCase):
    def test_full_policy_charges_draft_judge_and_original_fallback(self):
        import json
        from chisel.adapters.qa import solve
        from chisel.harness.jev_interleave import CHECKS

        class Legacy:
            @staticmethod
            def truncate_context(text):
                return text

            @staticmethod
            def build_user(item):
                return item["context"] + item["question"]

        item = {"id": "synthetic", "question": "Which label?", "context": "The label is amber."}
        draft = json.dumps({"target": "Requested label", "answer": "amber", "evidence": item["context"]})
        checks = json.dumps(dict.fromkeys(CHECKS, "unknown"))
        replies = iter((draft, checks, draft, checks, "<answer>original</answer>"))
        records = []

        def call(messages, **kwargs):
            records.append(kwargs)
            return {"raw_response": {"choices": [{"message": {"content": next(replies)}}]},
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                    "call_file": f"calls/{len(records)}.json"}

        settings = {"backend": "qwen_typed_decision_proxy", "maximum_drafts": 2,
                    "draft_max_tokens": 192, "judge_max_tokens": 96}
        result = solve(item, "interleave", "delete", {"delete": "Frozen CHISEL"},
                       {"interleave": settings}, Legacy(), call)
        self.assertEqual(result["usage"]["total_tokens"], 60)
        self.assertEqual(result["stages"], ["draft", "proxy_judge", "draft", "proxy_judge", "fallback"])
        self.assertEqual(result["answer"], "<answer>original</answer>")
        self.assertTrue(result["fallback"])
        self.assertFalse(result["true_jev_used"])
        self.assertFalse(result["calibration_claimed"])
        self.assertEqual(records[-1]["maximum"], 512)
    def test_bad_judge_never_skips_the_charged_original_fallback(self):
        import json
        from chisel.adapters.qa import solve

        item = {"id": "synthetic", "question": "Question", "context": "Exact source."}
        replies = iter((json.dumps({"target": "label", "answer": "amber", "evidence": item["context"]}), "{}", "<answer>original</answer>"))
        counter = []

        def call(messages, **kwargs):
            counter.append(kwargs)
            return {"raw_response": {"choices": [{"message": {"content": next(replies)}}]},
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                    "call_file": f"call-{len(counter)}"}

        legacy = unittest.mock.Mock()
        legacy.truncate_context.side_effect = lambda text: text
        legacy.build_user.return_value = "Original canonical input"
        result = solve(item, "interleave", "modify", {"modify": "Original rules"},
                       {"interleave": {"backend": "qwen_typed_decision_proxy", "maximum_drafts": 2, "draft_max_tokens": 192, "judge_max_tokens": 96}},
                       legacy, call)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["usage"]["total_tokens"], 18)
