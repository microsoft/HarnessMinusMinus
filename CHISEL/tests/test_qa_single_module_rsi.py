"""Extracted synthetic controller unit tests; no benchmark or provider calls."""
from copy import deepcopy
import json
import pytest

from chisel.learning import qa_meta_rsi as patches

from chisel.learning import qa_single_module_rsi as method

def skill():
    text = "Synthetic QA instructions.\n"
    for heading in patches.SECTION_IDS:
        text += f"## {heading}\n- Use the supplied source carefully before answering.\n"
    text += "## Answer Contract\nReturn exactly <answer>one answer</answer>."
    return patches.initial_skill({"delete": text, "modify": text})

def items():
    return [
        {"id": f"synthetic-{index}", "group": f"group-{index}",
         "question": f"What is the synthetic marker numbered {index}?",
         "context": f"Entire synthetic source paragraph {index}."}
        for index in range(2)
    ]

def blocks(em=0, f1=None, tokens=100, *, mixed=False):
    output = {}
    for repeat in range(4):
        current = repeat % 2 if mixed else em
        output[str(repeat)] = {
            item["id"]: {
                "exact_match": current, "f1": float(current if f1 is None else f1),
                "tokens": tokens, "prompt_tokens": tokens - 5, "completion_tokens": 5,
                "request_sha256": patches.text_hash(item["id"]),
                "prediction": "amber-target" if current else "wrong",
                "raw_actor_content": "<answer>amber-target</answer>" if current else "<answer>wrong</answer>",
            }
            for item in items()
        }
    return output

def cases():
    return method.training_cases(
        items(), blocks(mixed=True), {item["id"]: ["amber-target"] for item in items()},
        {item["id"]: {"variant": "delete"} for item in items()}, repeats=4, partition="train",
    )

def diagnoses(evidence=None, module="evidence"):
    return [{
        "evidence": case["evidence"], "module": module, "success_repeat": "1", "failure_repeat": "0",
        "rationale": "A synthetic paired-output discrepancy.",
        "suggested_change": "Check the supplied supporting relation.",
    } for case in (evidence or cases())]

def limits():
    return {"maximum_changed_rule_ids": 4, "maximum_rule_characters": 1000,
            "maximum_profile_growth_bytes": 512}

def test_compact_step_changes_only_one_module_and_preserves_prior_steps():
    original = skill()
    compact = deepcopy(original)
    compact["rules"] = {rule: "- Use the source." for rule in compact["rules"]}
    first, receipt = method.compact_module_candidate(
        original, original, compact, module="evidence", mode="clause",
    )
    second, _ = method.compact_module_candidate(
        first, original, compact, module="traps", mode="clause",
    )
    assert receipt["other_modules_unchanged"]
    assert second["layout"] == original["layout"]
    for rule in original["rules"]:
        expected = compact["rules"][rule] if rule.split(".")[1] in {"evidence", "traps"} else original["rules"][rule]
        assert second["rules"][rule] == expected
    with pytest.raises(ValueError, match="previously unchanged"):
        method.compact_module_candidate(first, original, compact, module="evidence", mode="clause")

def test_compact_step_rejects_layout_changes_and_unknown_modules():
    original, compact = skill(), skill()
    changed = deepcopy(original)
    changed["layout"]["delete"][0]["literal"] = "Changed immutable instruction.\n"
    with pytest.raises(ValueError, match="protected layout"):
        method.compact_module_candidate(changed, original, compact, module="evidence", mode="clause")
    with pytest.raises(ValueError, match="declared module"):
        method.compact_module_candidate(original, original, compact, module="router", mode="clause")

def test_integration_requires_independent_lineages_from_one_parent():
    original = skill()
    compact = deepcopy(original)
    compact["rules"] = {rule: "- Use the source." for rule in compact["rules"]}
    children = {
        module: method.compact_module_candidate(original, original, compact, module=module, mode="clause")[0]
        for module in ("evidence", "traps")
    }
    integrated, receipt = method.integrate_compact_modules(original, children)
    assert receipt["shared_parent_verified"]
    assert receipt["integration_quality_established"] is False
    for rule in original["rules"]:
        expected = compact["rules"][rule] if rule.split(".")[1] in children else original["rules"][rule]
        assert integrated["rules"][rule] == expected
    chained, _ = method.compact_module_candidate(
        children["evidence"], original, compact, module="traps", mode="clause",
    )
    with pytest.raises(ValueError, match="identical original"):
        method.integrate_compact_modules(original, {"traps": chained})

def test_integration_rejects_cross_module_edits_even_with_matching_parent():
    original = skill()
    changed = deepcopy(original)
    changed["parent_sha256"] = patches.digest(original)
    changed["rules"]["delete.normalization.r01"] = "- Use the source."
    with pytest.raises(ValueError, match="neighboring module"):
        method.integrate_compact_modules(original, {"evidence": changed})

def test_compact_gate_accepts_observed_quality_ties_but_never_guarantees_unseen_inputs():
    before, after = blocks(1)["0"], blocks(1, tokens=70)["0"]
    result = method.compact_quality_gate(before, after, minimum_saving=0.1)
    assert result["passed"]
    assert result["observed_em_losses"] == result["observed_f1_losses"] == 0
    assert result["unseen_task_guarantee"] is False
    assert result["paper_admitted"] is False

@pytest.mark.parametrize("child", [blocks(0, tokens=70)["0"], blocks(1, f1=0.5, tokens=70)["0"], blocks(1, tokens=100)["0"]])
def test_compact_gate_rejects_quality_or_cost_regression(child):
    assert not method.compact_quality_gate(blocks(1)["0"], child)["passed"]

def test_compact_gate_rejects_partial_or_unknown_usage():
    before, after = blocks(1)["0"], blocks(1, tokens=70)["0"]
    after.pop("synthetic-1")
    with pytest.raises(ValueError, match="complete paired"):
        method.compact_quality_gate(before, after)
    after = blocks(1, tokens=70)["0"]
    after["synthetic-0"]["tokens"] = 0
    with pytest.raises(ValueError, match="usage"):
        method.compact_quality_gate(before, after)

def test_no_skill_target_is_strict_for_both_score_and_actual_tokens():
    reference = {
        f"synthetic-{index}": {"exact_match": 1, "f1": 1.0, "tokens": 100,
                               "prompt_tokens": 90, "completion_tokens": 10}
        for index in range(1000)
    }
    candidate = {
        task: {"exact_match": int(index < 733), "f1": float(index < 733), "tokens": 99,
               "prompt_tokens": 90, "completion_tokens": 9}
        for index, task in enumerate(reference)
    }
    assert not method.no_skill_cost_gate(reference, candidate)["passed"]
    candidate["synthetic-733"].update({"exact_match": 1, "f1": 1.0})
    gate = method.no_skill_cost_gate(reference, candidate)
    assert gate["passed"]
    assert gate["candidate"]["correct"] == 734
    assert not gate["f1_is_acceptance_constraint"]
    assert not gate["chisel_quality_nonregression_required"]
    for row in candidate.values():
        row.update({"tokens": 100, "completion_tokens": 10})
    assert not method.no_skill_cost_gate(reference, candidate)["passed"]

def test_module_progress_requires_real_score_gain_and_cost_reduction():
    parent = blocks(0)["0"]
    candidate = blocks(1, tokens=70)["0"]
    assert method.module_progress_gate(parent, candidate)["passed"]
    assert not method.module_progress_gate(parent, blocks(0, tokens=70)["0"])["passed"]
    assert not method.module_progress_gate(parent, blocks(1, tokens=100)["0"])["passed"]

def test_module_lineage_integration_preserves_full_parent_chain():
    original = skill()
    compact = deepcopy(original)
    compact["rules"] = {rule: "- Use the source." for rule in compact["rules"]}
    first, _ = method.compact_module_candidate(original, original, compact, module="evidence", mode="clause")
    value = {
        "parent_skill_sha256": patches.digest(first), "hypothesis": "Refine only the selected module.",
        "risk": "Synthetic contract test.", "evidence": ["SOURCE"],
        "changes": [{"rule_id": "delete.evidence.r01", "text": "- Use source."}],
    }
    second, _ = patches.apply_skill_proposal(first, value, evidence_ids={"SOURCE"}, forbidden=[],
                                            max_changes=1, max_rule_characters=100, max_profile_growth_bytes=0)
    integrated, receipt = method.integrate_module_lineages(original, {"evidence": [first, second]})
    assert patches.render_skill(integrated) == patches.render_skill(second)
    assert receipt["accepted_lineage_hashes"]["evidence"] == [patches.digest(original), patches.digest(first), patches.digest(second)]
    with pytest.raises(ValueError, match="parent chain"):
        method.integrate_module_lineages(original, {"evidence": [second]})
    altered = deepcopy(second)
    altered["rules"]["delete.traps.r01"] = "- Incorrect cross-module modification."
    with pytest.raises(ValueError, match="neighboring"):
        method.integrate_module_lineages(original, {"evidence": [first, altered]})

def test_no_skill_target_rejects_mismatched_pairs_and_invalid_threshold():
    reference, candidate = blocks(1)["0"], blocks(1, tokens=70)["0"]
    for value in (True, float("nan"), -1, 100, "73.3"):
        with pytest.raises(ValueError, match="EM threshold"):
            method.no_skill_cost_gate(reference, candidate, minimum_em_percent=value)
    candidate.pop("synthetic-1")
    with pytest.raises(ValueError, match="complete paired"):
        method.no_skill_cost_gate(reference, candidate)

def proposal(parent=None):
    return {
        "parent_skill_sha256": patches.digest(parent or skill()),
        "hypothesis": "Repair a synthetic evidence-selection issue.",
        "risk": "A shorter rule could omit a useful distinction.",
        "evidence": ["T001", "T002"],
        "changes": [{"rule_id": "delete.evidence.r01", "text": "- Match the supplied evidence."}],
    }

def assessment(parent=None, child=None, aa=None, cohort=None, generic=None):
    child = child or blocks(1, tokens=70)
    generic = generic or blocks(0, tokens=child["0"]["synthetic-0"]["tokens"])
    return method.assess_candidate(
        cohort or items(), parent or blocks(0), child,
        aa or blocks(0), generic, repeats=4, changed=True, bootstrap_samples=100, seed=12,
    )

def test_partitioning_is_label_blind_disjoint_and_retains_duplicate_groups():
    identities = {"a1": "a", "a2": "a", "b": "b", "c": "c", "excluded": "x"}
    value = method.select_partitions(
        identities, {"x"}, counts=dict.fromkeys(method.PARTITIONS, 1), seed=1, maximum_tasks=4,
    )
    assert sum(map(len, value["task_ids"].values())) == 4
    assert any({"a1", "a2"} <= set(tasks) for tasks in value["task_ids"].values())
    assert len({group for groups in value["groups"].values() for group in groups}) == 3
    with pytest.raises(ValueError, match="no duplicate dropping"):
        method.select_partitions(
            identities, {"x"}, counts=dict.fromkeys(method.PARTITIONS, 1), seed=1, maximum_tasks=3,
        )

@pytest.mark.parametrize("change", ["missing_repeat", "missing_task", "fractional_em", "usage", "length"])
def test_invalid_repeat_inventories_and_native_usage_fail(change):
    value = blocks()
    if change == "missing_repeat":
        value.pop("3")
    elif change == "missing_task":
        value["0"].pop("synthetic-0")
    elif change == "fractional_em":
        value["0"]["synthetic-0"]["exact_match"] = 0.5
    elif change == "usage":
        value["0"]["synthetic-0"]["tokens"] += 1
    else:
        value["0"]["synthetic-0"]["prompt_tokens"] += 1
        value["0"]["synthetic-0"]["tokens"] += 1
    with pytest.raises(ValueError):
        method.validate_blocks(items(), value, 4)

def test_training_categories_are_per_task_not_votes_per_rollout():
    for value, category in ((blocks(1), "positive"), (blocks(0), "negative"),
                            (blocks(mixed=True), "contrastive")):
        result = method.training_cases(
            items(), value, {item["id"]: ["amber-target"] for item in items()},
            {item["id"]: {"variant": "delete"} for item in items()}, repeats=4, partition="train",
        )
        assert {case["category"] for case in result} == {category}
        assert all(len(case["observations"]) == 4 for case in result)
        assert all(case["context"].startswith("Entire synthetic source") for case in result)

def test_training_rejects_heldout_data_or_different_request_bodies():
    value = blocks(mixed=True)
    answers = {item["id"]: ["amber-target"] for item in items()}
    routes = {item["id"]: {"variant": "delete"} for item in items()}
    with pytest.raises(ValueError, match="only training"):
        method.training_cases(items(), value, answers, routes, repeats=4, partition="confirmation")
    value["1"]["synthetic-0"]["request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="byte-identical"):
        method.training_cases(items(), value, answers, routes, repeats=4, partition="train")

@pytest.mark.parametrize("key,value", [
    ("evidence", "T999"), ("module", "router"), ("success_repeat", "0"),
    ("failure_repeat", "1"), ("failure_repeat", "4"), ("rationale", ""), ("suggested_change", ""),
])
def test_diagnosis_must_cite_actual_same_task_contrast(key, value):
    diagnosis = diagnoses()[0]
    diagnosis[key] = value
    with pytest.raises(ValueError):
        method.validate_diagnosis(diagnosis, cases()[0])

def test_no_culprit_is_an_explicit_valid_diagnosis_not_a_forced_patch():
    values = diagnoses(module=None)
    for value in values:
        value["suggested_change"] = ""
    selected = method.select_module(cases(), values, 2)
    assert selected["status"] == "STOPPED_INSUFFICIENT_CONTRAST_SUPPORT"
    assert selected["module"] is None

def test_duplicate_questions_and_repeated_diagnoses_do_not_create_support():
    evidence = cases()
    evidence[1]["group"] = evidence[0]["group"]
    selected = method.select_module(evidence, diagnoses(evidence), 2)
    assert selected["votes"]["evidence"]["groups"] == 1
    assert selected["module"] is None
    with pytest.raises(ValueError, match="unique"):
        method.select_module(cases(), diagnoses() * 2, 2)

def test_single_target_is_selected_deterministically_from_training_only():
    selected = method.select_module(cases(), diagnoses(), 2)
    assert selected["module"] == "evidence"
    assert selected["votes"]["evidence"]["groups"] == 2
    assert selected["target_reselection_permitted"] is False
    assert selected["selection_uses_validation_or_confirmation"] is False
    assert method.select_module(list(reversed(cases())), list(reversed(diagnoses())), 2) == selected

def test_fixed_sibling_proposals_share_parent_controller_and_support():
    evidence, parent = cases(), skill()
    selected = method.select_module(evidence, diagnoses(), 2)
    payloads = [
        json.loads(method.proposal_messages(parent, selected, evidence, diagnoses(), index, limits())[1]["content"])
        for index in (0, 1)
    ]
    for payload in payloads:
        assert payload["parent_skill_sha256"] == patches.digest(parent)
        assert payload["controller"] == method.CONTROLLER
        assert "validation" not in payload
        assert payload["selected_module"] == "evidence"
        assert len(payload["training_cases"]) == 2
    assert payloads[0].pop("recipe") != payloads[1].pop("recipe")
    assert payloads[0] == payloads[1]
    with pytest.raises(ValueError, match="two frozen"):
        method.proposal_messages(parent, selected, evidence, diagnoses(), 2, limits())

def test_module_patch_schema_locks_parent_evidence_and_module_without_backend_uniqueitems():
    parent, evidence = skill(), cases()
    schema = method.module_patch_schema(parent, "evidence", evidence, limits())
    properties = schema["properties"]
    assert properties["parent_skill_sha256"]["enum"] == [patches.digest(parent)]
    assert properties["evidence"]["type"] == "array"
    assert properties["evidence"]["items"]["enum"] == ["T001", "T002"]
    assert properties["changes"]["items"]["properties"]["rule_id"]["enum"] == ["delete.evidence.r01"]
    assert "uniqueItems" not in json.dumps(schema)
    evidence[0]["partition"] = "confirmation"
    with pytest.raises(ValueError, match="training-only"):
        method.module_patch_schema(parent, "evidence", evidence, limits())

def test_only_supported_module_and_profile_can_be_rewritten():
    parent, evidence = skill(), cases()
    selected = method.select_module(evidence, diagnoses(), 2)
    child, receipt = method.compile_proposal(parent, proposal(parent), selected, evidence, limits())
    assert receipt["changed_rule_ids"] == ["delete.evidence.r01"]
    assert receipt["distinct_support_groups"] == 2
    assert child["layout"] == parent["layout"]
    for rule in parent["rules"]:
        if rule != "delete.evidence.r01":
            assert child["rules"][rule] == parent["rules"][rule]
    assert patches.render_skill(child)["modify"] == patches.render_skill(parent)["modify"]

@pytest.mark.parametrize("rule", [
    "delete.normalization.r01", "modify.evidence.r01", "delete.contract.r01",
])
def test_cross_module_unobserved_profile_and_contract_edits_fail(rule):
    value = proposal()
    value["changes"][0]["rule_id"] = rule
    with pytest.raises(ValueError):
        method.compile_proposal(
            skill(), value, method.select_module(cases(), diagnoses(), 2), cases(), limits(),
        )

@pytest.mark.parametrize("change", ["one_group", "heldout", "stale_parent", "memorized_answer"])
def test_patch_evidence_and_parent_are_bound(change):
    value = proposal()
    if change == "one_group":
        value["evidence"] = ["T001"]
    elif change == "heldout":
        value["evidence"] = ["T001", "heldout-identity"]
    elif change == "stale_parent":
        value["parent_skill_sha256"] = "0" * 64
    else:
        value["changes"][0]["text"] = "- Return amber-target."
    with pytest.raises(ValueError):
        method.compile_proposal(
            skill(), value, method.select_module(cases(), diagnoses(), 2), cases(), limits(),
        )

def test_support_payload_keeps_distinct_groups_and_complete_cases():
    evidence = cases()
    selected = method.select_module(evidence, diagnoses(), 2)
    chosen = method.support_cases(evidence, selected, 4)
    assert chosen == evidence
    assert all(len(case["observations"]) == 4 for case in chosen)
    with pytest.raises(ValueError, match="sufficient"):
        method.support_cases(evidence, selected, 1)

def test_success_requires_quality_cost_and_grouped_interval_support():
    value = assessment()
    assert value["point_gate"]["accepted"]
    assert value["confirmation_gate"]
    assert value["independent_groups"] == 2
    assert value["tasks"] == 2
    assert value["repeats"] == 4
    assert value["repeated_requests_are_independent_tasks"] is False
    assert value["aa_guard_is_confidence_bound"] is False
    assert value["paper_admitted"] is False
    assert all(value["source_group_intervals"][key][0] > 0
               for key in ("exact_match", "f1", "ten_percent_token_slack"))

@pytest.mark.parametrize("child", [
    blocks(0, tokens=70), blocks(1, tokens=91), blocks(1, f1=0, tokens=70),
])
def test_quality_ties_missing_saving_and_f1_failure_stop_promotion(child):
    value = assessment(child=child)
    assert not value["point_gate"]["accepted"]
    assert not value["confirmation_gate"]

def test_aa_variability_cannot_be_claimed_as_a_content_gain():
    value = assessment(aa=blocks(1))
    assert not value["point_gate"]["gates"]["effect_exceeds_descriptive_aa_guard"]
    assert not value["confirmation_gate"]

def test_one_group_does_not_become_many_independent_units():
    cohort = items()
    cohort[1]["group"] = cohort[0]["group"]
    value = assessment(cohort=cohort)
    assert value["point_gate"]["accepted"]
    assert value["independent_groups"] == 1
    assert not value["interval_support"]

def test_bootstrap_is_reproducible_with_duplicate_tasks_nested_inside_groups():
    first, second = assessment(), assessment()
    assert first == second
    assert len(first["per_repeat"]) == 4

def test_aa_input_mismatch_is_fatal_even_when_scores_match():
    with pytest.raises(ValueError, match="A/A input counts"):
        assessment(aa=blocks(0, tokens=101))

def test_native_input_matched_generic_tie_does_not_establish_content_benefit():
    value = assessment(generic=blocks(1, tokens=70))
    assert not value["point_gate"]["gates"]["both_endpoints_above_native_input_matched_generic"]
    assert not value["confirmation_gate"]
    with pytest.raises(ValueError, match="exactly match"):
        assessment(generic=blocks(0, tokens=71))

def test_generic_body_control_preserves_other_modules_and_answer_contract():
    parent = skill()
    parent["rules"]["delete.evidence.r01"] = "- " + "synthetic-content " * 100
    before = deepcopy(parent)
    source = patches.render_skill(parent)["delete"]
    control, receipt = method.length_matched_control(
        parent, "delete", "evidence", lambda text: len(("prefix " + text + " suffix").split()),
    )
    assert parent == before
    assert len(control.split()) == len(source.split())
    assert control[:control.index("## Context-Grounded Evidence Matching")] == (
        source[:source.index("## Context-Grounded Evidence Matching")]
    )
    assert control[control.index("## Clue Interpretation and Answer Type"):] == (
        source[source.index("## Clue Interpretation and Answer Type"):]
    )
    assert receipt["behavioral_neutrality_assumed"] is False
    assert receipt["outcomes_used"] is False

def test_candidate_selection_retains_parent_if_nothing_passes():
    good, bad = assessment(), assessment(child=blocks(0, tokens=70))
    assert method.choose_candidate({"bad": bad}) is None
    assert method.choose_candidate({"bad": bad, "good": good}) == "good"
    assert method.choose_candidate({"b": good, "a": deepcopy(good)}) == "a"
