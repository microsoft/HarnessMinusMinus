from copy import deepcopy
import json

import pytest

from chisel.learning.counterfactual_invocation import (
    build_policy, fit_value_gate, propose, public_state_sha256, retrieve_demonstrations, route,
)


def state(word="alpha"):
    return {
        "task": f"select {word}", "observation": f"{word} available",
        "actions": ["choose alpha", "choose beta"],
        "history": [{"action": "inspect", "observation": f"found {word}"}],
    }


def example(index=0, *, group=None, word="alpha", action="choose alpha"):
    return {
        "example_id": f"example-{index}", "task_id": f"task-{index}",
        "group_id": group or f"group-{index}", "public_state": state(word), "action": action,
    }


def pair(index=0, *, group=None, actor=1, compiled=1, saving=20):
    return {
        "pair_id": f"pair-{index}", "task_id": f"task-{index}",
        "group_id": group or f"group-{index}", "prefix_sha256": "a" * 64,
        "public_state": state(), "candidate_action": "choose alpha",
        "actor_valid": True, "compiled_valid": True, "actor_success": actor,
        "compiled_success": compiled, "actor_tokens": 100, "compiled_tokens": 100 - saving,
    }


def test_hash_normalizes_only_action_order():
    original = state()
    changed = deepcopy(original)
    changed["actions"].reverse()
    assert public_state_sha256(original) == public_state_sha256(changed)
    changed["observation"] += " "
    assert public_state_sha256(original) != public_state_sha256(changed)


@pytest.mark.parametrize("extra", ["answer", "reward", "task_id", "expert_plan"])
def test_future_or_identity_fields_rejected(extra):
    value = {**state(), extra: "secret"}
    with pytest.raises(ValueError):
        public_state_sha256(value)


@pytest.mark.parametrize("actions", [[], ["x", "x"], [True], "x"])
def test_invalid_native_actions_rejected(actions):
    with pytest.raises(ValueError):
        public_state_sha256({**state(), "actions": actions})


def test_history_is_completed_public_information_only():
    value = {**state(), "history": [{"action": "inspect", "reward": 1}]}
    with pytest.raises(ValueError):
        public_state_sha256(value)


def test_empty_native_history_action_is_preserved_without_invention():
    value = {**state(), "history": [{"action": "", "observation": "rejected request"}]}
    assert public_state_sha256(value)
    assert value["history"][0]["action"] == ""
    value["history"][0]["action"] = None
    with pytest.raises(ValueError):
        public_state_sha256(value)


def test_learned_native_action_and_order_invariance():
    policy = build_policy([example(i) for i in range(4)])
    query = state()
    first = propose(policy, query)
    query["actions"].reverse()
    assert propose(policy, query) == first
    assert first["action"] == "choose alpha"
    assert 0 <= first["confidence"] <= 1
    assert first["source_example_ids"]


def test_group_exclusion_precedes_retrieval():
    policy = build_policy([example(i, group="same") for i in range(200)])
    result = propose(policy, state(), excluded_groups={"same"})
    assert result["action"] is None
    assert not result["support_groups"]


def test_excluded_nearest_group_cannot_hide_remaining_evidence():
    examples = [example(i, group="same") for i in range(200)]
    examples.append(example(201, group="different"))
    result = propose(build_policy(examples), state(), excluded_groups={"same"})
    assert result["action"] == "choose alpha"
    assert result["support_groups"] == ["different"]


def test_unknown_state_abstains():
    policy = build_policy([example()])
    query = {
        "task": "unrelated zzz", "observation": "unknown qqq",
        "actions": ["unfamiliar command"], "history": [],
    }
    assert propose(policy, query)["action"] is None


def test_public_retrieval_keeps_group_exclusion_and_bounds():
    policy = build_policy([example(i) for i in range(4)])
    rows = retrieve_demonstrations(policy, state(), excluded_groups={"group-0"}, limit=2)
    assert len(rows) == 2
    assert all(row["group_id"] != "group-0" for _, row in rows)
    with pytest.raises(ValueError):
        retrieve_demonstrations(policy, state(), limit=True)


def test_non_native_and_duplicate_demonstrations_rejected():
    with pytest.raises(ValueError):
        build_policy([example(action="invented")])
    with pytest.raises(ValueError):
        build_policy([example(), example()])


def test_serialization_and_source_mutation_do_not_change_policy():
    examples = [example(i) for i in range(3)]
    policy = build_policy(examples)
    expected = propose(policy, state())
    examples[0]["public_state"]["observation"] = "mutated"
    assert propose(json.loads(json.dumps(policy)), state()) == expected


def test_positive_value_uses_terminal_outcomes():
    gate = fit_value_gate([pair(i, actor=0, compiled=1) for i in range(3)])
    decision = route(gate, state(), {"action": "choose alpha"})
    assert decision["commit"]
    assert decision["estimated_regret"] == -1
    assert decision["estimated_saved_tokens"] == 20


@pytest.mark.parametrize("actor,compiled,saving,reason", [
    (1, 0, 20, "positive_regret"),
    (1, 1, 0, "no_positive_suffix_saving"),
    (1, 1, -20, "no_positive_suffix_saving"),
])
def test_harm_and_full_suffix_cost_are_not_ignored(actor, compiled, saving, reason):
    gate = fit_value_gate([pair(i, actor=actor, compiled=compiled, saving=saving) for i in range(3)])
    decision = route(gate, state(), {"action": "choose alpha"})
    assert not decision["commit"]
    assert decision["reason"] == reason


def test_many_prefixes_from_one_group_are_not_independent():
    gate = fit_value_gate([pair(i, group="same") for i in range(20)])
    decision = route(gate, state(), {"action": "choose alpha"})
    assert not decision["commit"]
    assert decision["support_groups"] == ["same"]


def test_gate_excludes_target_group_and_requires_native_action_match():
    gate = fit_value_gate([pair(i) for i in range(3)])
    assert not route(gate, state(), {"action": "choose alpha"}, excluded_groups={"group-0"})["commit"]
    assert not route(gate, state(), {"action": "choose beta"})["commit"]
    assert not route(gate, state(), {"action": None})["commit"]
    with pytest.raises(ValueError):
        route(gate, state(), {"action": "not native"})


@pytest.mark.parametrize("key,value", [
    ("actor_success", True), ("compiled_success", 2), ("actor_tokens", True),
    ("compiled_tokens", -1), ("actor_valid", "true"), ("compiled_valid", False),
    ("prefix_sha256", "unverified"),
])
def test_invalid_counterfactual_evidence_rejected(key, value):
    record = {**pair(), key: value}
    with pytest.raises(ValueError):
        fit_value_gate([record])


def test_duplicate_pair_rejected():
    with pytest.raises(ValueError):
        fit_value_gate([pair(), pair()])


def test_shuffled_targets_cannot_reintroduce_the_excluded_group():
    records = [{**pair(i), "label_group_id": "held-out-group"} for i in range(3)]
    gate = fit_value_gate(records)
    result = route(gate, state(), {"action": "choose alpha"},
                   excluded_groups={"held-out-group"}, min_groups=1)
    assert not result["commit"]
    assert result["support_groups"] == []


def test_shuffled_label_origins_remain_the_independent_units():
    records = [{**pair(i), "label_group_id": "one-source-group"} for i in range(5)]
    result = route(fit_value_gate(records), state(), {"action": "choose alpha"})
    assert not result["commit"]
    assert result["support_groups"] == ["one-source-group"]


@pytest.mark.parametrize("origin", ["", None, True])
def test_invalid_label_origin_is_rejected(origin):
    with pytest.raises(ValueError):
        fit_value_gate([{**pair(), "label_group_id": origin}])


def test_group_shuffle_preserves_retrieval_support_without_heldout_labels():
    records = [pair(i, actor=0, compiled=1) for i in range(6)]
    for i in range(6, 12):
        row = pair(i, actor=1, compiled=0)
        row["public_state"] = state("beta")
        row["candidate_action"] = "choose beta"
        records.append(row)
    gate = fit_value_gate(records)
    excluded = {"group-11"}
    normal = route(gate, state(), {"action": "choose alpha"}, excluded_groups=excluded)
    shuffled = deepcopy(gate)
    shuffled["label_permutation_seed"] = 20260930
    control = route(shuffled, state(), {"action": "choose alpha"}, excluded_groups=excluded)
    assert normal["estimated_regret"] == -1
    assert control["estimated_regret"] > normal["estimated_regret"]
    assert len(normal["support_groups"]) == len(control["support_groups"]) == 6
    assert not excluded.intersection(control["support_groups"])


def test_group_shuffle_requires_matching_prefix_counts():
    gate = fit_value_gate([pair(0), pair(1, group="group-0"), pair(2)])
    gate["label_permutation_seed"] = 12
    with pytest.raises(ValueError, match="equal prefix counts"):
        route(gate, state(), {"action": "choose alpha"}, min_groups=1)


@pytest.mark.parametrize("kwargs", [
    {"min_groups": True}, {"min_groups": 0}, {"max_harm": True},
    {"max_harm": float("nan")}, {"max_harm": 2},
])
def test_gate_parameters_are_strict(kwargs):
    gate = fit_value_gate([pair()])
    with pytest.raises(ValueError):
        route(gate, state(), {"action": "choose alpha"}, **kwargs)


def test_ids_are_not_predictive_text():
    before = [example(i) for i in range(3)]
    after = deepcopy(before)
    for row in after:
        row["task_id"] += " new unknown task words"
        row["group_id"] += " renamed"
    first = propose(build_policy(before), state())
    second = propose(build_policy(after), state())
    assert first["action"] == second["action"]
    assert first["confidence"] == second["confidence"]
