from copy import deepcopy
import hashlib
import json
import runpy
import sys
from types import ModuleType

import pytest

from chisel.learning import counterfactual_invocation as invocation
from chisel.learning.fragment_contracts import compile_library
from chisel.learning import residual_programs as residual


TASK = "alpha beta gamma delta other actor prior"
ACTIONS = ["alpha", "beta", "gamma", "delta", "other", "actor", "prior"]
ORIGINAL_MODES = (
    "anchored_guarded_consensus", "anchored_consensus", "anchored_single",
    "unanchored_guarded_consensus",
)
ANCHORED = ORIGINAL_MODES[:-1]


def state(observation="s0", *, actions=None, task=TASK, history=None):
    return {
        "task": task, "observation": observation, "actions": list(ACTIONS if actions is None else actions),
        "history": [] if history is None else history,
    }


def corpus(name="p", *, actions=("alpha", "beta"), groups=("g0", "g1"), prefixes=1,
           observations=None, history=(), task=TASK, family="opaque-family"):
    examples, demos, transitions = [], [], {}
    for group in groups:
        for prefix in range(prefixes):
            task_id = f"{name}-{group}-{prefix}"
            ids = [f"{task_id}-{step}" for step in range(len(actions))]
            path = (observations(group, prefix) if callable(observations)
                    else list(observations) if observations is not None
                    else [f"s{step}" for step in range(len(actions) + 1)])
            examples.append({
                "task_id": task_id, "group_id": group, "start_turn": len(history),
                "demonstration_ids": ids, "task_family": family,
            })
            for step, action in enumerate(actions):
                demos.append({
                    "example_id": ids[step], "task_id": task_id, "group_id": group,
                    "action": action, "public_state": state(path[step], task=task),
                })
                transitions[ids[step]] = {
                    "post_observation": path[step + 1], "pre_action_history": list(history) + list(actions[:step]),
                    "task_family": family,
                }
    return [{"program_id": name, "actions": list(actions), "examples": examples}], demos, transitions


def prepared(*bundles, excluded_groups=frozenset()):
    programs, demos, transitions = [], [], {}
    for candidates, sources, feedback in bundles or (corpus(),):
        programs.extend(candidates)
        demos.extend(sources)
        transitions.update(feedback)
    library = compile_library(programs, demos, transitions, [], excluded_groups=excluded_groups)
    return library, invocation.build_policy(demos), demos


def controller(library, policy, mode="anchored_guarded_consensus", **kwargs):
    return residual.ResidualProgramController(
        library, policy, mode=mode,
        **{"group_id": "target", "task_family": "opaque-family", **kwargs},
    )


def retrieval(monkeypatch, rows):
    calls = []

    def retrieve(policy, public_state, *, excluded_groups=frozenset(), limit=128):
        invocation.validate_public_state(public_state)
        calls.append({"excluded_groups": excluded_groups, "limit": limit})
        return [(score, row) for score, row in rows if row["group_id"] not in excluded_groups][:limit]

    monkeypatch.setattr(invocation, "retrieve_demonstrations", retrieve)
    return calls


def starts(library, demos, score=1.0):
    return [(score, row) for row in demos if row["example_id"] in library["entry_index"]]


def instance(library, name):
    return next(program["instance_id"] for program in library["programs"] if program["program_id"] == name)


def anchor(control, action="alpha", observation="s1", before=None):
    result = control.decide(state() if before is None else before)
    assert not result["decision"]["commit"]
    assert result["proposal"]["action"] is None
    return result, control.observe(action, observation, compiled=False)


@pytest.mark.parametrize("mode", ANCHORED)
def test_initial_decision_waits_for_actual_actor_and_never_advances_speculatively(mode):
    library, policy, demos = prepared()
    control = controller(library, policy, mode)
    with pytest.raises(ValueError, match="decide must precede"):
        control.observe("alpha", "s1", compiled=False)
    result = control.decide(deepcopy(demos[0]["public_state"]))
    assert result["proposal"] == {"action": None}
    assert result["decision"]["reason"] == "actor_anchor_required"
    assert all(h["step"] == 0 for h in result["decision"]["anchor_candidates"])
    with pytest.raises(ValueError, match="observe must follow"):
        control.decide(state())
    with pytest.raises(ValueError, match="pending committed"):
        control.observe("alpha", "s1", compiled=True)
    feedback = control.observe("alpha", "s1", compiled=False)
    assert feedback["reason"] == "actor_anchor_accepted"
    assert feedback["hypotheses"][0]["step"] == 1
    assert feedback["hypotheses"][0]["anchor_turn"] == 0
    next_decision = control.decide(state("s1"))
    assert next_decision["decision"]["commit"]
    assert next_decision["proposal"]["action"] == "beta"
    assert next_decision["decision"]["hypotheses"][0]["step"] == 1


@pytest.mark.parametrize("mode", ANCHORED)
def test_actual_actor_action_selects_a_different_suffix_than_highest_retrieval(monkeypatch, mode):
    library, policy, demos = prepared(
        corpus("highest", actions=("alpha", "beta"), groups=("a0", "a1")),
        corpus("actor_choice", actions=("other", "gamma"), groups=("b0", "b1")),
    )
    rows = [(0.99 if row["group_id"].startswith("a") else 0.70, row)
            for _, row in starts(library, demos)]
    retrieval(monkeypatch, rows)
    control = controller(library, policy, mode)
    decision, feedback = anchor(control, action="other")
    assert decision["decision"]["anchor_candidates"][0]["program_id"] == "highest"
    assert feedback["support_groups"] == ["b0", "b1"]
    assert feedback["hypothesis_count"] == 1
    assert feedback["hypotheses"][0]["program_id"] == "actor_choice"
    assert any(row["reason"] == "actor_action_mismatch" for row in feedback["rejections"])
    result = control.decide(state("s1"))
    assert result["decision"]["commit"]
    assert result["proposal"]["action"] == "gamma"


def test_disagreement_defers_and_actor_reanchors_from_current_not_hypothetical_state(monkeypatch):
    library, policy, demos = prepared(
        corpus("one", actions=("alpha", "beta")),
        corpus("two", actions=("alpha", "gamma")),
        corpus("fresh", actions=("other", "delta"), observations=("s1", "fresh", "done")),
    )
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy)
    anchor(control)
    result = control.decide(state("s1"))
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "ambiguous_continuations"
    assert [row["action"] for row in result["decision"]["competing_actions"]] == ["beta", "gamma"]
    assert all(h["step"] == 0 for h in result["decision"]["anchor_candidates"])
    observed = control.observe("other", "fresh", compiled=False)
    assert observed["hypothesis_count"] == 1
    assert observed["hypotheses"][0]["program_id"] == "fresh"
    assert observed["hypotheses"][0]["step"] == 1
    assert observed["hypotheses"][0]["anchor_turn"] == 1
    assert observed["completed_action_count"] == 2
    assert control.decide(state("fresh"))["proposal"]["action"] == "delta"


def test_agreeing_instances_use_retrieved_independent_groups_not_program_provenance(monkeypatch):
    library, policy, demos = prepared(corpus("one"), corpus("two"))
    one = next(row for row in demos if row["task_id"] == "one-g0-0" and row["action"] == "alpha")
    two = next(row for row in demos if row["task_id"] == "two-g1-0" and row["action"] == "alpha")
    retrieval(monkeypatch, [(0.9, one), (0.8, two)])
    control = controller(library, policy)
    _, feedback = anchor(control)
    assert feedback["hypothesis_count"] == 2
    assert feedback["source_count"] == 2
    assert feedback["support_groups"] == ["g0", "g1"]
    assert all(len(h["support_groups"]) == 1 for h in feedback["hypotheses"])
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"
    single = controller(library, policy, "anchored_single")
    _, feedback = anchor(single)
    assert not feedback["hypotheses"]
    assert all(row["reason"] == "single_instance_insufficient_groups" for row in feedback["rejections"])
    assert not single.decide(state("s1"))["decision"]["commit"]


def test_many_retrieved_prefixes_from_one_group_cannot_supply_the_missing_group(monkeypatch):
    library, policy, demos = prepared(corpus(prefixes=6))
    retrieval(monkeypatch, [(score, row) for score, row in starts(library, demos)
                            if row["group_id"] == "g0"])
    control = controller(library, policy)
    _, feedback = anchor(control)
    assert feedback["source_count"] == 6
    assert feedback["support_groups"] == ["g0"]
    assert feedback["actor_required"]
    assert feedback["boundary_reason"] == "insufficient_source_groups"
    result = control.decide(state("s1"))
    assert not result["decision"]["commit"]
    assert result["decision"]["boundary_reason"] == "insufficient_source_groups"


def test_self_and_declared_excluded_groups_never_support_retrieval(monkeypatch):
    library, policy, demos = prepared(
        corpus(groups=("target", "held", "g0", "g1")), excluded_groups={"target", "held"},
    )
    rows = [(1.0, row) for row in demos if row["action"] == "alpha"]
    calls = retrieval(monkeypatch, rows)
    control = controller(library, policy)
    first, feedback = anchor(control)
    assert calls == [{"excluded_groups": frozenset({"target", "held"}), "limit": 128}]
    assert feedback["support_groups"] == ["g0", "g1"]
    assert first["decision"]["source_count"] == 2
    assert control.decide(state("s1"))["decision"]["support_group_count"] == 2


@pytest.mark.parametrize("groups", [("target", "g0"), ("g0", "g1")])
def test_target_must_not_supply_fitted_guard_or_projection_evidence(groups):
    library, policy, _ = prepared(corpus(groups=groups))
    if "target" in groups:
        with pytest.raises(ValueError, match="recompile"):
            controller(library, policy)
    else:
        controller(library, policy)


def test_ended_short_program_vetoes_a_supported_longer_extension(monkeypatch):
    library, policy, demos = prepared(
        corpus("short", actions=("alpha", "beta")),
        corpus("long", actions=("alpha", "beta", "gamma")),
    )
    retrieval(monkeypatch, [
        (score, row) for score, row in starts(library, demos)
        if not (row["task_id"].startswith("short") and row["group_id"] == "g1")
    ])
    control = controller(library, policy)
    anchor(control)
    result = control.decide(state("s1"))
    assert result["proposal"]["action"] == "beta"
    feedback = control.observe("beta", "s2", compiled=True)
    assert feedback["completed_instance_ids"] == [instance(library, "short")]
    assert feedback["hypothesis_count"] == 2
    assert feedback["competing_actions"][0]["action"] == "gamma"
    assert feedback["actor_required"]
    result = control.decide(state("s2"))
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "actor_boundary_required"
    assert result["decision"]["boundary_reason"] == "completed_hypothesis"
    control.observe("other", "s2", compiled=False)
    assert not control.decide(state("s2"))["decision"]["commit"]


def test_precondition_pruning_can_resolve_disagreement_with_independent_survivors(monkeypatch):
    library, policy, demos = prepared(
        corpus("one", actions=("alpha", "beta"), observations=("s0", "s1", "z")),
        corpus("two", actions=("alpha", "gamma"), observations=("s0", "s1", "y")),
    )
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy)
    anchor(control)
    result = control.decide(state("s1 z"))
    assert result["decision"]["commit"]
    assert result["proposal"]["action"] == "gamma"
    assert result["decision"]["hypothesis_count"] == 1
    assert result["decision"]["rejections"][0]["reason"] == "step_precondition_failed"


def test_postcondition_pruning_retains_only_feedback_consistent_continuations(monkeypatch):
    library, policy, demos = prepared(
        corpus("one", actions=("alpha", "beta", "gamma"), observations=("s0", "s1", "a", "end")),
        corpus("two", actions=("alpha", "beta", "delta"), observations=("s0", "s1", "b", "end")),
    )
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy)
    anchor(control)
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"
    observed = control.observe("beta", "a", compiled=True)
    assert observed["hypothesis_count"] == 1
    assert observed["hypotheses"][0]["program_id"] == "one"
    assert observed["hypotheses"][0]["step"] == 2
    assert observed["support_groups"] == ["g0", "g1"]
    assert not observed["actor_required"]
    assert any(row["reason"] == "step_postcondition_failed" for row in observed["rejections"])
    assert control.decide(state("a"))["proposal"]["action"] == "gamma"


def test_postcondition_uses_the_same_pre_matched_clause_not_any_post_clause(monkeypatch):
    library, policy, demos = prepared(corpus(
        groups=("g0", "g1", "g2", "g3"),
        observations=lambda group, _: ("", "u", "v") if group in ("g0", "g1") else ("r", "", "v"),
    ))
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy)
    first = control.decide(state(""))
    assert first["decision"]["anchor_candidates"][0]["precondition_clauses"] == [0]
    feedback = control.observe("alpha", "", compiled=False)
    assert feedback["checks"][0]["postcondition_clauses"] == []
    assert not feedback["hypotheses"]
    assert not control.decide(state(""))["decision"]["commit"]


def test_failed_local_postcondition_returns_to_actor_without_advancing_or_undoing():
    library, policy, _ = prepared(corpus(actions=("alpha", "beta", "gamma")))
    control = controller(library, policy)
    anchor(control)
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"
    feedback = control.observe("beta", "s1", compiled=True)
    assert not feedback["hypotheses"]
    assert feedback["checks"][0]["passed"] is False
    assert feedback["checks"][0]["step"] == 1
    assert feedback["completed_action_count"] == 2
    assert feedback["native_action_count"] == 2
    assert feedback["actor_required"]
    result = control.decide(state("s1"))
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "actor_boundary_required"


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
def test_every_local_action_requires_current_native_membership(monkeypatch, mode):
    library, policy, demos = prepared()
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy, mode)
    first = control.decide(state())
    control.observe("alpha", "s1", compiled=first["decision"]["commit"])
    result = control.decide(state("s1", actions=["alpha", "actor"]))
    assert not result["decision"]["commit"]
    assert result["proposal"]["action"] is None
    assert any(row["reason"] == "next_action_not_native" for row in result["decision"]["rejections"])


def test_unanchored_mode_can_start_only_on_agreement_and_preserves_actor_boundary():
    library, policy, _ = prepared()
    control = controller(library, policy, "unanchored_guarded_consensus")
    first = control.decide(state())
    assert first["decision"]["commit"]
    assert first["decision"]["reason"] == "local_entry_consensus"
    assert first["proposal"]["action"] == "alpha"
    control.observe("alpha", "s1", compiled=True)
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"
    feedback = control.observe("beta", "s2", compiled=True)
    assert feedback["actor_required"]
    assert control.decide(state())["decision"]["reason"] == "actor_boundary_required"


def test_unanchored_disagreement_still_allows_actual_actor_to_choose_an_anchor(monkeypatch):
    library, policy, demos = prepared(corpus("one"), corpus("two", actions=("other", "gamma")))
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy, "unanchored_guarded_consensus")
    first = control.decide(state())
    assert not first["decision"]["commit"]
    assert first["decision"]["reason"] == "ambiguous_continuations"
    feedback = control.observe("other", "s1", compiled=False)
    assert feedback["hypotheses"][0]["program_id"] == "two"
    assert control.decide(state("s1"))["proposal"]["action"] == "gamma"


def test_guardless_mode_removes_only_steps_not_actor_anchoring_support_or_consensus(monkeypatch):
    library, policy, demos = prepared(corpus("one"), corpus("two", actions=("alpha", "gamma")))
    retrieval(monkeypatch, starts(library, demos))
    guarded = controller(library, policy)
    unguarded = controller(library, policy, "anchored_consensus")
    guarded_first, unguarded_first = guarded.decide(state("s1")), unguarded.decide(state("s1"))
    assert not guarded_first["decision"]["commit"] and not unguarded_first["decision"]["commit"]
    assert guarded_first["decision"]["anchor_candidates"] == []
    assert len(unguarded_first["decision"]["anchor_candidates"]) == 2
    guarded.observe("alpha", "no change", compiled=False)
    feedback = unguarded.observe("alpha", "no change", compiled=False)
    assert all(not check["checked"] and check["passed"] is None for check in feedback["checks"])
    assert feedback["support_group_count"] == 2
    result = unguarded.decide(state("no change"))
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "ambiguous_continuations"


def test_single_mode_selects_one_supported_actor_compatible_instance_deterministically(monkeypatch):
    library, policy, demos = prepared(
        corpus("short"), corpus("z-long", actions=("alpha", "gamma", "delta")),
        corpus("a-long", actions=("alpha", "beta", "gamma")),
    )
    retrieval(monkeypatch, list(reversed(starts(library, demos, 0.8))))
    control = controller(library, policy, "anchored_single")
    _, feedback = anchor(control)
    assert feedback["hypothesis_count"] == 1
    assert feedback["hypotheses"][0]["program_id"] == "a-long"
    assert feedback["support_group_count"] == 2
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"
    consensus = controller(library, policy)
    anchor(consensus)
    assert consensus.decide(state("s1"))["decision"]["reason"] == "ambiguous_continuations"


def test_single_skips_higher_ranked_instances_without_two_actual_retrieved_groups(monkeypatch):
    library, policy, demos = prepared(corpus("high"), corpus("supported", actions=("alpha", "gamma")))
    rows = [(0.99 if row["task_id"].startswith("high") else 0.7, row)
            for _, row in starts(library, demos)
            if not (row["task_id"].startswith("high") and row["group_id"] == "g1")]
    retrieval(monkeypatch, rows)
    control = controller(library, policy, "anchored_single")
    _, feedback = anchor(control)
    assert feedback["hypotheses"][0]["program_id"] == "supported"
    assert any(row["reason"] == "single_instance_insufficient_groups" for row in feedback["rejections"])


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
def test_first_action_non_native_family_and_goal_bindings_apply_to_all_modes(monkeypatch, mode):
    library, policy, demos = prepared()
    retrieval(monkeypatch, starts(library, demos))
    for query, reason in (
        (state(actions=["actor"]), "first_action_not_native"),
        (state(task="alpha only"), "source_goal_words_missing"),
    ):
        result = controller(library, policy, mode).decide(query)
        assert not result["decision"]["commit"]
        assert any(row["reason"] == reason for row in result["decision"]["rejections"])
    result = controller(library, policy, mode, task_family="other-family").decide(state())
    assert not result["decision"]["commit"]
    assert all(row["reason"] == "task_family_mismatch" for row in result["decision"]["rejections"])


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
def test_prerequisites_use_only_actual_native_feedback_and_never_instruction_history(monkeypatch, mode):
    library, policy, demos = prepared(corpus(history=("prior",)))
    retrieval(monkeypatch, starts(library, demos))
    query = state("instructions prior", history=[{"action": "prior", "observation": "prior"}])
    control = controller(library, policy, mode)
    result = control.decide(query)
    assert not result["decision"]["commit"]
    assert any(row["reason"] == "history_prerequisites_missing"
               for row in result["decision"]["rejections"])
    control.observe("prior", "s0", compiled=False, native_valid=False)
    assert not control.decide(query)["decision"]["commit"]
    control.observe("prior", "s0", compiled=False)
    result = control.decide(state())
    assert result["decision"]["native_action_count"] == 1
    if mode == "unanchored_guarded_consensus":
        assert result["decision"]["commit"]
    else:
        assert result["decision"]["anchor_candidates"]
        assert not result["decision"]["commit"]


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
def test_net_guards_are_not_enforced_even_when_no_net_witness_exists(monkeypatch, mode):
    library, policy, demos = prepared(corpus(history=("alpha", "beta"), observations=("s0", "s1", "s0")))
    assert library["programs"][0]["positive_net_guard"] is None
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy, mode)
    for action in ("alpha", "beta"):
        assert not control.decide(state())["decision"]["commit"]
        control.observe(action, "s0", compiled=False)
    first = control.decide(state())
    assert first["decision"]["net_checks"] is False
    control.observe("alpha", "s1", compiled=first["decision"]["commit"])
    result = control.decide(state("s1"))
    assert result["decision"]["commit"]
    feedback = control.observe("beta", "s0", compiled=True)
    assert feedback["boundary_reason"] == "completed_hypothesis"
    assert feedback["completed_instance_ids"]
    assert feedback["net_checks"] is False


def test_unanchored_seen_entry_ignores_irrelevant_history_but_actor_can_reauthorize(monkeypatch):
    library, policy, demos = prepared(corpus(history=("alpha", "beta")))
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy, "unanchored_guarded_consensus")
    for action in ("alpha", "beta"):
        assert not control.decide(state())["decision"]["commit"]
        control.observe(action, "s0", compiled=False)
    for before, after in (("s0", "s1"), ("s1", "s2")):
        result = control.decide(state(before))
        assert result["decision"]["commit"]
        control.observe(result["proposal"]["action"], after, compiled=True)
    assert not control.decide(state())["decision"]["commit"]
    control.observe("other", "s0", compiled=False)
    changed_history = state(history=[{"action": "other", "observation": "s0"}])
    blocked = control.decide(changed_history)
    assert not blocked["decision"]["commit"]
    assert any(row["reason"] == "repeated_local_entry" for row in blocked["decision"]["rejections"])
    feedback = control.observe("alpha", "s1", compiled=False)
    assert feedback["reason"] == "actor_anchor_accepted"
    assert feedback["hypotheses"][0]["step"] == 1
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"


def test_fresh_actor_reauthorizes_anchored_residual_even_at_the_same_entry_key(monkeypatch):
    library, policy, demos = prepared(corpus(history=("alpha", "beta")))
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy)
    for action in ("alpha", "beta"):
        control.decide(state())
        control.observe(action, "s0", compiled=False)
    keys = []
    for _ in range(2):
        first = control.decide(state())
        assert not first["decision"]["commit"]
        feedback = control.observe("alpha", "s1", compiled=False)
        keys.append(feedback["hypotheses"][0]["entry_key"])
        assert control.decide(state("s1"))["decision"]["commit"]
        control.observe("beta", "s2", compiled=True)
    assert keys[0] == keys[1]


def test_actor_override_of_local_proposal_reanchors_without_using_hypothetical_action(monkeypatch):
    library, policy, demos = prepared(
        corpus("old", actions=("alpha", "beta", "gamma")),
        corpus("new", actions=("other", "delta"), observations=("s1", "fresh", "done")),
    )
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy)
    anchor(control)
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"
    feedback = control.observe("other", "fresh", compiled=False)
    assert feedback["native_action_count"] == 2
    assert feedback["hypotheses"][0]["program_id"] == "new"
    assert feedback["hypotheses"][0]["step"] == 1
    assert control.decide(state("fresh"))["proposal"]["action"] == "delta"


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
def test_action_list_order_and_serialized_inputs_do_not_change_outputs(mode):
    library, policy, demos = prepared()
    original = controller(library, policy, mode)
    restored = controller(json.loads(json.dumps(library)), json.loads(json.dumps(policy)), mode)
    first = deepcopy(demos[0]["public_state"])
    reverse = deepcopy(first)
    reverse["actions"].reverse()
    a, b = original.decide(first), restored.decide(reverse)
    assert a == b
    assert a == json.loads(json.dumps(a, allow_nan=False))
    one = original.observe("alpha", "s1", compiled=a["decision"]["commit"])
    two = restored.observe("alpha", "s1", compiled=b["decision"]["commit"])
    assert one == two
    assert one == json.loads(json.dumps(one, allow_nan=False))
    assert original.decide(state("s1")) == restored.decide(
        state("s1", actions=list(reversed(ACTIONS))),
    )


def test_input_and_return_mutation_cannot_change_pending_or_future_control():
    library, policy, _ = prepared()
    control = controller(library, policy)
    query = state()
    result = control.decide(query)
    result["decision"]["anchor_candidates"][0]["sources"].clear()
    query["actions"].clear()
    library["programs"][0]["actions"][1] = "mutated"
    policy["records"][0]["action"] = "mutated"
    observed = control.observe("alpha", "s1", compiled=False)
    observed["hypotheses"][0]["sources"].clear()
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
def test_episode_reset_clears_history_residual_counter_and_registry(mode):
    library, policy, _ = prepared()
    control = controller(library, policy, mode)
    first = control.decide(state())
    with pytest.raises(ValueError, match="pending"):
        control.reset_episode()
    control.observe("alpha", "s1", compiled=first["decision"]["commit"])
    control.reset_episode()
    assert control.decide(state()) == controller(library, policy, mode).decide(state())


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"mode": True}, {"mode": []},
    {"group_id": True}, {"group_id": ""}, {"task_family": None}, {"task_family": True},
    {"entry_threshold": True}, {"entry_threshold": None}, {"entry_threshold": "0.5"},
    {"entry_threshold": float("nan")}, {"entry_threshold": float("inf")},
    {"entry_threshold": -0.1}, {"entry_threshold": 1.1},
    {"minimum_groups": True}, {"minimum_groups": 0}, {"minimum_groups": 1},
    {"minimum_groups": 2.0}, {"minimum_groups": None},
])
def test_strict_constructor_types_and_parameters(kwargs):
    library, policy, _ = prepared()
    with pytest.raises(ValueError):
        controller(library, policy, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"compiled": 0}, {"compiled": 1}, {"compiled": None},
    {"native_valid": 0}, {"native_valid": 1}, {"native_valid": "true"},
    {"action": None}, {"action": True}, {"observation": False}, {"observation": None},
])
def test_invalid_feedback_preserves_pending_decision(kwargs):
    library, policy, _ = prepared()
    control = controller(library, policy)
    control.decide(state())
    with pytest.raises(ValueError):
        control.observe(**{"action": "alpha", "observation": "s1", "compiled": False, **kwargs})
    assert control.observe("alpha", "s1", compiled=False)["hypotheses"][0]["step"] == 1


def test_invalid_native_and_mismatched_compiled_actions_do_not_consume_pending_feedback():
    library, policy, _ = prepared()
    control = controller(library, policy)
    control.decide(state())
    with pytest.raises(ValueError, match="native action set"):
        control.observe("not a member", "s1", compiled=False)
    control.observe("alpha", "s1", compiled=False)
    control.decide(state("s1"))
    with pytest.raises(ValueError, match="pending committed"):
        control.observe("other", "s2", compiled=True)
    feedback = control.observe("beta", "s2", compiled=True)
    assert feedback["completed_instance_ids"]
    with pytest.raises(ValueError, match="decide must precede"):
        control.observe("beta", "s2", compiled=True)


def test_non_native_feedback_does_not_anchor_or_add_history_facts():
    library, policy, _ = prepared(corpus(history=("prior",)))
    control = controller(library, policy)
    control.decide(state())
    feedback = control.observe("prior", "s0", compiled=False, native_valid=False)
    assert feedback["native_action_count"] == 0
    assert feedback["reason"] == "actor_action_not_native"
    result = control.decide(state())
    assert not result["decision"]["anchor_candidates"]
    assert all(row["reason"] == "history_prerequisites_missing" for row in result["decision"]["rejections"])


@pytest.mark.parametrize("extra", ["official_success", "task_id", "reward", "future_action"])
def test_nonpublic_state_is_rejected_without_consuming_a_turn(extra):
    library, policy, _ = prepared()
    control = controller(library, policy)
    with pytest.raises(ValueError, match="Public state"):
        control.decide({**state(), extra: "hidden"})
    assert control.decide(state())["decision"]["turn"] == 0


@pytest.mark.parametrize("mutation", [
    lambda lib: lib.update(schema_version="unknown"),
    lambda lib: lib.update(entry_index={"unbound": ["missing"]}),
    lambda lib: lib["parameters"].update(minimum_groups=True),
    lambda lib: lib["programs"][0]["steps"][0]["positive_guard"].update(clauses=[]),
])
def test_library_schema_and_positive_guard_validation(mutation):
    library, policy, _ = prepared()
    mutation(library)
    with pytest.raises(ValueError):
        controller(library, policy)


@pytest.mark.parametrize("mutation", [
    lambda policy: policy.update(schema_version="unknown"),
    lambda policy: policy.update(records=[]),
    lambda policy: policy.update(records=[True]),
    lambda policy: policy["index"].update(with_action=True),
    lambda policy: policy["index"].update(idf={}),
])
def test_memory_policy_requires_the_validated_source_index(mutation):
    library, policy, _ = prepared()
    mutation(policy)
    with pytest.raises(ValueError):
        controller(library, policy)


def test_retrieved_source_cannot_be_rebound_to_an_existing_library_id(monkeypatch):
    library, policy, demos = prepared()
    row = deepcopy(demos[0])
    row["public_state"]["observation"] = "changed source"
    retrieval(monkeypatch, [(1.0, row)])
    with pytest.raises(ValueError, match="compiled source"):
        controller(library, policy).decide(state())


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
@pytest.mark.parametrize("length", [2, 3, 4])
def test_local_streak_is_bounded_by_actual_source_length_with_a_mandatory_actor_boundary(mode, length):
    actions = ("alpha", "beta", "gamma", "delta")[:length]
    library, policy, _ = prepared(corpus(actions=actions))
    control = controller(library, policy, mode)
    local_steps = []
    for step, action in enumerate(actions):
        decision = control.decide(state(f"s{step}"))
        commit = decision["decision"]["commit"]
        assert commit is (step > 0 or mode == "unanchored_guarded_consensus")
        if commit:
            assert decision["proposal"]["action"] == action
            local_steps.append(step)
        observed = control.observe(action, f"s{step + 1}", compiled=commit)
        assert observed["completed_action_count"] == step + 1
    assert local_steps == list(range(0 if mode == "unanchored_guarded_consensus" else 1, length))
    assert observed["completed_instance_ids"]
    assert observed["hypotheses"][0]["step"] == length
    assert control.decide(state(f"s{length}"))["decision"]["reason"] == "actor_boundary_required"


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
@pytest.mark.parametrize("second_score,accepted", [(0.55, True), (0.549, False)])
def test_threshold_filters_retrieved_support_before_group_accounting(monkeypatch, mode, second_score, accepted):
    library, policy, demos = prepared()
    rows = [(1.0 if row["group_id"] == "g0" else second_score, row)
            for _, row in starts(library, demos)]
    retrieval(monkeypatch, rows)
    control = controller(library, policy, mode)
    first = control.decide(state())
    control.observe("alpha", "s1", compiled=first["decision"]["commit"])
    next_decision = control.decide(state("s1"))
    assert next_decision["decision"]["commit"] is accepted
    if not accepted:
        assert any(row["reason"] == "below_entry_threshold" for row in first["decision"]["rejections"])


def test_retrieval_duplicates_do_not_inflate_hypotheses_sources_or_group_support(monkeypatch):
    library, policy, demos = prepared()
    row = demos[0]
    retrieval(monkeypatch, [(0.9, row)] * 12)
    control = controller(library, policy)
    first, feedback = anchor(control)
    assert first["decision"]["hypothesis_count"] == 1
    assert first["decision"]["source_count"] == 1
    assert first["decision"]["support_group_count"] == 1
    assert sum(row["reason"] == "duplicate_source_reference"
               for row in first["decision"]["rejections"]) == 11
    assert feedback["actor_required"]
    assert not control.decide(state("s1"))["decision"]["commit"]


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
def test_actual_actor_unmatched_action_does_not_invent_an_anchor_or_a_future_counter(mode):
    library, policy, _ = prepared()
    control = controller(library, policy, mode)
    control.decide(state())
    observed = control.observe("other", "s0", compiled=False)
    assert observed["reason"] == "actor_has_no_supported_anchor"
    assert not observed["hypotheses"]
    assert observed["actor_required"] is (mode != "unanchored_guarded_consensus")
    next_decision = control.decide(state())
    assert next_decision["decision"]["commit"] is (mode == "unanchored_guarded_consensus")
    assert all(h["step"] == 0 for h in next_decision["decision"]["hypotheses"])


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
def test_non_native_compiled_feedback_is_an_error_without_consuming_pending_decision(mode):
    library, policy, _ = prepared()
    control = controller(library, policy, mode)
    first = control.decide(state())
    control.observe("alpha", "s1", compiled=first["decision"]["commit"])
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"
    pending = deepcopy(control._pending)
    with pytest.raises(ValueError, match="Compiled feedback must be native-valid"):
        control.observe("beta", "s2", compiled=True, native_valid=False)
    assert control._pending == pending
    assert control._turn == control._native_count == 1
    assert control._native_actions == {"alpha"}
    result = control.observe("beta", "s2", compiled=True, native_valid=True)
    assert result["completed_action_count"] == 2
    assert result["native_action_count"] == 2
    assert result["completed_instance_ids"]
    assert result["actor_required"]
    assert control.decide(state("s2"))["decision"]["reason"] == "actor_boundary_required"


@pytest.mark.parametrize("mode", residual.MODES)
def test_empty_library_returns_explicit_actor_deferral(mode):
    library = compile_library([], [], {}, [])
    _, policy, _ = prepared()
    result = controller(library, policy, mode).decide(state())
    assert result["proposal"]["action"] is None
    assert not result["decision"]["commit"]
    assert result["decision"]["hypotheses"] == result["decision"]["anchor_candidates"] == []
    assert result["decision"]["support_group_count"] == 0


def test_single_source_dissenter_vetoes_a_larger_majority(monkeypatch):
    library, policy, demos = prepared(
        corpus("majority", groups=("g0", "g1", "g2")),
        corpus("minority", actions=("alpha", "gamma"), groups=("g3", "g4")),
    )
    retrieval(monkeypatch, [(score, row) for score, row in starts(library, demos)
                            if row["group_id"] != "g4"])
    control = controller(library, policy)
    anchor(control)
    result = control.decide(state("s1"))
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "ambiguous_continuations"
    groups = {row["action"]: row["support_groups"] for row in result["decision"]["competing_actions"]}
    assert groups == {"beta": ["g0", "g1", "g2"], "gamma": ["g3"]}


def test_feedback_pruning_below_group_minimum_requires_actor_instead_of_invented_program_support(
    monkeypatch,
):
    library, policy, demos = prepared(
        corpus("one", actions=("alpha", "beta", "gamma"), observations=("s0", "s1", "a", "end")),
        corpus("two", actions=("alpha", "beta", "gamma"), observations=("s0", "s1", "b", "end")),
    )
    retrieval(monkeypatch, [(score, row) for score, row in starts(library, demos)
                            if (row["task_id"].startswith("one") and row["group_id"] == "g0")
                            or (row["task_id"].startswith("two") and row["group_id"] == "g1")])
    control = controller(library, policy)
    anchor(control)
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"
    feedback = control.observe("beta", "a", compiled=True)
    assert feedback["hypothesis_count"] == 1
    assert feedback["support_groups"] == ["g0"]
    assert feedback["boundary_reason"] == "insufficient_source_groups"
    assert not control.decide(state("a"))["decision"]["commit"]


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
def test_modes_share_family_entry_pool_when_step_preconditions_hold(monkeypatch, mode):
    library, policy, demos = prepared(corpus("a"), corpus("b"))
    retrieval(monkeypatch, starts(library, demos))
    result = controller(library, policy, mode).decide(state())
    assert result["decision"]["hypothesis_count"] == 2
    assert result["decision"]["source_count"] == 4
    assert result["decision"]["support_groups"] == ["g0", "g1"]
    assert {h["instance_id"] for h in result["decision"]["hypotheses"]} == {
        instance(library, "a"), instance(library, "b"),
    }
    assert result["decision"]["net_checks"] is False
    assert result["decision"]["step_guard_variant"] == (None if mode == "anchored_consensus" else "positive")


def test_explicit_native_family_is_required_and_never_inferred_from_task_text(monkeypatch):
    library, policy, demos = prepared(corpus(family=None))
    retrieval(monkeypatch, starts(library, demos))
    result = controller(library, policy).decide(state(task=TASK + " opaque-family"))
    assert not result["decision"]["commit"]
    assert all(row["reason"] == "source_task_family_missing"
               for row in result["decision"]["rejections"])
    library, policy, demos = prepared(corpus(family="completely arbitrary native label"))
    retrieval(monkeypatch, starts(library, demos))
    control = controller(library, policy, task_family="completely arbitrary native label")
    anchor(control)
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"


@pytest.mark.parametrize("mode", ORIGINAL_MODES)
def test_exact_public_decision_feedback_and_source_audit_keys(mode):
    library, policy, _ = prepared()
    control = controller(library, policy, mode)
    common = {
        "schema_version", "mode", "turn", "library_behavior_hash", "memory_policy_hash",
        "group_id", "task_family", "entry_threshold", "minimum_groups", "step_guard_variant",
        "net_checks", "native_action_count", "seen_entry_count",
    }
    summary = {
        "hypotheses", "hypothesis_count", "source_count", "support_groups",
        "support_group_count", "completed_instance_ids", "competing_actions",
    }
    first = control.decide(state())
    assert set(first) == {"proposal", "decision"}
    assert set(first["proposal"]) == {"action"}
    assert set(first["decision"]) == common | summary | {
        "commit", "reason", "state_sha256", "boundary_reason", "anchor_candidates", "rejections",
    }
    hypothesis = first["decision"]["hypotheses"][0]
    assert set(hypothesis) == {
        "instance_id", "program_id", "step", "length", "next_action", "anchor_turn",
        "entry_key", "precondition_clauses", "sources", "support_groups",
    }
    assert set(hypothesis["sources"][0]) == {"example_id", "group_id", "similarity"}
    assert set(first["decision"]["competing_actions"][0]) == {
        "action", "support_groups", "instance_ids", "source_count",
    }
    observed = control.observe("alpha", "s1", compiled=first["decision"]["commit"])
    assert set(observed) == common | summary | {
        "action", "observation", "compiled", "native_valid", "reason", "checks",
        "rejections", "actor_required", "boundary_reason", "completed_action_count",
    }
    assert set(observed["checks"][0]) == {
        "instance_id", "step", "support_groups", "checked", "passed",
        "precondition_clauses", "postcondition_clauses",
    }
    decision = control.decide(state("s1", actions=["alpha", "actor"]))
    assert set(decision["decision"]["rejections"][0]) == {
        "stage", "reason", "instance_id", "step", "source_example_ids", "support_groups",
    }
    assert all(type(value) is bool for value in (
        first["decision"]["commit"], observed["compiled"], observed["native_valid"],
        observed["actor_required"], observed["net_checks"],
    ))


def test_import_and_execution_require_only_the_public_v5_fragment_hash_helper(monkeypatch):
    library, policy, _ = prepared()
    assert library["schema_version"] == "chisel-observable-fragment-contracts/v5"
    v5_surface = ModuleType("chisel.learning.fragment_contracts")
    v5_surface.library_behavior_hash = residual.library_behavior_hash
    monkeypatch.setitem(sys.modules, "chisel.learning.fragment_contracts", v5_surface)
    loaded = runpy.run_path(residual.__file__)
    restored = loaded["ResidualProgramController"](
        json.loads(json.dumps(library)), json.loads(json.dumps(policy)),
        group_id="target", task_family="opaque-family",
    )
    baseline = controller(library, policy)
    for before, action, after in (("s0", "alpha", "s1"), ("s1", "beta", "s2")):
        result = restored.decide(state(before))
        assert result == baseline.decide(state(before))
        assert restored.observe(action, after, compiled=result["decision"]["commit"]) == baseline.observe(
            action, after, compiled=result["decision"]["commit"],
        )


def test_teacher_replay_stops_at_mismatch_without_consuming_hypothetical_feedback():
    library, policy, _ = prepared(corpus(actions=("alpha", "beta", "gamma")))
    control = controller(library, policy)
    initial = control.decide(state())
    assert not initial["decision"]["commit"]
    control.observe("alpha", "s1", compiled=False, native_valid="alpha" in ACTIONS)
    proposed = control.decide(state("s1"))
    assert proposed["decision"]["commit"]
    assert proposed["proposal"]["action"] == "beta"
    teacher_action = "other"
    assert proposed["proposal"]["action"] != teacher_action
    assert control._turn == 1
    assert control._native_actions == {"alpha"}
    assert control._active[0]["step"] == 1
    with pytest.raises(ValueError, match="pending committed"):
        control.observe(teacher_action, "unexecuted counterfactual feedback", compiled=True)
    assert control._turn == 1
    assert control._native_actions == {"alpha"}


@pytest.mark.parametrize("mode", residual.MODES)
@pytest.mark.parametrize("action", ["", "   ", "not a native action"])
def test_malformed_actor_feedback_is_recorded_without_an_anchor_or_native_history(mode, action):
    library, policy, _ = prepared()
    control = controller(library, policy, mode)
    control.decide(state())
    feedback = control.observe(action, "actor action rejected", compiled=False, native_valid=False)
    assert feedback["action"] == action
    assert feedback["observation"] == "actor action rejected"
    assert feedback["reason"] == "actor_action_not_native"
    assert feedback["turn"] == 0
    assert feedback["completed_action_count"] == 1
    assert feedback["native_action_count"] == feedback["seen_entry_count"] == 0
    assert feedback["hypotheses"] == feedback["checks"] == []
    assert control._active == []
    assert control._native_actions == control._native_words == set()
    assert control._pending is None
    assert feedback == json.loads(json.dumps(feedback, allow_nan=False))
    next_decision = control.decide(state(history=[
        {"action": action, "observation": "actor action rejected"},
    ]))
    assert next_decision["decision"]["turn"] == 1
    assert next_decision["decision"]["native_action_count"] == 0
    assert all(h["step"] == 0 and h["anchor_turn"] is None
               for h in next_decision["decision"]["hypotheses"])


def test_empty_actor_override_discards_residual_but_preserves_earlier_actual_history():
    library, policy, _ = prepared()
    control = controller(library, policy)
    anchor(control)
    assert control.decide(state("s1"))["proposal"]["action"] == "beta"
    feedback = control.observe("", "s0", compiled=False, native_valid=False)
    assert feedback["hypotheses"] == []
    assert feedback["completed_action_count"] == 2
    assert feedback["native_action_count"] == 1
    assert control._native_actions == {"alpha"}
    assert control._native_words == {"alpha"}
    assert not control.decide(state())["decision"]["commit"]


@pytest.mark.parametrize("action,native_valid", [
    ("", False), ("", True), ("   ", False), ("   ", True), ("not native", True),
])
def test_empty_or_nonnative_compiled_feedback_remains_an_error(action, native_valid):
    library, policy, _ = prepared()
    control = controller(library, policy)
    anchor(control)
    control.decide(state("s1"))
    pending = deepcopy(control._pending)
    with pytest.raises(ValueError):
        control.observe(action, "rejected feedback", compiled=True, native_valid=native_valid)
    assert control._pending == pending
    assert control._turn == control._native_count == 1
    assert control.observe("beta", "s2", compiled=True)["completed_instance_ids"]


EFFECT_ACTION = "open crate 1"
EFFECT_PROGRAM = (EFFECT_ACTION, "visit crate 2")
EFFECT_NATIVE = [EFFECT_ACTION, "visit crate 2", "finish crate 2", "probe widget 1", "actor", "prior"]


def effect_corpus(name="effect", *, actions=EFFECT_PROGRAM, groups=("g0", "g1"),
                  prefixes=1, history=(), task="inspect widget", observations=None, natives=None):
    candidates, demos, transitions = corpus(
        name, actions=actions, groups=groups, prefixes=prefixes, history=history, task=task,
        observations=observations or ["closed", "revealed widget", "finished widget", "end"][
            :len(actions) + 1
        ],
    )
    for source in candidates[0]["examples"]:
        prior = [{"action": action, "observation": "prior feedback"} for action in history]
        for step, identifier in enumerate(source["demonstration_ids"]):
            demo = next(row for row in demos if row["example_id"] == identifier)
            if natives is not None:
                native = natives(source["group_id"], step)
            else:
                native = list(dict.fromkeys([*actions[step:], *EFFECT_NATIVE]))
                if step and actions[0] in native:
                    native.remove(actions[0])
            demo["public_state"]["actions"] = list(native)
            demo["public_state"]["history"] = deepcopy(prior)
            prior.append({"action": demo["action"],
                          "observation": transitions[identifier]["post_observation"]})
    return candidates, demos, transitions


def effect_states(demos, name="effect", group="g0"):
    before = next(row for row in demos if row["example_id"] == f"{name}-{group}-0-0")
    after = next(row for row in demos if row["example_id"] == f"{name}-{group}-0-1")
    return deepcopy(before["public_state"]), deepcopy(after["public_state"])


def observe_effect_actor(control, before, after, action=EFFECT_ACTION):
    result = control.decide(before)
    assert not result["decision"]["commit"]
    assert result["decision"]["anchor_candidates"] == []
    feedback = control.observe(action, after["observation"], compiled=False)
    assert feedback["reason"] == "actor_effect_pending"
    assert feedback["hypotheses"] == []
    assert not feedback["actor_required"]
    return result, feedback


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
def test_effect_memory_is_action_conditioned_before_ranking_and_built_only_once(monkeypatch, mode):
    library, policy, demos = prepared(
        effect_corpus("desired"),
        effect_corpus("distractors", actions=("rival crate 1", "visit crate 2"), prefixes=70),
    )
    before, after = effect_states(demos, "desired")
    control = controller(library, policy, mode)
    original_build, original_retrieve = invocation.build_policy, invocation.retrieve_demonstrations
    built, queried = [], []

    def build(records):
        built.append(deepcopy(records))
        return original_build(records)

    def retrieve(memory, current, *, excluded_groups, limit):
        queried.append((deepcopy(memory), deepcopy(current), excluded_groups, limit))
        return original_retrieve(memory, current, excluded_groups=excluded_groups, limit=limit)

    monkeypatch.setattr(invocation, "build_policy", build)
    monkeypatch.setattr(invocation, "retrieve_demonstrations", retrieve)
    observe_effect_actor(control, before, after)
    assert built == queried == []
    result = control.decide(after)
    assert result["decision"]["commit"]
    assert result["proposal"]["action"] == "visit crate 2"
    assert len(built) == len(queried) == 1
    assert len(built[0]) == 2
    assert {row["example_id"] for row in built[0]} == {"desired-g0-0-1", "desired-g1-0-1"}
    assert all(row["action"] != EFFECT_ACTION for row in built[0])
    assert all(EFFECT_ACTION not in row["public_state"]["actions"] for row in built[0])
    assert queried[0][3] == 128
    assert queried[0][2] == frozenset({"target"})
    assert queried[0][1] == after
    assert result["decision"]["effect_memory"]["record_count"] == 2
    assert result["decision"]["hypotheses"][0]["step"] == 1
    assert result["decision"]["hypotheses"][0]["anchor_turn"] == 0
    control.observe("visit crate 2", "finished widget", compiled=True)
    assert not control.decide(before)["decision"]["commit"]
    control.observe(EFFECT_ACTION, after["observation"], compiled=False)
    assert control.decide(after)["decision"]["commit"]
    assert len(built) == 1
    assert len(queried) == 2
    cached = control._effect_cache[EFFECT_ACTION]
    assert cached == json.loads(json.dumps(cached, allow_nan=False))
    control.observe("visit crate 2", "finished widget", compiled=True)
    control.reset_episode()
    observe_effect_actor(control, before, after)
    assert control.decide(after)["decision"]["commit"]
    assert len(built) == 1


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
def test_actual_effect_state_recovers_sources_lost_by_pre_action_similarity(mode):
    original_observation = " ".join(f"sourceword{index}" for index in range(90))
    library, policy, demos = prepared(effect_corpus(
        observations=[original_observation, "revealed widget", "finished widget"],
    ))
    before, after = effect_states(demos)
    before["observation"] = " ".join(f"queryword{index}" for index in range(90))
    baseline = controller(library, policy)
    first = baseline.decide(before)
    assert not first["decision"]["anchor_candidates"]
    baseline.observe(EFFECT_ACTION, after["observation"], compiled=False)
    assert not baseline.decide(after)["decision"]["commit"]
    control = controller(library, policy, mode)
    observe_effect_actor(control, before, after)
    result = control.decide(after)
    assert result["decision"]["entry_threshold"] == 0.55
    assert result["decision"]["minimum_groups"] == 2
    assert result["decision"]["commit"]
    assert result["proposal"]["action"] == "visit crate 2"
    assert result["decision"]["support_groups"] == ["g0", "g1"]
    assert all(check["passed"] for check in result["decision"]["actor_effect_checks"])


def test_new_goal_affordance_invalidates_stale_search_suffix_only_in_full_effect_mode():
    library, policy, demos = prepared(effect_corpus())
    before, after = effect_states(demos)
    after["actions"].append("collect widget 7")
    results = {}
    for mode in residual.EFFECT_MODES:
        control = controller(library, policy, mode)
        observe_effect_actor(control, before, after)
        results[mode] = control.decide(after)
    full = results["anchored_effect_consensus"]
    assert not full["decision"]["commit"]
    assert full["decision"]["support_group_count"] == 0
    assert {row["reason"] for row in full["decision"]["rejections"]} == {"goal_affordance_mismatch"}
    assert len(full["decision"]["rejections"]) == 2
    check = full["decision"]["rejections"][0]
    assert check["source_goal_roles"] == check["current_goal_roles"] == ["widget"]
    assert check["source_affordance_count"] == 1
    assert check["current_affordance_count"] == 2
    ablation = results["anchored_effect_no_affordance_check"]
    assert ablation["decision"]["commit"]
    assert ablation["proposal"]["action"] == "visit crate 2"
    assert ablation["decision"]["support_groups"] == ["g0", "g1"]


@pytest.mark.parametrize("change,allowed", [
    ("numeric_instance_only", True), ("non_goal_action", True),
    ("goal_action_verb", False), ("goal_entity_type", False), ("non_entity_integer", False),
])
def test_affordance_templates_normalize_only_entity_ids_and_ignore_non_goal_actions(change, allowed):
    library, policy, demos = prepared(effect_corpus())
    before, after = effect_states(demos)
    if change == "numeric_instance_only":
        after["actions"] = ["probe widget 900" if a == "probe widget 1" else a for a in after["actions"]]
    elif change == "non_goal_action":
        after["actions"].append("different operation crate 900")
    elif change == "goal_action_verb":
        after["actions"] = ["different widget 1" if a == "probe widget 1" else a for a in after["actions"]]
    elif change == "goal_entity_type":
        after["actions"] = ["probe widgetish 1" if a == "probe widget 1" else a for a in after["actions"]]
    else:
        after["actions"].append("probe widget 1 + 7")
    control = controller(library, policy, "anchored_effect_consensus")
    observe_effect_actor(control, before, after)
    result = control.decide(after)
    assert result["decision"]["commit"] is allowed
    if not allowed:
        assert any(row["reason"] == "goal_affordance_mismatch" for row in result["decision"]["rejections"])


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
def test_effect_source_prerequisites_cannot_be_satisfied_by_the_anchor_action_itself(mode):
    library, policy, demos = prepared(effect_corpus(history=(EFFECT_ACTION,)))
    before, after = effect_states(demos)
    control = controller(library, policy, mode)
    observe_effect_actor(control, before, after)
    assert EFFECT_ACTION in control._native_actions
    result = control.decide(after)
    assert not result["decision"]["commit"]
    assert {row["reason"] for row in result["decision"]["rejections"]} == {"history_prerequisites_missing"}
    assert result["decision"]["native_action_count"] == 1
    assert result["decision"]["effect_memory"]["record_count"] == 2


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
def test_effect_entry_goal_bindings_use_pre_actor_task_not_repaired_post_task(mode):
    library, policy, demos = prepared(effect_corpus(task="inspect widget by crate"))
    before, after = effect_states(demos)
    before["task"] = "inspect widget"
    control = controller(library, policy, mode)
    observe_effect_actor(control, before, after)
    result = control.decide(after)
    assert not result["decision"]["commit"]
    assert {row["reason"] for row in result["decision"]["rejections"]} == {"source_goal_words_missing"}


def test_pre_actor_validated_history_allows_a_genuinely_satisfied_prerequisite():
    library, policy, demos = prepared(effect_corpus(history=("prior",)))
    before, after = effect_states(demos)
    control = controller(library, policy, "anchored_effect_consensus")
    assert not control.decide(before)["decision"]["commit"]
    control.observe("prior", before["observation"], compiled=False)
    assert not control.decide(before)["decision"]["commit"]
    control.observe(EFFECT_ACTION, after["observation"], compiled=False)
    assert control.decide(after)["decision"]["commit"]
    assert control._native_history == ["prior", EFFECT_ACTION]


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
@pytest.mark.parametrize("which,reason", [
    ("actor_pre", "actor_precondition_failed"),
    ("actor_post", "actor_postcondition_failed"),
    ("current_pre", "step_precondition_failed"),
])
def test_effect_modes_keep_first_actor_pre_post_and_current_step_preconditions(mode, which, reason):
    library, policy, demos = prepared(effect_corpus())
    before, after = effect_states(demos)
    actual_post = after["observation"]
    if which == "actor_pre":
        before["observation"] += " revealed"
    elif which == "actor_post":
        actual_post = "unchanged"
    else:
        after["observation"] += " finished"
    control = controller(library, policy, mode)
    control.decide(before)
    control.observe(EFFECT_ACTION, actual_post, compiled=False)
    result = control.decide(after)
    assert not result["decision"]["commit"]
    assert any(row["reason"] == reason for row in result["decision"]["rejections"])


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
@pytest.mark.parametrize("source_task,current_task,reason", [
    ("inspect mystery", "inspect mystery", "source_goal_types_unbound"),
    ("inspect widget", "inspect mystery", "current_goal_types_unbound"),
    ("inspect widget widget", "inspect widget widget", "ambiguous_goal_type_roles"),
    ("inspect widget by crate", "inspect crate by widget", "incompatible_goal_type_roles"),
])
def test_goal_type_roles_must_be_bound_and_compatible_in_both_effect_modes(
    mode, source_task, current_task, reason,
):
    library, policy, demos = prepared(effect_corpus(task=source_task))
    before, after = effect_states(demos)
    before["task"] = after["task"] = current_task
    control = controller(library, policy, mode)
    observe_effect_actor(control, before, after)
    result = control.decide(after)
    assert not result["decision"]["commit"]
    assert any(row["reason"] == reason for row in result["decision"]["rejections"])


@pytest.mark.parametrize("minimum_groups,committed", [(2, True), (3, False)])
def test_later_step_affordance_pruning_is_per_source_and_recomputes_group_support(minimum_groups, committed):
    actions = (EFFECT_ACTION, "visit crate 2", "finish crate 2")

    def natives(group, step):
        result = list(EFFECT_NATIVE)
        if step:
            result.remove(EFFECT_ACTION)
        if group == "g2" and step == 2:
            result.append("collect widget 1")
        return result

    library, policy, demos = prepared(effect_corpus(
        actions=actions, groups=("g0", "g1", "g2"), natives=natives,
        observations=["closed", "revealed widget", "middle widget", "end"],
    ))
    before, after = effect_states(demos)
    control = controller(library, policy, "anchored_effect_consensus", minimum_groups=minimum_groups)
    observe_effect_actor(control, before, after)
    first = control.decide(after)
    assert first["decision"]["commit"]
    assert first["decision"]["support_groups"] == ["g0", "g1", "g2"]
    control.observe("visit crate 2", "middle widget", compiled=True)
    current = deepcopy(next(row["public_state"] for row in demos if row["example_id"] == "effect-g0-0-2"))
    result = control.decide(current)
    assert result["decision"]["commit"] is committed
    assert result["decision"]["support_groups"] == ["g0", "g1"]
    assert result["decision"]["source_count"] == 2
    rejected = next(row for row in result["decision"]["rejections"]
                    if row["reason"] == "goal_affordance_mismatch")
    assert rejected["support_groups"] == ["g2"]
    assert rejected["source_step_demonstration_id"] == "effect-g2-0-2"


def test_excluded_groups_are_absent_before_effect_policy_and_entity_vocabulary_fitting(monkeypatch):
    library, policy, demos = prepared(
        effect_corpus(groups=("g0", "g1", "target", "held")),
        effect_corpus("private", groups=("target", "held"),
                      natives=lambda group, step: [*EFFECT_NATIVE, "reveal privateentity 1"]),
        excluded_groups={"target", "held"},
    )
    before, after = effect_states(demos)
    control = controller(library, policy, "anchored_effect_consensus")
    assert "privateentity" not in control._goal_types
    build_policy = invocation.build_policy
    built = []

    def build(records):
        built.append(deepcopy(records))
        return build_policy(records)

    monkeypatch.setattr(invocation, "build_policy", build)
    observe_effect_actor(control, before, after)
    result = control.decide(after)
    assert result["decision"]["commit"]
    assert {row["group_id"] for row in built[0]} == {"g0", "g1"}
    assert control._effect_cache[EFFECT_ACTION]["policy"] == build_policy(built[0])
    assert result["decision"]["effect_memory"]["source_groups"] == ["g0", "g1"]
    assert result["decision"]["support_groups"] == ["g0", "g1"]


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
def test_effect_modes_keep_completed_short_hypothesis_veto_and_native_membership(mode):
    library, policy, demos = prepared(
        effect_corpus("short"),
        effect_corpus("long", actions=(EFFECT_ACTION, "visit crate 2", "finish crate 2"),
                      observations=["closed", "revealed widget", "finished widget", "end"]),
    )
    before, after = effect_states(demos, "short")
    control = controller(library, policy, mode)
    observe_effect_actor(control, before, after)
    first = control.decide(after)
    assert first["decision"]["commit"]
    feedback = control.observe("visit crate 2", "finished widget", compiled=True)
    assert feedback["hypothesis_count"] == 2
    assert feedback["completed_instance_ids"] == [instance(library, "short")]
    assert feedback["actor_required"]
    assert control.decide(after)["decision"]["reason"] == "actor_boundary_required"
    control = controller(library, policy, mode)
    observe_effect_actor(control, before, after)
    after["actions"].remove("visit crate 2")
    rejected = control.decide(after)
    assert not rejected["decision"]["commit"]
    assert any(row["reason"] == "next_action_not_native" for row in rejected["decision"]["rejections"])


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
@pytest.mark.parametrize("score,accepted", [(0.55, True), (0.549, False)])
def test_effect_retrieval_keeps_the_original_threshold_and_group_minimum(monkeypatch, mode, score, accepted):
    library, policy, demos = prepared(effect_corpus())
    before, after = effect_states(demos)
    control = controller(library, policy, mode)

    def retrieve(memory, current, *, excluded_groups, limit):
        assert limit == 128
        return [(1.0 if row["group_id"] == "g0" else score, row) for row in memory["records"]]

    monkeypatch.setattr(invocation, "retrieve_demonstrations", retrieve)
    observe_effect_actor(control, before, after)
    result = control.decide(after)
    assert result["decision"]["commit"] is accepted
    if not accepted:
        assert result["decision"]["support_groups"] == ["g0"]
        assert any(row["reason"] == "below_entry_threshold" for row in result["decision"]["rejections"])


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
def test_empty_actor_feedback_clears_pending_effect_without_history_or_memory_fit(mode):
    library, policy, demos = prepared(effect_corpus())
    before, after = effect_states(demos)
    control = controller(library, policy, mode)
    control.decide(before)
    feedback = control.observe("", "rejected", compiled=False, native_valid=False)
    assert feedback["reason"] == "actor_action_not_native"
    assert feedback["completed_action_count"] == 1
    assert feedback["native_action_count"] == 0
    assert feedback["effect_memory"] is None
    assert control._effect_cache == {}
    assert control._effect_anchor is None
    assert control._native_history == []
    assert control.decide(after)["decision"]["reason"] == "actor_anchor_required"


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
def test_effect_mode_serialization_and_native_action_order_are_deterministic(mode):
    library, policy, demos = prepared(effect_corpus())
    before, after = effect_states(demos)
    first = controller(library, policy, mode)
    second = controller(json.loads(json.dumps(library)), json.loads(json.dumps(policy)), mode)
    reversed_before, reversed_after = deepcopy(before), deepcopy(after)
    reversed_before["actions"].reverse()
    reversed_after["actions"].reverse()
    assert first.decide(before) == second.decide(reversed_before)
    assert first.observe(EFFECT_ACTION, after["observation"], compiled=False) == second.observe(
        EFFECT_ACTION, after["observation"], compiled=False,
    )
    result = first.decide(after)
    assert result == second.decide(reversed_after)
    assert result == json.loads(json.dumps(result, allow_nan=False))


@pytest.mark.parametrize("mode,expected", [
    ("anchored_guarded_consensus", "460e0885de9bae6e925b7a2b92a5904b611919c8540460a50e199e982a7dee75"),
    ("anchored_consensus", "765f6db23f446a4d4b84be31db657355d896436749103879733648c2d8e24195"),
    ("anchored_single", "7491b9a9e9c2b76184f15f262ff9993f8846b541c388b753489e35729813ff55"),
    ("unanchored_guarded_consensus", "cbe518174c8ea8a7e75a209da00cdcc8b72f7c1b76d7c41009bc30c46f496383"),
])
def test_all_original_modes_keep_pre_revision_complete_synthetic_traces(mode, expected):
    library, policy, _ = prepared()
    control = controller(library, policy, mode)
    trace = []
    for before, action, after in (
        ("s0", "alpha", "s1"), ("s1", "beta", "s2"), ("s0", "other", "s0"),
        ("s0", "", "s0"), ("s0", "alpha", "s1"), ("s1", "beta", "s2"),
    ):
        decision = control.decide(state(before))
        trace.append(decision)
        compiled = decision["decision"]["commit"] and decision["proposal"]["action"] == action
        trace.append(control.observe(action, after, compiled=compiled, native_valid=action in ACTIONS))
    assert hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == expected


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
def test_effect_modes_still_import_with_only_the_original_v5_public_hash_helper(monkeypatch, mode):
    library, policy, demos = prepared(effect_corpus())
    before, after = effect_states(demos)
    v5_surface = ModuleType("chisel.learning.fragment_contracts")
    v5_surface.library_behavior_hash = residual.library_behavior_hash
    monkeypatch.setitem(sys.modules, "chisel.learning.fragment_contracts", v5_surface)
    loaded = runpy.run_path(residual.__file__)
    control = loaded["ResidualProgramController"](
        library, policy, group_id="target", task_family="opaque-family", mode=mode,
    )
    observe_effect_actor(control, before, after)
    assert control.decide(after)["decision"]["commit"]


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
def test_effect_disagreement_returns_to_actor_and_reanchors_from_fresh_saved_context(mode):
    library, policy, demos = prepared(
        effect_corpus("one"),
        effect_corpus("two", actions=(EFFECT_ACTION, "finish crate 2")),
        effect_corpus(
            "fresh", actions=("prior", "visit crate 2"),
            observations=["revealed widget", "different widget", "finished widget"],
        ),
    )
    before, after = effect_states(demos, "one")
    control = controller(library, policy, mode)
    observe_effect_actor(control, before, after)
    conflict = control.decide(after)
    assert not conflict["decision"]["commit"]
    assert conflict["decision"]["reason"] == "ambiguous_continuations"
    assert {row["action"] for row in conflict["decision"]["competing_actions"]} == {
        "visit crate 2", "finish crate 2",
    }
    assert conflict["decision"]["anchor_candidates"] == []
    feedback = control.observe("prior", "different widget", compiled=False)
    assert feedback["reason"] == "actor_effect_pending"
    assert feedback["effect_memory"]["actor_action"] == "prior"
    assert control._effect_anchor["native_history"] == [EFFECT_ACTION]
    _, current = effect_states(demos, "fresh")
    result = control.decide(current)
    assert result["decision"]["commit"]
    assert result["decision"]["hypotheses"][0]["program_id"] == "fresh"
    assert result["decision"]["hypotheses"][0]["anchor_turn"] == 1
    assert set(control._effect_cache) == {EFFECT_ACTION, "prior"}


def test_unknown_actor_actions_do_not_build_or_grow_effect_memory_cache(monkeypatch):
    library, policy, demos = prepared(effect_corpus())
    before, _ = effect_states(demos)
    control = controller(library, policy, "anchored_effect_consensus")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unknown first actions must not fit a memory or invoke global retrieval")

    monkeypatch.setattr(invocation, "build_policy", forbidden)
    monkeypatch.setattr(invocation, "retrieve_demonstrations", forbidden)
    for action in ("actor", "prior", "actor"):
        assert not control.decide(before)["decision"]["commit"]
        control.observe(action, before["observation"], compiled=False)
    result = control.decide(before)
    assert not result["decision"]["commit"]
    assert result["decision"]["effect_memory"]["status"] == "no_source_action"
    assert result["decision"]["effect_memory"]["record_count"] == 0
    assert control._effect_cache == {}


def test_effect_memory_does_not_invent_missing_or_rebound_source_step_states():
    library, policy, demos = prepared(effect_corpus())
    remaining = [row for row in demos if row["example_id"] != "effect-g0-0-1"]
    with pytest.raises(ValueError, match="matching source-step"):
        controller(library, invocation.build_policy(remaining), "anchored_effect_consensus")
    changed = deepcopy(demos)
    next(row for row in changed if row["example_id"] == "effect-g0-0-1")["public_state"]["task"] = (
        "different source task"
    )
    with pytest.raises(ValueError, match="matching source-step"):
        controller(library, invocation.build_policy(changed), "anchored_effect_consensus")
    assert controller(library, policy, "anchored_effect_consensus")


@pytest.mark.parametrize("mode", residual.EFFECT_MODES)
def test_effect_retrieval_metadata_keys_and_source_effect_links_are_explicit(mode):
    library, policy, demos = prepared(effect_corpus())
    before, after = effect_states(demos)
    control = controller(library, policy, mode)
    initial, actor_feedback = observe_effect_actor(control, before, after)
    assert initial["decision"]["schema_version"] == residual.EFFECT_CONTROLLER_SCHEMA
    assert initial["decision"]["effect_memory"] is None
    assert actor_feedback["effect_memory"]["status"] == "retrieval_pending"
    result = control.decide(after)
    decision = result["decision"]
    assert decision["affordance_schema"] == residual.AFFORDANCE_SCHEMA
    assert decision["goal_role_checks"] is True
    assert decision["affordance_equality_check"] is (mode == "anchored_effect_consensus")
    assert set(decision["effect_memory"]) == {
        "schema_version", "status", "actor_action", "actor_turn", "pre_actor_state_sha256",
        "pre_actor_native_history_sha256", "memory_hash", "record_count",
        "source_reference_count", "source_groups",
    }
    assert decision["effect_memory"]["schema_version"] == residual.EFFECT_MEMORY_SCHEMA
    assert decision["effect_memory"]["status"] == "ready"
    assert set(decision["actor_effect_checks"][0]) == {
        "instance_id", "source_example_id", "group_id", "precondition_clauses",
        "postcondition_clauses", "passed",
    }
    assert set(decision["hypotheses"][0]["sources"][0]) == {
        "example_id", "group_id", "similarity", "effect_example_id",
    }
    assert decision["hypotheses"][0]["sources"][0]["effect_example_id"].endswith("-1")
