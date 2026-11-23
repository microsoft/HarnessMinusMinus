from copy import deepcopy
import hashlib
import json

import pytest

from chisel.learning import counterfactual_invocation as invocation
from chisel.learning import fragment_contracts as contracts


def public(observation="x", *, actions=None, task="alpha beta gamma delta", history=None):
    return {
        "task": task, "observation": observation,
        "actions": list(actions if actions is not None else ["act alpha", "act beta"]),
        "history": [] if history is None else history,
    }


def bundle(*, name="p", length=2, groups=("g0", "g1"), prefixes=1, observations=None,
           task="alpha beta gamma delta", pre_action_history=None, task_family=None):
    actions = [f"act {word}" for word in ("alpha", "beta", "gamma", "delta")][:length]
    examples, demonstrations, transitions = [], [], {}
    for group in groups:
        for prefix in range(prefixes):
            task_id = f"{name}-task-{group}-{prefix}"
            source_task = task(group, prefix) if callable(task) else task
            family = task_family(group, prefix) if callable(task_family) else task_family
            history = (pre_action_history(group, prefix) if callable(pre_action_history)
                       else [] if pre_action_history is None else pre_action_history)
            path = (observations(group, prefix) if callable(observations) else
                    observations if observations is not None else
                    ["x", "alpha", "beta", "gamma", "delta"][:length + 1])
            assert len(path) == length + 1
            ids = [f"{task_id}-demo-{step}" for step in range(length)]
            examples.append({
                "task_id": task_id, "group_id": group, "start_turn": len(history),
                "demonstration_ids": ids,
                **({"task_family": family} if family is not None else {}),
            })
            for step, action in enumerate(actions):
                demonstrations.append({
                    "example_id": ids[step], "task_id": task_id, "group_id": group, "action": action,
                    "public_state": public(path[step], actions=actions, task=source_task),
                })
                transitions[ids[step]] = {
                    "post_observation": path[step + 1],
                    "pre_action_history": list(history) + actions[:step],
                    **({"task_family": family} if family is not None else {}),
                }
    return [{
        "program_id": name, "actions": actions, "examples": examples,
        "learned_applicability_guard": False, "deployment_approved": False,
    }], demonstrations, transitions


def compile_bundle(*, negatives=None, minimum_groups=2, max_literals=1,
                   excluded_groups=frozenset(), negative_programs=None, **kwargs):
    programs, demonstrations, transitions = bundle(**kwargs)
    library = contracts.compile_library(
        programs, demonstrations, transitions, [] if negatives is None else negatives,
        minimum_groups=minimum_groups, max_literals=max_literals, excluded_groups=excluded_groups,
        negative_programs=negative_programs,
    )
    return library, invocation.build_policy(demonstrations), demonstrations


def negative(observation, *, group="negative", action="act alpha", post=None, task_family=None):
    return {
        "group_id": group, "action": action, "observation": observation,
        "post_observation": observation if post is None else post,
        **({"task_family": task_family} if task_family is not None else {}),
    }


def cycle_negative(observation="x", *, group="cycle", task_family=None, actions=None,
                   pre_state=None, post_state=None, pre_action_history=None,
                   post_action_history=None):
    actions = ["act alpha", "act beta"] if actions is None else actions
    before = public(observation, actions=actions) if pre_state is None else pre_state
    after = deepcopy(before) if post_state is None else post_state
    history = list(actions if pre_action_history is None else pre_action_history)
    return {
        "group_id": group, "task_family": task_family, "actions": list(actions),
        "observation": before["observation"], "post_observation": after["observation"],
        "pre_state_sha256": invocation.public_state_sha256(before),
        "post_state_sha256": invocation.public_state_sha256(after),
        "pre_action_history": history,
        "post_action_history": history + list(actions) if post_action_history is None
        else list(post_action_history),
    }


def controller(library, policy, mode="positive_contract", **kwargs):
    return contracts.FragmentController(
        library, policy, group_id="target", mode=mode, **kwargs,
    )


def fixed_neighbors(monkeypatch, rows):
    calls = []

    def retrieve(policy, state, *, excluded_groups=frozenset(), limit=128):
        invocation.validate_public_state(state)
        calls.append({"excluded_groups": excluded_groups, "limit": limit})
        return [(score, row) for score, row in rows if row["group_id"] not in excluded_groups][:limit]

    monkeypatch.setattr(invocation, "retrieve_demonstrations", retrieve, raising=False)
    return calls


def literals(guard):
    return [
        tuple((literal["sign"], (literal["action"],) if "action" in literal
               else tuple(literal["tokens"])) for literal in clause["literals"])
        for clause in guard["clauses"]
    ]


def prime_native_history(control, actions):
    for action in actions:
        control.observe(action, "earlier native feedback", compiled=False, native_valid=True)
    return control


def test_signed_witness_preconditions_and_actual_postconditions():
    library, policy, _ = compile_bundle(observations=["", "alpha", ""])
    steps = library["programs"][0]["steps"]
    assert literals(steps[0]["positive_guard"]) == [(("+", ("alpha",)),)]
    assert literals(steps[1]["positive_guard"]) == [(("-", ("alpha",)),)]
    for mode in ("positive_contract", "refined_contract", "refined_no_memory"):
        control = controller(library, policy, mode)
        first = control.decide(public(""))
        assert first["decision"]["commit"]
        assert first["proposal"]["step"] == 0
        feedback = control.observe(first["proposal"]["action"], "ALPHA!", compiled=True)
        assert feedback["postcondition_passed"] is True
        assert feedback["next_step"] == 1
        second = control.decide(public("alpha"))
        assert second["proposal"]["step"] == 1
        assert second["decision"]["precondition_checked"]
        result = control.observe(second["proposal"]["action"], "", compiled=True)
        assert result["postcondition_passed"] is True
        assert result["fragment_complete"] is True


def test_bigrams_are_ordered_observable_changes_not_bags_of_words():
    library, policy, demos = compile_bundle(observations=["a b", "b a", "a b"])
    assert literals(library["programs"][0]["steps"][0]["positive_guard"]) == [
        (("+", ("b", "a")),),
    ]
    control = controller(library, policy)
    result = control.decide(demos[0]["public_state"])
    assert result["decision"]["commit"]
    feedback = control.observe(result["proposal"]["action"], "a b", compiled=True)
    assert feedback["postcondition_passed"] is False


def test_exclusion_precedes_all_positive_and_negative_fitting():
    def paths(group, _):
        return ["private", "secret", "hidden"] if group == "target" else ["x", "alpha", "beta"]

    negatives = [
        negative("x", group="target"),
        negative("", group="kept"),
        negative("x", group="changing", post="alpha"),
        negative("x", group="case_change", post="X"),
        negative("x", group="whitespace_change", post="x "),
        negative("", group="kept"),
    ]
    library, policy, _ = compile_bundle(
        groups=("g0", "g1", "target"), observations=paths, negatives=negatives,
        excluded_groups={"target"},
    )
    program = library["programs"][0]
    assert program["source_groups"] == ["g0", "g1"]
    assert {row["group_id"] for row in program["examples"]} == {"g0", "g1"}
    assert library["fit_groups"] == ["g0", "g1", "kept"]
    assert library["counts"]["excluded_examples"] == 1
    assert library["counts"]["excluded_demonstrations"] == 2
    assert library["counts"]["excluded_negative_transitions"] == 1
    assert library["counts"]["ignored_non_stutter_negatives"] == 3
    assert library["counts"]["duplicate_negative_transitions"] == 1
    assert library["counts"]["usable_negative_transitions"] == 1
    for step in program["steps"]:
        for variant in ("positive_guard", "refined_guard"):
            guard = step[variant]
            assert "target" not in guard["negative_groups"]
            assert all("target" not in row["positive_groups"] for row in guard["clauses"])
            assert "secret" not in json.dumps(literals(guard))
    assert controller(library, policy).decide(public())["decision"]["commit"]


def test_same_eligible_program_pool_and_negative_group_penalty():
    negatives = [negative("", group="one-negative") for _ in range(7)]
    negatives += [negative(f"noise {index}", group="one-negative") for index in range(5)]
    negatives += [negative("", group="other-action", action="unrelated")]
    plain, _, _ = compile_bundle()
    refined, _, _ = compile_bundle(negatives=negatives)
    assert [row["program_id"] for row in plain["programs"]] == [
        row["program_id"] for row in refined["programs"]
    ]
    step = refined["programs"][0]["steps"][0]
    assert literals(step["positive_guard"]) == literals(
        plain["programs"][0]["steps"][0]["positive_guard"],
    )
    assert literals(step["positive_guard"]) == [(("+", ("alpha",)),)]
    assert literals(step["refined_guard"]) == [(("-", ("x",)),)]
    clause = step["positive_guard"]["clauses"][0]
    assert clause["negative_precondition_count"] == 6
    assert clause["negative_precondition_group_count"] == 1
    assert clause["negative_precondition_groups"] == ["one-negative"]
    assert step["refined_guard"]["accepted_negative_groups"] == []
    assert "other-action" not in refined["fit_groups"]
    assert step["refined_guard"]["objective"] == refined["objectives"]["refined"]


def test_refinement_retains_a_supported_clause_even_when_penalty_is_large():
    library, policy, _ = compile_bundle(
        negatives=[negative("x", group=f"n{index}") for index in range(5)],
    )
    assert len(library["programs"]) == 1
    clause = library["programs"][0]["steps"][0]["refined_guard"]["clauses"][0]
    assert clause["positive_group_count"] == 2
    assert clause["selection"]["objective_gain"] == -3
    assert controller(library, policy, "refined_contract").decide(public())["decision"]["commit"]


def test_many_prefixes_do_not_become_independent_source_groups():
    library, _, _ = compile_bundle(groups=("only",), prefixes=8)
    assert library["programs"] == []
    assert library["rejected_programs"][0]["reason"] == "insufficient_source_groups"
    library, _, _ = compile_bundle(prefixes=5)
    clause = library["programs"][0]["steps"][0]["positive_guard"]["clauses"][0]
    assert clause["positive_transition_count"] == 10
    assert clause["positive_group_count"] == 2
    assert clause["positive_groups"] == ["g0", "g1"]


def test_each_step_needs_a_group_supported_change():
    library, _, _ = compile_bundle(observations=["x", "alpha", "alpha"])
    assert library["programs"] == []
    assert library["rejected_programs"][0]["reason"] == "step_without_supported_change_witness"
    assert library["rejected_programs"][0]["first_unsupported_step"] == 1
    library, _, _ = compile_bundle(
        observations=lambda group, _: ["", group, ""],
    )
    assert library["programs"] == []
    assert library["rejected_programs"][0]["first_unsupported_step"] == 0


def test_missing_observations_are_counted_and_never_repaired():
    programs, demos, transitions = bundle(groups=("g0", "g1", "g2", "g3"))
    del transitions[demos[2]["example_id"]]
    missing_demo = demos.pop(4)
    transitions["unreferenced"] = {"post_observation": "unused", "pre_action_history": []}
    library = contracts.compile_library(programs, demos, transitions, [])
    assert library["counts"]["incomplete_examples"] == 2
    assert library["counts"]["missing_transition_references"] == 1
    assert library["counts"]["missing_demonstration_references"] == 1
    assert library["counts"]["unknown_transition_ids"] == 2
    assert library["programs"][0]["source_groups"] == ["g0", "g3"]
    assert missing_demo["example_id"] not in library["entry_index"]
    assert demos[2]["example_id"] not in library["entry_index"]


def test_or_clauses_cover_source_groups_and_are_bounded():
    groups = tuple(f"g{index}" for index in range(8))
    library, _, _ = compile_bundle(
        groups=groups, observations=lambda group, _: ["", f"token{int(group[1:]) // 2}", ""],
    )
    for step in library["programs"][0]["steps"]:
        for variant in ("positive_guard", "refined_guard"):
            guard = step[variant]
            assert len(guard["clauses"]) == 3
            assert guard["counts"]["covered_positive_group_count"] == 6
            assert guard["counts"]["positive_group_count"] == 8
            assert all(clause["positive_group_count"] == 2 for clause in guard["clauses"])


def test_postcondition_cannot_switch_to_a_clause_whose_precondition_failed(monkeypatch):
    library, policy, demos = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        observations=lambda group, _: ["", "alpha", "final"] if group in ("g0", "g1")
        else ["beta", "", "final"],
    )
    guard = library["programs"][0]["steps"][0]["positive_guard"]
    assert literals(guard) == [(("+", ("alpha",)),), (("-", ("beta",)),)]
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy)
    first = control.decide(public(""))
    assert first["decision"]["matched_precondition_clauses"] == [0]
    observed = control.observe(first["proposal"]["action"], "", compiled=True)
    assert observed["postcondition_passed"] is False
    assert observed["matched_postcondition_clauses"] == []


def test_arity_two_refinement_learns_joint_signed_change_and_serializes(monkeypatch):
    negatives = [negative("x alpha", group="n1"), negative("", group="n2")]
    library, policy, demos = compile_bundle(negatives=negatives, max_literals=2)
    step = library["programs"][0]["steps"][0]
    assert literals(step["positive_guard"]) == [(("+", ("alpha",)),)]
    assert literals(step["refined_guard"]) == [(("+", ("alpha",)), ("-", ("x",)))]
    clause = step["refined_guard"]["clauses"][0]
    assert clause["positive_group_count"] == 2
    assert clause["negative_precondition_group_count"] == 0
    restored = json.loads(json.dumps(library))
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(restored, policy, "refined_contract")
    assert not control.decide(public(""))["decision"]["commit"]
    control.observe("actor", "x", compiled=False)
    first = control.decide(public("x"))
    assert first["decision"]["commit"]
    result = control.observe(first["proposal"]["action"], "alpha x", compiled=True)
    assert result["postcondition_passed"] is False


def test_conjunction_support_requires_same_transition_not_just_same_group():
    library, _, _ = compile_bundle(
        prefixes=2, max_literals=2,
        observations=lambda _, prefix: ["x", "y", ""] if prefix == 0 else ["", "a", ""],
    )
    counts = library["programs"][0]["steps"][0]["positive_guard"]["counts"]
    assert counts["eligible_atom_count"] == 3
    assert counts["candidate_clause_count"] == 4


def test_conjunction_atom_pool_is_bounded_and_deterministic():
    words = " ".join(f"t{index}" for index in range(40))
    programs, demos, transitions = bundle(observations=["", words, words + " ending"])
    library = contracts.compile_library(programs, demos, transitions, [], max_literals=2)
    counts = library["programs"][0]["steps"][0]["refined_guard"]["counts"]
    assert counts["eligible_atom_count"] > 32
    assert counts["conjunction_atom_count"] == 32
    assert counts["candidate_clause_count"] <= counts["eligible_atom_count"] + 32 * 31 // 2
    programs[0]["examples"].reverse()
    assert contracts.compile_library(
        programs, list(reversed(demos)), dict(reversed(list(transitions.items()))), [],
        max_literals=2,
    ) == library


def test_native_retrieval_ties_prefer_length_then_id_independent_of_action_order():
    base, demos, transitions = bundle(length=4)
    programs = []
    for identifier, length in (("z-short", 2), ("z-long", 3), ("a-long", 3)):
        program = deepcopy(base[0])
        program["program_id"] = identifier
        program["actions"] = program["actions"][:length]
        for example in program["examples"]:
            example["demonstration_ids"] = example["demonstration_ids"][:length]
        programs.append(program)
    library = contracts.compile_library(programs, demos, transitions, [])
    policy = invocation.build_policy(demos)
    original = deepcopy(demos[0]["public_state"])
    first = controller(library, policy).decide(original)
    original["actions"].reverse()
    second = controller(library, policy).decide(original)
    assert first == second
    assert first["proposal"]["program_id"] == "a-long"
    assert first["proposal"]["planned_burst_length"] == 3
    assert first["proposal"]["entry_source"]["example_id"] == demos[0]["example_id"]
    assert library["entry_index"][demos[0]["example_id"]] == sorted(
        row["instance_id"] for row in library["programs"]
    )


def test_entry_prefers_similarity_before_length(monkeypatch):
    short, first, first_transitions = bundle(name="short")
    long, second, second_transitions = bundle(name="long", length=3)
    library = contracts.compile_library(
        short + long, first + second, {**first_transitions, **second_transitions}, [],
    )
    policy = invocation.build_policy(first + second)
    fixed_neighbors(monkeypatch, [(0.8, second[0]), (0.9, first[0])])
    state = public(actions=["act gamma", "act beta", "act alpha"])
    assert controller(library, policy).decide(state)["proposal"]["program_id"] == "short"


@pytest.mark.parametrize("mode", contracts.MODES)
@pytest.mark.parametrize("score,accepted", [(0.55, True), (0.549, False)])
def test_threshold_is_inclusive_and_shared_across_modes(monkeypatch, mode, score, accepted):
    library, policy, demos = compile_bundle(excluded_groups={"withheld"})
    calls = fixed_neighbors(monkeypatch, [(score, demos[0])])
    result = controller(library, policy, mode).decide(public())
    assert result["decision"]["commit"] is accepted
    retrievals = 2 if mode == "independent_actions" and accepted else 1
    assert calls == [{"excluded_groups": frozenset({"target", "withheld"}), "limit": 128}] * retrievals
    if not accepted:
        assert result["decision"]["entry_rejections"][0]["reason"] == "below_entry_threshold"


@pytest.mark.parametrize("mode", contracts.MODES)
def test_first_fragment_action_must_be_native(monkeypatch, mode):
    library, policy, demos = compile_bundle()
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    result = controller(library, policy, mode).decide(public(actions=["act beta"]))
    assert result["proposal"]["action"] is None
    assert not result["decision"]["commit"]
    assert result["decision"]["entry_rejections"][0]["reason"] == "first_action_not_native"


@pytest.mark.parametrize("mode", contracts.MODES)
def test_goal_word_requirements_are_generic_and_not_source_identifiers(monkeypatch, mode):
    library, policy, demos = compile_bundle(name="private_floor_key", task="Alpha, BETA requirement")
    source = library["programs"][0]["examples"][0]
    assert source["goal_words"] == ["alpha", "beta"]
    predictive = [source["goal_words"], [
        literals(step["positive_guard"]) for step in library["programs"][0]["steps"]
    ]]
    assert "private_floor_key" not in json.dumps(predictive)
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    result = controller(library, policy, mode).decide(public(task="alpha requirement"))
    assert not result["decision"]["commit"]
    assert result["decision"]["entry_rejections"][0]["reason"] == "source_task_text_mismatch"


def test_identical_actions_have_separate_goal_conditioned_instances_and_guards():
    library, _, _ = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        task=lambda group, _: "alpha beta plain" if group in ("g0", "g1") else "alpha beta tagged",
        observations=lambda group, _: ["x", "plain", "end"] if group in ("g0", "g1")
        else ["x", "tagged", "end"],
        pre_action_history=lambda group, _: ["alpha"] if group in ("g0", "g1") else ["beta"],
    )
    assert library["schema_version"] == "chisel-observable-fragment-contracts/v5"
    assert len(library["programs"]) == 2
    assert {row["program_id"] for row in library["programs"]} == {"p"}
    assert len({row["instance_id"] for row in library["programs"]}) == 2
    assert library["counts"]["input_programs"] == library["counts"]["retained_programs"] == 1
    assert library["counts"]["input_conditioned_instances"] == 2
    assert library["counts"]["retained_instances"] == 2
    for program in library["programs"]:
        plain = program["task_text"] == "alpha beta plain"
        assert program["source_groups"] == (["g0", "g1"] if plain else ["g2", "g3"])
        assert program["history_prerequisite_words"] == (["alpha"] if plain else ["beta"])
        for variant in ("positive_guard", "refined_guard"):
            guard = program["steps"][0][variant]
            assert literals(guard) == [(("+", ("plain" if plain else "tagged",)),)]
            assert guard["clauses"][0]["positive_group_count"] == 2
            assert guard["positive_groups"] == program["source_groups"]
        for example in program["examples"]:
            assert library["entry_index"][example["demonstration_ids"][0]] == [program["instance_id"]]


def test_group_support_cannot_be_pooled_across_different_goals():
    library, _, _ = compile_bundle(task=lambda group, _: f"alpha beta {group}")
    assert library["programs"] == []
    assert library["counts"]["input_conditioned_instances"] == 2
    assert library["counts"]["rejected_instances"] == 2
    assert library["counts"]["rejected_programs"] == 1
    assert all(row["reason"] == "insufficient_source_groups" for row in library["rejected_programs"])
    assert all(row["counts"]["source_group_count"] == 1 for row in library["rejected_programs"])


def test_goal_splitting_can_retain_one_instance_and_reject_another():
    library, _, _ = compile_bundle(
        groups=("g0", "g1", "g2"),
        task=lambda group, _: "alpha beta first" if group != "g2" else "alpha beta second",
    )
    assert len(library["programs"]) == len(library["rejected_programs"]) == 1
    assert library["programs"][0]["task_text"] == "alpha beta first"
    assert library["rejected_programs"][0]["task_text"] == "alpha beta second"
    assert library["counts"]["retained_programs"] == 1
    assert library["counts"]["rejected_programs"] == 0
    assert library["counts"]["retained_instances"] == library["counts"]["rejected_instances"] == 1
    assert library["fit_groups"] == ["g0", "g1"]


@pytest.mark.parametrize("mode", contracts.MODES)
def test_only_case_and_whitespace_are_normalized_for_goal_matching(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        task=lambda group, _: " ALPHA\t beta\n" if group == "g0" else "alpha beta",
    )
    assert len(library["programs"]) == 1
    assert library["programs"][0]["task_text"] == "alpha beta"
    assert library["programs"][0]["source_groups"] == ["g0", "g1"]
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    accepted = controller(library, policy, mode).decide(public(task=" \nAlpha  BETA\t"))
    assert accepted["decision"]["commit"]
    assert accepted["proposal"]["entry_source"]["task_text"] == "alpha beta"
    for task in ("beta alpha", "alpha beta extra", "alpha, beta", "alpha gamma"):
        denied = controller(library, policy, mode).decide(public(task=task))
        assert not denied["decision"]["commit"]
        assert denied["decision"]["entry_rejections"][0]["reason"] == "source_task_text_mismatch"


@pytest.mark.parametrize("mode", contracts.MODES)
def test_entry_skips_higher_similarity_instances_of_another_goal(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        task=lambda group, _: "alpha beta first" if group in ("g0", "g1") else "alpha beta second",
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0]), (0.8, demos[4])])
    result = controller(library, policy, mode).decide(public(task="alpha beta second"))
    assert result["decision"]["commit"]
    assert result["proposal"]["program_id"] == "p"
    assert result["proposal"]["entry_source"]["task_text"] == "alpha beta second"
    selected = next(row for row in library["programs"] if row["task_text"] == "alpha beta second")
    assert result["proposal"]["instance_id"] == selected["instance_id"]
    assert result["decision"]["entry_rejections"][0]["reason"] == "source_task_text_mismatch"


def test_history_prerequisites_intersect_every_source_example_not_group_representatives():
    histories = {
        ("g0", 0): ["alpha beta gamma irrelevant"],
        ("g0", 1): ["alpha beta gamma"],
        ("g1", 0): ["alpha beta delta"],
        ("g1", 1): ["alpha beta gamma delta"],
    }
    library, _, _ = compile_bundle(
        prefixes=2, pre_action_history=lambda group, prefix: histories[group, prefix],
    )
    program = library["programs"][0]
    assert program["history_prerequisite_words"] == ["alpha", "beta"]
    assert len(program["examples"]) == 4
    assert program["counts"]["source_group_count"] == 2
    histories["g0", 1] = []
    library, _, _ = compile_bundle(
        prefixes=2, pre_action_history=lambda group, prefix: histories[group, prefix],
    )
    assert library["programs"][0]["history_prerequisite_words"] == []


def test_history_prerequisite_can_be_a_task_word_absent_from_the_fragment_actions():
    library, policy, _ = compile_bundle(pre_action_history=["gamma"])
    program = library["programs"][0]
    assert program["examples"][0]["goal_words"] == ["alpha", "beta"]
    assert program["history_prerequisite_words"] == ["gamma"]
    control = controller(library, policy)
    denied = control.decide(public())
    assert denied["decision"]["entry_rejections"][0]["missing_history_prerequisite_words"] == [
        "gamma",
    ]
    control.observe("actual gamma", "instructions alpha beta", compiled=False)
    assert control.decide(public())["decision"]["commit"]


def test_goal_instance_ids_and_history_requirements_survive_ordering_and_serialization():
    programs, demos, transitions = bundle(
        groups=("g0", "g1", "g2", "g3"),
        task=lambda group, _: "alpha beta first" if group in ("g0", "g1") else "alpha beta second",
        pre_action_history=["alpha", "alpha"],
    )
    negatives = [negative("x", group="n1"), negative("", group="n2")]
    library = contracts.compile_library(programs, demos, transitions, negatives, max_literals=2)
    policy = invocation.build_policy(demos)
    programs[0]["examples"].reverse()
    assert library == contracts.compile_library(
        programs, list(reversed(demos)), dict(reversed(list(transitions.items()))),
        list(reversed(negatives)), max_literals=2,
    )
    restored = json.loads(json.dumps(library, allow_nan=False))
    assert restored == library
    transitions[demos[0]["example_id"]]["pre_action_history"].append("mutated")
    assert all(len(example["pre_action_history"]) == 2
               for program in restored["programs"] for example in program["examples"])
    control = controller(restored, policy)
    control.observe("actual alpha", "x", compiled=False)
    result = control.decide(public(task="alpha beta second"))
    assert result["decision"]["commit"]
    assert result["proposal"]["entry_source"]["task_text"] == "alpha beta second"
    assert result["proposal"]["entry_source"]["history_prerequisite_words"] == ["alpha"]


def test_excluded_and_incomplete_examples_do_not_change_history_prerequisites():
    programs, demos, transitions = bundle(
        groups=("g0", "g1", "target", "incomplete"),
        pre_action_history=lambda group, _: ["alpha"] if group in ("g0", "g1") else [],
    )
    del transitions[demos[-1]["example_id"]]
    library = contracts.compile_library(programs, demos, transitions, [], excluded_groups={"target"})
    assert library["programs"][0]["history_prerequisite_words"] == ["alpha"]
    assert library["programs"][0]["source_groups"] == ["g0", "g1"]
    assert library["counts"]["excluded_examples"] == library["counts"]["incomplete_examples"] == 1


def test_source_task_and_observation_instruction_text_cannot_supply_history_words():
    library, _, _ = compile_bundle(
        task="alpha beta gamma delta",
        observations=["instructions alpha beta gamma delta", "changed", "final"],
        pre_action_history=["unrelated action"],
    )
    assert library["programs"][0]["history_prerequisite_words"] == ["action", "unrelated"]
    library, _, _ = compile_bundle(pre_action_history=["unrelated action", "act alpha"])
    assert library["programs"][0]["history_prerequisite_words"] == [
        "act", "action", "alpha", "unrelated",
    ]


@pytest.mark.parametrize("mode", contracts.MODES)
@pytest.mark.parametrize("initial_actor_first", [False, True])
def test_every_mode_requires_observed_action_history_including_initial_actor(
    monkeypatch, mode, initial_actor_first,
):
    library, policy, demos = compile_bundle(
        pre_action_history=["ALPHA"],
        observations=["instructions alpha beta gamma delta", "result", "final"],
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, mode)
    state = public(
        "instructions alpha beta gamma delta",
        history=[{"action": "unobserved alpha", "observation": "alpha instructions"}],
    )
    if not initial_actor_first:
        denied = control.decide(state)
        assert not denied["decision"]["commit"]
        assert denied["decision"]["history_prerequisites_enabled"]
        assert denied["decision"]["completed_action_count"] == 0
        rejection = denied["decision"]["entry_rejections"][0]
        assert rejection["reason"] == "history_prerequisites_missing"
        assert rejection["missing_history_prerequisite_words"] == ["alpha"]
        control.observe("alphabet actual action", "alpha beta instructions", compiled=False)
        assert not control.decide(state)["decision"]["commit"]
    feedback = control.observe("actual ALPHA action", "unrelated observation", compiled=False)
    assert feedback["completed_action_count"] == (1 if initial_actor_first else 2)
    accepted = control.decide(state)
    assert accepted["decision"]["commit"]
    assert accepted["decision"]["entry_prerequisites_checked"]
    assert accepted["proposal"]["entry_source"]["history_prerequisite_words"] == ["alpha"]


@pytest.mark.parametrize("mode", contracts.MODES)
def test_compiled_actual_actions_supply_later_history_prerequisites(monkeypatch, mode):
    first, first_demos, first_transitions = bundle(name="first")
    prior = ["act alpha", "act beta"] if mode in _COMPOSED_SOURCE_MODES else ["beta"]
    second, second_demos, second_transitions = bundle(
        name="second", length=3, pre_action_history=prior,
    )
    library = contracts.compile_library(
        first + second, first_demos + second_demos, {**first_transitions, **second_transitions}, [],
    )
    policy = invocation.build_policy(first_demos + second_demos)
    fixed_neighbors(monkeypatch, [(1.0, second_demos[0]), (0.9, first_demos[0])])
    if mode == "independent_actions":
        monkeypatch.setattr(invocation, "propose", lambda *args, **kwargs: {"action": "act beta"})
    control = controller(library, policy, mode)
    actions = ["act alpha", "act beta", "act gamma"]
    for step, (before, after) in enumerate((("x", "alpha"), ("alpha", "beta"))):
        result = control.decide(public(before, actions=actions))
        assert result["proposal"]["program_id"] == "first"
        feedback = control.observe(result["proposal"]["action"], after, compiled=True)
        assert feedback["completed_action_count"] == step + 1
    assert control.decide(public("x next", actions=actions))["decision"][
        "reason"
    ] == "actor_boundary_required"
    control.observe("neutral actor", "x next", compiled=False)
    result = control.decide(public("x next", actions=actions))
    assert result["decision"]["commit"]
    assert result["proposal"]["program_id"] == "second"
    assert result["proposal"]["entry_source"]["history_prerequisite_words"] == (
        ["act", "alpha", "beta"] if mode in _COMPOSED_SOURCE_MODES else ["beta"]
    )
    assert result["decision"]["completed_action_count"] == 3


def test_no_memory_removes_only_reentry_registry_not_counter_or_history(monkeypatch):
    library, policy, demos = compile_bundle(pre_action_history=["alpha"])
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, "refined_no_memory")
    assert not control.decide(public())["decision"]["commit"]
    control.observe("initial alpha", "x", compiled=False)
    for iteration in range(2):
        for step, (before, after) in enumerate((("x", "alpha"), ("alpha", "beta"))):
            result = control.decide(public(before))
            assert result["decision"]["commit"]
            assert result["proposal"]["step"] == step
            feedback = control.observe(result["proposal"]["action"], after, compiled=True)
            assert feedback["next_step"] == (1 if step == 0 else None)
            assert feedback["seen_entry_count"] == 0
            assert feedback["completed_action_count"] == 2 + iteration * 3 + step
        assert control.decide(public())["decision"]["reason"] == "actor_boundary_required"
        control.observe("neutral actor", "x", compiled=False)
    control.reset_episode()
    denied = control.decide(public())
    assert not denied["decision"]["commit"]
    assert denied["decision"]["completed_action_count"] == 0
    assert denied["decision"]["entry_rejections"][0]["reason"] == "history_prerequisites_missing"


@pytest.mark.parametrize("mode", contracts.MODES)
def test_episode_reset_clears_completed_action_prerequisites_in_every_mode(monkeypatch, mode):
    library, policy, demos = compile_bundle(pre_action_history=["alpha", "alpha"])
    assert library["programs"][0]["history_prerequisite_words"] == ["alpha"]
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, mode)
    control.observe("initial alpha", "x", compiled=False)
    first = control.decide(public())
    control.observe(first["proposal"]["action"], "alpha", compiled=True)
    control.reset_episode()
    denied = control.decide(public())
    assert not denied["decision"]["commit"]
    assert denied["decision"]["completed_action_count"] == 0
    assert denied["decision"]["entry_rejections"][0]["reason"] == "history_prerequisites_missing"
    control.observe("new episode alpha", "x", compiled=False)
    assert control.decide(public())["proposal"]["step"] == 0


@pytest.mark.parametrize("mode", ["positive_contract", "refined_contract", "refined_no_memory"])
def test_failed_postcondition_does_not_erase_the_actual_executed_action(monkeypatch, mode):
    first, first_demos, first_transitions = bundle(name="first")
    second, second_demos, second_transitions = bundle(
        name="second", length=3, pre_action_history=["alpha"],
    )
    library = contracts.compile_library(
        first + second, first_demos + second_demos, {**first_transitions, **second_transitions}, [],
    )
    policy = invocation.build_policy(first_demos + second_demos)
    fixed_neighbors(monkeypatch, [(1.0, second_demos[0]), (0.9, first_demos[0])])
    control = controller(library, policy, mode)
    actions = ["act alpha", "act beta", "act gamma"]
    first = control.decide(public(actions=actions))
    assert first["proposal"]["program_id"] == "first"
    feedback = control.observe(first["proposal"]["action"], "x", compiled=True)
    assert feedback["postcondition_passed"] is False
    assert feedback["completed_action_count"] == 1
    control.observe("neutral actor", "x next", compiled=False)
    result = control.decide(public("x next", actions=actions))
    assert result["decision"]["commit"]
    assert result["proposal"]["program_id"] == "second"
    assert result["decision"]["completed_action_count"] == 2


@pytest.mark.parametrize("mode", contracts.MODES)
def test_active_goal_change_aborts_in_all_modes(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode)
    first = control.decide(public())
    control.observe(first["proposal"]["action"], "alpha", compiled=True)
    denied = control.decide(public("alpha", task="alpha beta gamma delta other"))
    assert not denied["decision"]["commit"]
    assert denied["decision"]["reason"] == "active_task_text_mismatch"
    assert denied["proposal"]["step"] == 1


def test_supplied_family_pools_paraphrases_not_different_families():
    library, _, _ = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        task=lambda group, _: f"alpha beta request {group}",
        task_family=lambda group, _: "opaque-a" if group in ("g0", "g1") else "opaque-b",
        observations=lambda group, _: ["x", "first", "end"] if group in ("g0", "g1")
        else ["x", "second", "end"],
        pre_action_history=lambda group, _: ["prepare sigma", f"unique_{group}"],
        negatives=[
            negative("x", group="na", task_family="opaque-a"),
            negative("", group="nb", task_family="opaque-b"),
            negative("x", group="unlabeled"),
        ],
    )
    assert len(library["programs"]) == 2
    assert library["counts"]["input_conditioned_instances"] == 2
    assert library["counts"]["negative_transitions_unused_by_retained_instances"] == 1
    assert "unlabeled" not in library["fit_groups"]
    for program in library["programs"]:
        first = program["task_family"] == "opaque-a"
        assert program["program_id"] == "p"
        assert program["conditioning"] == "task_family"
        assert program["task_text"] is None
        assert program["source_groups"] == (["g0", "g1"] if first else ["g2", "g3"])
        assert program["history_prerequisite_words"] == ["prepare", "sigma"]
        assert len({row["task_text"] for row in program["examples"]}) == 2
        for variant in ("positive_guard", "refined_guard"):
            guard = program["steps"][0][variant]
            assert guard["positive_groups"] == program["source_groups"]
            assert guard["negative_groups"] == (["na"] if first else ["nb"])
            assert guard["clauses"][0]["positive_group_count"] == 2
        assert literals(program["steps"][0]["positive_guard"]) == [
            (("+", ("first" if first else "second",)),),
        ]
        for example in program["examples"]:
            assert library["entry_index"][example["demonstration_ids"][0]] == [program["instance_id"]]


def test_group_support_cannot_cross_supplied_families_even_for_identical_tasks():
    library, _, _ = compile_bundle(task_family=lambda group, _: group)
    assert library["programs"] == []
    assert library["counts"]["rejected_instances"] == 2
    assert all(row["counts"]["source_group_count"] == 1 for row in library["rejected_programs"])


def test_family_and_exact_text_conditioning_have_separate_identity_namespaces():
    same_text = "alpha beta gamma delta"
    library, _, _ = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        task_family=lambda group, _: same_text if group in ("g0", "g1") else None,
    )
    assert len(library["programs"]) == 2
    assert {row["conditioning"] for row in library["programs"]} == {"task_family", "exact_task_text"}
    assert len({row["instance_id"] for row in library["programs"]}) == 2


@pytest.mark.parametrize("mode", contracts.MODES)
def test_family_entry_accepts_paraphrases_but_checks_every_source_word_binding(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        task=lambda group, _: f"alpha beta source {group}",
        task_family="opaque-label",
        pre_action_history=lambda group, _: ["prepare sigma", f"unique_{group}"],
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, mode, task_family="opaque-label")
    state = public("instructions prepare sigma", task="a different request for beta and ALPHA")
    rejected = control.decide(state)
    assert not rejected["decision"]["commit"]
    assert rejected["decision"]["entry_rejections"][0]["missing_history_prerequisite_words"] == [
        "prepare", "sigma",
    ]
    control.observe("PREPARE sigma", "x", compiled=False, native_valid=True)
    missing_binding = control.decide(public("instructions beta", task="a request for alpha"))
    assert not missing_binding["decision"]["commit"]
    assert missing_binding["decision"]["entry_rejections"][0]["reason"] == "source_goal_words_missing"
    control.observe("neutral", "x", compiled=False)
    accepted = control.decide(public(task="a different request for beta and ALPHA"))
    assert accepted["decision"]["commit"]
    assert accepted["proposal"]["task_family"] == "opaque-label"
    assert accepted["proposal"]["conditioning"] == "task_family"
    assert accepted["proposal"]["entry_source"]["goal_words"] == ["alpha", "beta"]
    assert accepted["proposal"]["entry_source"]["history_prerequisite_words"] == ["prepare", "sigma"]


@pytest.mark.parametrize("mode", contracts.MODES)
def test_family_entry_skips_a_higher_similarity_different_family(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        task_family=lambda group, _: "opaque-a" if group in ("g0", "g1") else "opaque-b",
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0]), (0.8, demos[4])])
    result = controller(library, policy, mode, task_family="opaque-b").decide(
        public(task="beta then alpha"),
    )
    assert result["decision"]["commit"]
    assert result["proposal"]["task_family"] == "opaque-b"
    assert result["decision"]["entry_rejections"][0]["reason"] == "task_family_mismatch"


@pytest.mark.parametrize("mode", contracts.MODES)
@pytest.mark.parametrize("source_family,current_family,reason", [
    ("opaque", None, "task_family_required"),
    (None, "opaque", "source_task_family_missing"),
    ("opaque", "OPAQUE", "task_family_mismatch"),
])
def test_missing_or_different_family_never_falls_back_to_inferred_labels(
    monkeypatch, mode, source_family, current_family, reason,
):
    library, policy, demos = compile_bundle(task_family=source_family)
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    denied = controller(library, policy, mode, task_family=current_family).decide(public())
    assert not denied["decision"]["commit"]
    assert denied["decision"]["entry_rejections"][0]["reason"] == reason


def test_explicit_no_family_metadata_preserves_exact_text_fallback():
    programs, demos, transitions = bundle(task="opaque family instructions alpha beta")
    for example in programs[0]["examples"]:
        example["task_family"] = None
    for transition in transitions.values():
        transition["task_family"] = None
    library = contracts.compile_library(programs, demos, transitions, [])
    assert library["programs"][0]["conditioning"] == "exact_task_text"
    assert library["programs"][0]["task_family"] is None
    policy = invocation.build_policy(demos)
    assert controller(library, policy).decide(demos[0]["public_state"])["decision"]["commit"]
    assert not controller(library, policy).decide(public())["decision"]["commit"]


def test_native_filtered_source_history_may_be_shorter_than_source_turn_count():
    programs, demos, transitions = bundle(
        task_family="opaque", pre_action_history=["prepare sigma"],
    )
    for example in programs[0]["examples"]:
        example["start_turn"] += 2
    library = contracts.compile_library(programs, demos, transitions, [])
    assert library["programs"][0]["history_prerequisite_words"] == ["prepare", "sigma"]
    assert all(row["start_turn"] == 3 and len(row["pre_action_history"]) == 1
               for row in library["programs"][0]["examples"])
    control = controller(library, invocation.build_policy(demos), task_family="opaque")
    control.observe("prepare sigma", "x", compiled=False)
    assert control.decide(public())["decision"]["commit"]


@pytest.mark.parametrize("mode", contracts.MODES)
def test_non_native_actor_feedback_is_logged_but_never_supplies_prerequisites(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        task_family="opaque", pre_action_history=["prepare sigma"],
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, mode, task_family="opaque")
    feedback = control.observe(
        "prepare sigma", "instructions prepare sigma", compiled=False, native_valid=False,
    )
    assert feedback["action"] == "prepare sigma"
    assert feedback["observation"] == "instructions prepare sigma"
    assert feedback["native_valid"] is False
    assert feedback["history_recorded"] is False
    assert feedback["reason"] == "actor_action_not_native"
    assert feedback["completed_action_count"] == 1
    assert feedback["native_action_count"] == 0
    denied = control.decide(public("instructions prepare sigma"))
    assert not denied["decision"]["commit"]
    assert denied["decision"]["entry_rejections"][0]["reason"] == "history_prerequisites_missing"
    feedback = control.observe("prepare sigma", "x", compiled=False, native_valid=True)
    assert feedback["completed_action_count"] == 2
    assert feedback["native_action_count"] == 1
    assert feedback["history_recorded"] is True
    assert control.decide(public())["decision"]["commit"]


@pytest.mark.parametrize("mode", contracts.MODES)
def test_non_native_compiled_feedback_aborts_without_history_or_postcondition_credit(
    monkeypatch, mode,
):
    first, first_demos, first_transitions = bundle(name="first", task_family="opaque")
    second, second_demos, second_transitions = bundle(
        name="second", task_family="opaque", pre_action_history=["alpha"],
    )
    library = contracts.compile_library(
        first + second, first_demos + second_demos, {**first_transitions, **second_transitions}, [],
    )
    policy = invocation.build_policy(first_demos + second_demos)
    neighbors = [(1.0, first_demos[0])]
    fixed_neighbors(monkeypatch, neighbors)
    control = controller(library, policy, mode, task_family="opaque")
    result = control.decide(public())
    feedback = control.observe(result["proposal"]["action"], "alpha", compiled=True, native_valid=False)
    assert feedback["fragment_aborted"]
    assert not feedback["fragment_complete"]
    assert feedback["next_step"] is None
    assert feedback["postcondition_passed"] is None
    assert not feedback["postcondition_checked"]
    assert feedback["reason"] == "compiled_action_not_native"
    assert feedback["completed_action_count"] == 1
    assert feedback["native_action_count"] == 0
    boundary = control.decide(public("x next"))["decision"]
    assert boundary["reason"] == "actor_boundary_required"
    assert boundary["boundary_reason"] == "compiled_action_not_native"
    control.observe("neutral", "alpha instructions", compiled=False, native_valid=True)
    neighbors[:] = [(1.0, second_demos[0])]
    denied = control.decide(public("x next"))
    assert not denied["decision"]["commit"]
    assert denied["decision"]["entry_rejections"][0]["missing_history_prerequisite_words"] == ["alpha"]
    assert denied["decision"]["completed_action_count"] == 2
    assert denied["decision"]["native_action_count"] == 1


def test_non_native_feedback_does_not_erase_earlier_native_history():
    library, policy, _ = compile_bundle(pre_action_history=["alpha"])
    control = controller(library, policy, "refined_no_memory")
    control.observe("alpha", "x", compiled=False)
    control.observe("", "actual malformed action feedback", compiled=False, native_valid=False)
    result = control.decide(public())
    assert result["decision"]["commit"]
    assert result["decision"]["native_action_count"] == 1
    assert result["decision"]["completed_action_count"] == 2
    control.observe(result["proposal"]["action"], "alpha", compiled=True)
    control.reset_episode()
    assert not control.decide(public())["decision"]["commit"]


def test_family_library_and_native_feedback_round_trip_without_aliasing():
    programs, demos, transitions = bundle(
        task=lambda group, _: f"alpha beta request {group}",
        task_family="opaque", pre_action_history=["prepare sigma"],
    )
    library = contracts.compile_library(programs, demos, transitions, [], max_literals=2)
    policy = invocation.build_policy(demos)
    restored = json.loads(json.dumps(library, allow_nan=False))
    programs[0]["examples"].reverse()
    assert library == contracts.compile_library(
        programs, list(reversed(demos)), dict(reversed(list(transitions.items()))), [], max_literals=2,
    )
    assert library == restored
    transitions[demos[0]["example_id"]]["task_family"] = "changed"
    control = controller(restored, policy, task_family="opaque")
    control.observe("prepare sigma", "x", compiled=False, native_valid=True)
    result = control.decide(deepcopy(demos[0]["public_state"]))
    assert result["decision"]["commit"]
    feedback = control.observe(result["proposal"]["action"], "alpha", compiled=True, native_valid=True)
    assert json.loads(json.dumps(feedback, allow_nan=False)) == feedback
    assert feedback["native_action_count"] == feedback["completed_action_count"] == 2


@pytest.mark.parametrize("mode", contracts.MODES)
def test_family_source_bindings_are_specific_to_the_selected_entry(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        task=lambda group, _: "act alpha beta" if group == "g0" else "alpha beta request",
        task_family="opaque",
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0]), (0.8, demos[2])])
    result = controller(library, policy, mode, task_family="opaque").decide(
        public(task="another alpha beta request"),
    )
    assert result["decision"]["commit"]
    assert result["proposal"]["entry_source"]["example_id"] == demos[2]["example_id"]
    assert result["decision"]["entry_rejections"][0]["reason"] == "source_goal_words_missing"


@pytest.mark.parametrize("mode", contracts.MODES)
def test_family_continuation_allows_paraphrases_but_preserves_literal_bindings(monkeypatch, mode):
    library, policy, demos = compile_bundle(task_family="opaque")
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    for task, accepted in (("beta and alpha again", True), ("alpha without the other binding", False)):
        control = controller(library, policy, mode, task_family="opaque")
        first = control.decide(public())
        control.observe(first["proposal"]["action"], "alpha", compiled=True)
        result = control.decide(public("alpha", task=task))
        assert result["decision"]["commit"] is accepted
        assert result["proposal"]["step"] == 1
        if not accepted:
            assert result["decision"]["reason"] == "active_goal_words_missing"


def test_exclusion_precedes_family_history_and_negative_fitting():
    programs, demos, transitions = bundle(
        groups=("g0", "g1", "target"),
        task=lambda group, _: f"alpha beta {group}",
        task_family="opaque",
        pre_action_history=lambda group, _: [] if group == "target" else ["prepare sigma"],
    )
    library = contracts.compile_library(
        programs, demos, transitions,
        [negative("x", group="target", task_family="opaque")], excluded_groups={"target"},
    )
    program = library["programs"][0]
    assert program["source_groups"] == ["g0", "g1"]
    assert program["history_prerequisite_words"] == ["prepare", "sigma"]
    assert library["fit_groups"] == ["g0", "g1"]
    assert program["steps"][0]["refined_guard"]["negative_groups"] == []
    assert library["counts"]["excluded_examples"] == 1
    assert library["counts"]["excluded_negative_transitions"] == 1


@pytest.mark.parametrize("metadata", ["example_only", "transition_only", "different", "later_missing"])
def test_partial_or_conflicting_family_metadata_is_rejected(metadata):
    programs, demos, transitions = bundle(task_family="opaque")
    first, second = demos[0]["example_id"], demos[1]["example_id"]
    if metadata == "example_only":
        for transition in transitions.values():
            transition.pop("task_family")
    elif metadata == "transition_only":
        for example in programs[0]["examples"]:
            example.pop("task_family")
    elif metadata == "different":
        transitions[first]["task_family"] = "other"
    else:
        transitions[second].pop("task_family")
    with pytest.raises(ValueError, match="task_family metadata"):
        contracts.compile_library(programs, demos, transitions, [])


@pytest.mark.parametrize("value", [True, 1, "", " ", [], {}])
@pytest.mark.parametrize("location", ["example", "transition", "negative", "controller"])
def test_family_labels_are_strict_opaque_strings_or_absent(value, location):
    programs, demos, transitions = bundle()
    negatives = []
    if location == "example":
        programs[0]["examples"][0]["task_family"] = value
    elif location == "transition":
        transitions[demos[0]["example_id"]]["task_family"] = value
    elif location == "negative":
        negatives.append(negative("x", task_family=value))
    else:
        library = contracts.compile_library(programs, demos, transitions, [])
        with pytest.raises(ValueError, match="task_family"):
            controller(library, invocation.build_policy(demos), task_family=value)
        return
    with pytest.raises(ValueError, match="task_family"):
        contracts.compile_library(programs, demos, transitions, negatives)


@pytest.mark.parametrize("value", ["", " "])
def test_source_native_history_cannot_contain_empty_action_strings(value):
    programs, demos, transitions = bundle(pre_action_history=[value])
    with pytest.raises(ValueError, match="native source action"):
        contracts.compile_library(programs, demos, transitions, [])


@pytest.mark.parametrize("mutation", [
    lambda lib: lib["programs"][0].update(task_family="other"),
    lambda lib: lib["programs"][0].update(conditioning="exact_task_text"),
    lambda lib: lib["programs"][0].update(conditioning=True),
    lambda lib: lib["programs"][0].update(task_text="invented exact goal"),
    lambda lib: lib["programs"][0]["examples"][0].update(task_family="other"),
    lambda lib: lib["programs"][0]["examples"][0].update(task_family=None),
    lambda lib: lib["programs"][0]["examples"][0].update(task_text="alpha changed binding"),
    lambda lib: lib["programs"][0]["examples"][0].update(goal_words=[]),
    lambda lib: lib["parameters"].update(negative_conditioning="cross_family"),
])
def test_family_library_cannot_reassign_provenance_or_weaken_bindings(mutation):
    library, policy, _ = compile_bundle(task_family="opaque")
    mutation(library)
    with pytest.raises(ValueError):
        controller(library, policy, task_family="opaque")


@pytest.mark.parametrize("mode,chosen", [
    ("ordered_fragment", "a-first"), ("independent_actions", "a-first"),
    ("positive_contract", "b-next"), ("refined_contract", "b-next"),
    ("refined_no_memory", "b-next"),
])
def test_guarded_entry_scans_the_next_candidate_but_controls_do_not(monkeypatch, mode, chosen):
    first, first_demos, first_transitions = bundle(name="a-first")
    second, second_demos, second_transitions = bundle(
        name="b-next", observations=["ready", "gamma", "delta"],
    )
    library = contracts.compile_library(
        first + second, first_demos + second_demos, {**first_transitions, **second_transitions}, [],
    )
    policy = invocation.build_policy(first_demos + second_demos)
    fixed_neighbors(monkeypatch, [(0.9, first_demos[0]), (0.8, second_demos[0])])
    result = controller(library, policy, mode).decide(public("alpha ready"))
    assert result["decision"]["commit"]
    assert result["proposal"]["program_id"] == chosen
    assert result["decision"]["entry_guards_used"] is (mode not in (
        "ordered_fragment", "independent_actions",
    ))
    if chosen == "b-next":
        assert result["decision"]["entry_rejections"][0]["reason"] == "entry_precondition_failed"


@pytest.mark.parametrize("mode", [
    "ordered_fragment", "positive_contract", "refined_contract", "refined_no_memory",
])
def test_program_counter_moves_only_after_execution_and_requires_actor_boundary(mode):
    library, policy, demos = compile_bundle(length=3)
    control = controller(library, policy, mode)
    assert control.observe("initial actor", "x", compiled=False)["reason"] == "actor_feedback"
    for index, observation in enumerate(("x", "alpha", "beta")):
        result = control.decide(public(observation, actions=library["programs"][0]["actions"]))
        assert result["decision"]["commit"]
        assert result["proposal"]["step"] == index
        with pytest.raises(ValueError, match="observe"):
            control.decide(demos[0]["public_state"])
        with pytest.raises(ValueError, match="pending action"):
            control.observe("wrong action", observation, compiled=True)
        feedback = control.observe(result["proposal"]["action"], ("alpha", "beta", "gamma")[index],
                                   compiled=True)
        assert feedback["step"] == index
        assert feedback["next_step"] == (index + 1 if index < 2 else None)
        with pytest.raises(ValueError, match="pending committed"):
            control.observe(result["proposal"]["action"], observation, compiled=True)
    assert feedback["fragment_complete"]
    deferred = control.decide(demos[0]["public_state"])
    assert not deferred["decision"]["commit"]
    assert deferred["decision"]["reason"] == "actor_boundary_required"
    assert deferred["decision"]["boundary_reason"] == "fragment_completed"
    with pytest.raises(ValueError, match="observe"):
        control.decide(demos[0]["public_state"])
    control.observe("actor", "x changed", compiled=False)
    assert control.decide(public(
        "x changed", actions=library["programs"][0]["actions"],
    ))["decision"]["commit"]


@pytest.mark.parametrize("mode", [
    "ordered_fragment", "positive_contract", "refined_contract", "refined_no_memory",
])
def test_non_native_continuation_aborts_without_repair(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode)
    first = control.decide(public())
    control.observe(first["proposal"]["action"], "alpha", compiled=True)
    denied = control.decide(public("alpha", actions=["act alpha"]))
    assert denied["proposal"]["action"] is None
    assert denied["proposal"]["stored_action"] == "act beta"
    assert denied["proposal"]["step"] == 1
    assert denied["decision"]["reason"] == "fragment_action_not_native"
    assert not denied["decision"]["commit"]


@pytest.mark.parametrize("mode", ["positive_contract", "refined_contract", "refined_no_memory"])
def test_precondition_failure_aborts_before_execution(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode)
    first = control.decide(public())
    control.observe(first["proposal"]["action"], "alpha", compiled=True)
    denied = control.decide(public("beta"))
    assert not denied["decision"]["commit"]
    assert denied["decision"]["reason"] == "precondition_failed"
    assert denied["decision"]["precondition_checked"]
    with pytest.raises(ValueError, match="pending committed"):
        control.observe("act beta", "", compiled=True)
    actor = control.observe("actor", "x", compiled=False)
    assert actor["cleared_boundary"] == "precondition_failed"


@pytest.mark.parametrize("mode", ["positive_contract", "refined_contract", "refined_no_memory"])
def test_failed_postcondition_defers_and_keeps_episode_memory(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode)
    first = control.decide(public())
    feedback = control.observe(first["proposal"]["action"], "x", compiled=True)
    assert feedback["fragment_aborted"]
    assert feedback["postcondition_passed"] is False
    assert not feedback["fragment_complete"]
    assert feedback["next_step"] is None
    assert control.decide(public())["decision"]["boundary_reason"] == "postcondition_failed"
    actor = control.observe("actor", "x", compiled=False)
    assert actor["cleared_boundary"] == "postcondition_failed"
    decision = control.decide(public())["decision"]
    assert decision["commit"] is (mode == "refined_no_memory")
    if mode != "refined_no_memory":
        assert decision["reason"] == "repeated_fragment_entry_state"


@pytest.mark.parametrize("mode", contracts.MODES)
def test_episode_memory_blocks_reentry_not_continuation_and_reset_is_explicit(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode)
    for observation, after in (("x", "alpha"), ("alpha", "beta")):
        decision = control.decide(public(observation))
        control.observe(decision["proposal"]["action"], after, compiled=True)
    control.observe("actor", "x", compiled=False)
    repeated = public(actions=["act beta", "act alpha"])
    decision = control.decide(repeated)["decision"]
    assert decision["commit"] is (mode in ("refined_no_memory", "composed_no_net_checks"))
    with pytest.raises(ValueError, match="pending"):
        control.reset_episode()
    control.observe("actor", "x", compiled=False)
    control.reset_episode()
    assert control.decide(public())["decision"]["commit"]
    assert controller(library, policy, mode).decide(public())["decision"]["commit"]


def test_independent_policy_actions_share_the_selected_burst_length(monkeypatch):
    library, policy, demos = compile_bundle(length=4)
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    proposed = []

    def propose(policy, state, *, excluded_groups):
        proposed.append(excluded_groups)
        return {"action": "alternative", "source_example_ids": ["p0-source"]}

    monkeypatch.setattr(invocation, "propose", propose)
    independent = controller(library, policy, "independent_actions")
    ordered = controller(library, policy, "ordered_fragment")
    state = public(actions=library["programs"][0]["actions"] + ["alternative"])
    for step in range(4):
        actual = independent.decide(state)
        stored = ordered.decide(state)
        assert actual["proposal"]["planned_burst_length"] == stored["proposal"][
            "planned_burst_length"
        ] == 4
        assert actual["proposal"]["step"] == stored["proposal"]["step"] == step
        assert actual["proposal"]["action"] == "alternative"
        assert stored["proposal"]["action"] == library["programs"][0]["actions"][step]
        assert not actual["decision"]["precondition_checked"]
        independent.observe("alternative", "unchanged", compiled=True)
        ordered.observe(stored["proposal"]["action"], "unchanged", compiled=True)
    assert proposed == [frozenset({"target"})] * 4
    assert independent.decide(state)["decision"]["reason"] == "actor_boundary_required"
    assert ordered.decide(state)["decision"]["reason"] == "actor_boundary_required"
    assert len(proposed) == 4


@pytest.mark.parametrize("action", [None, "not a native action"])
def test_independent_proposal_must_still_be_native(monkeypatch, action):
    library, policy, demos = compile_bundle()
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    monkeypatch.setattr(invocation, "propose", lambda *args, **kwargs: {"action": action})
    result = controller(library, policy, "independent_actions").decide(public())
    assert result["proposal"]["action"] is None
    assert not result["decision"]["commit"]
    assert result["decision"]["reason"] == "no_native_independent_action"


def test_actor_override_does_not_advance_or_mark_an_unexecuted_entry():
    library, policy, _ = compile_bundle()
    control = controller(library, policy)
    assert control.decide(public())["proposal"]["step"] == 0
    feedback = control.observe("actor substitute", "x", compiled=False)
    assert feedback["fragment_aborted"]
    assert feedback["seen_entry_count"] == 0
    assert control.decide(public())["proposal"]["step"] == 0


@pytest.mark.parametrize("mode", contracts.MODES)
def test_controller_rejects_target_group_evidence_in_either_training_role(mode):
    library, policy, _ = compile_bundle(groups=("g0", "target"))
    with pytest.raises(ValueError, match="recompile"):
        controller(library, policy, mode)
    library, policy, _ = compile_bundle(negatives=[negative("x", group="target")])
    with pytest.raises(ValueError, match="recompile"):
        controller(library, policy, mode)


def test_library_and_decision_serialization_and_input_immutability():
    programs, demos, transitions = bundle()
    negatives = [negative("")]
    snapshot = deepcopy((programs, demos, transitions, negatives))
    library = contracts.compile_library(programs, demos, transitions, negatives)
    policy = invocation.build_policy(demos)
    assert (programs, demos, transitions, negatives) == snapshot
    assert library == json.loads(json.dumps(library, allow_nan=False))
    control = controller(json.loads(json.dumps(library)), policy)
    programs[0]["actions"][0] = "changed"
    demos[0]["public_state"]["observation"] = "changed"
    transitions.clear()
    negatives.clear()
    result = control.decide(public())
    assert result["proposal"]["action"] == "act alpha"
    assert result == json.loads(json.dumps(result, allow_nan=False))
    feedback = control.observe(result["proposal"]["action"], "alpha", compiled=True)
    assert feedback == json.loads(json.dumps(feedback, allow_nan=False))


def test_empty_library_defers_without_inventing_a_candidate():
    library = contracts.compile_library([], [], {}, [])
    _, policy, _ = compile_bundle()
    result = controller(library, policy).decide(public())
    assert result["proposal"]["action"] is None
    assert result["decision"]["reason"] == "no_eligible_fragment"


@pytest.mark.parametrize("kwargs", [
    {"minimum_groups": True}, {"minimum_groups": 0}, {"minimum_groups": -1},
    {"minimum_groups": 2.0}, {"minimum_groups": None}, {"max_literals": True},
    {"max_literals": 0}, {"max_literals": 3}, {"max_literals": 1.0},
    {"excluded_groups": ["g0"]}, {"excluded_groups": "g0"}, {"excluded_groups": {True}},
    {"excluded_groups": {""}},
])
def test_invalid_compilation_parameters(kwargs):
    programs, demos, transitions = bundle()
    with pytest.raises(ValueError):
        contracts.compile_library(programs, demos, transitions, [], **kwargs)


@pytest.mark.parametrize("position,value", [
    (0, {}), (0, None), (1, {}), (1, None), (2, []), (2, None), (3, {}), (3, None),
])
def test_invalid_input_container_types(position, value):
    inputs = [*bundle(), []]
    inputs[position] = value
    with pytest.raises(ValueError):
        contracts.compile_library(*inputs)


@pytest.mark.parametrize("mutation", [
    lambda p, d, t, n: p.append(deepcopy(p[0])),
    lambda p, d, t, n: d.append(deepcopy(d[0])),
    lambda p, d, t, n: p[0]["examples"].append(deepcopy(p[0]["examples"][0])),
    lambda p, d, t, n: p[0].pop("program_id"),
    lambda p, d, t, n: p[0].update(actions=["only"]),
    lambda p, d, t, n: p[0].update(actions=["a"] * 5),
    lambda p, d, t, n: p[0].update(actions=[True, "b"]),
    lambda p, d, t, n: p[0].update(actions=("a", "b")),
    lambda p, d, t, n: p[0].update(examples={}),
    lambda p, d, t, n: p[0]["examples"][0].update(start_turn=True),
    lambda p, d, t, n: p[0]["examples"][0].update(start_turn=-1),
    lambda p, d, t, n: p[0]["examples"][0].update(group_id="wrong"),
    lambda p, d, t, n: p[0]["examples"][0].update(demonstration_ids=["one"]),
    lambda p, d, t, n: p[0]["examples"][0].update(demonstration_ids=["one", "one"]),
    lambda p, d, t, n: d[0].update(action="not native"),
    lambda p, d, t, n: d[0].update(group_id="conflicting"),
    lambda p, d, t, n: d[0]["public_state"].update(task_id="not public"),
    lambda p, d, t, n: d[0]["public_state"].update(history=[{"action": "x", "reward": 1}]),
    lambda p, d, t, n: d[0]["public_state"].update(actions=[True]),
    lambda p, d, t, n: d[1]["public_state"].update(observation="not consecutive"),
    lambda p, d, t, n: d[1]["public_state"].update(task="changed task"),
    lambda p, d, t, n: t.update({d[0]["example_id"]: {}}),
    lambda p, d, t, n: t[d[0]["example_id"]].update(post_observation=None),
    lambda p, d, t, n: t[d[0]["example_id"]].pop("pre_action_history"),
    lambda p, d, t, n: t[d[0]["example_id"]].update(pre_action_history=None),
    lambda p, d, t, n: t[d[0]["example_id"]].update(pre_action_history=()),
    lambda p, d, t, n: t[d[0]["example_id"]].update(pre_action_history=[True]),
    lambda p, d, t, n: t[d[0]["example_id"]].update(pre_action_history=[1]),
    lambda p, d, t, n: t[d[0]["example_id"]].update(pre_action_history=["invented prior action"]),
    lambda p, d, t, n: t[d[1]["example_id"]].update(pre_action_history=["wrong previous action"]),
    lambda p, d, t, n: t.update({True: {"post_observation": ""}}),
    lambda p, d, t, n: n.append(negative(True)),
    lambda p, d, t, n: n.append(negative("", group=True)),
    lambda p, d, t, n: n.append(negative("", action=True)),
])
def test_invalid_or_duplicated_training_evidence_is_rejected(mutation):
    programs, demos, transitions = bundle()
    negatives = []
    mutation(programs, demos, transitions, negatives)
    with pytest.raises(ValueError):
        contracts.compile_library(programs, demos, transitions, negatives)


@pytest.mark.parametrize("kwargs", [
    {"mode": "old_value_router"}, {"mode": True}, {"mode": []},
    {"group_id": True}, {"group_id": ""},
    {"entry_threshold": True}, {"entry_threshold": None}, {"entry_threshold": "0.55"},
    {"entry_threshold": float("inf")}, {"entry_threshold": float("nan")},
    {"entry_threshold": -0.1}, {"entry_threshold": 1.1},
])
def test_invalid_controller_parameters(kwargs):
    library, policy, _ = compile_bundle()
    arguments = {"group_id": "target", "mode": "positive_contract", **kwargs}
    with pytest.raises(ValueError):
        contracts.FragmentController(library, policy, **arguments)


@pytest.mark.parametrize("mutation", [
    lambda lib: lib.update(schema_version="unknown"),
    lambda lib: lib.update(schema_version="chisel-observable-fragment-contracts/v1"),
    lambda lib: lib.update(schema_version="chisel-observable-fragment-contracts/v2"),
    lambda lib: lib.update(schema_version="chisel-observable-fragment-contracts/v3"),
    lambda lib: lib.update(parameters={}),
    lambda lib: lib["parameters"].update(minimum_groups=True),
    lambda lib: lib["parameters"].update(max_literals=True),
    lambda lib: lib["parameters"].update(max_clauses=True),
    lambda lib: lib["parameters"].update(task_conditioning="task_word_overlap"),
    lambda lib: lib["parameters"].update(history_prerequisites=False),
    lambda lib: lib["counts"].update(retained_programs=True),
    lambda lib: lib.update(programs={}),
    lambda lib: lib["programs"].append(deepcopy(lib["programs"][0])),
    lambda lib: lib["programs"][0].update(task_text="unnormalized ALPHA"),
    lambda lib: lib["programs"][0].update(task_text=True),
    lambda lib: lib["programs"][0].update(instance_id="wrong"),
    lambda lib: lib["programs"][0].update(program_id="wrong"),
    lambda lib: lib["programs"][0].update(history_prerequisite_words=["alpha"]),
    lambda lib: lib["programs"][0].update(history_prerequisite_words=True),
    lambda lib: lib["programs"][0].update(steps=[]),
    lambda lib: lib["programs"][0]["steps"][0].update(step=False),
    lambda lib: lib["programs"][0]["steps"][0].update(action="wrong"),
    lambda lib: lib["programs"][0]["examples"][0].update(start_turn=True),
    lambda lib: lib["programs"][0]["examples"][0].pop("pre_action_history"),
    lambda lib: lib["programs"][0]["examples"][0].update(pre_action_history=["invented"]),
    lambda lib: lib["programs"][0]["examples"][0].update(pre_action_history=[True]),
    lambda lib: lib["programs"][0]["examples"][0].update(goal_words=["two words"]),
    lambda lib: lib["programs"][0]["examples"][0].update(start_state_sha256="wrong"),
    lambda lib: lib["programs"][0]["steps"][0]["positive_guard"].update(clauses=[]),
    lambda lib: lib["programs"][0]["steps"][0]["positive_guard"]["clauses"][0].update(
        positive_group_count=True,
    ),
    lambda lib: lib["programs"][0]["steps"][0]["positive_guard"]["clauses"][0].update(
        positive_groups=["only"],
    ),
    lambda lib: lib["programs"][0]["steps"][0]["positive_guard"]["clauses"][0].update(
        literals=[{"sign": "?", "tokens": ["alpha"]}],
    ),
    lambda lib: lib["programs"][0]["steps"][0]["positive_guard"]["clauses"][0].update(
        literals=[{"sign": "+", "tokens": [True]}],
    ),
    lambda lib: lib.update(entry_index={"unknown": ["p"]}),
    lambda lib: lib.update(fit_groups=["other"]),
])
def test_invalid_persisted_library_is_rejected(mutation):
    library, policy, _ = compile_bundle()
    mutation(library)
    with pytest.raises(ValueError):
        controller(library, policy)


@pytest.mark.parametrize("policy", [
    None, {}, {"schema_version": "unknown", "records": [], "index": {}},
    {"schema_version": invocation.POLICY_SCHEMA, "records": True, "index": {}},
    {"schema_version": invocation.POLICY_SCHEMA, "records": [{}], "index": {"with_action": 0}},
])
def test_invalid_policy_schema_is_rejected(policy):
    library, _, _ = compile_bundle()
    with pytest.raises(ValueError):
        controller(library, policy)


@pytest.mark.parametrize("kwargs", [
    {"compiled": 0}, {"compiled": 1}, {"compiled": None}, {"compiled": "true"},
    {"native_valid": 0}, {"native_valid": 1}, {"native_valid": None}, {"native_valid": "true"},
    {"native_valid": []}, {"native_valid": {}},
    {"action": True}, {"action": None}, {"observation": None}, {"observation": False},
])
def test_invalid_observe_types_do_not_consume_a_pending_action(kwargs):
    library, policy, _ = compile_bundle()
    control = controller(library, policy)
    first = control.decide(public())
    arguments = {"action": first["proposal"]["action"], "observation": "alpha",
                 "compiled": True, **kwargs}
    with pytest.raises(ValueError):
        control.observe(**arguments)
    assert control.observe(first["proposal"]["action"], "alpha", compiled=True)["next_step"] == 1


def test_hidden_state_and_future_fields_are_rejected_by_decide():
    library, policy, _ = compile_bundle()
    control = controller(library, policy)
    with pytest.raises(ValueError, match="Public state"):
        control.decide({**public(), "official_success": 1})
    assert control.decide(public())["decision"]["commit"]


def test_policy_source_identity_cannot_be_rebound_to_an_entry_id(monkeypatch):
    library, policy, demos = compile_bundle()
    altered = deepcopy(demos[0])
    altered["public_state"]["observation"] = "rebound"
    fixed_neighbors(monkeypatch, [(1.0, altered)])
    with pytest.raises(ValueError, match="compiled demonstration"):
        controller(library, policy).decide(public())


def test_behavior_hash_is_stable_serializable_and_non_mutating():
    library, _, _ = compile_bundle(max_literals=2, task_family="opaque")
    before = deepcopy(library)
    digest = contracts.library_behavior_hash(library)
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert digest == contracts.library_behavior_hash(json.loads(json.dumps(library)))
    assert library == before
    empty = contracts.compile_library([], [], {}, [])
    assert contracts.library_behavior_hash(empty) == contracts.library_behavior_hash(deepcopy(empty))
    assert contracts.library_behavior_hash(empty) != digest


@pytest.mark.parametrize("mode", contracts.MODES)
def test_behavior_hash_ignores_metadata_only_refits_and_preserves_runtime_outputs(mode):
    library, policy, _ = compile_bundle(max_literals=2)
    changed = deepcopy(library)
    changed["semantics"] = "Updated descriptive note; executable contract unchanged"
    changed["objectives"] = {"positive": "Positive fit summary", "refined": "Refined fit summary"}
    changed["fitted_at"] = "synthetic-later-timestamp"
    changed["parameters"]["fitted_at"] = "synthetic-later-timestamp"
    changed["counts"]["input_programs"] += 3
    for program in changed["programs"]:
        program["counts"]["complete_examples"] += 3
        program["source_tasks"].reverse()
        program["source_timestamp"] = "synthetic-later-timestamp"
        for source in program["examples"]:
            source["start_turn"] += 2
            source["pre_action_history"] = [f"aux_{source['group_id']}"]
            source["timestamp"] = "synthetic-later-timestamp"
        for step in program["steps"]:
            for variant in ("positive_guard", "refined_guard"):
                guard = step[variant]
                guard["objective"] = f"Updated {variant} diagnostic summary"
                guard["counts"]["candidate_clause_count"] += 5
                for clause in guard["clauses"]:
                    clause["selection"]["objective_gain"] += 7
                    clause["source_timestamp"] = "synthetic-later-timestamp"
    assert changed != library
    assert contracts.library_behavior_hash(changed) == contracts.library_behavior_hash(library)
    first, second = controller(library, policy, mode), controller(changed, policy, mode)
    for before, after in (("x", "alpha"), ("alpha", "beta")):
        left, right = first.decide(public(before)), second.decide(public(before))
        assert left == right
        assert first.observe(left["proposal"]["action"], after, compiled=True) == second.observe(
            right["proposal"]["action"], after, compiled=True,
        )
    assert first.decide(public()) == second.decide(public())


def test_behavior_hash_ignores_rejected_candidates_and_unused_fit_inputs():
    programs, demos, transitions = bundle()
    original = contracts.compile_library(programs, demos, transitions, [])
    extra, extra_demos, extra_transitions = bundle(name="unused", groups=("one-group",))
    refitted = contracts.compile_library(
        programs + extra, demos + extra_demos, {**transitions, **extra_transitions},
        [negative("unrelated", action="not a retained action")],
    )
    assert refitted["counts"] != original["counts"]
    assert refitted["rejected_programs"]
    assert contracts.library_behavior_hash(refitted) == contracts.library_behavior_hash(original)


def test_behavior_hash_canonicalizes_program_example_and_mapping_order():
    programs, demos, transitions = bundle(
        groups=("g0", "g1", "g2", "g3"),
        task_family=lambda group, _: "a" if group in ("g0", "g1") else "b",
    )
    library = contracts.compile_library(programs, demos, transitions, [], excluded_groups={"z", "y"})
    changed = deepcopy(library)
    changed["programs"].reverse()
    changed["fit_groups"].reverse()
    changed["parameters"]["excluded_groups"].reverse()
    changed["entry_index"] = dict(reversed(list(changed["entry_index"].items())))
    for program in changed["programs"]:
        program["examples"].reverse()
    changed = dict(reversed(list(changed.items())))
    assert changed != library
    assert contracts.library_behavior_hash(changed) == contracts.library_behavior_hash(library)


@pytest.mark.parametrize("kwargs", [
    {"name": "other-program"}, {"length": 3}, {"prefixes": 2}, {"groups": ("g0", "g2")},
    {"task_family": "other-family"}, {"task_family": None},
    {"task": "alpha beta changed source"}, {"task": "act alpha beta"},
    {"pre_action_history": ["prepare sigma"]}, {"minimum_groups": 1},
    {"max_literals": 2}, {"excluded_groups": {"other-excluded-group"}},
])
def test_behavior_hash_changes_with_instances_entry_bindings_requirements_and_config(kwargs):
    original, _, _ = compile_bundle(task_family="opaque")
    changed, _, _ = compile_bundle(**{"task_family": "opaque", **kwargs})
    assert contracts.library_behavior_hash(changed) != contracts.library_behavior_hash(original)


def test_behavior_hash_covers_exact_goal_text():
    original, _, _ = compile_bundle(task="alpha beta first")
    changed, _, _ = compile_bundle(task="alpha beta second")
    assert contracts.library_behavior_hash(original) != contracts.library_behavior_hash(changed)


def test_behavior_hash_covers_literal_action_sequence():
    programs, demos, transitions = bundle()
    for demo in demos:
        demo["public_state"]["actions"].append("alpha act")
    original = contracts.compile_library(programs, demos, transitions, [])
    programs[0]["actions"][0] = "alpha act"
    for demo in demos:
        if demo["action"] == "act alpha":
            demo["action"] = "alpha act"
    for transition in transitions.values():
        transition["pre_action_history"] = [
            "alpha act" if action == "act alpha" else action
            for action in transition["pre_action_history"]
        ]
    changed = contracts.compile_library(programs, demos, transitions, [])
    assert contracts.library_behavior_hash(changed) != contracts.library_behavior_hash(original)


def test_behavior_hash_covers_runtime_source_state_binding():
    library, _, _ = compile_bundle()
    changed = deepcopy(library)
    changed["programs"][0]["examples"][0]["start_state_sha256"] = "0" * 64
    assert contracts.library_behavior_hash(changed) != contracts.library_behavior_hash(library)


def test_behavior_hash_covers_fitted_groups_that_reject_target_construction():
    original, policy, _ = compile_bundle()
    changed, _, _ = compile_bundle(negatives=[negative("alpha", group="new-fit-group")])
    for variant in ("positive_guard", "refined_guard"):
        assert literals(original["programs"][0]["steps"][0][variant]) == literals(
            changed["programs"][0]["steps"][0][variant],
        )
    assert contracts.library_behavior_hash(changed) != contracts.library_behavior_hash(original)
    contracts.FragmentController(original, policy, group_id="new-fit-group", mode="positive_contract")
    with pytest.raises(ValueError, match="recompile"):
        contracts.FragmentController(
            changed, policy, group_id="new-fit-group", mode="positive_contract",
        )


@pytest.mark.parametrize("variant", ["positive_guard", "refined_guard"])
@pytest.mark.parametrize("change", ["sign", "tokens", "arity"])
def test_behavior_hash_covers_each_selected_guard_literal(variant, change):
    library, _, _ = compile_bundle(max_literals=2)
    changed = deepcopy(library)
    atoms = changed["programs"][0]["steps"][0][variant]["clauses"][0]["literals"]
    if change == "sign":
        atoms[0]["sign"] = "-"
    elif change == "tokens":
        atoms[0]["tokens"] = ["changed", "bigram"]
    else:
        atoms.append({"sign": "-", "tokens": ["extra"]})
    assert contracts.library_behavior_hash(changed) != contracts.library_behavior_hash(library)


def test_behavior_hash_preserves_bigram_token_order():
    library, _, _ = compile_bundle(observations=["a b", "b a", "a b"])
    changed = deepcopy(library)
    changed["programs"][0]["steps"][0]["positive_guard"]["clauses"][0]["literals"][0][
        "tokens"
    ].reverse()
    assert contracts.library_behavior_hash(changed) != contracts.library_behavior_hash(library)


def test_behavior_hash_canonicalizes_conjunction_order_but_preserves_logged_clause_indices():
    library, policy, _ = compile_bundle(
        negatives=[negative("x alpha", group="n1"), negative("", group="n2")], max_literals=2,
    )
    changed = deepcopy(library)
    atoms = changed["programs"][0]["steps"][0]["refined_guard"]["clauses"][0]["literals"]
    assert len(atoms) == 2
    atoms.reverse()
    assert contracts.library_behavior_hash(changed) == contracts.library_behavior_hash(library)
    assert controller(library, policy, "refined_contract").decide(public()) == controller(
        changed, policy, "refined_contract",
    ).decide(public())

    library, policy, _ = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        observations=lambda group, _: ["", "alpha", "final"] if group in ("g0", "g1")
        else ["beta", "", "final"],
    )
    changed = deepcopy(library)
    changed["programs"][0]["steps"][0]["positive_guard"]["clauses"].reverse()
    assert contracts.library_behavior_hash(changed) != contracts.library_behavior_hash(library)
    assert controller(library, policy).decide(public(""))["decision"][
        "matched_precondition_clauses"
    ] == [0]
    assert controller(changed, policy).decide(public(""))["decision"][
        "matched_precondition_clauses"
    ] == [1]


@pytest.mark.parametrize("mutation", [
    lambda lib: lib.update(schema_version="unknown"),
    lambda lib: lib["parameters"].update(minimum_groups=True),
    lambda lib: lib.update(entry_index={}),
    lambda lib: lib["programs"][0].update(instance_id="unbound"),
    lambda lib: lib["programs"][0].update(history_prerequisite_words=["unsupported"]),
    lambda lib: lib["objectives"].update(positive=None),
    lambda lib: lib["programs"][0]["steps"][0]["refined_guard"].update(objective=False),
    lambda lib: lib["counts"].update(input_programs=True),
])
def test_behavior_hash_rejects_invalid_libraries_without_mutation(mutation):
    library, _, _ = compile_bundle()
    mutation(library)
    before = deepcopy(library)
    with pytest.raises(ValueError):
        contracts.library_behavior_hash(library)
    assert library == before


@pytest.mark.parametrize("library", [None, True, [], {}])
def test_behavior_hash_rejects_invalid_top_level_types(library):
    with pytest.raises(ValueError):
        contracts.library_behavior_hash(library)


def test_exact_library_program_and_guard_public_keys():
    library, _, _ = compile_bundle(max_literals=2, task_family="opaque")
    assert set(library) == {
        "schema_version", "semantics", "parameters", "objectives", "programs", "entry_index",
        "fit_groups", "net_fit_groups", "counts", "rejected_programs",
    }
    assert set(library["parameters"]) == {
        "minimum_groups", "max_literals", "max_clauses", "conjunction_atom_limit", "excluded_groups",
        "negative_kind", "task_conditioning", "history_prerequisites", "negative_conditioning",
        "negative_program_kind", "negative_program_conditioning",
        "net_working_state", "composed_reentry",
    }
    program = library["programs"][0]
    assert set(program) == {
        "program_id", "instance_id", "conditioning", "task_family", "task_text",
        "history_prerequisite_words", "actions", "source_groups", "source_tasks", "examples",
        "steps", "counts", "positive_net_guard", "refined_net_guard", "reentry_projection",
    }
    assert set(program["examples"][0]) == {
        "task_id", "group_id", "start_turn", "task_family", "task_text", "demonstration_ids",
        "goal_words", "pre_action_history", "start_state_sha256",
    }
    assert set(program["steps"][0]) == {"step", "action", "positive_guard", "refined_guard"}
    for variant in ("positive_guard", "refined_guard"):
        guard = program["steps"][0][variant]
        assert set(guard) == {
            "objective", "clauses", "positive_groups", "negative_groups",
            "covered_positive_groups", "accepted_negative_groups", "counts",
        }
        clause = guard["clauses"][0]
        assert set(clause) == {
            "literals", "positive_groups", "positive_group_count", "positive_demonstration_ids",
            "positive_transition_count", "negative_precondition_groups",
            "negative_precondition_group_count", "negative_precondition_count", "selection",
        }
        assert set(clause["literals"][0]) == {"sign", "tokens"}
        assert set(clause["selection"]) == {
            "new_positive_group_count", "new_negative_precondition_group_count", "objective_gain",
        }


_DECISION_KEYS = {
    "commit", "reason", "mode", "guard_variant", "precondition_checked",
    "matched_precondition_clauses", "memory_enabled", "task_conditioning",
    "history_prerequisites_enabled", "entry_prerequisites_checked", "completed_action_count",
    "native_action_count", "task_family", "state_sha256", "action_selection",
    "net_guard_variant", "net_precondition_checked", "matched_net_precondition_clauses",
}
_ACTIVE_PROPOSAL_KEYS = {
    "action", "program_id", "instance_id", "step", "conditioning", "task_family",
    "planned_burst_length", "stored_action", "step_source_demonstration_id",
    "source_groups", "entry_source",
}
_OBSERVE_COMMON_KEYS = {
    "action", "observation", "native_valid", "history_recorded", "compiled", "reason",
    "postcondition_checked", "postcondition_passed", "fragment_complete", "fragment_aborted",
    "actor_required", "seen_entry_count", "completed_action_count", "native_action_count",
    "net_guard_variant", "net_postcondition_checked", "net_postcondition_passed",
    "matched_net_precondition_clauses", "matched_net_postcondition_clauses",
}
_OBSERVE_COMPILED_KEYS = _OBSERVE_COMMON_KEYS | {
    "program_id", "instance_id", "step", "planned_burst_length", "executed_steps", "next_step",
    "guard_variant", "matched_precondition_clauses", "matched_postcondition_clauses",
}


@pytest.mark.parametrize("mode", contracts.MODES)
def test_exact_decide_and_observe_keys_for_commits_and_actor_boundaries(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode)
    initial = control.observe("initial actor", "x", compiled=False)
    assert set(initial) == _OBSERVE_COMMON_KEYS | {"cleared_boundary"}
    for before, after in (("x", "alpha"), ("alpha", "beta")):
        result = control.decide(public(before))
        assert set(result) == {"proposal", "decision"}
        assert result["decision"]["commit"]
        assert set(result["decision"]) == _DECISION_KEYS | {"entry_rejections", "entry_guards_used"}
        assert set(result["proposal"]) == _ACTIVE_PROPOSAL_KEYS | (
            {"independent_proposal"} if mode == "independent_actions" else set()
        )
        assert set(result["proposal"]["entry_source"]) == {
            "example_id", "group_id", "task_id", "similarity", "goal_words", "task_text",
            "task_family", "history_prerequisite_words",
        }
        if mode == "independent_actions":
            assert set(result["proposal"]["independent_proposal"]) == {
                "action", "confidence", "margin", "support_groups", "source_example_ids", "reason",
            }
        feedback = control.observe(result["proposal"]["action"], after, compiled=True)
        assert set(feedback) == _OBSERVE_COMPILED_KEYS
    boundary = control.decide(public())
    assert boundary["proposal"] == {"action": None}
    assert set(boundary["decision"]) == _DECISION_KEYS | {"boundary_reason"}
    actor = control.observe("non-native actor", "actual feedback", compiled=False, native_valid=False)
    assert set(actor) == _OBSERVE_COMMON_KEYS | {"cleared_boundary"}
    if mode in _STEP_ONLY_MODES and mode != "refined_no_memory":
        repeated = control.decide(public())
        assert repeated["proposal"] == {"action": None}
        assert set(repeated["decision"]) == _DECISION_KEYS


@pytest.mark.parametrize("mode", contracts.MODES)
def test_exact_rejection_and_non_native_feedback_keys(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode)
    denied = control.decide(public(actions=["not a fragment action"]))
    assert denied["proposal"] == {"action": None}
    assert set(denied["decision"]) == _DECISION_KEYS | {"entry_rejections", "entry_guards_used"}
    assert set(denied["decision"]["entry_rejections"][0]) == {
        "program_id", "instance_id", "task_family", "source_example_id", "reason",
        "missing_history_prerequisite_words",
    }
    control.observe("actor", "x", compiled=False)
    accepted = control.decide(public())
    feedback = control.observe(
        accepted["proposal"]["action"], "alpha", compiled=True, native_valid=False,
    )
    assert set(feedback) == _OBSERVE_COMPILED_KEYS
    assert feedback["fragment_aborted"]


@pytest.mark.parametrize("mode", ["positive_contract", "refined_contract", "refined_no_memory"])
def test_exact_precondition_abort_keys(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode)
    result = control.decide(public())
    control.observe(result["proposal"]["action"], "alpha", compiled=True)
    failed = control.decide(public("beta"))
    assert not failed["decision"]["commit"]
    assert set(failed["decision"]) == _DECISION_KEYS
    assert set(failed["proposal"]) == _ACTIVE_PROPOSAL_KEYS
    assert failed["proposal"]["action"] is None


_STEP_ONLY_MODES = (
    "independent_actions", "ordered_fragment", "positive_contract",
    "refined_contract", "refined_no_memory",
)
_COMPOSED_MODES = ("composed_contract", "refined_composed")
_COMPOSED_ABLATIONS = (
    "composed_no_net_checks", "composed_raw_reentry", "composed_no_reentry_memory",
)
_COMPOSED_SOURCE_MODES = (*_COMPOSED_MODES, *_COMPOSED_ABLATIONS)
_PROJECTED_MEMORY_MODES = (*_COMPOSED_MODES, "composed_no_net_checks")
_KNOWN_ACTIONS = ("act alpha", "act beta")


def test_net_guard_is_mined_from_endpoints_not_conjoined_step_guards():
    library, _, _ = compile_bundle(
        observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    program = library["programs"][0]
    assert literals(program["steps"][0]["positive_guard"]) == [(("+", ("alpha",)),)]
    assert literals(program["steps"][1]["positive_guard"]) == [(("-", ("alpha",)),)]
    assert literals(program["positive_net_guard"]) == [(("+", ("beta",)),)]
    assert literals(program["refined_net_guard"]) == literals(program["positive_net_guard"])
    clause = program["positive_net_guard"]["clauses"][0]
    assert clause["positive_demonstration_ids"] == sorted(
        source["demonstration_ids"][0] for source in program["examples"]
    )
    assert clause["positive_group_count"] == 2
    assert library["counts"]["step_guard_pairs_equal"] == 2
    assert library["counts"]["net_guard_pairs_equal"] == 1
    assert library["counts"]["net_guard_pairs_different"] == 0


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
def test_net_cycle_fails_after_both_step_witnesses_pass_without_undoing_actions(monkeypatch, mode):
    tail = [
        {"action": "act alpha", "observation": "alpha"},
        {"action": "act beta", "observation": "x"},
    ]
    entry = public("x", history=tail)
    cycle = cycle_negative(pre_state=entry, post_state=deepcopy(entry))
    assert cycle["pre_state_sha256"] == cycle["post_state_sha256"]
    library, policy, demos = compile_bundle(
        observations=["x", "alpha beta", "beta"], negative_programs=[cycle],
        pre_action_history=_KNOWN_ACTIONS,
    )
    assert library["counts"]["usable_negative_programs"] == 1
    assert library["counts"]["net_guard_pairs_equal"] == 1
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS)
    first = control.decide(entry)
    assert first["decision"]["commit"]
    assert first["decision"]["net_precondition_checked"]
    assert first["decision"]["matched_net_precondition_clauses"] == [0]
    observed = control.observe(first["proposal"]["action"], "alpha", compiled=True)
    assert observed["postcondition_passed"] is True
    assert observed["net_postcondition_checked"] is False
    assert observed["next_step"] == 1
    second = control.decide(public("alpha"))
    assert second["proposal"]["step"] == 1
    assert not second["decision"]["net_precondition_checked"]
    observed = control.observe(second["proposal"]["action"], "x", compiled=True)
    assert observed["postcondition_passed"] is True
    assert observed["net_postcondition_checked"]
    assert observed["net_postcondition_passed"] is False
    assert observed["matched_net_postcondition_clauses"] == []
    assert observed["reason"] == "net_postcondition_failed"
    assert observed["fragment_aborted"] and not observed["fragment_complete"]
    assert observed["executed_steps"] == 2
    assert observed["native_action_count"] == 4
    assert observed["observation"] == entry["observation"]
    assert observed["next_step"] is None
    deferred = control.decide(entry)
    assert deferred["decision"]["boundary_reason"] == "net_postcondition_failed"
    control.observe("actor response", "x", compiled=False)
    repeated = control.decide(entry)
    assert repeated["decision"]["reason"] == "repeated_fragment_entry_state"
    assert repeated["decision"]["native_action_count"] == 5


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
def test_net_precondition_is_checked_only_at_entry_and_success_requires_final_effect(mode):
    library, policy, _ = compile_bundle(observations=["x", "alpha beta", "beta"])
    control = controller(library, policy, mode)
    first = control.decide(public("x"))
    control.observe(first["proposal"]["action"], "alpha beta", compiled=True)
    second = control.decide(public("alpha beta"))
    assert second["decision"]["commit"]
    assert not second["decision"]["net_precondition_checked"]
    assert second["decision"]["matched_net_precondition_clauses"] == [0]
    feedback = control.observe(second["proposal"]["action"], "beta", compiled=True)
    assert feedback["postcondition_passed"] is True
    assert feedback["net_postcondition_passed"] is True
    assert feedback["matched_net_postcondition_clauses"] == [0]
    assert feedback["fragment_complete"] and feedback["actor_required"]
    assert control.decide(public("x"))["decision"]["reason"] == "actor_boundary_required"


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
def test_net_precondition_rejection_defers_before_any_local_execution(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS)
    denied = control.decide(public("x beta"))
    assert denied["proposal"]["action"] is None
    assert not denied["decision"]["commit"]
    assert denied["decision"]["net_precondition_checked"]
    assert denied["decision"]["entry_rejections"][0]["reason"] == "entry_net_precondition_failed"
    assert denied["decision"]["completed_action_count"] == 2
    control.observe("actor", "x", compiled=False)
    assert control.decide(public("x"))["proposal"]["step"] == 0


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
def test_net_postcondition_cannot_switch_to_an_unmatched_entry_clause(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        observations=lambda group, _: ["", "alpha", "beta"] if group in ("g0", "g1")
        else ["gamma", "alpha gamma", ""],
        pre_action_history=_KNOWN_ACTIONS,
    )
    net = library["programs"][0]["positive_net_guard"]
    assert literals(net) == [(("+", ("beta",)),), (("-", ("gamma",)),)]
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS)
    first = control.decide(public(""))
    assert first["decision"]["matched_net_precondition_clauses"] == [0]
    control.observe(first["proposal"]["action"], "alpha", compiled=True)
    second = control.decide(public("alpha"))
    feedback = control.observe(second["proposal"]["action"], "", compiled=True)
    assert feedback["postcondition_passed"] is True
    assert feedback["net_postcondition_passed"] is False
    assert feedback["matched_net_postcondition_clauses"] == []


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
def test_non_native_final_action_cannot_receive_net_effect_credit(mode):
    library, policy, _ = compile_bundle(observations=["x", "alpha beta", "beta"])
    control = controller(library, policy, mode)
    first = control.decide(public("x"))
    control.observe(first["proposal"]["action"], "alpha beta", compiled=True)
    second = control.decide(public("alpha beta"))
    feedback = control.observe(second["proposal"]["action"], "beta", compiled=True, native_valid=False)
    assert feedback["reason"] == "compiled_action_not_native"
    assert feedback["net_postcondition_checked"] is False
    assert feedback["net_postcondition_passed"] is None
    assert feedback["fragment_aborted"] and not feedback["fragment_complete"]
    assert feedback["native_action_count"] == 1


def test_no_net_witness_retains_the_step_only_pool_and_needs_independent_groups():
    library, policy, _ = compile_bundle(
        observations=["x", "alpha", "x"], pre_action_history=_KNOWN_ACTIONS,
    )
    assert len(library["programs"]) == 1
    assert library["programs"][0]["positive_net_guard"] is None
    assert library["programs"][0]["refined_net_guard"] is None
    assert library["counts"]["instances_without_net_guard"] == 1
    assert library["net_fit_groups"] == []
    for mode in _STEP_ONLY_MODES:
        control = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS)
        for before, after in (("x", "alpha"), ("alpha", "x")):
            result = control.decide(public(before))
            assert result["decision"]["commit"]
            feedback = control.observe(result["proposal"]["action"], after, compiled=True)
        assert feedback["fragment_complete"]
        assert feedback["net_postcondition_passed"] is None
    for mode in _COMPOSED_SOURCE_MODES:
        denied = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS).decide(
            public("x"),
        )
        assert not denied["decision"]["commit"]
        assert denied["decision"]["entry_rejections"][0]["reason"] == "no_supported_net_guard"
    library, _, _ = compile_bundle(
        prefixes=5, observations=lambda group, _: ["x", "alpha beta", "beta"] if group == "g0"
        else ["x", "alpha beta", "x"],
        pre_action_history=_KNOWN_ACTIONS,
    )
    assert len(library["programs"]) == 1
    assert library["programs"][0]["positive_net_guard"] is None


@pytest.mark.parametrize("mode", _STEP_ONLY_MODES)
def test_cycle_negatives_change_only_net_fitting_not_step_only_behavior(mode):
    base, policy, _ = compile_bundle(
        observations=["x y", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    changed, _, _ = compile_bundle(
        observations=["x y", "alpha beta", "beta"],
        pre_action_history=_KNOWN_ACTIONS,
        negative_programs=[cycle_negative("x", group="net-only-group")],
    )
    assert base["programs"][0]["steps"] == changed["programs"][0]["steps"]
    assert base["fit_groups"] == changed["fit_groups"]
    assert literals(base["programs"][0]["refined_net_guard"]) != literals(
        changed["programs"][0]["refined_net_guard"],
    )
    assert "net-only-group" in changed["net_fit_groups"]
    assert contracts.runtime_contract_hash(base, mode) == contracts.runtime_contract_hash(changed, mode)
    first, second = controller(base, policy, mode), controller(changed, policy, mode)
    prime_native_history(first, _KNOWN_ACTIONS)
    prime_native_history(second, _KNOWN_ACTIONS)
    for before, after in (("x y", "alpha beta"), ("alpha beta", "beta")):
        left, right = first.decide(public(before)), second.decide(public(before))
        assert left == right
        assert first.observe(left["proposal"]["action"], after, compiled=True) == second.observe(
            right["proposal"]["action"], after, compiled=True,
        )
    assert first.decide(public()) == second.decide(public())
    contracts.FragmentController(changed, policy, mode=mode, group_id="net-only-group")
    with pytest.raises(ValueError, match="recompile"):
        contracts.FragmentController(
            changed, policy, mode="composed_contract", group_id="net-only-group",
        )


def test_step_stutters_never_supply_net_counterexamples():
    base, _, _ = compile_bundle(observations=["x", "alpha beta", "beta"])
    changed, _, _ = compile_bundle(
        observations=["x", "alpha beta", "beta"], negatives=[negative("")],
    )
    assert literals(base["programs"][0]["steps"][0]["refined_guard"]) != literals(
        changed["programs"][0]["steps"][0]["refined_guard"],
    )
    assert base["programs"][0]["positive_net_guard"] == changed["programs"][0]["positive_net_guard"]
    assert base["programs"][0]["refined_net_guard"] == changed["programs"][0]["refined_net_guard"]


def test_cycle_negatives_require_original_hash_equality_and_exact_sequence_and_family():
    cycle = cycle_negative("x", task_family="opaque")
    changed_state = public("x", actions=["act alpha", "act beta", "extra"])
    different_hashes = cycle_negative("x", task_family="opaque", post_state=changed_state)
    assert different_hashes["observation"] == different_hashes["post_observation"]
    assert different_hashes["pre_state_sha256"] != different_hashes["post_state_sha256"]
    same_group_other_state = cycle_negative(
        task_family="opaque", pre_state=changed_state, post_state=deepcopy(changed_state),
    )
    library, _, _ = compile_bundle(
        task_family="opaque", observations=["x y", "alpha beta", "beta"],
        pre_action_history=_KNOWN_ACTIONS,
        negative_programs=[
            cycle, deepcopy(cycle), different_hashes, same_group_other_state,
            cycle_negative("x", task_family="different-family"),
            cycle_negative("x", task_family="opaque", actions=["act beta", "act alpha"]),
        ],
    )
    assert library["counts"]["duplicate_negative_programs"] == 1
    assert library["counts"]["ignored_non_cycle_negative_programs"] == 1
    assert library["counts"]["usable_negative_programs"] == 4
    assert library["counts"]["negative_programs_unused_by_retained_net_guards"] == 2
    program = library["programs"][0]
    assert program["positive_net_guard"]["clauses"][0]["negative_precondition_count"] == 2
    assert program["positive_net_guard"]["clauses"][0]["negative_precondition_group_count"] == 1
    assert literals(program["positive_net_guard"]) == [(("+", ("beta",)),)]
    assert literals(program["refined_net_guard"]) == [(("-", ("y",)),)]
    assert library["counts"]["step_guard_pairs_equal"] == 2
    assert library["counts"]["step_guard_pairs_different"] == 0
    assert library["counts"]["net_guard_pairs_different"] == 1


def test_target_cycle_groups_are_excluded_before_net_refinement():
    library, policy, _ = compile_bundle(
        task_family="opaque", observations=["x y", "alpha beta", "beta"],
        pre_action_history=_KNOWN_ACTIONS,
        negative_programs=[cycle_negative("x", group="target", task_family="opaque")],
        excluded_groups={"target"},
    )
    assert library["counts"]["excluded_negative_programs"] == 1
    assert library["counts"]["usable_negative_programs"] == 0
    assert "target" not in library["net_fit_groups"]
    assert library["counts"]["net_guard_pairs_equal"] == 1
    control = prime_native_history(
        controller(library, policy, "refined_composed", task_family="opaque"), _KNOWN_ACTIONS,
    )
    assert control.decide(
        public("x y"),
    )["decision"]["commit"]


@pytest.mark.parametrize("mutation", [
    lambda row: row.pop("pre_state_sha256"),
    lambda row: row.update(pre_state_sha256=True),
    lambda row: row.update(post_state_sha256="not an original state hash"),
    lambda row: row.update(post_observation="contradicts the equal state hashes"),
    lambda row: row.update(actions=True),
    lambda row: row.update(actions=["one"]),
    lambda row: row.update(actions=["one"] * 5),
    lambda row: row.update(actions=[True, "two"]),
    lambda row: row.update(group_id=True),
    lambda row: row.update(task_family=True),
])
def test_invalid_or_unverified_cycle_negative_schema_is_rejected(mutation):
    record = cycle_negative()
    mutation(record)
    with pytest.raises(ValueError):
        compile_bundle(negative_programs=[record])


@pytest.mark.parametrize("records", [True, {}, (), "cycles"])
def test_negative_programs_must_be_an_explicit_record_list(records):
    with pytest.raises(ValueError, match="negative_programs"):
        compile_bundle(negative_programs=records)


def test_runtime_hash_aliases_equal_guards_without_mode_or_logging_labels():
    library, policy, _ = compile_bundle(observations=["x", "alpha beta", "beta"])
    for left, right in (
        ("positive_contract", "refined_contract"), ("composed_contract", "refined_composed"),
    ):
        digest = contracts.runtime_contract_hash(library, left)
        assert digest == contracts.runtime_contract_hash(library, right)
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        first, second = controller(library, policy, left), controller(library, policy, right)
        for before, after in (("x", "alpha beta"), ("alpha beta", "beta")):
            one, two = first.decide(public(before)), second.decide(public(before))
            assert one["proposal"] == two["proposal"]
            assert one["decision"]["mode"] != two["decision"]["mode"]
            assert one["decision"]["commit"] == two["decision"]["commit"]
            seen_one = first.observe(one["proposal"]["action"], after, compiled=True)
            seen_two = second.observe(two["proposal"]["action"], after, compiled=True)
            for field in ("fragment_complete", "fragment_aborted", "next_step", "actor_required"):
                assert seen_one[field] == seen_two[field]
        assert first.decide(public())["decision"]["commit"] is False
        assert second.decide(public())["decision"]["commit"] is False


@pytest.mark.parametrize("left,right", [
    ("independent_actions", "ordered_fragment"),
    ("ordered_fragment", "positive_contract"),
    ("refined_contract", "refined_no_memory"),
    ("positive_contract", "composed_contract"),
])
def test_runtime_hash_distinguishes_action_selection_memory_and_active_effects(left, right):
    library, _, _ = compile_bundle()
    assert contracts.runtime_contract_hash(library, left) != contracts.runtime_contract_hash(
        library, right,
    )


def test_runtime_hash_selects_only_active_guards_and_ignores_fit_diagnostics():
    library, _, _ = compile_bundle(
        observations=["x y", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    changed, _, _ = compile_bundle(
        observations=["x y", "alpha beta", "beta"],
        pre_action_history=_KNOWN_ACTIONS,
        negative_programs=[cycle_negative("x")],
    )
    assert contracts.runtime_contract_hash(changed, "positive_contract") == contracts.runtime_contract_hash(
        changed, "refined_contract",
    )
    assert contracts.runtime_contract_hash(changed, "composed_contract") != contracts.runtime_contract_hash(
        changed, "refined_composed",
    )
    assert contracts.runtime_contract_hash(library, "positive_contract") == contracts.runtime_contract_hash(
        changed, "positive_contract",
    )
    assert contracts.library_behavior_hash(library) != contracts.library_behavior_hash(changed)
    metadata = deepcopy(library)
    metadata["parameters"]["max_literals"] = 2
    metadata["parameters"]["minimum_groups"] = 1
    metadata["counts"]["step_guard_pairs_different"] += 4
    metadata["objectives"]["positive"] = "Different diagnostic summary"
    for mode in contracts.MODES:
        assert contracts.runtime_contract_hash(metadata, mode) == contracts.runtime_contract_hash(
            library, mode,
        )


def test_runtime_hash_ignores_clause_indices_but_preserves_joint_literal_clauses():
    library, _, _ = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        observations=lambda group, _: ["", "alpha", "beta"] if group in ("g0", "g1")
        else ["gamma", "alpha gamma", ""],
        pre_action_history=_KNOWN_ACTIONS,
    )
    changed = deepcopy(library)
    changed["programs"][0]["refined_net_guard"]["clauses"].reverse()
    assert contracts.library_behavior_hash(changed) != contracts.library_behavior_hash(library)
    assert contracts.runtime_contract_hash(changed, "composed_contract") == contracts.runtime_contract_hash(
        changed, "refined_composed",
    )
    changed["programs"][0]["refined_net_guard"]["clauses"][0]["literals"][0]["tokens"] = ["new"]
    assert contracts.runtime_contract_hash(changed, "composed_contract") != contracts.runtime_contract_hash(
        changed, "refined_composed",
    )


def test_runtime_hash_omits_unexecutable_net_instances():
    library, _, _ = compile_bundle(
        observations=["x", "alpha", "x"], negatives=[negative("")],
        pre_action_history=_KNOWN_ACTIONS,
    )
    assert library["programs"][0]["positive_net_guard"] is None
    assert contracts.runtime_contract_hash(library, "positive_contract") != contracts.runtime_contract_hash(
        library, "refined_contract",
    )
    assert contracts.runtime_contract_hash(library, "composed_contract") == contracts.runtime_contract_hash(
        library, "refined_composed",
    )


@pytest.mark.parametrize("mode", [True, None, "unknown", []])
def test_runtime_hash_rejects_invalid_modes(mode):
    library, _, _ = compile_bundle()
    with pytest.raises(ValueError, match="mode"):
        contracts.runtime_contract_hash(library, mode)


@pytest.mark.parametrize("mutation", [
    lambda lib: lib.update(schema_version="chisel-observable-fragment-contracts/v3"),
    lambda lib: lib.update(net_fit_groups=["unaccounted"]),
    lambda lib: lib["programs"][0].pop("positive_net_guard"),
    lambda lib: lib["programs"][0].update(positive_net_guard=None),
    lambda lib: lib["programs"][0]["positive_net_guard"].update(clauses=[]),
    lambda lib: lib["programs"][0]["refined_net_guard"]["clauses"][0].update(
        positive_group_count=True,
    ),
    lambda lib: lib["parameters"].update(negative_program_kind="observation_only_stutter"),
])
def test_new_net_schema_is_validated_by_controller_and_both_hashes(mutation):
    library, policy, _ = compile_bundle()
    mutation(library)
    with pytest.raises(ValueError):
        controller(library, policy, "composed_contract")
    with pytest.raises(ValueError):
        contracts.library_behavior_hash(library)
    with pytest.raises(ValueError):
        contracts.runtime_contract_hash(library, "composed_contract")


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
@pytest.mark.parametrize("length", [3, 4])
def test_longer_net_contracts_check_only_the_actual_last_transition(mode, length):
    library, policy, _ = compile_bundle(length=length)
    control = controller(library, policy, mode)
    actions = library["programs"][0]["actions"]
    observations = ["x", "alpha", "beta", "gamma", "delta"]
    for step in range(length):
        result = control.decide(public(observations[step], actions=actions))
        assert result["decision"]["commit"]
        assert result["proposal"]["step"] == step
        feedback = control.observe(result["proposal"]["action"], observations[step + 1], compiled=True)
        assert feedback["net_postcondition_checked"] is (step == length - 1)
        assert feedback["net_postcondition_passed"] is (True if step == length - 1 else None)
        assert feedback["fragment_complete"] is (step == length - 1)


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
@pytest.mark.parametrize("missing_guard", [True, False])
def test_net_entry_can_scan_to_the_next_supported_candidate(monkeypatch, mode, missing_guard):
    first, first_demos, first_transitions = bundle(
        name="first", observations=["x", "alpha", "x"] if missing_guard
        else ["x", "alpha beta", "beta"],
        pre_action_history=_KNOWN_ACTIONS,
    )
    second, second_demos, second_transitions = bundle(
        name="second", observations=["x beta", "alpha gamma", "gamma"],
        pre_action_history=_KNOWN_ACTIONS,
    )
    library = contracts.compile_library(
        first + second, first_demos + second_demos, {**first_transitions, **second_transitions}, [],
    )
    policy = invocation.build_policy(first_demos + second_demos)
    fixed_neighbors(monkeypatch, [(1.0, first_demos[0]), (0.8, second_demos[0])])
    result = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS).decide(
        public("x beta"),
    )
    assert result["decision"]["commit"]
    assert result["proposal"]["program_id"] == "second"
    assert result["decision"]["entry_rejections"][0]["reason"] == (
        "no_supported_net_guard" if missing_guard else "entry_net_precondition_failed"
    )


def test_net_entry_clause_cache_cannot_be_changed_through_returned_logging_lists():
    library, policy, _ = compile_bundle(observations=["x", "alpha beta", "beta"])
    control = controller(library, policy, "composed_contract")
    first = control.decide(public("x"))
    first["decision"]["matched_net_precondition_clauses"].clear()
    feedback = control.observe(first["proposal"]["action"], "alpha beta", compiled=True)
    assert feedback["matched_net_precondition_clauses"] == [0]
    feedback["matched_net_precondition_clauses"].append(999)
    second = control.decide(public("alpha beta"))
    assert second["decision"]["matched_net_precondition_clauses"] == [0]
    final = control.observe(second["proposal"]["action"], "beta", compiled=True)
    assert final["net_postcondition_passed"] is True


def test_reused_cycle_state_hash_cannot_claim_conflicting_observations():
    first = cycle_negative("x")
    second = {**first, "observation": "different", "post_observation": "different"}
    with pytest.raises(ValueError, match="different observations"):
        compile_bundle(negative_programs=[first, second])


def test_hashes_cover_both_net_variants_but_runtime_uses_only_the_active_one():
    library, _, _ = compile_bundle(pre_action_history=_KNOWN_ACTIONS)
    changed = deepcopy(library)
    changed["programs"][0]["positive_net_guard"]["clauses"][0]["literals"][0]["tokens"] = ["new"]
    assert contracts.library_behavior_hash(changed) != contracts.library_behavior_hash(library)
    for mode in (*_STEP_ONLY_MODES, "refined_composed"):
        assert contracts.runtime_contract_hash(changed, mode) == contracts.runtime_contract_hash(
            library, mode,
        )
    assert contracts.runtime_contract_hash(changed, "composed_contract") != contracts.runtime_contract_hash(
        library, "composed_contract",
    )


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
def test_same_observation_with_new_validated_action_history_is_net_progress(mode):
    library, policy, _ = compile_bundle(observations=["x", "alpha", "x"])
    program = library["programs"][0]
    assert program["positive_net_guard"]["clauses"][0]["literals"] == [
        {"field": "history_action", "sign": "+", "action": "act alpha"},
    ]
    assert program["positive_net_guard"]["clauses"][0]["positive_group_count"] == 2
    assert program["reentry_projection"]["positive"] == {
        "history_actions": ["act alpha"], "history_words": [],
    }
    control = controller(library, policy, mode)
    first = control.decide(public("x"))
    assert first["decision"]["commit"] and first["decision"]["net_precondition_checked"]
    control.observe(first["proposal"]["action"], "alpha", compiled=True)
    second = control.decide(public("alpha"))
    feedback = control.observe(second["proposal"]["action"], "x", compiled=True)
    assert feedback["postcondition_passed"] is True
    assert feedback["net_postcondition_passed"] is True
    assert feedback["fragment_complete"] and not feedback["fragment_aborted"]
    assert feedback["observation"] == "x"
    assert feedback["native_action_count"] == 2
    assert control.decide(public("x"))["decision"]["reason"] == "actor_boundary_required"


def test_information_gaining_observation_cycles_are_counted_and_not_fit_as_negatives():
    gaining = cycle_negative("x", pre_action_history=[])
    no_progress = cycle_negative("x", pre_action_history=list(_KNOWN_ACTIONS))
    assert gaining["pre_state_sha256"] == gaining["post_state_sha256"]
    library, _, _ = compile_bundle(
        observations=["x", "alpha", "x"], negative_programs=[gaining, no_progress],
    )
    assert library["counts"]["input_negative_programs"] == 2
    assert library["counts"]["ignored_information_gaining_observation_cycles"] == 1
    assert library["counts"]["usable_negative_programs"] == 1
    program = library["programs"][0]
    for variant in ("positive_net_guard", "refined_net_guard"):
        guard = program[variant]
        assert guard["counts"]["negative_transition_count"] == 1
        assert guard["clauses"][0]["negative_precondition_count"] == 0
    assert contracts.runtime_contract_hash(library, "composed_contract") == contracts.runtime_contract_hash(
        library, "refined_composed",
    )


def test_cycle_progress_uses_exact_action_sets_not_history_order_counts_or_tokens():
    unchanged = cycle_negative(
        "x", pre_action_history=["act alpha", "act beta"],
        post_action_history=["act beta", "act alpha", "act alpha"],
    )
    same_tokens_new_action = cycle_negative(
        "x", pre_action_history=["alpha act", "beta act"],
        post_action_history=["alpha act", "beta act", "act alpha", "act beta"],
    )
    library, _, _ = compile_bundle(negative_programs=[unchanged, same_tokens_new_action])
    assert library["counts"]["usable_negative_programs"] == 1
    assert library["counts"]["ignored_information_gaining_observation_cycles"] == 1


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
def test_net_history_facts_are_exact_native_strings_not_instruction_or_public_history_text(
    monkeypatch, mode,
):
    library, policy, demos = compile_bundle(task_family="opaque", observations=["x", "alpha", "x"])
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, mode, task_family="opaque")
    control.observe("act alpha", "instruction: act alpha", compiled=False, native_valid=False)
    control.observe("ACT alpha", "instruction: act alpha", compiled=False, native_valid=True)
    state = public(
        "x", task="instructions act alpha then act beta",
        history=[{"action": "act alpha", "observation": "instructions act alpha"}],
    )
    result = control.decide(state)
    assert result["decision"]["commit"]
    assert result["decision"]["matched_net_precondition_clauses"] == [0]
    assert result["decision"]["native_action_count"] == 1
    control.observe(result["proposal"]["action"], "alpha", compiled=True)
    second = control.decide({**state, "observation": "alpha"})
    feedback = control.observe(second["proposal"]["action"], "x", compiled=True)
    assert feedback["net_postcondition_passed"] is True


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
def test_composed_reentry_tracks_relevant_history_but_ignores_unrelated_history_churn(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        observations=lambda group, _: ["x", "alpha", "x"] if group in ("g0", "g1")
        else ["z", "alpha z", "final"],
        pre_action_history=lambda group, _: [] if group in ("g0", "g1") else list(_KNOWN_ACTIONS),
    )
    projection = library["programs"][0]["reentry_projection"]["positive"]
    assert projection == {"history_actions": ["act alpha"], "history_words": []}
    clauses = library["programs"][0]["positive_net_guard"]["clauses"]
    assert clauses[0]["literals"] == [
        {"field": "history_action", "sign": "+", "action": "act alpha"},
    ]
    assert clauses[1]["literals"] == [
        {"field": "observation", "sign": "+", "tokens": ["final"]},
    ]
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, mode)
    for iteration in range(2):
        entry = public("x", history=[] if iteration == 0 else [
            {"action": "unrelated one", "observation": "x"},
        ])
        first = control.decide(entry)
        assert first["decision"]["commit"]
        assert first["decision"]["matched_net_precondition_clauses"] == ([0, 1] if iteration == 0 else [1])
        control.observe(first["proposal"]["action"], "alpha", compiled=True)
        second = control.decide(public("alpha"))
        feedback = control.observe(second["proposal"]["action"], "x", compiled=True)
        assert feedback["net_postcondition_passed"] is (iteration == 0)
        assert feedback["fragment_complete"] is (iteration == 0)
        control.observe("unrelated one" if iteration == 0 else "unrelated two", "x", compiled=False)
    changed_raw_history = public("x", history=[
        {"action": "unrelated two", "observation": "x"},
    ])
    rejected = control.decide(changed_raw_history)
    assert not rejected["decision"]["commit"]
    assert rejected["decision"]["reason"] == "repeated_fragment_entry_state"
    assert rejected["decision"]["entry_rejections"][0]["reason"] == "repeated_fragment_entry_state"
    assert rejected["decision"]["native_action_count"] == 6


@pytest.mark.parametrize("mode", contracts.MODES)
def test_only_composed_modes_replace_raw_history_in_the_reentry_key(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    assert library["programs"][0]["reentry_projection"]["positive"] == {
        "history_actions": [], "history_words": ["act", "alpha", "beta"],
    }
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS)
    for before, after in (("x", "alpha"), ("alpha", "x")):
        result = control.decide(public(before))
        assert result["decision"]["commit"]
        control.observe(result["proposal"]["action"], after, compiled=True)
    control.observe("act unrelated", "x", compiled=False)
    state = public("x", actions=["act beta", "act alpha"], history=[
        {"action": "act unrelated", "observation": "x"},
    ])
    result = control.decide(state)
    assert result["decision"]["commit"] is (mode not in _PROJECTED_MEMORY_MODES)
    if mode in _PROJECTED_MEMORY_MODES:
        assert result["decision"]["reason"] == "repeated_fragment_entry_state"


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
def test_composed_reentry_memory_is_per_instance_not_a_global_state_ban(monkeypatch, mode):
    first, first_demos, first_transitions = bundle(
        name="first", observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    second, second_demos, second_transitions = bundle(
        name="second", observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    library = contracts.compile_library(
        first + second, first_demos + second_demos, {**first_transitions, **second_transitions}, [],
    )
    policy = invocation.build_policy(first_demos + second_demos)
    fixed_neighbors(monkeypatch, [(1.0, first_demos[0]), (0.9, second_demos[0])])
    control = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS)
    for before, after in (("x", "alpha beta"), ("alpha beta", "beta")):
        result = control.decide(public(before))
        assert result["proposal"]["program_id"] == "first"
        control.observe(result["proposal"]["action"], after, compiled=True)
    control.observe("irrelevant actor", "x", compiled=False)
    result = control.decide(public("x"))
    assert result["decision"]["commit"]
    assert result["proposal"]["program_id"] == "second"
    assert result["decision"]["entry_rejections"][0]["program_id"] == "first"
    assert result["decision"]["entry_rejections"][0]["reason"] == "repeated_fragment_entry_state"


def test_hashes_include_typed_history_atoms_and_active_reentry_projection():
    library, _, _ = compile_bundle(observations=["x", "alpha", "x"])
    changed = deepcopy(library)
    changed["programs"][0]["positive_net_guard"]["clauses"][0]["literals"][0]["action"] = "act beta"
    changed["programs"][0]["reentry_projection"]["positive"]["history_actions"] = ["act beta"]
    assert contracts.library_behavior_hash(changed) != contracts.library_behavior_hash(library)
    assert contracts.runtime_contract_hash(changed, "composed_contract") != contracts.runtime_contract_hash(
        library, "composed_contract",
    )
    for mode in (*_STEP_ONLY_MODES, "refined_composed"):
        assert contracts.runtime_contract_hash(changed, mode) == contracts.runtime_contract_hash(
            library, mode,
        )
    assert contracts.runtime_contract_hash(library, "composed_contract") == contracts.runtime_contract_hash(
        library, "refined_composed",
    )


@pytest.mark.parametrize("mutation", [
    lambda lib: lib["programs"][0]["positive_net_guard"]["clauses"][0]["literals"][0].pop("field"),
    lambda lib: lib["programs"][0]["positive_net_guard"]["clauses"][0]["literals"][0].update(
        field="instructions",
    ),
    lambda lib: lib["programs"][0]["positive_net_guard"]["clauses"][0]["literals"][0].update(
        action=True,
    ),
    lambda lib: lib["programs"][0]["positive_net_guard"]["clauses"][0]["literals"][0].update(
        tokens=["act alpha"],
    ),
    lambda lib: lib["programs"][0]["positive_net_guard"]["clauses"][0]["literals"][0].update(sign="-"),
    lambda lib: lib["programs"][0]["positive_net_guard"]["clauses"][0]["literals"][0].update(
        action="unexecuted action",
    ),
    lambda lib: lib["programs"][0]["reentry_projection"]["positive"]["history_actions"].append(
        "irrelevant action",
    ),
    lambda lib: lib["programs"][0]["reentry_projection"]["positive"].update(history_words=["invented"]),
    lambda lib: lib["programs"][0].pop("reentry_projection"),
    lambda lib: lib["parameters"].update(composed_reentry="all_history"),
])
def test_typed_net_atoms_and_projection_require_valid_source_provenance(mutation):
    library, policy, _ = compile_bundle(observations=["x", "alpha", "x"])
    mutation(library)
    with pytest.raises(ValueError):
        controller(library, policy, "composed_contract")
    with pytest.raises(ValueError):
        contracts.runtime_contract_hash(library, "composed_contract")


@pytest.mark.parametrize("mutation", [
    lambda row: row.pop("pre_action_history"),
    lambda row: row.pop("post_action_history"),
    lambda row: row.update(pre_action_history=True),
    lambda row: row.update(post_action_history=[True]),
    lambda row: row.update(post_action_history=["act alpha"]),
    lambda row: row.update(post_action_history=["act alpha", "act beta", "unrecorded action"]),
])
def test_negative_program_history_must_be_valid_monotone_and_trace_bound(mutation):
    record = cycle_negative("x")
    mutation(record)
    with pytest.raises(ValueError):
        compile_bundle(negative_programs=[record])


def test_identical_text_in_observation_and_action_history_is_a_different_typed_fact():
    programs, demos, transitions = bundle(observations=["", "beta", "alpha"])
    renamed = {"act alpha": "alpha", "act beta": "beta"}
    programs[0]["actions"] = [renamed[action] for action in programs[0]["actions"]]
    for demo in demos:
        demo["action"] = renamed[demo["action"]]
        demo["public_state"]["actions"] = [renamed[action] for action in demo["public_state"]["actions"]]
    for transition in transitions.values():
        transition["pre_action_history"] = [renamed[action] for action in transition["pre_action_history"]]
    library = contracts.compile_library(programs, demos, transitions, [])
    guard = library["programs"][0]["positive_net_guard"]
    assert guard["counts"]["eligible_atom_count"] == 3
    assert guard["clauses"][0]["literals"] == [
        {"field": "history_action", "sign": "+", "action": "alpha"},
    ]
    observation_guard = deepcopy(library)
    observation_guard["programs"][0]["positive_net_guard"]["clauses"][0]["literals"] = [
        {"field": "observation", "sign": "+", "tokens": ["alpha"]},
    ]
    observation_guard["programs"][0]["reentry_projection"]["positive"]["history_actions"] = []
    assert contracts.runtime_contract_hash(library, "composed_contract") != contracts.runtime_contract_hash(
        observation_guard, "composed_contract",
    )
    policy = invocation.build_policy(demos)
    for candidate, accepted in ((library, False), (observation_guard, True)):
        control = controller(candidate, policy, "composed_contract")
        control.observe("alpha", "", compiled=False)
        result = control.decide(public("", actions=["alpha", "beta"]))
        assert result["decision"]["commit"] is accepted


def test_extra_net_field_metadata_does_not_change_step_literal_semantics():
    library, policy, _ = compile_bundle()
    changed = deepcopy(library)
    for step in changed["programs"][0]["steps"]:
        for variant in ("positive_guard", "refined_guard"):
            for clause in step[variant]["clauses"]:
                for literal in clause["literals"]:
                    literal["field"] = "history_action"
                    literal["action"] = "ignored step metadata"
    for mode in _STEP_ONLY_MODES:
        assert contracts.runtime_contract_hash(library, mode) == contracts.runtime_contract_hash(
            changed, mode,
        )
        assert controller(library, policy, mode).decide(public()) == controller(
            changed, policy, mode,
        ).decide(public())


@pytest.mark.parametrize("mode", contracts.MODES)
def test_training_interventions_are_disabled_by_default_and_empty_opt_in_is_identical(mode):
    library, policy, _ = compile_bundle()
    baseline = controller(library, policy, mode)
    disabled = controller(library, policy, mode, allow_training_interventions=False)
    enabled = controller(library, policy, mode, allow_training_interventions=True)
    instance_id = library["programs"][0]["instance_id"]
    for control in (baseline, disabled):
        with pytest.raises(ValueError, match="disabled"):
            control.exclude_entry(public(), instance_id)
    controls = (baseline, disabled, enabled)
    for control in controls:
        control.observe("initial native action", "x", compiled=False)
    for before, after in (("x", "alpha"), ("alpha", "beta")):
        results = [control.decide(public(before)) for control in controls]
        assert results[0] == results[1] == results[2]
        feedback = [
            control.observe(result["proposal"]["action"], after, compiled=True)
            for control, result in zip(controls, results)
        ]
        assert feedback[0] == feedback[1] == feedback[2]
    boundaries = [control.decide(public()) for control in controls]
    assert boundaries[0] == boundaries[1] == boundaries[2]


@pytest.mark.parametrize("flag", [0, 1, None, "true", [], {}])
def test_training_intervention_flag_requires_a_bool(flag):
    library, policy, _ = compile_bundle()
    with pytest.raises(ValueError, match="allow_training_interventions must be a bool"):
        controller(library, policy, allow_training_interventions=flag)


@pytest.mark.parametrize("instance_id", ["unknown", "p", "", None, True, 1, []])
def test_training_exclusion_rejects_unknown_or_invalid_instances_without_state_changes(instance_id):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, allow_training_interventions=True)
    with pytest.raises(ValueError):
        control.exclude_entry(public(), instance_id)
    assert control._entry_exclusions == {}
    assert control.decide(public()) == controller(library, policy).decide(public())


@pytest.mark.parametrize("extra", ["task_id", "group_id", "official_success", "reward", "future_label"])
def test_training_exclusion_rejects_identity_evaluator_and_future_state_fields(extra):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, allow_training_interventions=True)
    with pytest.raises(ValueError, match="Public state"):
        control.exclude_entry({**public(), extra: "not public"}, library["programs"][0]["instance_id"])
    assert control._entry_exclusions == {}
    assert control.decide(public())["decision"]["commit"]


@pytest.mark.parametrize("mode", contracts.MODES)
def test_training_exclusion_rejects_pending_and_mid_fragment_use(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode, allow_training_interventions=True)
    instance_id = library["programs"][0]["instance_id"]
    first = control.decide(public())
    with pytest.raises(ValueError, match="pending"):
        control.exclude_entry(public(), instance_id)
    assert control._entry_exclusions == {}
    feedback = control.observe(first["proposal"]["action"], "alpha", compiled=True)
    assert feedback["next_step"] == 1
    with pytest.raises(ValueError, match="mid-fragment"):
        control.exclude_entry(public("alpha"), instance_id)
    assert control._entry_exclusions == {}
    second = control.decide(public("alpha"))
    assert second["proposal"]["step"] == 1
    assert second["proposal"]["instance_id"] == instance_id


def test_pending_actor_feedback_also_prevents_entry_intervention():
    library, policy, _ = compile_bundle()
    control = controller(library, policy, allow_training_interventions=True)
    result = control.decide(public(actions=["other native action"]))
    assert not result["decision"]["commit"]
    with pytest.raises(ValueError, match="pending"):
        control.exclude_entry(public(), library["programs"][0]["instance_id"])
    control.observe("other native action", "x", compiled=False)
    control.exclude_entry(public(), library["programs"][0]["instance_id"])
    denied = control.decide(public())
    assert not denied["decision"]["commit"]
    assert denied["decision"]["entry_rejections"][0]["reason"] == "training_entry_excluded"


@pytest.mark.parametrize("mode", contracts.MODES)
@pytest.mark.parametrize("excluded_program", ["first", "second"])
def test_training_exclusion_is_per_instance_not_an_action_blacklist(
    monkeypatch, mode, excluded_program,
):
    first, first_demos, first_transitions = bundle(name="first")
    second, second_demos, second_transitions = bundle(name="second")
    library = contracts.compile_library(
        first + second, first_demos + second_demos, {**first_transitions, **second_transitions}, [],
    )
    policy = invocation.build_policy(first_demos + second_demos)
    fixed_neighbors(monkeypatch, [(1.0, first_demos[0]), (0.9, second_demos[0])])
    control = controller(library, policy, mode, allow_training_interventions=True)
    instance = next(row["instance_id"] for row in library["programs"]
                    if row["program_id"] == excluded_program)
    event = control.exclude_entry(public(), instance)
    assert event["reason"] == "training_entry_excluded"
    result = control.decide(public())
    assert result["decision"]["commit"]
    assert result["proposal"]["program_id"] == ("second" if excluded_program == "first" else "first")
    assert result["proposal"]["action"] == "act alpha"
    if excluded_program == "first":
        rejected = result["decision"]["entry_rejections"][0]
        assert rejected["reason"] == "training_entry_excluded"
        assert rejected["instance_id"] == instance
        assert rejected["exclusion_key"] == event["exclusion_key"]
    else:
        assert result["decision"]["entry_rejections"] == []


@pytest.mark.parametrize("mode", contracts.MODES)
def test_excluded_entry_persists_across_equivalent_states_and_irrelevant_history(monkeypatch, mode):
    library, policy, demos = compile_bundle()
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, mode, allow_training_interventions=True)
    instance = library["programs"][0]["instance_id"]
    event = control.exclude_entry(public(), instance)
    assert not event["already_excluded"]
    first = control.decide(public())
    assert not first["decision"]["commit"]
    assert first["decision"]["reason"] == "no_eligible_fragment"
    assert first["decision"]["entry_rejections"][0]["reason"] == "training_entry_excluded"
    control.observe("unrelated native action", "x", compiled=False)
    changed = public(actions=["act beta", "act alpha"], history=[
        {"action": "unrelated native action", "observation": "x"},
    ])
    repeated = control.exclude_entry(changed, instance)
    assert repeated["already_excluded"]
    assert repeated["exclusion_key"] == event["exclusion_key"]
    assert repeated["state_sha256"] != event["state_sha256"]
    denied = control.decide(changed)
    assert not denied["decision"]["commit"]
    assert denied["decision"]["entry_rejections"][0]["exclusion_key"] == event["exclusion_key"]
    control.observe("act alpha", "instructions act alpha", compiled=False, native_valid=False)
    assert control.exclude_entry(public(), instance)["already_excluded"]
    assert not control.decide(public())["decision"]["commit"]


@pytest.mark.parametrize("mode", contracts.MODES)
@pytest.mark.parametrize("change", ["observation", "actions", "task"])
def test_changed_current_observable_state_can_permit_a_previously_excluded_entry(
    monkeypatch, mode, change,
):
    library, policy, demos = compile_bundle(task_family="opaque")
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(
        library, policy, mode, task_family="opaque", allow_training_interventions=True,
    )
    control.exclude_entry(public(), library["programs"][0]["instance_id"])
    changed = public()
    if change == "observation":
        changed["observation"] = "x changed"
    elif change == "actions":
        changed["actions"].append("new native action")
    else:
        changed["task"] = "another alpha beta request"
    result = control.decide(changed)
    assert result["decision"]["commit"]
    assert result["proposal"]["instance_id"] == library["programs"][0]["instance_id"]
    assert result["decision"]["entry_rejections"] == []


@pytest.mark.parametrize("mode", contracts.MODES)
def test_newly_satisfied_history_prerequisites_change_the_exclusion_key(monkeypatch, mode):
    library, policy, demos = compile_bundle(pre_action_history=["prior alpha"])
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, mode, allow_training_interventions=True)
    instance = library["programs"][0]["instance_id"]
    control.exclude_entry(public(), instance)
    control.observe("prior alpha", "x", compiled=False)
    result = control.decide(public())
    assert result["decision"]["commit"]
    assert result["proposal"]["instance_id"] == instance
    assert result["decision"]["entry_rejections"] == []


@pytest.mark.parametrize("mode", _COMPOSED_MODES)
def test_selected_full_history_action_can_invalidate_an_entry_exclusion(monkeypatch, mode):
    library, policy, demos = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        observations=lambda group, _: ["x", "alpha", "x"] if group in ("g0", "g1")
        else ["z", "alpha z", "final"],
        pre_action_history=lambda group, _: [] if group in ("g0", "g1") else list(_KNOWN_ACTIONS),
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, mode, allow_training_interventions=True)
    instance = library["programs"][0]["instance_id"]
    control.exclude_entry(public("x"), instance)
    assert not control.decide(public("x"))["decision"]["commit"]
    control.observe("act alpha", "x", compiled=False, native_valid=True)
    result = control.decide(public("x"))
    assert result["decision"]["commit"]
    assert result["proposal"]["instance_id"] == instance
    assert result["decision"]["matched_net_precondition_clauses"] == [1]


def test_training_exclusion_changes_only_intervention_state_and_not_library_hashes():
    library, policy, _ = compile_bundle()
    control = controller(library, policy, allow_training_interventions=True)
    control.observe("initial native", "x", compiled=False)
    state = public()
    original_state, original_library, original_policy = deepcopy((state, library, policy))
    fields = (
        "_active", "_pending", "_actor_required", "_seen_entries", "_completed_actions",
        "_native_actions", "_native_action_set", "_native_action_words",
    )
    before = {field: deepcopy(getattr(control, field)) for field in fields}
    hashes = (contracts.library_behavior_hash(library),
              contracts.runtime_contract_hash(library, "positive_contract"))
    instance = library["programs"][0]["instance_id"]
    event = control.exclude_entry(state, instance)
    assert set(event) == {
        "reason", "program_id", "instance_id", "state_sha256", "exclusion_key", "already_excluded",
    }
    assert event == json.loads(json.dumps(event, allow_nan=False))
    assert len(event["exclusion_key"]) == 64
    assert set(event["exclusion_key"]) <= set("0123456789abcdef")
    assert {field: getattr(control, field) for field in fields} == before
    assert (state, library, policy) == (original_state, original_library, original_policy)
    assert control.exclude_entry(state, instance)["already_excluded"]
    assert len(control._entry_exclusions[instance]) == 1
    assert hashes == (contracts.library_behavior_hash(library),
                      contracts.runtime_contract_hash(library, "positive_contract"))


def test_exclusion_does_not_clear_a_required_actor_boundary_or_undo_execution():
    library, policy, _ = compile_bundle()
    control = controller(library, policy, "ordered_fragment", allow_training_interventions=True)
    for before, after in (("x", "alpha"), ("alpha", "beta")):
        result = control.decide(public(before))
        feedback = control.observe(result["proposal"]["action"], after, compiled=True)
    assert feedback["fragment_complete"] and feedback["native_action_count"] == 2
    state = public("x next")
    control.exclude_entry(state, library["programs"][0]["instance_id"])
    boundary = control.decide(state)["decision"]
    assert boundary["reason"] == "actor_boundary_required"
    assert boundary["boundary_reason"] == "fragment_completed"
    assert boundary["completed_action_count"] == boundary["native_action_count"] == 2
    control.observe("actor after boundary", "x next", compiled=False)
    denied = control.decide(state)
    assert not denied["decision"]["commit"]
    assert denied["decision"]["entry_rejections"][0]["reason"] == "training_entry_excluded"
    assert denied["decision"]["native_action_count"] == 3


@pytest.mark.parametrize("mode", contracts.MODES)
def test_episode_reset_clears_entry_exclusions_without_disabling_opt_in(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode, allow_training_interventions=True)
    instance = library["programs"][0]["instance_id"]
    control.exclude_entry(public(), instance)
    assert not control.decide(public())["decision"]["commit"]
    control.observe("actor", "x", compiled=False)
    control.reset_episode()
    assert control._entry_exclusions == {}
    assert control.decide(public())["decision"]["commit"]
    control.observe("actor override", "x", compiled=False, native_valid=False)
    assert not control.exclude_entry(public(), instance)["already_excluded"]


@pytest.mark.parametrize("mode", contracts.MODES)
def test_recorded_actor_prefix_replay_reconstructs_the_same_intervention_key(mode):
    library, policy, _ = compile_bundle()
    events = [("unrelated", "x", True), ("act alpha", "x", False), ("another native", "x", True)]
    controls = [controller(library, policy, mode, allow_training_interventions=True) for _ in range(2)]
    results = []
    for control in controls:
        for action, observation, native_valid in events:
            control.observe(action, observation, compiled=False, native_valid=native_valid)
        results.append(control.exclude_entry(public(), library["programs"][0]["instance_id"]))
    assert results[0] == results[1]
    assert controls[0].decide(public()) == controls[1].decide(public())


@pytest.mark.parametrize("mode", contracts.MODES)
def test_training_exclusion_does_not_make_an_ineligible_alternative_eligible(monkeypatch, mode):
    first, first_demos, first_transitions = bundle(name="first")
    second, second_demos, second_transitions = bundle(name="second", task="other alpha beta task")
    library = contracts.compile_library(
        first + second, first_demos + second_demos, {**first_transitions, **second_transitions}, [],
    )
    policy = invocation.build_policy(first_demos + second_demos)
    fixed_neighbors(monkeypatch, [(1.0, first_demos[0]), (0.9, second_demos[0])])
    control = controller(library, policy, mode, allow_training_interventions=True)
    target = next(row["instance_id"] for row in library["programs"] if row["program_id"] == "first")
    control.exclude_entry(public(), target)
    result = control.decide(public())
    assert not result["decision"]["commit"]
    assert result["proposal"]["action"] is None
    assert [row["reason"] for row in result["decision"]["entry_rejections"]] == [
        "training_entry_excluded", "source_task_text_mismatch",
    ]


@pytest.mark.parametrize("mode", contracts.MODES)
def test_registered_entry_exclusion_never_turns_into_a_mid_fragment_action_ban(mode):
    library, policy, _ = compile_bundle(
        observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    control = prime_native_history(
        controller(library, policy, mode, allow_training_interventions=True), _KNOWN_ACTIONS,
    )
    control.exclude_entry(public("alpha beta"), library["programs"][0]["instance_id"])
    control.observe("actor returns to start", "x", compiled=False)
    for step, (before, after) in enumerate((("x", "alpha beta"), ("alpha beta", "beta"))):
        result = control.decide(public(before))
        assert result["decision"]["commit"]
        assert result["proposal"]["step"] == step
        feedback = control.observe(result["proposal"]["action"], after, compiled=True)
    assert feedback["fragment_complete"] and not feedback["fragment_aborted"]


# Captured from the pre-ablation v5 code using only the synthetic fixture below.
_V5_MODE_REGRESSIONS = [
    (
        "independent_actions",
        "db70702018e3807a94b445ac5cdbfc76e4bfe5e60c78f59e13e904fb2352f3dc",
        "eaa020695e4e9d291ea695a09454a707ea214863d706223ef692768bf8dc3fc0",
    ),
    (
        "ordered_fragment",
        "d4422235b42562c17f572d2b05531ea78e268064724f115f705637491148fc04",
        "d79b67039de1e3b37b04e6939bd6174c2853b05d08bf71fa21430840bdf68165",
    ),
    (
        "positive_contract",
        "a1a2e6e8c950c7d2c0aeaa070fe519d587a09ca87e1dffc1d8528cd2d633ba02",
        "28960cb2a226e9d5fc2177dfb689d63e9c100bb549f6479b254cb3fc1bd6f777",
    ),
    (
        "refined_contract",
        "a1a2e6e8c950c7d2c0aeaa070fe519d587a09ca87e1dffc1d8528cd2d633ba02",
        "9b02cb723bc6b0c05201cf594963bdb4c6f8b1937df018deadcf3ad929965951",
    ),
    (
        "refined_no_memory",
        "a1fa97fc709c4aab567aef3f402ede87128d5b7ccec3461c34c6d15a3fd7e0e9",
        "36459fddf7769771c27f3f5fc067ff696430c6c70d5c07e269219daa55fba4b3",
    ),
    (
        "composed_contract",
        "d78aad92c82bc290b3ff918154214a0ef2f1fd92e9545820d52cd6a47bddb2af",
        "cfa3f31cc99cd08baf7a707b65f1131f4d88f15ed620e0857a1bf940bbf1e5ba",
    ),
    (
        "refined_composed",
        "d78aad92c82bc290b3ff918154214a0ef2f1fd92e9545820d52cd6a47bddb2af",
        "83ac429c289f1f4d72ad57a4c4deba8037e9ec614d13a62fb04d53b12222b686",
    ),
]


@pytest.mark.parametrize("mode,expected_runtime_hash,expected_trace_hash", _V5_MODE_REGRESSIONS)
def test_original_v5_hashes_and_complete_synthetic_outputs_remain_unchanged(
    mode, expected_runtime_hash, expected_trace_hash,
):
    library, policy, _ = compile_bundle()
    assert contracts.library_behavior_hash(library) == (
        "5fe440177f5516bbea4fbc89fb7f1955451c2d1eb68972f6a1288e9d9967aaaa"
    )
    assert contracts.runtime_contract_hash(library, mode) == expected_runtime_hash
    control = controller(library, policy, mode)
    trace = [control.observe("act beta", "x", compiled=False)]
    for before, after in (
        ("x", "alpha"), ("alpha", "beta"), ("x", "x"), ("x", "x"),
        ("x changed", "alpha"), ("alpha", "beta"),
    ):
        result = control.decide(public(before))
        trace.append(result)
        compiled = result["decision"]["commit"]
        action = result["proposal"]["action"] if compiled else "act beta"
        trace.append(control.observe(action, after, compiled=compiled))
    assert hashlib.sha256(json.dumps(
        trace, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest() == expected_trace_hash


@pytest.mark.parametrize("mode", _COMPOSED_ABLATIONS)
def test_composed_ablations_keep_the_net_supported_pool(mode):
    library, policy, _ = compile_bundle(
        observations=["x", "alpha", "x"], pre_action_history=_KNOWN_ACTIONS,
    )
    assert library["programs"][0]["positive_net_guard"] is None
    control = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS)
    denied = control.decide(public("x"))
    assert not denied["decision"]["commit"]
    assert denied["decision"]["entry_rejections"][0]["reason"] == "no_supported_net_guard"


def test_no_net_checks_removes_net_preconditions_but_still_checks_step_effects(monkeypatch):
    library, policy, demos = compile_bundle(
        observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    baseline = prime_native_history(controller(library, policy, "composed_contract"), _KNOWN_ACTIONS)
    control = prime_native_history(
        controller(library, policy, "composed_no_net_checks"), _KNOWN_ACTIONS,
    )
    denied = baseline.decide(public("x beta"))
    assert denied["decision"]["entry_rejections"][0]["reason"] == "entry_net_precondition_failed"
    allowed = control.decide(public("x beta"))
    assert allowed["decision"]["commit"]
    assert allowed["decision"]["precondition_checked"]
    assert allowed["decision"]["net_precondition_checked"] is False
    assert allowed["decision"]["matched_net_precondition_clauses"] == []
    feedback = control.observe(allowed["proposal"]["action"], "x beta", compiled=True)
    assert feedback["postcondition_passed"] is False
    assert feedback["reason"] == "postcondition_failed"
    assert feedback["net_postcondition_checked"] is False
    assert feedback["net_postcondition_passed"] is None


def test_no_net_checks_removes_only_net_postcondition_enforcement():
    library, policy, _ = compile_bundle(
        observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    results = {}
    for mode in ("composed_contract", "composed_no_net_checks"):
        control = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS)
        for before, after in (("x", "alpha"), ("alpha", "x")):
            decision = control.decide(public(before))
            assert decision["decision"]["commit"]
            feedback = control.observe(decision["proposal"]["action"], after, compiled=True)
            assert feedback["postcondition_passed"] is True
        results[mode] = feedback
        assert feedback["native_action_count"] == 4
        assert control.decide(public())["decision"]["reason"] == "actor_boundary_required"
    assert results["composed_contract"]["net_postcondition_passed"] is False
    assert results["composed_contract"]["fragment_aborted"]
    assert results["composed_no_net_checks"]["net_postcondition_checked"] is False
    assert results["composed_no_net_checks"]["net_postcondition_passed"] is None
    assert results["composed_no_net_checks"]["fragment_complete"]


def test_no_net_checks_retains_net_derived_relevant_history_projection(monkeypatch):
    library, policy, demos = compile_bundle(observations=["x", "alpha", "x"])
    assert library["programs"][0]["reentry_projection"]["positive"] == {
        "history_actions": ["act alpha"], "history_words": [],
    }
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    control = controller(library, policy, "composed_no_net_checks")
    actions = ["act alpha", "act beta", "unrelated one", "unrelated two"]
    for iteration in range(2):
        for before, after in (("x", "alpha"), ("alpha", "x")):
            result = control.decide(public(before, actions=actions))
            assert result["decision"]["commit"]
            assert result["decision"]["net_precondition_checked"] is False
            feedback = control.observe(result["proposal"]["action"], after, compiled=True)
            assert feedback["net_postcondition_passed"] is None
        assert feedback["fragment_complete"]
        control.observe("unrelated one" if iteration == 0 else "unrelated two", "x", compiled=False)
    denied = control.decide(public("x", actions=actions, history=[
        {"action": "unrelated two", "observation": "x"},
    ]))
    assert not denied["decision"]["commit"]
    assert denied["decision"]["reason"] == "repeated_fragment_entry_state"
    assert len(control._seen_entries) == 2


def test_raw_reentry_changes_only_raw_history_sensitivity(monkeypatch):
    library, policy, demos = compile_bundle(
        observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    for mode, allowed in (("composed_contract", False), ("composed_raw_reentry", True)):
        control = prime_native_history(controller(library, policy, mode), _KNOWN_ACTIONS)
        for before, after in (("x", "alpha"), ("alpha", "x")):
            result = control.decide(public(before))
            feedback = control.observe(result["proposal"]["action"], after, compiled=True)
        assert feedback["net_postcondition_passed"] is False
        control.observe("act beta", "x", compiled=False)
        changed_history = public("x", history=[{"action": "act beta", "observation": "x"}])
        result = control.decide(changed_history)
        assert result["decision"]["commit"] is allowed
        if not allowed:
            assert result["decision"]["reason"] == "repeated_fragment_entry_state"


def test_raw_reentry_keeps_instance_identity(monkeypatch):
    first, first_demos, first_transitions = bundle(
        name="first", observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    second, second_demos, second_transitions = bundle(
        name="second", observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    library = contracts.compile_library(
        first + second, first_demos + second_demos, {**first_transitions, **second_transitions}, [],
    )
    policy = invocation.build_policy(first_demos + second_demos)
    fixed_neighbors(monkeypatch, [(1.0, first_demos[0]), (0.9, second_demos[0])])
    control = prime_native_history(controller(library, policy, "composed_raw_reentry"), _KNOWN_ACTIONS)
    for before, after in (("x", "alpha beta"), ("alpha beta", "beta")):
        result = control.decide(public(before))
        assert result["proposal"]["program_id"] == "first"
        control.observe(result["proposal"]["action"], after, compiled=True)
    control.observe("act beta", "x", compiled=False)
    result = control.decide(public("x"))
    assert result["decision"]["commit"]
    assert result["proposal"]["program_id"] == "second"
    assert result["decision"]["entry_rejections"][0]["program_id"] == "first"
    assert result["decision"]["entry_rejections"][0]["reason"] == "repeated_fragment_entry_state"


def test_no_reentry_memory_preserves_net_checks_native_history_counter_and_actor_boundary():
    library, policy, _ = compile_bundle(
        observations=["x", "alpha beta", "beta"], pre_action_history=_KNOWN_ACTIONS,
    )
    unprimed = controller(library, policy, "composed_no_reentry_memory")
    denied = unprimed.decide(public())
    assert denied["decision"]["entry_rejections"][0]["reason"] == "history_prerequisites_missing"
    control = prime_native_history(
        controller(library, policy, "composed_no_reentry_memory"), _KNOWN_ACTIONS,
    )
    for step, (before, after) in enumerate((("x", "alpha"), ("alpha", "x"))):
        result = control.decide(public(before))
        assert result["decision"]["commit"]
        assert result["proposal"]["step"] == step
        feedback = control.observe(result["proposal"]["action"], after, compiled=True)
        assert feedback["seen_entry_count"] == 0
    assert feedback["net_postcondition_passed"] is False
    assert feedback["fragment_aborted"] and feedback["actor_required"]
    assert control.decide(public())["decision"]["reason"] == "actor_boundary_required"
    control.observe("act beta", "x", compiled=False)
    again = control.decide(public())
    assert again["decision"]["commit"]
    assert again["proposal"]["step"] == 0
    assert again["decision"]["native_action_count"] == 5
    assert again["decision"]["net_precondition_checked"]
    assert not control._seen_entries


@pytest.mark.parametrize("mode", _COMPOSED_ABLATIONS)
def test_composed_ablations_preserve_native_action_membership(mode):
    library, policy, _ = compile_bundle()
    control = controller(library, policy, mode)
    first = control.decide(public())
    control.observe(first["proposal"]["action"], "alpha", compiled=True)
    denied = control.decide(public("alpha", actions=["act alpha"]))
    assert not denied["decision"]["commit"]
    assert denied["decision"]["reason"] == "fragment_action_not_native"


def test_ablation_runtime_hashes_cover_checks_raw_scope_and_disabled_registry():
    library, _, _ = compile_bundle()
    modes = ("composed_contract", *_COMPOSED_ABLATIONS)
    hashes = {mode: contracts.runtime_contract_hash(library, mode) for mode in modes}
    assert len(set(hashes.values())) == 4
    assert hashes["composed_contract"] == contracts.runtime_contract_hash(library, "refined_composed")


def test_no_net_check_hash_ignores_unenforced_observation_clauses_but_keeps_history_projection():
    library, _, _ = compile_bundle(pre_action_history=_KNOWN_ACTIONS)
    changed = deepcopy(library)
    changed["programs"][0]["positive_net_guard"]["clauses"][0]["literals"][0]["tokens"] = ["new"]
    mode = "composed_no_net_checks"
    assert contracts.runtime_contract_hash(library, mode) == contracts.runtime_contract_hash(changed, mode)
    for checked in ("composed_contract", "composed_raw_reentry", "composed_no_reentry_memory"):
        assert contracts.runtime_contract_hash(library, checked) != contracts.runtime_contract_hash(
            changed, checked,
        )
    library, _, _ = compile_bundle()
    changed = deepcopy(library)
    changed["programs"][0]["positive_net_guard"]["clauses"][0]["literals"][0]["action"] = "act beta"
    changed["programs"][0]["reentry_projection"]["positive"]["history_actions"] = ["act beta"]
    assert contracts.runtime_contract_hash(library, mode) != contracts.runtime_contract_hash(changed, mode)


@pytest.mark.parametrize("mode", _COMPOSED_ABLATIONS)
def test_composed_ablation_hashes_use_positive_not_inactive_refined_contracts(mode):
    library, _, _ = compile_bundle()
    changed = deepcopy(library)
    changed["programs"][0]["refined_net_guard"]["clauses"][0]["literals"][0]["action"] = "act beta"
    changed["programs"][0]["reentry_projection"]["refined"]["history_actions"] = ["act beta"]
    assert contracts.runtime_contract_hash(library, mode) == contracts.runtime_contract_hash(changed, mode)


def test_raw_reentry_replaces_rather_than_adds_to_the_relevant_history_projection(monkeypatch):
    library, policy, demos = compile_bundle(
        groups=("g0", "g1", "g2", "g3"),
        observations=lambda group, _: ["x", "alpha", "x"] if group in ("g0", "g1")
        else ["z", "alpha z", "final"],
        pre_action_history=lambda group, _: [] if group in ("g0", "g1") else list(_KNOWN_ACTIONS),
    )
    fixed_neighbors(monkeypatch, [(1.0, demos[0])])
    for mode, allowed in (("composed_contract", True), ("composed_raw_reentry", False)):
        control = controller(library, policy, mode)
        for before, after in (("x", "alpha"), ("alpha", "x")):
            result = control.decide(public(before))
            feedback = control.observe(result["proposal"]["action"], after, compiled=True)
        assert feedback["net_postcondition_passed"] is True
        control.observe("act beta", "x", compiled=False)
        result = control.decide(public("x"))
        assert result["decision"]["commit"] is allowed
        assert result["decision"]["native_action_count"] == 3
        if not allowed:
            assert result["decision"]["reason"] == "repeated_fragment_entry_state"


def test_no_reentry_memory_does_not_forget_native_action_facts_used_by_net_guards():
    library, policy, _ = compile_bundle()
    control = controller(library, policy, "composed_no_reentry_memory")
    for before, after in (("x", "alpha"), ("alpha", "beta")):
        result = control.decide(public(before))
        feedback = control.observe(result["proposal"]["action"], after, compiled=True)
    assert feedback["fragment_complete"]
    assert feedback["seen_entry_count"] == 0
    control.observe("act beta", "x", compiled=False)
    denied = control.decide(public("x"))
    assert not denied["decision"]["commit"]
    assert denied["decision"]["entry_rejections"][0]["reason"] == "entry_net_precondition_failed"
    assert denied["decision"]["native_action_count"] == 3


def test_all_positive_composed_ablations_keep_the_same_safe_training_exclusion_key():
    library, policy, _ = compile_bundle()
    events = []
    state = public(history=[{"action": "act beta", "observation": "x"}])
    for mode in ("composed_contract", *_COMPOSED_ABLATIONS):
        control = controller(library, policy, mode, allow_training_interventions=True)
        control.observe("act beta", "x", compiled=False)
        events.append(control.exclude_entry(state, library["programs"][0]["instance_id"]))
        denied = control.decide(state)
        assert not denied["decision"]["commit"]
        assert denied["decision"]["entry_rejections"][0]["reason"] == "training_entry_excluded"
    assert all(event == events[0] for event in events)
