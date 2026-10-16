from copy import deepcopy
import hashlib
import json

import pytest

from chisel.learning import observed_feedback_plans as events


FIRST = "touch orb 9"
NEXT = "continue orb 9"
OBSERVATION = "Touched orb 9."
EVENT_MODES = ("event_guarded", "event_no_novelty", "event_unchecked")


def record(index=0, *, group=None, action=None, post=None, task=None):
    return {
        "source_id": f"source-{index}", "task_id": task or f"task-{index}",
        "group_id": f"group-{index}" if group is None else group,
        "action": action or f"touch orb {index + 1}",
        "post_observation": f"Touched orb {index + 1}." if post is None else post,
    }


def catalog():
    return events.fit_feedback_catalog([record(0), record(1)])


def novelty_catalog():
    return events.fit_feedback_catalog([
        record(0), record(1),
        record(2, action="unseal case1", post="Unsealed case1; visible orb2."),
        record(3, action="unseal case2", post="Unsealed case2; visible orb3."),
    ])


def plan(menu, *, first=FIRST, continuation=NEXT, event_id=None):
    return {
        "reasoning": "Choose the native action and one conditional continuation.",
        "action": first,
        "continuation": None if continuation is None else {
            "action": continuation,
            "after_event": event_id or menu["action_events"][first][0],
        },
    }


def parsed(menu, value=None, *, allow=True):
    return events.parse_event_plan(
        json.dumps(plan(menu) if value is None else value),
        native_actions=menu["native_actions"], menu=menu, allow_continuation=allow,
    )


def installed(mode="event_guarded", *, catalogue=None, first=FIRST, next_action=NEXT,
              task="inspect orb", observation="initial", native_actions=None):
    catalogue = catalog() if catalogue is None else catalogue
    native_actions = [first] if native_actions is None else native_actions
    menu = events.catalog_for_actions(catalogue, native_actions)
    value = plan(menu, first=first, continuation=None if mode == "structured_single" else next_action)
    control = events.EventPlanState(catalogue, task, mode)
    control.install(value, current_observation=observation, current_actions=native_actions, menu=menu)
    return control, value, menu


def decided(mode, feedback=OBSERVATION, *, first=FIRST, native=None, **kwargs):
    control, value, menu = installed(mode, first=first, **kwargs)
    control.observe(first, feedback, native_valid=True)
    result = control.decide(feedback, [NEXT] if native is None else native)
    return control, result, value, menu


def test_catalog_learns_slots_and_empirical_group_provenance_not_task_text():
    rows = [record(0, task="not-a-predictive-goal77"), record(1)]
    result = events.fit_feedback_catalog(rows)
    assert result["schema_version"] == events.CATALOG_SCHEMA
    assert set(result["entity_vocabulary"]) == {"orb"}
    assert result["entity_vocabulary"]["orb"]["source_groups"] == ["group-0", "group-1"]
    assert len(result["events"]) == 1
    identifier, event = next(iter(result["events"].items()))
    assert identifier.startswith("ev_") and len(identifier) == 67
    assert event["action_shape"] == ["touch ", {"slot": 0}, ""]
    assert event["feedback_template"] == ["Touched ", {"slot": 0}, "."]
    assert event["argument_count"] == 1
    assert event["match_type"] == "exact_match"
    assert event["source_count"] == event["support_group_count"] == 2
    assert event["support_groups"] == ["group-0", "group-1"]
    assert {row["source_id"] for row in event["sources"]} == {"source-0", "source-1"}
    assert "goal" not in json.dumps(event["action_shape"] + event["feedback_template"])


def test_source_duplicates_are_counted_once_and_conflicting_ids_are_errors():
    rows = [record(0), record(1)]
    original = events.fit_feedback_catalog(rows)
    repeated = events.fit_feedback_catalog(rows + [deepcopy(rows[0])] * 8)
    assert repeated["events"] == original["events"]
    assert repeated["entity_vocabulary"] == original["entity_vocabulary"]
    assert repeated["counts"]["duplicate_records"] == 8
    assert repeated["counts"]["retained_records"] == 2
    with pytest.raises(ValueError, match="Conflicting duplicate source_id"):
        events.fit_feedback_catalog(rows + [{**rows[0], "post_observation": "Different."}])


def test_many_prefixes_from_one_group_do_not_satisfy_independent_support():
    result = events.fit_feedback_catalog([record(index, group="one-group") for index in range(12)])
    assert result["events"] == {}
    assert result["counts"]["rejected_low_support_events"] == 1
    rejection = result["rejections"][0]
    assert rejection["reason"] == "insufficient_independent_groups"
    assert rejection["source_count"] == 12
    assert rejection["support_group_count"] == 1
    assert rejection["support_groups"] == ["one-group"]


def test_group_exclusion_precedes_vocabulary_deduplication_and_template_fitting():
    rows = [record(0), record(1), record(2, group="target", action="touch secret8", post="Hidden.")]
    rows.append({**rows[0], "group_id": "target", "action": "touch private9", "post_observation": "Hidden."})
    result = events.fit_feedback_catalog(rows, excluded_groups={"target"})
    assert set(result["entity_vocabulary"]) == {"orb"}
    assert result["counts"]["excluded_records"] == 2
    assert result["counts"]["retained_records"] == 2
    assert result["events"] == catalog()["events"]
    for event in result["events"].values():
        assert "target" not in event["support_groups"]
    excluded_support = events.fit_feedback_catalog([record(0), record(1)], excluded_groups={"group-1"})
    assert excluded_support["events"] == {}
    assert excluded_support["rejections"][0]["support_group_count"] == 1


def test_bound_templates_transfer_current_entity_instances_without_guessing_ids():
    result = events.fit_feedback_catalog([
        record(0, action="link orb1 to tile2", post="Linked tile2 with orb1."),
        record(1, action="link orb7 to tile8", post="Linked tile8 with orb7."),
    ])
    actions = ["link orb20 to tile30", "link orb21 to tile31"]
    menu = events.catalog_for_actions(result, actions)
    assert len(menu["events"]) == 1
    assert next(iter(menu["events"].values()))["catalog_event_id"] in result["events"]
    for action, event_ids in menu["action_events"].items():
        assert len(event_ids) == 1
        event = menu["events"][event_ids[0]]
        assert event["first_actions"] == actions
        arguments = ["orb20", "tile30"] if "orb20" in action else ["orb21", "tile31"]
        assert event["bindings"][action] == arguments
        assert event["match_type"] == "exact_match"
        feedback = "Linked tile30 with orb20." if "orb20" in action else "Linked tile31 with orb21."
        control = events.EventPlanState(result, "inspect orb", "event_guarded")
        installed_event = control.install(
            plan(menu, first=action), current_observation="initial", current_actions=actions, menu=menu,
        )
        assert installed_event["event_arguments"] == arguments
        control.observe(action, feedback, native_valid=True)
        decision = control.decide(feedback, [NEXT])
        assert decision["commit"]
        assert decision["expected_feedback"] == feedback
    different_type = events.catalog_for_actions(result, ["link bead4 to plate5"])
    assert next(iter(different_type["events"].values()))["bindings"] == {
        "link bead4 to plate5": ["bead4", "plate5"],
    }


def test_repeated_argument_identity_is_not_conflated_with_two_different_instances():
    result = events.fit_feedback_catalog([
        record(0, action="pair orb1 with orb1", post="Paired orb1."),
        record(1, action="pair orb2 with orb2", post="Paired orb2."),
    ])
    menu = events.catalog_for_actions(result, ["pair orb9 with orb9", "pair orb9 with orb8"])
    assert menu["action_events"]["pair orb9 with orb9"]
    assert not menu["action_events"]["pair orb9 with orb8"]
    with pytest.raises(ValueError, match="inconsistent literal spellings"):
        events.fit_feedback_catalog([record(0, action="pair Orb1 with orb1")])


def test_case_is_preserved_while_whitespace_is_normalized_for_templates():
    result = events.fit_feedback_catalog([
        record(0, action="touch\t orb 1", post=" Touched\norb 1.\t"),
        record(1, action="touch  orb 2", post="Touched orb 2."),
    ])
    menu = events.catalog_for_actions(result, ["touch   orb 9", "Touch orb 9"])
    assert menu["action_events"]["touch   orb 9"]
    assert menu["action_events"]["Touch orb 9"] == []
    assert next(iter(menu["events"].values()))["bindings"]["touch   orb 9"] == ["orb 9"]
    first = "touch   orb 9"
    control, _, _ = installed(catalogue=result, first=first)
    feedback = "\nTouched \t orb 9.\n"
    control.observe(first, feedback, native_valid=True)
    assert control.decide(feedback, [NEXT])["commit"]


def test_unbound_feedback_reference_truncates_at_first_entity_not_in_the_action():
    result = novelty_catalog()
    assert set(result["entity_vocabulary"]) == {"orb", "case"}
    menu = events.catalog_for_actions(result, ["unseal case9"])
    event = next(iter(menu["events"].values()))
    assert event["match_type"] == "prefix_match"
    assert event["feedback_template"] == ["Unsealed ", {"slot": 0}, "; visible "]
    assert event["bindings"]["unseal case9"] == ["case9"]
    assert "orb2" not in event["description"] and "orb3" not in event["description"]
    assert result["counts"]["prefix_records"] == 2
    unknown = events.fit_feedback_catalog([
        record(0, action="wait", post="Visible unknown1 then later orb2"),
        record(1, action="wait", post="Visible unknown9 then later orb8"),
    ])
    assert unknown["entity_vocabulary"] == {}
    assert next(iter(unknown["events"].values()))["feedback_template"] == ["Visible "]


def test_empty_unbound_prefix_is_rejected_instead_of_matching_every_observation():
    result = events.fit_feedback_catalog([
        record(0, action="wait", post="orb1 appears"), record(1, action="wait", post="orb2 appears"),
    ])
    assert result["events"] == {}
    assert result["counts"]["rejected_records"] == 2
    assert all(row["reason"] == "empty_unbound_prefix" for row in result["rejections"])


def test_bound_reference_spelling_mismatch_is_reported_not_repaired():
    result = events.fit_feedback_catalog([
        record(0, action="touch Orb1", post="Touched orb1."),
        record(1, action="touch Orb2", post="Touched orb2."),
    ])
    assert result["events"] == {}
    assert all(row["reason"] == "bound_feedback_reference_spelling_mismatch" for row in result["rejections"])


def test_same_native_action_can_offer_multiple_observed_event_outcomes():
    result = events.fit_feedback_catalog([
        record(0), record(1),
        record(2, action="touch orb3", post="Unchanged orb3."),
        record(3, action="touch orb4", post="Unchanged orb4."),
    ])
    menu = events.catalog_for_actions(result, [FIRST, "touch  orb 9"])
    assert len(menu["events"]) == 2
    assert len(menu["action_events"][FIRST]) == 2
    assert {row["feedback_template"][0] for row in menu["events"].values()} == {"Touched ", "Unchanged "}
    assert all(row["first_actions"] == sorted([FIRST, "touch  orb 9"])
               for row in menu["events"].values())


def test_menu_limit_selects_group_support_then_id_not_repeated_source_count():
    rows = [record(index, group=f"a{index}", action="act", post="Outcome A") for index in range(3)]
    rows += [record(10 + index, group=f"b{index % 2}", action="act", post="Outcome B")
             for index in range(15)]
    rows += [record(40 + index, group=f"c{index}", action="act", post="Outcome C") for index in range(2)]
    result = events.fit_feedback_catalog(rows)
    menu = events.catalog_for_actions(result, ["act"], maximum_events=1)
    assert next(iter(menu["events"].values()))["feedback_template"] == ["Outcome A"]
    assert menu["counts"] == {
        "candidate_events": 3, "selected_events": 1, "truncated_events": 2,
        "candidate_bindings": 3, "selected_bindings": 1, "truncated_bindings": 2,
    }
    full = events.catalog_for_actions(result, ["act"], maximum_events=3)
    ordered = sorted(full["events"], key=lambda key: (
        -full["events"][key]["support_group_count"], full["events"][key]["catalog_event_id"],
    ))
    assert list(full["events"]) == ordered


@pytest.mark.parametrize("allow", [False, True])
def test_schema_has_native_enum_and_no_provider_unsupported_keywords(allow):
    result = catalog()
    native = [FIRST, "other native command"]
    menu = events.catalog_for_actions(result, native)
    schema = events.response_schema(native, menu, allow_continuation=allow)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"reasoning", "action", "continuation"}
    assert schema["properties"]["action"]["enum"] == sorted(native)
    assert schema["properties"]["action"]["maxLength"] == 160
    assert schema["properties"]["reasoning"]["maxLength"] == 1800
    if allow:
        continuation = schema["properties"]["continuation"]["anyOf"][1]
        assert continuation["additionalProperties"] is False
        assert set(continuation["required"]) == {"action", "after_event"}
        assert continuation["properties"]["after_event"]["enum"] == sorted(menu["events"])
    else:
        assert schema["properties"]["continuation"] == {"type": "null"}
    payload = json.dumps(schema)
    assert "uniqueItems" not in payload
    assert '"pattern"' not in payload
    assert schema == events.response_schema(list(reversed(native)), menu, allow_continuation=allow)


def test_empty_menu_allows_all_native_first_actions_but_only_null_continuation():
    menu = events.catalog_for_actions(catalog(), ["other native command"])
    assert not menu["events"]
    schema = events.response_schema(menu["native_actions"], menu, allow_continuation=True)
    assert schema["properties"]["action"]["enum"] == ["other native command"]
    assert schema["properties"]["continuation"] == {"type": "null"}
    value = plan(menu, first="other native command", continuation=None)
    assert parsed(menu, value) == value
    value["continuation"] = {"action": NEXT, "after_event": "guessed"}
    with pytest.raises(ValueError, match="must be null"):
        parsed(menu, value)


def test_event_modes_have_identical_generation_and_single_does_not_render_unused_catalogue():
    menu = events.catalog_for_actions(novelty_catalog(), [FIRST, "unseal case9"])
    prompts = [events.prompt_contract(mode, menu) for mode in EVENT_MODES]
    assert prompts[0].encode() == prompts[1].encode() == prompts[2].encode()
    assert "observed TRAIN feedback patterns, not guaranteed" in prompts[0]
    assert "BEFORE the continuation" in prompts[0]
    assert prompts[0].count("Events: id | action | match | feedback") == 1
    for identifier in menu["events"]:
        assert prompts[0].count(identifier) == 1
    for row in menu["events"].values():
        assert all(source not in prompts[0] for source in row["source_ids"])
    single = events.prompt_contract("structured_single", menu)
    assert all(identifier not in single for identifier in menu["events"])
    assert "Event menu" not in single
    assert '"continuation" must be null' in single
    assert "schema enum" in single


def test_parse_requires_known_event_allowed_for_the_exact_first_action():
    menu = events.catalog_for_actions(novelty_catalog(), [FIRST, "unseal case8", "other"])
    value = plan(menu)
    assert parsed(menu, value) == value
    assert NEXT not in menu["native_actions"]
    value["continuation"]["after_event"] = "me_guessed"
    with pytest.raises(ValueError, match="Unknown after_event"):
        parsed(menu, value)
    value = plan(menu, event_id=menu["action_events"]["unseal case8"][0])
    with pytest.raises(ValueError, match="not allowed"):
        parsed(menu, value)
    value = plan(menu)
    value["action"] = FIRST.upper()
    with pytest.raises(ValueError, match="First action"):
        parsed(menu, value)
    value["action"] = "other"
    with pytest.raises(ValueError, match="not allowed"):
        parsed(menu, value)


@pytest.mark.parametrize("text", [
    "", "  ", "null", "[]", "true", "{} {}", "```json\n{}\n```",
    '{"reasoning":"x","action":"a","action":"b","continuation":null}',
    '{"reasoning":NaN,"action":"a","continuation":null}',
    '{"reasoning":"x","action":"a","continuation":Infinity}',
    '{"reasoning":"x","action":"a","continuation":{"after_event":-1e9999}}',
])
def test_strict_json_rejects_empty_duplicate_nonfinite_or_ambiguous_model_payloads(text):
    menu = events.catalog_for_actions(catalog(), [FIRST])
    with pytest.raises(ValueError):
        events.parse_event_plan(text, native_actions=[FIRST], menu=menu, allow_continuation=True)


@pytest.mark.parametrize("mutation", [
    lambda p: p.pop("reasoning"), lambda p: p.pop("continuation"),
    lambda p: p.update(reasoning=True), lambda p: p.update(reasoning=" "),
    lambda p: p.update(reasoning="x" * 1801), lambda p: p.update(action=True),
    lambda p: p.update(action="x" * 161), lambda p: p.update(extra="forbidden"),
    lambda p: p.update(continuation=False), lambda p: p.update(continuation=[p["continuation"]]),
    lambda p: p["continuation"].update(action=" "), lambda p: p["continuation"].update(action="x" * 161),
    lambda p: p["continuation"].update(after_event=True),
    lambda p: p["continuation"].pop("after_event"),
    lambda p: p["continuation"].update(observation_contains_all=["invented"]),
    lambda p: p.update(tool_calls=[]), lambda p: p.update(reward=1),
])
def test_invalid_plan_fields_types_and_bounds_are_rejected(mutation):
    menu = events.catalog_for_actions(catalog(), [FIRST])
    value = plan(menu)
    mutation(value)
    with pytest.raises(ValueError):
        parsed(menu, value)


def test_single_mode_rejects_a_non_null_continuation_even_with_a_valid_event():
    menu = events.catalog_for_actions(catalog(), [FIRST])
    with pytest.raises(ValueError, match="must be null"):
        parsed(menu, allow=False)
    state = events.EventPlanState(catalog(), "inspect orb", "structured_single")
    with pytest.raises(ValueError, match="must be null"):
        state.install(plan(menu), current_observation="initial", current_actions=[FIRST], menu=menu)


@pytest.mark.parametrize("mode,commit", [
    ("event_guarded", False), ("event_no_novelty", True), ("event_unchecked", False),
])
def test_new_goal_instance_after_unsealing_is_a_separate_barrier(mode, commit):
    cat = novelty_catalog()
    first = "unseal case9"
    feedback = "Unsealed case9; visible orb8."
    _, result, _, _ = decided(mode, feedback, catalogue=cat, first=first, task="inspect orb")
    assert result["commit"] is commit
    assert result["novel_goal_entities"] == [{"type": "orb", "instance": "8"}]
    assert result["event_checked"] is (mode != "event_unchecked")
    assert result["event_matched"] is (True if mode != "event_unchecked" else None)
    assert result["novelty_checked"] is (mode != "event_no_novelty")
    assert result["novelty_passed"] is (False if mode != "event_no_novelty" else None)
    if not commit:
        assert result["reason"] == "new_goal_information"
        assert result["actor_required"]


@pytest.mark.parametrize("mode", EVENT_MODES)
def test_bound_goal_entity_reappearing_in_feedback_is_not_new_information(mode):
    _, result, _, _ = decided(mode)
    assert result["commit"]
    assert result["novel_goal_entities"] == []
    assert result["goal_types"] == ["orb"]
    assert result["native_member"] is True
    assert result["guard_checked"] is True and result["guard_passed"] is True


@pytest.mark.parametrize("prior,feedback", [
    ("Already visible orb8", "Unsealed case9; visible orb8."),
    ("initial", "Unsealed case9; visible tile8."),
    ("Already visible Orb8", "Unsealed case9; visible orb8."),
])
def test_preobserved_entities_and_unrelated_new_types_do_not_trip_goal_barrier(prior, feedback):
    _, result, _, _ = decided(
        "event_guarded", feedback, catalogue=novelty_catalog(),
        first="unseal case9", observation=prior, task="inspect orb",
    )
    assert result["commit"]
    assert result["novel_goal_entities"] == []


def test_goal_type_vocabulary_is_train_only_and_exact_task_words_not_substrings():
    cat = novelty_catalog()
    control, _, _ = installed(
        "event_guarded", catalogue=cat, first="unseal case9", task="inspect orbs",
    )
    control.observe("unseal case9", "Unsealed case9; visible orb8.", native_valid=True)
    decision = control.decide("Unsealed case9; visible orb8.", [NEXT])
    assert decision["goal_types"] == []
    assert decision["commit"]
    cat = events.fit_feedback_catalog([
        record(0, action="wait", post="Visible mystery1"),
        record(1, action="wait", post="Visible mystery2"),
    ])
    assert "mystery" not in cat["entity_vocabulary"]


@pytest.mark.parametrize("mode,commit", [
    ("event_guarded", False), ("event_no_novelty", False), ("event_unchecked", True),
])
def test_event_unchecked_disables_only_event_matching_when_no_novel_goal_fact_exists(mode, commit):
    _, result, _, _ = decided(mode, "different actual feedback")
    assert result["commit"] is commit
    assert result["event_checked"] is (mode != "event_unchecked")
    assert result["event_matched"] is (False if mode != "event_unchecked" else None)
    assert result["novelty_checked"] is (mode != "event_no_novelty")
    assert result["novel_goal_entities"] == []
    if not commit:
        assert result["reason"] == "feedback_event_mismatch"


def test_unchecked_event_mismatch_does_not_disable_novelty_barrier():
    _, decision, _, _ = decided("event_unchecked", "unexpected orb8 appears")
    assert decision["event_matched"] is None
    assert decision["novelty_passed"] is False
    assert decision["reason"] == "new_goal_information"
    assert not decision["commit"]


@pytest.mark.parametrize("feedback,commit", [
    ("Touched orb 9.", True), (" Touched\torb 9.\n", True),
    ("touched orb 9.", False), ("Touched orb 9. More text", False), ("Touched orb 99.", False),
])
def test_exact_matching_preserves_case_and_instance_identity_not_substring_matches(feedback, commit):
    _, result, _, _ = decided("event_guarded", feedback)
    assert result["commit"] is commit
    assert result["match_type"] == "exact_match"
    assert result["event_matched"] is commit


@pytest.mark.parametrize("feedback,commit", [
    ("Unsealed case9; visible orb8.", True),
    ("Unsealed case9; visible orb88; unrelated later text.", True),
    ("Unsealed case9; visiblesomething", False),
    ("unsealed case9; visible orb8.", False),
])
def test_prefix_matching_retains_literal_boundary_and_does_not_guess_suffixes(feedback, commit):
    _, result, _, _ = decided(
        "event_no_novelty", feedback, catalogue=novelty_catalog(), first="unseal case9",
    )
    assert result["commit"] is commit
    assert result["match_type"] == "prefix_match"
    assert result["expected_feedback"] == "Unsealed case9; visible "


@pytest.mark.parametrize("mode", EVENT_MODES)
def test_current_native_membership_is_mandatory_in_every_event_mode(mode):
    control, result, _, _ = decided(mode, native=["not the continuation"])
    assert result["reason"] == "continuation_not_native"
    assert result["action"] is None
    assert result["candidate_action"] == NEXT
    assert result["guard_checked"] is False and result["guard_passed"] is None
    assert result["actor_required"]
    with pytest.raises(ValueError, match="pending"):
        control.decide(OBSERVATION, [NEXT])
    with pytest.raises(ValueError, match="No pending"):
        control.observe(NEXT, "not actually executed", native_valid=True)


@pytest.mark.parametrize("mode", EVENT_MODES)
def test_only_recorded_first_action_feedback_can_be_used_to_decide(mode):
    control, _, _ = installed(mode)
    with pytest.raises(ValueError, match="pending"):
        control.decide(OBSERVATION, [NEXT])
    control.observe(FIRST, "actual different feedback", native_valid=True)
    with pytest.raises(ValueError, match="actual first-action feedback"):
        control.decide(OBSERVATION, [NEXT])
    with pytest.raises(ValueError, match="actual first-action feedback"):
        control.decide(" actual different feedback ", [NEXT])
    result = control.decide("actual different feedback", [NEXT])
    assert result["commit"] is (mode == "event_unchecked")


@pytest.mark.parametrize("mode", events.MODES)
def test_one_continuation_maximum_with_strict_phase_and_actor_boundaries(mode):
    control, value, menu = installed(mode)
    with pytest.raises(ValueError, match="outstanding"):
        control.install(value, current_observation="initial", current_actions=[FIRST], menu=menu)
    with pytest.raises(ValueError, match="differs"):
        control.observe("wrong action", OBSERVATION, native_valid=True)
    first = control.observe(FIRST, OBSERVATION, native_valid=True)
    assert first["completed_actions"] == 1
    with pytest.raises(ValueError, match="No pending"):
        control.observe(FIRST, OBSERVATION, native_valid=True)
    result = control.decide(OBSERVATION, [NEXT])
    assert result["commit"] is (mode != "structured_single")
    if mode != "structured_single":
        with pytest.raises(ValueError, match="pending"):
            control.decide(OBSERVATION, [NEXT])
        with pytest.raises(ValueError, match="outstanding"):
            control.install(value, current_observation=OBSERVATION, current_actions=[FIRST], menu=menu)
        feedback = control.observe(NEXT, "final actual feedback", native_valid=True)
        assert feedback["actor_required"] and feedback["completed_actions"] == 2
        assert not feedback["continuation_queued"]
        boundary = control.decide("final actual feedback", [NEXT, FIRST])
        assert not boundary["commit"]
        assert boundary["reason"] == "actor_boundary_required"
    else:
        assert result["reason"] == "actor_boundary_required"
    with pytest.raises(ValueError, match="pending"):
        control.decide("ignored new query", [FIRST])
    control.install(value, current_observation="current", current_actions=[FIRST], menu=menu)
    assert control.observe(FIRST, OBSERVATION, native_valid=True)["action_role"] == "actor"


def test_owned_plans_catalogs_menus_and_audits_cannot_mutate_installed_state():
    cat = catalog()
    menu = events.catalog_for_actions(cat, [FIRST])
    value = parsed(menu)
    control = events.EventPlanState(cat, "inspect orb", "event_guarded")
    original = deepcopy((cat, menu, value))
    install = control.install(value, current_observation="initial", current_actions=[FIRST], menu=menu)
    value["action"] = "changed"
    value["continuation"]["action"] = "changed next"
    menu["events"].clear()
    cat["events"].clear()
    control.observe(FIRST, OBSERVATION, native_valid=True)
    result = control.decide(OBSERVATION, [NEXT])
    assert result["commit"] and result["action"] == NEXT
    assert result["plan_sha256"] == install["plan_sha256"]
    result["novel_goal_entities"].append({"type": "invented", "instance": "1"})
    assert control.observe(NEXT, "observed final", native_valid=True)["actor_required"]
    assert original[0]["events"] and original[1]["events"]


def test_state_rejects_menu_from_another_catalogue_and_never_installs_guessed_feedback():
    cat = catalog()
    menu = events.catalog_for_actions(cat, [FIRST])
    other = events.fit_feedback_catalog([record(0, post="Other outcome"), record(1, post="Other outcome")])
    control = events.EventPlanState(other, "inspect orb", "event_guarded")
    with pytest.raises(ValueError, match="another catalogue"):
        control.install(plan(menu), current_observation="initial", current_actions=[FIRST], menu=menu)
    event = next(iter(menu["events"].values()))
    event["feedback_template"] = ["Invented observation"]
    with pytest.raises(ValueError):
        events.EventPlanState(cat, "inspect orb", "event_guarded").install(
            plan(menu), current_observation="initial", current_actions=[FIRST], menu=menu,
        )


def test_catalogue_and_menu_ordering_serialization_and_no_input_mutation():
    rows = [record(0), record(1), record(2, action="touch tile3", post="Touched tile3.")]
    original = deepcopy(rows)
    first = events.fit_feedback_catalog(rows)
    assert rows == original
    second = events.fit_feedback_catalog(list(reversed(deepcopy(rows))))
    assert first == second == json.loads(json.dumps(first, allow_nan=False))
    a = events.catalog_for_actions(first, [FIRST, "touch tile4"])
    b = events.catalog_for_actions(second, ["touch tile4", FIRST])
    assert a == b == json.loads(json.dumps(a, allow_nan=False))
    assert events.prompt_contract("event_guarded", a) == events.prompt_contract("event_guarded", b)


def test_catalogue_minimum_groups_is_enforced_without_manufacturing_an_event():
    rows = [record(0), record(1)]
    rejected = events.fit_feedback_catalog(rows, minimum_groups=3)
    assert rejected["events"] == {}
    assert rejected["rejections"][0]["reason"] == "insufficient_independent_groups"
    accepted = events.fit_feedback_catalog(rows + [record(2)], minimum_groups=3)
    assert len(accepted["events"]) == 1
    assert next(iter(accepted["events"].values()))["support_group_count"] == 3


@pytest.mark.parametrize("mode", events.MODES)
def test_empty_catalogue_supports_only_actor_actions_without_invented_events(mode):
    cat = events.fit_feedback_catalog([])
    menu = events.catalog_for_actions(cat, [FIRST])
    value = plan(menu, continuation=None)
    control = events.EventPlanState(cat, "inspect orb", mode)
    assert events.response_schema([FIRST], menu, allow_continuation=mode != "structured_single")[
        "properties"
    ]["continuation"] == {"type": "null"}
    control.install(value, current_observation="initial", current_actions=[FIRST], menu=menu)
    control.observe(FIRST, "actual feedback", native_valid=True)
    assert not control.decide("actual feedback", [NEXT])["commit"]
    assert '"continuation" must be null' in events.prompt_contract(mode, menu)


@pytest.mark.parametrize("actual", ["", "different"])
def test_empty_exact_feedback_is_an_exact_observed_pattern_not_a_wildcard(actual):
    cat = events.fit_feedback_catalog([record(0, post=""), record(1, post="")])
    _, result, _, _ = decided("event_guarded", actual, catalogue=cat)
    assert result["expected_feedback"] == ""
    assert result["match_type"] == "exact_match"
    assert result["commit"] is (actual == "")


def test_reasoning_task_or_event_description_never_supplies_novel_entity_evidence():
    cat = novelty_catalog()
    first = "unseal case9"
    menu = events.catalog_for_actions(cat, [first])
    value = plan(menu, first=first)
    value["reasoning"] = "I imagine orb8 is visible."
    control = events.EventPlanState(cat, "inspect orb and anticipate orb8", "event_guarded")
    control.install(value, current_observation="initial", current_actions=[first], menu=menu)
    feedback = "Unsealed case9; visible case8."
    control.observe(first, feedback, native_valid=True)
    result = control.decide(feedback, [NEXT])
    assert result["novel_goal_entities"] == []
    assert result["commit"]


@pytest.mark.parametrize("stage", ["first", "continuation"])
def test_native_invalid_actual_feedback_consumes_the_event_plan(stage):
    control, _, _ = installed()
    action = FIRST
    if stage == "continuation":
        control.observe(FIRST, OBSERVATION, native_valid=True)
        assert control.decide(OBSERVATION, [NEXT])["commit"]
        action = NEXT
    observed = control.observe(action, "native rejection", native_valid=False)
    assert observed["actor_required"]
    assert not observed["continuation_queued"]
    assert observed["reason"] == (
        "actor_action_not_native" if stage == "first" else "continuation_action_not_native"
    )
    assert not control.decide("native rejection", [FIRST, NEXT])["commit"]


@pytest.mark.parametrize("mutation", [
    lambda menu: menu.update(schema_version="unknown"),
    lambda menu: menu.update(catalog_sha256="unverified"),
    lambda menu: menu.update(native_actions=["different native set"]),
    lambda menu: menu.update(maximum_events=True),
    lambda menu: menu["action_events"].update({FIRST: []}),
    lambda menu: next(iter(menu["events"].values())).update(first_actions=["invented"]),
    lambda menu: next(iter(menu["events"].values())).update(support_group_count=True),
    lambda menu: next(iter(menu["events"].values())).update(source_count=0),
    lambda menu: next(iter(menu["events"].values())).update(description="invented feedback"),
    lambda menu: menu["counts"].update(selected_events=True),
])
def test_unknown_or_inconsistent_menu_bindings_fail_before_installation(mutation):
    menu = events.catalog_for_actions(catalog(), [FIRST])
    value = plan(menu)
    mutation(menu)
    with pytest.raises(ValueError):
        events.parse_event_plan(
            json.dumps(value), native_actions=[FIRST], menu=menu, allow_continuation=True,
        )
    control = events.EventPlanState(catalog(), "inspect orb", "event_guarded")
    with pytest.raises(ValueError):
        control.install(value, current_observation="initial", current_actions=[FIRST], menu=menu)
    assert control._phase == "empty"
    assert control._plan is None


@pytest.mark.parametrize("kwargs", [
    {"minimum_groups": True}, {"minimum_groups": 1}, {"minimum_groups": 2.0},
    {"excluded_groups": []}, {"excluded_groups": "g"}, {"excluded_groups": {True}},
])
def test_catalogue_parameters_are_strict(kwargs):
    with pytest.raises(ValueError):
        events.fit_feedback_catalog([record(0), record(1)], **kwargs)


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(source_id=True), lambda row: row.update(source_id=""),
    lambda row: row.update(task_id=None), lambda row: row.update(group_id=False),
    lambda row: row.update(action=""), lambda row: row.update(action=1),
    lambda row: row.update(post_observation=False), lambda row: row.pop("post_observation"),
    lambda row: row.update(outcome=1), lambda row: row.update(task_description="not permitted"),
])
def test_source_records_have_only_public_action_effect_and_provenance_fields(mutation):
    value = record(0)
    mutation(value)
    with pytest.raises(ValueError):
        events.fit_feedback_catalog([value])


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, None])
def test_event_cap_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="maximum_events"):
        events.catalog_for_actions(catalog(), [FIRST], maximum_events=value)


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_boolean_flags_are_not_integer_or_string_coercions(value):
    menu = events.catalog_for_actions(catalog(), [FIRST])
    with pytest.raises(ValueError, match="allow_continuation"):
        events.response_schema([FIRST], menu, allow_continuation=value)
    with pytest.raises(ValueError, match="allow_continuation"):
        parsed(menu, allow=value)
    control, _, _ = installed()
    with pytest.raises(ValueError, match="native_valid"):
        control.observe(FIRST, OBSERVATION, native_valid=value)
    assert control.observe(FIRST, OBSERVATION, native_valid=True)["completed_actions"] == 1


@pytest.mark.parametrize("mode", [True, None, "unknown", [], "guarded_plan"])
def test_mode_validation_is_strict(mode):
    with pytest.raises(ValueError, match="mode"):
        events.EventPlanState(catalog(), "inspect orb", mode)
    with pytest.raises(ValueError, match="mode"):
        events.prompt_contract(mode, events.catalog_for_actions(catalog(), [FIRST]))


@pytest.mark.parametrize("mutation", [
    lambda cat: cat.update(schema_version="unknown"),
    lambda cat: cat.update(minimum_groups=True),
    lambda cat: cat.update(minimum_groups=1),
    lambda cat: next(iter(cat["events"].values())).update(argument_count=True),
    lambda cat: next(iter(cat["events"].values())).update(support_group_count=True),
    lambda cat: next(iter(cat["events"].values())).update(support_groups=["one"]),
    lambda cat: next(iter(cat["events"].values())).update(feedback_template=["invented"]),
    lambda cat: next(iter(cat["events"].values()))["action_shape"][1].update(slot=True),
    lambda cat: cat.update(excluded_groups=["group-0"]),
])
def test_invalid_catalogue_metadata_templates_and_provenance_are_not_trusted(mutation):
    cat = catalog()
    mutation(cat)
    with pytest.raises(ValueError):
        events.catalog_for_actions(cat, [FIRST])
    with pytest.raises(ValueError):
        events.EventPlanState(cat, "inspect orb", "event_guarded")


_AUDIT = {
    "schema_version", "mode", "phase", "catalog_sha256", "plan_sha256",
    "menu_sha256", "completed_actions", "actor_required", "boundary_reason", "goal_types",
}
_DECISION = _AUDIT | {
    "commit", "action", "candidate_action", "reason", "guard_checked", "guard_passed", "native_member",
    "event_id", "catalog_event_id", "match_type", "expected_feedback", "event_checked", "event_matched",
    "novelty_checked", "novelty_passed", "novel_goal_entities",
    "event_arguments",
}


@pytest.mark.parametrize("mode", events.MODES)
def test_exact_public_output_keys_and_json_round_trips(mode):
    cat = catalog()
    menu = events.catalog_for_actions(cat, [FIRST])
    assert set(cat) == {
        "schema_version", "minimum_groups", "excluded_groups", "entity_vocabulary",
        "events", "counts", "rejections", "semantics",
    }
    assert set(menu) == {
        "schema_version", "catalog_sha256", "native_actions", "maximum_events",
        "events", "action_events", "counts",
    }
    control = events.EventPlanState(cat, "inspect orb", mode)
    initial = control.decide("initial", [FIRST])
    assert set(initial) == _DECISION
    value = plan(menu, continuation=None if mode == "structured_single" else NEXT)
    install = control.install(value, current_observation="initial", current_actions=[FIRST], menu=menu)
    assert set(install) == _AUDIT | {
        "reason", "action", "event_id", "catalog_event_id", "event_arguments", "continuation_queued",
    }
    observed = control.observe(FIRST, OBSERVATION, native_valid=True)
    assert set(observed) == _AUDIT | {
        "reason", "action", "observation", "native_valid", "action_role", "event_id",
        "catalog_event_id", "event_arguments", "continuation_queued",
    }
    decision = control.decide(OBSERVATION, [NEXT])
    assert set(decision) == _DECISION
    assert decision["commit"] is (mode != "structured_single")
    for result in (initial, install, observed, decision):
        assert result == json.loads(json.dumps(result, allow_nan=False))


@pytest.mark.parametrize("mode", EVENT_MODES)
def test_many_argument_bindings_share_one_event_and_one_identical_prompt_template(mode):
    cat = catalog()
    actions = [f"touch orb {number}" for number in range(1, 81)]
    small = events.catalog_for_actions(cat, actions[:1], maximum_events=1)
    large = events.catalog_for_actions(cat, actions, maximum_events=1)
    assert set(small["events"]) == set(large["events"]) == {"e0"}
    identifier = next(iter(large["events"]))
    event = large["events"][identifier]
    assert event["catalog_event_id"] == next(iter(cat["events"]))
    assert event["source_count"] == event["support_group_count"] == 2
    assert event["first_actions"] == sorted(actions)
    assert event["bindings"] == {action: [action.removeprefix("touch ")] for action in sorted(actions)}
    assert all(ids == [identifier] for ids in large["action_events"].values())
    assert large["counts"] == {
        "candidate_events": 1, "selected_events": 1, "truncated_events": 0,
        "candidate_bindings": 80, "selected_bindings": 80, "truncated_bindings": 0,
    }
    prompt = events.prompt_contract(mode, large)
    assert prompt == events.prompt_contract(mode, small)
    assert prompt.count(identifier) == 1
    assert all(action not in prompt for action in actions)
    assert '"bindings"' not in prompt and '"first_actions"' not in prompt
    assert "$0/$1" in prompt
    displayed = prompt.split("Events: id | action | match | feedback\n", 1)[1]
    assert displayed == "e0 | touch $0 | exact | Touched $0."
    assert event["catalog_event_id"] not in prompt
    schema = events.response_schema(actions, large, allow_continuation=True)
    assert schema["properties"]["continuation"]["anyOf"][1]["properties"]["after_event"]["enum"] == [
        identifier,
    ]
    assert schema["properties"]["action"]["enum"] == sorted(actions)


def test_template_cap_counts_outcomes_not_current_entities_or_bindings():
    cat = events.fit_feedback_catalog([
        record(0), record(1),
        record(2, action="touch orb 3", post="Unchanged orb 3."),
        record(3, action="touch orb 4", post="Unchanged orb 4."),
    ])
    actions = [f"touch orb {number}" for number in range(50)]
    menu = events.catalog_for_actions(cat, actions, maximum_events=1)
    assert len(menu["events"]) == 1
    assert menu["counts"] == {
        "candidate_events": 2, "selected_events": 1, "truncated_events": 1,
        "candidate_bindings": 100, "selected_bindings": 50, "truncated_bindings": 50,
    }
    assert all(len(ids) == 1 for ids in menu["action_events"].values())
    assert next(iter(menu["events"])) == "e0"
    assert menu["events"]["e0"]["catalog_event_id"] == min(cat["events"])


@pytest.mark.parametrize("mode", EVENT_MODES)
@pytest.mark.parametrize("chosen", [FIRST, "touch orb 8"])
def test_runtime_binds_the_chosen_first_action_not_the_first_menu_binding(mode, chosen):
    cat = catalog()
    actions = [FIRST, "touch orb 8"]
    menu = events.catalog_for_actions(cat, actions)
    shared_event = next(iter(menu["events"]))
    assert menu["action_events"][FIRST] == menu["action_events"]["touch orb 8"] == [shared_event]
    value = plan(menu, first=chosen, event_id=shared_event)
    assert parsed(menu, value) == value
    control = events.EventPlanState(cat, "inspect orb", mode)
    output = control.install(value, current_observation="initial", current_actions=actions, menu=menu)
    argument = chosen.removeprefix("touch ")
    assert output["event_arguments"] == [argument]
    feedback = f"Touched {argument}."
    observed = control.observe(chosen, feedback, native_valid=True)
    assert observed["event_arguments"] == [argument]
    result = control.decide(feedback, [NEXT])
    assert result["commit"]
    assert result["event_id"] == shared_event
    assert result["catalog_event_id"] == menu["events"][shared_event]["catalog_event_id"]
    assert result["event_arguments"] == [argument]
    assert result["expected_feedback"] == feedback
    assert result["novel_goal_entities"] == []


def test_shared_event_cannot_match_feedback_for_a_different_native_argument_binding():
    cat = catalog()
    actions = [FIRST, "touch orb 8"]
    menu = events.catalog_for_actions(cat, actions)
    control = events.EventPlanState(cat, "inspect orb", "event_no_novelty")
    control.install(
        plan(menu, first="touch orb 8"), current_observation="initial", current_actions=actions, menu=menu,
    )
    control.observe("touch orb 8", OBSERVATION, native_valid=True)
    result = control.decide(OBSERVATION, [NEXT])
    assert not result["commit"]
    assert result["event_matched"] is False
    assert result["reason"] == "feedback_event_mismatch"
    assert result["expected_feedback"] == "Touched orb 8."
    assert result["event_arguments"] == ["orb 8"]


def test_single_prompt_stays_uninflated_and_shares_the_same_native_enum():
    cat = catalog()
    one = events.catalog_for_actions(cat, [FIRST])
    actions = [f"touch orb {number}" for number in range(80)]
    many = events.catalog_for_actions(cat, actions)
    single = events.prompt_contract("structured_single", many)
    assert single == events.prompt_contract("structured_single", one)
    assert "feedback_template" not in single
    assert "Event menu" not in single and '"slot"' not in single
    assert all(identifier not in single for identifier in many["events"])
    single_schema = events.response_schema(actions, many, allow_continuation=False)
    plan_schema = events.response_schema(actions, many, allow_continuation=True)
    assert single_schema["properties"]["action"] == plan_schema["properties"]["action"]
    assert single_schema["properties"]["continuation"] == {"type": "null"}


@pytest.mark.parametrize("mutation", [
    lambda event: event["bindings"].update({FIRST: ["orb 8"]}),
    lambda event: event["bindings"].update({FIRST: [True]}),
    lambda event: event["bindings"].update({FIRST: []}),
    lambda event: event["bindings"].clear(),
    lambda event: event["bindings"].update({"touch orb 8": ["orb 8"]}),
    lambda event: event.update(argument_count=True),
    lambda event: event.update(catalog_event_id="invented"),
    lambda event: event["feedback_template"][1].update(slot=99),
])
def test_altered_argument_binding_maps_are_rejected_not_guessed_or_repaired(mutation):
    cat = catalog()
    menu = events.catalog_for_actions(cat, [FIRST])
    value = plan(menu)
    mutation(next(iter(menu["events"].values())))
    with pytest.raises(ValueError):
        events.parse_event_plan(json.dumps(value), native_actions=[FIRST], menu=menu, allow_continuation=True)
    control = events.EventPlanState(cat, "inspect orb", "event_guarded")
    with pytest.raises(ValueError):
        control.install(value, current_observation="initial", current_actions=[FIRST], menu=menu)
    assert control._phase == "empty"


def test_installed_binding_and_returned_argument_audits_are_owned_copies():
    cat = catalog()
    actions = [FIRST, "touch orb 8"]
    menu = events.catalog_for_actions(cat, actions)
    control = events.EventPlanState(cat, "inspect orb", "event_guarded")
    value = plan(menu, first=FIRST)
    installed_event = control.install(value, current_observation="initial", current_actions=actions, menu=menu)
    installed_event["event_arguments"].clear()
    event = next(iter(menu["events"].values()))
    event["bindings"][FIRST][:] = ["orb 100"]
    event["feedback_template"][0] = "Invented "
    feedback = control.observe(FIRST, OBSERVATION, native_valid=True)
    assert feedback["event_arguments"] == ["orb 9"]
    feedback["event_arguments"].append("invented")
    decision = control.decide(OBSERVATION, [NEXT])
    assert decision["commit"]
    assert decision["event_arguments"] == ["orb 9"]
    assert decision["expected_feedback"] == OBSERVATION
    decision["event_arguments"].clear()
    final = control.observe(NEXT, "actual feedback", native_valid=True)
    assert final["event_arguments"] == ["orb 9"]


def test_factored_menu_versions_and_event_record_shape_are_explicit():
    cat = catalog()
    menu = events.catalog_for_actions(cat, [FIRST])
    assert cat["schema_version"] == "chisel-observed-feedback-catalog/v1"
    assert menu["schema_version"] == "chisel-grounded-feedback-event-menu/v3"
    assert events.STATE_SCHEMA == "chisel-observed-feedback-event-plan-state/v3"
    assert set(next(iter(menu["events"].values()))) == {
        "catalog_event_id", "action_shape", "argument_count", "feedback_template",
        "match_type", "description", "first_actions", "bindings",
        "source_count", "source_ids", "support_groups", "support_group_count",
    }
    menu["schema_version"] = "chisel-grounded-feedback-event-menu/v2"
    with pytest.raises(ValueError):
        events.response_schema([FIRST], menu, allow_continuation=True)


@pytest.mark.parametrize("mode", events.MODES)
@pytest.mark.parametrize("character", ["r", "\u03b1"])
def test_all_new_event_modes_share_the_exact_1800_character_reasoning_bound(mode, character):
    cat = catalog()
    menu = events.catalog_for_actions(cat, [FIRST])
    allow = mode != "structured_single"
    value = plan(menu, continuation=NEXT if allow else None)
    value["reasoning"] = character * 1800
    schema = events.response_schema([FIRST], menu, allow_continuation=allow)
    assert schema["properties"]["reasoning"] == {
        "type": "string", "minLength": 1, "maxLength": 1800,
    }
    assert schema["properties"]["action"]["maxLength"] == 160
    if allow:
        continuation = schema["properties"]["continuation"]["anyOf"][1]["properties"]
        assert continuation["action"]["maxLength"] == continuation["after_event"]["maxLength"] == 160
    prompt = events.prompt_contract(mode, menu)
    assert "reasoning: nonempty, <=1800 characters" in prompt
    assert "6000" not in prompt
    assert parsed(menu, value, allow=allow) == value
    state = events.EventPlanState(cat, "inspect orb", mode)
    assert state.install(
        value, current_observation="initial", current_actions=[FIRST], menu=menu,
    )["reason"] == "actor_action_pending"
    value["reasoning"] += character
    with pytest.raises(ValueError, match="reasoning exceeds 1800 characters"):
        parsed(menu, value, allow=allow)
    untouched = events.EventPlanState(cat, "inspect orb", mode)
    with pytest.raises(ValueError, match="reasoning exceeds 1800 characters"):
        untouched.install(value, current_observation="initial", current_actions=[FIRST], menu=menu)
    assert untouched._phase == "empty"
    assert untouched._plan is None


@pytest.mark.parametrize("mode", events.MODES)
@pytest.mark.parametrize("truncation", ["reasoning", "outer_brace", "tail"])
def test_remaining_truncated_event_responses_are_explicit_errors_not_partial_plans(mode, truncation):
    menu = events.catalog_for_actions(catalog(), [FIRST])
    allow = mode != "structured_single"
    value = plan(menu, continuation=NEXT if allow else None)
    value["reasoning"] = "r" * 1800
    encoded = json.dumps(value)
    truncated = (
        encoded[:100] if truncation == "reasoning" else
        encoded[:-1] if truncation == "outer_brace" else encoded[:-10]
    )
    with pytest.raises(ValueError, match="Expected exactly one JSON object") as error:
        events.parse_event_plan(truncated, native_actions=[FIRST], menu=menu, allow_continuation=allow)
    assert isinstance(error.value.__cause__, json.JSONDecodeError)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()


def test_catalogue_fitting_and_enum_single_generation_are_exactly_unchanged():
    cat = novelty_catalog()
    assert canonical_hash(cat) == "aa63220a9526f5f58e450067094507a6026c582332098b824ada0a5fd0aa2d5b"
    menu = events.catalog_for_actions(cat, [FIRST, "unseal case9", "other native command"])
    schema = events.response_schema(menu["native_actions"], menu, allow_continuation=False)
    assert canonical_hash(schema) == "9325e100b3717e5a94f50f4b080ebdcb71144f0f618033b566a20f777ba08a73"
    prompt = events.prompt_contract("structured_single", menu)
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "db10d7934c835d3f9402a12471de5af661d313f0b970deb8c5f3d2c6be67c40e"
    )
    assert len(prompt) == 462
    assert [row["catalog_event_id"] for row in menu["events"].values()] == [
        "ev_0f4b8cd410ba045086d8819c68319d3c193836cff4699448973ae1371efacd86",
        "ev_aba6c96a024a31b2aa06d292502d88d57b9b38a32ec23ec335bcbc599d74a7f4",
    ]


def nine_event_fixture():
    rows, actions = [], []
    for index, letter in enumerate("ABCDEFGHI"):
        actions.append(f"cmd{letter} orb9")
        for offset in range(2):
            number = offset + 1
            rows.append(record(
                index * 2 + offset, group=f"g{offset}", action=f"cmd{letter} orb{number}",
                post=f"Feedback {letter}: orb{number}.",
            ))
    cat = events.fit_feedback_catalog(rows)
    return cat, events.catalog_for_actions(cat, actions)


def test_aliases_preserve_selected_catalogue_ids_ranking_bindings_and_provenance():
    cat, full = nine_event_fixture()
    expected = sorted(cat["events"], key=lambda identifier: (
        -cat["events"][identifier]["support_group_count"], identifier,
    ))
    assert list(full["events"]) == [f"e{index}" for index in range(9)]
    assert [event["catalog_event_id"] for event in full["events"].values()] == expected
    for alias, event in full["events"].items():
        original = cat["events"][event["catalog_event_id"]]
        for key in ("action_shape", "argument_count", "feedback_template", "match_type",
                    "support_groups", "source_count", "support_group_count"):
            assert event[key] == original[key]
        assert event["source_ids"] == sorted(row["source_id"] for row in original["sources"])
        for action in event["first_actions"]:
            assert full["action_events"][action] == [alias]
            assert event["bindings"][action] == ["orb9"]
    limited = events.catalog_for_actions(cat, list(reversed(full["native_actions"])), maximum_events=4)
    assert [row["catalog_event_id"] for row in limited["events"].values()] == expected[:4]
    assert full["catalog_sha256"] == limited["catalog_sha256"] == canonical_hash(cat)


def test_compact_nine_event_rendering_is_materially_smaller_than_the_previous_json_menu():
    cat, menu = nine_event_fixture()
    previous = json.dumps({
        event["catalog_event_id"]: {
            "action_shape": event["action_shape"], "match_type": event["match_type"],
            "feedback_template": event["feedback_template"],
        }
        for event in menu["events"].values()
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    prompt = events.prompt_contract("event_guarded", menu)
    compact = prompt.split("Events: id | action | match | feedback\n", 1)[1]
    assert len(compact.splitlines()) == len(menu["events"]) == 9
    assert len(compact) < 0.4 * len(previous)
    assert all(identifier not in prompt for identifier in cat["events"])
    assert all(event["description"] not in prompt for event in menu["events"].values())
    assert all(event["source_ids"][0] not in prompt for event in menu["events"].values())
    for index, line in enumerate(compact.splitlines()):
        alias, command, match, feedback = line.split(" | ")
        assert alias == f"e{index}"
        assert "$0" in command and "$0" in feedback
        assert match == "exact"
    two_event_menu = events.catalog_for_actions(novelty_catalog(), [FIRST, "unseal case9", "other native command"])
    assert len(events.prompt_contract("event_guarded", two_event_menu)) < 0.6 * 2009


@pytest.mark.parametrize("mode", EVENT_MODES)
def test_prompt_offers_a_plausible_branch_without_predicting_it_and_uses_the_same_schema(mode):
    _, menu = nine_event_fixture()
    prompt = events.prompt_contract(mode, menu)
    assert "PLAUSIBLE event branch without predicting that branch will occur" in prompt
    assert "mismatch returns to actor" in prompt
    assert "Use null if no suitable branch" in prompt
    assert "after one continuation return to actor" in prompt
    assert "reasoning: nonempty, <=1800 characters" in prompt
    assert prompt == events.prompt_contract("event_guarded", menu)
    schema = events.response_schema(menu["native_actions"], menu, allow_continuation=True)
    assert schema["properties"]["continuation"]["anyOf"][1]["properties"]["after_event"]["enum"] == [
        f"e{index}" for index in range(9)
    ]
    assert schema["properties"]["action"]["maxLength"] == 160
    assert schema["properties"]["reasoning"]["maxLength"] == 1800
    assert "uniqueItems" not in json.dumps(schema) and '"pattern"' not in json.dumps(schema)


def decode_compact_template(text):
    parts, literal, index = [], [], 0
    escapes = {"\\": "\\", "|": "|", "$": "$", "n": "\n", "r": "\r", "t": "\t"}
    while index < len(text):
        if text[index] == "\\":
            code = text[index + 1]
            if code == "u":
                literal.append(chr(int(text[index + 2:index + 6], 16)))
                index += 6
            else:
                literal.append(escapes[code])
                index += 2
        elif text[index] == "$":
            end = index + 1
            while end < len(text) and text[end] in "0123456789":
                end += 1
            assert end > index + 1
            parts.extend(["".join(literal), {"slot": int(text[index + 1:end])}])
            literal = []
            index = end
        else:
            literal.append(text[index])
            index += 1
    return parts + ["".join(literal)]


@pytest.mark.parametrize("template", [
    ["touch ", {"slot": 0}, " then ", {"slot": 0}, ""],
    ["literal $0 | \\ value\nnew\rline\t", {"slot": 0}, "1 next"],
    [" ", {"slot": 1}, " "],
    ["prefix "],
    ["\u2028\u2029\x85\x00", {"slot": 0}, "\\n $1 |"],
    ["exact empty? ", {"slot": 12}, "3"],
    [""],
])
def test_compact_template_escaping_is_one_line_and_unambiguous(template):
    compact = events._compact_template(template)
    assert "\n" not in compact and "\r" not in compact and "\t" not in compact
    assert "\u2028" not in compact and "\u2029" not in compact
    assert decode_compact_template(compact) == template


def test_display_delimiters_and_escapes_never_change_runtime_literal_matching():
    cat = events.fit_feedback_catalog([
        record(0, action=r"route orb1 | path\left", post=r"Routed orb1 | $literal\next"),
        record(1, action=r"route orb2 | path\left", post=r"Routed orb2 | $literal\next"),
    ])
    first, feedback = r"route orb9 | path\left", r"Routed orb9 | $literal\next"
    menu = events.catalog_for_actions(cat, [first])
    prompt = events.prompt_contract("event_guarded", menu)
    line = prompt.split("Events: id | action | match | feedback\n", 1)[1]
    alias, displayed_action, match, displayed_feedback = line.split(" | ")
    assert alias == "e0" and match == "exact"
    assert decode_compact_template(displayed_action) == menu["events"]["e0"]["action_shape"]
    assert decode_compact_template(displayed_feedback) == menu["events"]["e0"]["feedback_template"]
    control = events.EventPlanState(cat, "inspect orb", "event_guarded")
    control.install(plan(menu, first=first), current_observation="initial", current_actions=[first], menu=menu)
    control.observe(first, feedback, native_valid=True)
    result = control.decide(feedback, [NEXT])
    assert result["commit"]
    assert result["expected_feedback"] == feedback


@pytest.mark.parametrize("alias", ["e99", "e00", "E0", "me_guessed", ""])
def test_unknown_or_noncanonical_aliases_are_rejected(alias):
    menu = events.catalog_for_actions(catalog(), [FIRST])
    value = plan(menu)
    value["continuation"]["after_event"] = alias
    with pytest.raises(ValueError):
        parsed(menu, value)


def test_long_catalogue_id_is_not_a_valid_request_local_alias():
    menu = events.catalog_for_actions(catalog(), [FIRST])
    value = plan(menu)
    value["continuation"]["after_event"] = menu["events"]["e0"]["catalog_event_id"]
    with pytest.raises(ValueError, match="Unknown after_event alias"):
        parsed(menu, value)


def test_alias_swapping_is_rejected_even_with_updated_action_relations():
    _, menu = nine_event_fixture()
    menu["events"]["e0"], menu["events"]["e1"] = menu["events"]["e1"], menu["events"]["e0"]
    menu["action_events"] = {
        action: sorted(alias for alias, event in menu["events"].items() if action in event["first_actions"])
        for action in menu["native_actions"]
    }
    with pytest.raises(ValueError, match="support-ranked"):
        events.response_schema(menu["native_actions"], menu, allow_continuation=True)


def test_alias_gap_or_duplicate_catalogue_identity_is_rejected():
    _, menu = nine_event_fixture()
    gap = deepcopy(menu)
    gap["events"]["e9"] = gap["events"].pop("e0")
    with pytest.raises(ValueError, match="consecutive request-local"):
        events.prompt_contract("event_guarded", gap)
    duplicate = deepcopy(menu)
    duplicate["events"]["e1"] = deepcopy(duplicate["events"]["e0"])
    with pytest.raises(ValueError, match="Invalid factored event identity"):
        events.prompt_contract("event_guarded", duplicate)


@pytest.mark.parametrize("mode", EVENT_MODES)
def test_state_records_both_request_alias_and_catalogue_identity_through_execution(mode):
    cat = novelty_catalog()
    for first, actual in ((FIRST, OBSERVATION), ("unseal case9", "Unsealed case9; visible case8.")):
        menu = events.catalog_for_actions(cat, [first])
        catalog_id = menu["events"]["e0"]["catalog_event_id"]
        control = events.EventPlanState(cat, "inspect orb", mode)
        installed_plan = control.install(
            plan(menu, first=first), current_observation="initial", current_actions=[first], menu=menu,
        )
        observed = control.observe(first, actual, native_valid=True)
        decision = control.decide(actual, [NEXT])
        completed = control.observe(NEXT, "actual continuation feedback", native_valid=True)
        for audit in (installed_plan, observed, decision, completed):
            assert audit["event_id"] == "e0"
            assert audit["catalog_event_id"] == catalog_id
            assert audit["catalog_sha256"] == canonical_hash(cat)
            assert audit["menu_sha256"] == canonical_hash(menu)
        assert decision["commit"]
        assert decision["event_matched"] is (None if mode == "event_unchecked" else True)
        assert decision["novelty_passed"] is (None if mode == "event_no_novelty" else True)


def test_double_digit_aliases_preserve_selected_order_after_sorted_json_serialization():
    rows = []
    for index, letter in enumerate("ABCDEFGHIJKL"):
        for group in range(2):
            rows.append(record(
                index * 2 + group, group=f"g{group}",
                action=f"touch orb{group + 1}", post=f"Outcome {letter}: orb{group + 1}.",
            ))
    cat = events.fit_feedback_catalog(rows)
    menu = events.catalog_for_actions(cat, ["touch orb9"])
    restored = json.loads(json.dumps(menu, sort_keys=True))
    aliases = [f"e{index}" for index in range(12)]
    assert list(menu["events"]) == aliases
    schema = events.response_schema(menu["native_actions"], restored, allow_continuation=True)
    assert schema["properties"]["continuation"]["anyOf"][1]["properties"]["after_event"]["enum"] == aliases
    prompt = events.prompt_contract("event_guarded", menu)
    assert events.prompt_contract("event_guarded", restored) == prompt
    lines = prompt.split("Events: id | action | match | feedback\n", 1)[1].splitlines()
    assert [line.split(" | ", 1)[0] for line in lines] == aliases
    assert [menu["events"][alias]["catalog_event_id"] for alias in aliases] == sorted(cat["events"])
