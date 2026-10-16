from copy import deepcopy
import json

import pytest

from chisel.learning import entry_guard_refinement as guards


HARMFUL = "harmful_entry_supported"
HELPFUL = "helpful_entry_supported"
UNRESOLVED = "unresolved"


def example(group="h0", label=HARMFUL, *, item="widget", item_id="1", other_id="2",
            support="rack", support_id="1", last="prime", history=None,
            intervention_kind="exclude_entry"):
    selected, other, holder = f"{item} {item_id}", f"{item} {other_id}", f"{support} {support_id}"
    program = [f"inspect {selected}", f"place {selected} in {holder}"]
    return {
        "group_id": group, "task_family": "opaque-native-family", "program_actions": program,
        "public_state": {
            "task": f"place {item} in {support}",
            "observation": f"{selected} beside {other} and {holder}",
            "actions": program + [f"inspect {other}", f"inspect {holder}"], "history": [],
        },
        "validated_action_history": [f"{last} {selected}"] if history is None else history,
        "evidence_label": label, "intervention_kind": intervention_kind,
    }


def extract(vocabulary, row, *, representation="lifted"):
    return guards.extract_guard_features(
        vocabulary, row["public_state"], row["program_actions"], row["validated_action_history"],
        task_family=row["task_family"], representation=representation,
    )


def decide(model, row, **kwargs):
    return guards.decide_entry_guard(
        model, row["public_state"], row["program_actions"], row["validated_action_history"],
        task_family=row["task_family"], **kwargs,
    )


def training_rows():
    return [example("h0"), example("h1"), example("c0", HELPFUL, last="different")]


def test_source_native_vocabulary_records_provenance_and_ignores_instruction_text():
    rows = training_rows()
    rows[0]["public_state"]["task"] += " instructiononly 99"
    rows[0]["public_state"]["observation"] += " observationonly 88"
    rows[0]["public_state"]["history"] = [{"action": "unvalidated 77", "observation": "text"}]
    vocabulary = guards.fit_entity_vocabulary(rows)
    assert vocabulary["schema_version"] == guards.VOCABULARY_SCHEMA
    assert [row["name"] for row in vocabulary["types"]] == ["rack", "widget"]
    assert vocabulary["fit_groups"] == ["c0", "h0", "h1"]
    assert all(row["source_groups"] == ["c0", "h0", "h1"] for row in vocabulary["types"])
    assert all(row["source_ids"] for row in vocabulary["types"])
    assert vocabulary == json.loads(json.dumps(vocabulary, allow_nan=False))


def test_object_and_instance_renaming_preserves_positional_roles_and_features():
    first = example("h0")
    renamed = example("h1", item="vessel", item_id="7", other_id="8",
                      support="shelf", support_id="9")
    vocabulary = guards.fit_entity_vocabulary([first, renamed])
    a, b = extract(vocabulary, first), extract(vocabulary, renamed)
    assert a["status"] == b["status"] == "ok"
    assert a["shape"] == b["shape"]
    assert a["shape_id"] == b["shape_id"]
    assert a["features"] == b["features"]
    assert a["bindings"] != b["bindings"]
    assert a["shape"]["arguments"] == [0, 1]
    assert a["shape"]["role_type_partition"] == [0, 1]
    assert a["shape"]["goal"] == ["place", {"role": 0}, "in", {"role": 1}]
    assert "widget" not in json.dumps(a["features"])
    assert "vessel" not in json.dumps(b["features"])


def test_numbered_goal_role_swaps_remain_distinct_even_for_same_type():
    row = example()
    row["program_actions"] = ["inspect widget 1", "place widget 1 near widget 2"]
    row["public_state"]["actions"] = row["program_actions"] + ["inspect widget 3"]
    row["public_state"]["task"] = "place widget 1 near widget 2"
    swapped = deepcopy(row)
    swapped["public_state"]["task"] = "place widget 2 near widget 1"
    vocabulary = guards.fit_entity_vocabulary([row])
    first, second = extract(vocabulary, row), extract(vocabulary, swapped)
    assert first["status"] == second["status"] == "ok"
    assert first["shape"]["goal"] == second["shape"]["goal"]
    assert first["shape"]["role_type_partition"] == second["shape"]["role_type_partition"] == [0, 0]
    assert first["shape"]["arguments"] == [0, 1]
    assert second["shape"]["arguments"] == [1, 0]
    assert first["shape_id"] != second["shape_id"]
    assert first["features"]["program.arg.0.goal_role"] == 0
    assert second["features"]["program.arg.0.goal_role"] == 1


@pytest.mark.parametrize("task,program,reason", [
    ("place widget near widget", ["inspect widget 1", "place widget 1 near widget 2"],
     "ambiguous_unnumbered_goal_roles"),
    ("place widget in rack", ["compare widget 1 with widget 2", "place widget 1 in rack 1"],
     "ambiguous_program_instances_for_goal_role"),
    ("place widget in rack", ["inspect widget 1", "inspect widget 1"],
     "unbound_goal_role"),
    ("place widget in rack", ["inspect stand 1", "place widget 1 in rack 1"],
     "unbound_program_argument"),
    ("place widget 99 in rack 1", ["inspect widget 1", "place widget 1 in rack 1"],
     "goal_reference_not_native"),
    ("opaque words only", ["inspect widget 1", "place widget 1 in rack 1"],
     "unbound_goal_roles"),
])
def test_ambiguous_or_unbound_lifting_abstains_without_guessed_roles(task, program, reason):
    row = example()
    row["public_state"]["task"] = task
    row["program_actions"] = program
    row["public_state"]["actions"] = list(dict.fromkeys(program + ["inspect rack 1", "inspect widget 2"]))
    vocabulary = guards.fit_entity_vocabulary([row])
    result = extract(vocabulary, row)
    assert result["status"] == "abstain"
    assert result["reason"] == reason
    assert result["features"] == {}
    assert result["shape"] is result["shape_id"] is None
    assert result["bindings"] == []


def test_explicit_repeated_entity_mentions_preserve_identity_and_occurrence_order():
    row = example()
    row["public_state"]["task"] = "inspect widget 1 then place widget 1 in rack 1"
    result = extract(guards.fit_entity_vocabulary([row]), row)
    assert result["status"] == "ok"
    assert [token["role"] for token in result["shape"]["goal"] if isinstance(token, dict)] == [0, 0, 1]


def test_native_affordances_distinguish_selected_from_another_instance_of_the_same_type():
    row = example()
    vocabulary = guards.fit_entity_vocabulary([row])
    result = extract(vocabulary, row)
    templates = [json.loads(key.split(":", 1)[1]) for key in result["features"]
                 if key.startswith("native.template:")]
    assert ["inspect", {"role": 0}] in templates
    assert ["inspect", {"other_of_roles": [0], "entity": 0}] in templates
    assert result["features"]["observation.arg.0.selected"] is True
    assert result["features"]["observation.arg.0.other_same_type"] is True
    changed = deepcopy(row)
    changed["public_state"]["actions"].remove("inspect widget 1")
    changed["public_state"]["observation"] = "widget 2 rack 1"
    second = extract(vocabulary, changed)
    assert second["shape_id"] == result["shape_id"]
    assert second["features"]["program.native.0"] is False
    assert second["features"]["observation.arg.0.selected"] is False
    assert second["features"]["observation.arg.0.other_same_type"] is True


def test_last_validated_action_mentions_the_exact_selected_argument_not_ever_seen_type_words():
    row = example(history=["prime widget 1", "later widget 2"])
    row["public_state"]["history"] = [{"action": "fabricated widget 1", "observation": "text"}]
    vocabulary = guards.fit_entity_vocabulary([row])
    first = extract(vocabulary, row)
    assert json.loads(first["features"]["history.arg.0.last"]) == ["prime", {"role": 0}]
    assert first["features"]["history.arg.1.last"] is None
    row["validated_action_history"].append("different widget 1")
    second = extract(vocabulary, row)
    assert json.loads(second["features"]["history.arg.0.last"]) == ["different", {"role": 0}]
    row["validated_action_history"] = ["prime widget 01"]
    assert extract(vocabulary, row)["features"]["history.arg.0.last"] is None
    row["validated_action_history"] = []
    assert extract(vocabulary, row)["features"]["history.arg.0.last"] is None


def test_native_action_order_and_serialization_do_not_change_feature_shapes():
    row = example()
    vocabulary = guards.fit_entity_vocabulary([row])
    original = extract(vocabulary, row)
    row["public_state"]["actions"].reverse()
    assert extract(json.loads(json.dumps(vocabulary)), row) == original
    assert original == json.loads(json.dumps(original, allow_nan=False))


def test_sparse_group_supported_guard_uses_task_level_labels_and_matches_renamed_roles():
    rows = training_rows()
    rows[1] = example("h1", item="vessel", item_id="7", other_id="8", support="shelf", support_id="9")
    model = guards.fit_entry_guards(rows)
    assert model["status"] == "admitted_rules"
    assert len(model["rules"]) == 1
    rule = model["rules"][0]
    assert rule["harmful_groups"] == ["h0", "h1"]
    assert rule["helpful_conflict_groups"] == []
    assert rule["counts"]["harmful_group_count"] == 2
    assert len(rule["literals"]) == 1
    assert rule["literals"][0]["feature"] == "history.arg.0.last"
    for row in rows[:2]:
        result = decide(model, row, group_id="fresh-group")
        assert result["exclude"]
        assert result["reason"] == "group_supported_empirical_exclusion"
        assert result["support_groups"] == ["h0", "h1"]
    assert not decide(model, rows[2])["exclude"]
    later_other = example(history=["prime widget 1", "different widget 2"])
    assert decide(model, later_other)["exclude"]
    later_selected = example(history=["prime widget 1", "different widget 1"])
    assert not decide(model, later_selected)["exclude"]


def test_conjunction_of_two_observable_literals_can_remove_all_helpful_conflicts():
    harmful = [example("h0"), example("h1")]
    helpful_native = example("c0", HELPFUL)
    helpful_native["public_state"]["actions"].remove(helpful_native["program_actions"][1])
    helpful_observation = example("c1", HELPFUL)
    helpful_observation["public_state"]["observation"] = "widget 2 rack 1"
    rows = harmful + [helpful_native, helpful_observation]
    singletons = guards.fit_entry_guards(rows, max_literals=1)
    assert singletons["status"] == "no_admitted_rule"
    model = guards.fit_entry_guards(rows, max_literals=2)
    assert model["status"] == "admitted_rules"
    assert all(len(rule["literals"]) == 2 for rule in model["rules"])
    assert all(rule["counts"]["harmful_group_count"] >= 2 for rule in model["rules"])
    assert all(rule["helpful_conflict_groups"] == [] for rule in model["rules"])
    assert all(decide(model, row)["exclude"] for row in harmful)
    assert not decide(model, helpful_native)["exclude"]
    assert not decide(model, helpful_observation)["exclude"]


def test_many_prefixes_and_exact_duplicates_are_not_independent_harmful_groups():
    rows = [example("one-group", history=["prime widget 1"] * count) for count in range(1, 12)]
    model = guards.fit_entry_guards(rows + [deepcopy(rows[0])])
    assert model["status"] == "no_admitted_rule"
    assert model["rules"] == []
    assert model["counts"]["duplicate_examples"] == 1
    assert model["scopes"][0]["reason"] == "insufficient_independent_harmful_groups"
    assert model["scopes"][0]["harmful_groups"] == ["one-group"]
    assert any(row["reason"] == "duplicate_source_example" for row in model["rejections"])


@pytest.mark.parametrize("helpful_group", ["c0", "h0"])
def test_any_matching_helpful_group_prevents_admission(helpful_group):
    rows = [example("h0"), example("h1"), example(helpful_group, HELPFUL)]
    model = guards.fit_entry_guards(rows)
    assert model["status"] == "no_admitted_rule"
    assert model["rules"] == []
    conflicts = [row for row in model["rejections"] if row["reason"] == "helpful_group_conflict"]
    assert conflicts and all(row["helpful_conflict_groups"] == [helpful_group] for row in conflicts)
    result = decide(model, rows[0])
    assert result["exclude"] is False
    assert result["reason"] == "no_admitted_rule"
    assert result["support_groups"] == []


def test_unresolved_records_never_supply_vocabulary_support_or_helpful_labels():
    unresolved = example("u0", UNRESOLVED, item="unknownitem", support="unknownholder")
    only_unresolved = guards.fit_entry_guards([unresolved])
    assert only_unresolved["status"] == "no_admitted_rule"
    assert only_unresolved["reason"] == "all_evidence_unresolved"
    assert only_unresolved["vocabulary"]["types"] == []
    assert only_unresolved["evidence"] == []
    one_group = guards.fit_entry_guards([example("h0"), example("u0", UNRESOLVED)])
    assert one_group["rules"] == []
    base = guards.fit_entry_guards(training_rows())
    with_unresolved = guards.fit_entry_guards(training_rows() + [unresolved, example("u1", UNRESOLVED)])
    assert with_unresolved["rules"] == base["rules"]
    assert with_unresolved["vocabulary"]["types"] == base["vocabulary"]["types"]
    assert with_unresolved["counts"]["unresolved_examples"] == 2
    assert all(row["evidence_label"] != UNRESOLVED for row in with_unresolved["evidence"])


def test_group_exclusion_precedes_vocabulary_features_candidate_construction_and_conflicts():
    harmful = [example("h0"), example("h1")]
    held = example("held", HELPFUL, item="vessel", item_id="7", other_id="8",
                   support="shelf", support_id="9")
    unexcluded = guards.fit_entry_guards(harmful + [held])
    assert unexcluded["rules"] == []
    excluded = guards.fit_entry_guards(harmful + [held], excluded_groups={"held"})
    reference = guards.fit_entry_guards(harmful, excluded_groups={"held"})
    assert excluded["rules"] == reference["rules"]
    assert excluded["evidence"] == reference["evidence"]
    assert excluded["vocabulary"]["types"] == reference["vocabulary"]["types"]
    assert excluded["fit_groups"] == ["h0", "h1"]
    assert excluded["counts"]["excluded_examples"] == 1
    assert {row["name"] for row in excluded["vocabulary"]["types"]} == {"widget", "rack"}
    assert decide(excluded, held, group_id="held")["exclude"] is False
    assert decide(excluded, held)["reason"] == "unknown_native_entity_type"
    vocabulary = guards.fit_entity_vocabulary(harmful + [held], excluded_groups={"held"})
    assert vocabulary["types"] == excluded["vocabulary"]["types"]


def test_family_and_goal_role_shape_prevent_cross_scope_exclusion():
    model = guards.fit_entry_guards(training_rows())
    wrong_family = example()
    wrong_family["task_family"] = "different-opaque-family"
    assert decide(model, wrong_family)["reason"] == "no_rule_for_family_and_shape"
    swapped = example()
    swapped["public_state"]["task"] = "place rack in widget"
    result = decide(model, swapped)
    assert not result["exclude"]
    assert result["reason"] == "no_rule_for_family_and_shape"


def test_literal_fallback_is_explicit_and_cannot_generalize_an_unbound_program():
    rows = [example("h0"), example("h1")]
    for row in rows:
        row["public_state"]["task"] = "unspecified requested change"
    lifted = guards.fit_entry_guards(rows)
    assert lifted["rules"] == []
    assert lifted["reason"] == "no_extractable_training_evidence"
    assert all(row["reason"] == "unbound_goal_roles" for row in lifted["rejections"])
    literal = guards.fit_entry_guards(rows, representation="literal")
    assert literal["status"] == "admitted_rules"
    assert decide(literal, rows[0])["exclude"]
    assert decide(literal, rows[0])["representation"] == "literal"
    renamed = example(item="vessel", item_id="7", other_id="8", support="shelf", support_id="9")
    renamed["public_state"]["task"] = rows[0]["public_state"]["task"]
    assert not decide(literal, renamed)["exclude"]
    assert decide(literal, renamed)["reason"] == "no_rule_for_family_and_shape"
    features = extract(literal["vocabulary"], rows[0], representation="literal")
    assert features["bindings"] == []
    assert features["shape"]["program"] == rows[0]["program_actions"]


def test_sparse_rule_cap_is_prospective_and_each_unselected_scope_is_audited():
    rows = []
    for shape in range(4):
        for group in range(2):
            row = example(f"g{shape}-{group}")
            old = row["program_actions"][0]
            row["program_actions"][0] = f"operation{shape} widget 1"
            row["public_state"]["actions"] = [
                row["program_actions"][0] if action == old else action
                for action in row["public_state"]["actions"]
            ]
            rows.append(row)
    model = guards.fit_entry_guards(rows, max_rules=1)
    assert len(model["rules"]) == 1
    assert model["parameters"]["max_rules"] == 1
    assert sum(scope["status"] == "no_admitted_rule" for scope in model["scopes"]) == 3
    assert any(row["reason"] == "prospective_rule_cap" for row in model["rejections"])


def test_empty_or_insufficient_evidence_returns_explicit_no_admitted_rule():
    for rows in ([], [example("h0")], [example("c0", HELPFUL)], [example("u0", UNRESOLVED)]):
        model = guards.fit_entry_guards(rows)
        assert model["status"] == "no_admitted_rule"
        assert model["rules"] == []
        assert model["reason"]
        result = decide(model, example("fresh"))
        assert result["exclude"] is False
        assert result["reason"] == "no_admitted_rule"
        assert result["support_group_count"] == 0


def test_training_ids_and_labels_never_enter_predictive_feature_payloads():
    row = example("PRIVATE_GROUP_IDENTIFIER_92837")
    vocabulary = guards.fit_entity_vocabulary([row])
    features = extract(vocabulary, row)
    payload = json.dumps({"shape": features["shape"], "features": features["features"]})
    assert row["group_id"] not in payload
    assert row["evidence_label"] not in payload
    assert "task_id" not in payload
    assert not {"group_id", "evidence_label", "task_id"} & features["features"].keys()


def test_deterministic_serialization_input_order_native_action_order_and_no_mutation():
    rows = training_rows() + [example("u0", UNRESOLVED)]
    before = deepcopy(rows)
    model = guards.fit_entry_guards(rows)
    assert rows == before
    reversed_rows = list(reversed(deepcopy(rows)))
    for row in reversed_rows:
        row["public_state"]["actions"].reverse()
    assert guards.fit_entry_guards(reversed_rows) == model
    restored = json.loads(json.dumps(model, allow_nan=False, sort_keys=True))
    assert restored == model
    assert decide(restored, rows[0]) == decide(model, rows[0])
    rows[0]["program_actions"][0] = "changed after fit"
    assert decide(model, before[0])["exclude"]


def test_query_group_is_an_explicit_leakage_check_not_a_feature():
    model = guards.fit_entry_guards(training_rows())
    with pytest.raises(ValueError, match="refit"):
        decide(model, example(), group_id="h0")
    with pytest.raises(ValueError, match="group_id"):
        decide(model, example(), group_id=True)
    assert decide(model, example(), group_id="new-group")["exclude"]


@pytest.mark.parametrize("kwargs", [
    {"minimum_harmful_groups": True}, {"minimum_harmful_groups": 1},
    {"minimum_harmful_groups": 0}, {"minimum_harmful_groups": 2.0},
    {"max_literals": True}, {"max_literals": 0}, {"max_literals": 3},
    {"max_rules": True}, {"max_rules": 0}, {"max_rules": 33},
    {"representation": True}, {"representation": "automatic_fallback"},
    {"excluded_groups": []}, {"excluded_groups": "h0"}, {"excluded_groups": {True}},
])
def test_invalid_fitting_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        guards.fit_entry_guards(training_rows(), **kwargs)


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(group_id=True),
    lambda row: row.update(task_family=None),
    lambda row: row.update(task_family=""),
    lambda row: row.update(evidence_label=True),
    lambda row: row.update(evidence_label="safe"),
    lambda row: row.pop("evidence_label"),
    lambda row: row.update(program_actions=True),
    lambda row: row.update(program_actions=["one"]),
    lambda row: row.update(program_actions=["a"] * 5),
    lambda row: row.update(program_actions=[True, "two"]),
    lambda row: row.update(validated_action_history=[True]),
    lambda row: row.update(validated_action_history="prime widget 1"),
    lambda row: row.update(task_id="forbidden"),
    lambda row: row.update(evaluator_state={"success": 1}),
    lambda row: row["public_state"].update(reward=1),
    lambda row: row["public_state"].update(task_id="forbidden"),
    lambda row: row["public_state"].update(actions=[True]),
    lambda row: row["public_state"].update(history=[{"action": "x", "future_success": 1}]),
])
def test_invalid_examples_and_hidden_state_are_rejected(mutation):
    row = example()
    mutation(row)
    with pytest.raises(ValueError):
        guards.fit_entity_vocabulary([row])
    with pytest.raises(ValueError):
        guards.fit_entry_guards([row])


@pytest.mark.parametrize("value", [None, {}, True, ()])
def test_training_container_must_be_an_explicit_list(value):
    with pytest.raises(ValueError):
        guards.fit_entry_guards(value)


@pytest.mark.parametrize("mutation", [
    lambda vocab: vocab.update(schema_version="unknown"),
    lambda vocab: vocab.update(syntax="guessed_from_observations"),
    lambda vocab: vocab["types"][0].update(native_command_count=True),
    lambda vocab: vocab["types"][0].update(name=True),
    lambda vocab: vocab["types"][0].update(source_groups=["unfitted"]),
    lambda vocab: vocab.update(excluded_groups=["h0"]),
])
def test_invalid_vocabulary_is_rejected_without_implicit_repair(mutation):
    row = example()
    vocabulary = guards.fit_entity_vocabulary([row])
    mutation(vocabulary)
    with pytest.raises(ValueError):
        extract(vocabulary, row)


@pytest.mark.parametrize("mutation", [
    lambda model: model.update(schema_version="unknown"),
    lambda model: model["parameters"].update(minimum_harmful_groups=True),
    lambda model: model["parameters"].update(minimum_harmful_groups=1),
    lambda model: model["parameters"].update(max_literals=3),
    lambda model: model["parameters"].update(candidate_literal_limit=1000),
    lambda model: model["evidence"][0].update(evidence_label=UNRESOLVED),
    lambda model: model["evidence"][0]["features"].update(evaluator_success=True),
    lambda model: model["evidence"][0]["features"].update({"program.arg.0.goal_role": True}),
    lambda model: model["rules"][0].update(harmful_groups=["invented0", "invented1"]),
    lambda model: model["rules"][0].update(helpful_conflict_groups=["c0"]),
    lambda model: model["rules"][0]["counts"].update(harmful_group_count=True),
    lambda model: model["rules"][0].update(literals=[{"feature": "program.native.0", "value": 1}]),
    lambda model: model["rules"][0].update(literals=[{"feature": "evaluator_success", "value": True}]),
    lambda model: model["scopes"][0]["shape"]["role_type_partition"].__setitem__(0, True),
    lambda model: model.update(status="no_admitted_rule"),
])
def test_invalid_models_cannot_bypass_support_types_or_helpful_conflict_checks(mutation):
    model = guards.fit_entry_guards(training_rows())
    mutation(model)
    with pytest.raises(ValueError):
        decide(model, example())


def test_conjunction_support_is_same_example_not_intersection_of_group_names():
    rows = []
    for group in ("h0", "h1"):
        selected_only = example(group)
        selected_only["public_state"]["actions"].remove(selected_only["program_actions"][1])
        native_only = example(group)
        native_only["public_state"]["observation"] = "widget 2 rack 1"
        rows.extend([selected_only, native_only])
    rows.append(example("c0", HELPFUL))
    neither = example("c1", HELPFUL)
    neither["public_state"]["actions"].remove(neither["program_actions"][1])
    neither["public_state"]["observation"] = "widget 2 rack 1"
    rows.append(neither)
    model = guards.fit_entry_guards(rows)
    impossible = [
        row for row in model["rejections"] if row["stage"] == "candidate"
        and {(literal["feature"], literal["value"]) for literal in row["literals"]} == {
            ("program.native.1", True), ("observation.arg.0.selected", True),
        }
    ]
    assert len(impossible) == 1
    assert impossible[0]["reason"] == "insufficient_harmful_groups"
    assert impossible[0]["harmful_groups"] == []
    assert impossible[0]["support_source_ids"] == []


def test_harmful_groups_are_not_pooled_across_families_or_different_role_shapes():
    first, other_family = example("h0"), example("h1")
    other_family["task_family"] = "other-native-family"
    family_model = guards.fit_entry_guards([first, other_family])
    assert family_model["rules"] == []
    swapped = example("h1")
    swapped["public_state"]["task"] = "place rack in widget"
    shape_model = guards.fit_entry_guards([first, swapped])
    assert shape_model["rules"] == []
    assert all(scope["reason"] == "insufficient_independent_harmful_groups"
               for scope in family_model["scopes"] + shape_model["scopes"])


def test_candidate_literal_cap_is_explicit_bounded_and_audited():
    rows = [example("h0"), example("h1")]
    for row in rows:
        row["public_state"]["actions"].extend(f"operation{i} widget 1" for i in range(50))
    model = guards.fit_entry_guards(rows)
    assert model["parameters"]["candidate_literal_limit"] == 32
    assert model["scopes"][0]["counts"]["candidate_literals"] > 32
    assert model["scopes"][0]["counts"]["tested_candidates"] <= 32 + 32 * 31 // 2
    assert any(row["reason"] == "candidate_literal_cap" for row in model["rejections"])
    assert all(rule["counts"]["harmful_group_count"] == 2 for rule in model["rules"])


@pytest.mark.parametrize("field,value", [
    ("program.arg.0.goal_role", 1), ("program.arg.0.same_type.1", True),
])
def test_structural_identity_features_cannot_contradict_bound_roles(field, value):
    model = guards.fit_entry_guards(training_rows())
    model["evidence"][0]["features"][field] = value
    with pytest.raises(ValueError, match="contradict"):
        decide(model, example())


def test_public_api_key_sets_and_prediction_support_are_explicit():
    model = guards.fit_entry_guards(training_rows())
    assert model["schema_version"] == guards.MODEL_SCHEMA
    assert model["feature_schema_version"] == guards.FEATURE_SCHEMA
    assert set(model) == {
        "schema_version", "feature_schema_version", "representation", "vocabulary", "parameters",
        "excluded_groups", "fit_groups", "objective", "evidence", "scopes", "rules", "counts",
        "rejections", "status", "reason", "semantics",
        "intervention_kind",
    }
    assert set(model["vocabulary"]) == {
        "schema_version", "syntax", "types", "excluded_groups", "fit_groups", "counts", "rejections",
        "intervention_kind",
    }
    assert set(extract(model["vocabulary"], example())) == {
        "schema_version", "representation", "task_family", "status", "reason", "shape",
        "shape_id", "features", "bindings",
    }
    assert set(model["rules"][0]) == {
        "rule_id", "task_family", "shape_id", "literals", "harmful_groups",
        "helpful_conflict_groups", "support_source_ids", "counts",
        "intervention_kind", "directive",
    }
    result = decide(model, example())
    assert set(result) == {
        "exclude", "reason", "representation", "task_family", "shape_id",
        "matched_rule_ids", "support_groups", "support_group_count",
        "intervention_kind", "directive", "defer_to_actor", "actor_calls",
    }
    assert result == json.loads(json.dumps(result))
    assert type(result["exclude"]) is bool
    assert type(result["support_group_count"]) is int


def test_reported_r2_all_unresolved_label_pattern_admits_no_rule_without_feature_fitting(monkeypatch):
    rows = [
        example(
            f"synthetic-r2-group-{index // 2}", UNRESOLVED,
            item_id=str(index + 1), other_id=str(index + 101),
            intervention_kind="exclude_entry",
        )
        for index in range(20)
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("Unresolved labels must not trigger vocabulary/features/candidate fitting")

    monkeypatch.setattr(guards, "_references", forbidden)
    monkeypatch.setattr(guards, "extract_guard_features", forbidden)
    monkeypatch.setattr(guards, "_candidate", forbidden)
    model = guards.fit_entry_guards(rows, intervention_kind="exclude_entry")
    assert model["intervention_kind"] == "exclude_entry"
    assert model["status"] == "no_admitted_rule"
    assert model["reason"] == "all_evidence_unresolved"
    assert model["counts"]["input_examples"] == model["counts"]["unresolved_examples"] == 20
    assert model["counts"]["admitted_rules"] == 0
    assert model["rules"] == model["evidence"] == model["scopes"] == []
    assert model["vocabulary"]["types"] == model["fit_groups"] == []
    assert all(row["intervention_kind"] == "exclude_entry" for row in model["rejections"])
    decision = decide(model, example("fresh"))
    assert decision["intervention_kind"] == "exclude_entry"
    assert decision["reason"] == "no_admitted_rule"
    assert decision["directive"] == "none"
    assert decision["exclude"] is decision["defer_to_actor"] is False
    assert decision["actor_calls"] == 0


@pytest.mark.parametrize("kind", guards.INTERVENTION_KINDS)
def test_unresolved_evidence_never_requests_either_intervention(kind):
    rows = [example("u0", UNRESOLVED, intervention_kind=kind),
            example("u1", UNRESOLVED, intervention_kind=kind)]
    model = guards.fit_entry_guards(rows)
    assert model["intervention_kind"] == kind
    assert model["reason"] == "all_evidence_unresolved"
    result = decide(model, rows[0])
    assert result["directive"] == "none"
    assert result["exclude"] is result["defer_to_actor"] is False
    assert type(result["actor_calls"]) is int and result["actor_calls"] == 0


@pytest.mark.parametrize("representation", ["lifted", "literal"])
def test_synthetic_force_actor_evidence_requests_one_actor_call_not_instance_exclusion(representation):
    rows = training_rows()
    for row in rows:
        row["intervention_kind"] = "force_actor_once"
    model = guards.fit_entry_guards(rows, representation=representation)
    assert model["intervention_kind"] == model["vocabulary"]["intervention_kind"] == "force_actor_once"
    assert model["rules"]
    for record in model["evidence"] + model["scopes"] + model["rules"] + model["rejections"]:
        assert record["intervention_kind"] == "force_actor_once"
    assert all(rule["directive"] == "defer_to_actor_once" for rule in model["rules"])
    result = decide(model, rows[0])
    assert result["directive"] == "defer_to_actor_once"
    assert result["defer_to_actor"] is True
    assert result["exclude"] is False
    assert type(result["actor_calls"]) is int and result["actor_calls"] == 1
    assert result["reason"] == "group_supported_empirical_actor_deferral"
    assert result["support_groups"] == ["h0", "h1"]
    unmatched = decide(model, rows[2])
    assert unmatched["directive"] == "none"
    assert unmatched["defer_to_actor"] is unmatched["exclude"] is False
    assert unmatched["actor_calls"] == 0
    with pytest.raises(ValueError, match="actor deferral"):
        guards.decide_exclusion(
            model, rows[0]["public_state"], rows[0]["program_actions"],
            rows[0]["validated_action_history"], task_family=rows[0]["task_family"],
        )


def test_instance_exclusion_evidence_does_not_claim_actor_deferral():
    rows = training_rows()
    model = guards.fit_entry_guards(rows)
    assert all(rule["intervention_kind"] == "exclude_entry" for rule in model["rules"])
    assert all(rule["directive"] == "exclude_entry" for rule in model["rules"])
    result = guards.decide_exclusion(
        model, rows[0]["public_state"], rows[0]["program_actions"],
        rows[0]["validated_action_history"], task_family=rows[0]["task_family"],
    )
    assert result["exclude"] is True
    assert result["defer_to_actor"] is False
    assert result["directive"] == "exclude_entry"
    assert result["actor_calls"] == 0


@pytest.mark.parametrize("labels", [(HARMFUL, HARMFUL), (HARMFUL, HELPFUL), (UNRESOLVED, UNRESOLVED)])
@pytest.mark.parametrize("excluded", [frozenset(), frozenset({"b"})])
def test_mixed_intervention_meanings_are_rejected_even_without_resolved_labels(labels, excluded):
    rows = [
        example("a", labels[0], intervention_kind="exclude_entry"),
        example("b", labels[1], intervention_kind="force_actor_once"),
    ]
    with pytest.raises(ValueError, match="mixed intervention_kind"):
        guards.fit_entry_guards(rows, excluded_groups=excluded)
    with pytest.raises(ValueError, match="mixed intervention_kind"):
        guards.fit_entity_vocabulary(rows, excluded_groups=excluded)


@pytest.mark.parametrize("kind", guards.INTERVENTION_KINDS)
def test_explicit_kind_can_pin_empty_models_but_cannot_relabel_evidence(kind):
    model = guards.fit_entry_guards([], intervention_kind=kind)
    assert model["intervention_kind"] == model["vocabulary"]["intervention_kind"] == kind
    assert model["status"] == "no_admitted_rule"
    assert decide(model, example())["directive"] == "none"
    other = "force_actor_once" if kind == "exclude_entry" else "exclude_entry"
    with pytest.raises(ValueError, match="mixed intervention_kind"):
        guards.fit_entry_guards([example(intervention_kind=other)], intervention_kind=kind)
    with pytest.raises(ValueError, match="mixed intervention_kind"):
        guards.fit_entity_vocabulary([example(intervention_kind=other)], intervention_kind=kind)


@pytest.mark.parametrize("bad_kind", [True, 1, "", "actor", "defer_to_actor_once", [], {}])
def test_intervention_kind_requires_the_exact_measured_design(bad_kind):
    row = example(intervention_kind=bad_kind)
    with pytest.raises(ValueError, match="intervention_kind"):
        guards.fit_entry_guards([row])
    with pytest.raises(ValueError, match="intervention_kind"):
        guards.fit_entity_vocabulary([], intervention_kind=bad_kind)


def test_missing_evidence_kind_is_not_assumed_from_labels_or_public_text():
    row = example()
    row.pop("intervention_kind")
    row["public_state"]["observation"] = "force_actor_once exclude_entry"
    with pytest.raises(ValueError, match="schema"):
        guards.fit_entry_guards([row])


def test_intervention_meaning_is_metadata_not_a_predictive_feature():
    exclusion_rows = training_rows()
    actor_rows = deepcopy(exclusion_rows)
    for row in actor_rows:
        row["intervention_kind"] = "force_actor_once"
    exclusion = guards.fit_entry_guards(exclusion_rows)
    actor = guards.fit_entry_guards(actor_rows)
    assert extract(exclusion["vocabulary"], exclusion_rows[0]) == extract(actor["vocabulary"], actor_rows[0])
    assert [rule["literals"] for rule in exclusion["rules"]] == [
        rule["literals"] for rule in actor["rules"]
    ]
    assert exclusion["rules"][0]["rule_id"] != actor["rules"][0]["rule_id"]
    payload = json.dumps(extract(actor["vocabulary"], actor_rows[0])["features"])
    assert "intervention_kind" not in payload
    assert "force_actor_once" not in payload
    assert "defer_to_actor_once" not in payload
    assert guards.fit_entry_guards(list(reversed(actor_rows))) == actor
    restored = json.loads(json.dumps(actor, allow_nan=False))
    assert decide(restored, actor_rows[0]) == decide(actor, actor_rows[0])


@pytest.mark.parametrize("mutation", [
    lambda model: model.update(intervention_kind="exclude_entry"),
    lambda model: model["vocabulary"].update(intervention_kind="exclude_entry"),
    lambda model: model["evidence"][0].update(intervention_kind="exclude_entry"),
    lambda model: model["scopes"][0].update(intervention_kind="exclude_entry"),
    lambda model: model["rules"][0].update(intervention_kind="exclude_entry"),
    lambda model: model["rules"][0].update(directive="exclude_entry"),
    lambda model: model["rejections"][0].update(intervention_kind="exclude_entry"),
    lambda model: model["rules"][0].pop("intervention_kind"),
    lambda model: model.pop("intervention_kind"),
    lambda model: model.update(schema_version="chisel-causal-entry-exclusion-guards/v1"),
])
def test_artifacts_cannot_mix_or_reinterpret_actor_deferral_as_exclusion(mutation):
    rows = training_rows()
    for row in rows:
        row["intervention_kind"] = "force_actor_once"
    model = guards.fit_entry_guards(rows)
    mutation(model)
    with pytest.raises(ValueError):
        decide(model, rows[0])


def test_legacy_exclusion_api_rejects_even_an_unresolved_actor_intervention_model():
    row = example("u0", UNRESOLVED, intervention_kind="force_actor_once")
    model = guards.fit_entry_guards([row])
    with pytest.raises(ValueError, match="decide_entry_guard"):
        guards.decide_exclusion(
            model, row["public_state"], row["program_actions"],
            row["validated_action_history"], task_family=row["task_family"],
        )


@pytest.mark.parametrize("representation", ["lifted", "literal"])
def test_reported_r3_label_pattern_has_only_one_supporting_group_and_no_admitted_rule(representation):
    rows = [
        example("synthetic-support", "defer_preferred_supported", intervention_kind="force_actor_once"),
        example("synthetic-control", "compiled_route_preferred_supported", last="different",
                intervention_kind="force_actor_once"),
    ] + [
        example(f"synthetic-unresolved-{index}", UNRESOLVED, intervention_kind="force_actor_once")
        for index in range(18)
    ]
    model = guards.fit_entry_guards(rows, representation=representation)
    assert model["intervention_kind"] == "force_actor_once"
    assert model["status"] == "no_admitted_rule"
    assert model["reason"] == "insufficient_independent_intervention_groups"
    assert model["parameters"]["minimum_harmful_groups"] == 2
    assert model["counts"]["input_examples"] == 20
    assert model["counts"]["unresolved_examples"] == 18
    assert model["counts"]["intervention_support_group_count"] == 1
    assert model["counts"]["compiled_preferred_group_count"] == 1
    assert model["rules"] == []
    assert all(scope["counts"]["tested_candidates"] == 0 for scope in model["scopes"])
    assert {row["evidence_label"] for row in model["evidence"]} == {
        "defer_preferred_supported", "compiled_route_preferred_supported",
    }
    result = decide(model, example("fresh", intervention_kind="force_actor_once"))
    assert result["directive"] == "none"
    assert result["exclude"] is result["defer_to_actor"] is False
    assert result["actor_calls"] == 0
    with pytest.raises(ValueError, match="minimum_harmful_groups"):
        guards.fit_entry_guards(rows, minimum_harmful_groups=1, representation=representation)


def test_actor_preference_labels_preserve_group_not_repeat_or_alias_name_support(monkeypatch):
    rows = [
        example(
            "one-independent-group",
            "defer_preferred_supported" if index else HARMFUL,
            history=["prime widget 1"] * (index + 1),
            intervention_kind="force_actor_once",
        )
        for index in range(12)
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("One independent group cannot justify candidate construction")

    monkeypatch.setattr(guards, "_candidate", forbidden)
    model = guards.fit_entry_guards(rows)
    assert model["counts"]["retained_examples"] == 12
    assert model["counts"]["intervention_support_group_count"] == 1
    assert model["reason"] == "insufficient_independent_intervention_groups"
    assert model["rules"] == []


def test_synthetic_independent_actor_preferences_use_explicit_deferral_and_conflict_meanings():
    rows = [
        example("h0", "defer_preferred_supported", intervention_kind="force_actor_once"),
        example("h1", "defer_preferred_supported", intervention_kind="force_actor_once"),
        example("c0", "compiled_route_preferred_supported", intervention_kind="force_actor_once"),
    ]
    conflicting = guards.fit_entry_guards(rows)
    assert conflicting["rules"] == []
    assert any(row["reason"] == "helpful_group_conflict" for row in conflicting["rejections"])
    rows[2]["validated_action_history"] = ["different widget 1"]
    model = guards.fit_entry_guards(rows)
    assert model["counts"]["intervention_support_group_count"] == 2
    assert model["counts"]["compiled_preferred_group_count"] == 1
    assert {row["evidence_label"] for row in model["evidence"]} == {
        "defer_preferred_supported", "compiled_route_preferred_supported",
    }
    assert all(rule["harmful_groups"] == ["h0", "h1"] for rule in model["rules"])
    result = decide(model, rows[0])
    assert result["directive"] == "defer_to_actor_once"
    assert result["defer_to_actor"] is True and result["exclude"] is False
    assert result["actor_calls"] == 1
    assert decide(model, rows[2])["directive"] == "none"
    restored = json.loads(json.dumps(model, allow_nan=False))
    assert decide(restored, rows[0]) == result


@pytest.mark.parametrize("label", ["defer_preferred_supported", "compiled_route_preferred_supported"])
def test_actor_preference_labels_cannot_be_used_as_instance_exclusion_evidence(label):
    row = example(label=label)
    with pytest.raises(ValueError, match="incompatible evidence_label"):
        guards.fit_entity_vocabulary([row])
    with pytest.raises(ValueError, match="incompatible evidence_label"):
        guards.fit_entry_guards([row])


def test_group_exclusion_applies_before_actor_preference_support_accounting():
    rows = [
        example("h0", "defer_preferred_supported", intervention_kind="force_actor_once"),
        example("h1", "defer_preferred_supported", intervention_kind="force_actor_once"),
        example("c0", "compiled_route_preferred_supported", last="different",
                intervention_kind="force_actor_once"),
    ]
    model = guards.fit_entry_guards(rows, excluded_groups={"h1"})
    assert model["counts"]["intervention_support_group_count"] == 1
    assert model["rules"] == []
    assert model["reason"] == "insufficient_independent_intervention_groups"
    assert all("h1" not in scope["harmful_groups"] for scope in model["scopes"])
