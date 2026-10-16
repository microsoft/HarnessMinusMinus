"""Synthetic CoT-bank/Jev contracts; no benchmark data or model is loaded."""


from copy import deepcopy


import json


from types import SimpleNamespace


import pytest


from chisel.harness.gaia2_cot_jev import (
    at_path, audit_native_output_contract, bind, cot_response_format, decision_messages, digest, instantiate,
    fast_candidate_action_names, metered_request, observation_data, parse_bank, parse_cot_envelope,
    public_state_candidates,
    native_fast_execution_receipt,
)


def tools():
    return {
        "Store__read": SimpleNamespace(inputs={"item_id": {"type": "string"}}),
        "Store__list": SimpleNamespace(inputs={"offset": {"type": "integer"}, "limit": {"type": "integer"}}),
        "Store__delete": SimpleNamespace(inputs={"item_id": {"type": "string"}}),
    }


def bank():
    return {
        "note": "Read the exact named public item, or continue a nonempty page.",
        "candidates": [
            {"when": "The public item matches the requested name.", "action": "Store__read",
             "action_input": {"item_id": {"$item": ["id"]}},
             "foreach": {"path": [], "where": {"name": "synthetic blue"}}},
            {"when": "Another page is still required.", "action": "Store__list",
             "action_input": {"offset": {"$add": [{"$prev": ["offset"]}, 5]}, "limit": 5}},
        ],
    }


def test_public_records_are_inert_data_and_never_executed(tmp_path):
    data = observation_data("[Record(name='synthetic blue', id='a'), Record(name='other', id='b')]")
    assert data == [{"name": "synthetic blue", "id": "a"}, {"name": "other", "id": "b"}]
    with pytest.raises((ValueError, SyntaxError)):
        observation_data("__import__('os').system('touch never-created')")
    with pytest.raises(ValueError):
        observation_data("{'a':1, 'a':2}")
    assert not (tmp_path / "never-created").exists()


def test_candidates_come_only_from_cot_templates_and_public_bindings():
    original = bank()
    parsed = parse_bank({"jev_bank": original}, tools())
    rows, failures = instantiate(parsed, [{"name": "synthetic blue", "id": "visible-id"}],
                                 {"offset": 5}, tools(), set())
    assert not failures
    assert [row["action"] for row in rows] == [
        {"action": "Store__read", "action_input": {"item_id": "visible-id"}},
        {"action": "Store__list", "action_input": {"offset": 10, "limit": 5}},
    ]
    assert original == bank()
    seen = {digest(rows[0]["action"])}
    remaining, _ = instantiate(parsed, [{"name": "synthetic blue", "id": "visible-id"}],
                               {"offset": 5}, tools(), seen)
    assert len(remaining) == 1


def test_missing_paths_and_large_banks_defer_without_truncating():
    with pytest.raises((KeyError, ValueError)):
        at_path({}, ["missing"])
    with pytest.raises(ValueError):
        bind({"$add": [True, 1]}, {}, {})
    many = [{"name": "synthetic blue", "id": str(i)} for i in range(12)]
    rows, failures = instantiate(bank(), many, {"offset": 0}, tools(), set())
    assert rows == [] and failures[-1]["reason"] == "candidate_capacity_overflow_no_truncation"
    rows, failures = instantiate(bank(), None, {}, tools(), set())
    assert rows == [] and failures


def test_full_cot_retains_destructive_and_completion_decisions():
    value = bank()
    value["candidates"][0]["action"] = "Store__delete"
    with pytest.raises(ValueError, match="full CoT"):
        parse_bank({"jev_bank": value}, tools())


def test_typed_messages_have_all_actions_and_unknown_but_no_generated_action_request():
    rows, _ = instantiate(bank(), [{"name": "synthetic blue", "id": "a"}], {"offset": 0}, tools(), set())
    messages, choices = decision_messages("Read synthetic blue.", "complete public observation", bank(),
                                         rows, tools(), {})
    state = json.loads(messages[1]["content"])
    assert set(choices) == {"A", "B", "?"} and choices["?"] is None
    assert state["latest_observation"] == "complete public observation"
    assert "native_contract" in state["choices"]["A"]


def test_gateway_charges_one_control_token_and_preserves_slow_limits():
    config = {"cot_jev": {"enabled": True}}
    request = {
        "max_tokens": 1, "chisel_jev_decision": True, "allowed_token_ids": [12, 13],
        "logprobs": True, "top_logprobs": 2,
    }
    before = deepcopy(request)
    body, maximum, component = metered_request(request, config, "trial/actor", True)
    assert maximum == 1 and component == "jev_typed" and "chisel_jev_decision" not in body
    assert request == before
    assert metered_request({"max_tokens": 4096}, config, "trial/actor", True)[1:] == (4096, "cot")
    with pytest.raises(ValueError):
        metered_request(request, config, "trial/judge", True)
    with pytest.raises(ValueError):
        metered_request({**request, "top_logprobs": 1}, config, "trial/actor", True)


def test_cot_must_emit_a_bank_in_its_native_response_contract():
    schema = cot_response_format()["json_schema"]["schema"]
    assert set(schema["required"]) == {"reasoning", "action", "action_input", "jev_bank"}
    value = {"reasoning": "Read a known record.", "action": "Store__read",
             "action_input": {"item_id": "a"}, "jev_bank": bank()}
    assert parse_cot_envelope(json.dumps(value), tools()) == value
    value.pop("jev_bank")
    with pytest.raises(ValueError, match="required"):
        parse_cot_envelope(json.dumps(value), tools())


def test_reading_messages_is_not_confused_with_sending_them():
    catalogue = {"Messages__get_messages": SimpleNamespace(inputs={"offset": {"type": "integer"}})}
    value = {"note": "Read only.", "candidates": [
        {"when": "More observations needed.", "action": "Messages__get_messages",
         "action_input": {"offset": 0}},
    ]}
    assert parse_bank({"jev_bank": value}, catalogue) == value


def test_model_binding_shorthand_resolves_inert_fields_instead_of_becoming_an_id():
    assert bind("$obs[0].id", [{"id": "native-id"}], {}) == "native-id"
    assert bind("$prev.offset", {}, {"offset": 5}) == 5
    with pytest.raises((ValueError, KeyError)):
        bind("$obs[0].id", [{"event_id": "different-field"}], {})
    with pytest.raises(ValueError):
        bind("$obs[0].id or unsafe()", [{"id": "native-id"}], {})


def test_explicit_simulator_setting_allows_model_authored_state_changes():
    value = bank()
    value["candidates"][0]["action"] = "Store__delete"
    assert parse_bank({"jev_bank": value}, tools(), allow_routine_writes=True) == value


def test_candidate_schema_excludes_cot_only_actions_without_hiding_native_tools():
    catalogue = {
        **tools(),
        "SystemApp__wait_for_notification": SimpleNamespace(inputs={}),
        "AgentUserInterface__send_message_to_user": SimpleNamespace(inputs={}),
    }
    schema = cot_response_format(catalogue, allow_routine_writes=True)["json_schema"]["schema"]
    candidates = schema["properties"]["jev_bank"]["properties"]["candidates"]
    assert schema["properties"]["action"]["enum"] == sorted(catalogue)
    assert candidates["items"]["properties"]["action"]["enum"] == [
        "Store__delete", "Store__list", "Store__read",
    ]
    assert candidates["maxItems"] == 8
    assert "Store__delete" not in fast_candidate_action_names(catalogue)
    assert schema["properties"]["action_input"]["additionalProperties"] is True


def test_only_cot_tools_require_an_explicit_empty_bank_and_preserve_abstention():
    catalogue = {"SystemApp__wait_for_notification": SimpleNamespace(inputs={})}
    schema = cot_response_format(catalogue)["json_schema"]["schema"]
    candidates = schema["properties"]["jev_bank"]["properties"]["candidates"]
    assert candidates["maxItems"] == 0
    assert schema["properties"]["action"]["enum"] == list(catalogue)
    assert parse_bank({"jev_bank": {"note": "Wait requires full reasoning.", "candidates": []}},
                      catalogue) == {"note": "Wait requires full reasoning.", "candidates": []}


def test_native_audit_distinguishes_typed_control_from_slow_actor_and_judge():
    config = {"cot_jev": {"enabled": True}}
    request = {
        "role": "trial/actor", "component": "jev_typed",
        "json": {"max_tokens": 1, "allowed_token_ids": [12, 13], "logprobs": True,
                 "top_logprobs": 2},
    }
    response = {"usage": {"prompt_tokens": 17, "completion_tokens": 1, "total_tokens": 18}}
    audit_native_output_contract(request, response, config)
    audit_native_output_contract(
        {"role": "trial/actor", "component": "cot", "json": {"max_tokens": 4096}},
        response, config,
    )
    with pytest.raises(ValueError):
        audit_native_output_contract(request, {"usage": {"completion_tokens": 0}}, config)
    with pytest.raises(ValueError):
        audit_native_output_contract({**request, "role": "trial/judge"}, response, config)
    with pytest.raises(ValueError):
        audit_native_output_contract({**request, "component": "cot"}, response, config)
    with pytest.raises(ValueError):
        audit_native_output_contract(request, response, {"cot_jev": {"enabled": False}})


@pytest.mark.parametrize("native_id", (
    "749700000000000000000000000000ab",
    "921800000000000000000000000000cd",
))
def test_native_unquoted_identifier_shapes_remain_scalars(native_id):
    assert observation_data(native_id) == native_id
    template = {"note": "The native operation returned an opaque identifier.", "candidates": [
        {"when": "A read of the exact returned identifier is warranted.",
         "action": "Store__read", "action_input": {"item_id": {"$obs": []}}},
    ]}
    rows, failures, receipt = public_state_candidates(template, native_id, {}, tools(), set())
    assert not failures and receipt["status"] == "PARSED"
    assert receipt["native_scalar_identifier"]
    assert rows[0]["action"]["action_input"] == {"item_id": native_id}
    messages, _ = decision_messages("Read the returned identifier.", native_id, template, rows, tools(), {})
    assert json.loads(messages[1]["content"])["latest_observation"] == native_id
    assert json.loads(messages[1]["content"])["choices"]["A"]["native_contract"] == tools()["Store__read"].inputs


def test_native_scalar_identifier_does_not_invent_a_record_or_missing_fields():
    native_id = "749700000000000000000000000000ab"
    template = {"note": "An explicit record field is required.", "candidates": [
        {"when": "The record has an id.", "action": "Store__read",
         "action_input": {"item_id": {"$obs": ["id"]}}},
    ]}
    rows, failures, receipt = public_state_candidates(template, native_id, {}, tools(), set())
    assert rows == [] and failures and receipt["status"] == "PARSED"
    assert observation_data('{"id":"synthetic-id","count":3}') == {"id": "synthetic-id", "count": 3}
    assert observation_data("[Record(id='synthetic-id', optional=None)]") == [
        {"id": "synthetic-id", "optional": None},
    ]


@pytest.mark.parametrize("text", (
    "unknown native output",
    "__import__('os').system('never-execute')",
    "749700000000000000000000000000ab\n921800000000000000000000000000cd",
    '{"id":"one","id":"two"}',
    "NaN",
))
def test_unknown_public_state_defers_even_when_the_bank_has_literal_candidates(text):
    template = {"note": "Require interpretable current public evidence.", "candidates": [
        {"when": "Current public state warrants reading.", "action": "Store__read",
         "action_input": {"item_id": "literal-planned-id"}},
    ]}
    rows, failures, receipt = public_state_candidates(template, text, {}, tools(), set())
    assert rows == failures == []
    assert receipt["status"] == "UNKNOWN" and receipt["route"] == "full_cot"
    assert receipt["reason"] and receipt["error_type"]
    assert receipt["native_observation_preserved"]


def test_actual_native_call_and_observation_are_required_for_fast_execution_proof():
    action = {"action": "Store__read", "action_input": {"item_id": "synthetic-id"}}
    calls = [{"tool_name": "Store__read", "tool_arguments": '{"item_id":"synthetic-id"}'}]
    proof = native_fast_execution_receipt(action, calls, ["native public result"], step_completed=True)
    assert proof["status"] == "PASS" and proof["original_native_step_and_tool_executor_used"]
    assert proof["native_action_sha256"] == digest(action)
    for changed in (
        {"tool_calls": [], "observations": ["native public result"], "step_completed": True},
        {"tool_calls": calls, "observations": [], "step_completed": True},
        {"tool_calls": calls, "observations": ["native public result"], "step_completed": False},
        {"tool_calls": [{"tool_name": "Store__read", "tool_arguments": {"item_id": "another-id"}}],
         "observations": ["native public result"], "step_completed": True},
    ):
        assert native_fast_execution_receipt(action, **changed)["status"] == "UNVERIFIED"
