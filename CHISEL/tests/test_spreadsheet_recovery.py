"""Extracted synthetic controller unit tests; no benchmark or provider calls."""
import copy
import pytest

from chisel.sb2_protocol.jev_recovery import RecoveryPolicy, RecoverySession, public_rounds

from chisel.sb2_protocol.spec import ProtocolError

def recovery_payload(observations):
    messages = [{"role": "system", "content": "UNCHANGED_CONSTRAINTS"},
                {"role": "user", "content": "Complete the synthetic workbook task."}]
    for index, observation in enumerate(observations):
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call-{index}", "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"inspect"}'},
            }]},
            {"role": "tool", "tool_call_id": f"call-{index}", "content": observation},
        ])
    return {"model": "synthetic-model", "messages": messages, "max_tokens": 4096,
            "tools": [{"type": "function", "function": {"name": "bash"}}]}

def test_recovery_preserves_normal_requests_and_canonical_history():
    payload = recovery_payload(["different result", "new result"])
    before = copy.deepcopy(payload)
    projected, record = RecoverySession().project(payload)
    assert projected == before == payload
    assert record["recovery_request"] is False
    assert record["extra_model_calls"] == 0

def test_early_recovery_requires_completed_public_feedback_and_keeps_task():
    policy = RecoveryPolicy(recent_rounds=1, activation_after_rounds=1)
    for observations in ([], [""]):
        payload = recovery_payload(observations)
        assert RecoverySession(policy).project(payload)[0] == payload
    payload = recovery_payload(["first progress", "second progress", "new progress"])
    before = copy.deepcopy(payload)
    projected, record = RecoverySession(policy).project(payload)
    assert record["reason"] == "typed_bounded_continuation"
    assert record["retained_rounds"] == 1
    assert projected["messages"][-2:] == payload["messages"][-2:]
    assert projected["messages"][1] == payload["messages"][1]
    assert projected["messages"][0]["content"].startswith("UNCHANGED_CONSTRAINTS")
    assert "stalled" not in projected["messages"][0]["content"]
    assert projected["tools"] == payload["tools"]
    assert projected["max_tokens"] == 768
    assert payload == before

def test_early_recovery_waits_for_declared_round_and_restores_original():
    session = RecoverySession(RecoveryPolicy(activation_after_rounds=2))
    initial = recovery_payload(["progress"])
    assert session.project(initial)[0] == initial
    payload = recovery_payload(["first progress", "second progress"])
    assert session.project(payload)[1]["reason"] == "typed_bounded_continuation"
    session.observe_response({"choices": [{"finish_reason": "length"}]})
    assert session.project(payload)[0] == payload
    for _ in range(3):
        assert session.project(payload)[1]["recovery_request"]
    assert session.project(payload)[1]["reason"] == "original_periodic_refresh"
    failed = recovery_payload(["Traceback (most recent call last):\nSyntaxError: invalid"])
    assert session.project(failed)[1]["recovery_request"]
    assert session.project(failed)[0] == failed

def test_recovery_keeps_complete_recent_tool_pairs_and_original_task():
    payload = recovery_payload(["old result", "previous result", "same result", "same result"])
    before = copy.deepcopy(payload)
    projected, record = RecoverySession().project(payload)
    assert record["reason"] == "typed_repeated_feedback"
    assert projected["max_tokens"] == 768
    assert projected["messages"][0]["content"].startswith(payload["messages"][0]["content"])
    assert projected["messages"][1] == payload["messages"][1]
    assert [message["role"] for message in projected["messages"]].count("system") == 1
    assert projected["messages"][0]["role"] == "system"
    assert projected["messages"][-4:] == payload["messages"][-4:]
    assert projected["tools"] == payload["tools"]
    assert len(public_rounds(projected["messages"])[1]) == 2
    assert payload == before

def test_recovery_error_and_periodic_refresh_restore_full_original_request():
    session = RecoverySession()
    failed = recovery_payload(["Traceback (most recent call last):\nSyntaxError: invalid syntax"])
    assert session.project(failed)[1]["recovery_request"]
    projected, record = session.project(failed)
    assert projected == failed and record["reason"] == "original_after_recovery_error"
    successful = recovery_payload(["new progress"])
    for _ in range(3):
        assert session.project(successful)[1]["recovery_request"]
    projected, record = session.project(successful)
    assert projected == successful and record["reason"] == "original_periodic_refresh"

def test_no_periodic_refresh_retains_error_and_truncation_fallbacks():
    policy = RecoveryPolicy(activation_after_rounds=1, periodic_refresh=False)
    session = RecoverySession(policy)
    successful = recovery_payload(["new progress"])
    for _ in range(8):
        projected, record = session.project(successful)
        assert record["recovery_request"] and projected["max_tokens"] == 768
        assert record["reason"] != "original_periodic_refresh"
    session.observe_response({"choices": [{"finish_reason": "length"}]})
    assert session.project(successful)[0] == successful
    assert session.project(successful)[1]["recovery_request"]
    failed = recovery_payload(["Traceback (most recent call last): error"])
    projected, record = session.project(failed)
    assert projected == failed and record["reason"] == "original_after_recovery_error"
    assert RecoveryPolicy().periodic_refresh is True
    with pytest.raises(ProtocolError, match="periodic_refresh"):
        RecoveryPolicy(periodic_refresh=1)

def test_recovery_fails_closed_for_incomplete_round_and_never_increases_cap():
    payload = recovery_payload(["command not found"])
    payload["max_tokens"] = 128
    assert RecoverySession().project(payload)[0]["max_tokens"] == 128
    payload["messages"].pop()
    with pytest.raises(ProtocolError, match="incomplete native tool rounds"):
        RecoverySession().project(payload)
    with pytest.raises(ProtocolError, match="Invalid recovery policy"):
        RecoveryPolicy(maximum_fast_steps=True)

def test_recovery_ablations_isolate_context_and_generation_cap():
    payload = recovery_payload(["earlier", "earlier still", "same", "same"])
    cap_only, cap_record = RecoverySession(RecoveryPolicy(compact_history=False)).project(payload)
    history_only, history_record = RecoverySession(RecoveryPolicy(maximum_output_tokens=None)).project(payload)
    assert cap_only["max_tokens"] == 768 and cap_record["retained_rounds"] == 4
    assert history_only["max_tokens"] == 4096 and history_record["retained_rounds"] == 2
    assert cap_only["messages"][-8:] == payload["messages"][-8:]
    assert history_only["messages"][-4:] == payload["messages"][-4:]

def test_recovery_truncated_response_returns_to_original_without_extra_call():
    payload = recovery_payload(["same", "same"])
    session = RecoverySession()
    assert session.project(payload)[1]["recovery_request"]
    session.observe_response({"choices": [{"finish_reason": "length"}]})
    projected, record = session.project(payload)
    assert projected == payload
    assert record["reason"] == "original_after_incomplete_recovery_response"
    assert record["extra_model_calls"] == 0

def test_recovery_never_discards_intervening_system_constraints():
    payload = recovery_payload(["same", "same"])
    payload["messages"].insert(4, {"role": "system", "content": "Additional required constraint"})
    with pytest.raises(ProtocolError, match="intervening system"):
        RecoverySession().project(payload)

from chisel.sb2_protocol.jev_recovery import RecoveryPolicy, RecoverySession, public_rounds
from chisel.sb2_protocol.spec import ProtocolError
