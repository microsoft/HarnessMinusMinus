"""Extracted synthetic controller unit tests; no benchmark or provider calls."""
import json
import pytest



def test_jev_continuations_are_bounded_and_do_not_mutate_tool_schema():
    from chisel.harness.terminal_jev import TerminalJevSession

    session = TerminalJevSession({"mode": "guarded", "maximum_followups": 2})
    tools = session.tools([EXEC_TOOL])
    assert "followups" in tools[0]["function"]["parameters"]["properties"]
    assert "followups" not in EXEC_TOOL["function"]["parameters"]["properties"]
    step = {"command": "printf next", "stdout_contains": ["ready"], "stderr_empty": True}
    call = {"function": {"name": "exec_in_sandbox", "arguments": json.dumps({
        "command": "printf ready", "followups": [step]})}}
    assert session.parse(call) == [step]
    call["function"]["arguments"] = json.dumps({"command": "printf ready", "followups": [step] * 3})
    with pytest.raises(ValueError, match="execution bound"):
        session.parse(call)
    with pytest.raises(ValueError, match="closed, bounded"):
        TerminalJevSession({"mode": "guarded", "maximum_followups": True})

def test_jev_explicit_plan_requires_a_plan_or_explicit_abstention():
    from chisel.harness.terminal_jev import TerminalJevSession

    session = TerminalJevSession({"mode": "guarded", "maximum_followups": 2, "explicit_plan": True})
    assert session.tools([EXEC_TOOL])[0]["function"]["parameters"]["required"] == ["command", "followups"]
    call = {"function": {"name": "exec_in_sandbox", "arguments": '{"command":"pwd"}'}}
    with pytest.raises(ValueError, match=r"use \[\] to defer"):
        session.parse(call)
    call["function"]["arguments"] = '{"command":"pwd","followups":[]}'
    assert session.parse(call) == []
    assert session.report()["plans_seen"] == session.report()["empty_plans"] == 1

def test_jev_gates_abstain_on_missing_evidence_failure_or_timeout():
    from chisel.harness.terminal_jev import TerminalJevSession

    session = TerminalJevSession({"mode": "guarded", "maximum_followups": 2})
    step = {"command": "printf next", "stdout_contains": ["ready"], "stderr_empty": True}
    result = {"exit_code": 0, "timed_out": False, "stdout": "ready", "stderr": ""}
    assert session.gate(step, result, remaining_seconds=1)["execute"] is True
    silent = {**result, "stdout": ""}
    assert session.gate(step, silent, remaining_seconds=1)["execute"] is False
    assert session.gate({**step, "stdout_contains": []}, silent, remaining_seconds=1)["execute"] is True
    for changed in ({**result, "stdout": "unknown"}, {**result, "stderr": "warning"},
                    {**result, "exit_code": 1}, {**result, "exit_code": False},
                    {**result, "timed_out": True}, {**result, "stdout": None}):
        assert session.gate(step, changed, remaining_seconds=1)["execute"] is False
    assert session.gate(step, result, remaining_seconds=0)["execute"] is False
    control = TerminalJevSession({"mode": "success_only", "maximum_followups": 2})
    assert control.gate(step, {**result, "stdout": "other"}, remaining_seconds=1)["execute"] is True
    assert session.report()["neural_gate_calls"] == 0

# Original native tool schema fixture; no runner import.
EXEC_TOOL = {'type': 'function', 'function': {'name': 'exec_in_sandbox', 'description': 'Run one non-interactive shell command in the task container. Use it to inspect files, edit files, run programs, and validate the solution.', 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'The shell command to run.'}}, 'required': ['command'], 'additionalProperties': False}}}
