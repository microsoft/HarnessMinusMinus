"""Synthetic control tests; no benchmark data, models, evaluators, or file writes."""

from dataclasses import FrozenInstanceError, fields
import json
from pathlib import Path
import subprocess
import sys
from typing import cast
from unittest.mock import Mock

import pytest

from chisel.learning.supported_local_decision import (
    SCHEMA,
    SupportedCandidates,
    execute_supported_decision,
)


def test_first_supported_candidate_commits_without_selector_or_fallback():
    support = SupportedCandidates(True, ("open cabinet 1", "look"), "public prerequisite")
    native_result = object()
    commit = Mock(return_value=native_result)
    fallback = Mock(side_effect=AssertionError("Actor must not run"))

    result = execute_supported_decision(support, commit=commit, fallback=fallback)

    assert result.value is native_result
    commit.assert_called_once_with("open cabinet 1")
    fallback.assert_not_called()
    assert result.trace.to_dict() == {
        "schema_version": SCHEMA,
        "route": "commit",
        "reason": "supported_candidate_committed",
        "supported": True,
        "support_reason": "public prerequisite",
        "candidate_count": 2,
        "selector_called": False,
        "proposed_action": "open cabinet 1",
        "proposed_type": "str",
        "committed_action": "open cabinet 1",
    }


def test_selector_receives_the_ordered_candidates_and_can_choose_an_alternative():
    actions = ("first", "second")
    select = Mock(return_value="second")
    commit = Mock(return_value="native result")
    fallback = Mock()

    result = execute_supported_decision(
        SupportedCandidates(True, actions, "public alternatives"),
        select=select, commit=commit, fallback=fallback,
    )

    select.assert_called_once_with(actions)
    commit.assert_called_once_with("second")
    fallback.assert_not_called()
    assert result.trace.selector_called
    assert result.trace.committed_action == "second"


@pytest.mark.parametrize(
    ("supported", "actions", "reason"),
    [
        (False, ("candidate",), "unsupported_state"),
        (False, None, "unsupported_state"),
        (True, (), "empty_candidate_set"),
        (True, ("",), "invalid_candidate_set"),
        (True, ("   ",), "invalid_candidate_set"),
        (True, ("duplicate", "duplicate"), "invalid_candidate_set"),
        (True, ("candidate", 1), "invalid_candidate_set"),
        (True, ("candidate", []), "invalid_candidate_set"),
        (True, ["candidate"], "invalid_candidate_set"),
        (True, "candidate", "invalid_candidate_set"),
        (True, None, "invalid_candidate_set"),
        (True, {"candidate": True}, "invalid_candidate_set"),
    ],
)
def test_unusable_support_falls_back_once_without_selection(supported, actions, reason):
    select = Mock(side_effect=AssertionError("Selector must not run"))
    commit = Mock(side_effect=AssertionError("Commit must not run"))
    fallback = Mock(return_value="unchanged actor output")

    result = execute_supported_decision(
        SupportedCandidates(supported, cast(tuple[str, ...], actions), "adapter state"),
        select=select, commit=commit, fallback=fallback,
    )

    assert result.value == "unchanged actor output"
    assert result.trace.route == "fallback"
    assert result.trace.reason == reason
    assert result.trace.proposed_type is None
    assert result.trace.committed_action is None
    assert not result.trace.selector_called
    select.assert_not_called()
    commit.assert_not_called()
    fallback.assert_called_once_with()
    json.dumps(result.trace.to_dict())


@pytest.mark.parametrize("fallback_action", [None, "restore"])
def test_explicit_deferral_calls_only_fallback(fallback_action):
    select = Mock(return_value=fallback_action)
    commit = Mock()
    fallback = Mock(return_value="restored")

    result = execute_supported_decision(
        SupportedCandidates(True, ("keep",), "public source"),
        select=select, commit=commit, fallback=fallback, fallback_action=fallback_action,
    )

    assert result.trace.reason == "fallback_requested"
    assert result.trace.proposed_action == fallback_action
    assert result.trace.selector_called
    select.assert_called_once()
    commit.assert_not_called()
    fallback.assert_called_once_with()


@pytest.mark.parametrize(
    ("proposed", "reason"),
    [
        ("invented", "selection_not_supported"),
        ("", "invalid_selection"),
        ("  ", "invalid_selection"),
        (None, "invalid_selection"),
        (0, "invalid_selection"),
        (False, "invalid_selection"),
        ([], "invalid_selection"),
        ({"answer": "invented"}, "invalid_selection"),
    ],
)
def test_disallowed_selection_is_rejected_without_changing_native_fallback(proposed, reason):
    select = Mock(return_value=proposed)
    commit = Mock()
    fallback_result = object()
    fallback = Mock(return_value=fallback_result)

    result = execute_supported_decision(
        SupportedCandidates(True, ("keep",), "public source"),
        select=select, commit=commit, fallback=fallback, fallback_action="restore",
    )

    assert result.value is fallback_result
    assert result.trace.reason == reason
    assert result.trace.proposed_type == type(proposed).__name__
    assert result.trace.proposed_action == (proposed if isinstance(proposed, str) else None)
    assert result.trace.committed_action is None
    select.assert_called_once()
    commit.assert_not_called()
    fallback.assert_called_once_with()
    json.dumps(result.trace.to_dict())


def test_fallback_token_cannot_also_be_a_committable_candidate():
    select, commit, fallback = Mock(), Mock(), Mock()
    result = execute_supported_decision(
        SupportedCandidates(True, ("keep", "restore"), "public source"),
        select=select, commit=commit, fallback=fallback, fallback_action="restore",
    )
    assert result.trace.reason == "invalid_candidate_set"
    select.assert_not_called()
    commit.assert_not_called()
    fallback.assert_called_once()


@pytest.mark.parametrize("stage", ["select", "commit", "fallback"])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TypeError])
def test_callback_errors_propagate_without_retry_or_another_route(stage, error_type):
    error = error_type("synthetic callback failure")
    select = Mock(return_value="keep")
    commit = Mock(return_value="kept")
    fallback = Mock(return_value="restored")
    callbacks = {"select": select, "commit": commit, "fallback": fallback}
    callbacks[stage].side_effect = error

    with pytest.raises(error_type) as raised:
        execute_supported_decision(
            SupportedCandidates(stage != "fallback", ("keep",), "public source"),
            select=select, commit=commit, fallback=fallback,
        )

    assert raised.value is error
    assert select.call_count == int(stage != "fallback")
    assert commit.call_count == int(stage == "commit")
    assert fallback.call_count == int(stage == "fallback")


@pytest.mark.parametrize(
    ("supported", "reason"), [(1, "reason"), ("yes", "reason"), (True, ""), (False, None)]
)
def test_invalid_support_metadata_is_a_contract_error(supported, reason):
    with pytest.raises(ValueError):
        SupportedCandidates(supported, ("candidate",), reason)


def test_support_contract_is_immutable_and_has_no_task_or_privileged_payload_field():
    support = SupportedCandidates(True, ("candidate",), "public prerequisite")
    assert {field.name for field in fields(support)} == {"supported", "actions", "reason"}
    with pytest.raises(FrozenInstanceError):
        setattr(support, "supported", False)
    with pytest.raises(TypeError):
        SupportedCandidates(True, ("candidate",), "public prerequisite", reward=1)


def test_core_and_qa_import_with_only_the_standard_library():
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from chisel.learning import supported_local_decision, qa_typed_recovery\n"
        "assert supported_local_decision.SCHEMA\n"
        "assert not any(name.startswith(('torch', 'numpy', 'transformers', 'scripts', "
        "'chisel.evaluation')) for name in sys.modules)\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code, str(Path(__file__).resolve().parents[1])],
        check=True, capture_output=True, text=True,
    )
