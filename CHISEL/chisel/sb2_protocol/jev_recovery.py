"""Opt-in, public-feedback recovery around unchanged native spreadsheet tools."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
import shlex
from typing import Any

from .spec import ProtocolError


SCHEMA = "phiagent-sb2-typed-progress-recovery/v1"
ERROR_MARKERS = (
    "traceback (most recent call last)",
    "modulenotfounderror:",
    "syntaxerror:",
    "filenotfounderror:",
    "permissionerror:",
    "command not found",
    "submit_precondition_failed",
    "submit_rejected",
)
RECOVERY_GUIDANCE = (
    "Choose one supported native tool action that addresses the observed error or "
    "advances the required workbook. Avoid repeating unchanged inspections. "
    "Use a targeted range or existing script/transaction, not a whole-workbook dump. "
    "Preserve the complete requested edit and unrelated cells. An existing output "
    "file does not prove correctness. Do not claim completion without checking "
    "the requested changes and output path.\n"
    "</typed_progress_recovery>"
)
RECOVERY_INSTRUCTION = (
    "<typed_progress_recovery>\n"
    "Use the original task and tool contract. The preceding public actions stalled. "
    + RECOVERY_GUIDANCE
)
CONTINUATION_INSTRUCTION = (
    "<typed_progress_recovery>\n"
    "Use the original task and tool contract. Continue from the latest public tool feedback. "
    + RECOVERY_GUIDANCE
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class RecoveryPolicy:
    recent_rounds: int = 2
    maximum_fast_steps: int = 3
    maximum_output_tokens: int | None = 768
    compact_history: bool = True
    activation_after_rounds: int | None = None
    deterministic_followups: bool = False
    periodic_refresh: bool = True

    def __post_init__(self):
        for name, lower, upper in (
            ("recent_rounds", 1, 4),
            ("maximum_fast_steps", 1, 3),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ProtocolError(f"Invalid recovery policy: {name}")
        if (self.maximum_output_tokens is not None
                and (type(self.maximum_output_tokens) is not int or not 256 <= self.maximum_output_tokens <= 2048)):
            raise ProtocolError("Invalid recovery output-token allowance")
        if type(self.compact_history) is not bool:
            raise ProtocolError("Recovery compact_history must be a boolean")
        if type(self.deterministic_followups) is not bool:
            raise ProtocolError("deterministic_followups must be a boolean")
        if type(self.periodic_refresh) is not bool:
            raise ProtocolError("periodic_refresh must be a boolean")
        if (self.activation_after_rounds is not None
                and (type(self.activation_after_rounds) is not int
                     or not 1 <= self.activation_after_rounds <= 12)):
            raise ProtocolError("Invalid recovery activation round")


def public_rounds(messages: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    if not isinstance(messages, list) or not messages:
        raise ProtocolError("Recovery requires nonempty native messages")
    prefix = []
    rounds = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise ProtocolError("Unsupported native message schema")
        if message["role"] == "assistant":
            rounds.append([message])
        elif rounds:
            if message["role"] in {"system", "developer"}:
                raise ProtocolError("Recovery cannot reorder intervening system instructions")
            rounds[-1].append(message)
        else:
            prefix.append(message)
    if not any(message["role"] == "user" for message in prefix):
        raise ProtocolError("Original task is absent from the recovery prefix")
    for turn in rounds:
        calls = turn[0].get("tool_calls", [])
        if calls:
            expected = [call.get("id") for call in calls]
            actual = [message.get("tool_call_id") for message in turn[1:] if message["role"] == "tool"]
            if not all(isinstance(identity, str) and identity for identity in expected) or expected != actual:
                raise ProtocolError("Recovery cannot project incomplete native tool rounds")
    return prefix, rounds


def feedback(turn: list[dict]) -> str:
    return "\n".join(
        message["content"] for message in turn[1:]
        if message["role"] in {"tool", "user"} and isinstance(message.get("content"), str)
    )


def action_identity(turn: list[dict]) -> str:
    message = turn[0]
    return digest({
        "content": message.get("content"),
        "calls": [call.get("function") for call in message.get("tool_calls", [])],
    })


def python_saved_output(command: str, original_request: str) -> str | None:
    source = None
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    if len(words) == 3 and words[0] in ("python", "python3") and words[1] == "-c":
        source = words[2]
    else:
        direct = re.fullmatch(r"python3?\s+-\s*<<\s*['\"]?([A-Za-z_][A-Za-z_0-9]*)['\"]?\s*\n(.*?)\n\1\s*", command, re.S)
        saved = re.fullmatch(r"cat\s*>\s*([A-Za-z0-9_./-]+\.py)\s*<<\s*['\"]?([A-Za-z_][A-Za-z_0-9]*)['\"]?\s*\n(.*?)\n\2\s*\npython3?\s+\1\s*", command, re.S)
        if direct:
            source = direct[2]
        elif saved:
            source = saved[3]
    if source is None:
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    paths = [node.args[0].value for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "save"
             and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant)
             and isinstance(node.args[0].value, str)]
    if len(paths) != 1:
        return None
    output_path = paths[0]
    if (not re.fullmatch(r"/[A-Za-z0-9_./-]+\.xlsx", output_path)
            or ".." in PurePosixPath(output_path).parts
            or re.search(r"(?<![A-Za-z0-9_./-])" + re.escape(output_path) + r"(?![A-Za-z0-9_./-])", original_request) is None):
        return None
    return output_path


def deterministic_followup(payload: dict) -> dict:
    prefix, rounds = public_rounds(payload["messages"])
    abstain = {"action": None, "reason": "no_supported_deterministic_followup",
               "selection_model_calls": 0, "correctness_certified": False}
    if not rounds or not isinstance(payload.get("tools"), list):
        return abstain
    supported_tool = next((tool for tool in payload["tools"] if tool.get("type") == "function"
                           and tool.get("function", {}).get("name") == "bash"), None)
    if supported_tool is None:
        return abstain
    calls = rounds[-1][0].get("tool_calls", [])
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "bash":
        return abstain
    original_request = "\n".join(message.get("content", "") for message in prefix
                                 if message["role"] == "user" and isinstance(message.get("content"), str))
    observed = feedback(rounds[-1]).strip()
    if observed.startswith("<observation>") and observed.endswith("</observation>"):
        observed = observed[len("<observation>"):-len("</observation>")].strip()
    if any(marker in observed.casefold() for marker in (*ERROR_MARKERS, "error:", "exception:", "failed")):
        return abstain
    try:
        arguments = json.loads(calls[0]["function"]["arguments"])
        command = shlex.split(arguments["command"], posix=True)
    except (KeyError, TypeError, ValueError):
        return abstain
    saved_output = python_saved_output(arguments["command"], original_request)
    if saved_output is not None:
        inspection = (
            "import json,pathlib,zipfile; "
            f"target=pathlib.Path({saved_output!r}); "
            "assert target.is_file(), 'Missing output workbook'; "
            "archive=zipfile.ZipFile(target); invalid=archive.testzip(); "
            "assert 'xl/workbook.xml' in archive.namelist(), 'Missing workbook member'; archive.close(); "
            "assert invalid is None, 'Invalid workbook archive'; "
            "print(json.dumps({'status':'OUTPUT_INTEGRITY_VERIFIED','task_correctness_certified':False}))"
        )
        return {"action": {"name": "bash", "arguments": {"command": "python -c " + shlex.quote(inspection)}},
                "reason": "verify_explicit_python_save_output", "selection_model_calls": 0,
                "correctness_certified": False, "source_tool_call_id": calls[0]["id"],
                "source_feedback_sha256": digest(observed), "read_only": True, "automatic_submission": False}
    try:
        report = json.loads(observed)
    except (ValueError, TypeError):
        return abstain
    if (len(command) != 4 or PurePosixPath(command[0]).name != "sb2_apply"
            or any(not re.fullmatch(r"[A-Za-z0-9_./-]+", value) for value in command[1:])
            or PurePosixPath(command[1]).suffix != ".xlsx" or PurePosixPath(command[2]).suffix != ".xlsx"
            or PurePosixPath(command[3]).suffix != ".json"
            or any(".." in PurePosixPath(value).parts for value in command[1:])
            or not isinstance(report, dict) or report.get("unmodified_zip_parts_preserved") is not True
            or report.get("official_accuracy_claim") is not False
            or type(report.get("changed_cells")) is not int or report["changed_cells"] < 1
            or not isinstance(report.get("writes"), dict) or not report["writes"]
            or any(not isinstance(report.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", report[key])
                   for key in ("input_sha256", "output_sha256"))):
        return abstain
    output_path = command[2]
    if output_path not in original_request:
        return abstain
    inspection = (
        "import hashlib,json,pathlib,zipfile; "
        f"target=pathlib.Path({output_path!r}); "
        "observed=hashlib.sha256(target.read_bytes()).hexdigest(); "
        f"assert observed == {report['output_sha256']!r}, 'Output changed after transaction'; "
        "archive=zipfile.ZipFile(target); invalid=archive.testzip(); archive.close(); "
        "assert invalid is None, 'Invalid workbook archive'; "
        "print(json.dumps({'status':'OUTPUT_INTEGRITY_VERIFIED','task_correctness_certified':False}))"
    )
    action = {"name": "bash", "arguments": {"command": "python -c " + shlex.quote(inspection)}}
    return {"action": action, "reason": "verify_explicit_transaction_output", "selection_model_calls": 0,
            "correctness_certified": False, "output_sha256": report["output_sha256"],
            "source_tool_call_id": calls[0]["id"], "source_feedback_sha256": digest(report),
            "read_only": True, "automatic_submission": False}


def deterministic_response(payload: dict, decision: dict) -> dict:
    expected = deterministic_followup(payload)
    if decision != expected or decision["action"] is None:
        raise ProtocolError("A deterministic response requires the exact supported public action")
    identity = digest({"payload": payload, "decision": decision})
    return {"id": "deterministic-" + identity, "object": "chat.completion", "created": 0,
            "model": payload["model"], "choices": [{"index": 0, "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_" + identity[:24], "type": "function", "function": {
                        "name": decision["action"]["name"],
                        "arguments": json.dumps(decision["action"]["arguments"], separators=(",", ":"))}}]}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "phiagent_origin": "deterministic_controller_no_model_inference"}


class RecoverySession:
    def __init__(self, policy: RecoveryPolicy | None = None):
        self.policy = policy or RecoveryPolicy()
        self.active = False
        self.fast_steps = 0
        self.force_original_next = False
        self.records: list[dict] = []

    def observe_response(self, response: dict, *, protocol_failed: bool = False) -> None:
        if self.records and self.records[-1]["recovery_request"]:
            truncated = any(choice.get("finish_reason") == "length" for choice in response.get("choices", []))
            self.force_original_next = truncated or protocol_failed

    def project(self, payload: dict) -> tuple[dict, dict]:
        projected = deepcopy(payload)
        prefix, rounds = public_rounds(payload["messages"])
        latest = feedback(rounds[-1]) if rounds else ""
        error = any(marker in latest.lower() for marker in ERROR_MARKERS)
        repeated = (
            len(rounds) >= 2 and bool(latest)
            and action_identity(rounds[-1]) == action_identity(rounds[-2])
            and latest == feedback(rounds[-2])
        )
        early_continuation = (
            self.policy.activation_after_rounds is not None
            and len(rounds) >= self.policy.activation_after_rounds
            and bool(latest)
        )
        reason = "original_before_public_stall"
        recover = False
        refresh_due = self.policy.periodic_refresh and self.fast_steps >= self.policy.maximum_fast_steps
        if self.force_original_next:
            reason = "original_after_incomplete_recovery_response"
            self.force_original_next = False
            self.fast_steps = 0
        elif self.active and self.fast_steps and error:
            reason = "original_after_recovery_error"
            self.fast_steps = 0
        elif self.active and refresh_due:
            reason = "original_periodic_refresh"
            self.fast_steps = 0
        elif self.active or error or repeated or early_continuation:
            continuing = self.active
            self.active = True
            self.fast_steps += 1
            recover = True
            reason = (
                "typed_public_error" if error else
                "typed_repeated_feedback" if repeated else
                "typed_recovery_continuation" if continuing else
                "typed_bounded_continuation"
            )
            retained = rounds[-self.policy.recent_rounds:] if self.policy.compact_history else rounds
            prefix = deepcopy(prefix)
            instruction = RECOVERY_INSTRUCTION if error or repeated else CONTINUATION_INSTRUCTION
            if self.policy.activation_after_rounds is None:
                instruction = RECOVERY_INSTRUCTION
            if prefix[0]["role"] == "system" and isinstance(prefix[0].get("content"), str):
                prefix[0]["content"] += "\n\n" + instruction
            else:
                prefix.insert(0, {"role": "system", "content": instruction})
            projected["messages"] = deepcopy([*prefix, *(message for turn in retained for message in turn)])
            for key in ("max_tokens", "max_completion_tokens"):
                if key in payload:
                    if type(payload[key]) is not int or payload[key] < 1:
                        raise ProtocolError("Invalid native output-token allowance")
                    if self.policy.maximum_output_tokens is not None:
                        projected[key] = min(payload[key], self.policy.maximum_output_tokens)
            if "max_tokens" not in payload and "max_completion_tokens" not in payload:
                raise ProtocolError("Recovery requires an explicit native output-token allowance")
        record = {
            "schema_version": SCHEMA,
            "reason": reason,
            "recovery_active": self.active,
            "recovery_request": recover,
            "native_rounds": len(rounds),
            "retained_rounds": min(len(rounds), self.policy.recent_rounds)
            if recover and self.policy.compact_history else len(rounds),
            "request_sha256": digest(payload),
            "projected_sha256": digest(projected),
            "source_history_mutated": False,
            "extra_model_calls": 0,
            "typesafe_api_used": False,
            "calibration_claimed": False,
        }
        self.records.append(record)
        return projected, record