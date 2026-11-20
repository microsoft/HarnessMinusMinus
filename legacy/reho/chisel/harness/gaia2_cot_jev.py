"""CoT-authored action banks and same-backbone Jev-style finite-choice decisions."""

from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import re
import string
import time
from typing import Protocol, Sequence
from chisel.harness.gaia2_dual_rsi import combine_usage, well_typed
from chisel.harness.gaia2_public_literal_table import UnsupportedLiteral, encode_literal
from chisel.learning.alfworld_jev import LocalGate, select_local_decision

BANK_INSTRUCTION = """
Reason about the complete task and choose the next native action.
Return exactly one JSON object with reasoning, action, action_input, and jev_bank.
The reasoning string is your full deliberation. action and action_input are the
next native operation. Always include jev_bank:
{"note":"brief reusable plan and unresolved constraints",
 "candidates":[{"when":"when this next action is warranted",
 "action":"a native tool name","action_input":{}}]}.
The bank proposes alternatives for LATER public states; it does not execute a
sequence automatically. A separate finite-choice decision will select a
candidate or request fresh full reasoning. Prefer 2--4 candidates; never exceed 8.
Arguments may be literals, {"$obs":["field",0,"id"]} from the latest public
observation, {"$prev":["offset"]} from the last issued action's arguments,
or {"$add":[value1,value2]} for an explicitly planned numeric increment.
An optional foreach object {"path":["items"],"where":{"name":"literal name"}}
expands over matching public records; {"$item":["id"]} reads that record.
The expanded bank must still fit 8 actions; otherwise full reasoning is used.
Python-style public records expose their named fields, just as JSON objects do.
Use bindings for IDs that are not yet observed. Missing, ambiguous or malformed
bindings force full reasoning. Keep all task timing, filters and exceptions.
In this isolated simulator, candidates may include task-requested native reads
or state changes. Do not propose completion, waits or user-interface messages
as fast candidates. Use an empty bank when no reusable next action is justified.
This JSON envelope replaces only response serialization, not the native task,
tools or constraints. Do not return a separate Thought/Action text block.
"""

CHOICE_SYSTEM = (
    "Choose the next warranted native action from the supplied finite labels. "
    "The CoT plan and candidate conditions are hypotheses, not observations. "
    "Current public evidence overrides them. Check the requested goal, direction, "
    "timing, exceptions, argument bindings and whether an effect was already achieved. "
    "Issued actions are not by themselves proof of successful effects. "
    "Return ? if another action, missing evidence or fresh reasoning is needed. "
    "Return exactly one listed label, with no reasoning or new action text."
)

LABELS = string.ascii_uppercase[:8]

FORBIDDEN = ("complete", "finish", "wait", "sleep", "send", "reply", "delete",
             "remove", "buy", "order", "transfer", "cancel", "pay")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fast_candidate_action_names(tools, *, allow_routine_writes=False):
    forbidden = ("complete", "finish", "wait", "sleep") if allow_routine_writes else FORBIDDEN
    return sorted(
        name for name in tools
        if not name.startswith("AgentUserInterface__")
        and not any(
            name.split("__")[-1] == word or name.split("__")[-1].startswith(word + "_")
            for word in forbidden
        )
    )


def cot_response_format(tools=None, *, allow_routine_writes=False, continuation_bank=False):
    candidate = {
        "type": "object", "additionalProperties": False,
        "required": ["when", "action", "action_input"],
        "properties": {
            "when": {"type": "string"}, "action": {"type": "string"},
            "action_input": {"type": "object", "additionalProperties": True},
            "foreach": {"type": "object", "additionalProperties": True},
        },
    }
    result = {
        "type": "json_schema",
        "json_schema": {
            "name": "cot_native_action_and_jev_bank", "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "required": ["reasoning", "action", "action_input", "jev_bank"],
                "properties": {
                    "reasoning": {"type": "string"}, "action": {"type": "string"},
                    "action_input": {"type": "object", "additionalProperties": True},
                    "jev_bank": {
                        "type": "object", "additionalProperties": False,
                        "required": ["note", "candidates"],
                        "properties": {"note": {"type": "string"},
                                       "candidates": {"type": "array", "items": candidate}},
                    },
                },
            },
        },
    }
    if tools is not None:
        schema = result["json_schema"]["schema"]
        schema["properties"]["action"]["enum"] = sorted(tools)
        legal = fast_candidate_action_names(tools, allow_routine_writes=allow_routine_writes)
        candidates = schema["properties"]["jev_bank"]["properties"]["candidates"]
        if legal:
            candidates["items"]["properties"]["action"]["enum"] = legal
        else:
            candidates["maxItems"] = 0
        candidates["maxItems"] = min(candidates.get("maxItems", len(LABELS)), len(LABELS))
        if continuation_bank:
            from chisel.harness.gaia2_typed_continuation_bank import response_format
            result = response_format(result, tools, legal)
    elif continuation_bank:
        raise ValueError("Continuation banks require the actual native tool contracts")
    return result


def parse_cot_envelope(text, tools):
    value = json.loads(text)
    if (not isinstance(value, dict)
            or set(value) != {"reasoning", "action", "action_input", "jev_bank"}
            or not isinstance(value["reasoning"], str) or value["action"] not in tools
            or not isinstance(value["action_input"], dict)):
        raise ValueError("CoT did not supply its required native action and candidate-bank envelope")
    return value


def metered_request(body, config, role, native_actor):
    value = deepcopy(body)
    typed = value.pop("chisel_jev_decision", False)
    requested = value.pop("max_completion_tokens", value.get("max_tokens", 32768))
    if typed:
        labels = value.get("allowed_token_ids")
        if (typed is not True or not config.get("cot_jev", {}).get("enabled")
                or not role.endswith("/actor") or type(requested) is not int or requested != 1
                or value.get("logprobs") is not True or not isinstance(labels, list)
                or not 2 <= len(labels) <= 9 or len(set(labels)) != len(labels)
                or any(type(label) is not int or label < 0 for label in labels)
                or value.get("top_logprobs") != len(labels)):
            raise ValueError("Typed Jev requests require a complete legal-label distribution and one output token")
        maximum = 1
        component = "jev_typed"
    else:
        maximum = 4096 if native_actor else min(int(requested), 32768)
        component = "native_judge" if role.endswith("/judge") else "cot"
    return value, maximum, component


def audit_native_output_contract(request, response, config):
    role = request["role"]
    if not role.endswith(("/actor", "/judge")):
        return
    body = request["json"]
    if request.get("component") == "jev_typed":
        _, maximum, component = metered_request(
            {**body, "chisel_jev_decision": True}, config, role, True,
        )
        if (maximum != 1 or component != "jev_typed"
                or response.get("usage", {}).get("completion_tokens") != 1):
            raise ValueError("Every typed decision must charge its single control token")
    elif body.get("max_tokens") != 4096:
        raise ValueError("Native actor/judge output capacity changed")


def observation_data(text):
    def unique_object(pairs):
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate public fields")
        return dict(pairs)

    def nonfinite(value):
        raise ValueError("Nonfinite public literal: " + value)

    try:
        return json.loads(text, object_pairs_hook=unique_object, parse_constant=nonfinite)
    except (ValueError, TypeError):
        if isinstance(text, str) and re.fullmatch(r"[0-9a-f]{32}", text):
            return text
        payload = encode_literal(text)

    def unpack(value):
        if not isinstance(value, list):
            return value
        tag = value[0]
        if tag in ("l", "t"):
            return [unpack(child) for child in value[1]]
        if tag == "u":
            return -unpack(value[2]) if value[1] == "-" else unpack(value[2])
        if tag == "d":
            pairs = [(unpack(key), unpack(item)) for key, item in value[1]]
            if any(not isinstance(key, str) for key, _ in pairs) or len({key for key, _ in pairs}) != len(pairs):
                raise ValueError("Ambiguous public dictionary")
            return dict(pairs)
        if tag == "@":
            _, fields = payload["schemas"][value[1]]
            return {key: unpack(item) for key, item in zip(fields, value[2])}
        raise ValueError("Public data does not support ordered field binding")

    return unpack(payload["value"])


def at_path(value, path):
    if not isinstance(path, list) or len(path) > 8:
        raise ValueError("A bounded native field path is required")
    for key in path:
        if isinstance(value, dict) and isinstance(key, str):
            value = value[key]
        elif isinstance(value, list) and type(key) is int and 0 <= key < len(value):
            value = value[key]
        else:
            raise ValueError("Unresolved native field path")
    return deepcopy(value)


def bind(value, observation, previous, item=None):
    if isinstance(value, dict):
        operators = set(value) & {"$obs", "$prev", "$item", "$add"}
        if operators:
            if len(value) != 1:
                raise ValueError("Binding operators cannot hide additional fields")
            if "$obs" in value:
                return at_path(observation, value["$obs"])
            if "$prev" in value:
                return at_path(previous, value["$prev"])
            if "$item" in value:
                return at_path(item, value["$item"])
            terms = value["$add"]
            if not isinstance(terms, list) or len(terms) != 2:
                raise ValueError("Only explicit two-term numeric addition is supported")
            numbers = [bind(term, observation, previous, item) for term in terms]
            if any(type(number) not in (int, float) or not math.isfinite(number) for number in numbers):
                raise ValueError("Nonfinite or nonnumeric binding")
            return sum(numbers)
        return {key: bind(child, observation, previous, item) for key, child in value.items()}
    if isinstance(value, list):
        return [bind(child, observation, previous, item) for child in value]
    if isinstance(value, str) and value.startswith(("$obs", "$prev", "$item")):
        match = re.fullmatch(r"\$(obs|prev|item)((?:\.[A-Za-z_][A-Za-z0-9_]*|\[[0-9]+\])*)", value)
        if not match:
            raise ValueError("Malformed CoT binding shorthand")
        path = [
            int(index) if index else name
            for name, index in re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+)\]", match[2])
        ]
        return at_path({"obs": observation, "prev": previous, "item": item}[match[1]], path)
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("Unsupported native argument value")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Nonfinite native argument")
    return value


def parse_bank(action, tools, *, allow_routine_writes=False, continuation_bank=False):
    value = action.get("jev_bank")
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"note", "candidates"}
            or not isinstance(value["note"], str) or not 0 < len(value["note"]) <= 1200
            or not isinstance(value["candidates"], list) or len(value["candidates"]) > 8):
        raise ValueError("Invalid bounded CoT action bank")
    for candidate in value["candidates"]:
        if (not isinstance(candidate, dict)
                or set(candidate) not in (
                    {"when", "action", "action_input"},
                    {"when", "action", "action_input", "foreach"},
                )
                or not isinstance(candidate["when"], str) or not 0 < len(candidate["when"]) <= 500
                or candidate["action"] not in tools or not isinstance(candidate["action_input"], dict)):
            raise ValueError("Invalid CoT-authored candidate")
        if candidate["action"] not in fast_candidate_action_names(
            tools, allow_routine_writes=allow_routine_writes,
        ):
            raise ValueError("This action requires full CoT reasoning")
        if "foreach" in candidate:
            scope = candidate["foreach"]
            scopes = ({"path", "where"}, {"path", "where", "receipt"}) if continuation_bank else (
                {"path", "where"},
            )
            if (not isinstance(scope, dict) or set(scope) not in scopes
                    or not isinstance(scope["where"], dict)):
                raise ValueError("Invalid public-record expansion")
            if "receipt" in scope:
                from chisel.harness.gaia2_typed_continuation_bank import validate_receipt_reference
                validate_receipt_reference(scope["receipt"], with_path=False)
    return deepcopy(value)


def instantiate(bank, observation, previous, tools, seen):
    rows, failures, identities = [], [], set()
    for index, template in enumerate(bank["candidates"]):
        try:
            members = [None]
            if "foreach" in template:
                scope = template["foreach"]
                source = at_path(observation, scope["path"])
                if not isinstance(source, list):
                    raise ValueError("foreach requires a public record list")
                members = [
                    item for item in source if isinstance(item, dict)
                    and all(item.get(key) == expected for key, expected in scope["where"].items())
                ]
            for member in members:
                arguments = bind(template["action_input"], observation, previous, member)
                json.dumps(arguments, allow_nan=False)
                specifications = tools[template["action"]].inputs
                if set(arguments) - set(specifications) or any(
                    key not in arguments and "default" not in specification
                    or key in arguments and not well_typed(arguments[key], specification)
                    for key, specification in specifications.items()
                ):
                    raise ValueError("Candidate violates the declared native argument contract")
                action = {"action": template["action"], "action_input": arguments}
                fingerprint = digest(action)
                if fingerprint in identities or fingerprint in seen:
                    continue
                identities.add(fingerprint)
                rows.append({"action": action, "when": template["when"], "template": index})
        except (KeyError, IndexError, TypeError, ValueError) as error:
            failures.append({"template": index, "reason": str(error)})
    if len(rows) > len(LABELS):
        return [], failures + [{"reason": "candidate_capacity_overflow_no_truncation"}]
    return rows, failures


def public_state_candidates(bank, observation, previous, tools, seen, *, native_receipts=None):
    receipt = {
        "native_observation_sha256": hashlib.sha256(observation.encode()).hexdigest(),
        "native_observation_preserved": True,
    }
    try:
        public = observation_data(observation)
    except (SyntaxError, ValueError, TypeError, UnsupportedLiteral) as error:
        return [], [], {
            **receipt, "status": "UNKNOWN", "route": "full_cot",
            "error_type": type(error).__name__, "reason": str(error),
        }
    if native_receipts is None:
        rows, failures = instantiate(bank, public, previous, tools, seen)
    else:
        from chisel.harness.gaia2_typed_continuation_bank import instantiate as continuation
        rows, failures, used = continuation(bank, public, previous, tools, seen, native_receipts)
        receipt["bank_binding_receipts"] = used
    return rows, failures, {
        **receipt, "status": "PARSED", "public_value_type": type(public).__name__,
        "native_scalar_identifier": isinstance(public, str) and public == observation
        and re.fullmatch(r"[0-9a-f]{32}", observation) is not None,
    }


def native_fast_execution_receipt(action, tool_calls, observations, *, step_completed):
    if not step_completed or len(tool_calls) != 1 or not observations:
        return {"status": "UNVERIFIED", "reason": "Native step did not finish with one call and observation"}
    call = tool_calls[0]
    arguments = call["tool_arguments"]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError) as error:
            return {"status": "UNVERIFIED", "reason": "Native arguments are not JSON: " + str(error)}
    if call["tool_name"] != action["action"] or arguments != action["action_input"]:
        return {"status": "UNVERIFIED", "reason": "Native tool call differs from the selected CoT candidate"}
    return {
        "status": "PASS", "native_tool_name": call["tool_name"],
        "native_action_sha256": digest(action),
        "native_observation_sha256": [
            hashlib.sha256(observation.encode()).hexdigest() for observation in observations
        ],
        "original_native_step_and_tool_executor_used": True,
    }


class NativePublicLog(Protocol):
    def get_type(self) -> str: ...


def public_receipt_history(logs: Sequence[NativePublicLog]):
    receipts, diagnostics = [], []
    previous_tool = None
    for index, row in enumerate(logs):
        kind = row.get_type()
        if kind == "task":
            receipts, diagnostics, previous_tool = [], [], None
        elif kind == "tool_call":
            name = getattr(row, "tool_name", None)
            arguments = getattr(row, "tool_arguments", None)
            if not isinstance(name, str) or not isinstance(arguments, (str, dict)):
                raise ValueError("Native public tool-call fields are required for receipt history")
            previous_tool = {"tool_name": name, "tool_arguments": deepcopy(arguments)}
        elif kind == "observation":
            content = getattr(row, "content", None)
            if not isinstance(content, str):
                raise ValueError("Native public observation bytes are required for receipt history")
            checksum = hashlib.sha256(content.encode()).hexdigest()
            try:
                observation_data(content)
            except (SyntaxError, ValueError, TypeError, UnsupportedLiteral) as error:
                diagnostics.append({
                    "log_index": index, "native_observation_sha256": checksum,
                    "status": "UNKNOWN", "error_type": type(error).__name__,
                    "reason": str(error),
                    "observation": content,
                    "native_tool_call": deepcopy(previous_tool),
                })
                continue
            receipts.append({
                "log_index": index,
                "native_tool_call": deepcopy(previous_tool),
                "observation": content,
                "native_observation_sha256": checksum,
                "historical_receipt_not_current_state": True,
            })
    return receipts, diagnostics


def decision_messages(task, observation, bank, candidates, tools, previous, issued=(),
                      *, prior_public_receipts=None, prior_unknown_receipts=None,
                      bank_binding_receipts=None):
    choices = {label: json.dumps(row["action"], sort_keys=True, ensure_ascii=False)
               for label, row in zip(LABELS, candidates)}
    choices["?"] = None
    state = {
        "task": task, "latest_observation": observation, "prior_cot_note": bank["note"],
        "last_issued_action": previous,
        "issued_actions_since_cot": list(issued),
        "choices": {label: {**row, "native_contract": tools[row["action"]["action"]].inputs}
                    for label, row in zip(LABELS, candidates)},
        "unknown": "? means full CoT is required",
    }
    if prior_public_receipts is not None:
        state["prior_public_receipts"] = prior_public_receipts
        state["prior_unknown_receipts"] = (
            prior_unknown_receipts if prior_unknown_receipts is not None else []
        )
        state["receipt_scope"] = (
            "These are earlier original native tool observations in task order, not reasoning "
            "or a current-state oracle. Later observations or actions may supersede them. "
            "Check their originating calls, timing and already-observed effects; use ? when "
            "current evidence is still insufficient. UNKNOWN receipts retain original opaque "
            "feedback only and do not provide parsed fields or a binding authority."
        )
    if bank_binding_receipts is not None:
        state["bank_binding_receipts"] = bank_binding_receipts
        state["bank_receipt_scope"] = (
            "These original public tool receipts supplied the declared candidate bindings. "
            "They are historical evidence, not current-state or success guarantees. Check the "
            "latest observation and intervening issued actions; abstain if effects, timing or "
            "exceptions are unresolved. No receipt or candidate is truncated."
        )
    return [
        {"role": "system", "content": CHOICE_SYSTEM},
        {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))},
    ], choices
