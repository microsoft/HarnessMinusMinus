"""Synthetic public-feedback checks; no benchmark inputs or evaluator imports."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from chisel.harness.gaia2_jev import Gaia2Engine, PublicContinuation, action_object


TOOLS = {"Files__list", "Files__read", "final_answer", "AgentUserInterface__send_message_to_user"}


def response(followups=None):
    body = {"action": "Files__list", "action_input": {"directory": "/synthetic"}}
    if followups is not None:
        body["followups"] = followups
    return "Thought: Inspect the named file.\nAction:\n" + json.dumps(body)


def plan():
    return {"action": "Files__read", "action_input": {"file": "/synthetic/report.txt"},
            "observation_contains": ["report.txt"]}


class PublicContinuationTests(unittest.TestCase):
    def test_original_action_is_unchanged_without_plan(self):
        controller = PublicContinuation()
        self.assertEqual(controller.remember(response(), TOOLS), response())

    def test_no_stall_defers_to_original_model(self):
        controller = PublicContinuation()
        controller.remember(response([plan()]), TOOLS)
        action, record = controller.decide(observation="report.txt", error=False, intervened=False, legal_tools=TOOLS)
        self.assertIsNone(action)
        self.assertFalse(record["checks"]["stall_gate"])

    def test_repeated_public_feedback_latches_and_selects_without_model(self):
        controller = PublicContinuation()
        controller.remember(response([plan()]), TOOLS)
        controller.decide(observation="report.txt", error=False, intervened=False, legal_tools=TOOLS)
        stripped = controller.remember(response([plan()]), TOOLS)
        self.assertNotIn("followups", stripped)
        action, record = controller.decide(observation="report.txt", error=False, intervened=False, legal_tools=TOOLS)
        self.assertEqual(action_object(action)["action"], "Files__read")
        self.assertEqual(record["selection_model_calls"], 0)
        self.assertTrue(controller.stalled)

    def test_errors_missing_feedback_events_and_missing_tools_force_cot(self):
        for change in ({"error": True}, {"observation": None}, {"intervened": True},
                       {"observation": "different.txt"}, {"legal_tools": {"Files__list"}}):
            with self.subTest(change=change):
                controller = PublicContinuation(require_stall=False)
                controller.remember(response([plan()]), TOOLS)
                arguments = {"observation": "report.txt", "error": False, "intervened": False, "legal_tools": TOOLS}
                arguments.update(change)
                action, _ = controller.decide(**arguments)
                self.assertIsNone(action)
                self.assertEqual(controller.pending, [])

    def test_empty_predicates_invalid_plan_and_completion_tools_are_rejected(self):
        for changed in ({"observation_contains": []}, {"action": "final_answer"},
                        {"action": "AgentUserInterface__send_message_to_user"},
                        {"action": "Missing__tool"}, {"action": []}, {"extra": 1}, {"action_input": "not an object"}):
            with self.subTest(changed=changed):
                controller = PublicContinuation(require_stall=False)
                controller.remember(response([{**plan(), **changed}]), TOOLS)
                self.assertEqual(controller.pending, [])

    def test_periodic_refresh_is_mandatory(self):
        controller = PublicContinuation(require_stall=False)
        controller.remember(response([plan()] * 3), TOOLS)
        for _ in range(3):
            action, _ = controller.decide(observation="report.txt", error=False, intervened=False, legal_tools=TOOLS)
            self.assertIsNotNone(action)
        action, record = controller.decide(observation="report.txt", error=False, intervened=False, legal_tools=TOOLS)
        self.assertIsNone(action)
        self.assertFalse(record["checks"]["refresh_not_due"])

    def test_malformed_json_is_not_repaired_or_executed(self):
        for text in ("Action: {}", "Action: {} Action: {}", "Action: {} trailing", "Thought only"):
            self.assertIsNone(action_object(text))

    def test_configuration_is_bounded(self):
        for value in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                PublicContinuation(maximum_fast_steps=value)

    def test_native_adapter_preserves_request_and_charges_model_metadata(self):
        requests, records = [], []
        usage = {"prompt_tokens": 41, "completion_tokens": 9, "total_tokens": 50}

        def provider(messages, **kwargs):
            requests.append((messages, kwargs))
            return response(), usage

        agent = SimpleNamespace(logs=[], agent_id="synthetic", action_executor=SimpleNamespace(tools=TOOLS))
        engine = Gaia2Engine(agent, provider, mode="native", skill="synthetic rules", record=records.append)
        messages = [{"role": "system", "content": "Native system"}, {"role": "user", "content": "Synthetic task"}]
        text, actual = engine(messages, stop_sequences=["<end_action>"])
        self.assertEqual(requests, [(messages, {"stop_sequences": ["<end_action>"]})])
        self.assertEqual(actual, usage)
        self.assertEqual(text, response())

    def test_adapter_latches_error_then_preserves_native_step_for_deterministic_action(self):
        requests, records = [], []

        def provider(messages, **kwargs):
            requests.append(messages)
            return response([plan()]), {"total_tokens": 50}

        def log(kind, content):
            return SimpleNamespace(get_type=lambda: kind, content=content)

        agent = SimpleNamespace(logs=[], agent_id="synthetic", action_executor=SimpleNamespace(tools=TOOLS))
        engine = Gaia2Engine(agent, provider, mode="jev", skill="Rules", record=records.append)
        messages = [{"role": "system", "content": "Native system"}]
        engine(messages)
        self.assertNotIn("followups", requests[0][0]["content"])
        agent.logs.append(log("error", "Synthetic error"))
        engine(messages)
        self.assertIn("followups", requests[-1][0]["content"])
        agent.logs.append(log("observation", "report.txt"))
        text, metadata = engine(messages)
        self.assertEqual(len(requests), 2)
        self.assertEqual(action_object(text)["action"], "Files__read")
        self.assertEqual(metadata["total_tokens"], 0)
        self.assertFalse(records[-1]["provider_response"])
        agent.logs.append(log("task", "Different task"))
        engine(messages)
        self.assertFalse(engine.controller.stalled)
        self.assertNotIn("followups", requests[-1][0]["content"])







    def test_official_pypi_response_normalization_is_preserved(self):
        from chisel.harness.gaia2_usage import native_response_text

        text = "[[True]] False<end_action>extra"
        self.assertEqual(native_response_text(text, ["<end_action>"]), "[[true]] false")
        self.assertEqual(native_response_text(text, ["<end_action>"], judge=True), "[[True]] False")
        for verdict in ("[[True]]", "[[False]]", "[[Success]]", "[[Failure]]"):
            self.assertEqual(native_response_text(verdict, [], judge=True), verdict)
        with self.assertRaises(RuntimeError):
            native_response_text(None, [])

    def test_actor_normalization_preserves_literal_names_and_escaped_strings(self):
        from chisel.harness.gaia2_usage import native_response_text

        for literal in (
            "True North", "False Creek", "True", "False",
            'He said "True" and "False".', "First line\nTrue second line",
            r"\True\False", "",
        ):
            with self.subTest(literal=literal):
                action = {
                    "action": "Files__save",
                    "action_input": {"name": literal, "enabled": True, "flags": [False, True]},
                }
                text = "Action:\n" + json.dumps(action).replace('"enabled": true', '"enabled": True')
                normalized = native_response_text(text + "<end_action>ignored", ["<end_action>"])
                self.assertEqual(action_object(normalized), action)

    def test_actor_boolean_normalization_does_not_rewrite_parts_of_words(self):
        from chisel.harness.gaia2_usage import native_response_text

        text = "Thought: Inspect TrueNorth and Falsehood.\nAction:\n" + json.dumps({
            "action": "Files__read", "action_input": {"name": "TrueNorth"},
        })
        self.assertEqual(native_response_text(text, []), text)











if __name__ == "__main__":
    unittest.main()
