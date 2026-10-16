"""Extracted synthetic controller unit tests; no benchmark or provider calls."""
import json
import math
import unittest

from chisel.learning.alfworld_jev import LOCAL_DEFER, LocalGate, local_choice_messages, select_local_decision, RuleMemory, grounded_interaction, progress_choices, progress_choice_messages

def local_response(probabilities=None, selected="B"):
    probabilities = probabilities or {"A": 0.005, "B": 0.99, LOCAL_DEFER: 0.005}
    top = [{"token": label, "logprob": math.log(value)} for label, value in probabilities.items()]
    return {"model": "synthetic-qwen", "usage": {"prompt_tokens": 99, "completion_tokens": 1, "total_tokens": 100},
            "choices": [{"message": {"content": selected}, "logprobs": {"content": [
                {"token": selected, "logprob": math.log(probabilities[selected]), "top_logprobs": top}]}}]}

class RuleMemoryTests(unittest.TestCase):
    def test_stage_retrieval_keeps_exact_source_spans_and_guardrails(self):
        source = ("# Synthetic skill\n1. Take the exact target from its source.\n"
                  "2. Heat the held target with the microwave.\n"
                  "3. Move the target to the requested destination.\n"
                  "Never substitute a distractor.\nKeep track of observations.")
        memory = RuleMemory(source)
        heat = memory.retrieve("heat")
        self.assertEqual([row["text"] for row in heat], [
            "Heat the held target with the microwave.", "Never substitute a distractor."])
        self.assertNotEqual(heat, memory.retrieve("place"))
        for row in memory.entries:
            self.assertEqual(source[row["start"]:row["end"]], row["text"])
        self.assertEqual(memory.record()["source"], source)
        self.assertEqual(memory.record()["model_calls"], 0)
        self.assertLess(sum(len(row["text"]) for row in heat), len(source))
    def test_memory_does_not_invent_missing_rules_or_accept_unknown_stage(self):
        memory = RuleMemory("Inspect observations carefully.")
        self.assertEqual(memory.retrieve("cool"), [])
        with self.assertRaises(ValueError):
            memory.retrieve("hidden-evaluator-state")
        with self.assertRaises(ValueError):
            RuleMemory("")
    def test_search_and_ambiguous_targets_skip_discriminator(self):
        arguments = {"task": "put a mug in cabinet", "observation": "On table 1, you see a mug 1, and a mug 2.",
                     "actions": ["look", "go to cabinet 1"], "history": []}
        self.assertFalse(grounded_interaction(**arguments)["eligible"])
        arguments["actions"] += ["take mug 1 from table 1", "take mug 2 from table 1"]
        self.assertEqual(grounded_interaction(**arguments)["reason"], "cot_memory_ambiguous_interaction")
    def test_transform_prerequisite_comes_from_public_feedback_not_action_text(self):
        arguments = {"task": "put a clean mug in cabinet", "observation": "You arrive at cabinet 1.",
                     "actions": ["look", "move mug 1 to cabinet 1"],
                     "history": [{"action": "take mug 1 from table 1", "observation": "You pick up the mug 1 from the table 1."}]}
        self.assertFalse(grounded_interaction(**arguments)["eligible"])
        arguments["history"].append({"action": "clean mug 1 with sinkbasin 1", "observation": "Nothing happens."})
        self.assertFalse(grounded_interaction(**arguments)["eligible"])
        arguments["history"][-1]["observation"] = "You clean the mug 1 using the sinkbasin 1."
        routine = grounded_interaction(**arguments)
        self.assertEqual((routine["stage"], routine["grounded_action"]), ("place", "move mug 1 to cabinet 1"))
    def test_typed_memory_request_keeps_all_options_and_only_retrieved_rules(self):
        source = "Take the exact target. Heat it in the microwave. Move it to the requested destination. Never substitute a distractor."
        memory = RuleMemory(source)
        actions = ["look", "take mug 1 from table 1", "go to cabinet 1"]
        observation = "On table 1, you see a mug 1."
        routine = grounded_interaction(task="put a mug in cabinet", observation=observation, actions=actions, history=[])
        messages, choices = local_choice_messages(task="put a mug in cabinet", skill=source, observation=observation,
            actions=actions, history=[], last_cot_note="Take the visible mug.", rule_memory=memory, routine=routine)
        state = json.loads(messages[1]["content"])
        self.assertNotIn("skill", state)
        self.assertNotIn("grounded_action", state)
        self.assertEqual(state["rules"], ["Take the exact target.", "Never substitute a distractor."])
        self.assertEqual(set(choices.values()), {None, *actions})
        self.assertNotIn("Heat it", messages[1]["content"])
        self.assertNotIn("Move it", messages[1]["content"])
    def test_temperature_updates_replace_incompatible_public_state(self):
        history = [
            {"action": "take mug 1 from table 1", "observation": "You pick up the mug 1 from the table 1."},
            {"action": "heat mug 1 with microwave 1", "observation": "You heat the mug 1 using the microwave 1."},
            {"action": "cool mug 1 with fridge 1", "observation": "You cool the mug 1 using the fridge 1."},
        ]
        result = grounded_interaction(task="put a hot mug in cabinet", observation="You arrive at cabinet 1.",
            actions=["look", "move mug 1 to cabinet 1"], history=history)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["public_facts"]["properties"]["mug 1"], ["cool"])
    def test_hidden_fields_and_unknown_transitions_cannot_establish_state(self):
        arguments = {"task": "put a mug in cabinet", "observation": "You arrive at cabinet 1.",
                     "actions": ["look", "move mug 1 to cabinet 1"], "history": [
                         {"action": "take mug 1 from table 1", "observation": "Unknown response.",
                          "inventory": ["mug 1"], "reward": 1}]}
        self.assertEqual(grounded_interaction(**arguments)["reason"], "cot_memory_unknown_transition")
        arguments["history"] = [{"action": "look", "observation": "You arrive at cabinet 1.", "inventory": ["mug 1"]}]
        self.assertFalse(grounded_interaction(**arguments)["eligible"])

class TypedProgressTests(unittest.TestCase):
    initial = ("You are in the middle of a room. Looking quickly around you, you see a cabinet 1, "
               "a cabinet 2, a table 1, a microwave 1, and a fridge 1. Your task is to: put a hot mug in cabinet.")
    def choices(self, history, actions, task="put a hot mug in cabinet"):
        return progress_choices(task=task, initial_observation=self.initial, history=history, actions=actions)
    def test_repeated_search_excludes_only_publicly_inspected_places(self):
        history = [{"action": "go to cabinet 1", "observation": "You arrive at cabinet 1. In the cabinet 1, you see nothing."}]
        result = self.choices(history, ["look", "go to cabinet 1", "go to cabinet 2", "go to table 1"])
        self.assertEqual(result["allowed_actions"], ["go to cabinet 2", "go to table 1"])
        self.assertIn("go to cabinet 1", result["complete_native_actions"])
        self.assertEqual(result["public_state"]["inspected_locations"], ["cabinet 1"])
    def test_known_target_source_beats_new_search(self):
        history = [{"action": "go to table 1", "observation": "You arrive at table 1. On the table 1, you see a mug 1."},
                   {"action": "go to cabinet 1", "observation": "You arrive at cabinet 1. In the cabinet 1, you see nothing."}]
        result = self.choices(history, ["look", "go to table 1", "go to cabinet 2"])
        self.assertEqual(result["allowed_actions"], ["go to table 1"])
        self.assertEqual(result["reason"], "navigate_known_source")
    def test_explicit_transform_commits_state_and_stops_reheating(self):
        history = [{"action": "take mug 1 from table 1", "observation": "You pick up the mug 1 from the table 1."}]
        actions = ["heat mug 1 with microwave 1", "go to microwave 1", "go to cabinet 1"]
        self.assertEqual(self.choices(history, actions)["allowed_actions"], [actions[0]])
        history.append({"action": actions[0], "observation": "You heat the mug 1 using the microwave 1."})
        self.assertEqual(self.choices(history, actions)["allowed_actions"], ["go to cabinet 1"])
        self.assertEqual(self.choices(history, ["open cabinet 1", "go to cabinet 1"])["allowed_actions"], ["open cabinet 1"])
    def test_finite_choices_cannot_invent_actions_or_use_hidden_rewards(self):
        history = [{"action": "look", "observation": "You are facing the table 1. Next to it, you see nothing.",
                    "reward": 1, "expert_plan": ["go to cabinet 2"]}]
        result = self.choices(history, ["look", "go to cabinet 1", "go to cabinet 2"])
        messages, choices = progress_choice_messages(task="put a hot mug in cabinet", skill="unchanged skill",
                                                      observation="Public state.", progress=result)
        self.assertEqual(set(choices.values()), {"go to cabinet 1", "go to cabinet 2", None})
        self.assertNotIn("expert_plan", messages[1]["content"])
        result["allowed_actions"].append("go to secret place 9")
        with self.assertRaises(ValueError):
            progress_choice_messages(task="goal", skill="skill", observation="public", progress=result)
    def test_unsupported_goals_abstain_without_inventing_state(self):
        self.assertFalse(self.choices([], ["look"], task="unsupported complex goal")["eligible"])

class LocalDecisionTests(unittest.TestCase):
    def payload(self):
        return local_choice_messages(task="synthetic task", skill="original CHISEL skill", observation="Public state.",
            actions=["look", "go to table 1"], history=[{"action": "look", "observation": "Public.", "reward": 999}],
            last_cot_note="Go to the observed table.")
    def test_same_model_typed_decision_uses_provider_logprobs_not_self_report(self):
        messages, choices = self.payload()
        result = select_local_decision(local_response(), choices, expected_model="synthetic-qwen", gate=LocalGate())
        self.assertEqual(result["action"], "go to table 1")
        self.assertEqual(result["route"], "local")
        self.assertFalse(result["true_jev_used"])
        self.assertFalse(result["calibration_claimed"])
        self.assertEqual(result["usage"]["total_tokens"], 100)
        self.assertNotIn("reward", messages[1]["content"])
        self.assertEqual(json.loads(messages[1]["content"])["skill"], "original CHISEL skill")
    def test_uncertainty_or_abstention_reuses_cot(self):
        _, choices = self.payload()
        uncertain = local_response({"A": 0.4, "B": 0.55, LOCAL_DEFER: 0.05})
        abstained = local_response({"A": 0.005, "B": 0.005, LOCAL_DEFER: 0.99}, LOCAL_DEFER)
        for value in (uncertain, abstained):
            result = select_local_decision(value, choices, expected_model="synthetic-qwen", gate=LocalGate())
            self.assertEqual(result["route"], "cot")
            self.assertIsNone(result["action"])
    def test_missing_logprobs_or_incomplete_support_is_not_accepted(self):
        _, choices = self.payload()
        values = []
        missing = local_response()
        missing["choices"][0].pop("logprobs")
        values.append(missing)
        incomplete = local_response()
        incomplete["choices"][0]["logprobs"]["content"][0]["top_logprobs"].pop()
        values.append(incomplete)
        for value in values:
            with self.assertRaises(ValueError):
                select_local_decision(value, choices, expected_model="synthetic-qwen", gate=LocalGate())
    def test_unnormalized_native_scores_are_charged_and_fall_back_without_rescaling(self):
        _, choices = self.payload()
        value = local_response({"A": 0.005, "B": 0.5, LOCAL_DEFER: 0.005})
        result = select_local_decision(value, choices, expected_model="synthetic-qwen", gate=LocalGate())
        self.assertEqual(result["route"], "cot")
        self.assertEqual(result["reason"], "local_logprob_distribution_not_normalized")
        self.assertAlmostEqual(result["observed_probability_mass"], 0.51)
        self.assertIsNone(result["probabilities"])
        self.assertIsNone(result["action"])
        self.assertEqual(result["usage"]["total_tokens"], 100)
    def test_vocabulary_scores_outside_legal_labels_force_cot_without_normalizing(self):
        _, choices = self.payload()
        value = local_response()
        value["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0]["token"] = "Go"
        decision = select_local_decision(value, choices, expected_model="synthetic-qwen", gate=LocalGate())
        self.assertEqual(decision["route"], "cot")
        self.assertIsNone(decision["action"])
        self.assertIsNone(decision["probabilities"])
        self.assertEqual(decision["reason"], "local_logprob_support_outside_allowed_labels")
        self.assertEqual(decision["usage"]["total_tokens"], 100)
    def test_wrong_model_multitoken_or_nan_cannot_qualify(self):
        _, choices = self.payload()
        for key, value in (("model", "different-model"),
                           ("usage", {"prompt_tokens": 99, "completion_tokens": 2, "total_tokens": 101})):
            changed = local_response()
            changed[key] = value
            with self.assertRaises(ValueError):
                select_local_decision(changed, choices, expected_model="synthetic-qwen", gate=LocalGate())
        changed = local_response()
        changed["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0]["logprob"] = float("nan")
        with self.assertRaises(ValueError):
            select_local_decision(changed, choices, expected_model="synthetic-qwen", gate=LocalGate())
    def test_large_legal_action_inventory_cannot_be_silently_shortlisted(self):
        with self.assertRaises(ValueError):
            local_choice_messages(task="synthetic", skill="skill", observation="public", history=[],
                actions=[f"go to place {index}" for index in range(63)], last_cot_note="prior plan")
