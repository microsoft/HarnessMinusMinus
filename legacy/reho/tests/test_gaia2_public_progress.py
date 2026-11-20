"""Invented public-tool fixtures; no official data or evaluator imports."""
import json
from chisel.harness.gaia2_jev import action_object
from chisel.harness.gaia2_public_progress import NativePublicProgress, email_page

TOOLS = {"Mail__list_emails"}


def page(start, end, total=80):
    return f"ReturnedEmails(emails=[], emails_range=({start}, {end}), total_returned_emails=11, total_emails={total})"


def initial():
    return 'Thought: Inspect.\nAction:\n' + json.dumps({"action": "Mail__list_emails", "action_input": {"offset": 0, "limit": 10}})


def test_native_ranges_continue_past_three_steps_and_stop_at_end():
    c = NativePublicProgress(task="Summarize all emails", stalled=True)
    c.remember(initial(), TOOLS)
    for start in range(0, 70, 10):
        action, record = c.decide(observation=page(start, start+10), error=False, intervened=False, legal_tools=TOOLS)
        assert action_object(action)['action_input']['offset'] == start+10
        assert record['selection_model_calls'] == 0
        assert not record['periodic_refresh']
    action, _ = c.decide(observation=page(70, 80), error=False, intervened=False, legal_tools=TOOLS)
    assert action is None


def test_stall_latches_only_on_repeated_public_action_feedback():
    c = NativePublicProgress(task="Read all emails")
    c.remember(initial(), TOOLS)
    assert c.decide(observation=page(0, 10), error=False, intervened=False, legal_tools=TOOLS)[0] is None
    c.remember(initial(), TOOLS)
    assert c.decide(observation=page(0, 10), error=False, intervened=False, legal_tools=TOOLS)[0] is not None


def test_unsupported_tasks_events_errors_or_ranges_restore_model():
    for change in ({'error': True}, {'intervened': True}, {'legal_tools': set()}, {'observation': page(10, 20)},
                   {'observation': 'unstructured text'}, {'observation': page(0, 80)}):
        c = NativePublicProgress(task="Read all emails", stalled=True)
        c.remember(initial(), TOOLS)
        args = dict(observation=page(0, 10), error=False, intervened=False, legal_tools=TOOLS)
        assert c.decide(**(args | change))[0] is None
    c = NativePublicProgress(task="Read the latest email", stalled=True)
    c.remember(initial(), TOOLS)
    assert c.decide(observation=page(0, 10), error=False, intervened=False, legal_tools=TOOLS)[0] is None


def test_metadata_parser_does_not_execute_calls_or_accept_duplicate_fields():
    assert email_page("ReturnedEmails(emails=[], emails_range=(0,10), total_returned_emails=11, total_emails=__import__('os').system('false'))") is None
    assert email_page("ReturnedEmails(emails=[], emails_range=(0,10), total_returned_emails=11, total_emails=20, total_emails=30)") is None
    assert email_page("ReturnedEmails(emails=[Email(content='emails_range=(50,60)')], emails_range=(0,10), total_returned_emails=11, total_emails=20)") == (0,10,20)


def test_native_engine_uses_public_candidates_without_followup_prompt_or_refresh():
    from types import SimpleNamespace
    from chisel.harness.gaia2_jev import Gaia2Engine
    def log(kind, content):
        return SimpleNamespace(get_type=lambda: kind, content=content)
    agent = SimpleNamespace(logs=[log('task', 'Read all emails')], agent_id='synthetic',
                            action_executor=SimpleNamespace(tools=TOOLS))
    calls, decisions = [], []
    def model(messages, **kwargs):
        calls.append(messages)
        return initial(), {'total_tokens': 10}
    engine = Gaia2Engine(agent, model, mode='jev', skill='Inspect public progress.', record=decisions.append,
                         recovery_policy='native_public_progress_no_refresh')
    prompt = [{'role': 'system', 'content': 'Native instructions'}]
    engine(prompt)
    agent.logs.append(log('error', 'Synthetic prior action failed'))
    engine(prompt)
    for start in range(0, 50, 10):
        agent.logs.append(log('observation', page(start,start+10)))
        text, usage = engine(prompt)
        assert action_object(text)['action_input']['offset'] == start+10
        assert usage['total_tokens'] == 0
    assert len(calls) == 2
    assert all('followups' not in row[0]['content'] for row in calls)
    agent.logs.append(log('task','Read only the latest email'))
    engine(prompt)
    assert not engine.controller.stalled
    assert len(calls) == 3
