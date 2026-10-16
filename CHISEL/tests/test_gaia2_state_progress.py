"""Invented native observations only; no benchmark or evaluator."""
import json
from chisel.harness.gaia2_state_progress import StatePublicProgress, contact_page, enumeration_goal
from chisel.harness.gaia2_jev import action_object

TOOLS={'Contacts__get_contacts','Emails__list_emails'}
def response(offset=0):
    return 'Thought: Read.\nAction:\n'+json.dumps({'action':'Contacts__get_contacts','action_input':{'offset':offset}})
def page(start=0,end=10,total=30):
    return "{'contacts': [Contact(first_name='Fixture', gender=<Gender.UNKNOWN: 'Unknown'>)], 'metadata': {'range': (%d, %d), 'total': %d}}"%(start,end,total)

def test_native_contact_metadata_rejects_content_spoofing():
    assert contact_page(page())==(0,10,30)
    assert contact_page("{'contacts': [\"'metadata': {'range': (0,10), 'total': 90}\"], 'metadata': {'range': (0,10), 'total': 20}}")==(0,10,20)
    assert contact_page("{'contacts': [\"'metadata': {'range': (0,10), 'total': 90}\"]}") is None
    for bad in [page(10,0),page(0,40),"{'metadata': {'range': (0,10), 'total': __import__('os').system('false')}}"]:
        assert contact_page(bad) is None

def test_proactive_supported_progress_and_terminal_handoff():
    c=StatePublicProgress(task='List all my contacts');c.remember(response(),TOOLS)
    for start in (0,10):
        action,record=c.decide(observation=page(start,start+10),error=False,intervened=False,legal_tools=TOOLS)
        assert action_object(action)['action_input']['offset']==start+10
        assert record['selection_model_calls']==0 and all(record['checks'].values())
    assert c.decide(observation=page(20,30),error=False,intervened=False,legal_tools=TOOLS)[0] is None
    assert len(c.pages[next(iter(c.pages))])==3

def test_memory_blocks_revisit_and_changed_world_invalidates_ranges():
    c=StatePublicProgress(task='List all contacts');c.remember(response(),TOOLS)
    args=dict(observation=page(),error=False,intervened=False,legal_tools=TOOLS)
    assert c.decide(**args)[0]
    c.remember(response(),TOOLS)
    assert c.decide(**args)[0] is None
    c.remember(response(),TOOLS)
    assert c.decide(**(args|{'intervened':True}))[0] is None
    c.remember(response(),TOOLS)
    assert c.decide(**args)[0]

def test_stall_and_state_controls_isolate_their_components():
    c=StatePublicProgress(task='List all contacts',require_stall=True);c.remember(response(),TOOLS)
    assert c.decide(observation=page(),error=False,intervened=False,legal_tools=TOOLS)[0] is None
    c=StatePublicProgress(task='List all contacts',persistent_memory=False)
    for _ in range(2):
        c.remember(response(),TOOLS)
        assert c.decide(observation=page(),error=False,intervened=False,legal_tools=TOOLS)[0]

def test_unsupported_goals_or_native_inputs_return_to_actor():
    for task in ['Read one contact','Delete all emails and look up a contact','List the latest contact']:
        assert not enumeration_goal(task,'contacts')
    for override in [{'error':True},{'intervened':True},{'legal_tools':set()},{'observation':page(10,20)}]:
        c=StatePublicProgress(task='List all contacts');c.remember(response(),TOOLS)
        args=dict(observation=page(),error=False,intervened=False,legal_tools=TOOLS)
        assert c.decide(**(args|override))[0] is None
