"""Regression checks for schema-bound authorship and complete fallback charging."""
import json
from types import SimpleNamespace
import unittest
from chisel.harness.gaia2_structured_bank_v9 import generate
from chisel.harness.gaia2_fast_public_contract_v12 import ground_fast_messages


def tools():
    return {'Store__read':SimpleNamespace(description='Read an observed item.',output_type='string',inputs={'item_id':{'type':'string'}})}


def provider(responses):
    requests=[];records=[]
    def transport(endpoint,payload,**kwargs):
        requests.append(payload)
        body,inputs,outputs=responses.pop(0)
        return {'model':'pinned-model','choices':[{'message':{'content':body}}],
                'usage':{'prompt_tokens':inputs,'completion_tokens':outputs,'total_tokens':inputs+outputs}}
    return SimpleNamespace(config={'model':{'served_name':'pinned-model'}},endpoint=lambda:'metered/actor',
                           transport=transport,record=records.append),requests,records


class StructuredBankTests(unittest.TestCase):
    def test_single_schema_bound_cot_includes_readable_list_binding_contract(self):
        envelope={'reasoning':'Read the user-supplied identifier.','action':'Store__read','action_input':{'item_id':'literal-id'},
                  'jev_bank':{'note':'There is no further requested operation.','candidates':[]}}
        p,requests,records=provider([(json.dumps(envelope),100,60)])
        messages=[{'role':'system','content':'Native tool description.'},{'role':'user','content':'Read literal-id.'}]
        text,original,usage=generate(p,messages,tools(),['end_action'])
        assert len(requests)==1 and 'response_format'in requests[0] and 'stop'not in requests[0]
        contract=requests[0]['messages'][-1]['content']
        assert '"foreach"'in contract and '"$item"'in contract and '"$receipt"'in contract
        assert messages[-1]['content']=='Read literal-id.'
        assert 'literal-id'in text and json.loads(original)==envelope and usage['total_tokens']==160
        assert usage['completion_duration']>=0
        assert any(r.get('bank_field_present')for r in records)


    def check_invalid_envelope(self,invalid):
        fallback='Thought: Use the actual identifier.\nAction:\n{"action":"Store__read","action_input":{"item_id":"literal-id"}}'
        p,requests,records=provider([(invalid,100,40),(fallback,120,50)])
        text,original,usage=generate(p,[{'role':'system','content':'Native tools.'},{'role':'user','content':'Read literal-id.'}],tools(),['end_action'])
        assert len(requests)==2 and 'response_format'in requests[0] and 'response_format'not in requests[1]
        assert requests[1]['stop']==['end_action'] and original==fallback and 'literal-id'in text
        assert usage['prompt_tokens']==220 and usage['completion_tokens']==90 and usage['total_tokens']==310
        assert usage['completion_duration']>=0
        assert len([r for r in records if r.get('source')=='structured_bank_parse_fallback_v9'])==1

    def test_truncated_and_null_envelopes_keep_complete_costs(self):
        for invalid in ['{"reasoning":',None]:
            with self.subTest(invalid=invalid):
                self.check_invalid_envelope(invalid)

    def test_fast_input_has_exact_native_semantics_and_observed_return(self):
        state={'choices':{'A':{'action':{'action':'Store__read','action_input':{'item_id':'literal-id'}}}},'latest_observation':'{"uid":"literal-id"}'}
        messages=[{'role':'system','content':'Select a supported label.'},{'role':'user','content':json.dumps(state)}]
        call=SimpleNamespace(get_type=lambda:'tool_call',tool_name='Store__list',tool_arguments={'offset':0})
        observation=SimpleNamespace(get_type=lambda:'observation',content='{"uid":"literal-id"}')
        result=ground_fast_messages(messages,None,tools(),[call,observation]);actual=json.loads(result[1]['content'])
        assert actual['choices']['A']['native_tool_semantics']['description']=='Read an observed item.'
        assert actual['latest_native_return']['actual_tool_arguments']=={'offset':0}
        assert actual['latest_native_return']['actual_observation']==observation.content
        assert actual['latest_native_return']['write_success_not_inferred']
        assert json.loads(messages[1]['content'])==state

    def test_fast_input_does_not_invent_a_return_without_public_pair(self):
        messages=[{'role':'system','content':'Select a supported label.'},{'role':'user','content':json.dumps({'choices':{'A':{'action':{'action':'Store__read','action_input':{'item_id':'literal-id'}}}}})}]
        result=ground_fast_messages(messages,None,tools(),[])
        assert 'latest_native_return'not in json.loads(result[1]['content'])

if __name__=="__main__":
    unittest.main()
