"""Round trips and reference integrity for invented public observations."""
from copy import deepcopy
import hashlib
import json

import pytest

from chisel.harness.gaia2_public_context_codec_v14 import compact, compact_request, restore
from chisel.harness.gaia2_fused_bank_v16 import project


def test_nested_dictionary_restores_every_occurrence_and_preserves_input():
    repeated = 'invented public observation ' * 80
    original = {'latest': repeated, 'history': [{'text': repeated}, repeated], 'short': 'x', 'scalar': None}
    before = deepcopy(original)
    encoded = compact(original)
    assert len(encoded['public_text_dictionary']) == 1
    assert encoded['latest'] == {'$public_text': hashlib.sha256(repeated.encode()).hexdigest()}
    assert restore(encoded) == before and original == before
    assert len(json.dumps(encoded)) < len(json.dumps(original))


@pytest.mark.parametrize('original', [[], {'public_text_dictionary': {}}, {'value': {'$public_text': 'collision'}}])
def test_reference_collision_and_already_encoded_state_are_rejected(original):
    with pytest.raises(ValueError):
        compact(original)


def test_tampered_dictionary_is_rejected_instead_of_changing_evidence():
    encoded = compact({'x': 'a' * 300, 'y': 'a' * 300})
    key = next(iter(encoded['public_text_dictionary']))
    encoded['public_text_dictionary'][key] = 'changed'
    with pytest.raises(ValueError):
        restore(encoded)


def test_request_keeps_small_full_state_and_does_not_mutate_original():
    body = {'messages': [{'role': 'system', 'content': 'Native contracts.'},
                         {'role': 'user', 'content': json.dumps({'value': 'small'})}]}
    before = deepcopy(body)
    assert compact_request(body) == before and body == before


def test_request_preserves_system_and_decodes_original_large_public_state():
    state = {'observation': 'public ' * 300, 'previous': 'public ' * 300}
    body = {'messages': [{'role': 'system', 'content': 'Native contracts.'},
                         {'role': 'user', 'content': json.dumps(state)}], 'model': 'pinned-model'}
    before = deepcopy(body)
    result = compact_request(body)
    assert restore(json.loads(result['messages'][1]['content'])) == state
    assert result['messages'][0] == body['messages'][0]
    assert result['model'] == body['model'] and body == before


def observation(step, body):
    return {'role': 'user', 'content': f'[OUTPUT OF STEP {step}] Observation:\n***\n{body}\n***\n'}


def test_fused_projection_preserves_latest_and_unique_observation_bytes():
    repeated = 'invented data ' * 100
    unique = 'different public data ' * 80
    messages = [{'role': 'system', 'content': 'Native tools.'}, {'role': 'user', 'content': 'Original task.'},
                observation(1, repeated), observation(2, unique), observation(3, repeated)]
    before = deepcopy(messages)
    result, receipt = project(messages, {}, [])
    assert messages == before and result[-1] == before[-1]
    assert result[1] == before[1] and result[3] == before[3]
    assert 'Exact duplicate' in result[2]['content']
    assert receipt['exact_duplicate_references'] == [{'message': 2, 'retained_message': 4, 'retained_step': '3'}]
    assert receipt['unique_observations_removed'] == 0


def test_fused_tool_schema_compacts_only_an_exact_catalogue_match():
    records = [{'name': 'Store__read', 'description': 'Read an item.', 'output_type': 'string',
                'inputs': {'item': {'type': 'string'}}}]
    description = 'Verbose native contract ' * 50
    messages = [{'role': 'system', 'content': 'Tools:\n' + description}, {'role': 'user', 'content': 'Read item.'}]
    result, receipt = project(messages, {'Store__read': description}, records)
    assert receipt['schema_compacted']
    packed = result[0]['content'].split('\n', 1)[1].split('\n\n', 1)[0]
    assert json.loads(packed) == list(records[0].values())
    unchanged, receipt = project(messages, {'Store__read': 'not the catalogue'}, records)
    assert not receipt['schema_compacted']
    assert unchanged[-1] == messages[-1]
