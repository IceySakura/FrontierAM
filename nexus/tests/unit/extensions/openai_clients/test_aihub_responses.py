from nexus.extensions.openai_clients.aihub_responses import strip_responses_item_ids


def test_strip_responses_item_ids_keeps_encrypted_content_and_call_id() -> None:
    payload = [
        {
            'role': 'assistant',
            'reasoning_items': [{'type': 'reasoning', 'id': 'rs_1', 'encrypted_content': 'enc'}],
            'provider_specific_fields': {
                'responses_output_items': [
                    {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1', 'name': 'Bash', 'arguments': '{}'}
                ]
            },
        }
    ]

    stripped = strip_responses_item_ids(payload)

    assert stripped[0]['reasoning_items'] == [{'type': 'reasoning', 'encrypted_content': 'enc'}]
    call = stripped[0]['provider_specific_fields']['responses_output_items'][0]
    assert call == {'type': 'function_call', 'call_id': 'call_1', 'name': 'Bash', 'arguments': '{}'}
    assert payload[0]['reasoning_items'][0]['id'] == 'rs_1'
