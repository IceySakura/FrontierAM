import httpx
import pytest

from nexus.extensions.tasks.search import utils as search_utils


RUN_ID_ONE = 'search_00001'
RUN_ID_TWO = 'search_00002'
KEEP_RUN_ID = 'keep_me'
REWARD_URL = 'https://reward.example'
RESPONSES_MODEL = 'responses/api_azure_openai_gpt-5'
FLAT_TABLE_TEXT = 'company=Acme | revenue=10 || company=Beta | revenue=20'
MARKDOWN_TABLE_TEXT = '```markdown\n| company | revenue |\n| --- | --- |\n| Acme | 10 |\n```'
PLAIN_PROSE_TEXT = 'just plain prose'


def test_filter_excluded_results_matches_raw_and_normalized_instance_ids() -> None:
    results = [
        {'run_id': f'prefix_{RUN_ID_ONE}', 'config': {'task_config': {}}, 'metadata': {}},
        {'run_id': RUN_ID_TWO, 'config': {'task_config': {'instance_id': RUN_ID_TWO}}, 'metadata': {}},
        {'run_id': KEEP_RUN_ID, 'config': {'task_config': {'instance_id': KEEP_RUN_ID}}, 'metadata': {}},
    ]

    filtered = search_utils.filter_excluded_results(results, {RUN_ID_ONE, RUN_ID_TWO})

    assert [result['run_id'] for result in filtered] == [KEEP_RUN_ID]


def test_unified_to_dataframe_supports_flat_string_and_markdown_table() -> None:
    flat_df = search_utils.unified_to_dataframe(FLAT_TABLE_TEXT)
    markdown_df = search_utils.unified_to_dataframe(MARKDOWN_TABLE_TEXT)

    assert flat_df.to_dict(orient='records') == [
        {'company': 'Acme', 'revenue': '10'},
        {'company': 'Beta', 'revenue': '20'},
    ]
    assert markdown_df.to_dict(orient='records') == [{'company': 'Acme', 'revenue': 10}]


def test_unified_to_dataframe_rejects_text_without_table() -> None:
    with pytest.raises(ValueError):
        search_utils.unified_to_dataframe(PLAIN_PROSE_TEXT)


def test_extract_reward_score_and_error_context_include_request_ids() -> None:
    score = search_utils.extract_reward_score(
        {
            'data': {
                'request_id': 'req-1',
                'unique_request_id': 'uniq-1',
                'route_info': {'class1': 'a', 'class2': 'b', 'version': 'v1'},
                'reward': {'error_code': 0, 'final_score': 0.8},
            }
        }
    )

    request = httpx.Request('POST', REWARD_URL)
    response = httpx.Response(
        500,
        request=request,
        json={'data': {'request_id': 'req-2', 'unique_request_id': 'uniq-2'}},
    )

    assert score == 0.8
    assert search_utils.extract_error_context(response) == ' (request_id=req-2, unique_request_id=uniq-2)'


def test_resolve_openai_responses_reasoning_effort_parses_string_and_dict() -> None:
    parsed_dict = search_utils._resolve_openai_responses_reasoning_effort(
        RESPONSES_MODEL,
        '{"effort": "medium", "summary": "detailed"}',
    )
    parsed_string = search_utils._resolve_openai_responses_reasoning_effort(
        RESPONSES_MODEL,
        'high',
    )

    assert parsed_dict == {'effort': 'medium', 'summary': 'detailed'}
    assert parsed_string == {'effort': 'high', 'summary': 'auto'}


def test_safe_parse_imports_handles_literal_strings_and_iterables() -> None:
    class FakeArray:
        def tolist(self):
            return ['nexus.extensions', 'custom.module']

    assert search_utils._safe_parse_imports("['a', 'b']") == ['a', 'b']
    assert search_utils._safe_parse_imports(FakeArray()) == ['nexus.extensions', 'custom.module']
    assert search_utils._safe_parse_imports(None) == []
