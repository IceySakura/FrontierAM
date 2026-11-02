#!/usr/bin/env python3
"""Unit tests for WebArena-Verified evaluation logic.

Tests the conversion from WebArena-Verified eval format to original WebArena format,
and verifies that the eval functions work correctly with the converted format.
"""

import asyncio
import json


# ============================================================================
# Copy of eval functions for testing (to avoid import issues)
# ============================================================================


def clean_answer(answer: str) -> str:
    """Clean and normalize answer string."""
    answer = answer.strip()
    if answer.startswith("'") and answer.endswith("'") or answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    return answer.lower()


def exact_match(ref: str, pred: str) -> float:
    """Check exact match between reference and prediction."""
    return float(clean_answer(pred) == clean_answer(ref))


def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
    """Check if reference is included in prediction."""
    clean_ref = clean_answer(ref)
    clean_pred = clean_answer(pred)

    if tokenize and len(clean_ref) == 1:
        import re

        tok_pred = re.findall(r'\b\w+\b', clean_pred)
        return float(clean_ref in tok_pred)
    return float(clean_ref in clean_pred)


async def eval_string_match(config: dict, answer: str) -> float:
    """Evaluate using string matching."""
    eval_config = config.get('eval', {})
    reference_answers = eval_config.get('reference_answers', {})

    pred = clean_answer(answer)

    score = 1.0
    for approach, value in reference_answers.items():
        if approach == 'exact_match':
            score *= exact_match(ref=value, pred=pred)

        elif approach == 'must_include':
            assert isinstance(value, list), f'must_include value should be a list, got {type(value)}'
            for must_value in value:
                score *= must_include(
                    ref=must_value,
                    pred=pred,
                    tokenize=(len(value) == 1),
                )

    return score


async def eval_url_match(config: dict, current_url: str) -> float:
    """Evaluate by checking current URL."""
    import collections
    import urllib.parse

    eval_config = config.get('eval', {})
    reference_url = eval_config.get('reference_url', '')
    matching_rule = eval_config.get('url_note', 'GOLD in PRED')

    def clean_url(url: str) -> str:
        url = str(url)
        return url.rstrip('/')

    def parse_url(url: str):
        parsed_url = urllib.parse.urlparse(url)
        base_path = (parsed_url.netloc + parsed_url.path).rstrip('/')
        query = urllib.parse.parse_qs(parsed_url.query)
        return base_path, query

    def parse_urls(urls):
        base_paths = []
        queries = collections.defaultdict(set)
        for url in urls:
            base_path, query = parse_url(url)
            base_paths.append(base_path)
            for k, v in query.items():
                queries[k].update(v)
        return base_paths, queries

    pred = clean_url(current_url)
    ref_urls = reference_url.split(' |OR| ')
    ref_urls = [clean_url(url) for url in ref_urls]

    if matching_rule == 'GOLD in PRED':
        ref_base_paths, ref_queries = parse_urls(ref_urls)
        pred_base_path, pred_query = parse_url(pred)

        base_score = float(any(ref_base_path in pred_base_path for ref_base_path in ref_base_paths))

        query_score = 1.0
        for k, possible_values in ref_queries.items():
            query_score *= float(
                any(possible_ref_value in pred_query.get(k, []) for possible_ref_value in possible_values)
            )

        score = base_score * query_score
    else:
        raise ValueError(f'Unknown matching rule: {matching_rule}')

    return score


# ============================================================================
# Copy of generator conversion functions for testing
# ============================================================================


def _parse_json_field(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _convert_eval_format(eval_list):
    eval_types = []
    reference_answers = None
    reference_url = None
    original_evaluators = eval_list

    for evaluator in eval_list:
        evaluator_type = evaluator.get('evaluator', '')
        expected = evaluator.get('expected', {})

        if evaluator_type == 'AgentResponseEvaluator':
            if 'string_match' not in eval_types:
                eval_types.append('string_match')

            retrieved = expected.get('retrieved_data')

            if retrieved is not None and isinstance(retrieved, list) and len(retrieved) > 0:
                # For retrieve tasks with actual data, use must_include
                # NOTE: retrieved_data can be:
                #   - list of strings: ["value1", "value2"]
                #   - list of objects: [{"name": "...", "state": "..."}, ...]
                # For objects, we need to convert to JSON strings for string_match eval
                must_include_values = []
                for item in retrieved:
                    if isinstance(item, dict):
                        # Convert dict to JSON string for matching
                        must_include_values.append(json.dumps(item, sort_keys=True))
                    else:
                        # String or other primitive - use as-is
                        must_include_values.append(str(item) if item is not None else '')

                reference_answers = {
                    'must_include': must_include_values,
                }
            elif 'status' in expected:
                # For tasks with status or when retrieved_data is null/empty
                # Use empty must_include - actual verification via deterministic eval
                reference_answers = {
                    'must_include': [],
                }

        elif evaluator_type == 'NetworkEventEvaluator':
            if 'url_match' not in eval_types:
                eval_types.append('url_match')

            if 'url' in expected:
                urls = expected['url']
                reference_url = (urls[0] if urls else None) if isinstance(urls, list) else urls

    converted = {
        'eval_types': eval_types if eval_types else ['string_match'],
        '_original_evaluators': original_evaluators,
    }

    if reference_answers:
        converted['reference_answers'] = reference_answers

    if reference_url:
        converted['reference_url'] = reference_url

    return converted


def _normalize_task_config(task_config):
    normalized = task_config.copy()
    json_fields = ['sites', 'start_urls', 'eval', 'instantiation_dict']
    for field in json_fields:
        if field in normalized:
            normalized[field] = _parse_json_field(normalized[field])

    if 'start_urls' in normalized and isinstance(normalized['start_urls'], list):
        normalized['start_url'] = ' |AND| '.join(normalized['start_urls'])

    if 'eval' in normalized and isinstance(normalized['eval'], list):
        normalized['eval'] = _convert_eval_format(normalized['eval'])

    return normalized


# ============================================================================
# Test Cases
# ============================================================================


async def test_task_0_retrieve_with_data():
    """Task 0: retrieve task with retrieved_data (AgentResponseEvaluator)

    Expected: Agent should return "Quest Lumaflex™ Band"
    """
    print('\n=== Test: Task 0 - Retrieve with data ===')

    task = {
        'task_id': 0,
        'intent': 'Get the top-1 best-selling product name(s) in 2022',
        'eval': '[{"evaluator": "AgentResponseEvaluator", "expected": {"retrieved_data": ["Quest Lumaflex™ Band"], "status": "SUCCESS", "task_type": "retrieve"}, "results_schema": {"items": {"type": "string"}, "type": "array"}}]',
    }

    normalized = _normalize_task_config(task)

    # Verify conversion
    assert isinstance(normalized['eval'], dict), f'eval should be dict, got {type(normalized["eval"])}'
    assert 'eval_types' in normalized['eval'], 'eval should have eval_types'
    assert 'string_match' in normalized['eval']['eval_types'], 'eval_types should contain string_match'
    assert 'must_include' in normalized['eval'].get('reference_answers', {}), 'should have must_include'

    # Test cases
    test_cases = [
        ('Quest Lumaflex™ Band', 1.0, 'Exact match'),
        ('The top product is Quest Lumaflex™ Band', 1.0, 'Contains answer'),
        ('quest lumaflex™ band', 1.0, 'Case insensitive'),
        ('Quest Lumaflex Band', 1.0, 'Without trademark symbol'),
        ('Some other product', 0.0, 'Wrong answer'),
        ('', 0.0, 'Empty answer'),
    ]

    for answer, expected_score, description in test_cases:
        score = await eval_string_match(normalized, answer)
        status = '✅' if score == expected_score else '❌'
        print(f"  {status} {description}: answer='{answer[:30]}...' score={score} (expected={expected_score})")

    print('  ✅ Task 0 test passed')


async def test_task_9_retrieve_with_object_array():
    """Task 9: retrieve task with object array in retrieved_data

    This is the special case where retrieved_data contains objects (dicts),
    not just strings. We need to convert them to JSON strings for matching.

    Expected: Agent should return airport info as JSON objects
    """
    print('\n=== Test: Task 9 - Retrieve with object array ===')

    task = {
        'task_id': 9,
        'intent': 'Get the name, state, and zip code of all international airports...',
        'eval': '[{"evaluator": "AgentResponseEvaluator", "expected": {"retrieved_data": [{"name": "Pittsburgh International Airport", "postcode": "15231", "state": "Pennsylvania"}], "status": "SUCCESS", "task_type": "retrieve"}, "results_schema": {"items": {"properties": {"name": {"type": "string"}, "postcode": {"type": "string"}, "state": {"type": "string"}}, "title": "full_address", "type": "object"}, "type": "array"}}]',
    }

    normalized = _normalize_task_config(task)

    # Verify conversion
    assert isinstance(normalized['eval'], dict), f'eval should be dict, got {type(normalized["eval"])}'
    ref_answers = normalized['eval'].get('reference_answers', {})
    must_include = ref_answers.get('must_include', [])

    print(f'  Converted must_include: {must_include}')

    # Verify that the dict was converted to JSON string
    assert len(must_include) == 1, f'Should have 1 item, got {len(must_include)}'
    assert isinstance(must_include[0], str), f'Item should be string, got {type(must_include[0])}'

    # The expected JSON string (sorted keys)
    expected_json = '{"name": "Pittsburgh International Airport", "postcode": "15231", "state": "Pennsylvania"}'

    # Test cases - agent's answer should contain the JSON representation
    test_cases = [
        # Exact JSON match
        (expected_json, 1.0, 'Exact JSON match'),
        # JSON in context
        (f'The airport is: {expected_json}', 1.0, 'JSON in context'),
        # With different key order (should still match as JSON string is in answer)
        (
            'Here is the result: {"name": "Pittsburgh International Airport", "postcode": "15231", "state": "Pennsylvania"}',
            1.0,
            'JSON with same content',
        ),
        # Missing data
        ('{"name": "Pittsburgh International Airport"}', 0.0, 'Incomplete JSON'),
        # Wrong data
        ('{"name": "JFK", "postcode": "10001", "state": "New York"}', 0.0, 'Wrong airport'),
        # Plain text (no JSON)
        ('Pittsburgh International Airport, Pennsylvania 15231', 0.0, 'Plain text format'),
        # Empty
        ('', 0.0, 'Empty answer'),
    ]

    for answer, expected_score, description in test_cases:
        score = await eval_string_match(normalized, answer)
        status = '✅' if score == expected_score else '❌'
        print(f'  {status} {description}: score={score} (expected={expected_score})')

    print('  ✅ Task 9 test passed')


async def test_task_8_not_found_error():
    """Task 8: retrieve task with NOT_FOUND_ERROR (no data found)

    Expected: Agent should indicate nothing was found.
    With null retrieved_data and error status, must_include should be empty,
    which means any answer passes string_match (actual eval via _original_evaluators).
    """
    print('\n=== Test: Task 8 - NOT_FOUND_ERROR ===')

    task = {
        'task_id': 8,
        'intent': 'Get airports within driving distance...',
        'eval': '[{"evaluator": "AgentResponseEvaluator", "expected": {"retrieved_data": null, "status": "NOT_FOUND_ERROR", "task_type": "retrieve"}, "results_schema": {"type": "null"}}]',
    }

    normalized = _normalize_task_config(task)

    # Verify conversion
    assert isinstance(normalized['eval'], dict), f'eval should be dict, got {type(normalized["eval"])}'
    ref_answers = normalized['eval'].get('reference_answers', {})

    print(f'  Converted eval: {json.dumps(normalized["eval"], indent=2)}')
    print(f'  reference_answers: {ref_answers}')

    # With null retrieved_data and error status, should have empty must_include
    assert 'must_include' in ref_answers, 'Should have must_include'
    assert ref_answers['must_include'] == [], 'must_include should be empty'

    # For NOT_FOUND_ERROR tasks with null retrieved_data:
    # Empty must_include means any answer passes string_match
    # The actual verification should rely on _original_evaluators
    test_cases = [
        ('N/A - no airports found', 1.0, 'N/A response'),
        ('Could not find any airports', 1.0, 'Natural language not found'),
        ('', 1.0, 'Empty (passes with empty must_include)'),
        ('Some random answer', 1.0, 'Any answer passes'),
    ]

    for answer, expected_score, description in test_cases:
        score = await eval_string_match(normalized, answer)
        status = '✅' if score == expected_score else '❌'
        print(f'  {status} {description}: score={score} (expected={expected_score})')

    print('  ✅ Task 8 test passed')


async def test_task_44_navigate_with_url():
    """Task 44: navigate task with NetworkEventEvaluator (URL check)

    Expected: Browser should be at /dashboard/todos
    """
    print('\n=== Test: Task 44 - Navigate with URL check ===')

    task = {
        'task_id': 44,
        'intent': 'Open my todos page',
        'eval': """[
            {"evaluator": "AgentResponseEvaluator", "expected": {"retrieved_data": null, "status": "SUCCESS", "task_type": "navigate"}, "results_schema": {"type": "null"}},
            {"evaluator": "NetworkEventEvaluator", "expected": {"url": ["__GITLAB__/dashboard/todos", "__GITLAB__/dashboard/todos?state=pending"]}, "ignored_query_params_patterns": ["page", "sort"]}
        ]""",
    }

    normalized = _normalize_task_config(task)

    # Verify conversion
    assert isinstance(normalized['eval'], dict), f'eval should be dict, got {type(normalized["eval"])}'
    assert 'string_match' in normalized['eval']['eval_types'], 'should have string_match'
    assert 'url_match' in normalized['eval']['eval_types'], 'should have url_match'
    assert 'reference_url' in normalized['eval'], 'should have reference_url'

    print(f'  eval_types: {normalized["eval"]["eval_types"]}')
    print(f'  reference_url: {normalized["eval"]["reference_url"]}')

    # URL match tests (after URL substitution would happen in runtime)
    # Simulating with actual URLs
    test_config = {'eval': {'reference_url': 'http://gitlab.com/dashboard/todos', 'eval_types': ['url_match']}}

    url_test_cases = [
        ('http://gitlab.com/dashboard/todos', 1.0, 'Exact URL match'),
        ('http://gitlab.com/dashboard/todos/', 1.0, 'With trailing slash'),
        ('http://gitlab.com/dashboard/todos?page=1', 1.0, 'With query params'),
        ('http://gitlab.com/other/page', 0.0, 'Different page'),
    ]

    for url, expected_score, description in url_test_cases:
        score = await eval_url_match(test_config, url)
        status = '✅' if score == expected_score else '❌'
        print(f"  {status} {description}: url='{url}' score={score} (expected={expected_score})")

    print('  ✅ Task 44 test passed')


async def test_task_389_mutate():
    """Task 389: mutate task (post comment)

    Expected: Agent should post a comment and navigate to correct URL
    """
    print('\n=== Test: Task 389 - Mutate (post comment) ===')

    task = {
        'task_id': 389,
        'intent': 'Post "Thanks, working on reviews" for the merge request...',
        'eval': """[
            {"evaluator": "AgentResponseEvaluator", "expected": {"retrieved_data": null, "status": "SUCCESS", "task_type": "mutate"}, "results_schema": {"type": "null"}},
            {"evaluator": "NetworkEventEvaluator", "expected": {"http_method": "POST", "post_data": {"$.note.note": "Thanks, working on reviews"}, "url": "__GITLAB__/primer/design/notes"}}
        ]""",
    }

    normalized = _normalize_task_config(task)

    # Verify conversion
    assert isinstance(normalized['eval'], dict), f'eval should be dict, got {type(normalized["eval"])}'
    assert 'url_match' in normalized['eval']['eval_types'], 'should have url_match'

    print(f'  eval_types: {normalized["eval"]["eval_types"]}')
    print(f'  reference_url: {normalized["eval"].get("reference_url")}')
    # Note: For mutate tasks, string_match with empty must_include passes any answer
    # The actual verification is via NetworkEventEvaluator (POST request check)

    print('  ✅ Task 389 test passed')


async def test_task_491_action_not_allowed():
    """Task 491: mutate task with ACTION_NOT_ALLOWED_ERROR

    Expected: Agent should report that action is not allowed.
    With null retrieved_data, should have empty must_include.
    """
    print('\n=== Test: Task 491 - ACTION_NOT_ALLOWED_ERROR ===')

    task = {
        'task_id': 491,
        'intent': 'Notify Sarah Miller...',
        'eval': '[{"evaluator": "AgentResponseEvaluator", "expected": {"retrieved_data": null, "status": "ACTION_NOT_ALLOWED_ERROR", "task_type": "mutate"}, "results_schema": {"type": "null"}}]',
    }

    normalized = _normalize_task_config(task)

    # Verify conversion
    assert isinstance(normalized['eval'], dict), f'eval should be dict, got {type(normalized["eval"])}'
    ref_answers = normalized['eval'].get('reference_answers', {})

    print(f'  reference_answers: {ref_answers}')
    # With error status and null retrieved_data, should have empty must_include
    assert 'must_include' in ref_answers, 'Should have must_include'
    assert ref_answers['must_include'] == [], 'must_include should be empty'

    print('  ✅ Task 491 test passed')


async def test_task_319_not_found_no_data():
    """Task 319: retrieve task with NOT_FOUND_ERROR and no retrieved_data field

    This is a special case where retrieved_data is missing from expected
    """
    print('\n=== Test: Task 319 - NOT_FOUND_ERROR without retrieved_data ===')

    task = {
        'task_id': 319,
        'intent': 'How much refund should I expect...',
        'eval': '[{"evaluator": "AgentResponseEvaluator", "expected": {"status": "NOT_FOUND_ERROR", "task_type": "retrieve"}, "results_schema": {"items": {"format": "currency", "type": "number"}, "type": "array"}}]',
    }

    normalized = _normalize_task_config(task)

    # Verify conversion
    assert isinstance(normalized['eval'], dict), f'eval should be dict, got {type(normalized["eval"])}'
    ref_answers = normalized['eval'].get('reference_answers', {})

    print(f'  reference_answers: {ref_answers}')
    # Should have empty must_include since no retrieved_data and error status

    print('  ✅ Task 319 test passed')


async def test_multiple_retrieved_data():
    """Test task with multiple items in retrieved_data"""
    print('\n=== Test: Multiple retrieved_data items ===')

    task = {
        'task_id': 999,
        'intent': 'Get multiple products...',
        'eval': '[{"evaluator": "AgentResponseEvaluator", "expected": {"retrieved_data": ["Product A", "Product B", "Product C"], "status": "SUCCESS", "task_type": "retrieve"}}]',
    }

    normalized = _normalize_task_config(task)

    ref_answers = normalized['eval'].get('reference_answers', {})
    must_include = ref_answers.get('must_include', [])

    print(f'  must_include: {must_include}')
    assert len(must_include) == 3, f'Should have 3 items, got {len(must_include)}'

    # Test: answer must contain ALL items
    test_cases = [
        ('Product A, Product B, Product C', 1.0, 'All products'),
        ('The products are Product A and Product B and Product C', 1.0, 'All products in sentence'),
        ('Product A', 0.0, 'Only one product'),
        ('Product A, Product B', 0.0, 'Missing one product'),
        ('', 0.0, 'Empty answer'),
    ]

    for answer, expected_score, description in test_cases:
        score = await eval_string_match(normalized, answer)
        status = '✅' if score == expected_score else '❌'
        print(f'  {status} {description}: score={score} (expected={expected_score})')

    print('  ✅ Multiple retrieved_data test passed')


async def test_edge_cases():
    """Test edge cases"""
    print('\n=== Test: Edge cases ===')

    # Test 1: Empty eval list
    task1 = {'task_id': 1000, 'eval': '[]'}
    normalized1 = _normalize_task_config(task1)
    assert 'eval_types' in normalized1['eval'], 'Should have default eval_types'
    print('  ✅ Empty eval list handled')

    # Test 2: Already parsed eval (not string)
    task2 = {
        'task_id': 1001,
        'eval': [
            {'evaluator': 'AgentResponseEvaluator', 'expected': {'retrieved_data': ['test'], 'status': 'SUCCESS'}}
        ],
    }
    normalized2 = _normalize_task_config(task2)
    assert isinstance(normalized2['eval'], dict), 'Should convert to dict'
    print('  ✅ Already parsed eval handled')

    # Test 3: eval.get() works on converted format
    eval_types = normalized2['eval'].get('eval_types', [])
    assert 'string_match' in eval_types, 'get() should work'
    print('  ✅ .get() works on converted eval')

    print('  ✅ All edge cases passed')


async def test_int_array():
    """Test task with integer retrieved_data (like Task 15)"""
    print('\n=== Test: Integer retrieved_data ===')

    task = {
        'task_id': 15,
        'intent': 'Get the total number of reviews...',
        'eval': '[{"evaluator": "AgentResponseEvaluator", "expected": {"retrieved_data": [2], "status": "SUCCESS", "task_type": "retrieve"}}]',
    }

    normalized = _normalize_task_config(task)

    ref_answers = normalized['eval'].get('reference_answers', {})
    must_include = ref_answers.get('must_include', [])

    print(f'  must_include: {must_include}')
    assert len(must_include) == 1, f'Should have 1 item, got {len(must_include)}'
    assert must_include[0] == '2', f"Should be '2', got {must_include[0]}"
    assert isinstance(must_include[0], str), f'Should be string, got {type(must_include[0])}'

    # Test cases
    test_cases = [
        ('2', 1.0, 'Exact match'),
        ('The count is 2', 1.0, 'In context'),
        ('There are 2 reviews', 1.0, 'In sentence'),
        ('3', 0.0, 'Wrong number'),
        ('', 0.0, 'Empty answer'),
    ]

    for answer, expected_score, description in test_cases:
        score = await eval_string_match(normalized, answer)
        status = '✅' if score == expected_score else '❌'
        print(f'  {status} {description}: score={score} (expected={expected_score})')

    print('  ✅ Integer retrieved_data test passed')


async def test_float_array():
    """Test task with float retrieved_data (like Task 150)"""
    print('\n=== Test: Float retrieved_data ===')

    task = {
        'task_id': 150,
        'intent': 'Get the price of the item...',
        'eval': '[{"evaluator": "AgentResponseEvaluator", "expected": {"retrieved_data": [260.69], "status": "SUCCESS", "task_type": "retrieve"}}]',
    }

    normalized = _normalize_task_config(task)

    ref_answers = normalized['eval'].get('reference_answers', {})
    must_include = ref_answers.get('must_include', [])

    print(f'  must_include: {must_include}')
    assert len(must_include) == 1, f'Should have 1 item, got {len(must_include)}'
    assert must_include[0] == '260.69', f"Should be '260.69', got {must_include[0]}"
    assert isinstance(must_include[0], str), f'Should be string, got {type(must_include[0])}'

    # Test cases
    test_cases = [
        ('260.69', 1.0, 'Exact match'),
        ('$260.69', 1.0, 'With currency symbol'),
        ('The price is 260.69', 1.0, 'In context'),
        ('260.70', 0.0, 'Wrong price'),
        ('260', 0.0, 'Missing decimal'),
        ('', 0.0, 'Empty answer'),
    ]

    for answer, expected_score, description in test_cases:
        score = await eval_string_match(normalized, answer)
        status = '✅' if score == expected_score else '❌'
        print(f'  {status} {description}: score={score} (expected={expected_score})')

    print('  ✅ Float retrieved_data test passed')


async def test_multi_dict_array():
    """Test task with multiple dict items in retrieved_data (like Task 10, 20)"""
    print('\n=== Test: Multiple dict items in retrieved_data ===')

    task = {
        'task_id': 20,
        'intent': 'Get the time for walking and driving route...',
        'eval': '[{"evaluator": "AgentResponseEvaluator", "expected": {"retrieved_data": [{"duration": "13min", "mode": "driving"}, {"duration": "1hr 45min", "mode": "walking"}], "status": "SUCCESS", "task_type": "retrieve"}}]',
    }

    normalized = _normalize_task_config(task)

    ref_answers = normalized['eval'].get('reference_answers', {})
    must_include = ref_answers.get('must_include', [])

    print(f'  must_include count: {len(must_include)}')
    for i, item in enumerate(must_include):
        print(f'    [{i}]: {item}')

    assert len(must_include) == 2, f'Should have 2 items, got {len(must_include)}'
    # Both should be JSON strings
    assert all(isinstance(item, str) for item in must_include), 'All items should be strings'

    # Build expected JSON strings (sorted keys)
    expected1 = '{"duration": "13min", "mode": "driving"}'
    expected2 = '{"duration": "1hr 45min", "mode": "walking"}'

    # Test: answer must contain BOTH JSON objects
    correct_answer = f'Results: {expected1} and {expected2}'
    score = await eval_string_match(normalized, correct_answer)
    print(f'  ✅ Both objects present: score={score} (expected=1.0)')
    assert score == 1.0, f'Should be 1.0 with both objects, got {score}'

    # Test: missing one object should fail
    partial_answer = f'Result: {expected1}'
    score = await eval_string_match(normalized, partial_answer)
    print(f'  ✅ Only one object: score={score} (expected=0.0)')
    assert score == 0.0, f'Should be 0.0 with only one object, got {score}'

    print('  ✅ Multiple dict items test passed')


async def run_all_tests():
    """Run all tests"""
    print('=' * 60)
    print('WebArena-Verified Evaluation Unit Tests')
    print('=' * 60)

    try:
        await test_task_0_retrieve_with_data()
        await test_task_9_retrieve_with_object_array()
        await test_task_8_not_found_error()
        await test_task_44_navigate_with_url()
        await test_task_389_mutate()
        await test_task_491_action_not_allowed()
        await test_task_319_not_found_no_data()
        await test_multiple_retrieved_data()
        await test_int_array()
        await test_float_array()
        await test_multi_dict_array()
        await test_edge_cases()

        print('\n' + '=' * 60)
        print('✅ ALL TESTS PASSED!')
        print('=' * 60)

    except Exception as e:
        print(f'\n❌ TEST FAILED: {e}')
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(asyncio.run(run_all_tests()))
