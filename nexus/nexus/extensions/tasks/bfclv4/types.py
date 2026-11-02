"""BFCL v4 data models and category constants for Nexus integration."""

from __future__ import annotations

from typing import Any

from nexus.core import TaskRunInput, TaskRunOutput, WithScore


# ============================================================================
# Category Constants (from bfcl_eval.constants.category_mapping)
# ============================================================================

NON_LIVE_CATEGORY = [
    'simple_python',
    'simple_java',
    'simple_javascript',
    'multiple',
    'parallel',
    'parallel_multiple',
    'irrelevance',
]

LIVE_CATEGORY = [
    'live_simple',
    'live_multiple',
    'live_parallel',
    'live_parallel_multiple',
    'live_irrelevance',
    'live_relevance',
]

MULTI_TURN_CATEGORY = [
    'multi_turn_base',
    'multi_turn_miss_func',
    'multi_turn_miss_param',
    'multi_turn_long_context',
]

WEB_SEARCH_CATEGORY = [
    'web_search_base',
    'web_search_no_snippet',
]

MEMORY_CATEGORY = [
    'memory_kv',
    'memory_vector',
    'memory_rec_sum',
]

SINGLE_TURN_CATEGORY = NON_LIVE_CATEGORY + LIVE_CATEGORY
AGENTIC_CATEGORY = MEMORY_CATEGORY + WEB_SEARCH_CATEGORY
FORMAT_SENSITIVITY_CATEGORY = ['format_sensitivity']

# all_scoring: excludes memory (high cost in time & memory) and format_sensitivity
ALL_SCORING_CATEGORIES = SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY + WEB_SEARCH_CATEGORY
# all_scoring_with_memory: includes memory categories (requires --timeout 1800)
ALL_SCORING_WITH_MEMORY_CATEGORIES = ALL_SCORING_CATEGORIES + MEMORY_CATEGORY
ALL_CATEGORIES = ALL_SCORING_WITH_MEMORY_CATEGORIES + FORMAT_SENSITIVITY_CATEGORY

# Relevance/irrelevance categories
RELEVANCE_CATEGORIES = {'irrelevance', 'live_irrelevance', 'live_relevance'}

# Test collection mapping for convenience
TEST_COLLECTION_MAPPING = {
    'all': ALL_CATEGORIES,
    'all_scoring': ALL_SCORING_CATEGORIES,
    'all_scoring_with_memory': ALL_SCORING_WITH_MEMORY_CATEGORIES,
    'single_turn': SINGLE_TURN_CATEGORY,
    'multi_turn': MULTI_TURN_CATEGORY,
    'agentic': AGENTIC_CATEGORY,
    'live': LIVE_CATEGORY,
    'non_live': NON_LIVE_CATEGORY,
    'web_search': WEB_SEARCH_CATEGORY,
    'memory': MEMORY_CATEGORY,
}


# ============================================================================
# Category Type Helpers
# ============================================================================


def get_category_type(test_category: str) -> str:
    """Map a test category to its evaluation type.

    Returns one of: 'single_turn', 'relevance', 'multi_turn', 'agentic', 'format_sensitivity'
    """
    if 'format_sensitivity' in test_category:
        return 'format_sensitivity'
    if 'relevance' in test_category or 'irrelevance' in test_category:
        return 'relevance'
    if 'multi_turn' in test_category:
        return 'multi_turn'
    if 'web_search' in test_category or 'memory' in test_category:
        return 'agentic'
    return 'single_turn'


def parse_categories(categories_str: str) -> list[str]:
    """Parse a comma-separated categories string, expanding collection names.

    Example:
        'simple_python,multi_turn' -> ['simple_python', 'multi_turn_base', ...]
    """
    result: set[str] = set()
    for cat in categories_str.split(','):
        cat = cat.strip()
        if not cat:
            continue
        if cat in TEST_COLLECTION_MAPPING:
            result.update(TEST_COLLECTION_MAPPING[cat])
        elif cat in ALL_CATEGORIES:
            result.add(cat)
        else:
            raise ValueError(f'Unknown BFCL category: {cat}')
    return sorted(result)


# ============================================================================
# Task Input/Output Models
# ============================================================================


class BFCLTaskInput(TaskRunInput):
    """Input for a single BFCL test case.

    Attributes:
        test_entry: The BFCL test entry dict (question, function, id, etc.)
        test_category: The BFCL test category (e.g. 'simple_python', 'multi_turn_base')
        ground_truth: Ground truth entry for evaluation
        model_name: Model name for BFCL handler (e.g. 'glm-5')
        is_fc_model: Whether the model supports function calling natively
        temperature: Sampling temperature
    """

    test_entry: dict[str, Any]
    test_category: str
    ground_truth: dict[str, Any] | None = None
    model_name: str = 'deepseek-v3.2'
    is_fc_model: bool = True
    temperature: float = 0.0


class BFCLTaskOutput(TaskRunOutput, WithScore):
    """Output for a single BFCL test case.

    Attributes:
        score: 1.0 if the test passed, 0.0 otherwise
        test_id: The BFCL test entry ID
        test_category: The BFCL test category
        model_responses: Raw model responses from inference
        eval_result: Evaluation result details
        token_usage: Token usage statistics
        latency: Inference latency in seconds
        error: Error message if inference/evaluation failed
    """

    test_id: str
    test_category: str
    model_responses: Any = None
    eval_result: dict[str, Any] | None = None
    token_usage: dict[str, Any] | None = None
    latency: float | None = None
    error: str | None = None
