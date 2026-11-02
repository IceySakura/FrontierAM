"""BFCL evaluation wrappers for different test category types.

Thin wrappers around BFCL's existing checkers (ast_checker, multi_turn_checker,
agentic_checker) that provide a unified interface returning (valid, detail) tuples.
"""

from __future__ import annotations

import logging

from typing import Any

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
from bfcl_eval.eval_checker.ast_eval.ast_checker import (
    JAVA_TYPE_CONVERSION,
    MODEL_CONFIG_MAPPING,
    ast_checker,
)


# ---------------------------------------------------------------------------
# Fix BFCL bug: type conversion mappings and converter functions use capitalized
# type names ('String') but test data uses lowercase ('string'). This causes
# KeyError / ValueError for all simple_java and simple_javascript entries.
# Patch both Java and JS converters to accept lowercase 'string'.
# ---------------------------------------------------------------------------

# 1. Add lowercase 'string' to type conversion dicts
if 'string' not in JAVA_TYPE_CONVERSION:
    JAVA_TYPE_CONVERSION['string'] = str

from bfcl_eval.eval_checker.ast_eval.type_convertor.js_type_converter import JS_TYPE_CONVERSION  # noqa: E402


if 'string' not in JS_TYPE_CONVERSION:
    JS_TYPE_CONVERSION['string'] = str

# 2. Patch java_type_converter to normalize 'string' → 'String'
import bfcl_eval.eval_checker.ast_eval.type_convertor.java_type_converter as _java_mod  # noqa: E402


_original_java_type_converter = _java_mod.java_type_converter


def _patched_java_type_converter(value, expected_type, nested_type=None):
    if expected_type == 'string':
        expected_type = 'String'
    return _original_java_type_converter(value, expected_type, nested_type)


_java_mod.java_type_converter = _patched_java_type_converter  # ty: ignore[invalid-assignment]

import bfcl_eval.eval_checker.ast_eval.ast_checker as _ast_mod  # noqa: E402


_ast_mod.java_type_converter = _patched_java_type_converter  # ty: ignore[invalid-assignment]

# 3. Patch js_type_converter to normalize 'string' → 'String'
import bfcl_eval.eval_checker.ast_eval.type_convertor.js_type_converter as _js_mod  # noqa: E402


_original_js_type_converter = _js_mod.js_type_converter


def _patched_js_type_converter(value, expected_type, nested_type=None):
    if expected_type == 'string':
        expected_type = 'String'
    return _original_js_type_converter(value, expected_type, nested_type)


_js_mod.js_type_converter = _patched_js_type_converter  # ty: ignore[invalid-assignment]
_ast_mod.js_type_converter = _patched_js_type_converter  # ty: ignore[invalid-assignment]

# 4. Patch WebSearchAPI to use internal search forwarder instead of SerpAPI
from nexus.extensions.tasks.bfclv4.web_search_patch import apply_web_search_patch  # noqa: E402


apply_web_search_patch()

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (  # noqa: E402
    multi_turn_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: E402
    is_empty_execute_response,
)
from bfcl_eval.utils import is_empty_output, is_function_calling_format_output  # noqa: E402

from nexus.extensions.tasks.bfclv4.handler import NexusOpenAIHandler  # noqa: E402


logger = logging.getLogger(__name__)


def _ensure_model_in_config_mapping(model_name: str, is_fc_model: bool) -> str:
    """Ensure model_name exists in MODEL_CONFIG_MAPPING for ast_checker.

    BFCL's ast_checker uses MODEL_CONFIG_MAPPING to look up `underscore_to_dot`
    in convert_func_name(). This mapping is hardcoded with known model names.
    When testing internal/custom models via Nexus, the model name won't exist.

    Fix: dynamically inject a ModelConfig entry for the model if missing.
    The key field is `underscore_to_dot`:
      - True for FC models (OpenAI API replaces '.' with '_' in function names)
      - False for prompting models

    This is BFCL-local — only affects the eval checker lookup, not framework code.
    """
    # convert_func_name replaces '_' with '/' before lookup
    escaped = model_name.replace('_', '/')
    if escaped in MODEL_CONFIG_MAPPING:
        return model_name

    # Also try the original
    if model_name in MODEL_CONFIG_MAPPING:
        return model_name

    # Inject a synthetic entry — only underscore_to_dot matters for eval
    from bfcl_eval.constants.model_config import ModelConfig

    synthetic = ModelConfig(
        model_name=model_name,
        display_name=f'{model_name} (Nexus)',
        url='',
        org='Nexus',
        license='',
        model_handler=None,  # ty: ignore[invalid-argument-type]
        input_price=None,
        output_price=None,
        is_fc_model=is_fc_model,
        underscore_to_dot=is_fc_model,  # FC models need dot-to-underscore conversion
    )
    MODEL_CONFIG_MAPPING[escaped] = synthetic
    logger.info(
        f'Injected MODEL_CONFIG_MAPPING[{escaped!r}]: is_fc_model={is_fc_model}, underscore_to_dot={is_fc_model}'
    )
    return model_name


_eval_schema_cache: dict[str, dict] = {}


def _load_eval_function_schema(test_id: str, test_category: str) -> list[dict]:
    """Load function schemas with original (non-hint) types for evaluation.

    The official BFCL evaluation loads schemas with include_language_specific_hint=False
    so the AST checker sees native Java/JS types (boolean, integer, etc.) and applies
    the correct type converters. During inference, schemas use hint=True (all string).
    """
    if test_category not in _eval_schema_cache:
        from bfcl_eval.utils import load_dataset_entry

        entries = load_dataset_entry(test_category, include_prereq=False, include_language_specific_hint=False)
        _eval_schema_cache[test_category] = {e['id']: e['function'] for e in entries}
    return _eval_schema_cache[test_category].get(test_id, [])


def _get_language(test_category: str) -> Language:
    """Get the language enum for a test category."""
    if 'java' in test_category and 'javascript' not in test_category:
        return Language.JAVA
    if 'javascript' in test_category:
        return Language.JAVASCRIPT
    return Language.PYTHON


def evaluate_single_turn(
    handler: NexusOpenAIHandler,
    model_responses: Any,
    ground_truth: dict[str, Any],
    test_entry: dict[str, Any],
    test_category: str,
    model_name: str,
    is_fc_model: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate a single-turn (non-relevance) test case using AST checker.

    Args:
        handler: The BFCL handler for decoding responses
        model_responses: Raw model responses from inference
        ground_truth: Ground truth entry with 'ground_truth' key
        test_entry: The test entry dict
        test_category: BFCL test category
        model_name: Model name for checker

    Returns:
        (valid, detail) tuple
    """
    test_id = test_entry['id']
    possible_answer = ground_truth['ground_truth']
    prompt_function = test_entry['function']

    # For Java/JS categories, the AST checker needs the original function schemas
    # (with native types like "boolean", "integer") rather than the inference schemas
    # (where all types are converted to "string" with language-specific hints).
    # This mirrors the official BFCL behavior: inference uses hint=True, evaluation uses hint=False.
    language = _get_language(test_category)
    if language in (Language.JAVA, Language.JAVASCRIPT):
        prompt_function = _load_eval_function_schema(test_id, test_category)

    # Decode model response
    try:
        decoded = handler.decode_ast(model_responses, ReturnFormat.PYTHON, False)
    except Exception as e:
        return False, {
            'id': test_id,
            'error': f'Failed to decode AST: {e}',
            'error_type': 'ast_decoder:decoder_failed',
            'model_result_raw': model_responses,
        }

    # Check format validity
    if not is_function_calling_format_output(decoded):
        return False, {
            'id': test_id,
            'error': 'Output not in function calling format',
            'error_type': 'ast_decoder:decoder_wrong_output_format',
            'model_result_raw': str(model_responses),
            'model_result_decoded': str(decoded),
        }

    # Run AST checker
    _ensure_model_in_config_mapping(model_name, is_fc_model)
    checker_result = ast_checker(
        prompt_function,
        decoded,
        possible_answer,
        language,
        test_category,
        model_name,
    )

    if not checker_result['valid']:
        return False, {
            'id': test_id,
            'error': checker_result.get('error', []),
            'error_type': checker_result.get('error_type', 'ast_checker_failed'),
            'model_result_decoded': decoded,
            'possible_answer': possible_answer,
        }

    return True, {'id': test_id}


def evaluate_relevance(
    handler: NexusOpenAIHandler,
    model_responses: Any,
    test_entry: dict[str, Any],
    test_category: str,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate a relevance/irrelevance test case.

    For irrelevance: model should NOT produce valid function calls.
    For relevance: model SHOULD produce valid function calls.

    Args:
        handler: The BFCL handler for decoding responses
        model_responses: Raw model responses from inference
        test_entry: The test entry dict
        test_category: BFCL test category

    Returns:
        (valid, detail) tuple
    """
    test_id = test_entry['id']
    contain_func_call = False
    decoded_result = None

    try:
        decoded_result = handler.decode_ast(model_responses, language=ReturnFormat.PYTHON, has_tool_call_tag=False)
        contain_func_call = True
        if is_empty_output(decoded_result):
            contain_func_call = False
    except Exception:
        contain_func_call = False

    # irrelevance: success means no function call
    # relevance: success means function call present
    success = not contain_func_call if 'irrelevance' in test_category else contain_func_call

    if not success:
        detail: dict[str, Any] = {
            'id': test_id,
            'model_result': model_responses,
            'decoded_result': decoded_result,
        }
        if 'irrelevance' in test_category:
            detail['error'] = 'Valid syntax decoded when it should not have been'
            detail['error_type'] = 'irrelevance_error:decoder_success'
        else:
            detail['error'] = 'Failed to decode valid function call when expected'
            detail['error_type'] = 'relevance_error:decoder_failed'
        return False, detail

    return True, {'id': test_id}


def evaluate_multi_turn(
    handler: NexusOpenAIHandler,
    model_responses: list[list],
    ground_truth: dict[str, Any],
    test_entry: dict[str, Any],
    test_category: str,
    model_name: str,
    is_fc_model: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate a multi-turn test case.

    Args:
        handler: The BFCL handler for decoding responses
        model_responses: Multi-turn model responses (list of turns, each a list of steps)
        ground_truth: Ground truth entry with 'ground_truth' key (list of turns)
        test_entry: The test entry dict
        test_category: BFCL test category
        model_name: Model name for checker

    Returns:
        (valid, detail) tuple
    """
    test_id = test_entry['id']
    ground_truth_list = ground_truth['ground_truth']

    # Check turn count matches
    if not isinstance(model_responses, list) or len(model_responses) != len(ground_truth_list):
        return False, {
            'id': test_id,
            'error': f'Turn count mismatch: model={len(model_responses) if isinstance(model_responses, list) else "N/A"}, '
            f'expected={len(ground_truth_list)}',
            'error_type': 'multi_turn:force_terminated',
        }

    # Decode model results per turn per step
    multi_turn_decoded: list[list[list[str]]] = []
    for single_turn_results in model_responses:
        single_turn_decoded = []
        for step_result in single_turn_results:
            try:
                decoded: list[str] = handler.decode_execute(step_result, has_tool_call_tag=False)
                if is_empty_execute_response(decoded):
                    continue
                single_turn_decoded.append(decoded)
            except Exception:
                continue
        multi_turn_decoded.append(single_turn_decoded)

    # NOTE: The official BFCL evaluation does NOT call multi_turn_irrelevance_checker().
    # multi_turn_checker() skips turns with empty ground truth (the turns where the model
    # "should" avoid calling functions), so irrelevance is effectively not enforced.
    # We match official behavior here to ensure score alignment.

    # Run multi-turn checker
    _ensure_model_in_config_mapping(model_name, is_fc_model)
    checker_result = multi_turn_checker(
        multi_turn_decoded,
        ground_truth_list,
        test_entry,
        test_category,
        model_name,
    )

    if not checker_result['valid']:
        return False, {
            'id': test_id,
            'error': checker_result.get('error_message', 'Multi-turn check failed'),
            'error_type': checker_result.get('error_type', 'multi_turn_checker_failed'),
            'details': checker_result.get('details', {}),
        }

    return True, {'id': test_id}


def evaluate_agentic(
    handler: NexusOpenAIHandler,
    model_responses: list[list],
    ground_truth: dict[str, Any],
    test_entry: dict[str, Any],
    test_category: str,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate an agentic (web_search, memory) test case.

    Agentic tests check if the final non-function-call message contains
    the expected answer.

    Args:
        handler: The BFCL handler for decoding responses
        model_responses: Agentic model responses (list of one turn with multiple steps)
        ground_truth: Ground truth entry with 'ground_truth' key (list of possible answers)
        test_entry: The test entry dict
        test_category: BFCL test category

    Returns:
        (valid, detail) tuple
    """
    test_id = test_entry['id']
    possible_answers = ground_truth['ground_truth']

    # Agentic is single-turn multi-step, expect list of one element
    if not isinstance(model_responses, list) or len(model_responses) != 1:
        return False, {
            'id': test_id,
            'error': 'Model did not output expected agentic response format (list of one turn)',
            'error_type': 'agentic:inference_error',
        }

    # Find the last non-function-call message
    last_non_fc_message = None
    for step_result in model_responses[0]:
        try:
            decoded = handler.decode_execute(step_result, has_tool_call_tag=False)
            if is_empty_execute_response(decoded):
                last_non_fc_message = step_result
                continue
        except Exception:
            last_non_fc_message = step_result
            continue

    if last_non_fc_message is None:
        return False, {
            'id': test_id,
            'error': 'Cannot find last non-function-call message',
            'error_type': 'agentic:no_last_message',
        }

    # Check if the expected answer is in the last message
    checker_result = agentic_checker(last_non_fc_message, possible_answers)

    if not checker_result['valid']:
        return False, {
            'id': test_id,
            'error': checker_result.get('error_message', 'Expected answer not found'),
            'error_type': checker_result.get('error_type', 'agentic:answer_not_found'),
            'last_non_fc_message': last_non_fc_message,
            'possible_answers': possible_answers,
        }

    return True, {'id': test_id}
