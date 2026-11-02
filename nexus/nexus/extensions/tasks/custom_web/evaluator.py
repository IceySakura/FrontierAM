"""Evaluation module for custom web tasks.

This module integrates with the verifier module to evaluate task completion.
It supports multiple verification types:
- state: Verify application state changes
- retrieval: Verify retrieved information (with LLM semantic comparison)
- navigation: Verify page navigation paths
- action: Verify executed action sequences

The evaluation follows the groundtruth format from delivery.jsonl:
{
    "groundtruth": [
        {"type": "state", "expected_states": {...}, "options": {"mode": "include"}},
        {"type": "retrieval", "expected_content": "...", "options": {"similarity_threshold": 0.8}}
    ]
}

LLM 调用使用 ctx.get_openai_client('judge') 方式，与 nexus 框架保持一致。
"""

import logging

from typing import Any

from nexus.extensions.tasks.custom_web.verifier import (
    CompositeVerifier,
    VerificationResult,
)


logger = logging.getLogger(__name__)


async def evaluate_with_verifier(
    groundtruth: list[dict[str, Any]],
    agent_answer: str | None,
    executed_actions: list[str],
    executed_pages: list[str],
    current_state: dict[str, Any],
    question: str = '',
) -> VerificationResult:
    """Evaluate task completion using CompositeVerifier.

    Args:
        groundtruth: List of groundtruth items from task data
        agent_answer: Agent's final answer (for retrieval verification)
        executed_actions: List of executed action names
        executed_pages: List of visited page paths
        current_state: Current application state
        question: Task question/description for LLM context

    Returns:
        VerificationResult with score and details
    """
    if not groundtruth:
        logger.warning('No groundtruth provided, returning success')
        return VerificationResult.passed('No groundtruth to verify')

    # Create composite verifier (LLM client is obtained from context when needed)
    verifier = CompositeVerifier()

    # Build response based on groundtruth types
    response_list = _build_response_list(
        groundtruth=groundtruth,
        agent_answer=agent_answer,
        executed_actions=executed_actions,
        executed_pages=executed_pages,
        current_state=current_state,
        question=question,
    )

    # Run verification
    return await verifier.verify(
        groundtruth={'groundtruth': groundtruth},
        response={'response': response_list},
    )


def _build_response_list(
    groundtruth: list[dict[str, Any]],
    agent_answer: str | None,
    executed_actions: list[str],
    executed_pages: list[str],
    current_state: dict[str, Any],
    question: str = '',
) -> list[dict[str, Any]]:
    """Build response list matching groundtruth structure."""
    response_list = []

    for gt_item in groundtruth:
        gt_type = gt_item.get('type', '')

        if gt_type == 'state':
            response_list.append({
                'type': 'state',
                'executed_states': current_state,
                'question': question,
            })

        elif gt_type == 'retrieval':
            response_list.append({
                'type': 'retrieval',
                'executed_content': agent_answer or '',
                'question': question,
            })

        elif gt_type == 'navigation':
            response_list.append({
                'type': 'navigation',
                'executed_pages': executed_pages,
            })

        elif gt_type == 'action':
            response_list.append({
                'type': 'action',
                'executed_actions': executed_actions,
            })

        else:
            logger.warning(f'Unknown groundtruth type: {gt_type}')

    return response_list


# ============================================================================
# Convenience Functions for Direct Verification
# ============================================================================


async def verify_state(
    expected_states: dict[str, Any],
    executed_states: dict[str, Any],
    mode: str = 'include',
) -> VerificationResult:
    """Verify state changes."""
    from nexus.extensions.tasks.custom_web.verifier import StateVerifier

    verifier = StateVerifier()
    return await verifier.verify(
        groundtruth={'expected_states': expected_states},
        response={'executed_states': executed_states},
        options={'mode': mode},
    )


async def verify_retrieval(
    expected_content: str,
    executed_content: str,
    similarity_threshold: float = 0.8,
) -> VerificationResult:
    """Verify retrieved content using LLM semantic comparison."""
    from nexus.extensions.tasks.custom_web.verifier import RetrievalVerifier

    verifier = RetrievalVerifier()
    return await verifier.verify(
        groundtruth={'expected_content': expected_content},
        response={'executed_content': executed_content},
        options={'similarity_threshold': similarity_threshold},
    )


async def verify_navigation(
    expected_pages: list[str],
    executed_pages: list[str],
    mode: str = 'final',
) -> VerificationResult:
    """Verify page navigation."""
    from nexus.extensions.tasks.custom_web.verifier import NavigationVerifier

    verifier = NavigationVerifier()
    return await verifier.verify(
        groundtruth={'expected_pages': expected_pages},
        response={'executed_pages': executed_pages},
        options={'mode': mode},
    )


async def verify_actions(
    expected_actions: list[str],
    executed_actions: list[str],
    mode: str = 'contains',
) -> VerificationResult:
    """Verify executed actions."""
    from nexus.extensions.tasks.custom_web.verifier import ActionVerifier

    verifier = ActionVerifier()
    return await verifier.verify(
        groundtruth={'expected_actions': expected_actions},
        response={'executed_actions': executed_actions},
        options={'mode': mode},
    )


# ============================================================================
# Batch Evaluation
# ============================================================================


async def evaluate_batch(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a batch of task results."""
    total = len(results)
    successful = sum(1 for r in results if r.get('is_success', False))
    total_score = sum(r.get('score', 0.0) for r in results)

    type_stats: dict[str, dict[str, int]] = {}
    for r in results:
        verification = r.get('verification_results', {})
        details = verification.get('details', {})
        results_list = details.get('results', [])

        for item in results_list:
            item_type = item.get('type', 'unknown')
            if item_type not in type_stats:
                type_stats[item_type] = {'total': 0, 'passed': 0}

            type_stats[item_type]['total'] += 1
            if item.get('result', {}).get('success', False):
                type_stats[item_type]['passed'] += 1

    return {
        'total_tasks': total,
        'successful_tasks': successful,
        'success_rate': successful / total if total > 0 else 0.0,
        'average_score': total_score / total if total > 0 else 0.0,
        'verification_type_stats': type_stats,
    }
