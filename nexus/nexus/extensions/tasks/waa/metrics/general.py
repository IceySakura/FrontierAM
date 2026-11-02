"""General metrics for WAA evaluation.

All metrics are pure synchronous functions with signature:
    def metric_name(result: Any, rules: Any) -> float
"""

from __future__ import annotations

import json
import logging

from typing import Any

from nexus.extensions.tasks.waa.registry import metric_registry


logger = logging.getLogger('nexus.waa.metrics.general')


@metric_registry.register()
def exact_match(result: Any, rules: dict) -> float:
    """Exact match between result and expected value."""
    expect = rules['expected']
    return 1.0 if result == expect else 0.0


@metric_registry.register()
def check_direct_json_object(result: Any, rules: dict) -> float:
    """Compare two JSON objects directly.

    If expect_in_result is False (default): check that all expected keys match result.
    If expect_in_result is True: check that expected values are contained in result.
    """
    if isinstance(result, str):
        result = result.strip().replace("'", '"')
        result = json.loads(result)
    if result is None:
        return 0.0
    try:
        expect_in_result = rules.get('expect_in_result', False)
        if not expect_in_result:
            expected_json = rules['expected']
            for key in expected_json:
                expected_value = expected_json.get(key)
                if expected_value != result.get(key):
                    return 0.0
            return 1.0
        expected_json = rules['expected']
        for key in expected_json:
            if isinstance(expected_json.get(key), list):
                flag = 0
                expected_value_list = expected_json.get(key)
                for each_expected_value in expected_value_list:
                    if isinstance(result.get(key), list) and each_expected_value in result.get(key):
                        flag = 1
                        break
                if flag == 0:
                    return 0.0
            elif isinstance(expected_json.get(key), str):
                if expected_json.get(key) not in result.get(key):
                    return 0.0
            else:
                logger.debug('check_direct_json_object: expected value type not supported')
                return 0.0
        return 1.0
    except Exception:
        logger.debug('check_direct_json_object: result is not a valid json object')
        return 0.0
