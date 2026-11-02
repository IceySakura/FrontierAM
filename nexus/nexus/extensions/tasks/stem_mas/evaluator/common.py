"""Shared evaluator helpers for stem_mas."""

import logging
import re


logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


def strip_think_blocks(prediction: str) -> str:
    return _THINK_RE.sub('', prediction).strip()


def log_judge_input(prefix: str, question: str | None, prediction: str, reference: str, prompt: str) -> None:
    logger.debug('=' * 80)
    logger.debug(f'[{prefix}] Input:')
    logger.debug(f'  Question: {question[:200]}...' if question and len(question) > 200 else f'  Question: {question}')
    logger.debug(f'  Prediction: {prediction[:200]}...' if len(prediction) > 200 else f'  Prediction: {prediction}')
    logger.debug(f'  Reference: {reference[:200]}...' if len(reference) > 200 else f'  Reference: {reference}')
    logger.debug(f'[{prefix}] Full Prompt:\n{prompt}')
