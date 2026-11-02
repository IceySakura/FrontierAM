"""Reward-system evaluator for stem_mas."""

import logging
import os

from typing import Any

from nexus.extensions.agents.base.evaluation import Evaluator
from nexus.extensions.tasks.stem_mas.evaluator.common import strip_think_blocks
from nexus.utils.reward_system import RewardSystemClient


logger = logging.getLogger(__name__)


def _reward_system_token_configured(user_token: str | None) -> bool:
    return bool(user_token or os.getenv('NEXUS_REWARD_SYSTEM_TOKEN'))


class RewardSystemEvaluator(Evaluator):
    """Evaluator that delegates scoring to the reward system API.

    Parameters are split into two groups:
    - **Client-level** (shared across all samples): ``user_token``, ``exp_id``, ``ci_test``
    - **Per-sample** (vary per evaluation): ``router_id``, ``extra``, ``public_test``

    Per-sample fields are passed via the constructor so that each task invocation
    can create an evaluator with the correct per-sample context.
    """

    last_judge_result: dict[str, Any] | None = None

    def __init__(
        self,
        user_token: str | None = None,
        exp_id: str = '',
        router_id: str = '',
        extra: dict | None = None,
        public_test: dict | None = None,
        ci_test: bool = False,
    ):
        logger.debug(
            '[RewardSystemEvaluator:init] router_id=%s, exp_id=%s, ci_test=%s, token_configured=%s, '
            'extra_count=%d, public_test_count=%d',
            router_id,
            exp_id or '<empty>',
            ci_test,
            _reward_system_token_configured(user_token),
            len(extra or {}),
            len(public_test or {}),
        )
        self.client = RewardSystemClient(user_token=user_token, exp_id=exp_id)
        self.router_id = router_id
        self.extra = extra or {}
        self.public_test = public_test or {}
        self.ci_test = ci_test

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        clean_prediction = strip_think_blocks(prediction)
        prompt = question or ''
        messages = [
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': clean_prediction},
        ]

        try:
            score = await self.client.get_score(
                messages=messages,
                ref_answer=reference,
                router_id=self.router_id,
                extra_param=self.extra,
                unittests=self.public_test,
                ci_test=self.ci_test,
            )
        except Exception as e:
            logger.exception(
                '[RewardSystemEvaluator:evaluate] failed for router_id=%s with error=%s',
                self.router_id,
                e,
            )
            raise

        self.last_judge_result = {
            'score': score,
            'router_id': self.router_id,
            'ci_test': self.ci_test,
            'extra_keys': sorted(self.extra.keys()),
            'public_test_keys': sorted(self.public_test.keys()),
        }

        logger.info('[RewardSystem] router_id=%s, score=%.4f', self.router_id, score)
        return score

    def get_last_judge_result(self) -> dict | None:
        return self.last_judge_result
