import logging
import re

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    WithUserPrompt,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory


logger = logging.getLogger(__name__)


class SearchR1TaskInput(TaskRunInput, WithAgentRunner, WithUserPrompt):
    data_source: str = 'searchR1_nq'
    """Dataset/source identifier for routing reward function."""

    ground_truth: str | list[str] | dict
    """The expected result."""


class SearchR1TaskOutput(TaskRunOutput, WithScore):
    answer: str | None = None
    """The answer extracted from the agent's response."""
    score: float | None = None
    """Reward score; None means unsupported/unknown data source."""


@task_registry.register('search_r1_task')
async def run_search_r1_task(task_input: SearchR1TaskInput) -> SearchR1TaskOutput:
    """Search R1 task that requires the agent to find information using a search tool."""
    logger.debug(f'Running Search R1 task with prompt: {task_input.user_prompt}')

    # Run the agent with the user prompt directly
    agent_output = await task_input.run_agent(user_prompt=task_input.user_prompt)

    # Extract answer from the trajectory
    if not isinstance(agent_output, WithMainTrajectory):
        raise RuntimeError('Search R1 task requires agent output to be a subclass of WithMainTrajectory')

    last_assistant_content = next(
        (
            msg['content']
            for msg in reversed(agent_output.main_trajectory)
            if msg['role'] == 'assistant' and msg.get('content')
        ),
        None,
    )

    answer = None
    score = 0.0
    if last_assistant_content:
        extracted = extract_solution(last_assistant_content)
        if extracted is not None:
            answer = extracted.strip()
        score = compute_score(
            data_source=task_input.data_source,
            solution_str=last_assistant_content,
            ground_truth=task_input.ground_truth,
        )

    logger.info(
        f'Data source: {task_input.data_source}, Answer: {answer}, Ground truth: {task_input.ground_truth}, Score: {score}'
    )
    return SearchR1TaskOutput(answer=answer, score=score)


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in ['openai/gsm8k', 'gsm8k']:
        from verl.utils.reward_score import gsm8k

        return gsm8k.compute_score(solution_str, ground_truth)

    if data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval', 'HuggingFaceH4/MATH-500']:
        from verl.utils.reward_score import math_reward

        return math_reward.compute_score(solution_str, ground_truth)

    if data_source in [
        'searchR1_nq',
        'searchR1_triviaqa',
        'searchR1_popqa',
        'searchR1_hotpotqa',
        'searchR1_2wikimultihopqa',
        'searchR1_musique',
        'searchR1_bamboogle',
    ]:
        from verl.utils.reward_score import search_r1_like_qa_em

        return search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    logger.warning(f'unsupported data_source: {data_source}, score will be None')
    return None
