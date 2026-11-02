"""DeepEyes visual tool-calling task.

Implements the DeepEyes training task for visual reasoning with tool use.

Input format (task config):
{
    "system_prompt": "...",
    "user_prompt": [{"type": "image_url", ...}, {"type": "text", ...}] | "...",
    "images": [{"bytes": <bytes>, "path": "..."}],
    "ground_truth": "...",
    "question": "...",
    "reward_style": "model" | "rule",
    "max_iterations": 10,
}

The task:
1. Loads the image and stores it in tool_state so image_zoom_in_tool can access it
2. Runs the configured agent with the prompt
3. Computes the reward using LLM-as-a-judge (via the 'judge' OpenAI client)
"""

import base64
import io
import logging

from PIL import Image

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithMainTrajectory,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.projects.deepeyes.reward import compute_rule_score, compute_score
from nexus.extensions.projects.deepeyes.tools import DEEPEYES_TOOL_RECORDS_KEY


logger = logging.getLogger(__name__)


def _load_image(image_data: dict | Image.Image) -> Image.Image:
    """Load a PIL Image from the parquet image struct or a PIL Image directly.

    Accepts:
        PIL.Image.Image — returned as-is (converted to RGB)
        {'bytes': <bytes|str>, 'path': '...'}  — raw bytes or base64-encoded string from parquet/config
    """
    if isinstance(image_data, Image.Image):
        return image_data.convert('RGB')
    raw = image_data.get('bytes')
    if raw is not None:
        # Support both raw bytes and base64-encoded strings
        if isinstance(raw, str):
            raw = base64.b64decode(raw)
        return Image.open(io.BytesIO(raw)).convert('RGB')
    raise ValueError('image_data must be a PIL Image or a dict with a "bytes" key')


def _image_to_base64_url(image: Image.Image, fmt: str = 'JPEG') -> str:
    """Encode a PIL Image as a data-URL string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'


def _build_user_prompt_with_image(user_prompt: str | list[dict], image: Image.Image) -> str | list[dict]:
    """Inject the actual base64 image into the user prompt.

    The dataset user prompt may contain plain text with "<image>" markers or
    already-structured content parts. This function ensures the image is
    provided as an image_url content part for the main agent.
    """
    image_url_part = {'type': 'image_url', 'image_url': {'url': _image_to_base64_url(image)}}

    if isinstance(user_prompt, list):
        new_parts = []
        image_injected = False
        for part in user_prompt:
            if part.get('type') == 'image_url':
                new_parts.append(image_url_part)
                image_injected = True
            else:
                new_parts.append(part)
        if not image_injected:
            new_parts = [image_url_part, *new_parts]
        return new_parts

    text = str(user_prompt).replace('<image>', '').strip()
    return [
        image_url_part,
        {'type': 'text', 'text': text},
    ]


class DeepEyesTaskInput(TaskRunInput, WithAgentRunner):
    """DeepEyes task input.

    Fields map directly to the parquet data columns.

    system_prompt:        system prompt text for the agent
    user_prompt:          user prompt text or structured content parts
    images:               list of image structs [{'bytes': ..., 'path': ...}]
    ground_truth:         reference answer (from reward_model.ground_truth)
    question:             original question text (from extra_info.question)
    max_iterations:       max agent turns
    """

    system_prompt: str
    user_prompt: str | list[dict]
    images: list[dict | Image.Image]
    ground_truth: str
    question: str
    max_iterations: int = 10
    reward_style: str = 'model'  # 'model' for LLM-as-a-judge, 'rule' for boxed extraction


DeepEyesTaskInput.model_rebuild(_types_namespace={'Image': Image})


class DeepEyesTaskOutput(TaskRunOutput, WithScore):
    """DeepEyes task output."""

    response: str  # Last assistant message content
    correct: bool  # Whether acc_reward > 0.5


def _extract_last_response(trajectory: list[dict]) -> str:
    """Return the last assistant message content from a trajectory."""
    logger.debug(f'DeepEyes task: extracting last response from trajectory of {len(trajectory)} messages')
    for msg in reversed(trajectory):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [p.get('text', '') for p in content if p.get('type') == 'text']
                return ' '.join(texts)
    return ''


@task_registry.register('deepeyes')
async def deepeyes_task(task_input: DeepEyesTaskInput) -> DeepEyesTaskOutput:
    """DeepEyes visual reasoning task.

    Steps:
    1. Load the image from task_input.images[0]
    2. Inject the base64 image into the prompt messages
    3. Store the PIL Image in tool_state so image_zoom_in_tool can access it
    4. Run the configured agent
    5. Compute reward via LLM-as-a-judge using the 'judge' OpenAI client
    """
    if not task_input.images:
        raise ValueError('DeepEyes task requires at least one image')

    image = _load_image(task_input.images[0])
    user_prompt = _build_user_prompt_with_image(task_input.user_prompt, image)

    logger.debug(f'DeepEyes task: question="{task_input.question[:80]}", ground_truth="{task_input.ground_truth[:80]}"')

    # Store the PIL image in tool_state so the LOCAL image_zoom_in_tool can access it.
    # The tool reads context.tool_state['image'].
    ctx = require_context()
    ctx.tool_state['image'] = image
    ctx.tool_state[DEEPEYES_TOOL_RECORDS_KEY] = []

    # Run the agent through main_agent-compatible fields.
    agent_output = await task_input.run_agent(
        system_prompt=task_input.system_prompt,
        user_prompt=user_prompt,
        tools=['deepeyes/image_zoom_in_tool'],
        max_iterations=task_input.max_iterations,
    )

    # Extract response from trajectory
    trajectory = []
    if isinstance(agent_output, WithMainTrajectory):
        trajectory = agent_output.main_trajectory
    response = _extract_last_response(trajectory)
    tool_records = list(ctx.tool_state.get(DEEPEYES_TOOL_RECORDS_KEY, []))

    logger.debug(f'DeepEyes task: received response of {len(response)} chars')

    # Compute reward
    if task_input.reward_style == 'rule':
        result = compute_rule_score(
            solution_str=response,
            ground_truth=task_input.ground_truth,
            trajectory=trajectory,
            tool_records=tool_records,
        )
    else:
        # LLM-as-a-judge (falls back to 'main' if 'judge' client not configured)
        try:
            judge_client = ctx.get_openai_client('judge')
        except KeyError:
            logger.warning('DeepEyes task: no "judge" client configured, using "main" client for scoring')
            judge_client = ctx.openai_client

        result = await compute_score(
            solution_str=response,
            ground_truth=task_input.ground_truth,
            question=task_input.question,
            judge_client=judge_client,
            trajectory=trajectory,
            tool_records=tool_records,
        )

    logger.info(
        f'DeepEyes task: score={result.score:.3f}, correct={result.correct}, ground_truth="{task_input.ground_truth}"'
    )

    return DeepEyesTaskOutput(
        score=result.score,
        response=response,
        correct=result.correct,
    )
