"""MNIST scrambled-digit multiplication task.

Input format (``nexus.task_config`` in the parquet row)::

    {
        "user_prompt": "<image>...<image> <question>",
        "images": [<base64 png>, ...],
        "ground_truth": <int|str>,
        "max_iterations": 10,
    }

The task loads the images once and hands the *same* PIL objects to two places:

* the user prompt, as ``image_url`` content parts, so the model can see them;
* ``ctx.tool_state['image_list']``, so ``mnist/decode`` can address one by index.

Both are references to one list, not two copies — a tool parameter can only carry
an index (model output is JSON, and the pydantic-to-OpenAI schema admits only
JSON primitives), so the entities have to live somewhere the tool can reach.

Scoring mirrors ``calculator_task``: the last ``\\boxed{...}`` in the final
assistant message, compared to ``ground_truth`` as a string.
"""

from __future__ import annotations

import base64
import io
import logging

from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image
from pydantic import ConfigDict

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    WithUserPrompt,
    require_context,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory
from nexus.extensions.projects.mnist.tools import IMAGE_LIST_STATE_KEY, image_to_base64_url


logger = logging.getLogger(__name__)


class MnistTaskInput(TaskRunInput, WithAgentRunner, WithUserPrompt):
    # NOTE(arlenxu): the frozen parquet stores reward_model.ground_truth as a Python
    # int, and pydantic v2 rejects int for a `str` field in strict-by-default mode.
    # Without this, every row fails validation at task start.
    model_config = ConfigDict(coerce_numbers_to_str=True)

    images: list[str | dict | Image.Image]
    """Task images, in the order the prompt shows them: base64 str, parquet
    struct ``{'bytes': ...}``, or an already-decoded PIL image."""

    ground_truth: str
    """Expected product. Declared ``str`` (with the coercion above) so a parquet
    ``int`` becomes ``'42'`` rather than failing an ``int == str`` comparison at
    scoring time."""

    max_iterations: int = 10


MnistTaskInput.model_rebuild(_types_namespace={'Image': Image})


class MnistTaskOutput(TaskRunOutput, WithScore):
    answer: str | None = None
    """The answer extracted from the agent's final message."""


def _load_image(image_data: str | dict | Image.Image) -> Image.Image:
    """Load one PIL image from a base64 string, a parquet struct, or an image."""
    if isinstance(image_data, Image.Image):
        return image_data.convert('RGB')
    if isinstance(image_data, str):
        payload = image_data.split(',', 1)[1] if image_data.startswith('data:') else image_data
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert('RGB')
    raw = image_data.get('bytes')
    if raw is None:
        raise ValueError('image entry must be a base64 str, a PIL Image, or a dict with a "bytes" key')
    if isinstance(raw, str):
        raw = base64.b64decode(raw)
    return Image.open(io.BytesIO(raw)).convert('RGB')


def _build_user_prompt_with_images(
    user_prompt: str | Sequence[Mapping[str, Any]], images: list[Image.Image]
) -> list[dict]:
    """Turn the prompt into content parts with one image part per image.

    A ``<image>`` marker in the text is replaced positionally by its image, so
    the parts come out in the same order as ``images`` and therefore in the same
    order the tool's ``image_idx`` addresses. A prompt with no markers gets every
    image up front.

    Typed as a Sequence of Mappings rather than list[dict]: the caller hands over
    OpenAI's TypedDict content parts, and list is invariant, so list[dict] rejects them.
    """
    image_parts = [{'type': 'image_url', 'image_url': {'url': image_to_base64_url(image)}} for image in images]

    if not isinstance(user_prompt, str):
        parts: list[dict] = []
        cursor = 0
        for part in user_prompt:
            if part.get('type') == 'image_url' and cursor < len(image_parts):
                parts.append(image_parts[cursor])
                cursor += 1
            else:
                parts.append(dict(part))
        parts.extend(image_parts[cursor:])
        return parts

    text = str(user_prompt)
    if '<image>' not in text:
        return [*image_parts, {'type': 'text', 'text': text.strip()}]

    parts = []
    segments = text.split('<image>')
    for index, segment in enumerate(segments):
        if segment.strip():
            parts.append({'type': 'text', 'text': segment.strip()})
        if index < len(segments) - 1 and index < len(image_parts):
            parts.append(image_parts[index])
    parts.extend(image_parts[len(segments) - 1 :])
    return parts


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last ``\\boxed{...}`` span, tracking nested braces.

    Adapted from https://github.com/EleutherAI/lm-evaluation-harness

    Copied from ``calculator/task.py`` rather than imported: a project owns its own
    scoring, and reaching across into a sibling project would couple this task's
    reward to one nobody expects to change together with it.
    """
    idx = string.rfind('\\boxed')
    if '\\boxed ' in string:
        return '\\boxed ' + string.split('\\boxed ')[-1].split('$')[0]
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    right_brace_idx = None
    num_left_braces_open = 0
    for i in range(idx, len(string)):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Strip the ``\\boxed{}`` wrapper and return the inner content."""
    if '\\boxed ' in s:
        left = '\\boxed '
        assert s[: len(left)] == left, f'expected {left!r} prefix, got {s[:20]!r}'
        return s[len(left) :]

    left = '\\boxed{'
    assert s[: len(left)] == left, f'expected {left!r} prefix, got {s[:20]!r}'
    assert s[-1] == '}', f'expected closing brace, got {s[-1]!r}'
    return s[len(left) : -1]


def extract_boxed_answer(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}``, or None if absent/malformed."""
    boxed_str = last_boxed_only_string(text)
    if boxed_str is None:
        return None
    try:
        return remove_boxed(boxed_str)
    except AssertionError:
        return None


@task_registry.register('mnist_task')
async def run_mnist_task(task_input: MnistTaskInput) -> MnistTaskOutput:
    """Show the images, let the agent decode and multiply, score the boxed answer."""
    if not task_input.images:
        raise ValueError('mnist_task requires at least one image')

    images = [_load_image(entry) for entry in task_input.images]

    # The prompt parts and tool_state hold the SAME PIL objects: the tool decodes
    # by index into exactly what the model was shown.
    ctx = require_context()
    ctx.tool_state[IMAGE_LIST_STATE_KEY] = images

    user_prompt = _build_user_prompt_with_images(task_input.user_prompt, images)
    logger.debug(f'mnist_task: {len(images)} images, ground_truth={task_input.ground_truth}')

    agent_output = await task_input.run_agent(
        user_prompt=user_prompt,
        max_iterations=task_input.max_iterations,
    )

    if not isinstance(agent_output, WithMainTrajectory):
        raise RuntimeError('mnist_task requires agent output to be a subclass of WithMainTrajectory')

    last_assistant_content = next(
        (
            msg['content']
            for msg in reversed(agent_output.main_trajectory)
            if msg['role'] == 'assistant' and msg.get('content')
        ),
        None,
    )

    answer = None
    if last_assistant_content:
        extracted = extract_boxed_answer(str(last_assistant_content))
        if extracted is not None:
            answer = extracted.strip()

    score = 1.0 if answer is not None and answer == task_input.ground_truth else 0.0
    logger.info(f'mnist_task: answer={answer}, ground_truth={task_input.ground_truth}, score={score}')

    return MnistTaskOutput(score=score, answer=answer)
