"""Audio Re-Listen task — DeepEyes-style multi-turn audio reasoning.

The task ingests an audio clip + a question (multi-choice or open-ended),
embeds the audio in the user prompt as an OpenAI ``input_audio`` content part,
and lets the agent optionally invoke ``audio_relisten_tool`` to "re-listen"
to specific time ranges. Reward shape mirrors DeepEyes:

    final_score = 0.5 * acc_reward + 0.5 * tool_reward

Key difference vs. Echo's reference implementation: the re-listen behaviour
goes through a *real* OpenAI tool-call protocol (``tool_calls`` →
``role: tool`` message containing an ``input_audio`` part). No stop-token
hacks, no inference-engine-level audio splicing — just standard function
calling, exactly like DeepEyes' ``image_zoom_in_tool``.
"""

import base64
import logging
import os

from typing import Any

from openai.types.chat import ChatCompletionContentPartParam

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithMainTrajectory,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.projects.audio_relisten.audio_io import encode_wav_bytes, load_wav
from nexus.extensions.projects.audio_relisten.reward import compute_rule_score, compute_score

# Side-effect import: registers audio_relisten/audio_relisten_tool with the tool registry.
from nexus.extensions.projects.audio_relisten.tools import (  # noqa: F401
    AUDIO_FORMAT_STATE_KEY,
    AUDIO_RELISTEN_RECORDS_KEY,
    AUDIO_STATE_KEY,
    SAMPLE_RATE_STATE_KEY,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------


DEFAULT_SYSTEM_PROMPT = (
    'You are an expert audio-comprehension assistant. The user has shared a single audio clip '
    'and a question about it.\n\n'
    'You have access to one tool:\n'
    '  - audio_relisten_tool(time_range=[start_seconds, end_seconds], label?): re-listens to a '
    'specific portion of the *original* clip and returns that audio segment to you so you can '
    'examine it more closely.\n\n'
    'Use the tool whenever the first listen is not enough — for example to focus on a brief '
    'sound event, to verify when something happens, or to compare two moments. Call the tool at '
    'most a few times; do not loop indefinitely.\n\n'
    'When you are ready to answer, respond with the final answer wrapped in <answer>...</answer> '
    'tags, and nothing else after it. For multiple-choice questions, place the option text '
    '(or its letter) inside the tags.'
)


# ---------------------------------------------------------------------------
# Task I/O
# ---------------------------------------------------------------------------


class AudioRelistenTaskInput(TaskRunInput, WithAgentRunner):
    """Input for the audio_relisten task.

    audio_path:       absolute path to a WAV file on the local filesystem
    question:         the natural-language question (typically a stem + options)
    ground_truth:     reference answer text (verbatim option text or open-ended ground truth)
    options:          for multi-choice — the list of options (used by rule scoring)
    system_prompt:    overrides the default system prompt
    user_prompt_text: extra text accompanying the audio (default: just ``question``)
    reward_style:     'rule' (exact / option match) or 'model' (LLM-as-a-judge)
    audio_format:     'wav' or 'mp3' — passed through as the input_audio.format field
    max_iterations:   max agent turns
    """

    audio_path: str
    question: str
    ground_truth: str
    options: list[str] | None = None
    system_prompt: str | list[ChatCompletionContentPartParam] = DEFAULT_SYSTEM_PROMPT
    user_prompt_text: str | None = None
    reward_style: str = 'rule'
    audio_format: str = 'wav'
    max_iterations: int = 8


class AudioRelistenTaskOutput(TaskRunOutput, WithScore):
    response: str
    correct: bool
    n_tool_calls: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_last_response(trajectory: list[dict]) -> str:
    """Return the last assistant message content from a trajectory."""
    for msg in reversed(trajectory):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [p.get('text', '') for p in content if p.get('type') == 'text']
                return ' '.join(texts)
    return ''


def _format_question(question: str, options: list[str] | None) -> str:
    """Append A/B/C/D options to the question stem (if multi-choice)."""
    if not options:
        return question
    parts = [question.rstrip(), '']
    for i, opt in enumerate(options):
        letter = chr(ord('A') + i)
        parts.append(f'{letter}. {opt}')
    return '\n'.join(parts)


def _build_user_prompt(
    question_text: str,
    audio_b64: str,
    audio_format: str,
    user_prompt_text: str | None,
) -> list[dict[str, Any]]:
    """Build a multimodal user prompt: ``[input_audio, text]``."""
    text = user_prompt_text if user_prompt_text is not None else question_text
    return [
        {
            'type': 'input_audio',
            'input_audio': {'data': audio_b64, 'format': audio_format},
        },
        {'type': 'text', 'text': text},
    ]


# ---------------------------------------------------------------------------
# Task entry point
# ---------------------------------------------------------------------------


@task_registry.register('audio_relisten')
async def audio_relisten_task(task_input: AudioRelistenTaskInput) -> AudioRelistenTaskOutput:
    """Run an audio-comprehension question through the configured agent.

    Steps
    -----
    1. Load and decode the WAV file from ``audio_path``.
    2. Stash the raw waveform + sample-rate in ``ctx.tool_state`` so the
       ``audio_relisten_tool`` can slice it later.
    3. Re-encode the full clip as a base64 WAV and embed it in the user prompt
       as an OpenAI ``input_audio`` content part.
    4. Run the configured agent (e.g. ``main_agent``) with our tool registered.
    5. Score the final response (rule or LLM-judge).
    """
    if not os.path.isfile(task_input.audio_path):
        raise FileNotFoundError(f'Audio file not found: {task_input.audio_path}')

    samples, sample_rate = load_wav(task_input.audio_path)
    duration = float(samples.size) / float(sample_rate)
    logger.debug(
        'audio_relisten task: loaded audio_path=%s sr=%d duration=%.2fs n_samples=%d',
        task_input.audio_path,
        sample_rate,
        duration,
        int(samples.size),
    )

    # Re-encode the full clip as a 16-bit mono WAV for the user message.
    full_wav = encode_wav_bytes(samples, sample_rate)
    audio_b64 = base64.b64encode(full_wav).decode('ascii')
    audio_format = (task_input.audio_format or 'wav').lower()

    # Format the question (with options if multi-choice).
    question_text = _format_question(task_input.question, task_input.options)
    user_prompt = _build_user_prompt(
        question_text=question_text,
        audio_b64=audio_b64,
        audio_format=audio_format,
        user_prompt_text=task_input.user_prompt_text,
    )

    # Stash the raw waveform so audio_relisten_tool can serve future slice requests.
    ctx = require_context()
    ctx.tool_state[AUDIO_STATE_KEY] = samples
    ctx.tool_state[SAMPLE_RATE_STATE_KEY] = sample_rate
    ctx.tool_state[AUDIO_FORMAT_STATE_KEY] = audio_format
    ctx.tool_state[AUDIO_RELISTEN_RECORDS_KEY] = []

    logger.debug(
        'audio_relisten task: question=%r ground_truth=%r reward_style=%s',
        task_input.question[:100],
        task_input.ground_truth[:100],
        task_input.reward_style,
    )

    # Run the agent. main_agent natively accepts ``system_prompt`` / ``user_prompt`` /
    # ``tools`` / ``max_iterations``; deepeyes_agent accepts the same shape.
    agent_output = await task_input.run_agent(
        system_prompt=task_input.system_prompt,
        user_prompt=user_prompt,
        tools=['audio_relisten/audio_relisten_tool'],
        max_iterations=task_input.max_iterations,
    )

    trajectory: list[dict[str, Any]] = []
    if isinstance(agent_output, WithMainTrajectory):
        trajectory = agent_output.main_trajectory
    response = _extract_last_response(trajectory)
    tool_records = list(ctx.tool_state.get(AUDIO_RELISTEN_RECORDS_KEY, []))
    n_tool_calls = sum(1 for r in tool_records if r.get('success'))

    logger.debug(
        'audio_relisten task: response_len=%d n_tool_calls=%d',
        len(response),
        n_tool_calls,
    )

    if task_input.reward_style == 'model':
        try:
            judge_client = ctx.get_openai_client('judge')
        except KeyError:
            logger.warning('audio_relisten task: no "judge" client configured, falling back to "main"')
            judge_client = ctx.openai_client

        result = await compute_score(
            solution_str=response,
            ground_truth=task_input.ground_truth,
            question=task_input.question,
            judge_client=judge_client,
            trajectory=trajectory,
            tool_records=tool_records,
        )
    else:
        result = compute_rule_score(
            solution_str=response,
            ground_truth=task_input.ground_truth,
            options=task_input.options,
            trajectory=trajectory,
            tool_records=tool_records,
        )

    logger.info(
        'audio_relisten task: score=%.3f correct=%s gt=%r',
        result.score,
        result.correct,
        task_input.ground_truth,
    )

    return AudioRelistenTaskOutput(
        score=result.score,
        response=response,
        correct=result.correct,
        n_tool_calls=n_tool_calls,
    )
