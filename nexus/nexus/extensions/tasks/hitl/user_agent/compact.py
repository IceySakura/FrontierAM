"""Replacing a long user history with an account of it.

The history becomes the system prompt, then one ``role: user`` message holding the
account and the state of the requirement queue. No tool calls, no tool replies and no
earlier turns survive.

One user message is well-formed by construction. A history that keeps message
structure has to answer every tool call, end on a user message, and leave no assistant
message to prefill.
"""

import logging

from typing import Any

from nexus.core.openai_client import OpenAICompatibleClient
from nexus.extensions.tasks.hitl.trace.usage.parse import parse_usage
from nexus.extensions.tasks.hitl.trace.usage.profiles import DEFAULT_PROFILE
from nexus.extensions.tasks.hitl.user_agent.queue import Requirements
from nexus.extensions.tasks.hitl.utils.text import section, tagged


logger = logging.getLogger(__name__)


# Share of the context window the history may fill before it is replaced.
DEFAULT_COMPACT_PCT = 80

SUMMARY_PROMPT = """\
Below is your own conversation so far, as the person who wants this work done.

<conversation>
{text}
</conversation>

Write it up as a first-person account of what happened. Use these headings and drop any
that has nothing under it:

**What I asked for** -- each thing you asked, in the words you asked it.
**What I was told** -- what the worker reported doing, and what it claimed about it.
**What I checked myself** -- what you read or ran, and what you found. Keep the
specifics. File paths, commands, numbers, test names.
**What I am not satisfied with** -- anything you pushed back on, and anything you were
told that you have not verified.
**What I am waiting on** -- questions you asked that are unanswered, and work in
progress.

Keep every detail you would need in order to notice a later claim contradicting an
earlier one. Output the account only.

IMPORTANT: write the account in {language}. Keep the headings as given."""

ACCOUNT_HEADING = '## The conversation so far, in your own words'
QUEUE_HEADING = '## What you asked for, and where it stands'

CONTINUE_INSTRUCTION = (
    'The turn-by-turn record of this conversation has been dropped to save room. The '
    'account above is what you wrote to replace it. The list under it is read from the '
    'run rather than from your memory, and it is authoritative on what has been '
    'accepted. Carry on from here. The worker still has its own memory of the work. Ask '
    'it for anything the account does not cover.'
)

FAILED_INSTRUCTION = (
    'The turn-by-turn record of this conversation has been dropped to save room, and '
    'writing it up failed. What you asked for is all that survives. Ask the worker '
    'where things stand before deciding anything.'
)


def reported_input_tokens(response: Any) -> int:
    """The prompt size the provider reported, or 0 when it reported none.

    A zero counts as unreported.
    """
    dump = response if isinstance(response, dict) else None
    if dump is None:
        dump = response.model_dump() if hasattr(response, 'model_dump') else {}
    trace = parse_usage(DEFAULT_PROFILE, dump)
    return trace.input_tokens if trace.input_tokens > 0 else 0


class Budget:
    """Whether the history has grown past the share of the window it may fill.

    ``window`` and ``pct`` are settings. A client config carries no context size, and
    the model is named by a string the run chooses.

    The size read is the one reported for the last response. It lags by the current
    turn's appends.
    """

    def __init__(self, *, window: int = 0, pct: int = DEFAULT_COMPACT_PCT) -> None:
        self.window = window
        self.pct = pct
        self._warned = False

    @property
    def enabled(self) -> bool:
        return self.window > 0 and self.pct > 0

    @property
    def threshold(self) -> int:
        return self.window * self.pct // 100 if self.enabled else 0

    def exceeded(self, response: Any) -> bool:
        if not self.enabled:
            return False
        used = reported_input_tokens(response)
        if used <= 0:
            if not self._warned:
                self._warned = True
                logger.warning('the provider reported no prompt size. the history will not be compacted')
            return False
        return used > self.threshold


def transcript(history: list[dict[str, Any]]) -> str:
    """``history`` as text for the summary prompt, tool calls and replies included."""
    lines: list[str] = []
    for message in history:
        role = str(message.get('role') or 'unknown')
        if role == 'system':
            continue
        content = message.get('content')
        if isinstance(content, list):
            content = ' '.join(str(part.get('text') or '') for part in content if isinstance(part, dict))
        if content and str(content).strip():
            lines.append(f'[{role}]: {str(content).strip()}')
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {} if isinstance(call, dict) else {}
            lines.append(f'[{role} calls {function.get("name") or "?"}]: {function.get("arguments") or ""}')
    return '\n'.join(lines)


def queue_state(queue: Requirements) -> str:
    """Each requirement and its state, read off the run rather than the account.

    A single-requirement run gets the requirement without the numbering.
    """
    if queue.total == 1:
        state = f'accepted -- {queue.accepted[0]}' if queue.accepted else 'not accepted yet'
        return f'[{state}]\n{queue.segments[0]}'

    lines = [f'{queue.total} in total, and you are on number {queue.position}.', '']
    for position, text in enumerate(queue.segments, 1):
        if position <= len(queue.accepted):
            state = f'accepted -- {queue.accepted[position - 1]}' if queue.accepted[position - 1] else 'accepted'
        elif position == queue.position:
            state = 'working on this one now'
        else:
            state = 'not asked for yet'
        lines.append(f'{position}. [{state}]\n   {text}')
    return '\n'.join(lines)


async def summarize(
    history: list[dict[str, Any]],
    *,
    client: OpenAICompatibleClient,
    language: str,
    prompt: str = SUMMARY_PROMPT,
) -> str:
    """One request asking the model to account for its own conversation.

    ``language`` is the name the account is to be written in. Returns '' when the
    request fails. The caller reports that rather than retrying.
    """
    try:
        response = await client.create_completion(
            messages=[{'role': 'user', 'content': prompt.format(text=transcript(history), language=language)}]
        )
        return str(response.choices[0].message.content or '').strip()
    except Exception as exc:  # noqa: BLE001 - a failed summary must not end the run
        logger.warning('summarising the user history failed: %s', exc)
        return ''


async def compact(
    history: list[dict[str, Any]],
    *,
    client: OpenAICompatibleClient,
    queue: Requirements,
    language: str,
    prompt: str = SUMMARY_PROMPT,
) -> list[dict[str, Any]]:
    """``history`` replaced by its system prompt and one user message.

    The message holds the account, the queue state and what to do next, each in its own
    tag. A failed summary still returns the short shape.
    """
    system = [message for message in history[:1] if message.get('role') == 'system']
    account = await summarize(history, client=client, language=language, prompt=prompt)
    parts = [
        section(ACCOUNT_HEADING, account) if account else '',
        section(QUEUE_HEADING, queue_state(queue)),
        tagged('system-reminder', CONTINUE_INSTRUCTION if account else FAILED_INSTRUCTION),
    ]
    compacted = [*system, {'role': 'user', 'content': '\n\n'.join(part for part in parts if part)}]
    logger.info('replaced %d messages of user history with %d', len(history), len(compacted))
    return compacted


__all__ = [
    'ACCOUNT_HEADING',
    'CONTINUE_INSTRUCTION',
    'DEFAULT_COMPACT_PCT',
    'FAILED_INSTRUCTION',
    'QUEUE_HEADING',
    'SUMMARY_PROMPT',
    'Budget',
    'compact',
    'queue_state',
    'reported_input_tokens',
    'summarize',
    'transcript',
]
