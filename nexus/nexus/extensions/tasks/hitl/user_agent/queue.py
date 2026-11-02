"""The requirements a user holds, released one at a time.

A queue of length one is the common case: of the 4996 records in the chain
dataset, 497 hold a single requirement and the rest hold between two and nineteen.
A lone requirement is a queue that releases once and then has nothing left.

Release is pulled: accepting the current requirement is what makes the next one
available. How many there are is not told to the user by default.
"""

import logging

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


DEFAULT_TURN_CAP = 120
"""Turns one user gets before a run is cut off.

One place because the cap is declared twice over -- on the agent that enforces it and
on the task that sends it -- and the two disagreeing means whoever drives the agent
directly gets a different run length from whoever drives it through the task.
"""


class QueueError(ValueError):
    """A requirement queue cannot be built or advanced as asked."""


@dataclass
class Requirements:
    """An ordered set of requirements and how far through them the run is."""

    segments: tuple[str, ...]
    index: int = 0
    accepted: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.segments:
            raise QueueError('a run needs at least one requirement')
        blank = [position for position, text in enumerate(self.segments) if not text.strip()]
        if blank:
            raise QueueError(f'requirements at {blank} are empty')
        if not 0 <= self.index < len(self.segments):
            raise QueueError(f'index {self.index} is outside 0..{len(self.segments) - 1}')

    @classmethod
    def build(cls, segments: Sequence[str]) -> 'Requirements':
        """Build from raw text, dropping blanks so a count is never inflated."""
        kept = tuple(text.strip() for text in segments if text and text.strip())
        if not kept:
            raise QueueError('no requirements after dropping empty ones')
        return cls(segments=kept)

    @property
    def total(self) -> int:
        return len(self.segments)

    @property
    def current(self) -> str:
        return self.segments[self.index]

    @property
    def position(self) -> int:
        """1-based, for text a person reads."""
        return self.index + 1

    @property
    def exhausted(self) -> bool:
        """True once the last requirement has been accepted."""
        return len(self.accepted) >= self.total

    @property
    def remaining(self) -> int:
        return max(0, self.total - len(self.accepted))

    @property
    def on_last(self) -> bool:
        """True while the requirement in front of the user is the final one."""
        return self.index == self.total - 1

    def accept(self, summary: str = '') -> str | None:
        """Record acceptance of the current requirement and return the next.

        Returns None when nothing follows, which is what ends the run. Accepting
        an exhausted queue raises rather than silently doing nothing, because a
        caller doing that has lost track of whether the run is over.
        """
        if self.exhausted:
            raise QueueError('every requirement has already been accepted')
        self.accepted.append(summary)
        if self.exhausted:
            return None
        self.index += 1
        return self.current


def from_task_input(*, requirements: Sequence[str], user_prompt: Any = '') -> 'Requirements':
    """A queue from whichever field the task carried its requirements in.

    Two fields carry the same thing because tasks differ in what they can send.
    ``requirements`` is one entry per thing the user wants, in the order they will
    ask for them. ``user_prompt`` is the single combined statement every task already
    sends, and is all a task written before this recipe can offer.

    ``requirements`` wins whenever it is non-empty, and ``user_prompt`` is a fallback
    rather than an addition: a run carrying both is a chain-aware task that also
    filled the field its predecessor used, not a run being asked for something extra.
    Preferring the prompt would collapse a nineteen-segment chain into one segment
    and still score, so the field that can express the chain wins.

    A prompt that is not the first requirement is logged. The two disagreeing means
    they were built from different sources, and what reaches the user is
    ``requirements``.
    """
    from nexus.extensions.agents.installed_common import coerce_user_prompt

    segments = [text for text in requirements if text and text.strip()]
    prompt = coerce_user_prompt(user_prompt).strip() if user_prompt else ''

    if segments and prompt and prompt != segments[0]:
        logger.warning(
            'user_prompt is not requirements[0]; the user is given requirements. prompt=%.60r first=%.60r',
            prompt,
            segments[0],
        )
    if not segments and prompt:
        logger.info('no requirements were given; treating user_prompt as a single requirement')
        segments = [prompt]
    if not segments:
        raise QueueError('this run needs either requirements or a user_prompt; both were empty')
    return Requirements.build(segments)


__all__ = [
    'DEFAULT_TURN_CAP',
    'QueueError',
    'Requirements',
    'from_task_input',
]
