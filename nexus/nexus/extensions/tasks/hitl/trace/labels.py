"""Per-session trace labels carried on spans.

A label is a string stamped onto an LLM call's span and read back off the
exported trajectory, which lets one run-level ``ChatTrackerProcessor`` hold
several conversations at once -- a host user agent plus any number of concurrent
workers.

The path a label travels:

1. the sandbox stamps :data:`LABEL_FIELD` onto ``openai_params``
   (``actor/labelling/proxy.py``);
2. the host handler pops it and enters :func:`labelled`
   (``actor/labelling/handler.py``);
3. ``@observe(self_attrs=['_chat_phase'])`` on
   ``OpenAICompatibleClient.create_completion`` copies it onto the span, which
   requires :func:`enable_labels` to have prepared the client;
4. ``ChatTrackerProcessor.on_end`` reads it and the exporter surfaces it as
   ``trajectory['role']``;
5. :func:`filter_by_label` selects one session's share.

Requires the tracker in ``branch_last`` mode; ``global_first`` exports bare
message lists with nowhere to carry a label.
"""

import logging

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from nexus.trace.processors.chat_tracker import ChatTrackerProcessor, ToolsTrackingMode


logger = logging.getLogger(__name__)


REQUIRED_TOOLS_MODE: ToolsTrackingMode = 'branch_last'
"""The only tools mode a labelled export can be read back from.

``global_first`` exports each trajectory as a bare message list, which has
nowhere to carry a label, and keeps only the run's first tool schema.
"""

Trajectory = list[dict[str, Any]] | dict[str, Any]
"""One exported trajectory: a bare message list in ``global_first``, a dict carrying
``messages`` and ``role`` in ``branch_last``."""


# The label in effect for the current task.
_ACTIVE_LABEL: ContextVar[str | None] = ContextVar('hitl_active_label', default=None)


# Written onto ``openai_params`` in the sandbox, popped by the host handler
# before the call reaches the provider.
LABEL_FIELD = '_hitl_label'

# Prefix of every private field this module puts on a request.
# ``assert_no_label_fields`` checks none survive to the provider.
LABEL_FIELD_PREFIX = '_hitl'

# Label for a request whose owner could not be resolved.
UNATTRIBUTED = '_unattributed'

SEPARATOR = '/'

ACTOR_USER = 'user'
ACTOR_WORKER = 'worker'

# Marks a class built by enable_labels, so a second call is a no-op.
_LABELS_ENABLED = '_hitl_labels_enabled'


class LabelError(ValueError):
    """A name or label that would make attribution ambiguous."""


def validate_name(name: str, *, kind: str = 'name') -> str:
    """Return ``name``, raising :class:`LabelError` if it is empty or holds a ``/``.

    A ``/`` would shift every later label segment and still produce a
    well-formed label, so prefix matching would select the wrong sessions.
    """
    if not name:
        raise LabelError(f'{kind} must be non-empty')
    if SEPARATOR in name:
        raise LabelError(f'{kind} must not contain {SEPARATOR!r}: {name!r}')
    return name


def user_label(session: str = 'orchestrator') -> str:
    """Label for a host-side user-agent call."""
    return f'{ACTOR_USER}{SEPARATOR}{validate_name(session, kind="session")}'


def worker_label(session: str) -> str:
    """Label for one worker session: ``worker/<session>``."""
    return f'{ACTOR_WORKER}{SEPARATOR}{validate_name(session, kind="session")}'


def is_wellformed(label: str) -> bool:
    """True when ``label`` is one this module could have produced."""
    if label == UNATTRIBUTED:
        return True
    segments = label.split(SEPARATOR)
    if any(not segment for segment in segments):
        return False
    return segments[0] in {ACTOR_USER, ACTOR_WORKER} and len(segments) == 2


@contextmanager
def labelled(client: Any, label: str | None) -> Iterator[str]:
    """Label every completion made inside this block, on any client.

    Yields the label actually applied: an empty ``label`` resolves to
    :data:`UNATTRIBUTED`, since an unlabelled call inherits the nearest labelled
    ancestor in the chat tree and those ancestors are shared between sessions.

    The label is task-local, so overlapping blocks in different tasks do not see
    each other's. ``client`` is accepted for call-site readability and is not
    read; the client must already have been passed to :func:`enable_labels`.

    Synchronous on purpose: ``@observe`` snapshots ``_chat_phase`` before
    awaiting the wrapped call, so the label must be set at call time, and a
    synchronous ``with`` cannot yield partway through.
    """
    resolved = label or UNATTRIBUTED
    token = _ACTIVE_LABEL.set(resolved)
    try:
        yield resolved
    finally:
        _ACTIVE_LABEL.reset(token)


def current_label() -> str | None:
    """The label in effect for this task, or None outside any :func:`labelled` block."""
    return _ACTIVE_LABEL.get()


class LabelReadingClient:
    """Mixin giving an OpenAI client a context-scoped ``_chat_phase``.

    Reads :func:`current_label` so ``@observe(self_attrs=['_chat_phase'])`` finds
    the active label on the instance. Assignment raises; use :func:`labelled`.
    """

    @property
    def _chat_phase(self) -> str | None:
        return current_label()

    @_chat_phase.setter
    def _chat_phase(self, value: object) -> None:
        raise AttributeError('_chat_phase is context-scoped and cannot be assigned; use the labelled() context manager')


def enable_labels(client: Any) -> Any:
    """Give ``client`` a ``_chat_phase`` that reads the active label.

    Rebinds ``client.__class__`` to a per-instance subclass carrying the
    property, leaving other instances of the original class untouched. Returns
    ``client``. Idempotent, and a no-op for a client whose class already reads
    the label (a :class:`LabelReadingClient` subclass).

    Required for any client serving labelled requests: the label reaches the span
    through ``@observe(self_attrs=['_chat_phase'])``, and a stock
    ``OpenAICompatibleClient`` has no such attribute, so its spans carry no label.

    Raises :class:`LabelError` for a client whose layout forbids rebinding
    (``object()``, an ``int``, a ``__slots__`` class). Such a client cannot carry
    a label at all, so failing here is the alternative to exporting every
    trajectory unattributed.
    """
    if isinstance(client, LabelReadingClient) or hasattr(type(client), _LABELS_ENABLED):
        return client

    base = type(client)
    try:
        labelled_class = type(
            f'Labelled{base.__name__}',
            (base,),
            {
                '_chat_phase': property(lambda self: current_label()),
                _LABELS_ENABLED: True,
                '__doc__': f'{base.__name__} whose _chat_phase reads the active HITL trace label.',
            },
        )
        # Single base and no added slots, so the instance layout is unchanged and
        # __class__ assignment is permitted.
        client.__class__ = labelled_class
    except TypeError as exc:
        raise LabelError(
            f'cannot label requests on a {base.__name__} client: {exc}. '
            'Its spans would carry no label, so every trajectory would export as '
            f'{UNATTRIBUTED}. Pass a normal client instance, or mix in LabelReadingClient.'
        ) from exc
    return client


def strip_label(params: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Split the label out of a request, leaving nothing for the provider.

    Returns ``(label, remaining_params)``. The caller must use the returned
    dict: ``sanitize_openai_params`` drops only a fixed set of keys, so any
    ``_hitl`` field left behind travels on to the provider as a completion
    parameter.
    """
    remaining = dict(params)
    label = remaining.pop(LABEL_FIELD, None)
    return (label if isinstance(label, str) and label else None), remaining


def assert_no_label_fields(params: dict[str, Any]) -> None:
    """Fail loudly if a private field would reach the provider."""
    leaked = sorted(key for key in params if key.startswith(LABEL_FIELD_PREFIX))
    if leaked:
        raise AssertionError(f'label fields must not reach the provider: {leaked}')


def label_of(trajectory: Trajectory) -> str | None:
    """A trajectory's label, or None when it carries none.

    Only the ``branch_last`` dict form carries one; ``global_first`` exports bare
    message lists.
    """
    if isinstance(trajectory, dict):
        role = trajectory.get('role')
        return role if isinstance(role, str) else None
    return None


def require_labelling_tracker() -> ChatTrackerProcessor:
    """The run-level chat tracker, refused unless it can carry labels.

    Called at agent startup so a misconfigured run stops before it spends anything:
    in ``global_first`` each trajectory exports as a bare message list with nowhere
    to put a label, so every session would export nothing and nothing would say so
    until the record was read.
    """
    from nexus.trace.context import get_processor

    processor = get_processor(ChatTrackerProcessor)
    if processor is None:
        raise LabelError(
            'no run-level chat_tracker is configured; the generator must pass '
            "processors={'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}}}"
        )
    # get_processor is annotated as returning the SpanProcessor base, so narrow to
    # the class actually asked for before reading its mode.
    if not isinstance(processor, ChatTrackerProcessor):
        raise LabelError(f'expected a ChatTrackerProcessor, got {type(processor).__name__}')
    if processor.tools_mode != REQUIRED_TOOLS_MODE:
        raise LabelError(
            f'the run-level chat_tracker is in {processor.tools_mode!r} mode, but labelled traces '
            f'require {REQUIRED_TOOLS_MODE!r}. In {processor.tools_mode!r} each trajectory is '
            'exported as a bare message list with nowhere to carry its label, so every session '
            "would export nothing at all. Configure the recipe with processors={'chat': "
            "{'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}}}."
        )
    return processor


def run_level_trajectories() -> dict[str, Any]:
    """Snapshot the run-level chat tracker.

    Returns an empty result when no tracker is installed, which is the
    before-the-run case rather than a misconfiguration -- unlike at startup, where
    :func:`require_labelling_tracker` refuses the same thing.
    """
    from nexus.trace.context import get_processor

    if get_processor(ChatTrackerProcessor) is None:
        return {'trajectories': []}
    return require_labelling_tracker().snapshot_result()


def filter_by_label(result: dict[str, Any], label: str) -> dict[str, Any]:
    """Keep only the trajectories carrying ``label``.

    Other keys (``tools`` and friends) are passed through untouched. The input
    is not mutated.

    A free function so post-processing can import the same implementation the
    manager uses.
    """
    filtered = dict(result)
    trajectories = result.get('trajectories')
    if not isinstance(trajectories, list):
        return filtered
    filtered['trajectories'] = [trajectory for trajectory in trajectories if label_of(trajectory) == label]
    return filtered


def group_by_label(result: dict[str, Any]) -> dict[str, list[Trajectory]]:
    """Bucket trajectories by label. Unlabelled ones land under UNATTRIBUTED."""
    buckets: dict[str, list[Trajectory]] = {}
    trajectories = result.get('trajectories')
    if not isinstance(trajectories, list):
        return buckets
    for trajectory in trajectories:
        buckets.setdefault(label_of(trajectory) or UNATTRIBUTED, []).append(trajectory)
    return buckets


def unattributed_count(result: dict[str, Any]) -> int:
    """How many trajectories carry no resolvable label.

    A count, not an error signal. A run may legitimately make calls outside any
    :func:`labelled` scope -- the SWE rubrics evaluator runs after the agent
    function and is deliberately unlabelled -- so those land here every run.

    What it is for is comparing against what a run expects: a jump means requests
    started arriving from somewhere new.
    """
    trajectories = result.get('trajectories')
    if not isinstance(trajectories, list):
        return 0
    return sum(1 for trajectory in trajectories if (label_of(trajectory) or UNATTRIBUTED) == UNATTRIBUTED)


__all__ = [
    'ACTOR_USER',
    'ACTOR_WORKER',
    'LABEL_FIELD',
    'REQUIRED_TOOLS_MODE',
    'UNATTRIBUTED',
    'LabelError',
    'LabelReadingClient',
    'Trajectory',
    'assert_no_label_fields',
    'current_label',
    'enable_labels',
    'filter_by_label',
    'group_by_label',
    'is_wellformed',
    'label_of',
    'labelled',
    'require_labelling_tracker',
    'strip_label',
    'unattributed_count',
    'user_label',
    'validate_name',
    'worker_label',
]
