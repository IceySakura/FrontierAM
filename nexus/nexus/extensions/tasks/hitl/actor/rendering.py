"""Rendering a worker turn into the tagged text the user simulator reads.

How much of the turn is available depends on the backend, because the raw material
does: both Claude Code backends record what the worker said and the tools it called
between saying things, from ``recent_events`` in PTY mode and from the stream-json
``assistant`` events in print mode; a backend with neither has only the combined
``output``. Each turns its own material into :class:`Step` values and
:func:`walk_steps` puts them in order, so the shape the simulator reads does not
depend on which backend produced it.

Rendering is reached through ``SessionManager.report``, which each backend
implements.

``WORKER_REPORT_PROTOCOL`` is the same contract stated for the simulator's system
prompt, so the tags and their explanation change together.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from nexus.extensions.tasks.hitl.utils.text import truncate


# Single tier for now. Adding one means extending this alias and branching in
# render_worker_turn; call sites that pass nothing are unaffected.
Detail = Literal['default']

DEFAULT_DETAIL: Detail = 'default'


WORKER_REPORT_PROTOCOL = """\
The worker's reply arrives in tagged sections:

- <worker_progress> -- how the turn went, in the order it happened: what the
  worker said, and `[used X: detail]` for each tool it called between saying
  things. Context, not results.
- <final_response> -- what the worker claims it accomplished. Treat this as a
  claim to verify, not as evidence.
- <worker_question> -- the worker is blocked and waiting, and this turn has no
  claimed result. Your next message is sent as the answer to it; answer it before
  doing anything else. A question may list the options it sees, with what each one
  costs; pick among them or say what you want instead.
- <note> -- a limitation of this run's reporting, not something the worker said.
"""


TOOL_STEP_PREFIX = '[used '
"""Opens a step standing for one tool the worker called."""

# What one tool's detail may run to before it is cut. Matches the limits the PTY
# proxy already applies when it builds recent_events
# (``agents/claude_code/interactive/events.py``), so a report is the same length
# whichever backend produced it.
DETAIL_HEAD = 150
DETAIL_TAIL = 50
TRUNCATED_MARKER = ' ...[truncated]... '

# What one section of a report may run to. A worker is an agent rather than a person
# pasting, so this is a backstop against one turn emitting a whole build log, not a
# budget: across 678 reports from the recorded batches the largest was 17,899
# characters and the 99th percentile 12,268, so nothing a worker normally says
# reaches it. The marker names how much went, since a silently cut report reads as a
# complete one.
SECTION_LIMIT = 60_000


@dataclass(frozen=True)
class Step:
    """One thing the worker did: said something, or called a tool.

    ``name`` empty means the step is speech and ``text`` is what was said.
    Otherwise ``name`` is the tool and ``text`` the argument worth naming.
    """

    text: str
    name: str = ''

    @property
    def is_tool(self) -> bool:
        return bool(self.name)


def said(text: str) -> Step:
    """A step for something the worker said."""
    return Step(text=text)


def used(name: str, detail: str = '') -> Step:
    """A step for a tool call, carrying the argument the backend picked out."""
    return Step(text=truncate_detail(' '.join(detail.split())) if detail else '', name=name)


def detail_from(arguments: object, fields: Sequence[str]) -> str:
    """The first of ``fields`` ``arguments`` carries as a non-empty string.

    A backend names the fields its harness uses; an empty result reports the tool by
    name alone.
    """
    if not isinstance(arguments, Mapping):
        return ''
    for field_name in fields:
        value = cast(Mapping[str, Any], arguments).get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return ''


def truncate_detail(value: str) -> str:
    """``value`` cut to the head and tail a report shows."""
    if len(value) <= DETAIL_HEAD + DETAIL_TAIL:
        return value
    return f'{value[:DETAIL_HEAD]}{TRUNCATED_MARKER}{value[-DETAIL_TAIL:]}'


def render_step(step: Step) -> str:
    """One line for a tool step, or the text itself for a spoken one."""
    if not step.is_tool:
        return step.text
    inside = f'{step.name}: {step.text}' if step.text else step.name
    return f'{TOOL_STEP_PREFIX}{inside}]'


def walk_steps(steps: Iterable[Step]) -> tuple[str, ...]:
    """``steps`` rendered in order, consecutive tool calls grouped into one block.

    A run of tool calls between two of the worker's messages belongs between those
    messages, so it becomes one block there -- one line per call, since a call now
    carries an argument and a single line would not stay readable.
    """
    rendered: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            rendered.append('\n'.join(pending))
            pending.clear()

    for step in steps:
        if step.is_tool:
            pending.append(render_step(step))
        elif step.text.strip():
            flush()
            rendered.append(step.text.strip())
    flush()
    return tuple(rendered)


def spoken_steps(steps: Iterable[Step]) -> tuple[Step, ...]:
    """The steps that are speech, in order. What a turn's final message comes from."""
    return tuple(step for step in steps if not step.is_tool and step.text.strip())


def read_steps(entries: object) -> tuple[Step, ...]:
    """``Step`` values from the ``text`` / ``tool_use`` dicts a backend recorded.

    A ``tool_use`` entry names its tool and the argument worth naming, which the
    backend picked out of its own harness's tool vocabulary. Entries of any other
    type are skipped.
    """
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ()
    steps: list[Step] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        item = cast(Mapping[str, Any], entry)
        kind = item.get('type')
        if kind == 'tool_use':
            if name := str(item.get('name') or ''):
                steps.append(used(name, str(item.get('detail') or '')))
        elif kind == 'text' and (text := str(item.get('text') or '')).strip():
            steps.append(said(text))
    return tuple(steps)


def report_from_steps(
    entries: object,
    *,
    output: str,
    question: str = '',
    empty_note: str = '',
    detail: Detail = DEFAULT_DETAIL,
) -> str:
    """One turn rendered from the steps a backend recorded.

    The last thing the worker said is its claim about the turn, and everything before
    it is how the turn went. ``output`` is that same claim as the backend reports it
    whole, and is what the claim is taken from: a step carries whatever the backend's
    event stream kept, which for the PTY backend is a 350/150 summary. The step is
    used only where ``output`` is empty, and a turn with no speech at all falls back
    to ``output`` and says so through ``empty_note``.
    """
    steps = read_steps(entries)
    spoken = spoken_steps(steps)
    if not spoken:
        return render_worker_turn(
            final=output,
            steps=walk_steps(steps),
            question=question,
            note=empty_note,
            detail=detail,
        )
    last = steps.index(spoken[-1])
    return render_worker_turn(
        final=output.strip() or spoken[-1].text.strip(),
        steps=walk_steps(steps[:last]),
        question=question,
        detail=detail,
    )


def render_questions(questions: object, *, agent: str = 'The worker') -> str:
    """One block per question, with the options it offered and their trade-offs.

    ``questions`` is an ``AskUserQuestion`` tool input's ``questions`` list. Each
    entry contributes its header and text, then one line per option: the label,
    the description of what choosing it costs, and the preview if it carries one.

    A question whose options are dropped reads as "you decide", which is what the
    simulator then answers -- the trade-off the worker wrote down is the only
    thing that makes the choice a choice.
    """
    if not isinstance(questions, list):
        return ''
    blocks: list[str] = []
    for index, raw_question in enumerate(questions, 1):
        if not isinstance(raw_question, Mapping):
            continue
        question = cast(Mapping[str, Any], raw_question)
        header = str(question.get('header') or '').strip()
        text = str(question.get('question') or '').strip()
        heading = f'{index}. {f"{header}: " if header else ""}{text}'
        lines = [heading]
        if question.get('multiSelect'):
            lines.append('   (several answers are allowed)')
        lines += _option_lines(question.get('options'))
        blocks.append('\n'.join(lines))
    if not blocks:
        return ''
    return f'{agent} is blocked and waiting on you.\n\n' + '\n\n'.join(blocks)


def _option_lines(options: object) -> list[str]:
    """The indented lines describing each option, or none when there are none."""
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        return []
    lines: list[str] = []
    for raw_option in options:
        if not isinstance(raw_option, Mapping):
            continue
        option = cast(Mapping[str, Any], raw_option)
        label = str(option.get('label') or '').strip()
        if not label:
            continue
        lines.append(f'   - {label}')
        for key in ('description', 'preview'):
            body = str(option.get(key) or '').strip()
            if body:
                lines.append(_indent(body, '     '))
    return lines


def _indent(body: str, prefix: str) -> str:
    return '\n'.join(f'{prefix}{line}' if line.strip() else '' for line in body.splitlines())


def render_worker_turn(
    *,
    final: str,
    steps: tuple[str, ...] = (),
    question: str = '',
    note: str = '',
    detail: Detail = DEFAULT_DETAIL,
) -> str:
    """Render one worker turn as tagged sections.

    ``<worker_progress>`` carries how the turn went and is emitted whether or not
    the turn ended in a question, so the simulator reads one shape rather than two.

    What ``final`` is depends on whether the turn ended in a question. Asking is not
    claiming, so a turn that asks has no claimed result: there ``final`` is the last
    thing the worker said before asking, which belongs to the progress. Otherwise it
    is the claim, and gets its own section.

    ``<final_response>`` is emitted even with no steps. A shape that varies with
    how many messages the worker happened to send gives nothing stable to read.
    """
    del detail  # one tier so far; kept in the signature so adding tiers is additive

    progress = list(steps)
    if question and final.strip():
        # Said last, so it goes last: the steps are in the order they happened.
        progress.append(final.strip())
    sections = []
    if progress:
        sections.append(_tag('worker_progress', '\n\n'.join(progress)))
    if question:
        sections.append(_tag('worker_question', question))
    else:
        sections.append(_tag('final_response', final))
    if note:
        sections.append(_tag('note', note))
    return '\n\n'.join(sections)


def split_messages(messages: object) -> tuple[str, tuple[str, ...]] | None:
    """Split a list of assistant messages into ``(final, earlier)``.

    Returns None when ``messages`` is not a usable list, which is how a caller
    tells "this backend cannot split" apart from "the worker sent one message".
    """
    if not isinstance(messages, list):
        return None
    segments = tuple(str(item).strip() for item in messages if str(item).strip())
    if not segments:
        return None
    return segments[-1], segments[:-1]


def _tag(name: str, body: str) -> str:
    """``body`` wrapped in ``<name>``, capped at ``SECTION_LIMIT``, kept when empty.

    ``utils.text.tagged`` drops an empty body; here an empty
    ``<final_response>`` still has to be emitted, so the section set does not
    change with how much the worker said.
    """
    return f'<{name}>\n{truncate(body.strip(), SECTION_LIMIT)}\n</{name}>'


__all__ = [
    'DEFAULT_DETAIL',
    'DETAIL_HEAD',
    'DETAIL_TAIL',
    'TOOL_STEP_PREFIX',
    'WORKER_REPORT_PROTOCOL',
    'Detail',
    'Step',
    'detail_from',
    'read_steps',
    'render_questions',
    'render_step',
    'render_worker_turn',
    'report_from_steps',
    'said',
    'spoken_steps',
    'split_messages',
    'truncate_detail',
    'used',
    'walk_steps',
]
