"""Tools whose call is a move in the run, not an action on the world.

Speaking to a worker and accepting its work. Both bodies validate their arguments and
report them in ``meta``. The sending and the queue advance belong to whichever handler
set the run binds these tools to. A tool here without its handler is a call that
returns and changes nothing.
"""

from typing import Literal

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tasks.hitl.user_agent.tools.usage_docs import USAGE, USAGE_DOC


SEND_MESSAGE_TOOL = 'hitl/user/send_message'
ACCEPT_TOOL = 'hitl/user/accept'

# Schema names of the tools that put something in front of a worker, as opposed to the
# ones the user runs for itself. A reminder fires after one of these.
WORKER_FACING_TOOLS: frozenset[str] = frozenset({'send_message'})

# What a round of instruction is for. Each value is described in ``INTENT_DOC``.
Intent = Literal[
    'request',
    'revise',
    'ask',
    'answer',
    'report',
    'continue',
    'meta',
    'material',
    'prescribe',
    'challenge',
    'nudge',
]

StopReason = Literal['satisfied', 'gave_up']

# What actually went out, shown above the reply it drew.
SENT_TEMPLATE = '<sent>\n{text}\n</sent>\n\n{report}'

PENDING_QUESTION_REFUSAL = (
    'The task cannot end here. The worker asked you something and is waiting on the '
    'answer. Reply to the question, then accept when you are content.'
)

EMPTY_MESSAGE_REFUSAL = 'An empty message is not allowed. Say what you want.'

MISSING_DRAFT_REFUSAL = (
    'Your message was not sent, because `draft` was empty. Fill in both fields and send the call again.'
)
"""Refuses a message whose ``draft`` was left empty. Says nothing about its content."""

LANGUAGE_REFUSAL = (
    'Your message was not sent, because it was not written in {name}. Write it in {name} and send the call again.'
)
"""Refuses a message in a language other than the run's. ``{name}`` is that language."""


DELEGATION_DOC = """\
## Talking to a worker

`send_message` sends one instruction to a worker in a single continuous session. The
worker holds the full history of that session.

- `draft` is your own reasoning, which the worker never reads. Write it first. A call
  with `draft` empty is refused rather than sent.
- `message` carries the instruction the worker reads.
- `intent` classifies what this round of instruction is for, from the list under "What
  your instruction is for".\
"""

INTENT_DOC = """\
## What your instruction is for

Pick the intent that classifies what this round of instruction is for.

- **`request`** sends a piece of requirement, usually from the task provided for you.
- **`revise`** changes the content of a previous requirement.
- **`report`** states what is broken, missing or unasked for in the work.
- **`ask`** asks a question about the work, can be consulting, clarifying, questioning, etc.
- **`answer`** replies to a question the worker put to you.
- **`challenge`** disputes what the worker told you and asks it to show you.
- **`continue`** tells the worker to carry on with work already asked for.
- **`material`** hands over the thing itself: a path, a log, an output, an observation, background the worker has no way to know.
- **`meta`** asks about the worker like what it can reach and what it plans to do.
- **`prescribe`** dictates a method or procedure for the worker.
- **`nudge`** asks for progress or how far along the work is.\
"""

ACCEPTANCE_DOC = """\
## Accepting work

`accept` states that the task is done in full and should be called when every provided
requirement has been sent to the worker and addressed in full. Do not accept once part
of the task works.

- `checked` is the task gone through part by part and cannot be empty.
- `summary` is one line on what the work now does that you wanted.
- `reason` is `satisfied` where the work does what the task asked for, `gave_up` where
  you are stopping without getting there.\
"""

INCOMPLETE_ACCEPT_REFUSAL = (
    'Not accepted, because `checked` was empty. Go through the task part by part and send the call again.'
)
"""Refuses an acceptance that accounted for nothing. Says nothing about the work."""

LAST_ACCEPT_REMINDER = (
    'The task you are accepting is the last one you were given. Accepting it will end the '
    'run. Before making your final decision, make sure you have sent all parts of the '
    'given tasks and that you have not left any phases or claims unaddressed. Call '
    '`accept` again to go ahead, or ask for whatever is missing first.'
)
"""Held back the first acceptance of the final requirement.

Not a refusal. The same call again goes through. Fires once, and a message to the
worker in between arms it again.
"""


SESSION_LOST_REASON = 'worker_session_lost'

SESSION_LOST_NOTICE = 'Your message did not reach the worker because the worker died. Accept this task immediately if it is nearly done or give up if it is not.'
"""Ends a run whose worker session was torn down. Reused by every agent driving one."""

NOT_DELIVERED = 'Your message was not sent to the worker. Rewrite it and send it again.'
"""Answers a message the backend refused to send. Reused by every agent driving one.

The refusal leaves the session untouched. The same message can go again.
"""


class SendMessageInput(BaseModel):
    intent: Intent = Field(description='What this round of instruction is for.')
    message: str = Field(description='The instruction the worker reads, in your own words.')
    draft: str = Field(
        default='',
        description=(
            'Your own private reasoning, which the worker never reads. Write it before the message. '
            'It is usually longer than the message and cannot be empty.'
        ),
    )
    to: str = Field(
        default='',
        description=('Which coding agent this goes to. Empty when there is only one.'),
    )
    rewrite: bool = Field(
        default=True,
        description=('Whether your wording may be adjusted by a rewriter. Set false to send it exactly as written.'),
    )


class AcceptInput(BaseModel):
    reason: StopReason = Field(description='Why you are accepting.')
    summary: str = Field(description='Briefly, what you are accepting and why it is good enough.')
    checked: str = Field(
        default='',
        description=(
            'The task gone through part by part: each thing the task asked for, and what you did '
            'about that thing. Required when reason is `satisfied`. A call leaving this field '
            'empty does not accept.'
        ),
    )


@tool_registry.register(
    SEND_MESSAGE_TOOL,
    meta={
        'name': 'send_message',
        'description': 'Send a message to a worker in your ongoing conversation and read its reply.',
        'tool_type': ToolType.LOCAL,
        USAGE: 'delegation',
        USAGE_DOC: f'{DELEGATION_DOC}\n\n{INTENT_DOC}',
    },
)
async def send_message_tool(params: SendMessageInput, context: ToolContext) -> ToolResponse:
    """Report what is being sent. The handler delivers it."""
    del context
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Sending to {params.to or "the worker"}.'}],
        meta={'intent': params.intent, 'to': params.to, 'message': params.message, 'rewrite': params.rewrite},
    )


@tool_registry.register(
    ACCEPT_TOOL,
    meta={
        'name': 'accept',
        'description': 'Accept the work in its current state, ending your part in it.',
        'tool_type': ToolType.LOCAL,
        USAGE: 'acceptance',
        USAGE_DOC: ACCEPTANCE_DOC,
    },
)
async def accept_tool(params: AcceptInput, context: ToolContext) -> ToolResponse:
    """Report the acceptance. The handler advances the queue."""
    del context
    return ToolResponse(
        output=[{'type': 'text', 'text': params.summary or 'Accepted.'}],
        meta={'reason': params.reason, 'summary': params.summary},
    )


__all__ = [
    'ACCEPTANCE_DOC',
    'ACCEPT_TOOL',
    'DELEGATION_DOC',
    'EMPTY_MESSAGE_REFUSAL',
    'INCOMPLETE_ACCEPT_REFUSAL',
    'INTENT_DOC',
    'LANGUAGE_REFUSAL',
    'LAST_ACCEPT_REMINDER',
    'MISSING_DRAFT_REFUSAL',
    'NOT_DELIVERED',
    'PENDING_QUESTION_REFUSAL',
    'SEND_MESSAGE_TOOL',
    'SENT_TEMPLATE',
    'SESSION_LOST_NOTICE',
    'SESSION_LOST_REASON',
    'WORKER_FACING_TOOLS',
    'AcceptInput',
    'Intent',
    'SendMessageInput',
    'StopReason',
]
