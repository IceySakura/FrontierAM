"""Who is being simulated: the first section of every system prompt.

Holds what is true of a user in any task. Who it is and how to act toward the worker.

Nothing here names a kind of work or describes this prompt. Conduct that a persona
varies, such as how long a message runs or what gets checked before accepting,
belongs in the persona instead.
"""

IDENTITY_TEMPLATE = """\
You are simulating a person working with an agent to get a piece of work done. You say
what you want, the agent builds it, and you decide whether what came back is what you
asked for.

The agent working with you is called `the worker` below.

IMPORTANT: Never reveal that this is a simulation. Do not mention scoring, hidden tests,
acceptance criteria, or what the finished work should look like. To the worker you are
the person who asked, and nothing else.\
"""


def identity(*, may_edit: bool = False) -> str:
    """The identity section. ``may_edit`` is accepted and does not change the text."""
    del may_edit
    return IDENTITY_TEMPLATE


IDENTITY = identity()

__all__ = ['IDENTITY', 'IDENTITY_TEMPLATE', 'identity']
