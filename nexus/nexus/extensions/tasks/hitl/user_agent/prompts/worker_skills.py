"""The worker's own skills: what it has, and how to have it run one.

Skills the run installed into the harness, which a user reaches by naming one at the
start of a message. Separate from the user's own skills
(``user_agent/tools/skill_tool.py``): those are text the user reads, these are
approaches the worker follows.
"""

from collections.abc import Mapping

from nexus.extensions.tasks.hitl.utils.text import section


def worker_command_lines(commands: Mapping[str, str], *, prefix: str) -> str:
    """One line per skill the worker can be asked for, or '' when there are none.

    ``prefix`` is ``SessionManager.command_prefix``. Empty means this harness has no
    way to be asked for one, so nothing is listed.

    Read by both :func:`worker_skills_section` and the heavy reminder
    (``user_agent/reminders.py``).
    """
    if not prefix:
        return ''
    return '\n'.join(
        f'- `{prefix}{name}` -- {description.strip()}'
        for name, description in sorted(commands.items())
        if name.strip() and description.strip()
    )


def worker_skills_section(commands: Mapping[str, str], *, prefix: str) -> str:
    """What the worker's own skills are and how to have it use one, or '' for none.

    Written out with the literal form rather than described, because the form is the
    whole of it: a user that knows a skill exists and not how to name it cannot use it.

    The form is shown with a placeholder rather than with one of the names. A name in
    the example reads as the one to reach for.
    """
    listed = worker_command_lines(commands, prefix=prefix)
    if not listed:
        return ''
    body = '\n'.join([
        'The worker has written approaches of its own it can follow. You can have it',
        'use one by naming it at the start of a message:',
        '',
        f'    {prefix}<skill>',
        f'    {prefix}<skill> <anything else you want to say>',
        '',
        'The name has to be the first thing in the message. Whatever follows it reaches',
        'the worker as an ordinary instruction, so the two can be said at once.',
        '',
        'What it has:',
        '',
        listed,
        '',
        "These are the worker's own, and separate from any skill of yours -- yours is",
        'text you read.',
    ])
    return section("## The worker's own skills", body)


__all__ = ['worker_command_lines', 'worker_skills_section']
