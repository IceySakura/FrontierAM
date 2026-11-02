"""What this user is working with: the directory, the tool, and how to address it."""

from collections.abc import Sequence


def environment_facts(*, working_dir: str, workers: Sequence[str], product: str = '') -> list[str]:
    """Where the work is, what the user is working with, and how to address it.

    ``product`` is ``SessionManager.product``: what a person would call the tool.

    The name in ``workers`` is a routing name. It appears in ``send_message``'s ``to``
    field and in what a turn's result is labelled, and a message body addresses the
    thing it is sent to in the second person.

    ``product`` defaults to empty, which drops the product clause rather than leaving a
    gap: a manager from outside this module may not name one.
    """
    facts = [f'The work is in {working_dir}.']
    named = product.strip()
    if len(workers) == 1:
        subject = f'{named}, a coding agent' if named else 'a coding agent'
        facts.append(
            f'You are working with {subject}. Write to it as `you` in a message; `{workers[0]}` is '
            'what these surroundings call it when they report back on it.'
        )
    elif len(workers) > 1:
        listed = ', '.join(f'`{worker}`' for worker in workers)
        subject = f'several instances of {named}' if named else 'several coding agents'
        facts.append(
            f'You are working with {subject}, which these surroundings call {listed}. Name one in '
            "`send_message`'s `to` field to pick who a message goes to, and write to it as `you` in "
            'the message itself.'
        )
    return facts


__all__ = ['environment_facts']
