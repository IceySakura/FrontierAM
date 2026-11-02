"""The user's own note on how it likes to work, quoted back to it.

A run may hand the worker a preference note (``worker_config['user_memory']``, placed
at the harness's memory path). The same text reaches the user here.

The note is quoted inside a tag rather than merged into the surrounding prose. It is
addressed to the worker in the second person.
"""

from nexus.extensions.tasks.hitl.utils.text import demote_headings, section, tagged


TAG = 'your_preferences'

PREFACE = """\
You have a long-term requirement for the worker describing your working preferences.
When working with the worker, you should remember to assess whether the worker's actions match the preferences.\
"""


def user_memory_section(text: str) -> str:
    """``text`` as the section the user reads, or '' for none.

    Headings inside the note are pushed down two levels. It is written to stand alone,
    with its sections opening at ``#``.
    """
    body = text.strip()
    if not body:
        return ''
    return section('## How you like to work', f'{PREFACE}\n\n{tagged(TAG, demote_headings(body, levels=2))}')


__all__ = ['PREFACE', 'TAG', 'user_memory_section']
