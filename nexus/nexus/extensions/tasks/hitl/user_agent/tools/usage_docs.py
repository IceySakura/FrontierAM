"""Metadata keys a tool uses to describe itself beyond its schema.

A tool's ``description`` says what it is and goes into the schema the model sees. A
``usage_doc`` says how the thing is used across turns and is collected into the system
prompt at assembly. ``usage`` names the group of tools sharing one such text, and each
group becomes one section of the prompt.

``ToolMeta`` reads only the keys it knows (``core/tool/registry.py:53-72``). A
``usage_doc`` never reaches the provider as part of the schema.
"""

from collections.abc import Mapping
from typing import Any


USAGE = 'usage'
USAGE_DOC = 'usage_doc'


class UsageDocError(ValueError):
    """A tool declares its usage in a way that cannot be assembled."""


def tool_meta(obj: Any) -> Mapping[str, Any]:
    """The meta dict recorded when ``obj`` was registered."""
    recorded = getattr(obj, 'nexus_meta', None)
    if not isinstance(recorded, Mapping):
        return {}
    meta = recorded.get('meta')
    return meta if isinstance(meta, Mapping) else {}


def usage_of(obj: Any) -> tuple[str, str]:
    """The ``(usage, usage_doc)`` pair a tool declares, or ``('', '')``.

    Declaring one without the other raises.
    """
    meta = tool_meta(obj)
    usage = meta.get(USAGE, '')
    usage_doc = meta.get(USAGE_DOC, '')
    if not isinstance(usage, str) or not isinstance(usage_doc, str):
        raise UsageDocError(f'{USAGE} and {USAGE_DOC} must be strings, got {usage!r} and {usage_doc!r}')
    if bool(usage) != bool(usage_doc.strip()):
        raise UsageDocError(
            f'a tool declaring {USAGE}={usage!r} must also declare a non-empty {USAGE_DOC} '
            f'(and the reverse); got usage_doc of {len(usage_doc)} chars'
        )
    return usage, usage_doc.strip()


def collect_usage_docs(tools: Mapping[str, Any]) -> list[tuple[str, str]]:
    """One ``(usage, usage_doc)`` per usage present in ``tools``.

    De-duplicated by usage. Two tools claiming one usage with different text raises.
    """
    collected: dict[str, str] = {}
    origin: dict[str, str] = {}
    for name in sorted(tools):
        usage, usage_doc = usage_of(tools[name])
        if not usage:
            continue
        previous = collected.get(usage)
        if previous is None:
            collected[usage] = usage_doc
            origin[usage] = name
            continue
        if previous != usage_doc:
            raise UsageDocError(
                f'{origin[usage]!r} and {name!r} both declare usage {usage!r} '
                'with different usage_doc; they must share one text'
            )
    return [(usage, collected[usage]) for usage in sorted(collected)]


__all__ = [
    'USAGE',
    'USAGE_DOC',
    'UsageDocError',
    'collect_usage_docs',
    'tool_meta',
    'usage_of',
]
