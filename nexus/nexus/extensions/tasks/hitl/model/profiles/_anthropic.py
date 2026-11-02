"""Request-arg shapes shared by the Anthropic-protocol worker profiles.

Claude reached over Bedrock and over Vertex takes the same thinking block, so the
one rule both have to obey lives here rather than in each profile.
"""

from typing import Any


def thinking_block(thinking_type: str) -> dict[str, Any] | None:
    """The ``thinking`` request arg for this type, or None to omit the field.

    ``display`` is only legal while thinking is on: a request carrying it next to
    ``type='disabled'`` is rejected. So the disabled block is the bare type, and
    only the enabled types ask for a summarized block, which is what lets the
    thinking content survive the round trip.
    """
    if not thinking_type:
        return None
    if thinking_type == 'disabled':
        return {'type': 'disabled'}
    return {'type': thinking_type, 'display': 'summarized'}


__all__ = ['thinking_block']
