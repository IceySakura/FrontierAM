"""Uploading each message once per trace, rather than once per later request.

The upstream tracker records the complete message list on every generation, so a
conversation of N calls uploads N + (N-1) + ... + 1 messages. A message already
uploaded is replaced here by a marker naming where it first appeared; the first
upload carries it in full.

Identity is the message's own content rather than its position, since concurrent
worker sessions and Claude Code's own sub-calls each carry an independent history and
arrive interleaved.
"""

import hashlib
import json
import logging

from typing import Any


logger = logging.getLogger(__name__)


DIGEST_SIZE = 8
"""Bytes of the message digest, which is a dictionary key and is never uploaded."""

MESSAGES_FIELD = 'messages'
TOOLS_FIELD = 'tools'


def fingerprint(message: Any) -> str:
    """A digest of ``message`` as it would be uploaded, over its keys in sorted order."""
    return hashlib.blake2b(
        json.dumps(message, sort_keys=True, default=str).encode(), digest_size=DIGEST_SIZE
    ).hexdigest()


class MessageLedger:
    """What has already been uploaded, and where.

    One ledger per trace, so a message shared by two labels is uploaded once. A marker
    names the label alongside the step, which several labels number from 1.
    """

    def __init__(self) -> None:
        self._where: dict[str, str] = {}
        self._tools: dict[str, str] = {}
        self.uploaded = 0
        self.referenced = 0

    def rewrite(self, messages: Any, *, where: str) -> Any:
        """``messages`` with the already-uploaded ones replaced by markers.

        ``where`` names this observation for a marker pointing back at it.
        """
        if not isinstance(messages, list):
            return messages
        out: list[Any] = []
        for message in messages:
            digest = fingerprint(message)
            first_seen = self._where.get(digest)
            if first_seen is None:
                self._where[digest] = where
                self.uploaded += 1
                out.append(message)
            else:
                self.referenced += 1
                out.append(f'<same as {first_seen}>')
        return out

    def tools_first_seen(self, tools: Any, *, where: str) -> str:
        """Where this exact tool set was uploaded, or '' when this is the first time."""
        if not tools:
            return ''
        digest = fingerprint(tools)
        first_seen = self._tools.get(digest)
        if first_seen is None:
            self._tools[digest] = where
            return ''
        return first_seen

    def summary(self) -> dict[str, int]:
        """What the ledger spared, for the trace's own metadata."""
        total = self.uploaded + self.referenced
        return {
            'messages_uploaded': self.uploaded,
            'messages_referenced': self.referenced,
            'messages_total': total,
        }


def deduplicate(payload: Any, ledger: MessageLedger, *, where: str) -> Any:
    """One observation's ``input`` with its repeated parts replaced by markers.

    Anything that is not a message list is passed through unchanged.
    """
    if not isinstance(payload, dict) or MESSAGES_FIELD not in payload:
        return payload
    out = dict(payload)
    out[MESSAGES_FIELD] = ledger.rewrite(payload[MESSAGES_FIELD], where=where)
    tools = out.get(TOOLS_FIELD)
    if tools:
        first_seen = ledger.tools_first_seen(tools, where=where)
        if first_seen:
            out[TOOLS_FIELD] = f'<same {len(tools)} tools as {first_seen}>'
    return out


class DeduplicatingSpan:
    """A Langfuse span that rewrites the ``input`` of every observation under it.

    Wrapping is inherited: an observation started through this one is wrapped too, so
    wrapping a label's span covers its subagents' generations. ``__getattr__`` passes
    everything else through to the SDK.

    ``scope`` prefixes the markers this span's observations produce.
    """

    def __init__(self, inner: Any, ledger: MessageLedger, *, scope: str = '') -> None:
        self._inner = inner
        self._ledger = ledger
        self._scope = scope

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _where(self, name: Any) -> str:
        step = str(name or '?')
        return f'{self._scope} {step}' if self._scope else step

    def start_observation(self, **kwargs: Any) -> 'DeduplicatingSpan':
        if 'input' in kwargs:
            kwargs['input'] = deduplicate(kwargs['input'], self._ledger, where=self._where(kwargs.get('name')))
        return DeduplicatingSpan(self._inner.start_observation(**kwargs), self._ledger, scope=self._scope)

    def update(self, **kwargs: Any) -> Any:
        if 'input' in kwargs:
            kwargs['input'] = deduplicate(kwargs['input'], self._ledger, where=self._where(kwargs.get('name')))
        return self._inner.update(**kwargs)


__all__ = [
    'DIGEST_SIZE',
    'DeduplicatingSpan',
    'MessageLedger',
    'deduplicate',
    'fingerprint',
]
