"""Tests for nexus.utils.common helpers."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from pydantic import BaseModel

from nexus.utils.common import coerce_bool, pformat, pformat_for_display


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (True, True),
        (False, False),
        (None, False),
        ('true', True),
        ('TRUE', True),
        (' Yes ', True),
        ('1', True),
        ('on', True),
        ('false', False),
        ('no', False),
        ('0', False),
        ('off', False),
        ('', False),
    ],
)
def test_coerce_bool_accepts_known_forms(value, expected):
    assert coerce_bool(value, name='flag') is expected


@pytest.mark.parametrize('value', ['sometimes', 'maybe', 'yo', 2, 1.5, [], {}])
def test_coerce_bool_rejects_ambiguous(value):
    with pytest.raises(ValueError, match='flag must be a boolean'):
        coerce_bool(value, name='flag')


# ---------------------------------------------------------------------------
# pformat_for_display: __pformat_exclude__ convention
# ---------------------------------------------------------------------------


class _NoisyOutput(BaseModel):
    """A tiny stand-in for a task output that hides one field from display."""

    keep_me: str = 'hello'
    hide_me: str = 'internal-handle'

    __pformat_exclude__: ClassVar[frozenset[str]] = frozenset({'hide_me'})


def test_pformat_for_display_drops_excluded_fields():
    """Fields listed in ``__pformat_exclude__`` disappear from the display string."""
    out = _NoisyOutput()
    rendered = pformat_for_display(out)
    parsed = json.loads(rendered)
    # ``hide_me`` is silenced ...
    assert 'hide_me' not in parsed
    # ... but the model itself and its serialisation are untouched.
    assert out.hide_me == 'internal-handle'
    assert 'hide_me' in out.model_dump()
    # And other fields remain visible in the display output.
    assert parsed['keep_me'] == 'hello'


def test_pformat_for_display_matches_pformat_when_no_convention_declared():
    """Regular pydantic models without ``__pformat_exclude__`` render identically."""

    class _Plain(BaseModel):
        a: int = 1
        b: str = 'x'

    obj = _Plain()
    assert pformat_for_display(obj) == pformat(obj)


def test_pformat_for_display_leaves_non_pydantic_objects_alone():
    """Dicts / lists / primitives fall through to ``pformat``."""
    payload = {'artifact_id': 'art_xyz', 'artifact_uri': 'artifact:///a/b'}
    # Dicts don't carry ``__pformat_exclude__``, so both keys survive.
    parsed = json.loads(pformat_for_display(payload))
    assert parsed == payload


def test_pformat_exclude_convention_hides_artifact_id_from_display():
    """Advanced case: task output uses ``__pformat_exclude__`` to hide ``artifact_id``.

    The common case for artifact-producing tasks is to declare only
    ``artifact_uri`` on the subclass. Rarely, a platform layer may also
    persist ``artifact_id``; when it does, ``__pformat_exclude__`` (a generic
    mechanism, not artifact-specific) keeps the noisy internal handle out of
    the terminal log while ``model_dump`` and ``--output`` JSON still carry it.
    """
    from typing import ClassVar

    from nexus.core.task import TaskRunOutput

    class _Out(TaskRunOutput):
        artifact_uri: str | None = None
        artifact_id: str | None = None
        filename: str = 'r.md'

        __pformat_exclude__: ClassVar[frozenset[str]] = frozenset({'artifact_id'})

    out = _Out(artifact_id='art_01KZ', artifact_uri='artifact:///alice/r.md')
    parsed = json.loads(pformat_for_display(out))
    # ``artifact_id`` is hidden from the display ...
    assert 'artifact_id' not in parsed
    # ... but is still present on the model and in full serialisation, so
    # platform-layer consumers (download URL regeneration, persistence) can
    # still read it.
    assert out.artifact_id == 'art_01KZ'
    dumped = out.model_dump()
    assert dumped['artifact_id'] == 'art_01KZ'
    # ``artifact_uri`` remains visible everywhere -- that's the human-readable
    # reference we actively want the user to see and copy-paste.
    assert parsed['artifact_uri'] == 'artifact:///alice/r.md'
