"""Unit tests for the ``artifact:///`` URI parser."""

from __future__ import annotations

import pytest

from nexus.utils.artifact.errors import ArtifactPathError
from nexus.utils.artifact.uri import (
    SELF_OWNER,
    format_aggregated_shared_uri,
    format_artifact_uri,
    format_self_uri,
    parse_artifact_uri,
)


def test_bare_path_has_no_owner_and_no_scheme():
    parsed = parse_artifact_uri('/models/v1/model.bin')
    assert parsed.owner is None
    assert parsed.is_aggregated_shared_view is False
    assert parsed.is_uri is False
    assert parsed.segments == ('models', 'v1', 'model.bin')


def test_bare_path_without_leading_slash_still_splits():
    parsed = parse_artifact_uri('models/model.bin')
    assert parsed.owner is None
    assert parsed.segments == ('models', 'model.bin')


def test_uri_with_owner_extracts_first_segment_as_owner():
    """The three-slash form treats the first path segment as owner."""
    parsed = parse_artifact_uri('artifact:///alice/models/model.bin')
    assert parsed.owner == 'alice'
    assert parsed.is_aggregated_shared_view is False
    assert parsed.is_uri is True
    # Segments below the owner are what the client walks against list_resources.
    assert parsed.segments == ('models', 'model.bin')


def test_uri_with_owner_and_no_path_is_top_level_reference():
    """``artifact:///<owner>/`` denotes the owner's top level (no segments)."""
    parsed = parse_artifact_uri('artifact:///alice/')
    assert parsed.owner == 'alice'
    assert parsed.segments == ()


def test_uri_with_owner_no_trailing_slash_is_also_top_level():
    """``artifact:///<owner>`` (no slash) has no path -- treat as the top level."""
    parsed = parse_artifact_uri('artifact:///alice')
    assert parsed.owner == 'alice'
    assert parsed.segments == ()


def test_bare_three_slashes_is_the_aggregated_shared_view():
    """``artifact:///`` (empty path) marks the URI as the aggregated shared listing."""
    parsed = parse_artifact_uri('artifact:///')
    assert parsed.is_aggregated_shared_view is True
    assert parsed.owner is None
    assert parsed.segments == ()


def test_legacy_two_slash_form_rejected_with_migration_hint():
    """The old``artifact://<owner>/`` form is refused with a nudge toward three slashes."""
    with pytest.raises(ArtifactPathError, match='did you mean'):
        parse_artifact_uri('artifact://alice/models/model.bin')


def test_legacy_shared_alias_rejected_with_hint_about_removal():
    """``artifact://@shared/`` used to be the alias; make its removal discoverable."""
    with pytest.raises(ArtifactPathError, match='legacy .* alias has been removed'):
        parse_artifact_uri('artifact://@shared/')


def test_invalid_owner_login_rejected():
    """First path segment must look like a Taihu login."""
    with pytest.raises(ArtifactPathError, match='invalid owner'):
        parse_artifact_uri('artifact:///not a login/foo')


def test_empty_owner_rejected():
    """Four slashes (empty first segment) is not a valid owner reference."""
    with pytest.raises(ArtifactPathError, match='missing an owner'):
        parse_artifact_uri('artifact:////models')


def test_nfc_normalisation_applied_to_segments():
    """Path segments are NFC-normalised so equivalent code points compare equal."""
    # ``é`` composed vs. ``e`` + combining acute; both must produce the same
    # NFC form so downstream name comparisons don't split hairs.
    composed = parse_artifact_uri('artifact:///alice/caf\u00e9/file')
    decomposed = parse_artifact_uri('artifact:///alice/cafe\u0301/file')
    assert composed.segments == decomposed.segments


def test_format_uri_round_trips_a_parsed_uri():
    parsed = parse_artifact_uri('artifact:///alice/models/model.bin')
    assert parsed.owner is not None
    assert format_artifact_uri(parsed.owner, *parsed.segments) == 'artifact:///alice/models/model.bin'


def test_format_uri_with_trailing_slash_marks_folder():
    """A trailing slash on the formatted URI marks the referent as a folder."""
    assert format_artifact_uri('alice', 'reports', trailing_slash=True) == 'artifact:///alice/reports/'


def test_format_uri_top_level_still_ends_with_slash():
    """``format_artifact_uri('alice', trailing_slash=True)`` yields the owner's root."""
    assert format_artifact_uri('alice', trailing_slash=True) == 'artifact:///alice/'


def test_format_uri_rejects_bad_segments():
    with pytest.raises(ValueError):
        format_artifact_uri('alice', 'has/slash')
    with pytest.raises(ValueError):
        format_artifact_uri('alice', '')


def test_format_aggregated_shared_uri_is_the_canonical_literal():
    """The helper hides the literal so future grammar tweaks stay in one place."""
    assert format_aggregated_shared_uri() == 'artifact:///'
    # And the parser must round-trip it as the aggregated view.
    parsed = parse_artifact_uri(format_aggregated_shared_uri())
    assert parsed.is_aggregated_shared_view is True


# ---------------------------------------------------------------------------
# artifact:///~ self-reference
# ---------------------------------------------------------------------------


def test_self_uri_top_level_parses_correctly():
    """artifact:///~ parses with owner=SELF_OWNER and no segments."""
    parsed = parse_artifact_uri('artifact:///~')
    assert parsed.owner == SELF_OWNER
    assert parsed.is_self_view is True
    assert parsed.is_aggregated_shared_view is False
    assert parsed.is_uri is True
    assert parsed.segments == ()


def test_self_uri_with_trailing_slash_is_same_as_without():
    """artifact:///~/ and artifact:///~ are equivalent (trailing slash stripped)."""
    parsed = parse_artifact_uri('artifact:///~/')
    assert parsed.owner == SELF_OWNER
    assert parsed.segments == ()


def test_self_uri_with_subpath_captures_segments():
    """artifact:///~/models/v1 has owner=~ and two path segments."""
    parsed = parse_artifact_uri('artifact:///~/models/v1')
    assert parsed.owner == SELF_OWNER
    assert parsed.is_self_view is True
    assert parsed.segments == ('models', 'v1')


def test_self_uri_is_uri_flag():
    assert parse_artifact_uri('artifact:///~').is_uri is True


# ---------------------------------------------------------------------------
# artifact://~ shorthand (two-slash exception for the self-reference URI)
# ---------------------------------------------------------------------------


def test_self_uri_shorthand_top_level_parses_correctly():
    """artifact://~ (two slashes) is accepted as an alias for artifact:///~."""
    parsed = parse_artifact_uri('artifact://~')
    assert parsed.owner == SELF_OWNER
    assert parsed.is_self_view is True
    assert parsed.is_aggregated_shared_view is False
    assert parsed.is_uri is True
    assert parsed.segments == ()


def test_self_uri_shorthand_trailing_slash_is_same_as_without():
    """artifact://~/ and artifact://~ are equivalent."""
    parsed = parse_artifact_uri('artifact://~/')
    assert parsed.owner == SELF_OWNER
    assert parsed.segments == ()


def test_self_uri_shorthand_with_subpath_captures_segments():
    """artifact://~/models/v1 must behave exactly like artifact:///~/models/v1."""
    parsed = parse_artifact_uri('artifact://~/models/v1')
    assert parsed.owner == SELF_OWNER
    assert parsed.is_self_view is True
    assert parsed.segments == ('models', 'v1')


def test_self_uri_shorthand_rejects_tilde_glued_to_extra_chars():
    """artifact://~foo (no path separator after ~) must be rejected clearly."""
    with pytest.raises(ArtifactPathError, match='self-reference must be followed by'):
        parse_artifact_uri('artifact://~foo')


def test_self_uri_shorthand_only_applies_to_tilde_owner():
    """Two-slash form is still rejected for regular logins (e.g. artifact://alice/...)."""
    with pytest.raises(ArtifactPathError, match='did you mean'):
        parse_artifact_uri('artifact://alice/models/model.bin')


def test_format_self_uri_no_segments():
    assert format_self_uri() == 'artifact:///~'


def test_format_self_uri_with_trailing_slash():
    assert format_self_uri(trailing_slash=True) == 'artifact:///~/'


def test_format_self_uri_with_segments():
    assert format_self_uri('models', 'v1') == 'artifact:///~/models/v1'


def test_format_artifact_uri_accepts_tilde_owner():
    """format_artifact_uri must accept SELF_OWNER ('~') as a valid owner."""
    uri = format_artifact_uri(SELF_OWNER, 'tasks', '2026')
    assert uri == 'artifact:///~/tasks/2026'


def test_self_uri_does_not_match_regular_owner_regex():
    """Make sure '~' is explicitly allowed rather than accidentally matching _OWNER_RE."""
    import re

    _OWNER_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9._-]{0,62}$')
    assert not _OWNER_RE.match('~'), "'~' must NOT match the employee-login regex"
    # …yet parse_artifact_uri accepts it.
    parsed = parse_artifact_uri('artifact:///~')
    assert parsed.owner == '~'


def test_self_view_is_false_for_real_owner():
    parsed = parse_artifact_uri('artifact:///alice/models')
    assert parsed.is_self_view is False


def test_self_view_is_false_for_aggregated_view():
    parsed = parse_artifact_uri('artifact:///')
    assert parsed.is_self_view is False


def test_self_view_is_false_for_bare_path():
    parsed = parse_artifact_uri('/models/report.pdf')
    assert parsed.is_self_view is False
