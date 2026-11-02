"""Parser and formatter for the ``artifact:///`` URI scheme.

The artifact service addresses every resource by an opaque id. On the wire,
paths like ``/models/v1/model.bin`` are already used by the CLI and SDK; this
module extends that model with a fully-qualified URI form so that same-named
resources shared by different employees remain distinguishable at the
addressing layer -- and so that a URI, once printed, is a self-contained
reference that any caller can paste into a subsequent command.

The grammar is intentionally tiny::

    artifact-uri  := "artifact:///" path?
    path          := owner ("/" segment)*
    owner         := <employee login, matches _OWNER_RE below>
                | "~"   (self-reference sentinel, resolved at runtime)

Note the **three** slashes after the scheme. ``artifact:`` has an empty
authority (there is no network host to talk to; the "owner" is a logical
namespace, not a DNS name), which follows the same convention as
``file:///path``. Owner therefore lives as the *first path segment*, not as
the authority.

Semantics (interpreted by ``ArtifactClient``, not by the server):

* ``artifact:///`` (empty path): aggregated shared-with-me view across all
  owners. Only meaningful for *listing* commands; any attempt to descend
  into it is rejected because ``artifact:///`` is not a real folder --
  children come from many different owners and must each be reached via
  their own owner-scoped URI.
* ``artifact:///~[/<path...>]``: self-reference. ``~`` is resolved to the
  caller's own login (via ``GET /v1/whoami``) at execution time, making
  ``artifact:///~`` equivalent to ``artifact:///<caller>`` without knowing
  the login in advance. Useful in scripts that must not hardcode usernames.
  The shorthand ``artifact://~`` / ``artifact://~/<path>`` (**two** slashes)
  is accepted as an alias — this is an intentional exception to the
  three-slash rule so users can save a keystroke on the very common
  "myself" URI; every other two-slash owner form remains rejected.
* ``artifact:///<owner>``: the owner's top level (equivalent to a trailing
  ``/`` and empty tail).
* ``artifact:///<owner>/<path...>``: navigate to that resource. If ``owner``
  is the caller (per ``GET /v1/whoami``), the top-level walk uses the owned
  view; otherwise it uses the shared view filtered by ``owner_id ==
  <owner>``. The rest of the walk is by folder id.
* Bare paths (``/tasks/xxx`` or ``tasks/xxx``) remain valid and mean "the
  caller's own resource at that path".

Legacy ``artifact://<owner>/<path>`` (two slashes) is rejected with a hint,
including the earlier ``artifact://@shared/`` alias which no longer exists.

``format_artifact_uri`` is the inverse used by the CLI when it needs to print
a canonical URI back to the user.
"""

from __future__ import annotations

import re
import unicodedata

from dataclasses import dataclass

from nexus.utils.artifact.errors import ArtifactPathError


SCHEME = 'artifact:'
# Canonical prefix. All artifact URIs start with exactly three slashes after
# the scheme (empty authority, then a path that starts with ``/``).
_URI_PREFIX = 'artifact:///'
# Shorthand for the self-reference URI. ``artifact://~`` (two slashes) is an
# intentional exception to the three-slash rule: it lets scripts write the
# very common "myself" URI a keystroke shorter. Every other two-slash form
# (e.g. ``artifact://alice/x``) remains rejected with a migration hint.
_SELF_URI_SHORTHAND = 'artifact://~'

# Employee login shape. Taihu's ``LoginName`` in practice is an OA account
# (letters/digits/dot/dash/underscore, must start with a letter). We keep the
# validation deliberately loose so it does not need to track upstream policy
# changes, but strict enough to reject obviously malformed owners such as
# empty strings or characters that would break URI parsing.
_OWNER_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9._-]{0,62}$')

# Self-reference sentinel.``artifact:///~`` (and ``artifact:///~/<path>``)
# is resolved to the caller's own login at command-execution time via
# ``GET /v1/whoami``.  Using ``~`` avoids hardcoding the login in scripts
# while still composing full URIs that are unambiguous.
SELF_OWNER = '~'


@dataclass(frozen=True)
class ParsedUri:
    """Result of :func:`parse_artifact_uri`.

    Attributes:
        is_aggregated_shared_view: ``True`` iff the URI is bare
            ``artifact:///``. Callers that only support single-resource
            addressing (download, info, ...) should reject such URIs.
        is_self_view: ``True`` iff the owner segment is the ``~``
            self-reference sentinel (``artifact:///~[/<path>]``).
            Callers must resolve ``~`` to the actual login via
            ``whoami()`` before handing the path to the server.
        owner: The URI's owner login, or ``None`` for a bare-path URI whose
            owner is implicitly the caller (and also for the aggregated
            ``artifact:///`` view, where the "owner" spans many employees).
            Equals :data:`SELF_OWNER` (``'~'``) for the self-reference form.
        segments: NFC-normalized path segments below the owner (already
            ``str``; may be empty for a ``artifact:///<owner>`` top-level
            reference).
    """

    is_aggregated_shared_view: bool
    owner: str | None
    segments: tuple[str, ...]

    @property
    def is_self_view(self) -> bool:
        """Whether the URI uses the ``artifact:///~`` self-reference sentinel."""
        return self.owner == SELF_OWNER

    @property
    def is_uri(self) -> bool:
        """Whether the source string used the ``artifact:///`` scheme."""
        return self.is_aggregated_shared_view or self.owner is not None


def parse_artifact_uri(text: str) -> ParsedUri:
    """Parse either an ``artifact:///`` URI or a bare slash-path.

    Bare paths are equivalent to ``artifact:///<caller>/<path>``: the
    caller's identity is resolved by :meth:`ArtifactClient.whoami` at
    resolution time, so we do not need it here.

    The special form ``artifact:///~[/<path>]`` uses ``~`` as a
    self-reference placeholder; callers must substitute it with the actual
    login before issuing server requests.

    Raises:
        ArtifactPathError: If the URI is malformed, uses an invalid owner
            login, uses the legacy two-slash form, or attempts to descend
            into the aggregated ``artifact:///`` view.
    """

    if text is None:
        raise ArtifactPathError('artifact URI must not be None', path=text)
    raw = str(text)

    if not raw.startswith(SCHEME):
        # Bare paths keep their previous meaning: "the caller's own resource".
        # We still validate them here so callers only have to parse once.
        segments = _split_and_normalize(raw)
        return ParsedUri(is_aggregated_shared_view=False, owner=None, segments=segments)

    # Shorthand: ``artifact://~`` (two slashes) and ``artifact://~/<path>``
    # are accepted as aliases for the three-slash canonical form. This is the
    # single exception to the three-slash rule (see module docstring); any
    # other two-slash form falls through to the migration-hint branch below.
    if raw.startswith(_SELF_URI_SHORTHAND):
        trailer = raw[len(_SELF_URI_SHORTHAND) :]
        if trailer in ('', '/'):
            return ParsedUri(is_aggregated_shared_view=False, owner=SELF_OWNER, segments=())
        if trailer.startswith('/'):
            segments = _split_and_normalize(trailer[1:])
            return ParsedUri(is_aggregated_shared_view=False, owner=SELF_OWNER, segments=segments)
        # ``artifact://~foo`` -- ``~`` must be terminated by ``/`` or end of URI.
        raise ArtifactPathError(
            f'artifact URI {raw!r}: the `~` self-reference must be followed by `/` or end of URI',
            path=raw,
        )

    # Anything using the ``artifact:`` scheme must have the three-slash
    # prefix. Reject earlier two-slash forms (which used to put owner in the
    # authority segment) with a clear migration hint so users are not left
    # guessing what changed.
    if not raw.startswith(_URI_PREFIX):
        remainder = raw[len(SCHEME) :]
        hint = ''
        if remainder.startswith('//'):
            # Legacy artifact://<owner>/<path> or artifact://@shared/.
            legacy = remainder[2:]
            if legacy.startswith('@shared'):
                hint = ' (the legacy `artifact://@shared/` alias has been removed; use `artifact:///`)'
            elif legacy:
                # Migrate two-slash form to three-slash for the hint.
                hint = f' (did you mean `artifact:///{legacy}`?)'
        raise ArtifactPathError(
            f'artifact URI {raw!r} must start with `artifact:///` (three slashes){hint}',
            path=raw,
        )

    path = raw[len(_URI_PREFIX) :]

    if not path:
        # ``artifact:///`` -- aggregated shared-with-me view.
        return ParsedUri(is_aggregated_shared_view=True, owner=None, segments=())

    # First path segment is the owner; the rest is the resource path.
    owner, _, rest = path.partition('/')
    if not owner:
        # Handles pathological input like ``artifact:////foo`` where the
        # leading slash makes the owner segment empty.
        raise ArtifactPathError(
            f'artifact URI {raw!r} is missing an owner. Use `artifact:///<owner>/<path>` '
            'or bare `artifact:///` for the aggregated shared-with-me listing.',
            path=raw,
        )

    # ``~`` is the self-reference sentinel: accepted as-is, resolved to the
    # caller's login by the command layer.
    if owner == SELF_OWNER:
        segments = _split_and_normalize(rest)
        return ParsedUri(is_aggregated_shared_view=False, owner=SELF_OWNER, segments=segments)

    if not _OWNER_RE.match(owner):
        raise ArtifactPathError(
            f'artifact URI {raw!r} has an invalid owner {owner!r}',
            path=raw,
        )

    segments = _split_and_normalize(rest)
    return ParsedUri(is_aggregated_shared_view=False, owner=owner, segments=segments)


def format_artifact_uri(owner: str, *segments: str, trailing_slash: bool = False) -> str:
    """Build a canonical ``artifact:///<owner>/<segments...>`` URI.

    ``owner`` may be :data:`SELF_OWNER` (``'~'``) to produce a self-reference
    URI that callers can resolve to their own login at runtime.

    Only used by the CLI to print stable references in listings, so this is
    intentionally strict: an invalid owner or empty-string segment is
    programmer error. ``trailing_slash`` appends a ``/`` after the last
    segment (or right after the owner when there are no segments); pass it
    when the URI names a folder so callers can spot folders at a glance.
    """

    if owner != SELF_OWNER and not _OWNER_RE.match(owner):
        raise ValueError(f'invalid artifact URI owner: {owner!r}')
    normalized = [unicodedata.normalize('NFC', segment) for segment in segments]
    for segment in normalized:
        if not segment or '/' in segment:
            raise ValueError(f'invalid artifact URI segment: {segment!r}')
    path = '/'.join([owner, *normalized])
    if trailing_slash:
        path = f'{path}/'
    return f'{_URI_PREFIX}{path}'


def format_self_uri(*segments: str, trailing_slash: bool = False) -> str:
    """Return a canonical ``artifact:///~[/<segments>]`` self-reference URI.

    Equivalent to ``format_artifact_uri(SELF_OWNER, *segments, ...)``.
    Kept as a named helper so call-sites read more clearly and the
    ``~`` literal is not scattered through the codebase.
    """
    return format_artifact_uri(SELF_OWNER, *segments, trailing_slash=trailing_slash)


def format_aggregated_shared_uri() -> str:
    """Return the canonical URI for the aggregated shared-with-me view.

    Kept as a helper so callers do not hardcode the literal, and so future
    grammar tweaks stay in one place.
    """
    return _URI_PREFIX


def _split_and_normalize(path: str) -> tuple[str, ...]:
    return tuple(unicodedata.normalize('NFC', segment) for segment in path.split('/') if segment)
