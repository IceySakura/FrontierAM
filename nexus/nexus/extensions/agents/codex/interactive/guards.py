"""Guards for caller-supplied CLI args and environment.

``extra_cli_args`` is appended after the flags this module owns, and most CLIs let
a later occurrence win. So a caller could otherwise redirect ``--remote`` at
another app-server, move ``-C`` to a different workspace, or change the sandbox
that protects unattended tool execution.

The same applies to ``-c key=value``: Codex's generic config override can reach
provider, sandbox, and approval settings that Nexus writes itself.

Both are rejected rather than sanitised: quietly dropping an argument the caller
asked for would be its own surprise.
"""

from __future__ import annotations

import tomllib

from typing import cast


class UnsafeCliArgumentError(ValueError):
    """Raised when extra_cli_args would override a Nexus-owned flag."""


class UnsafeEnvOverrideError(ValueError):
    """Raised when extra_env would override a Nexus-owned variable."""


# Flags whose value this module decides. Long forms cover ``--flag=value`` too.
RESERVED_FLAGS = frozenset({
    '-a',
    '--ask-for-approval',
    '--approve-for-me',
    '--remote',
    '--remote-auth-token-env',
    '-C',
    '--cd',
    '-s',
    '--sandbox',
    '--no-alt-screen',
    '--full-auto',
    '--dangerously-bypass-approvals-and-sandbox',
    '--dangerously-bypass-hook-trust',
    # Provider selection: the session points Codex at its own Responses proxy, and
    # either of these redirects the model elsewhere (or fails to start).
    '--oss',
    '--local-provider',
})

# Short flags clap also accepts concatenated with their value: ``-anever`` is
# ``-a never``, ``-C/tmp`` is ``-C /tmp``, ``-ck=v`` is ``-c k=v``. Matching only
# the exact token would let every one of those through.
RESERVED_SHORT_PREFIXES = ('-a', '-C', '-c', '-s')

# Config keys that must come from Nexus's own config.toml, never from -c.
RESERVED_CONFIG_PREFIXES = (
    'model_provider',
    'model_providers',
    'approval_policy',
    'approvals_reviewer',
    'sandbox_mode',
    'sandbox_permissions',
    'sandbox_workspace_write',
    'features.hooks',
    'features.default_mode_request_user_input',
    'hooks',
    'allow_managed_hooks_only',
    'shell_environment_policy',
    'experimental_use_freeform_apply_patch',
)

# Environment this module owns; re-applied after a caller's extra_env is merged.
# PATH is deliberately absent: whether callers may set it is an existing
# repo-wide contract for installed agents, not something to change here.
PROTECTED_ENV_KEYS = ('CODEX_HOME', 'NEXUS_AGENT_PROXY_URL', 'NEXUS_PROXY_API_KEY')
PROTECTED_FEATURES = frozenset({'hooks', 'default_mode_request_user_input'})


def _flag_name(token: str) -> str:
    """``--remote=x`` -> ``--remote``; leaves non-flags untouched."""
    return token.split('=', 1)[0] if token.startswith('-') else token


def _reject_reserved_config_key(raw: str) -> None:
    """Raise if ``raw`` (a ``key=value`` string) targets a reserved config key."""
    key = raw.split('=', 1)[0].strip().strip('"').strip("'")
    lowered = key.casefold()
    for reserved in RESERVED_CONFIG_PREFIXES:
        if lowered == reserved or lowered.startswith(f'{reserved}.'):
            raise UnsafeCliArgumentError(
                f'may not override the {key!r} config key: it would bypass the approval, '
                'sandbox, provider, or hook settings this session configures.'
            )


def _reject_protected_feature(raw: str) -> None:
    feature = raw.strip().casefold().removeprefix('features.')
    if feature in PROTECTED_FEATURES:
        raise UnsafeCliArgumentError(f'extra_cli_args may not change the {raw!r} feature: codex_interactive owns it.')


def validate_extra_cli_args(args: list[str] | None) -> list[str]:
    """Return *args* unchanged, or raise if they touch anything Nexus owns.

    Covers three shapes a naive exact-match check misses:
    ``--flag=value``, concatenated short flags (``-anever``), and bare
    positionals -- Codex treats a leftover positional as the startup PROMPT, so
    one slipping through would silently inject a task.
    """
    tokens = list(args or [])
    value_for: str | None = None
    for index, token in enumerate(tokens):
        if value_for is not None:
            # Consumed as the previous flag's value; not a positional.
            if value_for in ('--enable', '--disable'):
                _reject_protected_feature(token)
            value_for = None
            continue

        name = _flag_name(token)
        if name in RESERVED_FLAGS:
            raise UnsafeCliArgumentError(
                f'extra_cli_args may not set {name!r}: codex_interactive owns it '
                '(approval policy, sandbox, relay endpoint, workspace, and screen mode are '
                'fixed by the session).'
            )

        if token.startswith('-') and not token.startswith('--'):
            # Short flag, possibly with its value attached.
            for prefix in RESERVED_SHORT_PREFIXES:
                if token.startswith(prefix):
                    if token == prefix:
                        break  # exact form: handled above / value follows
                    raise UnsafeCliArgumentError(
                        f'extra_cli_args may not set {token!r}: {prefix!r} is owned by the session, '
                        'and a concatenated short flag sets it just as the separated form does.'
                    )

        if name in ('-c', '--config'):
            # The value may be attached (-c k=v, -ck=v, --config=k=v) or follow.
            if '=' in token:
                _reject_reserved_config_override(token.split('=', 1)[1])
            elif token not in ('-c', '--config'):
                _reject_reserved_config_override(token[len('-c') :])
            elif index + 1 < len(tokens):
                _reject_reserved_config_override(tokens[index + 1])
                value_for = name
            continue

        if name in ('--enable', '--disable') and '=' in token:
            _reject_protected_feature(token.split('=', 1)[1])

        if not token.startswith('-'):
            raise UnsafeCliArgumentError(
                f'extra_cli_args may not contain the bare argument {token!r}: Codex would take a '
                'leftover positional as the startup prompt, injecting a task the caller never sent. '
                'Pass values attached to their flag (e.g. ["-s", "read-only"]).'
            )
        if token in _VALUE_TAKING_FLAGS:
            value_for = token
    return tokens


# Long flags that consume the following token as their value, so that token is
# not a stray positional. Anything not listed must attach its value with '='.
_VALUE_TAKING_FLAGS = frozenset({
    '-m',
    '--model',
    '-p',
    '--profile',
    '-i',
    '--image',
    '--add-dir',
    '--enable',
    '--disable',
})


def _reject_reserved_keys_in_table(table: dict[str, object], path: tuple[str, ...] = ()) -> None:
    """Walk a parsed TOML table and reject every reserved key path in it.

    Recursion is what makes the check sound: a reserved key can sit at any depth,
    including inside an inline table, where looking only at the outermost name
    sees ``features`` and misses ``features.default_mode_request_user_input``.
    """
    for key, value in table.items():
        current = (*path, key)
        checked = current[2:] if len(current) > 2 and current[0].casefold() == 'profiles' else current
        if checked:
            _reject_reserved_config_key('.'.join(checked))
        if isinstance(value, dict):
            _reject_reserved_keys_in_table(cast(dict[str, object], value), current)


def _reject_reserved_config_override(raw: str) -> None:
    """Reject one ``-c``/``--config`` value that targets a reserved key.

    Codex accepts a TOML value on the right-hand side, so the assignment has to be
    parsed rather than split: ``features={default_mode_request_user_input=false}``
    names no reserved key before the ``=`` yet sets one. Unparseable input is
    rejected instead of waved through -- a value this module cannot read is a value
    it cannot vouch for.
    """
    text = raw.strip()
    # A bare inline table (``{sandbox_mode="x"}``) is not a TOML document, so it
    # would fall through to the left-hand-side check and match nothing. Wrap it so
    # its keys are seen; the synthetic root name is dropped by the walk below.
    candidates = [text, f'_ = {text}'] if text.startswith('{') else [text]
    for candidate in candidates:
        try:
            parsed = tomllib.loads(candidate)
        except tomllib.TOMLDecodeError:
            continue
        root = parsed.get('_') if candidate is not text else parsed
        if isinstance(root, dict):
            _reject_reserved_keys_in_table(cast(dict[str, object], root))
        return
    # Not parseable in any form (e.g. a bare key name, or a value Codex accepts but
    # tomllib does not). Fall back to the left-hand side, which still catches the
    # dotted forms.
    _reject_reserved_config_key(text)


def validate_extra_config_toml(extra_config_toml: str | None) -> None:
    """Reject inline TOML that sets a reserved key.

    ``extra_config_toml`` is appended to the generated config.toml, so a later
    definition wins there just as a later CLI flag does.
    """
    if not extra_config_toml:
        return
    try:
        config = tomllib.loads(extra_config_toml)
    except tomllib.TOMLDecodeError as error:
        raise UnsafeCliArgumentError(f'extra_config_toml is invalid TOML: {error}') from error

    _reject_reserved_keys_in_table(config)


def apply_protected_env(env: dict[str, str], owned: dict[str, str]) -> dict[str, str]:
    """Merge *owned* over *env* so caller values cannot win.

    Called after ``extra_env`` has been merged in, which is what makes the
    protection effective: ordering here is the whole point.
    """
    merged = dict(env)
    for key in PROTECTED_ENV_KEYS:
        if key in owned:
            merged[key] = owned[key]
    return merged


def assert_env_not_overriding(extra_env: dict[str, str] | None) -> None:
    """Reject an ``extra_env`` that tries to set a protected variable.

    ``apply_protected_env`` already makes such an attempt ineffective; failing
    loudly instead tells the caller its setting was never going to apply.
    """
    for key in PROTECTED_ENV_KEYS:
        if extra_env and key in extra_env:
            raise UnsafeEnvOverrideError(
                f'extra_env may not set {key!r}: it is owned by the session '
                '(Codex home isolation and proxy routing depend on it).'
            )


__all__ = [
    'PROTECTED_ENV_KEYS',
    'RESERVED_CONFIG_PREFIXES',
    'RESERVED_FLAGS',
    'UnsafeCliArgumentError',
    'UnsafeEnvOverrideError',
    'apply_protected_env',
    'assert_env_not_overriding',
    'validate_extra_cli_args',
]
