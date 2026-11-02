"""Read typed values out of untyped session config.

Session config reaches a manager as ``Mapping[str, object]`` -- from a recipe
literal, a generator, or environment interpolation -- so every read needs a type
check and a fallback. These return the fallback for anything unusable rather
than raising: a config typo should not abort a run mid-turn.

``strict_int`` and friends do not exist here on purpose. Validation belongs where
the value enters the process: a generator reads the environment through
``utils.generator_util``, whose ``_env_int`` / ``_env_bool`` / ``_env_choice``
refuse a malformed value before a sandbox is started.
"""

from collections.abc import Iterable, Mapping


def optional_dict(value: object) -> dict[str, object] | None:
    """``value`` as a string-keyed dict, or None if it is not a mapping."""
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return None


def optional_string(value: object) -> str | None:
    """``value`` if it is a string, else None."""
    return value if isinstance(value, str) else None


def optional_bool(value: object, *, default: bool) -> bool:
    """``value`` if it is a bool, else ``default``.

    Only a real bool counts; ``'true'`` and ``1`` are not accepted, so a config
    that means to switch something on has to say so unambiguously.
    """
    return value if isinstance(value, bool) else default


def optional_int(value: object) -> int | None:
    """``value`` as an int, accepting a decimal string, else None.

    Bools are rejected even though ``bool`` is a subclass of ``int``: ``True``
    arriving where a count belongs is a config error, not the number 1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def optional_float(value: object, *, default: float) -> float:
    """``value`` as a float, else ``default``. Bools rejected as in :func:`optional_int`."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def optional_str_list(value: object) -> list[str] | None:
    """``value`` as a list of strings, or None if it is not a list."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def string_map(value: object) -> dict[str, str]:
    """``value`` as a flat string-to-string dict; empty for anything else."""
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    return {}


def pick(config: Mapping[str, object], keys: Iterable[str]) -> dict[str, object]:
    """The entries of ``config`` whose key is in ``keys`` and whose value is set.

    Used to forward a whitelisted subset of session config to a sandbox, where
    an unexpected key would be passed to the CLI verbatim.
    """
    return {key: config[key] for key in keys if config.get(key) is not None}


__all__ = [
    'optional_bool',
    'optional_dict',
    'optional_float',
    'optional_int',
    'optional_str_list',
    'optional_string',
    'pick',
    'string_map',
]
