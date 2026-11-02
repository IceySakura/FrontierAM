"""Immutable collection types with helpful error messages.

These classes wrap dict and list to prevent modification, providing
clear error messages that guide users to the correct approach.
"""

from __future__ import annotations

from typing import Any


class ImmutableDict(dict):
    """A dict that raises an error when modified, with a helpful message.

    Nested dicts are also converted to ImmutableDict recursively.

    Args:
        *args: Positional arguments passed to dict constructor
        field_name: Name used in error messages to identify this dict
        error_message: Custom error message format (should contain {field_name} placeholder)
        **kwargs: Keyword arguments passed to dict constructor
    """

    _field_name: str
    _error_message: str

    _default_error_message: str = (
        '{field_name} is immutable. To update state, return the changes in ToolResponse.context_updates instead.'
    )

    def __new__(
        cls, *args: Any, field_name: str = 'field', error_message: str | None = None, **kwargs: Any
    ) -> ImmutableDict:
        instance = super().__new__(cls)
        # Store attributes via object.__setattr__ to avoid any custom __setattr__
        object.__setattr__(instance, '_field_name', field_name)
        object.__setattr__(instance, '_error_message', error_message or cls._default_error_message)
        return instance

    def __init__(self, *args: Any, field_name: str = 'field', error_message: str | None = None, **kwargs: Any) -> None:
        # First initialize as a regular dict
        temp = dict(*args, **kwargs)
        # Recursively convert nested dicts
        converted = {k: self._make_immutable(v, f'{self._field_name}[{k!r}]') for k, v in temp.items()}
        super().__init__(converted)

    def __reduce__(self) -> tuple:
        """Support pickle by serializing as a plain dict + metadata."""
        return (_rebuild_immutable_dict, (dict(self), self._field_name, self._error_message))

    def _make_immutable(self, value: Any, path: str) -> Any:
        """Recursively convert nested dicts to ImmutableDict."""
        if isinstance(value, dict) and not isinstance(value, ImmutableDict):
            return ImmutableDict(value, field_name=path, error_message=self._error_message)
        if isinstance(value, list):
            return ImmutableList(value, field_name=path, error_message=self._error_message)
        return value

    def _raise_immutable_error(self) -> None:
        raise TypeError(self._error_message.format(field_name=self._field_name))

    def __setitem__(self, key, value):
        self._raise_immutable_error()

    def __delitem__(self, key):
        self._raise_immutable_error()

    def clear(self):
        self._raise_immutable_error()

    def pop(self, *args):
        self._raise_immutable_error()

    def popitem(self):
        self._raise_immutable_error()

    def setdefault(self, *args):
        self._raise_immutable_error()

    def update(self, *args, **kwargs):
        self._raise_immutable_error()


class ImmutableList(list):
    """A list that raises an error when modified, with a helpful message.

    Nested dicts and lists are also converted recursively.

    Args:
        iterable: Initial items for the list
        field_name: Name used in error messages to identify this list
        error_message: Custom error message format (should contain {field_name} placeholder)
    """

    _field_name: str
    _error_message: str

    _default_error_message: str = (
        '{field_name} is immutable. To update state, return the changes in ToolResponse.context_updates instead.'
    )

    def __new__(cls, iterable: Any = (), field_name: str = 'field', error_message: str | None = None) -> ImmutableList:
        instance = super().__new__(cls)
        object.__setattr__(instance, '_field_name', field_name)
        object.__setattr__(instance, '_error_message', error_message or cls._default_error_message)
        return instance

    def __init__(self, iterable: Any = (), field_name: str = 'field', error_message: str | None = None) -> None:
        converted = [self._make_immutable(item, f'{self._field_name}[{i}]') for i, item in enumerate(iterable)]
        super().__init__(converted)

    def __reduce__(self) -> tuple:
        """Support pickle by serializing as a plain list + metadata."""
        return (_rebuild_immutable_list, (list(self), self._field_name, self._error_message))

    def _make_immutable(self, value: Any, path: str) -> Any:
        """Recursively convert nested dicts/lists to immutable versions."""
        if isinstance(value, dict) and not isinstance(value, ImmutableDict):
            return ImmutableDict(value, field_name=path, error_message=self._error_message)
        if isinstance(value, list) and not isinstance(value, ImmutableList):
            return ImmutableList(value, field_name=path, error_message=self._error_message)
        return value

    def _raise_immutable_error(self) -> None:
        raise TypeError(self._error_message.format(field_name=self._field_name))

    def __setitem__(self, index, value):
        self._raise_immutable_error()

    def __delitem__(self, index):
        self._raise_immutable_error()

    def append(self, value):
        self._raise_immutable_error()

    def extend(self, values):
        self._raise_immutable_error()

    def insert(self, index, value):
        self._raise_immutable_error()

    def remove(self, value):
        self._raise_immutable_error()

    def pop(self, *args):
        self._raise_immutable_error()

    def clear(self):
        self._raise_immutable_error()

    def sort(self, *args, **kwargs):
        self._raise_immutable_error()

    def reverse(self):
        self._raise_immutable_error()


def _rebuild_immutable_dict(data: dict, field_name: str, error_message: str) -> ImmutableDict:
    """Pickle reconstruction helper for ImmutableDict."""
    return ImmutableDict(data, field_name=field_name, error_message=error_message)


def _rebuild_immutable_list(data: list, field_name: str, error_message: str) -> ImmutableList:
    """Pickle reconstruction helper for ImmutableList."""
    return ImmutableList(data, field_name=field_name, error_message=error_message)


__all__ = ['ImmutableDict', 'ImmutableList']
