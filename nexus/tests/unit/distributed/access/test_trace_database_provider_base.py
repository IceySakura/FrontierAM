"""The backend-agnostic parts of the trace database providers.

These used to be three copies of the same few lines, one per provider, free to drift apart
on the message an operator reads when a request is routed to the wrong backend.
"""

from __future__ import annotations

import pytest

from nexus.distributed.access.trace_database_providers.base import (
    TraceDatabaseTarget,
    validate_account_name,
    validate_database_name,
    validate_table_name,
)


_PROVIDERS = ['mysql', 'tencent_cloud_dcdb', 'tencent_cloud_tdmysql']


class _Pinned(TraceDatabaseTarget):
    def __init__(self, database: str) -> None:
        self.database = database


def test_the_target_guard_accepts_the_database_it_was_pinned_to() -> None:
    _Pinned('runner')._require_database('runner')


def test_the_target_guard_rejects_another_database() -> None:
    """A different name means the request reached the wrong backend, not a wrong spelling."""
    with pytest.raises(ValueError, match='target mismatch'):
        _Pinned('runner')._require_database('other')


def test_the_target_guard_rejects_an_unusable_database_name() -> None:
    with pytest.raises(ValueError, match='invalid trace database name'):
        _Pinned('runner')._require_database('runner; DROP DATABASE runner')


@pytest.mark.parametrize('module_name', _PROVIDERS)
def test_every_provider_inherits_the_shared_target_guard(module_name: str) -> None:
    """Inheriting it is what keeps the three providers from re-deriving the check.

    ``TraceDatabaseAccountProvider`` is a ``Protocol`` and carries no implementation, so a
    provider naming only that base would silently need its own copy again.
    """
    import importlib

    module = importlib.import_module(f'nexus.distributed.access.trace_database_providers.{module_name}')
    providers = [getattr(module, name) for name in module.__all__]

    assert providers, f'{module_name} exposes no provider'
    for provider in providers:
        assert issubclass(provider, TraceDatabaseTarget)
        assert '_require_database' not in vars(provider), 'the guard is inherited, not restated'


@pytest.mark.parametrize(
    ('validate', 'value', 'message'),
    [
        (validate_account_name, 'root', 'invalid trace database username'),
        (validate_database_name, 'runner`', 'invalid trace database name'),
        (validate_table_name, 'events; DROP TABLE runs', 'invalid trace database table name'),
    ],
)
def test_identifier_validation_rejects_names_no_writer_produces(validate, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate(value)


def test_validate_table_name_is_exported() -> None:
    """All three providers import it, so leaving it out of __all__ misreported the contract."""
    from nexus.distributed.access.trace_database_providers import base

    assert 'validate_table_name' in base.__all__
