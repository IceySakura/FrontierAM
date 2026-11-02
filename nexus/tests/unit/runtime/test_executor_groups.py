"""Tests for executor registration via plain @register decorator."""

import pytest

from nexus.core.registry import Registry


class TestRegisterDecorator:
    def test_register_standalone_function(self):
        """@register decorates standalone functions directly."""
        reg = Registry('test_register_fn')

        @reg.register('do_something')
        async def do_something(x: int) -> int:
            return x * 2

        @reg.register('do_other')
        async def do_other(y: str) -> str:
            return y.upper()

        assert reg.get('do_something') is do_something
        assert reg.get('do_other') is do_other

    def test_register_not_found(self):
        reg = Registry('test_register_missing')
        with pytest.raises(Exception):
            reg.get('nonexistent')

    def test_register_with_different_name(self):
        """Registration name can differ from function name."""
        reg = Registry('test_register_alias')

        @reg.register('aliased_name')
        async def actual_function_name() -> None:
            pass

        assert reg.get('aliased_name') is actual_function_name
        with pytest.raises(Exception):
            reg.get('actual_function_name')

    def test_duplicate_register_raises(self):
        """Re-registering the same name raises DuplicateEntryError."""
        reg = Registry('test_register_dup')

        @reg.register('unique_name')
        async def first() -> None:
            pass

        with pytest.raises(Exception):

            @reg.register('unique_name')
            async def second() -> None:
                pass

    def test_duplicate_lazy_add_same_import_path_is_idempotent(self, caplog):
        """Lazy metadata discovery may see the same tool exposed by two extension modules."""
        reg = Registry('test_lazy_add_same_import')

        reg.add('shared_tool', 'pkg.module')
        reg.add('shared_tool', 'pkg.module')

        assert not [record for record in caplog.records if 'shared_tool' in record.message]
