"""Unit coverage for extension-provided CLI commands."""

from __future__ import annotations

import argparse


def test_register_module_commands_parser_is_lazy(monkeypatch):
    """The top-level CLI should register only the module_commands entrypoint."""
    from nexus.cli.extension_commands import register_module_commands_parser

    monkeypatch.setattr(
        'nexus.cli.extension_commands.module_registry.discover_and_register',
        lambda: (_ for _ in ()).throw(AssertionError('discovery should be lazy')),
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    register_module_commands_parser(subparsers)
    args = parser.parse_args(['module_commands', 'fake-command', '--flag'])

    assert args.command == 'module_commands'
    assert args.module_command_args == ['fake-command', '--flag']


def test_handle_module_commands_imports_only_selected_command(monkeypatch):
    """Extension command modules should load only after a concrete module command is selected."""
    from nexus.cli.extension_commands import handle_module_commands

    calls: list[str] = []
    discovered: list[bool] = []
    loaded: list[str] = []

    class FakeCommand:
        @staticmethod
        def register_parser(subparsers, command_name: str) -> None:
            parser = subparsers.add_parser(command_name)
            parser.add_argument('--flag', action='store_true')

        @staticmethod
        def handle_command(args) -> int:
            calls.append(args.module_command)
            return 7

    def fake_get(command_name: str):
        loaded.append(command_name)
        return FakeCommand

    monkeypatch.setattr(
        'nexus.cli.extension_commands.module_registry.discover_and_register',
        lambda: discovered.append(True),
    )
    monkeypatch.setattr('nexus.cli.extension_commands.command_registry.get', fake_get)

    args = argparse.Namespace(module_command_args=['fake-command-from-registry', '--flag'])

    assert handle_module_commands(args) == 7
    assert discovered == [True]
    assert loaded == ['fake-command-from-registry']
    assert calls == ['fake-command-from-registry']


def test_module_list_summary_includes_commands():
    """The modules CLI object summary should display command registrations."""
    from nexus.cli.commands.modules import _obj_summary

    assert _obj_summary({'commands': {'build-ubuntu-session-image': 'x.y'}}) == '1 command'
