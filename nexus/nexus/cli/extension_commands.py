"""Lazy CLI command dispatch for extension modules."""

from __future__ import annotations

import argparse

from nexus.core import EntryNotFoundError, command_registry, module_registry


def _command_names() -> list[str]:
    """Return lazily discovered module command names."""
    module_registry.discover_and_register()
    return sorted(entry.name for entry in command_registry.list_entries())


def _print_available_commands() -> None:
    """Print available extension module commands without importing their modules."""
    names = _command_names()
    if not names:
        print('No module commands registered.')
        return
    print('Available module commands:')
    for name in names:
        print(f'  {name}')


def handle_module_commands(args: argparse.Namespace) -> int:
    """Dispatch a lazily loaded extension module command."""
    command_args = list(getattr(args, 'module_command_args', None) or [])
    if not command_args or command_args[0] in {'-h', '--help'}:
        _print_available_commands()
        return 0

    command_name = command_args[0]
    module_registry.discover_and_register()
    try:
        command = command_registry.get(command_name)
    except EntryNotFoundError:
        print(f'Unknown module command: {command_name}')
        _print_available_commands()
        return 1

    parser = argparse.ArgumentParser(prog='nexus module_commands')
    command_subparsers = parser.add_subparsers(dest='module_command', required=True)
    command.register_parser(command_subparsers, command_name)
    parsed_args = parser.parse_args(command_args)
    return command.handle_command(parsed_args)


def register_module_commands_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the lazy module_commands CLI entrypoint."""
    parser = subparsers.add_parser(
        'module_commands',
        help='Run extension module commands',
        description='Run extension module commands lazily. Use `nexus module_commands` to list available commands.',
    )
    parser.add_argument('module_command_args', nargs=argparse.REMAINDER, help='Module command name and arguments')


__all__ = ['handle_module_commands', 'register_module_commands_parser']
