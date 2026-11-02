from pathlib import Path

from nexus.core.module_meta import discover_all_modules


EXPECTED_EXECUTORS = {
    'create_terminal_session',
    'destroy_terminal_session',
    'run_command_in_terminal_session',
    'start_command_in_terminal_session',
    'query_terminal_command_status',
    'send_keys_to_terminal_session',
    'capture_terminal_pane',
    'terminate_terminal_session_processes',
}


def _project_root() -> Path:
    import nexus

    return Path(nexus.__file__).parent.parent


def test_terminal_is_directory_module_with_all_executors():
    modules = discover_all_modules(_project_root(), include_compiled_manifest=False)
    terminal = next((m for m in modules if m.module_path == 'nexus.runtime.executors.terminal'), None)
    assert terminal is not None, 'terminal directory module not discovered'
    assert terminal.kind == 'directory'
    registered = set(terminal.objects.get('executors', {}).keys())
    assert registered == EXPECTED_EXECUTORS
