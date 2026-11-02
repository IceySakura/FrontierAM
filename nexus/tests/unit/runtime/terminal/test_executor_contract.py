import pytest

from pydantic import ValidationError

from nexus.core import executor_registry
from nexus.core.module_registry import module_registry
from nexus.runtime.executors.models.terminal import CreateTerminalSessionRequest


EXPECTED = [
    'create_terminal_session',
    'destroy_terminal_session',
    'run_command_in_terminal_session',
    'start_command_in_terminal_session',
    'query_terminal_command_status',
    'send_keys_to_terminal_session',
    'capture_terminal_pane',
    'terminate_terminal_session_processes',
]


@pytest.fixture(autouse=True)
def _discover():
    module_registry.discover_and_register()


@pytest.mark.parametrize('name', EXPECTED)
def test_executor_resolvable(name):
    fn = executor_registry.get(name)
    assert callable(fn)


@pytest.mark.parametrize(
    ('height', 'width'),
    [(0, 120), (65536, 120), (40, 0), (40, 65536)],
)
def test_create_session_rejects_dimensions_outside_pty_range(height, width):
    with pytest.raises(ValidationError):
        CreateTerminalSessionRequest(height=height, width=width)
