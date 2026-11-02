"""Guard the public import surface of the terminal package.

The legacy single-file terminal.py re-exported the request/response models via
`from ...models.terminal import *`, and upstream modules import them straight
from `nexus.runtime.executors.terminal` (e.g. swe_agent_rubrics/tools.py). The
package refactor must keep that surface intact.
"""

import importlib

import pytest


# Models that upstream code imports from the package path.
EXPECTED_MODELS = [
    'CommandInfo',
    'CreateTerminalSessionRequest',
    'CreateTerminalSessionResponse',
    'DestroyTerminalSessionRequest',
    'DestroyTerminalSessionResponse',
    'RunCommandInSessionRequest',
    'StartCommandInSessionRequest',
    'QueryCommandStatusRequest',
    'SendKeysToSessionRequest',
    'SendKeysToSessionResponse',
    'CapturePaneRequest',
    'CapturePaneResponse',
    'TerminateSessionProcessesRequest',
    'TerminateSessionProcessesResponse',
]


@pytest.mark.parametrize('name', EXPECTED_MODELS)
def test_model_importable_from_package(name):
    pkg = importlib.import_module('nexus.runtime.executors.terminal')
    assert hasattr(pkg, name), f'{name} must be importable from nexus.runtime.executors.terminal'


def test_session_and_executors_exported():
    pkg = importlib.import_module('nexus.runtime.executors.terminal')
    assert hasattr(pkg, 'TerminalSession')
    assert hasattr(pkg, 'create_terminal_session')
    assert hasattr(pkg, 'TERMINAL_SESSIONS')
