import os
import subprocess
import sys

from pathlib import Path


def test_root_exports_resolve_to_runner_types():
    script = """
import nexus
assert nexus.module_registry.list_modules()
from nexus import BatchItem, ModuleObjectConfig, ProcessorConfig, RunInput
from nexus.runner.batch import BatchItem as ExpectedBatchItem
from nexus.runner.single import (
    ModuleObjectConfig as ExpectedModuleObjectConfig,
    ProcessorConfig as ExpectedProcessorConfig,
    RunInput as ExpectedRunInput,
)
assert BatchItem is ExpectedBatchItem
assert ModuleObjectConfig is ExpectedModuleObjectConfig
assert ProcessorConfig is ExpectedProcessorConfig
assert RunInput is ExpectedRunInput
from nexus.runner import RunHooks, RunOutput, run
from nexus.runner import single
assert RunHooks is single.RunHooks
assert RunOutput is single.RunOutput
assert run is single.run
assert not hasattr(nexus, 'unknown_export')
exports = {}
exec('from nexus import *', exports)
for name in ('BatchItem', 'ModuleObjectConfig', 'ProcessorConfig', 'RunInput', 'module_registry'):
    assert exports[name] is getattr(nexus, name)
"""
    subprocess.run(
        [sys.executable, '-c', script],
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[2])},
        capture_output=True,
        text=True,
        check=True,
    )
