"""run-batch generator for the code_mt executor over harbor 0720 task dirs.

code_mt is "harbor_bridge + MT loop + host-side judge mixing": it grades with the
SAME harbor-native machinery (Verifier / GongfengEnvironment / separate-verifier)
and reads EVERY business/grading field from ``{task_dir}/task.toml`` via
``loader.load_task_spec`` at task runtime. So this generator does NOT re-implement
task discovery or runtime/agent/model/boot wiring — it DELEGATES all of it to
``harbor_bridge.generator.generate_configs`` (the single battle-tested source of
that logic) and only RETARGETS each emitted BatchItem to the code_mt task:

  * ``task_name``  harbor_bridge_task -> code_mt_task
  * ``imports``    swap the task module import (agent + noauth imports untouched)

Everything else in the RunInput is reused verbatim. The retarget is loss-free:
  * runtime_configs keep the ``main`` / ``verifier`` keys code_mt expects
    (infer_runtime_name / verifier_runtime_name defaults).
  * harbor_bridge-only task_config keys (instruction, run_nop,
    skip_environment_upload, ...) are silently ignored by CodeMTTaskInput
    (TaskRunInput has extra='ignore').
  * code_mt re-resolves verifier_mode + boot from task.toml itself, so passing
    harbor_bridge's already-correct values just matches.

Same env-var contract as harbor_bridge (HARBOR_DATASET_PATH, RUN_ORACLE,
AGENT_NAME, OPENAI_*, GONGFENG_*), so the existing run_oracle.sh works unchanged
apart from ``-g`` pointing at this module. For a real code_mt agent-loop, set
``AGENT_NAME=code_mt`` so the code_mt agent (instruction.md render + MT tools) runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from nexus.extensions.harbor_bridge.generator import (
    generate_configs as _harbor_generate_configs,
)
from nexus.extensions.harbor_bridge.generator import (
    get_batch_metadata as _harbor_batch_metadata,
)
from nexus.extensions.harbor_bridge.generator import (
    get_total_count,  # re-exported unchanged (task discovery is identical)
)
from nexus.runner.batch.batch_config import BatchItem


_HARBOR_TASK_IMPORT = 'nexus.extensions.harbor_bridge.task'
_CODE_MT_TASK_IMPORT = 'nexus.extensions.projects.code_mt.task.task'
# harbor_bridge.generator emits the agent import as
# ``nexus.extensions.agents.{agent_name}`` (generator.py). For the code_mt agent
# that STALE path no longer exists after the move to extensions/projects; run()
# imports every entry in ``imports`` (side-effect registration) BEFORE the
# registry lookup, so leaving it unrewritten raises ModuleNotFoundError in a
# clean wheel env. The registering module is agent/agent.py (the package
# __init__.py does NOT import it), so we must retarget to the *.agent.agent
# module, not the package.
_STALE_CODE_MT_AGENT_IMPORT = 'nexus.extensions.agents.code_mt'
_CODE_MT_AGENT_IMPORT = 'nexus.extensions.projects.code_mt.agent.agent'


def _retarget(item: BatchItem) -> BatchItem:
    cfg = item.config
    imports = [
        _CODE_MT_AGENT_IMPORT
        if imp == _STALE_CODE_MT_AGENT_IMPORT
        else _CODE_MT_TASK_IMPORT
        if imp == _HARBOR_TASK_IMPORT
        else imp
        for imp in cfg.imports
    ]
    if _CODE_MT_TASK_IMPORT not in imports:
        imports.append(_CODE_MT_TASK_IMPORT)
    new_cfg = cfg.model_copy(update={'task_name': 'code_mt_task', 'imports': imports})
    return item.model_copy(update={'config': new_cfg})


def generate_configs() -> Iterator[BatchItem]:
    """Yield harbor_bridge BatchItems retargeted to the code_mt task."""
    for item in _harbor_generate_configs():
        yield _retarget(item)


def get_batch_metadata() -> dict[str, Any]:
    md = _harbor_batch_metadata()
    md['task_name'] = 'code_mt_task'
    md['bridge'] = 'code_mt'
    return md


__all__ = ['generate_configs', 'get_batch_metadata', 'get_total_count']
