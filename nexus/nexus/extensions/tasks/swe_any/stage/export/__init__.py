"""SWE-Any Export — assemble → execution-grade → emit a Harbor dataset directory.

Deterministic stage (no LLM). For each feature that has a usable mask + oracle + issue it:

1. joins Mask/Eval/Issue on ``index`` into a SWE-bench-shaped record (:mod:`.schema`),
2. execution-grades it in-sandbox — run the oracle tests pre-gold (feature masked → FAIL) and
   post-gold (implementation restored → PASS) and derive/validate structured
   FAIL_TO_PASS / PASS_TO_PASS (gate: non-empty F2P and no regressions),
3. writes an experimental Harbor directory-per-instance artifact (:mod:`.harbor`).

The Harbor format adaptation is incomplete and still under test. Do not use the emitted Harbor
directories for evaluation; use ``swe_any_instances.jsonl`` through the SWE-Any dataset handler.
"""

from __future__ import annotations

from .harbor import write_dataset_toml, write_harbor_instance
from .runner import run_export_feature
from .schema import (
    ExportTaskInput,
    ExportTaskOutput,
    build_swebench_instance,
    empty_export_row,
    export_instance_id,
    export_unified_row,
    load_export_features,
)


__all__ = [
    'ExportTaskInput',
    'ExportTaskOutput',
    'build_swebench_instance',
    'empty_export_row',
    'export_instance_id',
    'export_unified_row',
    'load_export_features',
    'run_export_feature',
    'write_dataset_toml',
    'write_harbor_instance',
]
