from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from types import ModuleType
from typing import Any

from nexus.utils.otel._instrumentor import NexusModuleInstrumentor
from nexus.utils.otel.auto import is_enabled
from nexus.utils.otel.propagation import add_trace_state_entries


def _inject_correlation(original: Any) -> Any:
    @wraps(original)
    def new_row(repository: Any, request: Any, **kwargs: Any) -> Any:
        row = original(repository, request, **kwargs)
        if not is_enabled():
            return row
        try:
            batch = request.spec.get('batch')
            item_id = batch.get('item_id') if isinstance(batch, Mapping) else None
            # Child retry keys must never become the top-level submission correlation ID.
            is_child = request.parent_execution_id is not None or item_id is not None
            entries = (('hyway_instance_id', item_id),) if is_child else (('hyway_slice_id', request.idempotency_key),)
            row.trace_carrier_json = add_trace_state_entries(row.trace_carrier_json, entries) or None
        except Exception:
            # Correlation enrichment must not prevent persistence of the original carrier.
            pass
        return row

    return new_row


class NexusHywayInstrumentor(NexusModuleInstrumentor):
    _module_name = 'nexus.distributed.repositories.execution'

    def _patch_module(self, module: ModuleType) -> None:
        self._wrap(module.ExecutionRepository, '_new_execution_row', _inject_correlation)
