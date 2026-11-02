"""Shared trajectory record backend helpers."""

from __future__ import annotations

from typing import Any

from .types import NodePayloadRecord, Record


def _record_fields(record: Record) -> dict[str, Any]:
    if isinstance(record, NodePayloadRecord):
        return {
            'token_ids': record.token_ids,
            **{
                name: value
                for name, value in (
                    ('logprobs', record.logprobs),
                    ('topk_indices', record.topk_indices),
                    ('topk_values', record.topk_values),
                    ('routed_experts', record.routed_experts),
                    ('multi_modal_items', record.multi_modal_items),
                )
                if value is not None
            },
        }
    return dict(record.fields)
