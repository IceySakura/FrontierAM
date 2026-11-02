"""Media handling for the gateway's expanded token view.

The adapter is injected whole, the way a tokenizer is, so family differences stay inside
its implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from nexus.gateway.types import TOKEN_ID_DTYPE


class PromptExpansion(Protocol):
    token_ids: list[int]
    multi_modal_items: dict[str, list[dict[str, Any]]] | None


class GatewayMultiModalProtocol(Protocol):
    """The media contract the Processor requires, satisfied by one injected object.

    HyTuner injects its per-family ``MultiModalPlugin`` here; nexus never imports it, so any
    caller supplying these three methods can drive the expanded token view.
    """

    async def process_multi_modal_info(self, messages: list[dict]) -> dict[str, Any]:
        """Media this conversation carries, keyed by modality, or {} when text-only."""
        ...

    def get_placeholder_token_ids(self) -> Mapping[str, int]:
        """Raw-prompt placeholder token IDs keyed by media modality."""
        ...

    def expand_prompt(self, text: str, multi_modal_data: Mapping[str, Any]) -> PromptExpansion:
        """The exact expanded token ids and media tensors from one processor call."""
        ...


def partition_expanded_ids(
    raw_message_ids: list[np.ndarray],
    prompt_ids: np.ndarray,
    placeholder_id: int,
) -> list[np.ndarray]:
    """Partition processor output by raw spans without predicting visual token counts.

    Image placeholders must stay isolated by the template's boundary tokens: a flattened
    run of them would be consumed whole by the first placeholder, so every later one trips
    the missing image-run check below. Returned partitions remain views of prompt_ids.
    """
    raw_prompt_ids = np.concatenate(raw_message_ids) if raw_message_ids else np.empty(0, dtype=TOKEN_ID_DTYPE)
    # NOTE(jerryhrzhou): Collapsing each image run must reconstruct exactly one raw placeholder.
    image_mask = prompt_ids == placeholder_id
    keep = np.ones(len(prompt_ids), dtype=np.bool_)
    keep[1:] = ~(image_mask[1:] & image_mask[:-1])
    token_starts = np.flatnonzero(keep)
    collapsed_ids = prompt_ids[token_starts]

    common_length = min(len(raw_prompt_ids), len(collapsed_ids))
    mismatches = np.flatnonzero(raw_prompt_ids[:common_length] != collapsed_ids[:common_length])
    if mismatches.size or len(raw_prompt_ids) > len(collapsed_ids):
        mismatch_index = int(mismatches[0]) if mismatches.size else common_length
        if raw_prompt_ids[mismatch_index] == placeholder_id:
            raise ValueError(
                'processor output contains no image-token run for a raw image placeholder; '
                'the chat template must isolate each image with non-image boundary tokens'
            )
        raise ValueError(
            'processor output disagrees with the raw prompt at a non-image token; '
            'refusing to construct a potentially misaligned expanded token view'
        )
    if len(raw_prompt_ids) < len(collapsed_ids):
        remaining = len(prompt_ids) - int(token_starts[len(raw_prompt_ids)])
        raise ValueError(
            f'processor output has {remaining} unpartitioned token(s); '
            'the raw prompt and processor output have different lengths'
        )

    # NOTE(jerryhrzhou): The terminal offset also maps trailing empty messages without special cases.
    expanded_offsets = np.concatenate((token_starts, [len(prompt_ids)]))
    raw_boundaries = np.cumsum([0, *(len(ids) for ids in raw_message_ids)])
    boundaries = expanded_offsets[raw_boundaries]
    return [prompt_ids[start:stop] for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)]
