"""Version-tolerant Codex request metadata for Gateway policy inspection."""

from __future__ import annotations

import copy
import json
import logging
import re

from collections.abc import Mapping
from typing import Any

from nexus.runtime.server.services.codex_compaction import CODEX_TURN_METADATA_HEADER, CodexCompactionEvidence


logger = logging.getLogger(__name__)
_USER_AGENT_VERSION = re.compile(r'(?:codex_exec|codex_cli_rs|codex)/([^\s(]+)')


def build_codex_request_meta(
    request_body: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    configured_version: str | None,
    request_id: str,
    compaction_evidence: CodexCompactionEvidence | None,
    tool_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project available protocol facts without requiring one exact Codex version."""
    normalized_headers = {str(key).lower(): str(value) for key, value in headers.items()}
    user_agent = normalized_headers.get('user-agent', '')
    version_match = _USER_AGENT_VERSION.search(user_agent)
    observed_version = version_match.group(1) if version_match is not None else None
    scaffold_version = observed_version or configured_version or 'unknown'

    client_metadata = request_body.get('client_metadata')
    client_raw = client_metadata.get(CODEX_TURN_METADATA_HEADER) if isinstance(client_metadata, Mapping) else None
    header_raw = normalized_headers.get(CODEX_TURN_METADATA_HEADER)
    header_metadata = _parse_metadata(header_raw, 'turn_metadata')
    client_turn_metadata = _parse_metadata(client_raw, 'client_metadata')
    if header_metadata is not None:
        codex = header_metadata
        metadata_source = 'turn_metadata'
    elif client_turn_metadata is not None:
        codex = client_turn_metadata
        metadata_source = 'client_metadata'
    else:
        codex = {}
        metadata_source = 'none'

    request_meta: dict[str, Any] = {
        'request_id': request_id,
        'scaffold': 'codex',
        'scaffold_version': scaffold_version,
        'codex': copy.deepcopy(codex),
        'codex_metadata_source': metadata_source,
    }
    if configured_version is not None and configured_version != scaffold_version:
        request_meta['configured_scaffold_version'] = configured_version
    if compaction_evidence is not None:
        request_meta['codex_compaction_detector'] = compaction_evidence.detector
    if codex.get('request_kind') == 'compaction' or compaction_evidence is not None:
        request_meta['invocation_kind'] = 'summary_inference'
    if tool_identities:
        request_meta['codex_tool_identities'] = {
            str(name): identity.to_metadata() if hasattr(identity, 'to_metadata') else copy.deepcopy(identity)
            for name, identity in tool_identities.items()
        }
    return request_meta


def _parse_metadata(raw: Any, source: str) -> dict[str, Any] | None:
    if isinstance(raw, Mapping):
        return copy.deepcopy(dict(raw))
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning('Ignoring malformed Codex %s JSON; Gateway will keep the request on the main chain', source)
        return None
    if not isinstance(parsed, dict):
        logger.warning('Ignoring non-object Codex %s; Gateway will keep the request on the main chain', source)
        return None
    return parsed


__all__ = ['build_codex_request_meta']
