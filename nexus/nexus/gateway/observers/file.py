# /// script
# dependencies = []
#
# [tool.nexus]
# trajectory_observers = { trajectory_file = "nexus.gateway.observers.file" }
# ///

"""Final-only, best-effort trajectory export to a local or mounted Ceph directory."""

from __future__ import annotations

import copy
import gzip
import json
import os
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus.gateway.observers.base import TrajectoryObserver
from nexus.gateway.observers.registry import trajectory_observer_registry
from nexus.gateway.trajectory import Trajectory


def _trajectory_file_stem(session_id: str) -> str:
    """Map trajectory.session_id onto a single-path filename stem.

    Gateway session_id is HyTuner's trajectory_id, and the same string is the
    Langfuse Name/Session used for cross-search. Demo fixtures may contain `/`,
    which must not become extra directories under output_dir.
    """
    if not session_id or not session_id.strip():
        raise ValueError('trajectory_file requires a non-empty trajectory.session_id')
    stem = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in session_id)
    if not stem.strip('._-'):
        raise ValueError(f'trajectory.session_id {session_id!r} is not a usable file stem')
    return stem


def visualization_dump(trajectory: Trajectory) -> dict[str, Any]:
    """Export wire/semantic facts without training headers or temporary store handles.

    A shallow view drops training-only headers without copying tensors or mutating
    the source. Trajectory.dump constructs the independent plain-data result.
    Payloads from the runtime record store are deliberately not materialized.
    """
    view = copy.copy(trajectory)
    view.segment_training_data = {}
    return view.dump()


def _atomic_json(path: Path, value: dict[str, Any], *, compress: bool = False) -> None:
    """Publish only complete files; temporary files stay on the destination mount."""
    fd, temporary = tempfile.mkstemp(prefix='.trajectory-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as raw:
            if compress:
                with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                    for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(value):
                        stream.write(chunk.encode('utf-8'))
            else:
                for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(value):
                    raw.write(chunk.encode('utf-8'))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@trajectory_observer_registry.register('trajectory_file')
class FileTrajectoryObserver(TrajectoryObserver):
    """Write one immutable JSON artifact and small catalog entry per final callback.

    The output directory must be supplied explicitly through observer configuration.
    Errors are isolated and logged by TrajectoryObservationHub.
    """

    def __init__(self, output_dir: str, *, compress: bool = True) -> None:
        if not output_dir or not output_dir.strip():
            raise ValueError('trajectory_file requires a non-empty output_dir')
        if '://' in output_dir:
            raise ValueError('trajectory_file requires a local or mounted Ceph path, not an object-store URI')
        self.output_dir = Path(output_dir).expanduser().absolute()
        self.compress = compress

    @property
    def observes_updates(self) -> bool:
        return False

    def on_final(self, trajectory: Trajectory) -> None:
        artifact_id = _trajectory_file_stem(trajectory.session_id)
        created_at = datetime.now(timezone.utc).isoformat()
        data = visualization_dump(trajectory)
        summary = {
            'artifact_id': artifact_id,
            'created_at': created_at,
            'session_id': trajectory.session_id,
            'trajectory_id': trajectory.session_id,
            'state': trajectory.state,
            'node_count': len(trajectory.nodes),
            'chain_count': len(trajectory.chains),
            'final_reward': trajectory.final_reward,
            'termination_reason': trajectory.termination_reason,
        }
        filename = f'{artifact_id}.trajectory.json' + ('.gz' if self.compress else '')
        artifact = {
            **summary,
            'omitted': ['segment_training_data', 'token_account.payload_ref', 'external_node_payloads'],
            'trajectory': data,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.output_dir / filename, artifact, compress=self.compress)
        # Publish the catalog entry last. Readers never discover half-written artifacts.
        _atomic_json(self.output_dir / f'{artifact_id}.summary.json', {**summary, 'file': filename})
