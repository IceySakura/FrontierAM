"""SCIP toolchain — provision ``scip-python``, index a repo, decode ``.scip``.

The SCIP/protobuf-dependent side of the pipeline, kept apart from the ``stage``
packages and the protobuf-free ``graph`` model (its sibling under ``utils``):

- :mod:`.indexer` runs the ``scip-python`` CLI (plain ``subprocess``, no protobuf)
  to produce ``index.scip``; re-exported here.
- :mod:`.scip_pb2` is the committed, pre-generated protobuf bindings for the
  ``.scip`` format (regenerated from ``scip_proto/scip.proto`` via
  ``grpc_tools.protoc``). It needs the ``protobuf`` runtime, so it is imported as
  a submodule (``from ...utils.scip import scip_pb2``) rather than re-exported,
  keeping ``import ...utils.scip`` for the indexer alone protobuf-free.

The SCIP index → entity graph build lives in ``stage.parse.heuristics`` (it depends
on both ``scip_pb2`` and the ``graph`` model).
"""

from __future__ import annotations

from .indexer import ensure_scip_python, find_scip_python, run_scip_index


__all__ = [
    'ensure_scip_python',
    'find_scip_python',
    'run_scip_index',
]
