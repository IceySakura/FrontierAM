"""SWE-Any Parse — repo → SCIP index → entity graph (stage package).

Parse's two halves mirror Select's ``heuristics`` + ``prompts`` split:

- :mod:`.heuristics` is the deterministic step — it turns a SCIP index into the
  entity graph model (``build_graph`` / ``load_index``), re-exported here. It is
  the SCIP/protobuf-dependent side (pulls ``scip.scip_pb2``).
- :mod:`.prompts` holds the Claude Code prompt for the agentic seed-policy pass
  that follows indexing (it classifies repo paths so Select never seeds inside
  generated code); import it directly (``from ...parse.prompts import ...``).

    from nexus.extensions.tasks.swe_any.stage.parse import build_graph
    from nexus.extensions.tasks.swe_any.utils.graph import read_json

    graph = build_graph('parse_index.scip', graph_path='parse_graph.json')

Repo indexing (``scip-python``) and the ``.scip`` protobuf bindings live in the
shared ``utils.scip`` package; the protobuf-free graph model lives in ``utils.graph``. The
``is_test`` re-tagging Parse runs after the build lives in the shared
``classify_tests`` module (``annotate_tests``).
"""

from __future__ import annotations

from .heuristics import build_graph, load_index
from .runner import run_parse_operation, run_parse_repo, run_seed_policy_pass
from .schema import ParseTaskInput, ParseTaskOutput


__all__ = [
    'ParseTaskInput',
    'ParseTaskOutput',
    'build_graph',
    'load_index',
    'run_parse_operation',
    'run_parse_repo',
    'run_seed_policy_pass',
]
