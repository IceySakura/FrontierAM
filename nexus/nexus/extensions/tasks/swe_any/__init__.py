"""SWE-Any task package — large-scale SWE dataset construction.

Pipeline: **Parse → Select → Mask → Eval → Issue → Export**, all implemented (Parse
migrated from the standalone ``swe_any`` project).

Layout — four groups plus the root orchestration:
- ``stage/``     — the pipeline stages, each an ``heuristics`` (deterministic) +
  ``prompts`` (agentic) pair; stages depend only on ``utils``, never on each other:
  - ``parse``  — SCIP index → entity graph (``build_graph``); seed-policy prompt.
  - ``select`` — layered-BFS ``select_subgraph`` / ``sample_seeds``; slice agentic pass.
  - ``mask``   — keep entry signatures, delete the rest → draft mask/gold patch; refine.
  - ``eval``   — hide ALL feature tests (``test_mask_patch``) + author the forward
    ``test_gold_patch`` oracle; test-design.
  - ``issue``  — write a non-leaking ``problem_statement`` from the masked tree (agentic-only).
  - ``export`` — execution-grade (F2P/P2P) + Harbor emit (deterministic).
- ``utils/``     — shared libraries used across stages / generators / tasks:
  - ``graph``  — protobuf-free entity-graph model + SCIP symbol parsing + JSON IO.
  - ``scip``   — SCIP toolchain: ``scip-python`` indexer + committed ``scip_pb2``.
  - ``source_edit`` — graph-span stub/delete → unified diff + inverse.
  - ``classify_tests`` — Python ``is_test`` re-tagging (``annotate_tests``) + predicate.
  - ``constants`` / ``agent_output`` / ``batch_config`` — paths, agent-JSON reader, config merge.
- ``generator/`` — ``stage/`` and ``pipeline/`` batch generators mirroring ``config/``.
- ``scripts/``   — ``run_single.*`` / ``run_batch.*`` shell wrappers.
- ``task.py``    — legacy six-stage repo orchestration (registered).
- ``pipeline/``  — additive pre-feature, per-feature, and post-feature tasks (registered).
- ``runtime_setup.py`` — sandbox-side ``run_{parse,mask,eval,apply,grade}`` invoked via ``runtime.invoke``.
- ``config/``    — stage default YAML + client/custom patches (run-single & generators).

See ``task.py`` for orchestration and ``README.md`` for the entity/relation catalogue.
"""

# Importing this package is the public registration boundary used by every
# default config. Keep implementation imports here so all legacy and additive
# task names are available without requiring private module paths in YAML.
from nexus.extensions.tasks.swe_any import pipeline as pipeline
from nexus.extensions.tasks.swe_any import task as task


__all__ = ['pipeline', 'task']
