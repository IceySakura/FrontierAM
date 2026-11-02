"""Shared libraries used across SWE-Any stages, generators, and tasks.

- ``graph`` — protobuf-free entity-graph model + SCIP symbol parsing + JSON IO.
- ``scip`` — SCIP toolchain (``scip-python`` indexer + committed ``scip_pb2``).
- ``source_edit`` — graph-span source editing (stub/delete → unified diff + inverse).
- ``classify_tests`` — Python test classification over the graph.
- ``constants`` — sandbox paths, artifact filenames, and stage defaults.
- ``agent_output`` — tolerant reader for the JSON an agentic pass writes.
- ``batch_config`` — instance loading + config merging for the ``generator`` package.
"""

from __future__ import annotations
