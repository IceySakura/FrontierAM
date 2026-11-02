"""Schemas for the SWE-Any Parse stage."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore
from nexus.extensions.tasks.swe_any.utils import constants


class ParseTaskInput(TaskRunInput, WithAgentRunner):
    """Business inputs for one repo's Parse run (SCIP index, then seed policy)."""

    instance: dict[str, Any] | None = None
    instance_path: str = ''
    """Optional one-row JSONL; ignored when ``instance`` is provided."""

    # Identity + host IO
    repo_name: str = 'repo'
    output_dir: str = './outputs/swe_any'
    """Host dir; parse_graph.json is downloaded to ``output_dir/<repo_name>/parse_graph.json``."""

    # Sandbox-side layout
    repo_dir: str = constants.SANDBOX_REPO_DIR
    """Sandbox path to the repository source in the baked per-repo image (SWE-Env ``workdir``)."""
    base_commit: str = ''
    """Git commit restored before this stage; hydrated from the source SWE-Env instance."""
    swe_any_dir: str = constants.SANDBOX_SWE_ANY_DIR
    graph_filename: str = constants.GRAPH_FILENAME
    seed_policy_filename: str = constants.SEED_POLICY_FILENAME
    runtime_name: str = 'main'

    # Baked per-repo image (the workload image). Empty = rely on the provider default.
    image: str = ''
    image_override: str | None = None

    # Parse knobs
    project_name: str | None = None
    install_scip_python: bool = True
    npm_registry: str | None = 'http://mirrors.tencent.com/npm/'
    index_timeout: float = 3600.0
    npm_install_timeout: float = 600.0

    # Seed-policy pass: classify repo paths once so Select can shape its seed pool.
    # Soft by design — a failure here leaves parse_graph.json intact and Select falls
    # back to no exclusions rather than paying for a re-index.
    agent_timeout: float = 3600.0
    profile_dirs: int = Field(default=40, ge=1)
    """How many directories of the graph profile to show the agent."""

    # Host-side ceiling on the in-sandbox invoke (seconds).
    request_timeout: float = 5400.0
    download: bool = True
    # Skip early release so another stage in the same ExecutionContext can reuse it.
    # ExecutionContext.__aexit__ still releases all runtimes on success or failure.
    release_runtime: bool = False

    @model_validator(mode='after')
    def ensure_request_timeout_covers_parse(self) -> ParseTaskInput:
        """Leave enough time for npm installation, indexing, and invoke overhead."""
        self.request_timeout = max(self.request_timeout, self.npm_install_timeout + self.index_timeout + 300.0)
        return self


class ParseTaskOutput(TaskRunOutput, WithScore):
    """Result of one repo's Parse run."""

    repo_name: str
    nodes_internal: int = 0
    nodes_total: int = 0
    edges_internal: int = 0
    edges_total: int = 0
    sandbox_graph_path: str | None = None
    graph_path: str | None = None  # host path of the downloaded parse_graph.json
    index_path: str | None = None
    seed_policy_path: str | None = None  # host path of the downloaded parse_seed_policy.json
    exclude_globs: list[str] = Field(default_factory=list)
    seed_policy_error: str | None = None
    """Why the seed-policy pass produced no usable policy. Does not fail the task."""


__all__ = [
    'ParseTaskInput',
    'ParseTaskOutput',
]
