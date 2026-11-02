"""Host-side cache locations for OSWorld setup assets and evaluator fixtures.

Two rules keep the cache reusable while isolating evaluator references:

**Setup assets can share one flat directory.** ``setup._download_cache_path``
names every asset ``<uuid5(NAMESPACE_URL, url)>_<basename>``, which is globally
unique, so two tasks can share a directory -- and should, since that is what
makes the cache reusable across tasks and across runs.

**Evaluator reference files cannot.** ``getters.file.get_cloud_file`` names them
by their raw ``dest``, which repeats across tasks: in the official task set
``log_Gold.txt`` and ``test_suite.py`` each map to two different URLs. Pointing
every task at one shared directory therefore makes the second task silently
grade against the first task's reference file, so evaluator assets get a
per-task subdirectory.

Resolution order for both is: explicit argument (from ``task_config``) >
``NEXUS_OSWORLD_CACHE_DIR`` > packaged default. The explicit argument is what
platform runs must use -- ``run_single`` may not depend on process environment
(Agent-Bench platform spec 2.1), so the generator reads the env var offline and
bakes the result into the config.
"""

from __future__ import annotations

import os

from pathlib import Path


# nexus/extensions/tasks/os_world/cache -- resolved from this file, not the cwd.
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[1] / 'cache'

CACHE_DIR_ENV_VAR = 'NEXUS_OSWORLD_CACHE_DIR'

# Evaluator reference files live under this subdirectory of the cache root, one
# directory per task, so ``dest`` collisions between tasks cannot cross over.
EVALUATOR_SUBDIR = '_evaluator'


def cache_root(cache_dir: str | Path | None = None) -> Path:
    """Host-side cache root.

    Note the explicit ``.strip()`` emptiness check -- ``Path('')`` evaluates to
    ``Path('.')``, which is truthy, so an ``or``-based fallback silently resolves
    to the working directory instead.
    """
    if cache_dir is not None and str(cache_dir).strip():
        return Path(str(cache_dir).strip())
    override = os.environ.get(CACHE_DIR_ENV_VAR, '').strip()
    return Path(override) if override else DEFAULT_CACHE_ROOT


def resolve_setup_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Cache dir for setup ``download`` assets -- flat, shared by every task.

    Safe to share because ``setup._download_cache_path`` prefixes each name with
    ``uuid5(NAMESPACE_URL, url)``: distinct URLs can never collide, and two tasks
    pulling the *same* URL want the same bytes anyway. Concurrent writers are
    fine too -- ``setup._write_bytes_atomic`` writes to a unique temp name and
    ``Path.replace()``s it into place.
    """
    return cache_root(cache_dir)


def resolve_evaluator_cache_dir(task_id: str, cache_dir: str | Path | None = None) -> Path:
    """Per-task cache dir for evaluator reference files.

    Per-task rather than flat because ``get_cloud_file`` keys on the raw ``dest``
    name, which repeats across tasks (see module docstring). Keeping the task id
    in the path still lets a re-run of the same task reuse what it downloaded
    last time.
    """
    return cache_root(cache_dir) / EVALUATOR_SUBDIR / task_id
