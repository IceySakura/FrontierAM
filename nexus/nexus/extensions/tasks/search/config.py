"""YAML-based configuration for the Search benchmark runner.

``SearchConfig`` is a thin extension of :class:`nexus.runner.single.RunInput`:
everything the single-task runner already understands (``imports``, ``task_name``,
``task_config``, ``tool_configs``, ``runtime_provider_*``, ``agent_name``,
``agent_config``, ``openai_clients``, ``processors``, ``runner``) is inherited
unchanged. Search-runner-specific additions are ``benchmarks`` and ``workspace_path``.

Relationship with ``run-single`` YAML::

    SearchConfig == RunInput + {benchmarks: {name: parquet_path}, workspace_path: path}

Anything valid in a ``run-single`` YAML is valid here, plus ``benchmarks`` and ``workspace_path``.
OmegaConf interpolation (``${oc.env:FOO}`` / ``${oc.env:FOO,default}``) is
inherited from ``RunInput.from_dict`` — no custom placeholder syntax.

Search-specific values that used to live in ``prompt`` / ``toolset`` /
``evaluation`` are no longer top-level YAML sections; they belong on the
underlying task/agent inputs:

* ``prompt.system_prompt`` → ``agent_config.system_prompt`` (supports
  ``{today_date}`` and ``{finish_instruction}`` placeholders, resolved per row
  by the generator).
* ``toolset.versions[X] = [...]`` → ``agent_config.tools: [...]``
  (explicit tool-name list for the agent).
* ``evaluation.*`` → ``task_config.{eval_mode, enable_summary, run_oracle,
  training_length_exceeded}`` (YAML values are defaults; parquet rows override).
"""

from __future__ import annotations

import logging

from pathlib import Path

from pydantic import Field

from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)


class SearchConfig(RunInput):
    """RunInput + a benchmark registry.

    ``benchmarks`` maps a benchmark name (used as sub-directory / DB filename /
    log label) to the absolute Parquet dataset path consumed by the generator.
    """

    benchmarks: dict[str, str] = Field(default_factory=dict)
    workspace_path: Path | None = None


def load_search_config(path: str | Path) -> SearchConfig:
    """Load and validate a :class:`SearchConfig` from a YAML/JSON file.

    Delegates to :meth:`RunInput.from_file` so OmegaConf env-var interpolation
    and free-text-field escaping behave identically to the single-task runner.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f'Search config not found: {config_path}')

    # Reuse RunInput's loader: OmegaConf resolution + free-text escaping + pydantic validation.
    cfg = SearchConfig.from_file(config_path)
    logger.debug('Loaded search config from %s', config_path)
    return cfg


def load_search_config_if_exists(path: str | Path | None = None) -> SearchConfig | None:
    """Load config from *path* or the ``SEARCH_CONFIG_PATH`` environment variable.

    Returns ``None`` when no path is provided/set or the file is missing, so the
    env-var-only workflow remains the default.
    """
    import os

    if path is None:
        path = os.getenv('SEARCH_CONFIG_PATH')
    if not path:
        return None

    p = Path(path)
    if not p.is_file():
        logger.debug('Search config path set but file not found: %s', p)
        return None

    return load_search_config(p)


__all__ = [
    'SearchConfig',
    'load_search_config',
    'load_search_config_if_exists',
]
