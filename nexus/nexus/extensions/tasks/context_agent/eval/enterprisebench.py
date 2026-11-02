"""EnterpriseBench entry point — batch eval, delegating to Nexus-native orchestration.

This module is a **launcher, not an evaluator**, and is deliberately absent
from :data:`..eval.EVALUATORS`. Every other module in this package implements
``evaluate(model_output, task, llm)``: it grades a rollout's final *text* with
a judge model or a rule, routed by ``metadata.dataset_name``. EnterpriseBench
is not scored that way.

Its score is produced *inside the sandbox* by the delivery's own grader —
``tests/test.sh`` runs ``harbor-rewardkit`` against an OpenCode judge that
reads the mutated world through MCP — and surfaces as ``task_output.score``
from ``context_agent_enterprisebench_native_task``. Registering a judge here would compute a second,
locally-defined number that is not comparable with a published
EnterpriseBench figure, so the official reward is the only one reported.

The benchmark needs a Gongfeng sandbox whose main container *is* the world
image (SuiteCRM is Apache + MariaDB + PHP; zammad adds Elasticsearch).
``context_agent_general`` runs its seven common tools through that runtime,
while the world MCP server exposes only application tools; its Docker ``Core``
tool set is disabled by the dataset builder. So the chain is:

===============  =========================================================
Task             ``context_agent_enterprisebench_native_task``
Agent            ``context_agent_general`` (``nexus.extensions.agents.context_agent``)
Policy model      ``anthropic.claude-opus-4-8`` (Vertex Anthropic)
Dataset builder  ``tasks.context_agent.enterprisebench_native.dataset``
Scoring          OpenCode + ``gpt-5.6-sol`` via Nexus Responses proxy
Batch eval       ``tasks.context_agent.enterprisebench_native.eval`` (re-exported here)
===============  =========================================================

Run it from either path — they are the same code::

    python -m nexus.extensions.tasks.context_agent.eval.enterprisebench \\
        --delivery /path/to/tencent_aug24_delivery \\
        --workdir /path/to/eval_run \\
        --image-tag v2 --memory 10 --max-concurrent 8

Each task declares exactly one world, so the run needs six shared images
(``<prefix>-<world>:<tag>``) rather than three hundred: suitecrm 133,
appflowy 64, base 48, teable 31, suitecrm-zammad 14, zammad 10. ``--worlds``
restricts a run to the images that have actually been pushed, and the
preflight stage fails before any sandbox is spent if one is missing.
"""

from __future__ import annotations

from nexus.extensions.tasks.context_agent.enterprisebench_native.eval import (
    DEFAULT_IMAGE_PREFIX,
    DEFAULT_IMAGE_TAG,
    DEFAULT_MEMORY_GB,
    WORLD_TASK_COUNTS,
    aggregate,
    classify,
    main,
)


__all__ = [
    'DEFAULT_IMAGE_PREFIX',
    'DEFAULT_IMAGE_TAG',
    'DEFAULT_MEMORY_GB',
    'WORLD_TASK_COUNTS',
    'aggregate',
    'classify',
    'main',
]


if __name__ == '__main__':
    raise SystemExit(main())
