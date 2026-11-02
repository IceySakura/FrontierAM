"""Claw-Eval benchmark integration for Nexus framework.

Claw-Eval is an end-to-end AI agent evaluation benchmark with 104 tasks,
15+ mock services, Docker sandbox isolation, and multi-dimensional scoring.
This module provides complete integration with the Nexus framework.

Key differences from PinchBench integration:
- Does NOT use nexus's openclaw agent — uses its own agent loop or any compatible agent
- Provides per-task independent MCP tool servers backed by claw-eval's mock HTTP services
- Multi-dimensional scoring: completion * 0.8 + robustness * 0.2, gated by safety

Usage:
    from nexus.extensions.tasks.claw_eval import (
        ClawEvalTaskInput,
        ClawEvalTaskOutput,
        run_claw_eval_task,
    )

    input_data = ClawEvalTaskInput(
        task_dir="/path/to/claw-eval/tasks/T01zh_email_triage",
        claw_eval_root="/path/to/claw-eval",
    )
    result = await run_claw_eval_task(input_data)
    print(f"Score: {result.score}")
"""
