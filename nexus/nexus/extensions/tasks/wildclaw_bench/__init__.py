"""WildClawBench benchmark integration for Nexus framework.

WildClawBench is an end-to-end benchmark for evaluating AI agents on 60
real-world tasks across 6 categories (Productivity Flow, Code Intelligence,
Social Interaction, Search & Retrieval, Creative Synthesis, Safety Alignment).

Each task runs inside a Docker container with an OpenClaw agent, and is graded
by executing embedded Python grading code within the container — exactly
replicating the original WildClawBench pipeline.

Usage:
    from nexus.extensions.tasks.wildclaw_bench import (
        WildClawBenchTaskInput,
        WildClawBenchTaskOutput,
        run_wildclaw_bench_task,
        TaskLoader,
    )

    # Run a single task
    input_data = WildClawBenchTaskInput(
        task_file="/path/to/01_Productivity_Flow/task_1_arxiv_digest.md",
    )
    result = await run_wildclaw_bench_task(input_data)
    print(f"Score: {result.score}")
"""
