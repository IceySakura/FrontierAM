"""PinchBench benchmark integration for Nexus framework.

PinchBench is a benchmark for evaluating AI agents on real-world coding and
tool-use tasks. This module provides complete integration with the Nexus
framework for running PinchBench tasks.

Usage:
    from nexus.extensions.tasks.pinchbench import (
        PinchBenchTaskInput,
        PinchBenchTaskOutput,
        run_pinchbench_task,
        TaskLoader,
    )

    # Run a single task
    input_data = PinchBenchTaskInput(
        task_file="/path/to/task_00_sanity.md",
        workspace_path="/workspace",
    )
    result = await run_pinchbench_task(input_data)
    print(f"Score: {result.score}")

    # Load all tasks from a directory
    loader = TaskLoader(Path("/path/to/tasks"))
    tasks = loader.load_all_tasks()
"""
