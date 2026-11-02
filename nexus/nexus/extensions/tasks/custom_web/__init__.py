"""Custom Web task for browser-based automation.

This module provides:
- Custom web task execution with Docker-based websites
- Multi-type verifiers (state, retrieval, navigation, action)

Task format follows delivery.jsonl structure:
{
    "ID": "001",
    "task": "任务描述",
    "intentions": ["意图1", "意图2"],
    "classes": ["页面导航类", "状态调整类"],
    "groundtruth": [
        {"type": "state", "expected_states": {...}, "options": {...}},
        {"type": "retrieval", "expected_content": "...", "options": {...}}
    ]
}

Usage:
    from nexus.extensions.tasks.custom_web import (
        run_custom_web_task,
        CustomWebTaskInput,
        CustomWebTaskOutput,
        TaskData,
        ServiceConfig,
    )

    # Create task input
    task_input = CustomWebTaskInput(
        task_data=TaskData(
            ID="001",
            task="去茶颜悦色点2杯幽兰拿铁",
            intentions=["搜索商家", "加入购物车"],
            classes=["页面导航类", "状态调整类"],
            groundtruth=[
                {
                    "type": "state",
                    "expected_states": {"cart": [{"name": "幽兰拿铁", "quantity": 2}]},
                    "options": {"mode": "include"}
                }
            ]
        ),
        service_config=ServiceConfig(
            image="delivery-app:latest",
            port=80,
        ),
    )

    # Run task
    output = await run_custom_web_task(task_input)
"""
