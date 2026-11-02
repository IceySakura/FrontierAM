# 六模式的数据与导出工具

这些脚本不是额外的 agent/harness 启动通路。

- `export_single.py`：调用 `generators.generator_native.single_config()` 导出完整 YAML；
  `--all-modes` 一次导出六模式，`task_id` 和 `task_files` 同步生成。
- `export_platform_dataset.py`：默认调用同一原生 generator，将配置拆为逐题 JSONL、
  数据集公共参数与模型 client 服务表单，不输出已解析凭据。
- `build_osworld_dataset.py`：发布者从显式 `OSWORLD_EXAMPLES_ROOT` 打包
  `task.json`、setup/evaluator 附件；**唯一需要原始 checkout 的环节是数据发布**，
  不是本地或平台评测。
- `score_osworld.py`：离线汇总 batch `results.jsonl`，不重跑 evaluator。

```bash
python -m nexus.extensions.tasks.os_world.scripts.export_single \
  --all-modes --task-id TASK_ID -o /tmp/osworld-six
python -m nexus.extensions.tasks.os_world.scripts.export_platform_dataset \
  -o /tmp/dataset.jsonl --eval-set-out /tmp/eval-set.json --client-form-out /tmp/client-form.json
```

后一个命令使用 `OSWORLD_MODEL_FAMILY` / `OSWORLD_HARNESS` 选择模式。
环境变量和平台字段说明见 `docs/how-to/osworld-runbook.md`。
