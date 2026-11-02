# OSWorld 原生任务

只维护六种评测通路：GPT/Claude bare、GPT/Hunyuan Codex、Claude/Hunyuan CC。
本地与平台均由 `task.py` 读取 `task_files`，从 COS 获取题目及 setup/eval 资源。

## 目录职责

| 目录/文件 | 职责 |
|---|---|
| `task.py` | Nexus `os_world` task 注册、setup → agent → evaluator 生命周期 |
| `.env` / `.env.example` | 私有本地配置（不自动加载）/无密钥示例 |
| `generators/` | 六个 batch 入口、共享生成器、配置/client/harness 构建、`osworld_native` run part |
| `data/` | COS 数据集索引、选题、资源预取及缓存路径 |
| `execution/` | setup、命令、CDP/代理、提示词、工具、大沙盒策略、可选结果目录 hook |
| `evaluation/` | 官方评分编排、错误、评分上下文与 getter/metric 注册表 |
| `getters/` | 读取被测桌面状态与参考文件 |
| `metrics/` | 官方指标实现 |
| `prompts/claude_code/` / `assets/codex/` | Harness 指南和模型 catalog |
| `scripts/` | 发布数据准备、single YAML/平台参数导出、结果汇总 |
| `pyproject.toml` / `__init__.py` | 模块依赖和注册 |

AGS Tool 提供桌面服务和应用环境，task 负责 setup、执行和评分。
CC/Codex 通过 `execution_profile: osworld` 选择任务策略；
单元测试位于 `tests/unit/extensions/tasks/osworld/`。

## 发起

```bash
nexus run-single nexus/extensions/tasks/os_world/configs/osworld_gpt_bare.yaml -o /tmp/single.json
nexus run-batch -g nexus/extensions/tasks/os_world/generators/generator_native_gpt_bare.py \
  -d /tmp/batch.db --max-concurrent 1
```

六模式的全部 single/batch 命令与环境变量见 `docs/how-to/osworld-runbook.md`。
