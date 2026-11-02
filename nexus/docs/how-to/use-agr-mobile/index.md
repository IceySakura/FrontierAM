# 使用 AGR Mobile

这套用法面向公司内部，目标是通过 Nexus 官方 `run-batch` 入口，批量调度 `agr_mobile` runtime。这里不额外发明新的启动器，保留一层很薄的 generator 适配层。

相关文件：

- `docs/how-to/use-agr-mobile/agr_mobile_batch_config.yaml.example`
- `docs/how-to/use-agr-mobile/agr_mobile_batch_generator.py`

## 1. 设计原则

- `nexus run-batch` 仍然是唯一官方批量入口。
- `agr_mobile_batch_generator.py` 只负责把一个 YAML 配置拆成多条 `BatchItem`，不写死业务流程。
- 真正的任务链定义在 YAML 本身。你可以配成 Agent 驱动验证链路，也可以以后换成别的 task。
- 单条运行不再单独维护 main agent 示例；把 `--max-concurrent` 设成 `1`，本质上就是单条运行。

## 2. 当前默认任务链

当前示例 YAML 默认通过 `nexus.extensions.projects.android_demo` 导出的 `agr_mobile_agent_e2e` 任务。它不是纯流水线，而是“固定准备动作 + AI Agent 可视化验证”的混合模式：

- 连接已有沙箱，或按 `sandbox_template` 新建沙箱
- 获取设备信息并写入 output
- 检查本地 APK 缓存，不存在则下载应用宝 APK
- 上传 APK、安装 APK、授权、拉起应用宝
- 然后把界面控制权交给 `main_agent`
- Agent 只使用 `android/screenshot` 和 `android/device_info` 做只读验证
- Agent 停止后，再统一截图、执行 reset、拉取 `logcat`
- 如果是模板新建沙箱，run 结束后自动释放；如果是已有沙箱，只断连不删除

这样做的好处是：沙箱准备动作保持稳定，真正的页面确认交给 AI Agent，符合 Nexus 作为 GUI Agent 框架的定位。

## 3. 配置文件

先复制模板：

```bash
cp docs/how-to/use-agr-mobile/agr_mobile_batch_config.yaml.example \
  docs/how-to/use-agr-mobile/agr_mobile_batch_config.yaml
```

然后填写你自己的配置。当前模板字段分三类：

- RunInput 业务字段
  - `imports`
  - `task_name`
  - `task_config`
  - `runtime_provider_name`
  - `runtime_provider_config`
  - `agent_name`
  - `agent_config`
  - `openai_clients`
- Batch 拆分字段
  - `batch.mode`: `template_create` 或 `existing_sandbox_ids`
  - `batch.count`: 模板建沙箱模式下生成多少条 run
  - `batch.sandbox_ids`: 已有沙箱列表
- 调度字段
  - 不放在 YAML 内，统一由 `run-batch` 命令行控制

注意：`--max-concurrent`、`-d`、`--fail-fast`、`--retry-failed`、`--timeout` 这些是 Nexus batch 调度参数，不放在 YAML 里。

另外，示例里把 `openai_clients.main.config.client_args.max_retries` 设成了 `1`。原因很实际：如果模型域名不可达，默认重试会把 run 看起来拖成“长时间 running”，不利于排障。

## 4. 运行方式

模板建沙箱批量运行：

```bash
uv run nexus run-batch \
  -g docs/how-to/use-agr-mobile/agr_mobile_batch_generator.py \
  -d /tmp/agr_mobile_batch.db \
  --max-concurrent 5
```

如果你只是想单条验证，把并发改成 `1` 即可：

```bash
uv run nexus run-batch \
  -g docs/how-to/use-agr-mobile/agr_mobile_batch_generator.py \
  -d /tmp/agr_mobile_single.db \
  --max-concurrent 1
```

已有沙箱模式，把 YAML 改成：

```yaml
batch:
  mode: "existing_sandbox_ids"
  sandbox_ids:
    - "sandbox-a"
    - "sandbox-b"
```

然后正常运行：

```bash
uv run nexus run-batch \
  -g docs/how-to/use-agr-mobile/agr_mobile_batch_generator.py \
  -d /tmp/agr_mobile_existing.db \
  --max-concurrent 2
```

清理语义是固定的：

- `template_create` 模式下，task 结束后 provider 会断开 Appium 连接并删除自己创建的 sandbox
- `existing_sandbox_ids` 模式下，task 结束后只断连，不会删除你传入的 sandbox

## 5. Generator 的职责边界

`agr_mobile_batch_generator.py` 现在只做这些事：

- 读取 `agr_mobile_batch_config.yaml`
- 校验 `batch.mode`
- 按 `batch.count` 或 `batch.sandbox_ids` 生成多条 run
- 给每条 run 注入自己的 `runtime_configs.android`
- 给已有沙箱模式补 `conflict_groups`

它不会把“安装应用宝、截图、logcat、Agent 验证”这些流程写死在 generator 里。那部分属于 `task_name` 对应的 task 逻辑；当前默认是 `agr_mobile_agent_e2e`。

## 6. Output 说明

默认示例里：

- `task_config.output_dir` 配的是输出根目录
- generator 会自动把结果整理到 `output_dir/<batch.mode>/<timestamp>/`
- `task_config.apk_dir` 控制 APK 缓存目录

例如：

- `docs/how-to/use-agr-mobile/agr-mobile-output/template_create/20260413_153000/`
- `docs/how-to/use-agr-mobile/agr-mobile-output/existing_sandbox_ids/20260413_153500/`

如果你想手动固定这一层时间戳，可以在运行前设置环境变量 `AGR_BATCH_RUN_ID`。

每条 run 会在 `output_dir/<batch.mode>/<timestamp>/<run_id>/` 下生成至少这些文件：

其中 `reset_result.json` 里会包含 `stopped_packages`，可以直接看到 reset 实际关闭了哪些 app。

- `sandbox_info.json`
- `device_info.json`
- `upload_result.json`
- `install_result.json`
- `grant_result.json`
- `launch_result.json`
- `app_store_before_agent.png`
- `agent_output.json`
- `agent_final_answer.txt`
- `app_store_after_agent.png`
- `reset_result.json`
- `after_reset.png`
- `task_summary.json`
- `logcat.txt`

## 7. 后续扩展

如果你后面不想跑应用宝验证链路，而是想跑别的 task，不需要改 generator，只需要改 YAML：

- 替换 `imports`
- 替换 `task_name`
- 替换 `task_config`
- 如果新 task 需要 agent / model，再补 `agent_name`、`agent_config`、`openai_clients`

这样 generator 仍然保持通用，AGR mobile 只是被当作一种 runtime 使用。
