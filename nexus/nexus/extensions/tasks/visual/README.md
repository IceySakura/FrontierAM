# Visual Task

`visual` 定义本次视觉任务的输入：问题和图文、视频、文档内容，以及历史消息和待上传的目录或 ZIP，然后将输入副本交给配置的 Agent。输入上传由 Agent 负责；启用评分时，Task 会为评分服务读取本地图片。
通常搭配 `visual_agent` 使用。默认不在 Nexus 中评分，`task_output.score` 为 `null`；设置 `reward_system_router_id` 后由 reward system 评分。回答和轨迹仍由 runner 保存在 `agent_output`，Agent 上传后的目标路径保存在 `agent_output.workspace`。

```yaml
task_name: visual
task_config:
  user_prompt:
    - type: text
      text: 请结合图片和 workspace 中的资料进行分析。
    - type: image_url
      image_url:
        url: /data/example.png
  history: []
  upload_workspace: /data/project.zip

agent_name: visual_agent
agent_config:
  workspace: /workspace/project
  max_iterations: 30
  tools:
    - jupyter_with_image
  show_image_dimensions: true
```

完整运行配置见[图片输入示例](example.yaml)；需要 workspace 时，添加上面的 `upload_workspace` 和 `workspace` 字段。
设置 `VISUAL_AGENT_IMAGE`、`MODEL_API_KEY`、`MODEL_BASE_URL`、`MODEL_NAME` 后运行：

```bash
uv run nexus run-single nexus/extensions/tasks/visual/example.yaml -o /tmp/visual-output.json
```

## 输入和职责

| task_config 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `user_prompt` | 必填 | 文本或内容 part 列表，保留图文顺序及 `file.url`、`video_url` 等扩展字段。 |
| `history` | `null` | 已有对话，角色限 `user`、`assistant`、`tool`，非空时以 `assistant` 或 `tool` 结束。 |
| `upload_workspace` | `null` | 一个本地目录或 ZIP 路径，省略时不上传。 |

Task 定义本次问题和上传来源，Agent 负责将来源目录的内容复制或解压到自己的 workspace。
任务输入与校验都在 [task.py](task.py)；Task 复用 VisualAgent 的媒体资源函数，执行仍通过标准 Agent runner。
Task 使用框架原有的 `run_agent(user_prompt=..., history=..., upload_workspace=...)`，不传目标目录；目标由 `agent_config.workspace` 决定，默认 `/tmp`。
Agent 通过 `WithUserPrompt`、`WithHistory` 和资源模块的 `WithWorkspace` 接收参数，不修改 core。

runner 的合并优先级是 `agent_config → task 传参 → agent_input_overrides`。
Task 省略 `history`/`upload_workspace` 时传 `None`，表示本次没有历史或上传。Task 不定义或覆盖 workspace 目标目录。
Codex 可接收文本问题，但仍需配置 `working_dir`；上传与多模态能力由具体 Agent 实现，不会自动获得。

`task_config.user_prompt` 保留字面内容，不做 OmegaConf 插值；`upload_workspace` 支持 `${oc.env:WORKSPACE_SOURCE}`。
历史消息沿用 runner 现有插值规则，也不会自动上传其中的媒体引用。

搭配 `visual_agent` 时，图片 part 的 `image_url.url` 可以直接使用标准 COS HTTP/HTTPS 链接，包括内网端点和签名 URL。下载、图片编码和沙盒上传由 Agent 完成，Task 无需预先下载；普通 HTTP 图片链接仍由模型后端读取。详见 [COS 图片输入](../../agents/visual_agent/README.md#cos-图片-url)。

## 可选评分

评分配置属于 `task_config`，不会传给 Agent。`reward_system_router_id` 默认为 `null`，此时不加载评分客户端、不读取评分图片，也不调用评分服务；输出为 `{"score": null, "evaluation_details": null}`。若交给 HyTuner 在外部评分，必须启用 `enable_async_reward` 对应的 reward loop worker 配置；未配置异步评分时，`None` 可能被下游填成 0，不会自动调用评分服务。

```yaml
task_config:
  user_prompt: 图片中共有几个物体？
  reward_system_router_id: MultimodalReward-GeneralV2-0
  answer: "3"
  reward_system_extra: {}  # 路由需要的元数据
  public_test: {}          # 路由需要的公开测试
```

| 评分字段 | 默认值 | 说明 |
| --- | --- | --- |
| `reward_system_router_id` | `null` | `class1-class2-router_version`；非空时在 Nexus 内评分。请使用实际已部署的路由。 |
| `answer` | `""` | 参考答案，对应 reward system 的 `ref_answer`；不会暴露给 Agent。 |
| `reward_system_extra` | `null` | 透传到请求的 `extra`，例如 checklist 或评分模式。 |
| `public_test` | `null` | 透传到请求的 `public_test`。 |
| `reward_system_strip_multimodal_content` | `null` | `true` 对所有路由只保留文字，也可填需去掉多模态内容的路由 ID 列表。 |
| `reward_system_include_trajectory` | `false` | `true` 时把 Agent 产生的轨迹（工具调用和返回）按原顺序送评分；实际张数和消息数记录在 `evaluation_details.tool_images_sent` / `trajectory_messages_sent`。 |
| `reward_system_max_tool_images` | `0` | 需同时开启 `reward_system_include_trajectory`，否则校验报错：轨迹中的工具图只保留最近 N 张，更早的换成文字占位；`0` 不保留。 |

启用评分前运行 `uv run nexus modules install visual` 安装评分所需的 Polaris 客户端，并设置 `NEXUS_REWARD_SYSTEM_TOKEN`。实验名可用 `NEXUS_REWARD_SYSTEM_EXP_ID` 配置；服务发现与请求复用 Nexus 现有的 `RewardSystemClient`。客户端在 Agent 执行前创建，凭据缺失或超时配置无效会立即报错。默认只请求一次、超时 180 秒；可用 `REWARD_PLATFORM_TIMEOUT` 设置有限的正秒数。评分输入准备错误和程序错误直接抛出，不转成未判分值。

评分请求参照 hytuner 非 Agent 任务：空 system 消息、原始历史和问题、最终回答；默认不包含工具中间轮次，开启 `reward_system_include_trajectory` 后追加 Agent 生成的消息。保留图文顺序，本地图片转为 Data URI，HTTP(S)/COS 图片链接和已有 Data URI 原样传入。服务必须能读取这些远端链接；开启评分时，本地视频和文档路径会在任务校验阶段直接报错，需自行提供 Data URI/HTTP(S) 引用，或明确配置文字评分。workspace 内容不会自动附加到评分请求。

轨迹评分要求 Agent 输出非空 `main_trajectory` 和有效的 `generated_message_start`（visual agent 自动提供；直接使用 main agent 不提供这些字段），并配置非空 `reward_system_router_id`。缺少轨迹或边界、工具调用与返回不配对时直接报错；只生成最终回答、没有工具调用是合法情况。预算状态和压缩恢复提示不进入评分。发生上下文压缩（`trajectory_compacted=true`）时暂不支持轨迹评分，会明确报错；请关闭轨迹评分或禁用压缩。

图片上限只覆盖 `role=tool` 的图片，按图片数量计数；原始问题、历史以及自定义 hook 注入的 user/assistant 图片不受该上限约束，`tool_images_sent` 也只统计工具图。开启纯文字评分时统一移除图片，不插入省略占位。中间 assistant 消息移除独立的 `reasoning_content` 字段，并按 `THINK_END_TOKEN` 清除文本中的思考前缀。

最终回答优先取 `agent_output.final_answer`，其次取末条 assistant 消息或 Agent 的文本输出，并移除 `</think>` 前的推理内容。`THINK_END_TOKEN` 可覆盖结束标记。

按 `ReasoningReward` 约定处理退出结果：推理/工具执行等环境异常返回 `-100`（未判分），工具格式错误和被截断的轨迹返回 `0`，均不调用评分服务。评分网络异常、非有限值或不高于 `-99` 的不可用分数返回 `-100`，通过现有 span 属性记录 `REWARD_ERROR` 和评分耗时；正常分数写入 `task_output.score`，诊断写入 `evaluation_details`。未配置路由时始终保持 `score=null`。

## Workspace 路径语义

`workspace` 是一个目标目录，`upload_workspace` 是一个来源路径，不再接受 `source/target` 列表，也没有 `extract` 参数。

| upload_workspace | 行为 |
| --- | --- |
| 省略或 `null` | 跳过上传，不因默认 workspace 而创建 runtime。 |
| 本地目录 | 将内容按相对路径复制到 workspace，保留子目录、隐藏文件和空目录，不额外套源目录名。 |
| ZIP 文件 | 自动在沙盒内解压到 workspace，保留包内层级，不自动剥掉顶层目录。 |

例如 `/data/project/a.txt` 上传到 `/workspace/project` 后为 `/workspace/project/a.txt`；
ZIP 内的 `project/a.txt` 解压到 `/workspace` 后为 `/workspace/project/a.txt`。
目录内的 ZIP 保持为文件，不递归解压；其他普通文件不能单独作为上传来源。

来源必须在执行 Nexus 的机器上可读，相对路径基于进程工作目录，支持 `~`。
目标必须是绝对 POSIX 目录路径；已有目录合并、同名文件覆盖，其他文件保留。符号链接、特殊文件和 ZIP 越界路径会报错。
上传或解压失败会中止推理，已完成的写入不自动回滚。临时传输包在操作结束后清理。

上传使用 `main` runtime，与 `upload_images_to_runtime` 开关独立。Agent 上传成功后在 prompt 中告知 workspace 路径；
`agent_output.workspace` 返回一个目录字符串。此目录是资源根目录，代码解释器的执行目录仍按其 `work_dir` 配置决定。
需要直接看图仍使用 prompt 中的图片 part，不会自动内联 workspace 中的所有图片。

实现位于 [resources/workspace.py](../../agents/visual_agent/resources/workspace.py)，入口为 `upload_workspace(runtime, source_path, workspace='/tmp')`。
目录通过临时 ZIP 传输后展开，最终内容与目录递归复制一致；ZIP 直接上传解压。建议配置 `runtime_configs.main.worker_venv: true` 使用当前辅助代码。

## 配置与调用

`visual + visual_agent` 在 `task_config` 提供问题、历史和 `upload_workspace`，在 `agent_config.workspace` 设置目标。
`identity + visual_agent` 将所有参数直接放在 `agent_config`，`identity` 实现不变。

其他 Task 也可以调用：

```python
output = await task_input.run_agent(
    user_prompt=task_input.question,
    upload_workspace=task_input.workspace_source,
)
```

从旧配置迁移时，将原先的 `source` 改为 `upload_workspace`，将 `target` 改为 `agent_config.workspace`，删除列表包装和 `extract`。
需要合并多份资料时，先在本地组织成一个目录或 ZIP 再上传。
