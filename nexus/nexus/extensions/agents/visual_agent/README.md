# Visual Agent

`visual_agent` 准备图文输入并委托 `main_agent` 执行推理和工具循环。默认工具为 `jupyter_with_image`，模型可见名称为 `code_interpreter`，支持继续查看工具返回的图片。视频输入仍为 WIP。

## 快速开始

需要支持图片和工具调用的模型、runtime provider，以及安装了 `ipykernel` 和任务所需图像库的沙盒 Python 环境。以下片段使用 `identity` task，将输入和行为配置都放在 `agent_config`：

```yaml
task_name: identity
agent_name: visual_agent
agent_config:
  user_prompt:
    - type: text
      text: 读取图片中的小字，必要时裁剪放大。
    - type: image_url
      image_url:
        url: /data/example.png
  max_iterations: 30
  show_image_dimensions: true
```

输入文件必须在 Nexus 宿主机上可读。`user_prompt` 和 `system_prompt` 保留字面文本，环境变量应在构造配置前解析。运行修改后的 Nexus 时建议设置 `runtime_configs.main.worker_venv: true`。

## Task 组合与结果

搭配 `visual` 时，问题、历史、`upload_workspace` 放在 `task_config`，行为和目标 `workspace` 放在 `agent_config`。搭配 `identity` 时，所有输入放在 `agent_config`：

```yaml
task_name: identity
agent_name: visual_agent
agent_config:
  user_prompt: 分析 workspace 中的文件。
  upload_workspace: /data/project.zip
  workspace: /workspace/project
```

`visual` task 中未设置的 `history` 和 `upload_workspace` 仍以 `None` 传入，会覆盖 `agent_config` 同名字段；这两个输入应放在 `task_config`。

其他 Task 可以调用 `run_agent(user_prompt=..., history=..., upload_workspace=...)`；Agent 不依赖 `visual` task。Runner 优先级为 `agent_config` → Task 参数 → `agent_input_overrides`。

`agent_output` 包含 `final_answer`、`exit_status`、`iterations`、`main_trajectory` 和 `workspace`。图片历史限制仅影响模型请求，不裁剪保存的轨迹。`main_agent_metrics` 可采集内层推理和工具指标，外层委托 span 不会覆盖它们。

`visual` task 默认返回 `score: null`。配置 `task_config.reward_system_router_id` 可启用 reward system，见 [可选评分](../../tasks/visual/README.md#可选评分)；不需要 judge prompt 或独立 judge 模型。

## 媒体输入

| 类型 | 来源 | 行为 |
| --- | --- | --- |
| `image_url` | `image_url.url`：本地路径、Data URI 或 HTTP(S) URL | 本地/Data URI 图片上传沙盒并内联给模型；普通 HTTP URL 交给模型后端读取。 |
| `file` | `file.url`：本地路径或 Data URI；`filename` 仅用于展示 | 上传后改为沙盒路径提示；Data URI 在无法上传时可保留为内联文件。 |
| `video_url` | `video_url.url`：本地路径或 Data URI | WIP：改为沙盒路径提示，由 Python 读取视频帧。 |

保留图文顺序及图片 `detail`。兼容的 `image`/`video` part 会归一化为标准媒体格式。兼容 `file.file_data`，本地来源推荐使用 `file.url`。远端文件/视频需先由调用方解析。无效来源、编码失败、请求的上传失败或上传结果不完整都会中止运行，避免缺图样本进入评分。未配置 runtime 时图片可直接内联；本地文件和视频必须成功上传。

本地媒体、Data URI 解码结果以及上传的 workspace ZIP 均限制为 64 MiB；文件直接内联时同样校验非空、base64 格式和大小。超限在读取或上传前报错。数据集路径应来自可信来源，本功能不提供宿主目录访问隔离。

上传路径为 `/tmp/<来源哈希>/<文件名>`。哈希基于来源字符串，同一输入路径稳定；本地文件保留文件名，同批重名加数字后缀，Data URI 按媒体类型编号并使用 MIME 对应后缀。重复来源只上传一次，但保留各处消息位置。

### COS 图片 URL

Nexus 宿主机下载标准 COS HTTP(S) 图片并转为 Data URI。支持 `myqcloud.com`、`tencentcos.cn` 下的 `cos.<region>` / `cos-internal.<region>` 域名。签名参数原样发送，调用方负责提供有效签名。

相同 URL 去重下载，不继承 HTTP proxy，不跟随重定向；单图限制 60 秒、64 MiB，最多并发 4 个请求。MIME 按 URL 后缀推断，不使用 Pillow 解码或校验内容。网络、空响应和大小超限错误会中止输入准备，错误信息不包含签名。普通 HTTP 图片、COS 文档和视频不走此下载链路。

## Workspace 上传

`upload_workspace` 是一个宿主机目录或 ZIP，`workspace` 是目标目录，默认 `/tmp`。保留目录结构、隐藏文件和空目录；目录中的 ZIP 不递归解压。

已有目录合并，同名文件覆盖。目标必须为绝对 POSIX 路径；拒绝符号链接、特殊文件和 ZIP 路径越界。上传/解压失败中止推理，已完成写入不回滚，临时传输文件会清理。

Workspace 使用 `main` runtime，不受 `upload_images_to_runtime` 控制。Agent 告知模型其路径，但不自动内联所有文件。代码解释器工作目录需单独配置，详见 [visual task](../../tasks/visual/README.md#workspace-路径语义)。

## Agent 配置

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `system_prompt` | `You are a helpful assistant.` | 调用方指令，不额外注入视觉策略。 |
| `max_iterations` | `30` | 正整数；同时受 runner 超时和上下文容量限制。 |
| `tools` | `[jupyter_with_image]` | 非空列表替换默认工具，`null` / `[]` 回退到默认值。 |
| `multimodal_tool_output` | `true` | 保留工具结果中的图片。 |
| `show_image_dimensions` | `false` | 为输入及工具图片添加 `(宽x高 px)`。 |
| `image_history_limit` | `null` | 保留全部图片，或仅向模型发送最近 N 张；原始问题图片无特殊豁免。 |
| `upload_images_to_runtime` | `true` | 上传支持的图片、文件和视频。 |
| `input_image_paths` | `true` | 在消息中提供已上传图片的路径。 |
| `input_image_path_position` | `before` | `before` / `after`：将 `![名称](沙盒路径)` 及尺寸提示放在对应输入图片前 / 后。 |
| `inline_images` | `true` | 内联解析后的图片；关闭时应保留上传和路径提示。 |
| `parallel_tool_calls` | `true` | 并发执行独立调用；同一个 Jupyter 内核仍串行执行代码。 |
| `tool_error_continue` | `true` | 工具调用失败作为 observation 返回给模型并继续，而不是终止整个 run；视觉探索包含大量试错代码。 |
| `openai_client_key` | `main` | 选择模型客户端。 |

输入和工具图片均使用原始像素尺寸，与沙盒中 `Image.open(path).size` 一致，不应用 EXIF 旋转；上传字节保持原样。输入和工具图片统一解析 PNG/JPEG/GIF/BMP/WebP 头部，最多读取 64 KiB；过大的 JPEG 元数据可能导致尺寸提示省略。工具图片通过现有 `post_iteration_hook` 标注；自定义 hook 随后执行，其消息和下一轮请求覆盖仍有效。无法读取的尺寸省略。

VisualAgent 继承 `MainAgentInput`，自动透传所有主 Agent 字段，包括 `tool_search`、`wrap_failed_tool_response_as_tool_error`、`oom_runtime_recovery` 和多 Agent 配置。其余选项沿用 `main_agent`：MCP、工具/schema 覆盖、重试、`warmup_runtime`、工具结果限制、`post_iteration_hook`、`inference_time_timeout`、`observation_time_timeout`、`single_tool_timeout`。超时单位为秒，文字截断不限制图片字节。

设置 `agent_config.input_image_path_position: after` 可将路径提示放到每张输入图片后面，原有文字和多图顺序保持不变。`input_image_paths: false` 时不生成提示；`inline_images: false` 时只保留路径提示，不受前后位置影响。此选项只作用于输入图片，文件/视频提示及工具回图保持原来的格式。

Gateway 使用已注册的 `nexus_main_agent` policy，识别委托主 Agent 的 declared-chain 和 compaction 元数据。VisualAgent 默认开启 `tool_error_continue`（覆盖主 Agent 的默认 `false`；这是对所有 VisualAgent 使用方的全局默认，不只影响 PPT 任务）：工具调用失败、以及未被判定为传输失效的沙盒断连类错误（如 `aiohttp.ServerDisconnectedError` / `ClientOSError`），会作为 observation 返回给模型继续运行；只有被判定为传输失效的错误（`RuntimeTransportError`，重试耗尽后仍然致命）不受此开关影响、始终终止轨迹。Jupyter 内部作为普通工具结果返回的 Python 异常不受此开关影响。需要故障退出标记（`tool_execution_error` / `ENV_ERROR`）的评测或训练侧使用方，请显式设置 `tool_error_continue: false`（或环境变量 `NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE=false`，环境变量优先级最高），并自行权衡环境故障对评分轨迹的影响。

## 代码解释器

注册名为 `jupyter_with_image`，模型可见名和 `tool_configs` 配置键为 `code_interpreter`：

```yaml
tool_configs:
  code_interpreter:
    python_path: /opt/venv/bin/python
    work_dir: /tmp/jupyter_work
```

`python_path` 应指向沙盒中安装了内核及任务依赖的 Python。内核在调用间保持状态。用 `display(...)`、`plt.show()` 或保存新图片文件返回图片；`image.show()` 和覆盖旧文件不保证触发回图。裁剪、OCR 等通过模型生成 Python 完成，默认不启用独立工具。

文件找不到时检查消息中的沙盒路径；缺库时检查沙盒 Python 环境。工具回图未到下一轮模型时，检查 `multimodal_tool_output` 和 `image_history_limit`。
