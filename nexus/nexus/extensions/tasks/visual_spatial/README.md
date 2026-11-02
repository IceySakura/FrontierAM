# Visual Spatial Task

`visual_spatial` 继承 [`visual`](../visual/README.md) task，输入、上传和可选评分完全相同，只额外为 Agent 启用 4 个空间理解工具：

| 工具 | 模型可见名 | 作用 |
| --- | --- | --- |
| `visual/query_locate` | `query_locate` | 按描述定位所有实例，返回 `bbox_2d`、`point_2d` |
| `visual/query_depth` | `query_depth` | 一次查询多个点的米制深度 `depth_m` |
| `visual/query_pose` | `query_pose` | 单图物体朝向，或 `query='camera motion'` 时两图间的相机运动 |
| `visual/query_segment` | `query_segment` | 按描述分割，返回 `polygon_2d`、`area_pixels` |

工具代码在本 task 的 [tools.py](tools.py)，不修改 `visual_agent`。坐标统一为 0–999 归一化范围。

```yaml
task_name: visual_spatial
task_config:
  user_prompt:
    - type: text
      text: 笔记本和音箱哪个离镜头更近？
    - type: image_url
      image_url:
        url: /data/desk.jpg
  # 可选，默认只启用全部 4 个空间工具，不带代码解释器。
  # spatial_tools: [visual/query_locate, visual/query_depth]
  # base_tools: [jupyter_with_image]  # 需要代码解释器时显式加上

agent_name: visual_agent

tool_configs:
  spatial:
    base_url: ${oc.env:SPATIAL_BASE_URL,"http://29.177.195.212"}
    timeout: 150.0
    max_concurrency: 8
```

完整运行配置见 [example.yaml](example.yaml)。

## 评分

评分完全沿用 `visual` task：`reward_system_router_id`、`answer`、`reward_system_extra` 等字段含义相同，默认 `null` 不评分。评分服务只收到原始问题和最终回答，空间工具的中间调用不会进入评分。详见 [visual task 可选评分](../visual/README.md#可选评分)。

## 工具选择

Task 把 `base_tools + spatial_tools`（去重，保持顺序）作为 `tools` 传给 Agent，会覆盖 `agent_config.tools`。`base_tools` 默认为空，即只有空间工具；需要代码解释器等其他工具时在 `base_tools` 里列出。`spatial_tools` 只接受上表 4 个名字，写错会报错。

## 读图方式

工具在 Nexus 宿主进程运行，按模型给出的沙盒绝对路径，通过 `runtime.download_file` 从 runtime 读回图片（默认 `main`，可用 `tool_configs.spatial.runtime` 指定），转成 data URL 后发给空间服务。所以输入图片可以直接使用；在 `base_tools` 中加上代码解释器时，它在沙盒里新生成的文件也可以直接使用。工具只使用本次运行已经拉起的 runtime，不会为了读一张图去启动沙盒。

没有 runtime 时，可以传 base64 data URL，或用 `tool_configs.spatial.image_sources` 显式把路径映射到宿主可读来源。**只有在没有 runtime 时才会直接读宿主机路径**：有沙盒时模型看到的路径都在沙盒里，宿主机上同名路径是另一个文件，读它会把宿主机文件发给服务。路径写错但文件名恰好对上某张已知图片时，工具会报错并提示正确路径，不会静默替换。单张图片上限 64 MiB。

## 服务配置

四个工具共用 `tool_configs.spatial`：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `base_url` | `SPATIAL_BASE_URL` 或 `http://29.177.195.212` | 各工具在其后拼接固定端口和路径 |
| `locate_url` / `depth_url` / `pose_url` / `segment_url` | 无 | 覆盖单个工具的完整地址，也可用 `SPATIAL_LOCATE_URL` 等环境变量 |
| `timeout` | `150.0` | 单次请求超时，秒。必须大于服务端排队超时（locate 当前 120 s），否则排满队被 429 时本地已经先超时，拿不到退避提示 |
| `max_concurrency` | `8` | 进程内并发上限 |
| `runtime` | `main` | 读图使用的 runtime 名 |

请求不走环境代理。服务返回 HTTP 200 但 `ok=false`、网络失败、找不到图片时，工具返回可读的错误让模型改正，不会中止运行。HTTP 4xx（429 除外）返回 `meta.error = 'rejected'` 和服务给出的原因，告诉模型服务在线、是参数有问题；只有 5xx 和网络层失败才说服务可能不可用。使用前确认宿主机能访问服务。

### 背压与重试

服务端队列满时返回 **HTTP 429**（而不是排队到超时）。这不是故障，是主动卸载，所以和"连不上"分开处理：

- 收到 429 时按响应的 `Retry-After` 头等待后**重试一次**；头缺失或不可解析时等 5 s，超过 30 s 的值按 30 s 截断（这么长的值更可能来自链路上的网关而不是服务本身）。
- 被拒那次请求实际花掉的时间和等待时间都从同一个 `timeout` 预算里扣，剩余不足 1 s 就不再重试，重试不会让总耗时超出调用方允许的范围（等本地并发名额的时间不计入）。
- 第二次仍然 429 就放弃，返回 `meta.error = 'rate_limited'` 和 `meta.retry_after`，并明确告诉模型"服务在忙、不是挂了"。继续重试只会加深刚被要求让开的那个队列。

`max_concurrency` 是**每进程**上限。多个 Nexus 进程并行时，服务端看到的是各进程之和，这是触发 429 的实际条件。

### 服务警告

响应里的 `warnings` 会**附在工具输出末尾**（`Service warnings: ...`，最多 5 条）返回给模型，而不是只写日志。有些警告只有发请求的一方能处理——例如字段名写错导致请求被静默降级——模型读不到日志，不回传就等于没有。空结果时同样回传，因为"什么都没找到"正是最需要知道原因的时候。
