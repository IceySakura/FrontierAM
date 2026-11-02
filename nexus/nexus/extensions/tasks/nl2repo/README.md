# NL2RepoBench

NL2RepoBench 是一个代码生成评测任务（Nexus Task），用于衡量 LLM Agent 「从自然语言需求到完整代码仓库」的能力。每个实例提供一份 `start.md` 需求文档，Agent 需要在空白环境中从零实现完整的 Python 项目，最终通过项目自带的 pytest 测试套件评估代码正确率。

## 架构概览

任务分为 **推理** 和 **评估** 两个阶段，分别运行在独立的 Runtime 容器中：

```
┌─────────────────────────────────────────────────────────────────────┐
│  Host (nexus run-batch)                                             │
│                                                                     │
│  generator.py  ──→  task.py (入口)                                  │
│                        │                                            │
│                ┌───────┴───────┐                                    │
│                ▼               ▼                                    │
│          ┌──────────┐   ┌──────────┐                                │
│          │ 推理阶段  │   │ 评估阶段  │                                │
│          │ infer.py │   │ eval.py  │                                │
│          └────┬─────┘   └────┬─────┘                                │
│               │              │                                      │
│  ┌────────────▼──┐   ┌──────▼────────────┐                          │
│  │ infer_runtime │   │  eval_runtime     │                          │
│  │ (通用 Python  │   │ (项目基础镜像,     │                          │
│  │  镜像)        │   │  含预装依赖)       │                          │
│  └───────────────┘   └───────────────────┘                          │
│                                                                     │
│  数据流:                                                            │
│  DatasetClient ──签名URL──→ start.md ──→ infer_runtime              │
│  infer_runtime ──zip──→ Host /tmp/nl2repo/{run_id}/workspace.zip    │
│  Host zip ──→ eval_runtime ──pytest──→ score                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 推理阶段 (`infer.py`)

1. 通过 `DatasetClient.sign_urls` 获取 `start.md` 签名 URL并下载
2. 将 `start.md` 上传到 infer_runtime 的 `/workspace/`
3. 运行 CodeAct Agent，根据 `start.md` 生成完整项目代码在`/workspace/`目录下
4. 后处理：删除包管理文件和测试文件 (`post_process.py`)，打包为 `workspace.zip`
5. 将 zip 下载到 Host 本地 `/tmp/nl2repo/{run_id}/workspace.zip`
6. 释放 infer_runtime

### 评估阶段 (`eval.py`)

1. 启动 eval_runtime（使用项目专属的基础镜像，已预装依赖）
2. 从 Host 本地读取 `workspace.zip`，上传到 eval_runtime
3. 解压到 `/workspace/`
4. 按顺序执行 `test_commands`（通常是 `pip install -e .` + `pytest`）
5. 解析 pytest 输出，计算 `success_rate = passed / (passed + failed + errors)`
6. 释放 eval_runtime，清理 Host 本地临时文件

## 数据集格式 (JSONL)

每行一个 JSON 对象，字段说明：

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `instance_id` | string | 是 | 实例唯一标识，如 `aiofiles` |
| `start_md_cos_key` | string | 是 | `start.md` 在数据集中的相对路径，如 `aiofiles/start.md` |
| `test_files` | list[string] | 是 | 需要删除的测试文件/目录列表（推理后处理时移除） |
| `test_commands` | list[string] | 是 | 评估时执行的命令列表（按顺序执行） |
| `test_case_count` | int | 是 | 标注时的测试用例总数，仅供参考；实际执行的用例数可能因 repo 版本更新而增减，以 `pytest_results.real_total` 为准 |
| `eval_image` | string | 是 | 评估阶段容器镜像（含项目依赖） |
| `gold_git_repo` | string | 否 | 标准答案 Git 仓库 URL（mock agent 使用） |
| `gold_git_commit` | string | 否 | 标准答案 Git commit（可选，确保版本一致） |

示例：

```json
{
  "instance_id": "aiofiles",
  "start_md_cos_key": "aiofiles/start.md",
  "test_files": ["tests"],
  "test_commands": ["pip install -e .", "pytest --continue-on-collection-errors tests"],
  "test_case_count": 202,
  "eval_image": "mirrors.tencent.com/hunyuan_yanguan/nl2repobench.eval.aiofiles:1.0",
  "gold_git_repo": "https://github.com/Tinche/aiofiles"
}
```

数据集中的 `start_md_cos_key` 会与 `dataset_namespace`/`dataset_name`（默认 `swe-data`/`NL2RepoBench`）组合，通过 DatasetClient 签名 URL 下载。完整 COS 路径为 `swe-data/NL2RepoBench/{start_md_cos_key}`。

## 运行方法

```bash
cd ${NEXUS_DIR}

# 设置环境变量（见下方完整列表）
export NL2REPO_DATASET_PATH=dataset.jsonl
export OPENAI_API_KEY=xxx
...

# 执行
uv run nexus run-batch \
    -g nexus/extensions/tasks/nl2repo/generator.py \
    -d output.db \
    --max-concurrent 5
```

### 快速测试（单道题）

```bash
bash nexus/extensions/tasks/nl2repo/tests/test.sh
```

使用 `tests/test.jsonl`（含少量实例），用真实 Agent 跑推理 + 评估。

### 评估验证（mock agent）

```bash
bash nexus/extensions/tasks/nl2repo/tests/test_eval.sh
```

使用 `mock_nl2repo_agent` 直接 clone `gold_git_repo` 作为标准答案，跳过模型调用，仅验证评估流程。适用于确认 eval 镜像和 pytest 执行是否正常。

## 输出字段

DB 中每条 run 的 `task_output` 包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `score` | float | 归一化分数 [0, 1]，= passed / (passed + failed + errors) |
| `completed` | bool | 任务是否完成 |
| `pytest_results` | object | pytest 结果详情（见下方） |
| `local_zip_path` | string \| null | Host 本地 zip 路径（eval 后已清理） |
| `error` | string \| null | 错误信息，null 表示成功 |

`pytest_results` 对象：

| 字段 | 类型 | 说明 |
|---|---|---|
| `passed` | int | pytest passed 数量 |
| `failed` | int | pytest failed 数量 |
| `errors` | int | pytest errors 数量 |
| `total` | int | 数据集标注的测试用例总数 |
| `real_total` | int | 实际执行的测试用例总数 (passed + failed + errors) |
| `success_rate` | float | 通过率 = passed / real_total |

## 目录结构

```
nl2repo/
├── README.md              ← 本文件
├── __init__.py             # 包初始化，注册 task
├── task.py                 # 任务入口 + 数据模型（NL2RepoTaskInput/Output）
├── infer.py                # 推理阶段：下载 start.md → Agent → 后处理 → 保存 zip
├── eval.py                 # 评估阶段：加载 zip → pytest → 解析结果 → 清理
├── post_process.py         # 后处理函数（在 Runtime 中执行）
├── generator.py            # batch generator（读取环境变量 + JSONL 生成配置）
│
└── tests/
    ├── test.sh             # 快速测试脚本（真实 Agent）
    ├── test.jsonl          # 测试数据集（少量实例）
    ├── test_eval.sh        # 评估验证脚本（mock Agent）
    ├── test_eval.jsonl     # 评估测试数据集（104 个实例）
    └── mock_nl2repo_agent.py  # Mock agent：clone gold_git_repo 跳过推理
```
