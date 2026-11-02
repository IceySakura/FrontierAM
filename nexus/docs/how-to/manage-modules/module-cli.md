# Module CLI 使用指南

`nexus modules` 命令用于查看和管理 Nexus 扩展模块的依赖状态。

## 查看模块列表

```bash
nexus modules
```

输出所有已发现模块的列表，按类别分组，显示每个模块的名称、Host/Runtime 依赖状态和注册对象。

示例输出（截取部分）：

```
Module                        Host      Runtime    Objects

Agent (16)
  base                         ✓           ✓       2 tools
  claude_code                  ✓           ✓       1 agent
  codeact                      ✓           ✓       1 agent + 5 tools
  swe_agent                    ✓           ✓       1 agent + 3 tools

Task (27)
  bfclv4                     2/4 ✗         ✓       1 task
  echo                         ✓           ✓       1 task
  swe                        2/2 ✓         ✓       2 agents + 4 tasks + 3 tools
  tau2                       1/2 ✗         ✓       1 agent + 1 task

Tool (5)
  browser                      ✓           ✓       14 tools
  file_editor                  ✓         3/3 ✓     1 tool

80 modules, 76 OK, 4 missing
```

- **Host** / **Runtime** 列：`✓` 表示依赖全部已安装，`2/4 ✗` 表示 4 个依赖中只安装了 2 个
- **Objects** 列：模块注册的 agent、task、tool 等对象数量

## 安装模块依赖

### 安装指定模块的 host 依赖

```bash
nexus modules install swe_agent
```

安装 `swe_agent` 模块的 base 依赖 + host extras。

### 安装指定模块的 runtime 依赖

```bash
nexus modules install swe_agent --runtime
```

安装 `swe_agent` 模块的 base 依赖 + runtime extras。 通常在构建 Runtime 镜像时使用。

### 安装所有模块的 host 依赖

```bash
nexus modules install --all
```

### 安装所有模块的 runtime 依赖

```bash
nexus modules install --all --runtime
```

在 Dockerfile 中构建 Runtime 镜像时使用，为 Worker Venv 的符号链接提供完整的依赖池。

## 交互式 TUI

```bash
nexus modules --tui
```

启动基于 Textual 的交互式终端界面，提供：

- 按类别（agents、tasks、tools 等）分组浏览
- 可折叠的分类区块
- 查看每个模块的详细依赖和注册对象
- 一键安装缺失依赖

## 自动安装

设置环境变量 `NEXUS_AUTO_INSTALL=1` 可在运行时自动安装缺失的 host 侧依赖：

```bash
NEXUS_AUTO_INSTALL=1 nexus run-single workspace/example.yaml
```

当框架通过 `registry.get()` 懒加载某个模块时，如果发现其依赖未安装，会自动调用 `uv pip install` 安装后重试。

!!! warning "注意"

    `NEXUS_AUTO_INSTALL` 只适合本地开发调试。它不是 Nexus Cloud 的依赖安装机制；生产环境不应依赖运行中现场安装。

## Nexus Cloud 的模块依赖

Cloud source-package 提交会从实际 `RunInput` 解析任务、Agent、LLM Client、Runtime Provider、处理器和
`imports`，并从上传的 Nexus wheel 元数据中提取对应模块的 `base + host` 依赖。Builder 联合解析 Nexus 基础声明、
所选 extras 和模块 Host 声明，优先保留原始 `uv.lock` 中仍兼容的版本，将最终完整依赖集合固定到 `module-host.lock`，
再构建离线 wheelhouse。

Builder 的包索引由部署配置提供，忽略源码中的 `[tool.uv.sources]` 开发来源覆盖；
`[tool.uv.constraint-dependencies]` 中的显式版本约束继续生效。

- Single/Live 分析本次提交的规范化 `RunInput`。
- Batch 只分析 Task List 第一条规范化记录；整个 Batch 必须使用相同 Host 依赖集合。
- 声明允许时可以升级旧锁版本；声明之间存在真实冲突时构建失败。项目显式约束继续生效，原始 `uv.lock` 不被修改。
- image 和 `--config-only` 模式不执行自动补齐，依赖必须预装在 workload 镜像中。

因此，Cloud 用户无需设置 `NEXUS_AUTO_INSTALL`，也无需在 Manager 运行时调用 `nexus modules install`。

## 本地构建模块 wheelhouse

从任务输入、物化、Nexus wheel 到依赖 wheelhouse 的完整操作流程，统一见
[Cloud：提交前本地验证](../../cloud/local-validation.md)。本节仅说明命令参数与报告。

`nexus modules wheel` 使用与 Cloud Builder 相同的 selector 和联合解析逻辑。
运行环境需要 `builder` extra；读取 Task List 时同时使用 `distributed` extra。
完整流程页已在调用中提供这两个 extra。

| 参数 | 含义 |
| --- | --- |
| `--run-input PATH` | 已编译的 RunInput YAML/JSON；不是 `body` Profile 或完整 Compose 文件 |
| `--task-list PATH` | 已物化的 Task List JSONL，支持 v1/v2/v3，只分析第一条规范化记录 |
| `--project-root DIR` | Nexus source-package 项目根目录，默认当前目录 |
| `--lock FILE` | 必填，原始 `uv.lock`；有模块 Host 依赖时作为版本偏好，无模块 Host 依赖时按 frozen 锁构建 |
| `--nexus-wheel FILE` | 必填，本轮构建的精确 Nexus wheel |
| `--output DIR` | 必填，wheelhouse 输出目录 |
| `--extra NAME` | 可重复，额外启用源项目声明的 extra；不是选择 Agent 的参数 |

`--run-input` 与 `--task-list` 必须且只能选择一个。Task、Agent、LLM Client、Runtime Provider、处理器
等模块来自完整 RunInput 中的选择器及 `imports`；仅填写 `task_name` 不会推断出未指定的 scaffold。
整个 Batch 应使用相同 Host 依赖集合，异构任务的处理见
[Cloud Batch 模块依赖](../../cloud/submit-batch.md#模块依赖)。

命令构建 wheels 和最终依赖锁，并在临时 venv 中先离线安装 Nexus 与 extras，再安装模块 Host 依赖，
执行 `uv pip check`、最终锁版本比对以及 Nexus 导入/版本检查；
不把模块 Host 包安装到当前解释器。验证范围是所选模块声明的 Host Python 依赖，不包含沙箱软件或任务执行。

### 输出与报告

| 产物/字段 | 含义 |
| --- | --- |
| `wheels/` | Nexus wheel 和依赖 wheel 集合 |
| `locks/module-host.lock` | 有模块 Host 依赖时生成的完整最终依赖锁，包含基础、extras 和模块依赖；原始 `uv.lock` 不被改写 |
| `module-wheel-report.json` → `representative_item_id` | 实际分析的首条任务 ID；RunInput 模式为 null |
| `module-wheel-report.json` → `selectors` | 从 RunInput 提取的模块选择器 |
| `module-wheel-report.json` → `wheelhouse.module_names` / `module_paths` | 成功解析的模块集合 |
| `module-wheel-report.json` → `wheelhouse.module_host_requirements` | 模块声明的 Host 依赖 |
| `wheelhouse-metadata.json` | wheelhouse 的版本、文件与校验元数据 |

报告在构建成功后生成。失败时按命令错误定位到[物化、构建或离线安装阶段](../../cloud/troubleshooting.md#依赖构建失败)，
不要把当前解释器中已经能 import 视为 wheelhouse 构建通过。

### 从 Harbor Profile 物化后验证

物化和依赖校验的完整流程统一见 [Cloud：提交前本地验证](../../cloud/local-validation.md#1-准备项目和输出目录)。
该页并列提供 Generator、常规 Compose 和 [Harbor Compose](../../cloud/local-validation.md#harbor-compose)，
选择输入后共用物化和 wheel 校验流程；另有
[完整实验 YAML](../../cloud/local-validation.md#已有完整实验-yaml)和
[仅检查 Profile 组合](../../cloud/local-validation.md#只检查-profile-组合)的入口。
物化产物、成功判据、后续提交和本地 E2E 也在同一页说明。

## 模块发现机制

`nexus modules` 命令基于 Nexus 的元数据驱动发现系统。 它扫描以下目录中的 `pyproject.toml` 和 PEP 723 inline metadata：

- `nexus/extensions/agents/`
- `nexus/extensions/tasks/`
- `nexus/extensions/tools/`
- `nexus/extensions/projects/`
- `nexus/extensions/openai_clients/`
- `nexus/runtime/providers/`
- `nexus/runtime/executors/`
- `nexus/trace/processors/`

整个发现过程 **不导入任何 extension 代码**——只读取元数据文件。 详见 [Module 元数据格式](../../explanation/module-metadata.md)。
