# Module 元数据格式

每个 extension module 通过标准元数据格式声明其依赖和对象注册。

## 三层依赖模型

模块依赖分为三层，host 和 runtime 环境各自只安装需要的部分：

| 层 | 声明位置 | 安装在 host | 安装在 runtime |
|---|---|---|---|
| **Base（共享）** | `dependencies` | ✓ | ✓ |
| **Host Extras** | `optional-dependencies.host` | ✓ | ✗ |
| **Runtime Extras** | `optional-dependencies.runtime` | ✗ | ✓ |

- **Host 安装集** = `dependencies` + `host` extras
- **Runtime 安装集** = `dependencies` + `runtime` extras

## 目录模块 — `pyproject.toml`

```toml
[project]
name = "swe_agent"
version = "0.1.0"
dependencies = []                    # base 依赖（host 和 runtime 都需要）

[project.optional-dependencies]
host = []                            # host 侧独有依赖（orchestrator 上需要）
runtime = []                         # runtime 侧独有依赖（沙箱容器内需要）

[tool.nexus]
agents = { swe_agent = "nexus.extensions.agents.swe_agent.agent" }
run_parts = { swe_agent = "nexus.extensions.agents.swe_agent.run_parts" }
dataset_sources = { t3-harbor = "nexus.extensions.harbor_bridge.dataset_source" }

[tool.nexus.tools]
"swe_agent/bash" = "nexus.extensions.agents.swe_agent.tools"
"swe_agent/str_replace_editor" = "nexus.extensions.agents.swe_agent.tools"
```

## 单文件模块 — PEP 723 inline script metadata

```python
# /// script
# dependencies = []
#
# [optional-dependencies]
# host = []
# runtime = ["playwright>=1.57.0"]
#
# [tool.nexus.executors]
# create_browser_session = "nexus.runtime.executors.browser"
# browser_goto = "nexus.runtime.executors.browser"
# ///
```

## 规则

- `dependencies` = 共享基础依赖（host 和 runtime 都需要）
- `optional-dependencies.host` = host 侧独有依赖（orchestrator 上需要，runtime 不安装）
- `optional-dependencies.runtime` = runtime 侧独有依赖（沙箱容器内需要，host 不安装）
- `[tool.nexus]` 映射 registry 类型 → `{object_name: import_path}`，用于 lazy 注册
- `run_parts` 注册可复用的 partial `RunInput` 配置对象，供 `nexus run --use ...` 解析
- `dataset_sources` 注册 URI scheme Adapter，供 `nexus dataset` 和 `nexus run --data ...` 解析
- TOML inline tables（`{ k = "v" }`）**不能**跨行 — 多行场景使用 `[tool.nexus.tools]` 子表
- 所有对象注册使用 `registry.add(name, import_path)` — discovery 时不导入 extension 代码

## 模块位置

模块必须位于某个 module root 的**直接子目录**，即 `nexus/<root>/<module>/`。root 列在
`nexus/core/module_meta.py::MODULE_ROOTS` 中，其中 `extensions` 本身也是一个 root，因此
`nexus/extensions/harbor_bridge/` 这类顶层扩展模块与 `nexus/extensions/agents/xxx/` 一样会被
自动发现，无需在 core 里登记名字。

只有恰好一层的目录会被当作模块，所以模块自己携带的 `pyproject.toml`（例如
`nexus/extensions/<module>/configs/pyproject.toml`）不会被误认为独立模块；源码扫描和 wheel
扫描使用同一套规则。

## 安装依赖

- `nexus modules install <module>`：安装指定 module 的 host 侧依赖（base + host extras）
- `nexus modules install <module> --runtime`：安装指定 module 的 runtime 侧依赖（base + runtime extras）
- `nexus modules install --all`：安装所有 module 的 host 侧依赖
- `nexus modules install --all --runtime`：安装所有 module 的 runtime 侧依赖（Dockerfile 使用）
- 设置 `NEXUS_AUTO_INSTALL=1` 可在运行时自动安装缺失的 host 侧依赖
