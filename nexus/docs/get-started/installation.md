# Installation

Nexus 支持多种安装方式，你可以根据自己的使用场景选择合适的方式。

## 方式一：安装为 pip 包

如果你只是想使用 Nexus 提供的功能，而不需要修改 Nexus 源代码，推荐使用这种方式。

```bash
pip install tencent-nexus --extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
```

特定 module 可能需要额外依赖，可通过 `nexus modules install <module>` 按需安装，或设置环境变量 `NEXUS_AUTO_INSTALL=1` 在运行时自动安装缺失依赖。

!!! note "Python 版本要求"

    Nexus 支持 Python 3.12 - 3.14。 建议使用虚拟环境来隔离依赖。

## 方式二：基于 Nexus 仓库开发

如果你需要修改 Nexus 源代码，或者想要贡献代码，推荐使用这种方式。

### 1. 克隆仓库

```bash
git clone git@git.woa.com:nexus-agent/nexus.git
cd nexus
```

### 2. 创建虚拟环境并安装依赖

```bash
# 使用 uv 创建虚拟环境(推荐)
uv venv --python 3.13

# 以可编辑模式安装 Nexus 及其开发依赖
uv pip install -e ".[dev]"
```

### 3. 关于 pre-commit hook

项目配置了 pre-commit hook ，每次 `git commit` 前会自动运行 `make check` （ruff format ＋ ruff lint ＋ ty typecheck）。 任一检查未通过，提交会被拦截。

`make install-hooks` 会先安装开发依赖，再执行 `pre-commit install` 完成激活。 如果开发依赖已就绪，也可单独执行：

```bash
uv run pre-commit install
```

常用操作：

```bash
# 手动运行检查
make check

# 紧急情况下跳过 hook 提交(不推荐)
git commit --no-verify
```

!!! note "ruff 自动修复"

    `make check` 中的 ruff 会自动修复部分问题(格式化、import 排序等)。 若自动修复改动了已暂存的文件， pre-commit 会拦截本次提交，提示你 `git add` 重新暂存修复后再提交，避免未修复内容被静默提交。

!!! tip "关于 uv"

    uv 是一个快速的 Python 包管理工具。 如果你还没有安装 uv，可以通过以下命令安装:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

## 方式三：在其他项目中引入 Nexus

如果你正在开发自己的项目，需要基于 Nexus 扩展功能(例如添加自定义的 Agent、Tool 或 Task)，可以将 Nexus 作为依赖引入。

### 使用 uv 创建新项目

```bash
# 创建新的 Python 项目
uv init my-nexus-project
cd my-nexus-project

# 添加 Nexus 依赖
uv add tencent-nexus --index tencent=https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
```

### 使用其他包管理工具

如果你使用 pip 或其他包管理工具，只需将 Nexus 添加到你的依赖文件中:

```toml title="pyproject.toml"
[project]
dependencies = [
    "tencent-nexus"，
]

[[tool.uv.index]]
name = "tencent"
url = "https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple"
```

## 验证安装

安装完成后，可以通过以下命令验证安装是否成功：

```bash
nexus --help
```

对于 uv 管理的项目，可以执行：

```bash
uv run nexus --help
```

如果显示帮助信息，说明安装成功。
