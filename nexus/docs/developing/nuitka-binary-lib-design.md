# Nuitka 二进制 Nexus 分发设计

## 目标

Nexus 支持生成不携带 `nexus/**/*.py` 源码树的二进制分发包。目标产物是可直接安装的
compiled wheel，内部核心模块由 Nuitka 编译为可 import 的 extension module：

```text
nexus.cpython-312-x86_64-linux-gnu.so
```

第一阶段目标运行环境是 AGS Ubuntu 22.04 + CPython 3.12。不同 CPython ABI 需要分别构建
对应 wheel，例如 Python 3.13 需要 `cp313-cp313` 产物，不能复用 `cp312-cp312` 的 `.so`。

编译后的分发包需要满足：

- `import nexus` 加载 compiled module，而不是源码包。
- 内置 agent、tool、task、provider、executor、client、processor 等能力可正常注册和懒加载。
- `nexus modules`、`nexus modules install`、`nexus build-runtime` 等 CLI 能在 compiled wheel 中工作。
- `main_agent` 可以在二进制分发环境中调用 `str_replace_editor`、`jupyter_code_interpreter` 等工具。
- Docker runtime 在 host Nexus 是 compiled wheel 时，能把 compiled Nexus 复制进 runtime 镜像/容器。
- `inspect.getsource(nexus)` 不能从 compiled module 直接恢复 Nexus 源码。

## 分发形态

推荐交付形态是 manylinux compiled wheel，而不是裸 `.so`：

```text
tencent_nexus_binary-<version>-cp312-cp312-manylinux*.whl
```

wheel 中包含：

- 顶层 `nexus.cpython-312-x86_64-linux-gnu.so`
- Nuitka 生成的 `nexus.pyi`，如果存在
- `tencent_nexus_binary-<version>.dist-info/METADATA`
- `tencent_nexus_binary-<version>.dist-info/entry_points.txt`
- 最小目录资源 `nexus/_bundle/**`

`tencent-nexus-binary` 只是发布到 PyPI 的 distribution name；安装后导入包名仍然是 `nexus`，
CLI 仍然是 `nexus`。

`entry_points.txt` 提供命令入口：

```ini
[console_scripts]
nexus = nexus.cli.main:main
```

因此用户安装 wheel 后可以直接使用：

```bash
pip install tencent-nexus-binary
nexus modules
nexus modules install file_editor --runtime
```

裸 `.so` 适合作为调试或嵌入式产物，但不会自动生成 `nexus` console script。裸 `.so` 场景下如果需要 CLI，
调用方需要额外提供 wrapper，并确保 `PYTHONPATH` 指向 `.so` 所在目录。

## 最小资源目录

compiled wheel 不携带完整源码树，只保留运行和 runtime 镜像构建必须的非代码资源：

```text
nexus/_bundle/deployment/docker/Dockerfile.runtime
nexus/_bundle/deployment/runtime/start.sh
nexus/_bundle/deployment/runtime/gui_stack.sh
nexus/_bundle/deployment/runtime/docker-bootstrap.sh
nexus/_bundle/deployment/runtime/dind-runc.sh
nexus/_bundle/deployment/runtime/flake.nix
nexus/_bundle/deployment/runtime/flake.lock
nexus/_bundle/deployment/runtime/bash/**
nexus/_bundle/pyproject.toml
nexus/_bundle/README.md
```

`nexus/_bundle/pyproject.toml` 用于 binary 模式下读取项目版本和构建 runtime。扩展元数据不通过
`nexus/extensions/**/pyproject.toml` 分发，而是由编译期 manifest 提供。

常规 wheel 和 compiled wheel 共用 `pyproject.toml` 中的显式 `_bundle` force-include 清单；binary builder 通过
`deployment/pypi/stage_wheel_bundle.py` 读取同一份清单，避免两种分发形态的 runtime 资源发生漂移。

默认不打包以下内容：

- 内置扩展源码和扩展 README
- benchmark 数据、示例 JSONL 和测试 fixture
- 开发文档和站点构建文件
- 非 runtime 构建必需的 Dockerfile、本地二进制工具和大型静态资源

如果后续某个 task 在运行时确实依赖额外资源，需要把资源加入 `_bundle` allowlist，并补充对应测试。

## 内置扩展元数据

Nexus 的 module 系统依赖元数据，而不是导入所有 extension 代码。源码模式下，元数据来自：

- 目录模块的 `pyproject.toml`
- 单文件模块的 PEP 723 metadata

compiled wheel 中没有源码目录，因此构建时生成：

```text
nexus/_compiled_module_manifest.py
```

manifest 中的 `COMPILED_MODULES` 保存所有内置 `ModuleMeta` 需要的信息：

- `name`
- `module_path`
- `description`
- `version`
- `base_deps`
- `host_extras`
- `runtime_extras`
- `objects`
- `kind`

Nuitka 编译时显式包含该模块：

```bash
--include-module=nexus._compiled_module_manifest
```

运行 compiled wheel 时，`discover_all_modules()` 优先读取 `nexus._compiled_module_manifest` 并还原成
`ModuleMeta`。因此 `nexus modules` 和 registry discovery 不需要反编译 `.so`，也不依赖源码树。

## Module CLI

compiled wheel 支持完整 module CLI：

```bash
nexus modules
nexus modules install <module>
nexus modules install <module> --runtime
nexus modules install --all
nexus modules install --all --runtime
nexus modules --tui
```

依赖来源是 `ModuleMeta`：

- host install 使用 `base_deps + host_extras`
- runtime install 使用 `base_deps + runtime_extras`

在 compiled wheel 中，内置扩展的 `ModuleMeta` 来自编译期 manifest；源码开发模式下则来自仓库内置
module roots 中的 metadata。

`nexus modules` 发现缺失 host 依赖时会按现有 CLI 语义返回非零退出码，但列表和依赖检查逻辑仍正常工作。

## Runtime 依赖与 Docker Runtime

worker venv 依赖收集需要在三种安装形态下工作：

1. 构建 runtime 时若传入 Nexus wheel，优先从 wheel metadata 读取 module 依赖。
2. 运行在 compiled `nexus.so` 中时，从 `nexus._compiled_module_manifest` 读取 module 依赖。
3. 开发模式回退到源码目录扫描。

Docker runtime provider 在 package 模式下从 `nexus._bundle` 复制 runtime 构建资源；如果 host Nexus 是 compiled
wheel，还会把相邻的 `nexus.*.so` / `nexus.pyi` 一并复制进 runtime build context。这样容器内不会退回到源码包。

关键 runtime extras 示例：

- `file_editor`：`binaryornot`、`cachetools`、`charset-normalizer`
- `jupyter_code_interpreter`：`ipykernel`、`jupyter-client`

## 构建入口

compiled wheel 使用：

```bash
deployment/pypi/build_nuitka_binary_wheel.sh --python 3.12
```

脚本行为：

1. 在 manylinux 容器中选择目标 CPython ABI，例如 `cp312-cp312`。
2. 安装构建依赖、Nuitka 和项目依赖。
3. 生成 `nexus/_compiled_module_manifest.py`。
4. 使用 Nuitka module mode 编译完整 `nexus` 包。
5. 生成只包含 compiled module、dist-info 和最小 `_bundle` 的 raw wheel。
6. 使用 `auditwheel repair` 产出 manylinux wheel。
7. 退出时清理源码树中的临时 manifest 文件。

默认 manylinux 镜像：

```text
quay.io/pypa/manylinux_2_28_x86_64
```

支持不同 Python 版本：

```bash
deployment/pypi/build_nuitka_binary_wheel.sh --python 3.12
deployment/pypi/build_nuitka_binary_wheel.sh --python 3.13
deployment/pypi/build_nuitka_binary_wheel.sh --python cp312-cp312
```

普通源码 wheel 仍使用：

```bash
uv build
```

两套 wheel 构建互不影响。需要把常规 wheel 和 compiled wheel 一起发布到内部 PyPI 时，发布脚本可以在同一个
`dist/` 中生成所有待发布文件：

```bash
TENCENT_PYPI_BUILD_COMPILED=1 \
TENCENT_PYPI_COMPILED_PYTHONS="3.12 3.13" \
deployment/pypi/publish-tencent.sh
```

该流程会先把发布包名临时切换为 `tencent-nexus`，再执行 `uv build` 生成
`tencent_nexus-<version>-py3-none-any.whl`，随后逐个调用 `deployment/pypi/build_nuitka_binary_wheel.sh --out-dir dist`，
并通过 `NEXUS_BINARY_PACKAGE_NAME=tencent-nexus-binary` 生成
`tencent_nexus_binary-<version>-cp<abi>-cp<abi>-manylinux*.whl`，最后由同一次 `uv publish` 发布 `dist/`
下的所有发布文件。两个 distribution 都提供同一个顶层 `nexus` import，使用时二选一安装。

## 构建加速与源配置

manylinux 编译默认使用：

- `NUITKA_JOBS=32`
- `ccache`
- 持久 pip cache
- SJTU AlmaLinux/EPEL/PyPI 镜像源

可配置项：

```bash
NEXUS_BINARY_USE_SJTU_MIRRORS=0 deployment/pypi/build_nuitka_binary_wheel.sh --python 3.12
NUITKA_JOBS=8 deployment/pypi/build_nuitka_binary_wheel.sh --python 3.12
NEXUS_BINARY_CCACHE_DIR=/path/to/ccache deployment/pypi/build_nuitka_binary_wheel.sh --python 3.12
NEXUS_BINARY_PIP_INDEX_URL=https://mirror.sjtu.edu.cn/pypi/web/simple deployment/pypi/build_nuitka_binary_wheel.sh --python 3.12
```

如果镜像源缺少某个 Python 包，构建脚本会回退到官方 PyPI。用户安装 wheel 时也可以采用相同策略：默认使用企业或校园镜像源，
再配置 `extra-index-url=https://pypi.org/simple` 作为兜底。

## 验证

单元测试覆盖：

- manifest record 与 `ModuleMeta` 的转换。
- 只依赖 compiled manifest 时的内置 module discovery。
- runtime worker venv 从 compiled manifest 收集 runtime extras。
- Docker provider 在 binary package 模式下复制 compiled artifacts。

integration 测试覆盖：

- 在不包含源码目录的二进制环境中 `import nexus`。
- `main_agent`、`str_replace_editor`、`jupyter_code_interpreter` 注册可见。
- `file_editor` 与 Jupyter tool 可直接执行。
- `inspect.getsource(nexus)` 无法恢复源码。
- 在真实 Docker runtime 中运行 `main_agent`，顺序调用 Jupyter 和 file editor。

真实 Docker runtime 测试通过环境变量显式开启：

```bash
NEXUS_BINARY_LIB_DIR=/path/to/build/nexus-binary-wheel-cp312-cp312 \
NEXUS_BINARY_DOCKER_RUNTIME=1 \
NEXUS_BINARY_DOCKER_RUNTIME_IMAGE=nexus-runtime:1.1.3 \
pytest tests/integration/test_nuitka_binary_lib_smoke.py -q
```

compiled wheel 安装验证应使用干净 Ubuntu 22.04 + Python 3.12 容器，安装 `/wheels/*.whl` 后验证 CLI、registry
和源码不可恢复行为。
