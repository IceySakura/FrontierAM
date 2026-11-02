# SWE-Any Parse：从 SCIP 索引到实体关系图

Parse 的核心目标不是保存一份完整的 SCIP 副本，而是把 SCIP 提供的文件级语义索引压缩成后续
Select、Mask 和 Eval 可以稳定消费的实体关系图。

整体数据流如下：

```text
仓库源码
  |
  | scip-python index
  v
parse_index.scip                 # sandbox 内的原始 Protobuf 索引
  |
  | load_index + build_graph
  v
NetworkX DiGraph
  |
  | annotate_tests + write_json
  v
parse_graph.json                 # 下载到 host 的正式 Parse 产物
  |
  | 可选的 seed-policy agent
  v
parse_seed_policy.json           # 只影响 Select 的 seed pool，不修改图
```

相关实现：

- `utils/scip/indexer.py`：定位或安装 `scip-python`，执行索引命令。
- `stage/parse/heuristics.py`：读取 `.scip` 并构造实体关系图。
- `utils/graph/`：protobuf-free 的图模型、SCIP symbol 解析和 JSON IO。
- `utils/classify_tests.py`：基于 Python 文件约定重新标注测试节点。
- `runtime_setup.py::_run_parse_blocking`：串联索引、转换、测试标注和落盘。
- `stage/parse/runner.py`：通过 runtime 调用 sandbox 实现，并把 JSON 下载到 host。

## 1. `scip-python` 生成的原始文件包含什么

### 1.1 文件格式

`parse_index.scip` 是一个二进制 Protobuf 文件，顶层消息为 SCIP 协议定义的 `Index`：

```text
Index
├── metadata
├── documents[]
└── external_symbols[]
```

它是语义索引，不是 AST dump，也不是可以直接执行的调用图。它记录“哪个 symbol 在哪个位置被定义或
引用”，以及 indexer 能推导出的 symbol 元数据和关系。

当前使用 `@sourcegraph/scip-python` 生成 Python 索引。执行命令等价于：

```bash
scip-python index \
  --project-name <project_name> \
  --project-version 0.1.0 \
  --output parse_index.scip \
  .
```

### 1.2 `Index.metadata`

`metadata` 描述索引本身：

- `version`：SCIP 协议版本。
- `tool_info.name` / `tool_info.version`：索引器名称和版本。
- `tool_info.arguments`：索引器参数。
- `project_root`：索引根目录 URI，当前通常是 `file:///testbed`。
- `text_document_encoding`：源码文本编码。

最终的 `parse_graph.json` 只保留：

```json
{
  "project_root": "file:///testbed",
  "tool": "scip-python 0.6.6"
}
```

### 1.3 `Index.documents[]`

一个 `Document` 对应一个被索引的源码文件，主要包含：

- `relative_path`：相对 `project_root` 的规范化 POSIX 路径。
- `language`：语言标识。
- `position_encoding`：range 中 character offset 的编码方式。
- `symbols[]`：这个文件内定义的 symbol 元数据。
- `occurrences[]`：这个文件内所有定义、引用、import 等 symbol 出现位置。
- `text`：可选源码正文；indexer 通常不会填，消费者应从文件系统读取源码。

Parse 转换主要消费 `relative_path`、`symbols[]` 和 `occurrences[]`。

### 1.4 `SymbolInformation`

`Document.symbols[]` 的每项描述一个在该文件中定义的 symbol：

- `symbol`：全局唯一的 SCIP symbol 字符串。
- `documentation[]`：docstring、签名或 hover 文本。
- `relationships[]`：与其他 symbol 的语义关系。
- `kind`：class、function、method、field 等细粒度类型。
- `display_name`：显示名称。
- `signature_documentation`：结构化签名文档。
- `enclosing_symbol`：主要用于 local symbol 的 owner 信息。

一个 Python symbol 示例：

```text
scip-python python zoo-demo 0.1.0 `zoo.animals`/Dog#speak().
```

它可以拆成：

```text
scheme          = scip-python
package manager = python
package name    = zoo-demo
package version = 0.1.0
namespace       = zoo.animals
class           = Dog#
method          = speak().
```

SCIP descriptor 后缀表达 symbol 类型：

- `name/`：namespace。
- `name#`：type/class。
- `name().`：function/method。
- `name.`：term/field。
- `(name)`：parameter。
- `[name]`：type parameter。

当前 `scip-python 0.6.x` 经常不填写 `SymbolInformation.kind`，所以转换器没有依赖该字段，而是通过
symbol descriptor 的 `#` 和 `().` 区分 class 与 function。

### 1.5 `Occurrence`

`Document.occurrences[]` 表示一个 symbol 在源码中的一次出现，主要包含：

- `symbol`：该位置对应的 SCIP symbol。
- `symbol_roles`：bitset，例如 definition、import、test。
- `range`：symbol 名称自身的源码位置。
- `enclosing_range`：包围该 occurrence 的完整 AST 范围。
- `override_documentation`：该 occurrence 特有的文档。
- `syntax_kind`：语法高亮分类。
- `diagnostics[]`：该位置上的诊断信息。

旧式 range 编码为：

```text
[start_line, start_col, end_col]            # 单行
[start_line, start_col, end_line, end_col]  # 多行
```

行列号均为 0-based，范围为 half-open `[start, end)`。转换时单行三元组会被规范化为四元组：

```text
[line, start_col, line, end_col]
```

对 definition occurrence，`enclosing_range` 通常覆盖整个函数或类定义，包括 decorator 和 docstring。
这比只记录函数名所在的 `range` 更重要，因为 Mask 后续需要用它定位完整源码范围。

### 1.6 `Relationship`

`SymbolInformation.relationships[]` 可以表达：

- `is_reference`
- `is_implementation`
- `is_type_definition`
- `is_definition`

当前转换只使用 `is_implementation`，生成 implementation 边。普通调用/引用关系不是从
`Relationship.is_reference` 直接生成，而是从 `Occurrence` 的位置推导。

### 1.7 原始索引中存在、但当前图没有保留的信息

为了让后续阶段脱离 protobuf，并控制文件规模，当前转换会丢弃或压缩部分信息：

- 不保留 `Document.text`。
- 不保留每个 reference occurrence 的具体行列号。
- 不保留 diagnostics、syntax kind 和 occurrence override documentation。
- 不保留完整 signature documentation 和 display name。
- 不读取 `Index.external_symbols[]` 的完整元数据。
- 不创建 field、parameter、local variable、module 等节点。
- 不区分“函数调用”“类型引用”“装饰器引用”等 reference 子类型。

## 2. 最终 `parse_graph.json` 的框架

### 2.1 顶层结构

最终文件是一个 protobuf-free、可 round-trip 为 `networkx.DiGraph` 的 JSON：

```json
{
  "info": {
    "project_root": "file:///testbed",
    "tool": "scip-python 0.6.6"
  },
  "nodes": [],
  "edges": []
}
```

设计上使用普通 `DiGraph`，而不是 `MultiDiGraph`：

- 每个 SCIP symbol 对应一个节点。
- 每个 `(src, dst)` 只存在一条物理边。
- 同一对节点之间的多条语义事实保存在 `edges[].infos[]` 中。

这样既可以直接使用 NetworkX 的 predecessor/successor 和连通分量算法，也不会丢失同一节点对之间的
多种关系。

### 2.2 节点

节点分为内部节点和外部节点。

内部节点是在当前仓库中定义的 class 或 function：

```json
{
  "id": "scip-python python demo 0.1.0 pkg/Foo#run().",
  "info": {
    "kind": "function",
    "name": "run",
    "module": "pkg",
    "file": "pkg/foo.py",
    "owner": "scip-python python demo 0.1.0 pkg/Foo#",
    "def_line": 20,
    "enclosing_range": [20, 4, 35, 18],
    "is_test": false,
    "test_evidence": [],
    "documentation": "Run the operation.",
    "external": false
  }
}
```

字段含义：

- `id`：原始 SCIP symbol，也是图内唯一 id。
- `kind`：当前只会是 `class` 或 `function`。
- `name`：symbol 最后一个 descriptor 的名称。
- `module`：namespace descriptors 组成的 dotted module path。
- `file`：仓库相对源码路径。
- `owner`：直接 owner 的 SCIP symbol；顶层实体为 `null`。
- `def_line`：definition occurrence 的起始行，0-based。
- `enclosing_range`：整个定义的源码范围，0-based。
- `is_test`：是否位于测试文件。
- `test_evidence`：测试判定证据，可能包含 `scip_role`、`path`、`name`。
- `documentation`：`SymbolInformation.documentation[]` 拼接后的文本。
- `external`：内部节点固定为 `false`。

当一个被引用的 class/function 不在当前仓库内定义时，会按需创建最小外部节点：

```json
{
  "id": "scip-python python python-stdlib 3.11 builtins/ValueError#",
  "info": {
    "external": true
  }
}
```

外部节点不保证有 file、range、kind 或 documentation。

### 2.3 边

边的框架为：

```json
{
  "src": "<source symbol>",
  "dst": "<target symbol>",
  "infos": [
    {
      "type": "reference",
      "dst_external": false,
      "evidence": "occurrence"
    }
  ]
}
```

当前有三类关系：

#### E01：`enclosing`

```text
parent entity -> child entity
```

例如：

```text
Foo# -> Foo#run().
```

依据是 SCIP symbol descriptor 的 owner 层级，`evidence="descriptor"`。

#### E02：`implementation`

```text
implementation entity -> interface/base entity
```

依据是 `Relationship.is_implementation`，`evidence="is_implementation"`。

目标可能是仓库外的基类或接口，此时 `dst_external=true`。

#### E03：`reference`

```text
containing function -> referenced class/function
```

例如函数 `a()` 内出现对 `b()` 的引用：

```text
a(). -> b().
```

依据是非 definition、非 import 的 `Occurrence`，`evidence="occurrence"`。

这里的 `reference` 是“函数体内使用了目标实体”，不保证一定是运行时函数调用；类型引用、构造器使用或
其他语义引用也可能形成同类边。

### 2.4 测试节点标注

`scip-python` 很少可靠填写 SCIP Test role，因此图构建后会再次执行 `annotate_tests()`。

最终 `is_test` 的严格规则是：节点必须位于测试文件中。测试文件通过以下约定识别：

- 路径包含 `tests/`、`test/` 或 `testing/`。
- 文件名匹配 `test_*.py`、`*_test.py` 或 `conftest.py`。

函数名为 `test_*`、类名为 `Test*` 或 SCIP Test role 只作为 `test_evidence`，不会单独把生产文件中的
节点标为测试。

### 2.5 `parse_seed_policy.json` 与图的关系

seed policy 是 Parse 的另一个产物，但不属于 SCIP 到图的转换。

它只记录 Select 不应该从哪些路径开始采样：

```json
{
  "exclude_globs": ["generated/*"],
  "exclude_reasons": {
    "generated/*": "generated client code"
  },
  "notes": "Prefer hand-written modules."
}
```

被排除的路径仍完整保留在 `parse_graph.json` 中，因为它们可能是一个真实 feature 的依赖，只是不适合
作为 BFS seed。

## 3. 转换是怎么实现的

### 3.1 准备索引器

`runtime_setup._run_parse_blocking()` 在 sandbox 中执行：

1. 查找 `scip-python`。
2. 如果不存在且 `install_scip_python=true`，通过 npm 安装 `@sourcegraph/scip-python`。
3. 安装目录位于 SWE-Any sandbox artifact 目录，不放在仓库内，避免 `index .` 扫描 `node_modules`。
4. 使用 Nexus runtime 的 Node/npm 工具链。
5. 设置 `NODE_OPTIONS=--max-old-space-size=8192`，给大型仓库索引留出 V8 heap。

### 3.2 生成 `.scip`

`run_scip_index()` 在 `repo_dir` 下执行 `scip-python index`，得到：

```text
<repo_dir>/parse_index.scip
```

该文件默认只留在 sandbox。Host 正式下载的是转换后的 `parse_graph.json`。

### 3.3 读取 Protobuf

`load_index()` 使用仓库内提交的预生成 `scip_pb2.py`：

```python
index = scip_pb2.Index()
index.ParseFromString(Path(path).read_bytes())
```

这样 Parse 依赖 protobuf，但后续 Select、Mask、Eval 只依赖 JSON 图模型，不需要加载 SCIP bindings。

### 3.4 Pass 1：收集 definition 并创建实体节点

第一遍扫描所有 `Document.occurrences[]`：

1. 只保留带 definition role 的 occurrence。
2. 规范化 `range` 和 `enclosing_range`。
3. 建立：

```text
symbol -> (relative_path, def_line, enclosing_range, symbol_roles)
```

第二遍扫描所有 `Document.symbols[]`：

1. 解析 SCIP symbol descriptors。
2. 只接受最后一个 descriptor 为 `#` 或 `().` 的 symbol。
3. 分别映射成 `class` 或 `function`。
4. 将 definition occurrence 的文件、行号、完整范围和 test role 合并进节点。
5. 将 documentation 列表拼成一个字符串。

其他 symbol 类型在这个阶段被舍弃。

### 3.5 Pass 2：构建 enclosing 边

对每个内部实体：

1. 从 SCIP symbol 中删除最后一个 descriptor，得到直接 owner symbol。
2. 如果 owner 也是图中的实体，则添加：

```text
owner -> child, type=enclosing
```

纯 namespace/module 不作为实体节点，所以顶层函数不会获得 module enclosing 边。

### 3.6 Pass 3：构建 implementation 边

对每个内部实体的 `SymbolInformation.relationships[]`：

1. 选择 `is_implementation=true` 的关系。
2. 要求目标是内部节点，或者其 symbol descriptor 能识别为 class/function。
3. 添加：

```text
implementation -> base/interface, type=implementation
```

若目标不在当前仓库，`add_edge_info()` 会创建一个最小 external 节点。

### 3.7 Pass 4：构建 reference 边

首先按源码文件收集所有内部 function 的 `enclosing_range`。

然后重新扫描每个文件的 occurrence：

1. 跳过 definition 和 import occurrence。
2. 只保留目标可识别为 class/function 的 occurrence。
3. 根据 occurrence 起点，找到其所在位置最内层的 function `enclosing_range`。
4. 以该 function 作为 source，occurrence symbol 作为 target。
5. 添加 `reference` 关系。

当多个函数范围都包含 occurrence 时，选择行跨度最小、同一行时列跨度最小的范围，即最内层函数。

这一过程把 SCIP 的“某位置引用某 symbol”转换为图上的：

```text
哪个函数使用了哪个实体
```

位于 module 顶层、class body 但不在函数范围内的引用不会生成 reference 边。

### 3.8 重新标注测试节点

图构建完成后，`annotate_tests()`：

1. 遍历所有内部节点。
2. 根据文件路径重新计算 `is_test`。
3. 保留 SCIP role 和名称特征作为 `test_evidence`。

测试节点不会在 Parse 阶段删除；是否保留测试由下游图过滤和 Select 配置决定。

### 3.9 序列化与下载

`write_json()` 把 NetworkX 图写成：

```text
info + nodes[] + edges[]
```

其中 `edges_internal` / `edges_total` 统计的是 `infos[]` 中的关系事实数，不是 NetworkX 物理边数量。

Sandbox 返回图路径和节点/关系计数。`stage/parse/runner.py` 再将：

```text
<sandbox swe_any dir>/parse_graph.json
```

下载到：

```text
<output_dir>/<repo_name>/parse_graph.json
```

## 4. 当前边界和已知限制

- 目前只支持 Python；其他语言需要对应 SCIP indexer 和 symbol 语义适配。
- 当前转换器读取 deprecated `Occurrence.range` / `enclosing_range` 整数数组，尚未读取新的 typed range
  字段；升级到只写 typed range 的 indexer 前需要补兼容。
- `reference` 是静态语义引用，不是精确的运行时 call graph。
- 动态调用、反射、字符串导入和 monkey patch 通常无法由 SCIP 完整恢复。
- module-level 引用不会形成 reference 边。
- 外部节点只保留 `external=true`，不会消费 `Index.external_symbols[]` 中更完整的文档。
- 同一 `(src, dst)` 的多次 occurrence 会成为 `infos[]` 中的多条事实，但当前不保留每次 occurrence 的位置。
- `build_graph(graph_path=...)` 作为库函数时会复用已存在的 JSON；正式 Parse runtime 则始终从新生成的
  `.scip` 构图，批次重跑由 Nexus batch DB 的 resume 语义控制。
