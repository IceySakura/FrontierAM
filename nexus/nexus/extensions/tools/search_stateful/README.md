# Search Stateful Tools

## 概述

这是一个有状态的网页搜索工具集，提供了一套完整的浏览器模拟功能，可以在多次工具调用之间保持状态。该工具集主要用于支持智能体进行迭代式的信息检索任务。

## 核心功能

### 1. 状态管理 (`state.py`)
- **浏览器会话管理**: 为每个请求维护独立的浏览器状态（页面栈、浏览历史）
- **LRU缓存**: 使用 LRU 字典管理多个会话，自动清理过期状态
- **线程安全**: 使用异步锁确保并发访问的安全性
- **防泄露检测**: 
  - 检测页面内容是否包含原始问题（避免问题泄露）
  - 过滤包含测试集名称的页面

### 2. 搜索工具 (`search.py`)
- **功能**: 执行网页搜索并返回结果列表
- **参数**:
  - `query`: 搜索查询字符串
  - `topn`: 返回结果数量（默认10条）
  - `source`: 搜索来源（web/news）
- **特性**:
  - 使用 Google API 后端
  - 支持 URL 黑名单过滤
  - 自动将搜索结果添加到浏览器状态

### 3. 打开链接工具 (`open.py`)
- **功能**: 打开搜索结果中的链接或直接打开URL
- **参数**:
  - `id`: 链接ID（数字）或完整URL（字符串）
  - `cursor`: 页面游标位置
  - `loc`: 起始行号
  - `num_lines`: 显示行数
- **特性**:
  - 支持页面滚动和部分内容查看
  - 自动提取和清理页面内容
  - 检测并阻止包含敏感内容的页面

### 4. 查找工具 (`find.py`)
- **功能**: 在当前页面中查找精确匹配的文本
- **参数**:
  - `pattern`: 要查找的精确模式
  - `cursor`: 要搜索的页面游标
- **特性**:
  - 高亮显示匹配位置
  - 返回上下文信息

### 5. 其他组件
- **后端支持** (`backend.py`): Google API 搜索后端实现
- **页面内容** (`page_contents.py`): 页面内容的结构化表示
- **N-gram检测** (`check_leak_question_n_gram.py`): 基于n-gram的问题泄露检测

## 使用场景

这套工具主要用于构建能够进行复杂信息检索的智能体，典型场景包括：

1. **多轮信息查找**: 智能体可以先搜索，然后打开相关链接，再在页面中查找具体信息
2. **长文档浏览**: 支持分页显示和滚动，适合处理长文档
3. **安全的评测环境**: 内置的防泄露机制确保在评测任务中不会意外泄露问题或答案

## 架构设计

```
SimpleBrowserStateList (全局状态管理器)
    └── SimpleBrowserStateActor (单个会话的浏览器状态)
            ├── pages: dict[url, PageContents]
            └── page_stack: list[url]
```

- 每个 `request_id` 对应一个独立的浏览器会话
- 会话之间完全隔离，互不干扰
- 自动的资源管理和清理

## 配置选项

### API 凭证（必填）
工具需要配置 API 凭证才能使用：

```python
tool_configs = {
    'search_stateful': {
        'user': 'your_user_id',  # 必填：API 用户 ID
        'apikey': 'your_api_key',  # 必填：API 密钥
    },
}
```

### 默认配置
工具自带默认配置文件，位于 `data/` 目录下：
- `data/testset_name_to_filter.txt`: 测试集名称过滤列表
- `data/url_block_list.txt`: URL 黑名单列表

默认情况下，这些文件为空，可以根据需要编辑它们。

### 完整配置示例
```python
tool_configs = {
    'search_stateful': {
        # API 凭证（必填）
        'user': 'your_user_id',
        'apikey': 'your_api_key',
        # 可选配置
        'max_search_results': 10,  # 最大搜索结果数
        'testset_name_to_filter_file': '/path/to/your/testset_filter.txt',
        'url_block_list_file': '/path/to/your/url_blocklist.txt',
    },
}
```

配置项说明：
- `user`: API 用户 ID（**必填**）
- `apikey`: API 密钥（**必填**）
- `max_search_results`: 最大搜索结果数（可选，默认 10）
- `testset_name_to_filter_file`: 测试集名称过滤文件路径（可选，默认使用 data/testset_name_to_filter.txt）
- `url_block_list_file`: URL黑名单文件路径（可选，默认使用 data/url_block_list.txt）

详细配置说明请参考 `data/README.md`。

## 注意事项

1. 需要配置有效的 Google API 凭证
2. 配置文件使用相对路径，便于代码在不同环境下移植
3. 状态会占用内存，建议设置合理的 LRU 容量
4. 默认配置文件会被 git 追踪，可用于团队共享过滤规则
