# 配置文件说明

本目录包含 search_stateful 工具的默认配置文件。

## 文件说明

### testset_name_to_filter.txt
- **用途**: 测试集名称过滤列表
- **格式**: 每行一个测试集名称
- **作用**: 如果页面内容包含这些测试集名称，页面将被阻止访问（防止数据泄露）
- **示例**:
  ```
  GPQA
  HotpotQA
  TriviaQA
  ```

### url_block_list.txt
- **用途**: URL 黑名单列表
- **格式**: 每行一个 URL 片段或域名
- **作用**: 如果搜索结果的 URL 包含这些片段，该结果将被过滤掉
- **示例**:
  ```
  example-blocked-site.com
  spam-domain.net
  /admin/
  ```

## 使用方式

### 方式 1: 使用默认配置（推荐用于开发/测试）
直接使用本目录中的空文件，工具会自动加载。不需要任何额外配置。

### 方式 2: 通过配置文件覆盖（推荐用于生产环境）
在工具配置中指定自定义路径：

```python
tool_configs = {
    'search_stateful': {
        'testset_name_to_filter_file': '/path/to/your/custom/testset_filter.txt',
        'url_block_list_file': '/path/to/your/custom/url_blocklist.txt',
    },
}
```

### 方式 3: 修改默认文件
直接编辑本目录中的配置文件。这些文件会被 git 追踪，适合团队共享的过滤规则。

## 注意事项

1. 配置文件使用 UTF-8 编码
2. 空行会被自动忽略
3. 不支持注释语法（每行都会被当作有效配置）
4. 建议定期审查和更新过滤列表
