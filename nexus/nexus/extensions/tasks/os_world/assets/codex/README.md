# Codex OSWorld 内置资源与覆盖规则

这些资源仅用于 `codex_native_bridge` 的 `execution_profile: osworld`。
Claude Code 的 `guide_version: v2` 属于另一套指南，不适用于 Codex。

## 简写入口

```yaml
agent_name: codex_native_bridge
agent_config:
  execution_profile: osworld
  execution_options:
    model: gpt-6-astra
    version: 0.154.0
    guide_version: v1
    model_catalog_preset: gpt6_catalog_v8
    context_window: 262144
    max_output_tokens: 64000
    context_token_margin: 4096
    auto_compact_token_limit: 194048
```

- Codex `guide_version: v1` → `guide_codex_linux.md`。
- `model_catalog_preset: gpt6_catalog_v8` → `gpt6_catalog_v8.json`。
- `model_catalog_preset: gpt56_sol_catalog` → `gpt56_sol_catalog.json`，
  搭配 `model: gpt-5.6-sol`。
- 路径相对于安装包资源目录解析，不依赖 cwd，也不读取本机环境变量。
- 原始 catalog 是能力模板；运行时使用 OSWorld 公共预算覆盖其窗口和压缩阈值，
  再执行 CLI 版本适配与 classic-tools / local-compaction 校验。

## 覆盖顺序

两类资源独立解析，不通过 guide 选择 catalog：

1. 非 null 的 `guide_text` 优先于 `guide_version`。空字符串也视为显式内容。
2. 非 null 的 `model_catalog_json` 优先于 `model_catalog_preset`。
   接受原始 JSON 字符串和 JSON 对象；不与内置 catalog 合并。
3. 显式内容存在时，对应选择器不参与加载。无效显式 catalog 会直接报错，
   不会偷偷回退到内置配置。省略/null 内容时才使用选择器。
4. 两者都未提供时报必填错误，避免缺失配置被隐式默认值掩盖。
   选择器只接受上述注册名，不接受任意文件路径。
5. `context_window`、`max_output_tokens`、`context_token_margin` 仅用于
   **内置 catalog 展开**，默认分别为 262144、64000、4096。
   未指定压缩阈值时计算 `window - output - margin`；显式阈值不能侵占预留预算。
   `max_output_tokens` 在此只用于预算计算，不替代模型 client 的输出上限配置。
6. 显式 catalog 的窗口等字段不受上述 preset 预算字段改写；
   `auto_compact_token_limit` 是 CLI 参数，自定义实验需保证它与 catalog 一致。
7. 无论来自显式内容还是 preset，都保留既有的 CLI 版本适配
   （0.153.4 使用 null tool mode，0.154.0 使用 direct）、
   local-compaction 限制及最终 guide/catalog hash 审计。

完整 YAML 与自定义实验可以直接填写 `guide_text`、
`model_catalog_json`。`catalog_source_sha256` 通常应省略，让 adapter 计算；
改变 catalog 后应重新计算摘要。

Generator 的 `CODEX_GUIDE_FILE` / `CODEX_MODEL_CATALOG_JSON` 自定义文件入口保留。
它们在 runner 侧读取后以显式内容传入，因此遵循相同的最高优先级。
