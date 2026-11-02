# Single Turn system prompt 端到端验证

使用真实模型验证 `RunInput → identity task → single_turn → 模型服务`：单独的
`system_prompt` 前置，原始 `prompt` 中的 system、user、assistant 和续聊消息保持顺序。
模型需要返回分别来自两条 system 消息和此前 assistant 消息的随机标记。

`NEXUS_SINGLE_TURN_E2E_CONFIG` 指向已有 RunInput YAML；测试只读取其中的
`openai_clients.main`，支持原有环境变量引用。所选客户端的依赖需要已安装。
可使用 `workspace/local-configs/config-runinput.yaml` 等本地配置，不提交真实凭据。

```bash
NEXUS_SINGLE_TURN_E2E_CONFIG=/path/to/config-runinput.yaml \
  uv run pytest tests/e2e/test_single_turn_system_prompt.py -m e2e -q -s
```

结果保存在 gitignored 的 `workspace/single-turn-system-prompt-e2e/result.json`，
包含合成测试消息、模型回答和断言使用的标记，不包含访问凭据。
可用 `NEXUS_SINGLE_TURN_E2E_OUTPUT` 修改输出目录。测试不创建沙盒。
