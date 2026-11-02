"""aidata —— 通用的开放式长时程 agent 基准测试。

该任务接收任意的 `user_prompt` 与可选的 `context_files` ，暴露一个固定的
MCP 工具目录（pdf-tools ，加上以 LLM 后端模拟的消息、日程、会议室、
电话会议、在线文档、笔记以及邮件，全部打包在 `./servers/` 下），
并让 agent 自主运作。

所有 mock 响应都是根据 `user_prompt` 与 `context_files` 由 LLM 即时生成的
—— 无需按任务配置场景。 rubric 检查同样在评估时由 LLM 从 user prompt
派生，然后由一个二值 LLM judge 校验。
"""
