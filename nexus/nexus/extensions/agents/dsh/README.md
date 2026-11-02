# DeepSeek Harness ACP Agent

`dsh` Agent 仅通过官方 Agent Client Protocol（ACP）标准输入输出桥接运行 DeepSeek Harness。每次 Nexus
Agent 调用都会启动一个长生命周期的 DSH 进程并创建一个新的 ACP 会话，因此第一轮和后续各轮用户消息共享同一份
Harness 上下文。

运行时会安装精确版本的 `@deepseek-ai/dsh` 和 `@deepseek-ai/dsh-acp`。DSH 当前要求 Node.js
`^22.19.0` 或 `>=24`；如果运行时不满足该版本范围，安装器会在选择安装来源之前直接失败。不同 DSH 版本、ACP
版本、平台、libc 和 Node 主版本使用相互隔离的安装目录；会话主目录则位于这些不可变安装目录之外。

安装器会优先从腾讯 Generic 仓库匿名下载按版本组织的预构建 npm 安装树。归档保留标准 `dsh` 可执行入口、
profile、插件、npm 元数据以及在 CI 中编译的 `node-pty`，所以启动方式与本地 npm 安装完全一致，并且不要求运行时
提供编译器。如果对应产物不存在或仓库暂时不可用，安装器会回退到 npm。由于 DSH 的子进程能力依赖
`node-pty`，npm 生命周期脚本必须保持启用，因此回退路径需要 `make` 和 C++ 编译器（Debian/Ubuntu 中为
`build-essential`）。已发布产物如果出现校验和或身份不匹配，安装器会直接失败，不会用 npm 回退掩盖完整性问题。
仅当需要强制使用 npm 路径时，才设置 `prebuilt_base_url: null`。

每一组精确的 DSH/ACP 版本都采用相同的安装选择流程。已发布的预构建安装树会冻结并记录完整的依赖解析结果。
如果没有匹配的预构建产物，npm 回退路径会根据请求的两个顶层版本动态生成 package manifest，然后执行
`npm install`。该路径会解析当时最新的传递依赖范围，因此只提供尽力而为的兼容性，不保证可复现；Nexus wheel
中不会携带任何特定 DSH 版本的 npm lock。

Nexus 会为每个会话创建独立的 DSH 主目录和 profile。该 profile 会：

- 使用官方 `openai-completions` 适配器，通过运行时 `OpenAIProxy` 路由 `nexus-model`；
- 将 `@deepseek-ai/dsh-acp` 作为唯一前端；
- 持久化未经压缩和打包的 JSONL，并以 `turn/end` 作为每轮结果的权威记录；
- 禁用 DeepSeek 原生模型路由、遥测、计划审核、Web 搜索、任务、可续接子 Agent、工作流和 Ralph，确保模型调用
  不会超出 Nexus 为当前轮次注册的 LLM 桥接生命周期。

`get_user_input` 默认是一个返回 `None` 的异步函数，因此普通调用方只会执行一轮 ACP 对话。传入自定义函数后，
无需改变传输方式或进程模型，即可在同一会话中最多执行 `max_turns` 轮。`close_on_finish` 默认为 `true`。当前
v1 仅支持 worker 内新建会话；上游 ACP 桥接尚未实现跨 worker 的持久化恢复。
