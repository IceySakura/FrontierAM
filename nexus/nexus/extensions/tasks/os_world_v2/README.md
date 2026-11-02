# OSWorld V2（独立任务）

V2注册 `os_world_v2` task与 `osworld_v2` run-part，支持 `osworld-v2.1`。
官方源码固定到 `6f50176ede5a3009ea92849699524ca4903e7a5e`，
本地和平台均读取校验过的远端题包。

- [运行指南](../../../../docs/how-to/osworld-v2-runbook.md)：环境、六模式、平台参数、数据与产物。
- [设计说明](../../../../docs/developing/osworld-v2-hunyuan-design.md)：adapter、HITL、GitLab与独立grader。
- 私有环境模板：本目录 `.env.example`，复制为 `.env` 后显式加载。
- `generators/`：六模式配置；`scripts/`：配置导出、题包构建与诊断。
- `adapter/`：官方worker、传输、CC/Codex交互与评分适配。
- `deployment/`：独立CAD/PDF grader镜像构建。

CC使用 `claude_code_multiturn`，Codex使用 `codex_native_bridge`，
均选择 `execution_profile=osworld_v2`。V1使用独立入口，不共享题包格式。
