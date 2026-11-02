# Nexus Cloud

Nexus Cloud 在云端运行 Nexus 任务。你可以通过 CLI 提交项目，也可以由应用调用 HTTP API；提交后在控制台或通过 API 查询
状态、获取结果和停止任务。

## 选择任务类型

| 类型 | 适用场景 | 下一步 |
| --- | --- | --- |
| Single | 运行一个独立任务，验证配置或执行单次作业 | [快速开始](quickstart.md) |
| Batch | 对一组输入运行任务，控制并发、重试并读取各子任务结果 | [Batch 指南](submit-batch.md) |
| Live | 持续读取事件，并在任务请求输入时继续交互 | [Live 指南](live.md) |

Batch 的每个子任务都是独立的 Single；Live 可以使用 CLI 或 HTTP API 创建，后续交互使用事件与命令 API。

## 接入与阅读路线

[Web 控制台](https://cloud.nexus-agent.woa.com/v1/console)提供 Runs、Batches、My Environments 和 Shared Access。
浏览器使用 OA 登录；个人 CLI/HTTP 脚本使用太湖个人 PAT，线上应用使用太湖应用身份。入口与权限说明见
[认证与运行环境](access-and-environments.md)。

1. **第一次使用**：按[快速开始](quickstart.md)完成 Single 提交并确认业务结果。
2. **运行自己的任务**：按 [Batch 指南](submit-batch.md)或 [Live 指南](live.md)选择输入和提交方式。
3. **接入新的 Batch 输入或依赖**：可先做[提交前本地验证](local-validation.md)，编译任务并检查依赖构建；这不是每次提交的必需步骤。
4. **提交之后**：按[任务与结果](runs-and-results.md)查询状态、读取结果、停止任务和导出数据。
5. **遇到问题**：按[常见问题](troubleshooting.md)定位失败阶段；字段与路径查 [API 参考](reference/api.md)。

程序调用应使用与认证方式匹配的 API 入口；控制台地址和 API 地址用途不同。返回 URL 的使用规则统一见
[程序入口与地址](access-and-environments.md#程序入口与地址)。

## 三个容易混淆的概念

- **部署环境**：正式、测试等一套 Nexus Cloud 服务，决定控制台和 Control/Application API 地址。
- **运行环境（Environment）**：该部署中可申请的运行资源及授权，提交时通过 `--environment` 指定 Environment ID。
- **Runtime**：任务使用的沙盒或容器，例如 Gongfeng；其凭据和容量与 Nexus Manager 的资源额度分别管理。

共享任务、Run Trace Backend 和数据库只读账号属于可选能力，按需要查阅[任务与结果](runs-and-results.md)。
