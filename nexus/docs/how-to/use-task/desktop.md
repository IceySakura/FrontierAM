# Desktop 工具与 OSWorld-V1 任务

## 两类运行场景

- **通用 Desktop**：`desktop` 是工具集合，可与 `identity` task 配合，在适配的
  Docker、Gongfeng 或 AGS runtime 上启动应用、截图、输入和执行命令。
- **OSWorld-V1 benchmark**：当前维护的六模式使用 **AGS**，从 COS `task_files`
  读取题目与 setup/evaluator 资源。不是 Gongfeng Desktop 示例，也不依赖本地官方 examples。

完整六模式配置见
[OSWorld 原生运行指南](../osworld-runbook.md)。

## 通用 Desktop

已有 Gongfeng Desktop 示例仍用于通用桌面工具，不代表OSWorld评测入口：

```bash
nexus run-single docs/examples/configs/desktop.yaml -o /tmp/desktop.result.json
```

按该配置准备Gongfeng凭据、桌面镜像和模型服务。`desktop/start`负责GUI启动；
这些容器镜像准备步骤不是AGS原生OSWorld评测的前置条件。

## OSWorld-V1：AGS + COS

先按原生运行指南准备：

- `WEDATA_KEY`：COS数据访问凭据；
- `AGS_SECRET_ID`、`AGS_SECRET_KEY`、`AGS_OSWORLD_TOOL_ID`：AGS控制面；
- `NEXUS_MODEL/BASE_URL/API_KEY`或Hunyuan模型服务配置；
- `OSWORLD_DATASET_URI`：Run Part/batch使用的线上数据索引；
- `os_world[host]`模块声明的evaluator依赖。

完整YAML入口：

```bash
nexus run-single nexus/extensions/tasks/os_world/configs/osworld_gpt_bare.yaml \
  -o /tmp/osworld.result.json
```

Run Part入口：

```bash
export OSWORLD_DATASET_URI='hy_gui_eval/osworld@v1.1.1'
export TASK_ID=bb5e4c0d-f964-439c-97b6-bdb9747de3f4
nexus run-single \
  --use "osworld_native:model_family=gpt,harness=bare,task_id=$TASK_ID" \
  -o /tmp/osworld.result.json
```

两者都使用 `task_name: os_world`。完整YAML携带远程`task_files`定位；
Run Part/batch从线上索引生成它。不要给标准入口填写
`OSWORLD_EXAMPLES_ROOT`、`task_root`或本机`task_file`代替COS数据链路。

执行顺序：

```text
题目与资源加载 → AGS申请 → setup → agent → evaluator → 输出score → 释放AGS
```

GPT/Claude裸CUA分别使用 `cua_gpt_agent`、`cua_anthropic_agent`；
CC/Codex分别使用 `claude_code_native_bridge`、`codex_native_bridge`
及 `execution_profile: osworld`。两种harness的模型client与执行内核分离。
CC/Codex参数写在对应profile的 `execution_options` 中。

## 评分与输出

当前支持多metric等组合评分，具体以题目定义和getter/metric注册实现为准，
默认 `unsupported_evaluator_policy: fail`，不通过跳过未知metric冒充通过。

- `success=true`表示运行完成，不等于题目满分。
- 题目得分看 `task_output.score`；允许部分分。
- 失败原因看顶层 `error`、agent结束状态和evaluator结果。
- 轨迹和运行目录按原生指南配置；`--hold`可用于有意识地保留现场调试。

AGS底层接口与通用工具示例见
[使用 AGS OSWorld runtime](../use-runtime/use-ags-osworld.md)。
