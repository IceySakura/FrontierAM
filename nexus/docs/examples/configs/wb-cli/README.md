# wb-cli

`agent_name: workbuddy` 在 Gongfeng 沙盒里安装私有包 `@tencent/nexus-wb-cli`。默认装 dist-tag `latest`，实际装上的版本见日志 `WorkBuddy installed at ... (version ..., requested latest)`。要固定版本或换 tag，设 `WORKBUDDY_VERSION`（如 `5.9.0`、`hotfix`）。5.9.0 起 `wb --help` 没有 `--permission-mode`，冒烟不要传这个参数，也不用开 `compat_permission_mode_patch`。共享配置 `nexus/extensions/configs/agents/workbuddy.yaml` 默认开着它，用来把 `fullAccess` 送进 wb，只认 `compat.py` 里核对过的版本。这条冒烟走 Chat Completions（`CODEBUDDY_BASE_URL`），不要加 `--features '{"rollout":true}'`。`rollout` 只作用于 `models.json` 里 `"api": "openai-responses"` 的模型。

`wb` 命令本身的用法见 [wb-cli.md](./wb-cli.md)。那份说明覆盖安装、模型接入和 Responses API。本目录的冒烟不走那条路径。

## 环境变量

跑之前导出这些。冒烟不读别的文件。

| 变量 | 必填 | 作用 |
| --- | --- | --- |
| `GONGFENG_RUNTIME_TOKEN` | 是 | Gongfeng 沙盒 token |
| `GONGFENG_PURPOSE` | 否 | 配额桶。不设时用 `test` |
| `OPENAI_API_KEY` | 是 | 模型密钥 |
| `OPENAI_BASE_URL` | 否 | 要带 `/v1`。不设时用 `http://api.aihub.woa.com/standard/v1` |
| `OPENAI_MODEL` | 否 | 不设时用 `deepseek-flash` |
| `WORKBUDDY_VERSION` | 否 | npm 版本号或 dist-tag。不设时用 `latest` |
| `WORKBUDDY_NPM_USERNAME` | 是 | 安装 `@tencent/nexus-wb-cli` 的只读账号。必须和 token 一起设。申请方式见 [iWiki](https://iwiki.woa.com/p/4048045047) |
| `WORKBUDDY_NPM_TOKEN` | 是 | 上面这个账号的 token。申请方式见 [iWiki](https://iwiki.woa.com/p/4048045047) |
| `SWE_DATASET_PATH` | SWE 必填 | 完整实例的 JSONL。连通检查不用 |
| `SWE_INSTANCE_IDS` | 否 | 逗号分隔的实例 id。不设则按文件顺序取 |
| `SWE_LIMIT` | 否 | 最多跑几条。不设时是 `1` |

## 连通检查

```bash
uv run nexus run single docs/examples/configs/wb-cli/smoke.yaml \
  -o outputs/wb-cli-smoke.json
```

成功时 `outputs/wb-cli-smoke.json` 里 `score` 为 `1.0`，`agent_output.success` 为 `true`。

## SWE 冒烟

`swe_generator.py` 读取同目录的 `smoke.yaml`，再换成 `swe_task`。`SWE_DATASET_PATH` 指向一份带完整实例的 JSONL。

```bash
export SWE_DATASET_PATH=/absolute/path/cases.jsonl
export SWE_INSTANCE_IDS=stern__stern-300
export SWE_LIMIT=1
mkdir -p outputs
uv run nexus run batch \
  -g docs/examples/configs/wb-cli/swe_generator.py \
  -d outputs/wb-cli-swe.db \
  --max-concurrent 1
```

进程退出码 `0` 表示这道题跑完了。还要看任务的 `resolved` 和 `score`；CLI 跑完不等于补丁通过。

默认超时：agent 1800 秒，runtime RPC 2700 秒，runner 5400 秒。用 `SWE_AGENT_TIMEOUT`、`SWE_RUNTIME_TIMEOUT`、`SWE_RUNNER_TIMEOUT` 改，并且要满足 `0 < agent < runtime < runner`。
