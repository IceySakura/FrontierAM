# Hyness：用独立 App 配置做 System Prompt 实验

可以复用已发布的 Hyness 引擎和 App npm 包，复制 App 的 YAML、Harness 和 Markdown，
再让 Nexus 的 `agent_config.app_config` 指向副本。修改 System Prompt（SP）无需重新发包。
本文以 2026-10-21 验证过的 mini-SWE 双包为例：

```yaml
packages:
  - '@tencent/hyness@0.0.1-scaffold-apps.20261021.7'
  - '@tencent/hyness-mini-swe-agent@0.1.0-scaffold-apps.20261021.7'
```

配置在每次 Agent 启动时读取。实验之间选择不同副本，运行中的轨迹不应依赖文件热更新。
其它 App 可以沿用这个方法，但需要检查它实际选中的 Harness、prompt 引用和依赖，
不能直接套用本文的 mini-SWE 文件名。通用接入见 [Hyness App / Harness 接入](hyness.md)。

## 为什么需要独立副本

这组包的 `app_config_overlay` 只接受部分 runtime 参数（retry、stopOnToolError、
backgroundTasks、agents、compact）及 subagent 的 background 设置。
把 `mainAgent.prompt` 写进 overlay 会在启动时被拒绝。

`--system-prompt` 和 `--system-prompt-file` 可以替换 SP，但这版 CLI 会把它们视为显式调用方提示词，
额外添加环境、日期上下文，首条 user 消息也会变化。要做只改变 SP 文本的对照，应使用独立 App 副本，
保留 App 原有的 `runtime.context`、工具和 hooks。

mini-SWE 有两处 prompt 引用：

| 文件 | 字段 | 原值 |
| --- | --- | --- |
| `app.yaml` | `mainAgent.prompt` | `./prompts/system.md` |
| `harnesses/mini-swe-agent.yaml` | `agent.prompt` | `../prompts/mini-swe-agent.md` |

本例明确选择 `harness_id: mini-swe-agent`，实际 SP 来自该 Harness。
实测只改 `mainAgent.prompt` 仍使用原版 SP；下面把两处引用统一指向实验 Markdown。

## 1. 在 Agent 执行环境中准备副本

下面的准备脚本适用于普通 runtime：准备脚本与 Agent 使用同一用户、同一文件系统，
不额外配置 `execution_target`。`python3` 需能导入本次 Nexus 和 PyYAML；Node 版本需满足
`^22.19.0`、`^23.11.0` 或 `>=24.10.0`，npm 可访问配置的 registry。

**使用 Nexus 自己的 installer 准备依赖。**这样副本与随后启动的 CLI 会解析到同一份引擎安装。
单独执行一次 `npm install --prefix /somewhere`，即使版本号相同，也可能与 Nexus 的安装目录不同，
触发 `App and CLI resolve different Hyness engines`。

先把自己的完整 SP 保存为 Markdown，再运行以下命令。示例路径都在 Agent 执行环境中，
每个实验使用新的目录，并放在任务仓库（例如 `/testbed`）之外：

```bash
export HYNESS_PROMPT_MARKDOWN=/path/to/my-system.md
export HYNESS_PROMPT_EXPERIMENT_DIR=/tmp/nexus-hyness-sp-v1

python3 - <<'PY'
import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml

from nexus.extensions.agents.hyness.installer import install_hyness, verify_app_engine
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport

PACKAGES = [
    '@tencent/hyness@0.0.1-scaffold-apps.20261021.7',
    '@tencent/hyness-mini-swe-agent@0.1.0-scaffold-apps.20261021.7',
]


async def prepare():
    prompt = Path(os.environ['HYNESS_PROMPT_MARKDOWN']).resolve(strict=True)
    experiment = Path(os.environ['HYNESS_PROMPT_EXPERIMENT_DIR']).resolve()
    experiment.mkdir(parents=True, exist_ok=False)

    transport = AgentExecutionTransport()
    installed = await install_hyness(transport, PACKAGES, 'https://mirrors.tencent.com/npm/')
    prefix = Path(installed['prefix']).resolve()
    package = prefix / 'node_modules/@tencent/hyness-mini-swe-agent'
    app = experiment / 'app'

    shutil.copytree(package / 'apps/mini-swe-agent', app)
    # 保留命名插件的 dependencies / bundleDependencies 声明。
    shutil.copyfile(package / 'package.json', app / 'package.json')
    # 复用同一份已安装插件和引擎；副本中的 YAML/Markdown 是独立文件。
    (app / 'node_modules').symlink_to(package / 'node_modules', target_is_directory=True)
    (experiment / 'node_modules').symlink_to(prefix / 'node_modules', target_is_directory=True)
    shutil.copyfile(prompt, app / 'prompts/experiment.md')

    app_path = app / 'app.yaml'
    app_config = yaml.safe_load(app_path.read_text(encoding='utf-8'))
    app_config['mainAgent']['prompt'] = './prompts/experiment.md'
    app_path.write_text(yaml.safe_dump(app_config, sort_keys=False), encoding='utf-8')

    harness_path = app / 'harnesses/mini-swe-agent.yaml'
    harness = yaml.safe_load(harness_path.read_text(encoding='utf-8'))
    harness['agent']['prompt'] = '../prompts/experiment.md'
    harness_path.write_text(yaml.safe_dump(harness, sort_keys=False), encoding='utf-8')

    await verify_app_engine(transport, installed, str(app_path))
    metadata = {
        'packages': PACKAGES,
        'npm_prefix': str(prefix),
        'app_config': str(app_path),
        'prompt_sha256': hashlib.sha256((app / 'prompts/experiment.md').read_bytes()).hexdigest(),
    }
    (experiment / 'experiment.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


asyncio.run(prepare())
PY
```

得到的目录如下，链接目标是该 runtime 上 Nexus 实际使用的 npm 安装目录：

```text
/tmp/nexus-hyness-sp-v1/
├── experiment.json
├── node_modules -> <Nexus npm prefix>/node_modules
└── app/
    ├── package.json
    ├── node_modules -> <原 App 包>/node_modules
    ├── app.yaml
    ├── harnesses/mini-swe-agent.yaml
    └── prompts/
        ├── experiment.md
        └── …原 App 的其余 Markdown
```

`package.json` 不能省略：缺少 App 的依赖声明时，Hyness 可能转而向 registry 请求命名插件，
无法复用已经安装的 bundled dependencies。两条 `node_modules` 链接分别保留插件和引擎的解析路径；
原 npm 包中的文件无需修改。

保留原来的 `runtime.context.system: authored`、`user: task`、`currentTime: false`、
`initialUserTemplate`，以及工具、hooks 和提交规则。不要再叠加 `--system-prompt` 类 CLI 覆盖。
本例副本中的两个字段最终是：

```yaml
# app.yaml，保留文件其它字段
mainAgent:
  prompt: ./prompts/experiment.md
```

```yaml
# harnesses/mini-swe-agent.yaml，保留文件其它字段
agent:
  prompt: ../prompts/experiment.md
```

## 2. 生成独立的 Nexus Agent 配置

在提交实验的机器上，从 Nexus 仓库根目录运行下列命令。
它复制现有 `hyness_mini_swe_agent` profile，保留模型族适配、启动参数和工具 schema 设置，
固定上述两个包版本，并选择实验副本。

`HYNESS_EXPERIMENT_APP_CONFIG` 填写的是 **Agent 执行环境中的路径**；
输出的 `HYNESS_EXPERIMENT_PROFILE` 则是 **提交机器上的配置文件**：

```bash
export HYNESS_EXPERIMENT_APP_CONFIG=/tmp/nexus-hyness-sp-v1/app/app.yaml
export HYNESS_EXPERIMENT_PROFILE=/tmp/hyness-mini-swe-sp-v1.yaml

uv run python - <<'PY'
import os
from pathlib import Path

import yaml

base = Path('nexus/extensions/configs/agents/hyness_mini_swe_agent.yaml')
part = yaml.safe_load(base.read_text(encoding='utf-8'))
config = part['body']['agent_config']
config['packages'] = [
    '@tencent/hyness@0.0.1-scaffold-apps.20261021.7',
    '@tencent/hyness-mini-swe-agent@0.1.0-scaffold-apps.20261021.7',
]
config['app_config'] = os.environ['HYNESS_EXPERIMENT_APP_CONFIG']
# run-part 合并把 YAML null 视为删除；延迟解析才能显式传入 None。
config['app_config_overlay'] = '${oc.decode:null}'
config['harness_id'] = 'mini-swe-agent'
part['source'] = 'hyness-mini-swe-sp-v1'
output = Path(os.environ['HYNESS_EXPERIMENT_PROFILE'])
with output.open('x', encoding='utf-8') as stream:
    yaml.safe_dump(part, stream, sort_keys=False, allow_unicode=True)
print(output)
PY

export AGENT_CONFIG="$HYNESS_EXPERIMENT_PROFILE"
```

专用 `hyness_mini_swe_agent` profile 的 `app_config` 是固定包内路径，
仅设置 `HYNESS_APP_CONFIG` 不会覆盖它。这里通过外部 run-part 文件选择实验路径。
也可以使用通用 `AGENT_CONFIG=hyness`，但要自行保留实验需要的其它 Agent 设置。

## 3. 接到已有 SWE 实验

沿用原实验的模型、数据、runtime 和评分设置，只把 `AGENT_CONFIG` 换成上一步生成的文件。
下面展示共享 SWE generator 的调用；先在环境中配置所需凭据：

```bash
export AGENT_CONFIG=/tmp/hyness-mini-swe-sp-v1.yaml
export RUNTIME_CONFIG=swe_runtime
export LLM_CONFIG=/path/to/model.yaml
export DATASET_PATH=/path/to/swe-multi.jsonl
export SWE_LIMIT=2

uv run nexus run-batch -g nexus/extensions/configs/generator_swe_template.py \
  -d /tmp/hyness-mini-swe-sp-v1.db --max-concurrent 2
```

**该命令只选择实验配置，不会自动在新 sandbox 中执行第 1 步。**
Gongfeng 批量运行时，应在每个 runtime 调用 Agent 之前的初始化／任务准备阶段执行准备脚本，
或者预先准备有效的镜像、挂载与依赖路径。仅把配置文件放在提交机器的 `/tmp` 下不足以运行。

当前启动顺序是安装 npm 包 → 检查 App／overlay 路径及引擎一致性 → 执行 `pre_start_scripts`
→ 启动 Hyness。因此，不能先配置一个不存在的 App 路径，再依靠同一次调用的
`pre_start_scripts` 创建它。准备过程必须发生在调用 Agent 之前。

使用切换用户或 Compose 的 `execution_target` 时，需要按照该 target 的安装目录和权限准备副本；
第 1 步的默认 transport 脚本不能直接套用。把副本交接到另一台机器时，应在目标环境重新建立
依赖解析关系，不能直接搬运指向提交机器安装目录的绝对软链接。

每组实验保存独立的配置、Markdown 和 `experiment.json`。已生成的 batch DB 或 Parquet 如果固化了
Agent 配置，不会随共享 YAML 自动更新，需要重新生成或显式更新对应的实验输入。

## 验证结果与检查方法

2026-10-21 使用上述精确版本，真实启动 Nexus 的 `setup_and_run_hyness`、AnthropicProxy、
Hyness CLI、mini-SWE 插件和 bash 工具。模型由固定两次工具调用的本地响应器替代；
安装步骤使用事先安装好的双包路径。成功场景均完成文件修改，并以 `hook_stopped` 提交结束。

| 场景 | 结果 |
| --- | --- |
| 原版 App | 原版 SP；正常修改文件、提交 |
| overlay 覆盖 `mainAgent.prompt` | 启动失败：`Unrecognized key: "mainAgent"`，没有模型调用 |
| CLI `--system-prompt` | 新 SP 生效，但增加环境／日期上下文 |
| pre-start 写文件 + `--system-prompt-file` | 新 SP 生效，同样增加上下文 |
| 副本只改 `mainAgent.prompt` | 执行成功，实际仍使用 Harness 原版 SP |
| 副本同时更新两处 prompt 引用 | 两轮请求都精确使用实验 SP，保留原用户消息模板和工具行为 |

对原版与最后一个场景的两轮请求，只归一化 session ID 和每次不同的工作目录后，diff 仅有
system 文本变化。工具名称、描述、完整参数 schema、首条用户消息模板、工具调用参数、工具返回及
模型请求参数均一致；App 副本实验前后的原包资源哈希一致。

这是配置传递与执行行为验证，没有运行真实模型的 SWE 评分或 GPU RL 训练。
真实模型在不同 SP 下可以产生不同工具调用和答案，不能由上述固定响应器实验推断分数一致。

后续更换包或修改配置时，先检查实际模型请求里的完整 system、首条 user、工具 schema 和工具返回，
再比较任务得分。确认 SP 来自选中 Harness，并保留 App 的模板、工具及提交设置，
才能把得分变化归因到本次提示词实验。

相关实现：

- [Nexus App 路径解析与引擎校验](../../../nexus/extensions/agents/hyness/installer.py)
- [Nexus Hyness 启动顺序](../../../nexus/extensions/agents/hyness/runtime_setup.py)
- [mini-SWE Agent profile](../../../nexus/extensions/configs/agents/hyness_mini_swe_agent.yaml)
- [共享配置加载规则](../../../nexus/extensions/configs/README.md)
