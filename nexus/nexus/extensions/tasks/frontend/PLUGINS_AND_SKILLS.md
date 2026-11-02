# Plugins 与 Skills 接入与触发机制

本文以 `generator_midtrain.py` 与 `generator_midtrain_game.py` 为例，说明 Claude Code **Plugins** 与 **Skills** 是如何在 nexus 批量任务中：

1. 从本地（host 侧）打包
2. 注入到 sandbox 容器（runtime 侧）
3. 在 Claude Code agent 启动时被加载
4. 被 prompt / 模型调用而触发

> 涉及的两个文件：
> - `generator_midtrain.py`（标准版）
> - `generator_midtrain_game.py`（增加了 `.claude` 整目录覆盖能力）

---

## 0. 整体执行流水线

```
Host (本地机器)                                 Sandbox 容器 (runtime)
─────────────────────────────                  ───────────────────────────────────
generate_configs()                             pre_start_scripts (按顺序执行)
 ├─ 读取 dataset.jsonl                          ├─ mkdir -p /workspace
 ├─ 构建 final_prompt                            ├─ [可选] 同步 .claude/ 整目录   (game)
 │   (可加 /<skill> 前缀)                         ├─ [可选] plugin install 脚本
 ├─ 生成 plugin install 脚本                     ├─ [可选] skills 同步脚本
 ├─ 生成 skills 同步脚本                          └─ [可选] 写入 askuser_hook.py
 ├─ 生成 hook script writer                                 ▼
 └─ 组装 BatchItem(config)              Claude Code CLI 启动
                                         ├─ 读 ~/.claude/settings.json (含 enabledPlugins / hooks)
                                         ├─ 读 ~/.claude/plugins/installed_plugins.json
                                         ├─ 读 ~/.claude/skills/<name>/SKILL.md (作为 slash 命令)
                                         └─ 收到 final_prompt → 触发对应 plugin / skill
```

关键点：**host 侧不直接 ssh 进容器执行 `claude plugin install`**，而是把本地已安装的 plugin 文件**序列化成一段 Python 脚本**，作为 `pre_start_scripts` 在容器内执行，从而实现「离线、无网络」安装。Skills 的注入思路一致，但更简单（直接 base64 写文件）。

---

## 1. Plugins 的接入

### 1.1 本地准备（一次性）

Plugins 来自一个 Claude 官方 marketplace clone：

```python
_MARKETPLACE_NAME = 'claude-plugins-official'
_MARKETPLACE_ROOT = Path('~/.claude/plugins/marketplaces') / _MARKETPLACE_NAME
```

要求 host 侧已经执行过：

```bash
claude plugin marketplace add anthropics/claude-plugins-official
claude plugin install <plugin-name>
```

执行后，本地 `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/<plugin-name>/` 下会有完整插件源码（含 `.claude-plugin/plugin.json`）。

### 1.2 配置要安装哪些 plugin

通过环境变量 `CLAUDE_PLUGINS`（逗号分隔）决定：

```bash
CLAUDE_PLUGINS="frontend-design,skill-creator"
```

代码里的解析：

```550:574:nexus/extensions/tasks/frontend/generator_midtrain.py
CLAUDE_PLUGINS_TO_INSTALL: list[str] = _parse_comma_list(os.getenv('CLAUDE_PLUGINS', ''))
...
plugins_to_install = _parse_comma_list(
    _extra_env_value(extra_env, 'CLAUDE_PLUGINS', CLAUDE_PLUGINS_TO_INSTALL)
)
plugin_key = tuple(plugins_to_install)
if plugin_key not in plugin_install_scripts:
    plugin_install_scripts[plugin_key] = (
        _build_plugin_install_script(plugins_to_install) if plugins_to_install else None
    )
```

差异点：
- `generator_midtrain.py`：**支持每条样本通过 `extra_env["CLAUDE_PLUGINS"]` 覆盖**全局值，并按 `plugin_key` 缓存生成的安装脚本，避免重复构建。
- `generator_midtrain_game.py`：**只在 generator 启动时根据全局 env 构建一次**（`_build_plugin_install_script(CLAUDE_PLUGINS_TO_INSTALL)`），所有样本共享同一脚本。

### 1.3 把本地 plugin 序列化成"自包含"安装脚本

核心函数：`_build_plugin_install_script(plugins)`

它做了三件事：

1. **遍历每个 plugin 目录的全部文件**，全部读成字符串，记录到 `plugin_data[plugin_name@marketplace] = {files: {rel_path: text}, version, gitCommitSha}`。
2. **生成一段 Python 源码**（拼字符串），这段源码不会引用本地路径，而是把所有 plugin 文件内容 inline 进去，运行时直接 `os.makedirs + open(..., 'w').write(content)` 还原到容器里的：
   - `~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/...`（plugin 实际文件）
   - `~/.claude/plugins/marketplaces/<marketplace>/.claude-plugin/marketplace.json`（marketplace 元信息）
   - `~/.claude/plugins/installed_plugins.json`（"已安装"清单，scope=user）
   - `~/.claude/plugins/known_marketplaces.json`（已知 marketplace 元数据）
3. **修补 `~/.claude/settings.json`**：把 `enabledPlugins[<plugin@marketplace>] = True` 写入（保留已有 settings），并在 `extraKnownMarketplaces` 里登记 marketplace 来源。

这样容器内的 Claude Code CLI 启动时，会以为这些 plugin 是通过正常 `claude plugin install` 安装的——只是源是 GitHub repo 的"虚拟克隆"。

### 1.4 在 sandbox 中运行该脚本

通过 `agent_config.pre_start_scripts` 注入：

```629:651:nexus/extensions/tasks/frontend/generator_midtrain.py
config['agent_config'] = {
    'pre_start_scripts': [
        {'code': f'mkdir -p {working_dir}'},
        *(
            [{'interpreter': 'python', 'code': _plugin_install_script}]
            if _plugin_install_script else []
        ),
        *(
            [{'interpreter': 'python', 'code': _build_skills_install_script(SKILLS_DIRS_TO_INSTALL)}]
            if SKILLS_DIRS_TO_INSTALL else []
        ),
        *(
            [{'interpreter': 'python', 'code': _build_askuser_hook_script_writer()}]
            if not _env_bool('FRONTEND_DISABLE_HOOKS') else []
        ),
    ],
    ...
}
```

> `pre_start_scripts` 在 Claude Code CLI 启动**之前**执行，因此 plugin 文件就绪后，CLI 才会去读 `installed_plugins.json` 与 `enabledPlugins`。

### 1.5 如何被触发

- **Claude Code 启动后扫描** `~/.claude/plugins/cache/.../plugins.json` / `commands/`，把 plugin 内含的 slash commands、skills、hooks、subagent 等注册进 CLI。
- 用户/调度方在 `final_prompt` 里使用 plugin 暴露的能力（slash 命令、显式调用 plugin 工具、或 plugin 自带的 skill 名）即可触发。
- **plugin 内置的 skill**（例如 `frontend-design` plugin 自带的 `/frontend-design`）走的是和下面 §2 一样的 skill 触发路径。

---

## 2. Skills 的接入

Skills 比 plugin 简单很多——它就是 `~/.claude/skills/<skill-name>/` 目录下的一个 `SKILL.md` + 资源文件。Claude Code 会**自动**把每个目录注册成同名的 slash 命令。

### 2.1 本地准备

只要 host 上有以下结构即可（无需事先 install）：

```
/path/to/my_skills/
├── frontend_codesign/
│   ├── SKILL.md
│   └── ...
└── data_viz/
    ├── SKILL.md
    └── examples/
```

### 2.2 配置要同步哪些 skill 目录

环境变量 `FRONTEND_SKILLS_DIR`（逗号分隔的本地路径）：

```bash
FRONTEND_SKILLS_DIR="/path/to/my_skills/frontend_codesign,/path/to/my_skills/data_viz"
```

解析在 `_parse_skills_dirs()`：每个路径必须是已存在的目录，否则抛 `FileNotFoundError`。

> **目录名 = slash 命令名**，无需任何映射或注册。
> - 目录 `frontend_codesign` → 命令 `/frontend_codesign`
> - 目录 `frontend-codesign` → 命令 `/frontend-codesign`

### 2.3 同步到 sandbox

`_build_skills_install_script(skills_dirs)`：把目录里**所有文件 base64 编码**，拼成一段 Python，运行时在容器里写到 `~/.claude/skills/<dir_name>/<rel_path>`。

```239:267:nexus/extensions/tasks/frontend/generator_midtrain.py
def _build_skills_install_script(skills_dirs: list[Path]) -> str:
    ...
    for skill_dir in skills_dirs:
        skill_name = skill_dir.name
        for fp in sorted(skill_dir.rglob('*')):
            if fp.is_file():
                content_bytes = fp.read_bytes()
                rel_path = str(fp.relative_to(skill_dir))
                encoded = _b64.b64encode(content_bytes).decode()
                target = f'os.path.join(skills_root, {skill_name!r}, {rel_path!r})'
                lines.append(f'os.makedirs(os.path.dirname({target}), exist_ok=True)')
                lines.append(f'open({target}, "wb").write(base64.b64decode({encoded!r}))')
    ...
```

同样作为 `pre_start_scripts` 中的一项传入，运行时写盘。

### 2.4 如何被触发

Skills 有两种触发途径：

**(1) 显式 slash 触发（常用）**
通过 `SKILL_NAME` 环境变量，把 `/<skill>` 拼到 prompt 头：

```563:569:nexus/extensions/tasks/frontend/generator_midtrain.py
skill_name = str(_extra_env_value(extra_env, 'SKILL_NAME', os.getenv('SKILL_NAME', '')) or '').strip()
if skill_name:
    final_prompt = f'/{skill_name} ' + original_prompt
else:
    final_prompt = original_prompt
```

例如 `SKILL_NAME=frontend_codesign` 且 prompt 是 "做一个登录页"，最终发给 Claude Code 的就是：

```
/frontend_codesign 做一个登录页
```

Claude Code CLI 看到 `/frontend_codesign` 时，会读 `~/.claude/skills/frontend_codesign/SKILL.md` 作为系统指令注入到本轮上下文，然后才让模型处理用户输入。

> `generator_midtrain.py` 支持每条样本用 `extra_env["SKILL_NAME"]` 覆盖；
> `generator_midtrain_game.py` **只读全局 env**，且当 `SKILL_NAME` 为空时会**自动**在 prompt 末尾追加 `"\n\nPlease use skills to complete the task"`，提示模型自主选用 skill。

**(2) 模型自主调用（隐式）**
即使没有 `/skill` 前缀，Claude Code 也会把 `~/.claude/skills/*/SKILL.md` 的"摘要"放入系统提示中（行为由 SKILL.md 顶部的 frontmatter 决定）。模型在判断任务匹配 skill 描述时，会主动加载 skill 全文执行。`generator_midtrain_game.py` 那句兜底 prompt 就是利用这个机制。

---

## 3. `generator_midtrain_game.py` 特有：`.claude` 整目录覆盖

`game` 版本额外增加了一个机制——**直接覆盖整个 `~/.claude/` 目录**，比 plugin/skill 颗粒度更粗，可以一次塞入 settings、agents、commands、skills、references、scripts 等任意结构。

### 3.1 来源

硬编码的本地路径：

```261:261:nexus/extensions/tasks/frontend/generator_midtrain_game.py
_GAME_SKILLS_CLAUDE_DIR = Path('/apdcephfs_qy4/share_302593112/maxluxchen/workspace/tools/game_skills/.claude')
```

### 3.2 注入

`_build_claude_dir_sync_script(source_dir)` 跟 skills 注入逻辑几乎一样（base64 + 写文件），区别是目标根目录是 `~/.claude/` 而不是 `~/.claude/skills/`。

### 3.3 顺序

注意它在 `pre_start_scripts` 里的**位置**：

```645:668:nexus/extensions/tasks/frontend/generator_midtrain_game.py
'pre_start_scripts': [
    {'code': f'mkdir -p {working_dir}'},
    *([{'interpreter': 'python', 'code': _build_claude_dir_sync_script(_GAME_SKILLS_CLAUDE_DIR)}]
       if _GAME_SKILLS_CLAUDE_DIR.is_dir() else []),
    *([{'interpreter': 'python', 'code': _plugin_install_script}]
       if _plugin_install_script else []),
    *([{'interpreter': 'python', 'code': _build_skills_install_script(SKILLS_DIRS_TO_INSTALL)}]
       if SKILLS_DIRS_TO_INSTALL else []),
    *([{'interpreter': 'python', 'code': _build_askuser_hook_script_writer()}]
       if not _env_bool('FRONTEND_DISABLE_HOOKS') else []),
],
```

执行顺序很关键：
1. **先**铺整目录 → 提供基础底图
2. **再**装 plugin → plugin 会写 `~/.claude/plugins/...` 与 patch `settings.json`，可能覆盖第 1 步
3. **再**装 skills → 把额外 skill 目录补进 `~/.claude/skills/`
4. **最后**写 askuser hook 脚本

后面的步骤可以覆盖前面的同名文件，符合"基础包 + 增量补丁"的语义。

---

## 4. Hook：另一种"被触发"的机制

虽然题目主要是 plugin/skill，但 hook 与触发链密切相关，顺带说明：

### 4.1 注册：通过 `config_file_patches`

`_build_hook_patches()` 返回：

```477:495:nexus/extensions/tasks/frontend/generator_midtrain.py
patches: list[dict[str, Any]] = [
    {
        'path': '~/.claude/settings.json',
        'content': {
            'hooks': {
                'PreToolUse': [
                    {'matcher': 'ExitPlanMode',
                     'hooks': [{'type': 'command', 'command': _EXITPLAN_HOOK_CMD}]},
                    {'matcher': 'AskUserQuestion',
                     'hooks': [{'type': 'command', 'command': _ASKUSER_HOOK_CMD}]},
                ],
            },
        },
    },
]
```

这个 dict 会被 nexus runtime 合并写入 `~/.claude/settings.json`（与 plugin 的 `enabledPlugins` 同一个文件，互不冲突）。

### 4.2 触发

Claude Code 调用工具前会检查 `hooks.PreToolUse`：
- `ExitPlanMode` → 直接 `echo` 一段 JSON，`permissionDecision=allow`，让计划模式通过。
- `AskUserQuestion` → 调用 `/tmp/_nexus_askuser_hook.py`（同样由 pre_start_scripts 写入），由它返回选项。

这是为了让 `--print` 非交互模式下，需要用户交互的工具不被卡住。

### 4.3 可选：WebSearch MCP

当 `FRONTEND_ENABLE_SEARCH=1` 时，往 `~/.claude/mcp.json` 注入一个 SSE 类型的 `WebSearchMCP` 服务，并通过 `--disallowedTools WebSearch` 关闭内建工具。Claude Code 启动时会从 mcp.json 拉起 MCP server，模型把它当作普通工具调用——这就是触发方式。

---

## 5. 速查表：环境变量 → 行为

| 环境变量 | 作用 | 触发路径 |
|---|---|---|
| `CLAUDE_PLUGINS` | 要装哪些 plugin（逗号分隔） | 写入 `~/.claude/plugins/...` + `enabledPlugins`，CLI 启动时加载 |
| `FRONTEND_SKILLS_DIR` | 要同步哪些本地 skill 目录 | 写入 `~/.claude/skills/<name>/`，目录名 = slash 命令名 |
| `SKILL_NAME` | 在 prompt 头加 `/<skill>` | 显式触发对应 skill |
| `FRONTEND_DISABLE_HOOKS` | 关闭 ExitPlanMode/AskUserQuestion hooks | 跳过 hook 注入与脚本部署 |
| `FRONTEND_ENABLE_SEARCH` | 启用 WebSearch MCP | 写入 `~/.claude/mcp.json`，CLI 拉起 MCP server |
| `FRONTEND_SEARCH_MCP_URL` | WebSearch MCP 的 SSE URL | 同上 |

`generator_midtrain.py` 还允许在 dataset 的每条样本里通过 `extra_env` 覆盖 `CLAUDE_PLUGINS` 和 `SKILL_NAME`，做到**逐条样本不同的 plugin/skill 组合**；`generator_midtrain_game.py` 没有这层覆盖。

---

## 6. 两个 generator 的接入差异速览

| 方面 | `generator_midtrain.py` | `generator_midtrain_game.py` |
|---|---|---|
| Plugin 安装脚本 | 按 `plugin_key` 缓存，**支持 per-instance 覆盖** | 启动时构建一次，**全局共享** |
| Skill 触发 prompt | 仅 `SKILL_NAME` 设置时拼 `/<skill>` | 未设置 `SKILL_NAME` 时追加 `"Please use skills to complete the task"` |
| `.claude` 整目录覆盖 | 无 | 有，从 `_GAME_SKILLS_CLAUDE_DIR` 注入 |
| `extra_env` 覆盖 | 支持 `CLAUDE_PLUGINS`/`SKILL_NAME` per-instance | 不支持，全局 env 为准 |
| Prompt 字段 | `problem_statement` > `prompt` > `text` | 优先 `messages[role=user]`，否则 `prompt`/`question` |

其余（hook、MCP、LLM provider、resource override、EVAL_TIMES 多轮、SHUFFLE）逻辑完全一致。

---

## 7. 端到端调用示例

```bash
# 装 frontend-design plugin + 自定义 codesign skill，并强制 skill 触发
export CLAUDE_PLUGINS="frontend-design"
export FRONTEND_SKILLS_DIR="/abs/path/to/skills/frontend_codesign"
export SKILL_NAME="frontend_codesign"

export FRONTEND_DATASET_PATH="./data/frontend.jsonl"
export FRONTEND_RUNTIME_TOKEN="..."
export OPENAPI_API_KEY="..." OPENAPI_BASE_URL="..." OPENAPI_MODEL_NAME="..."

uv run nexus run-batch \
  -g nexus/extensions/tasks/frontend/generator_midtrain.py \
  -d ./outputs/frontend.db --max-concurrent 5
```

发生的事：

1. host：读 `~/.claude/plugins/marketplaces/.../frontend-design/` 全部文件 → 拼成 plugin 安装脚本。
2. host：读 `/abs/path/to/skills/frontend_codesign/` 全部文件 → 拼成 skills 同步脚本。
3. host：把 `final_prompt = "/frontend_codesign <原 problem_statement>"` 写进 `task_config`。
4. nexus 启动 sandbox，依次执行 `mkdir`、plugin 脚本、skills 脚本、askuser hook 脚本、写 `settings.json`/`mcp.json` 补丁。
5. Claude Code CLI 启动，发现 `enabledPlugins` 与 `~/.claude/skills/frontend_codesign/SKILL.md`。
6. CLI 收到 prompt `/frontend_codesign ...` → 加载 skill → 进入相应工作流，必要时调用 plugin 暴露的工具/agent，最终把 `/workspace` 打包返回给 host。
