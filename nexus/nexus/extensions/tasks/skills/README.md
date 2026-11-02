# SkillsBench Oracle 评测 — 工蜂沙箱复现指南

## 概述

SkillsBench 包含 87 个 task，Oracle 模式运行每个 task 的参考解法（solve.sh）并用测试验证（test.sh/test_outputs.py），期望 score=1.0。

本指南说明如何在工蜂沙箱上复现 Oracle 评测结果。当前通过率：**82/87**。

## 前置条件

### 1. 仓库

```bash
# skillsbench 仓库（含所有 task 和适配修改）
git clone git@git.woa.com:nexus-agent/skills-bench.git
# 默认路径：/data/home/<user>/code/skillsbench

# nexus 仓库（orchestration 框架）
# 确保 generator_gongfeng.py 包含 SocketIO RPC 和按 task.toml 分配资源的修改
```

### 2. 环境变量

运行前**必须** export 以下环境变量（配置文件不再硬编码默认值，未设置会导致运行失败）：

```bash
# skillsbench task 目录路径（必填）
export TASK_BASE_DIR="/data/home/<user>/code/skillsbench/tasks"

# 工蜂沙箱 token（必填）
export GONGFENG_RUNTIME_TOKEN="<your-gongfeng-token>"

# 主 API（大部分 task 使用，必填）
export OPENAI_BASE_URL="http://api.model-eval.woa.com"
export OPENAI_API_KEY="<your-key>?provider=aws_third&timeout=600"

# Whisper/TTS API（video-filler-word-remover, video-tutorial-indexer, pg-essay-to-audiobook 使用）
# 注意：这 3 个 task 需要 provider=openai，不是 aws_third
export OPENAI_WHISPER_BASE_URL="http://api.model-eval.woa.com/v1"
export OPENAI_WHISPER_API_KEY="<your-key>?provider=openai&timeout=600"

# GitHub
export GH_TOKEN="<github-pat>"

# HuggingFace
export HUGGINGFACE_API_TOKEN="<hf-token>"

# Modal（mhc-layer-impl 需要，当前额度受限）
export MODAL_TOKEN_ID="<modal-token-id>"
export MODAL_TOKEN_SECRET="<modal-token-secret>"

# Gemini 代理
export GEMINI_BASE_URL="http://trpc-gpt-eval.production.polaris:8080"
export GEMINI_API_KEY="<your-key>?provider=google&api=gemini_api&timeout=300"
```

### 3. Docker 镜像

所有 task 镜像已推送到 `mirrors.tencent.com/skill_and_mcp/skillsbench/<task-name>:latest`。

如需重建某个 task 镜像：

```bash
cd skillsbench/tasks/<task-name>/environment
docker build -t mirrors.tencent.com/skill_and_mcp/skillsbench/<task-name>:latest .
docker push mirrors.tencent.com/skill_and_mcp/skillsbench/<task-name>:latest
```

## 运行方式

### 跑单个 task

```bash
cd nexus
bash workspace/scripts/run_skills_oracle_gongfeng.sh -t <task-name> -o <output-name> -c 1
```

### 跑多个 task

```bash
bash workspace/scripts/run_skills_oracle_gongfeng.sh \
  -t task1,task2,task3 \
  -o <output-name> \
  -c 5
```

### 跑全量（87 个 task）

```bash
bash workspace/scripts/run_skills_oracle_gongfeng.sh -o fullrun_$(date +%m%d) -c 87
```

### 查看结果

结果存储在 SQLite 数据库中：

```bash
#数据看板
uv run nexus inspect-batch outputs/<output-name>.db

# 总体统计
sqlite3 outputs/<output-name>.db "
  SELECT
    SUM(CASE WHEN json_extract(task_output, '$.score') = 1.0 THEN 1 ELSE 0 END) as passed,
    COUNT(*) as total
  FROM runs WHERE status='success'
"

# 查看失败的 task
sqlite3 outputs/<output-name>.db "
  SELECT run_id, status, json_extract(task_output, '$.score') as score
  FROM runs
  WHERE status='failed' OR json_extract(task_output, '$.score') != 1.0
  ORDER BY run_id
"
```

## 关键适配改动

### nexus generator（generator_gongfeng.py）

1. **RPC 协议改为 SocketIO**：Engine.IO 心跳保活防止长时间运行的命令被中间代理断开（之前用 HTTP 长连接会 ServerDisconnectedError）
2. **按 task.toml 分配沙箱资源**：读取每个 task 的 cpus/memory_mb 配置，受沙箱上限约束（max 4 CPU, 12GB memory），而不是全局写死 2 核 10GB


## 未通过的 task（5/87）

### 需要外部 API/账号支持（2 个）

| Task | 依赖 | 说明 |
|------|------|------|
| mhc-layer-impl | Modal GPU 平台 | 计费额度受限 |
| scheduling-email-assistant | Google Calendar API | 需要有效 OAuth token 且日历状态与期望值匹配 |

### 模型已下线（1 个）

| Task | 说明 |
|------|------|
| pedestrian-traffic-counting | gemini-2.0-flash-exp 已被 Google 下线，换模型会改变做题流程，且统计次数波动大 |

### 参考解法本身有问题（2 个）

这两个 task 在同事本地 Docker 上也是 score=0，不是沙箱环境问题：

| Task | 说明 |
|------|------|
| fix-visual-stability | solve.sh 引入 1s setTimeout 延迟 + 缺少 fetch 错误处理，Playwright 5s 内等不到产品卡片 |
| speaker-diarization-subtitles | solve.sh 产出 DER=104%（阈值≤20%），语音分割质量远不达标 |

## 注意事项

1. **reserves-at-risk-calc** 的 test 期望值基于 IMF 金价数据，每月可能需要重新更新
2. **偶发失败**：部分 task 依赖外部 API（Whisper、LLM 等），可能偶尔返回异常结果导致 score=0，重跑通常可解决（如 citation-check, r2r-mpc-control, pddl-tpp-planning）
3. **taxonomy-tree-merge** 镜像必须使用 CPU-only PyTorch 2.2.0（CUDA 版和新版 PyTorch 在沙箱无 GPU 环境中 import 极慢），solve.sh 中已设置 `TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1` 和线程数限制
4. **并发数**：建议 87 或更高，每个 task 在独立沙箱容器中运行，不互相影响
