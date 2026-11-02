#!/bin/bash
# one_agent_infer — 多模型推理/评测脚本（原 YB OneAgentBench runner）
#
# 依次使用多个模型评测，每个模型的结果存在各自的路径下：
#   outputs/{bench_name}_{ts}/{model_name}/
#     ├── results.jsonl
#     ├── results.db
#     ├── files/
#     └── html/
#
# 用法:
#   bash run_infer.sh                          # 默认参数
#   bash run_infer.sh --max-concurrent 6       # 自定义并发数
#   bash run_infer.sh --csig-api-key sk-xxx    # 注入 CSIG 私有 key（也可放 config/.env）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 从 SCRIPT_DIR 向上查找包含 pyproject.toml 的目录作为 NEXUS_ROOT
_dir="$SCRIPT_DIR"
while [ "$_dir" != "/" ]; do
    if [ -f "$_dir/pyproject.toml" ]; then
        NEXUS_ROOT="$_dir"
        break
    fi
    _dir="$(dirname "$_dir")"
done
if [ -z "${NEXUS_ROOT:-}" ]; then
    echo "ERROR: Cannot find nexus project root (no pyproject.toml found above $SCRIPT_DIR)"
    exit 1
fi

# 默认参数
BENCH_NAME="YB_OneAgentBench_V0.2"
BENCH_ROOT="${NEXUS_ROOT}/yb-agent-openbench/benchmarks/releases/${BENCH_NAME}"
CONFIG="${SCRIPT_DIR}/config/base.yaml"
OUTPUT_BASE="${NEXUS_ROOT}/outputs"
MAX_CONCURRENT=5
RESUME_DIR=""  # 如果指定，复用已有目录（断点续跑）
LIMIT=""       # 只跑前 N 条
NUDGE_ON_EMPTY=""  # 空响应时 prefill 引导次数（0=关闭）
ENABLE_AUQ="${ENABLE_AUQ:-true}"  # 是否启用 ask_user_question 工具（true/false，默认 true）
HISTORY_MODE="${HISTORY_MODE:-qa_only}"  # 多轮 session 中 history 拼接方式：
                                         # "qa_only"=只保留 user query + assistant answer（默认）
                                         # "full"=完整保留所有 message（含 tool_calls 和 tool response）

# 待评测的模型列表
MODELS=(
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.2"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.3"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.4"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.5"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.4.1"
    # "openai/glm-5"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.2.8-step600"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.2.4-step600_HF_w8a8c8"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.3.3-mtp-step845"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.3.2-mtp-step845_w8a8c8"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.2.4-step600-2"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.2.3-step600"
    # "openai/HY3.0-dev-0405-OneAgent-SFT-v0.3.4-mtp-step614"
    # "openai/aws-claude-sonnet-4-6"
    # "openai/HY3.0-dev-0424-OneAgent-SFT-v1.3.2-mtp-HF-step465"
    # "openai/HY3.0-dev-0424-OneAgent-SFT-v1.3.3-mtp-HF-step465"
    # "openai/HY3.0-dev-0424-OneAgent-SFT-v1.2.2-mtp-step381-smooth-2"
    # "openai/HY3.0-dev-0424-OneAgent-SFT-v1.2.5-mtp-HF-step66"
    # "openai/deepops-claude-opus-4-6"
    # "openai/hy3-dev0405-jim"
    # "openai/dev-0424"
    # "openai/claude-sonnet-4.6"
    # "openai/claude-opus-4-6"
    "openai/taiji/hy3-a3b-sft-v2.3.1.1-step310_HF"
    # "openai/aws-claude-opus-4-6"
    # "openai/claude-sonnet-4-5-20250929"
)

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --bench-root)     BENCH_ROOT="$2";     shift 2 ;;
        --bench-name)     BENCH_NAME="$2";     shift 2 ;;
        --config)         CONFIG="$2";         shift 2 ;;
        --resume)         RESUME_DIR="$2";     shift 2 ;;
        --limit)          LIMIT="$2";          shift 2 ;;
        --sandbox-token)  SANDBOX_TOKEN="$2";  shift 2 ;;
        --csig-api-key)   ONE_AGENT_CSIG_API_KEY="$2"; shift 2 ;;
        --nudge-on-empty) NUDGE_ON_EMPTY="$2"; shift 2 ;;
        --enable-auq)     ENABLE_AUQ="true";  shift ;;
        --disable-auq)    ENABLE_AUQ="false"; shift ;;
        --history-mode)   HISTORY_MODE="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# 确定输出根目录：续跑用已有目录，否则新建带时间戳的目录
if [ -n "$RESUME_DIR" ]; then
    OUTPUT_DIR="$RESUME_DIR"
    echo "*** RESUME MODE: reusing $OUTPUT_DIR ***"
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="${OUTPUT_BASE}/${BENCH_NAME}_${TIMESTAMP}"
fi

# 检测 nexus 命令
if command -v nexus &>/dev/null; then
    NEXUS_CMD="nexus"
else
    NEXUS_CMD="python3 -m nexus"
fi

# 先加载 config/.env —— 里面可能包含：
#   COS_SECRET_ID / COS_SECRET_KEY / COS_REGION / COS_BUCKET   （用于 parse_file 上传）
#   SANDBOX_TOKEN                                              （沙箱鉴权）
#   以及其它任意想注入的环境变量
# 命令行参数（如 --sandbox-token / --csig-api-key）已经在上面的 argparse 阶段
# 赋给了 shell 变量；这里 source 时只对"当前 shell 中尚未设置"的变量生效是不够
# 的 —— bash 的 source 会直接覆盖同名变量。为了让"命令行 > .env"的优先级成立，
# 我们先把已经来自命令行的值保存下来，source 之后再还原。
_CLI_SANDBOX_TOKEN="${SANDBOX_TOKEN:-}"
_CLI_CSIG_API_KEY="${ONE_AGENT_CSIG_API_KEY:-}"

ENV_FILE="${SCRIPT_DIR}/config/.env"
if [ -f "$ENV_FILE" ]; then
    set -a  # 将随后 source 的变量自动 export
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "[INFO] Loaded env from $ENV_FILE"
else
    echo "[WARN] Env file not found: $ENV_FILE"
fi

# 命令行显式传入的 --sandbox-token 优先级最高
if [ -n "$_CLI_SANDBOX_TOKEN" ]; then
    SANDBOX_TOKEN="$_CLI_SANDBOX_TOKEN"
fi
unset _CLI_SANDBOX_TOKEN

# 命令行显式传入的 --csig-api-key 优先级最高（覆盖 .env 中的同名值）
if [ -n "$_CLI_CSIG_API_KEY" ]; then
    ONE_AGENT_CSIG_API_KEY="$_CLI_CSIG_API_KEY"
fi
unset _CLI_CSIG_API_KEY
# 仅在直连模式下校验 / 导出 CSIG key（litellm router 模式不需要）。
if [ "${ONE_AGENT_LLM_CLIENT:-litellm}" = "one_agent_direct" ]; then
    if [ -z "${ONE_AGENT_CSIG_API_KEY:-}" ] || \
       [ "${ONE_AGENT_CSIG_API_KEY}" = "REPLACE_WITH_YOUR_CSIG_API_KEY" ]; then
        echo "[WARN] ONE_AGENT_CSIG_API_KEY not set (via --csig-api-key or config/.env);"
        echo "       direct CSIG calls will fail with 401. Skip if you only call Taiji."
    else
        export ONE_AGENT_CSIG_API_KEY
    fi
fi

# 通用环境变量
export YB_BENCH_PATH="$BENCH_ROOT"
export YB_BENCH_CONFIG="$CONFIG"

# LLM client 选择：
#   - litellm（默认）：走原 litellm + 独立 router server（需另外起 proxy_server.py），
#     此时 ONE_AGENT_BASE_URL 指向 router。保持与原始逻辑一致（已验证稳定）。
#   - one_agent_direct：进程内直连 CSIG/Taiji，无需启动 router server，
#     无需依赖 litellm_router 仓库。CSIG/Taiji 的地址与密钥见 config/.env
#     （ONE_AGENT_CSIG_* / ONE_AGENT_TAIJI_*），未配置时回退内置默认值。
#     直连为新增功能、尚未充分验证；如需启用：ONE_AGENT_LLM_CLIENT=one_agent_direct bash run_infer.sh
export ONE_AGENT_LLM_CLIENT="${ONE_AGENT_LLM_CLIENT:-litellm}"
echo "[INFO] llm client: ${ONE_AGENT_LLM_CLIENT}"
# 以下两个仅在 litellm（router）模式下生效；direct 模式忽略。
export ONE_AGENT_API_KEY="${ONE_AGENT_API_KEY:-sk-placeholder}"
export ONE_AGENT_BASE_URL="${ONE_AGENT_BASE_URL:-http://127.0.0.1:8001/v1}"

# Taiji prompt cache（一致性哈希路由）:
#   - StrategyType=ConsistentHash + SessionID=<run_id>: 同一 session 多轮请求落同一实例,
#     命中该实例 prefix/KV cache, 减少首字耗时
#   - 设置 ONE_AGENT_TAIJI_STRATEGY="" 即关闭 (走 Taiji 默认轮询)
#   - SESSION_ID 留空时 direct client 会用 run_id 兜底
export ONE_AGENT_TAIJI_STRATEGY="${ONE_AGENT_TAIJI_STRATEGY:-ConsistentHash}"
export ONE_AGENT_TAIJI_SESSION_ID="${ONE_AGENT_TAIJI_SESSION_ID:-}"
echo "[INFO] taiji prompt cache: strategy=${ONE_AGENT_TAIJI_STRATEGY:-<off>} session_id=${ONE_AGENT_TAIJI_SESSION_ID:-<auto run_id>}"
# 采样参数（base.yaml 默认 0.7/0.8；这里默认 greedy=0/1.0 用于对照）
# 调用者可通过环境变量覆盖，例如：ONE_AGENT_TEMPERATURE=0.3 ONE_AGENT_TOP_P=0.9 ./run_infer.sh
# export ONE_AGENT_TEMPERATURE="${ONE_AGENT_TEMPERATURE:-0}"
# export ONE_AGENT_TOP_P="${ONE_AGENT_TOP_P:-1.0}"
# echo "[INFO] sampling: temperature=${ONE_AGENT_TEMPERATURE} top_p=${ONE_AGENT_TOP_P}"
if [ -n "${SANDBOX_TOKEN:-}" ]; then
    export SANDBOX_TOKEN
else
    echo "[WARN] SANDBOX_TOKEN not set (via --sandbox-token or config/.env);"
    echo "       sandbox-dependent tools may fail authentication."
fi
# Gongfeng runtime purpose: prod | eval | test
export ONE_AGENT_PURPOSE="test"
echo "[INFO] runtime purpose: ${ONE_AGENT_PURPOSE}"
if [ -n "$LIMIT" ]; then
    export YB_BENCH_LIMIT="$LIMIT"
fi
if [ -n "$NUDGE_ON_EMPTY" ]; then
    export NUDGE_ON_EMPTY
fi
export ENABLE_AUQ
echo "[INFO] ask_user_question: ${ENABLE_AUQ}"
export HISTORY_MODE
echo "[INFO] history_mode: ${HISTORY_MODE}"

if [ -z "${COS_SECRET_ID:-}" ] || [ -z "${COS_SECRET_KEY:-}" ] || [ -z "${COS_BUCKET:-}" ]; then
    echo "[WARN] COS credentials not set (COS_SECRET_ID/COS_SECRET_KEY/COS_BUCKET)."
    echo "       parse_file via MCP will fail; agent will only see file paths in sandbox."
fi

BENCH_JSONL="${BENCH_ROOT}/${BENCH_NAME}_benchmark.jsonl"
export YB_BENCH_JSONL="$BENCH_JSONL"

echo "╔══════════════════════════════════════════════════════╗"
echo "║      one_agent_infer — Multi-Model Evaluation       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Models to evaluate: ${#MODELS[@]}"
for m in "${MODELS[@]}"; do echo "  - ${m##*/}"; done
echo "Benchmark:      $BENCH_ROOT"
echo "Max concurrent: $MAX_CONCURRENT"
echo "History mode:   $HISTORY_MODE"
echo "ask_user_question: $ENABLE_AUQ"
echo ""

# ── 逐模型评测 ──
for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT="${MODEL##*/}"
    RESULT_DIR="${OUTPUT_DIR}/${MODEL_SHORT}"
    DATABASE="${RESULT_DIR}/results.db"
    RESULTS="${RESULT_DIR}/results.jsonl"
    FILES_DIR="${RESULT_DIR}/files"
    HTML_DIR="${RESULT_DIR}/html"
    TRAJ_DIR="${RESULT_DIR}/trajectory"

    export ONE_AGENT_MODEL="$MODEL"
    export NEXUS_OUTPUT_DIR="$RESULT_DIR"    # task.py 用此路径增量写出 per-turn trajectory/files
    mkdir -p "$RESULT_DIR"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Model: $MODEL_SHORT"
    echo "  Results dir: $RESULT_DIR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Step 1: 运行评测
    # NOTE: run-batch may exit non-zero when some tasks fail (partial success).
    # We still want Step 2/3 to process whatever results were produced, so we
    # capture the exit code instead of letting set -e kill the script.
    echo "── Step 1: Running evaluation ──"
    BATCH_ARGS=(
        --generator "${SCRIPT_DIR}/generator.py"
        --database "$DATABASE"
        --export-path "$RESULTS"
        --export-format jsonl
        --max-concurrent "$MAX_CONCURRENT"
        --timeout 18000
    )
    STEP1_RC=0
    $NEXUS_CMD run-batch "${BATCH_ARGS[@]}" || STEP1_RC=$?
    if [ "$STEP1_RC" -ne 0 ]; then
        echo "  [WARN] run-batch exited with code $STEP1_RC (some tasks may have failed)"
        echo "         Proceeding with Step 2/3 on available results..."
    fi

    # Guard: if results.jsonl doesn't exist or is empty, skip Step 2/3
    if [ ! -s "$RESULTS" ]; then
        echo "  [ERROR] No results produced ($RESULTS missing or empty). Skipping Step 2/3."
        continue
    fi

    # Step 2: 提取产出文件 & 生成 HTML
    echo ""
    echo "── Step 2: Extracting files & generating HTML ──"

    python3 -c "
import json, base64, os, sys
sys.path.insert(0, '${NEXUS_ROOT}')

results_path = '${RESULTS}'
bench_path = '${BENCH_JSONL}'
files_dir = '${FILES_DIR}'
html_dir = '${HTML_DIR}'
traj_dir = '${TRAJ_DIR}'

from nexus.extensions.tasks.one_agent_infer.export_html import build_html

bench_map = {}
if os.path.exists(bench_path):
    with open(bench_path, 'r', encoding='utf-8') as bf:
        for line in bf:
            line = line.strip()
            if line:
                rec = json.loads(line)
                bench_map[rec['task_id']] = rec

os.makedirs(files_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)
os.makedirs(traj_dir, exist_ok=True)
total_files = 0
total_html = 0
total_traj = 0

def process_one_turn(task_id, trajectory_str, downloaded_files, bench_map, chat_trajs=None, tools=None, rec_metadata=None, system_msg=None):
    global total_files, total_html, total_traj

    # Parse trajectory
    trajectory = []
    if chat_trajs and isinstance(chat_trajs[-1], list):
        trajectory = [m for m in chat_trajs[-1] if m.get('role') != 'system']
    else:
        try:
            trajectory = json.loads(trajectory_str) if trajectory_str else []
        except json.JSONDecodeError:
            trajectory = []

    has_user_in_traj = any(m.get('role') == 'user' for m in trajectory)

    # Export trajectory
    full_traj = chat_trajs[-1] if chat_trajs and isinstance(chat_trajs[-1], list) else trajectory
    # Prepend system prompt if missing and available
    if full_traj and full_traj[0].get('role') != 'system' and system_msg:
        full_traj = [system_msg] + full_traj
    traj_data = {'trajectory': full_traj, 'tools': tools or []}
    # Attach optional metadata fields from benchmark and results
    bench_rec = bench_map.get(task_id, {})
    if bench_rec.get('source_metadata'):
        traj_data['source_metadata'] = bench_rec['source_metadata']
    if rec_metadata:
        traj_data['metadata'] = rec_metadata
    if bench_rec.get('meta'):
        traj_data['meta'] = bench_rec['meta']
    traj_path = os.path.join(traj_dir, f'{task_id}.jsonl')
    with open(traj_path, 'w', encoding='utf-8') as out:
        out.write(json.dumps(traj_data, ensure_ascii=False) + '\n')
    print(f'  [TRAJ] {traj_path} ({len(full_traj)} msgs)')
    total_traj += 1

    # Extract output files
    for filename, b64content in (downloaded_files or {}).items():
        outpath = os.path.join(files_dir, task_id, filename)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, 'wb') as out:
            out.write(base64.b64decode(b64content))
        print(f'  [FILE] {outpath} ({os.path.getsize(outpath):,} bytes)')
        total_files += 1

    # Generate HTML
    bench_rec = bench_map.get(task_id, {})
    query = bench_rec.get('query', '')
    metadata = {
        'L1': bench_rec.get('L1', ''),
        'L2': bench_rec.get('L2', ''),
        'difficulty': bench_rec.get('difficulty', ''),
        'expected_output': bench_rec.get('expected_output', ''),
    }
    html_content = build_html(task_id, query, trajectory, metadata, downloaded_files or {},
                              skip_user_message=has_user_in_traj)
    html_path = os.path.join(html_dir, f'{task_id}.html')
    with open(html_path, 'w', encoding='utf-8') as out:
        out.write(html_content)
    print(f'  [HTML] {html_path} ({len(trajectory)} msgs, {len(html_content):,} bytes)')
    total_html += 1


def _split_timeout_session_trajectory(session_id, full_msgs, tools, bench_map, rec_metadata):
    """For timeout sessions: split recovered base trajectory into per-turn
    cumulative trajectory files so evaluation can score already-completed turns.
    Normal sessions generate per-turn files: session_id.jsonl, session_id_t2.jsonl, etc.
    This function replicates that structure from the single recovered trajectory.
    """
    global total_traj

    # Separate system messages from the rest
    system_msgs = [m for m in full_msgs if m.get('role') == 'system']
    non_system = [m for m in full_msgs if m.get('role') != 'system']

    # Find user message indices to identify turn boundaries
    user_indices = [i for i, m in enumerate(non_system) if m.get('role') == 'user']
    if not user_indices:
        return

    # For each turn, find where the turn ends (= next user msg or end of list)
    # Then check if the turn has a complete assistant reply (non-empty content)
    for turn_num, user_idx in enumerate(user_indices, start=1):
        next_user_idx = user_indices[turn_num] if turn_num < len(user_indices) else len(non_system)

        # Check if this turn has a substantive assistant reply
        has_complete_reply = False
        for j in range(user_idx + 1, next_user_idx):
            msg = non_system[j]
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if content and content.strip() and len(content.strip()) > 5:
                    has_complete_reply = True

        if not has_complete_reply:
            # Turn was not completed (timeout hit during this turn); stop here
            break

        # Build cumulative trajectory up to end of this turn (mimics normal behavior)
        cumulative = system_msgs + non_system[:next_user_idx]

        # Determine task_id for this turn
        if turn_num == 1:
            turn_task_id = session_id
        else:
            turn_task_id = f'{session_id}_t{turn_num}'

        # Write trajectory file
        bench_rec = bench_map.get(turn_task_id, {})
        traj_data = {'trajectory': cumulative, 'tools': tools}
        if bench_rec.get('source_metadata'):
            traj_data['source_metadata'] = bench_rec['source_metadata']
        if rec_metadata:
            traj_data['metadata'] = rec_metadata
        if bench_rec.get('meta'):
            traj_data['meta'] = bench_rec['meta']
        traj_path = os.path.join(traj_dir, f'{turn_task_id}.jsonl')
        with open(traj_path, 'w', encoding='utf-8') as out:
            out.write(json.dumps(traj_data, ensure_ascii=False) + '\n')
        print(f'  [TRAJ-TIMEOUT] {traj_path} ({len(cumulative)} msgs, turn {turn_num})')
        total_traj += 1


with open(results_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        to = rec.get('task_output', {}) or {}
        meta = rec.get('metadata', {}) or {}
        pr = rec.get('processor_results', {}) or {}
        chat_pr = pr.get('chat', {}) if isinstance(pr, dict) else {}
        chat_trajs = chat_pr.get('trajectories', []) if isinstance(chat_pr, dict) else []
        tools = chat_pr.get('tools', []) if isinstance(chat_pr, dict) else []

        # Check for multi-turn results
        turn_results = to.get('turn_results', [])
        if turn_results and len(turn_results) > 1:
            # Extract system prompt from chat_trajs for multi-turn trajectory export
            _system_msg = None
            if chat_trajs:
                for _t in chat_trajs:
                    if isinstance(_t, list) and _t and _t[0].get('role') == 'system':
                        _system_msg = _t[0]
                        break
            # Multi-turn session: process each turn separately
            for tr in turn_results:
                turn_task_id = tr.get('task_id', 'unknown')
                process_one_turn(
                    turn_task_id,
                    tr.get('trajectory', ''),
                    tr.get('downloaded_files', {}),
                    bench_map,
                    tools=tools,
                    rec_metadata=meta,
                    system_msg=_system_msg,
                )
        elif rec.get('status') == 'timeout' and (meta.get('n_turns', 1) > 1):
            # ── Timeout multi-turn session: split recovered trajectory into per-turn files ──
            # The session timed out before task.py could return turn_results normally.
            # We have a single recovered trajectory (from trace) that contains all
            # messages up to the point of timeout. Split it into per-turn files so
            # that already-completed turns can be properly evaluated.
            session_id = meta.get('session_id') or to.get('task_id') or meta.get('task_id', 'unknown')
            # Get the full trajectory from processor_results or task_output
            full_msgs = []
            if chat_trajs and isinstance(chat_trajs[-1], list):
                full_msgs = chat_trajs[-1]
            else:
                traj_str = to.get('trajectory', '')
                if traj_str:
                    try:
                        parsed = json.loads(traj_str)
                        if isinstance(parsed, list):
                            full_msgs = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
            if full_msgs:
                _split_timeout_session_trajectory(
                    session_id, full_msgs, tools, bench_map, meta
                )
            else:
                # Fallback: process as single-turn (only base task_id gets trajectory)
                task_id = to.get('task_id') or meta.get('task_id', 'unknown')
                process_one_turn(
                    task_id,
                    to.get('trajectory', ''),
                    to.get('downloaded_files', {}),
                    bench_map,
                    chat_trajs=chat_trajs if chat_trajs else None,
                    tools=tools,
                    rec_metadata=meta,
                )
        else:
            # Single-turn: use top-level fields (backward compat)
            task_id = to.get('task_id') or meta.get('task_id', 'unknown')
            process_one_turn(
                task_id,
                to.get('trajectory', ''),
                to.get('downloaded_files', {}),
                bench_map,
                chat_trajs=chat_trajs if chat_trajs else None,
                tools=tools,
                rec_metadata=meta,
            )

print(f'  Extracted {total_files} files, generated {total_html} HTML reports, {total_traj} trajectory files')
"

    # Step 3: 展开 results.jsonl，每个 turn 一行
    echo ""
    echo "── Step 3: Expanding results.jsonl to per-turn rows ──"

    python3 -c "
import json, os, sqlite3

results_path = '${RESULTS}'
database_path = '${DATABASE}'
expanded_path = results_path + '.tmp'

# ── De-duplicate results.jsonl by run_id BEFORE expansion ──
# When resuming from a checkpoint, export may append new session-level records
# on top of previously-expanded per-turn rows, causing duplicates after expansion.
# Fix: keep only the LAST occurrence of each run_id (newest result wins).
dedup_lines = {}  # run_id -> line (last-wins)
dedup_order = []  # preserve insertion order
with open(results_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rid = rec.get('run_id', '')
        if rid not in dedup_lines:
            dedup_order.append(rid)
        dedup_lines[rid] = line
# Write back deduplicated
with open(results_path, 'w', encoding='utf-8') as f:
    for rid in dedup_order:
        f.write(dedup_lines[rid] + '\n')
if len(dedup_order) < len(dedup_lines) + (0):  # always true, just a placeholder
    pass
n_before = sum(1 for _ in open(results_path))
print(f'  De-duplicated results.jsonl: {n_before} unique run_ids')

# ── Load per-run token usage from trace events in DB ──
run_token_usage = {}
if os.path.exists(database_path):
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute(
            \"SELECT run_id, event_data FROM run_trace_events \"
            \"WHERE event_data LIKE '%token%'\"
        )
        for run_id, ed_str in cursor.fetchall():
            ed = json.loads(ed_str)
            usage = (ed.get('attributes') or {}).get('outputs', {}).get('usage', {})
            if not usage:
                continue
            if run_id not in run_token_usage:
                run_token_usage[run_id] = {
                    'prompt_tokens': 0, 'completion_tokens': 0,
                    'total_tokens': 0, 'llm_calls': 0,
                }
            ru = run_token_usage[run_id]
            ru['prompt_tokens'] += usage.get('prompt_tokens', 0) or 0
            ru['completion_tokens'] += usage.get('completion_tokens', 0) or 0
            ru['total_tokens'] += usage.get('total_tokens', 0) or 0
            ru['llm_calls'] += 1
        conn.close()
        print(f'  Loaded token usage for {len(run_token_usage)} runs from DB')
    except Exception as e:
        print(f'  [WARN] Failed to load token usage from DB: {e}')

with open(results_path, 'r', encoding='utf-8') as fin, \
     open(expanded_path, 'w', encoding='utf-8') as fout:
    total_rows = 0
    seen_task_ids = set()

    # Load bench data to know all task_ids per session (for filling missing turns)
    bench_sessions = {}  # session_id -> [task_id_t1, task_id_t2, ...]
    bench_jsonl = os.getenv('YB_BENCH_JSONL', '')
    if bench_jsonl and os.path.exists(bench_jsonl):
        with open(bench_jsonl, 'r', encoding='utf-8') as bf:
            for bline in bf:
                bline = bline.strip()
                if not bline:
                    continue
                brec = json.loads(bline)
                sid = brec.get('session_id') or brec.get('task_id', '')
                bench_sessions.setdefault(sid, []).append({
                    'task_id': brec['task_id'],
                    'turn': brec.get('turn', 1),
                })
        for sid in bench_sessions:
            bench_sessions[sid].sort(key=lambda x: x['turn'])

    for line in fin:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        to = rec.get('task_output', {}) or {}
        meta = rec.get('metadata', {}) or {}
        turn_results = to.get('turn_results', []) or []
        orig_run_id = rec.get('run_id', '')

        # ── Extract fields needed by downstream SFT pipeline ──
        pr = rec.get('processor_results') or {}
        chat_pr = pr.get('chat', {}) if isinstance(pr, dict) else {}
        all_trajs = chat_pr.get('trajectories', []) if isinstance(chat_pr, dict) else []
        chat_tools = chat_pr.get('tools', []) if isinstance(chat_pr, dict) else []

        agent_trajs = [t for t in all_trajs if t and t[0].get('role') == 'system']
        judge_trajs = [t for t in all_trajs if t and t[0].get('role') != 'system']

        # Attach session-level token usage
        session_usage = run_token_usage.get(orig_run_id)

        rec.pop('agent_output', None)
        rec.pop('processor_results', None)

        if turn_results:
            for turn_idx, tr in enumerate(turn_results):
                tid = tr['task_id']
                if tid in seen_task_ids:
                    continue
                seen_task_ids.add(tid)

                turn_rec = dict(rec)
                turn_rec['task_output'] = {
                    'task_id': tid,
                    'session_id': to.get('session_id', ''),
                    'turn': tr.get('turn', 1),
                    'trajectory': tr.get('trajectory', ''),
                    'answer': tr.get('answer', ''),
                    'expected_output': to.get('expected_output', ''),
                    'downloaded_files': tr.get('downloaded_files', {}),
                    'error': tr.get('error'),
                }
                # Build per-turn trajectory: prefer turn_results[].trajectory
                # (which records only this turn's context), since agent_trajs from
                # processor may be the full session's accumulated trajectory.
                per_turn_traj = []
                traj_str = tr.get('trajectory', '')
                if traj_str:
                    try:
                        parsed_traj = json.loads(traj_str)
                        if isinstance(parsed_traj, list) and parsed_traj:
                            per_turn_traj = parsed_traj
                    except (json.JSONDecodeError, TypeError):
                        pass
                # Fallback to agent_trajs if turn has no trajectory string
                if not per_turn_traj and turn_idx < len(agent_trajs):
                    per_turn_traj = agent_trajs[turn_idx]
                # Prepend system prompt if missing
                if per_turn_traj and per_turn_traj[0].get('role') != 'system':
                    if agent_trajs and agent_trajs[0] and agent_trajs[0][0].get('role') == 'system':
                        per_turn_traj = [agent_trajs[0][0]] + per_turn_traj
                pr_chat = {
                    'trajectories': [per_turn_traj] if per_turn_traj else [],
                    'tools': chat_tools,
                }
                if judge_trajs:
                    pr_chat['judge_trajectories'] = judge_trajs
                turn_rec['processor_results'] = {'chat': pr_chat}
                turn_rec['run_id'] = 'yb_bench_' + tid
                turn_meta = dict(meta)
                turn_meta['task_id'] = tid
                turn_meta['turn'] = tr.get('turn', 1)
                turn_rec['metadata'] = turn_meta
                # Token usage: session total attached to each turn
                if session_usage:
                    turn_rec['token_usage'] = session_usage
                fout.write(json.dumps(turn_rec, ensure_ascii=False) + '\n')
                total_rows += 1
        else:
            if agent_trajs or judge_trajs or chat_tools:
                pr_chat = {
                    'trajectories': agent_trajs,
                    'tools': chat_tools,
                }
                if judge_trajs:
                    pr_chat['judge_trajectories'] = judge_trajs
                rec['processor_results'] = {'chat': pr_chat}
            if session_usage:
                rec['token_usage'] = session_usage

            # Write the session-level record (turn 1 or single-turn)
            task_id = to.get('task_id') or meta.get('task_id', '')
            seen_task_ids.add(task_id)
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            total_rows += 1

            # For multi-turn sessions with no turn_results (e.g. timeout),
            # fill missing turns from bench data so total rows = 190.
            # For timeout sessions, try to extract per-turn answers from the
            # recovered trajectory (which may contain completed earlier turns).
            session_id = meta.get('session_id') or task_id
            n_turns = meta.get('n_turns', 1)
            if n_turns > 1 and session_id in bench_sessions:
                # Parse the recovered trajectory to extract per-turn answers
                _recovered_turn_answers = {}  # turn_num -> answer
                _full_msgs = []
                if agent_trajs and agent_trajs[-1]:
                    _full_msgs = agent_trajs[-1]
                elif to.get('trajectory'):
                    try:
                        _parsed = json.loads(to['trajectory'])
                        if isinstance(_parsed, list):
                            _full_msgs = _parsed
                    except (json.JSONDecodeError, TypeError):
                        pass

                if _full_msgs:
                    _non_system = [m for m in _full_msgs if m.get('role') != 'system']
                    _user_indices = [i for i, m in enumerate(_non_system) if m.get('role') == 'user']
                    for _ti, _ui in enumerate(_user_indices, start=1):
                        _next_ui = _user_indices[_ti] if _ti < len(_user_indices) else len(_non_system)
                        # Find the last assistant content with substance in this turn
                        _last_answer = ''
                        for _j in range(_ui + 1, _next_ui):
                            _m = _non_system[_j]
                            if _m.get('role') == 'assistant':
                                _c = _m.get('content', '')
                                if _c and _c.strip() and len(_c.strip()) > 5:
                                    _last_answer = _c
                        if _last_answer:
                            _recovered_turn_answers[_ti] = _last_answer

                for bt in bench_sessions[session_id]:
                    tid = bt['task_id']
                    if tid in seen_task_ids:
                        continue
                    seen_task_ids.add(tid)
                    fill_rec = dict(rec)
                    fill_rec['run_id'] = 'yb_bench_' + tid
                    fill_meta = dict(meta)
                    fill_meta['task_id'] = tid
                    fill_meta['turn'] = bt['turn']
                    fill_rec['metadata'] = fill_meta
                    # Use recovered answer if this turn was completed before timeout
                    _turn_answer = _recovered_turn_answers.get(bt['turn'], '')
                    _turn_error = None if _turn_answer else (rec.get('error') or 'session terminated before this turn')
                    fill_rec['task_output'] = {
                        'task_id': tid,
                        'session_id': session_id,
                        'turn': bt['turn'],
                        'trajectory': '',
                        'answer': _turn_answer,
                        'expected_output': to.get('expected_output', ''),
                        'downloaded_files': {},
                        'error': _turn_error,
                    }
                    # If we recovered an answer, mark status as success for this turn
                    if _turn_answer:
                        fill_rec['status'] = 'success'
                        fill_rec['error'] = None
                    fout.write(json.dumps(fill_rec, ensure_ascii=False) + '\n')
                    total_rows += 1

os.replace(expanded_path, results_path)

# ── Final safety: deduplicate expanded results by task_id (last-wins) ──
final_dedup = {}
final_order = []
with open(results_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        tid = (rec.get('task_output') or {}).get('task_id') or (rec.get('metadata') or {}).get('task_id', '')
        if tid not in final_dedup:
            final_order.append(tid)
        final_dedup[tid] = line
n_dedup = len(final_order)
if n_dedup < total_rows:
    with open(results_path, 'w', encoding='utf-8') as f:
        for tid in final_order:
            f.write(final_dedup[tid] + '\n')
    print(f'  [DEDUP] Removed {total_rows - n_dedup} duplicate task_ids -> {n_dedup} final rows')
else:
    print(f'  Expanded to {total_rows} rows in results.jsonl (no duplicates)')
"

    echo ""
    echo "  ✓ ${MODEL_SHORT} done → ${RESULT_DIR}/"
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                  All models done!                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT="${MODEL##*/}"
    echo "  ${OUTPUT_DIR}/${MODEL_SHORT}/"
done
