#!/usr/bin/env bash
# web_judger 验证 —— 第 1 步：给数据集生成 flow 契约，并落成一份带契约的新数据集。
#
#   bash nexus/extensions/tasks/artifact/judges/web_judger/run_1_flow.sh
#
# 只调纯文字 LLM（不起容器、不碰浏览器），10 条大约半分钟。 产出：
#   experiments/web_judger_verify/flows.jsonl     每行 {instance_id, ok, contract, issues}
#   experiments/web_judger_verify/dataset.jsonl   原数据 + flow 字段 + judge_name=web_judger
#
# 契约是 f(需求)、和产物无关，所以生成一次可以反复跑 judge —— 改 executor/verifier 后
# 不必重跑这一步。
#
# 模型写死在 flow.py 的 GATEWAY_MODEL（opus-4-8 直连 model-eval 网关，不走 nexus 的
# litellm 层）—— 要换模型改那个常量，不是改这里。
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

WJ_DIR='nexus/extensions/tasks/artifact/judges/strategies/web_judger'
SRC="${SRC:-experiments/human2rm_align/subset_3d_web_10.jsonl}"
OUT_DIR="${OUT_DIR:-experiments/web_judger_verify}"

KEY_ALIAS="${KEY_ALIAS:-opus48}"           # 只用它的 key，model/base 由 flow.py 写死
export FLOW_CONCURRENCY="${FLOW_CONCURRENCY:-512}"

# 凭据在 .env 里（M_<alias>_KEY），和 run_batch 共用一份
_ENV_FILE='nexus/extensions/tasks/artifact/.env'
if [[ -f "${_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${_ENV_FILE}"
else
    echo "[flow] 找不到 ${_ENV_FILE} —— 需要 M_${KEY_ALIAS}_KEY" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
echo "[flow] 源数据 : ${SRC}  ($(wc -l < "${SRC}") 行)"
echo "[flow] 凭据   : M_${KEY_ALIAS}_KEY  并发=${FLOW_CONCURRENCY}"

# --- 1) 需求 → flow 契约 -----------------------------------------------------
python3 "${WJ_DIR}/flow.py" \
    --dataset "${SRC}" \
    --out "${OUT_DIR}/flows.jsonl" \
    --key-alias "${KEY_ALIAS}" \
    --concurrency "${FLOW_CONCURRENCY}"

# --- 2) 把契约塞回数据行 -----------------------------------------------------
# pipeline 只认 instance['flow']（解析好的 contract），不做文件 IO —— 拼接在这里做。
python3 - "${SRC}" "${OUT_DIR}/flows.jsonl" "${OUT_DIR}/dataset.jsonl" <<'PY'
import json, sys, collections

src, flows_path, out = sys.argv[1:4]
contracts = {}
for line in open(flows_path, encoding='utf-8'):
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get('ok') and row.get('contract'):
        contracts[str(row['instance_id'])] = row['contract']

kept = dropped = 0
types, flow_counts = collections.Counter(), []
with open(out, 'w', encoding='utf-8') as handle:
    for line in open(src, encoding='utf-8'):
        if not line.strip():
            continue
        inst = json.loads(line)
        contract = contracts.get(str(inst.get('instance_id')))
        if not contract:
            # 没契约的行直接不要 —— 带进去只会在 pipeline 里判 config_error 白跑一趟容器
            dropped += 1
            continue
        inst['flow'] = contract
        # 分派键只有 ``judge_name``。源数据里若残留已废弃的 ``pipeline_name``，
        # 顺手删掉 —— 它早已不被读取，留在输出里只会让人以为还有作用。
        inst['judge_name'] = 'web_judger'
        inst.pop('pipeline_name', None)
        handle.write(json.dumps(inst, ensure_ascii=False) + '\n')
        kept += 1
        types[contract.get('task_type', '?')] += 1
        flow_counts.append(len(contract.get('flows') or []))

print(f'\n[flow] 写出 {kept} 行 → {out}' + (f'（丢弃 {dropped} 行无契约）' if dropped else ''))
print(f'[flow] task_type: {dict(types)}')
if flow_counts:
    flow_counts.sort()
    print(f'[flow] flows/case: min={flow_counts[0]} p50={flow_counts[len(flow_counts)//2]} '
          f'max={flow_counts[-1]} 合计={sum(flow_counts)} 条 flow')
PY

echo
echo "[flow] 看一条契约: python3 ${WJ_DIR}/flow.py --show ${OUT_DIR}/flows.jsonl --index 0"
echo "[flow] 输出是离线 JSONL，不可直接交给 Harbor。请将各条 contract 放入相应任务的 tests/flow.json。"
echo "[flow] 准备好完整 Harbor 任务目录后，统一使用 artifact/run_harbor_test.sh；旧 JSONL 跑批入口已退役。"
