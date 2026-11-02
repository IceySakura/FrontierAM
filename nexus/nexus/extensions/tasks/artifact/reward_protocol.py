"""Reward-service protocol helpers for the artifact RL task.

Self-contained copy (no dependency on ``tasks.frontend``): error codes,
``Judge`` 枚举(判分结局，值即 RL 的 judge_code)、``build_reward_extra_info`` /
``build_judge_metrics``、产物空/不完整的诊断。 与 frontend task 完全独立。
"""

import base64
import binascii

from enum import IntEnum
from typing import Any


class Judge(IntEnum):
    """判分结局 —— 全系统只用这一个变量表示「这条数据算不算数」。

    值是发给 RL 的 judge_code(wire format，hytuner 读它)，**数值永不改动**；名字带主语
    前缀，一眼读出归因：

      - ``ARTIFACT_*``  产物的锅 = **模型问题** → penalty，负分**进梯度**
      - ``ROLLOUT_*``   这一轮没跑完 = 非模型 → maskout
      - ``EVAL_*``      判分/基建侧的锅 = 非模型 → maskout

    maskout 的含义是 score 压到 THRESHOLD(-99) 以下：既不进梯度，**也不进 GRPO 组内
    baseline**（0 分只做到前者，仍会把组内 std 撑大约 2 倍）。

    ``action`` 必须保持是 judge_code 的纯函数 —— 同一个 code 不能有时罚有时屏，否则
    ``training_action`` 与实际行为自相矛盾。要区分就加新 code（hytuner 只在
    nexus_agent_loop.py:962 打日志读 judge_code，从不分支，加 code 是安全的）。
    """

    SUCCESS = 0

    # ── 模型问题：产物有交付但不合格 → penalty，进梯度 ──────────────────────
    ARTIFACT_UNDEPLOYABLE = 1006
    ARTIFACT_MISSING = 1007

    # ── 非模型：rollout 没干净收尾（超时 / 步数上限 / 截断 / agent 崩）→ maskout ──
    # 细分原因不编码进 judge_code，由 artifact_empty_cause 记录，看 artifact/empty/{cause}。
    # 1005(原 INCOMPLETE_PROJECT) 已废弃并入本 code：判据同源(rollout 提前退出)、
    # reward 行为相同(都是 maskout)，剩多少文件由 artifact_file_count 表达。
    ROLLOUT_UNFINISHED = 1003

    # ── 非模型：判分/基建侧故障 → maskout ────────────────────────────────
    EVAL_DEPLOY_SANDBOX_UNAVAILABLE = 2009
    EVAL_VLM_CALL_FAILED = 3001
    EVAL_CONFIG_FAILED = 3002
    EVAL_AGENT_CALL_FAILED = 3003
    NEXUS_TASK_FAILED = 3004
    NEXUS_TASK_TIMEOUT = 3005
    EVAL_UNKNOWN_FAILED = 3099

    @property
    def action(self) -> str:
        if self is Judge.SUCCESS:
            return 'use_score'
        if self in _PENALTY_CODES:
            return 'penalty'
        return 'maskout'

    @property
    def outcome(self) -> str:
        return {'use_score': 'success', 'penalty': 'penalty'}[self.action] if self.action != 'maskout' else 'masked'


_PENALTY_CODES = frozenset({
    Judge.ARTIFACT_UNDEPLOYABLE,
    Judge.ARTIFACT_MISSING,
})
"""判成「模型问题」的 code —— 负分参与训练，**必须**配空串 termination_reason
（否则 hytuner 认出一个 maskout reason 并屏蔽掉，惩罚静默丢失）。"""


_TERM_REASON_BY_CODE = {
    Judge.SUCCESS: '',
    Judge.ARTIFACT_UNDEPLOYABLE: '',
    Judge.ARTIFACT_MISSING: '',
    # 判分侧故障统一报 REWARD_ERROR：模型无责，样本必须屏蔽而不是当低分训。
    Judge.ROLLOUT_UNFINISHED: 'REWARD_ERROR',
    Judge.EVAL_DEPLOY_SANDBOX_UNAVAILABLE: 'REWARD_ERROR',
    Judge.EVAL_VLM_CALL_FAILED: 'REWARD_ERROR',
    Judge.EVAL_CONFIG_FAILED: 'REWARD_ERROR',
    Judge.EVAL_AGENT_CALL_FAILED: 'REWARD_ERROR',
    Judge.EVAL_UNKNOWN_FAILED: 'REWARD_ERROR',
    # HyTuner recognizes these generic task-level reasons directly.
    Judge.NEXUS_TASK_FAILED: 'UNHANDLED_ERROR',
    Judge.NEXUS_TASK_TIMEOUT: 'TASK_TIMEOUT',
}
"""judge_code → hytuner 侧 ``termination_reason``。**这是唯一的跨仓库契约**。

hytuner 不看 judge_code 决定 maskout —— 它看 ``termination_reason`` 是否命中
``MASKOUT_TERMINATION_REASONS``（hytuner/utils/termination_reason.py）。命中则样本
不参与梯度：V1 走 is_maskout 把该 response 从训练组剔除，V2 把 loss_mask 清零。

因此每个 maskout code 必须映射到一个 hytuner **已定义**的枚举成员，否则匹配不上、
maskout 静默失效。判分侧故障用 ``REWARD_ERROR``（语义即「reward 没算出来」，且
hytuner 有现成的 traj/*/reward_scoring/reward_error_ratio 指标）；penalty /
use_score 必须是空串。

细分归因不靠 termination_reason 承载 —— 它要精确等于枚举值，没有携带子类型的余地。
逐 code 的可观测性由 ``build_judge_metrics`` 经 extra_reward_metrics 输出，那条路径
按 response_mask 取有效样本，而 maskout 不动 response_mask，所以故障样本的
judge/* 指标照常可见。

把这层映射做成字典而非散在 if-else 里，是为了让「加了 code 却忘配 term_reason」
在单测里立刻暴露（见 test_term_reason_table_is_consistent）。
"""


def term_reason_for(code: int) -> str:
    """judge_code → termination_reason。未知 code 按 maskout 兜底（宁可屏蔽也不误训）。"""
    try:
        return _TERM_REASON_BY_CODE[Judge(code)]
    except (ValueError, KeyError):
        return _TERM_REASON_BY_CODE[Judge.EVAL_UNKNOWN_FAILED]


ARTIFACT_EMPTY_CAUSES = (
    'agent_timeout',
    'max_steps',
    'response_truncated',
    'format_error',
    'collect_failed',
    'no_files_written',
    'unknown',
)
"""``artifact_empty_cause`` 的全部取值 —— 区分「模型问题 / 没跑完」的唯一依据。

只有 ``no_files_written``（agent 自认为干完了却没交付）算模型问题 → ARTIFACT_MISSING
penalty；其余一律 ROLLOUT_UNFINISHED maskout。

``max_steps`` / ``response_truncated`` / ``format_error`` 依赖 agent 上报
``termination_reason``：claude_code 会报，hycode 不会（见
``task._infer_termination_for_installed_agent``，目前只能从 stderr 反推出 TIMEOUT）。
保留这些取值是因为换 agent 就会命中，删掉以后还得加回来。
"""

# ── 模型问题的惩罚分：负分参与训练，让模型学到「交了不合格的产物」比什么都不做更差 ──
# 三个都是 -0.2 且独立成常量，方便日后分别调整；本轮不动幅度，好让指标 delta 可归因。
PENALTY_UNDEPLOYABLE = -0.2
"""产物自身部署失败（依赖装不上 / 构建失败 / 首页探活失败）。
基建侧的部署失败走 EVAL_DEPLOY_SANDBOX_UNAVAILABLE，是 maskout，不罚模型。"""

PENALTY_MISSING = -0.2
"""agent 正常收尾（SUCCESS / ENV_DONE）却一个文件都没写 —— 它自认为干完了，所以是
模型自己的问题，而不是评测故障。其余 empty 原因（超时 / 步数上限 / 截断 / 收集失败 /
unknown）都归 ROLLOUT_UNFINISHED 走 maskout：那些是没跑完，不该由模型担责。"""


def build_judge_metrics(
    judge_code: int,
    term_reason: str = '',
    artifact_empty_cause: str = '',
) -> dict[str, float]:
    """Build 0/1 indicator dict for extra_reward_metrics. Mean over batch = ratio."""
    # ARTIFACT_MISSING 和 ROLLOUT_UNFINISHED 都可能是「0 文件」，但归因相反（模型 / 非模型），
    # 所以指标上必须分开 —— 旧的 judge/no_output 把两者混在一个 key 里，看不出谁的锅。
    # artifact/empty/{cause} 的细分对两者都保留，因为 cause 才是诊断信息。
    artifact_empty = judge_code in (Judge.ROLLOUT_UNFINISHED, Judge.ARTIFACT_MISSING)
    metrics = {
        'judge/success': 1.0 if judge_code == Judge.SUCCESS else 0.0,
        # 模型问题
        'judge/artifact_undeployable': 1.0 if judge_code == Judge.ARTIFACT_UNDEPLOYABLE else 0.0,
        'judge/artifact_missing': 1.0 if judge_code == Judge.ARTIFACT_MISSING else 0.0,
        # 非模型
        'judge/rollout_unfinished': 1.0 if judge_code == Judge.ROLLOUT_UNFINISHED else 0.0,
        'judge/eval_deploy_sandbox_unavailable': (1.0 if judge_code == Judge.EVAL_DEPLOY_SANDBOX_UNAVAILABLE else 0.0),
        'judge/eval_vlm_call_failed': 1.0 if judge_code == Judge.EVAL_VLM_CALL_FAILED else 0.0,
        'judge/eval_agent_call_failed': 1.0 if judge_code == Judge.EVAL_AGENT_CALL_FAILED else 0.0,
        'judge/eval_config_failed': 1.0 if judge_code == Judge.EVAL_CONFIG_FAILED else 0.0,
        'judge/nexus_task_failed': 1.0 if judge_code == Judge.NEXUS_TASK_FAILED else 0.0,
        'judge/nexus_task_timeout': 1.0 if judge_code == Judge.NEXUS_TASK_TIMEOUT else 0.0,
        'judge/eval_unknown_failed': 1.0 if judge_code == Judge.EVAL_UNKNOWN_FAILED else 0.0,
        # 两个聚合：一眼看到「模型问题 vs 评测问题」的总占比
        'judge/model_fault': 1.0 if judge_code in _PENALTY_CODES else 0.0,
        'judge/eval_fault': 1.0 if 2000 <= judge_code < 4000 else 0.0,
        # 按 term_reason 子串匹配，口径与上面按 code 的不同，保留是为了兼容既有看板。
        'judge/eval_timeout': 1.0 if 'TIMEOUT' in term_reason else 0.0,
    }
    metrics['artifact/empty'] = 1.0 if artifact_empty else 0.0
    for cause in ARTIFACT_EMPTY_CAUSES:
        metrics[f'artifact/empty/{cause}'] = 1.0 if artifact_empty and artifact_empty_cause == cause else 0.0
    return metrics


def status_value(value: Any) -> str:
    """Return a stable string for enum-like agent status values."""
    if value is None:
        return ''
    raw = getattr(value, 'value', value)
    return str(raw)


def archive_size_bytes(archive_b64: str) -> int:
    if not archive_b64:
        return 0
    try:
        return len(base64.b64decode(archive_b64, validate=True))
    except (binascii.Error, ValueError):
        return max(0, len(archive_b64) * 3 // 4)


def artifact_empty_cause(infer_metadata: dict[str, Any]) -> str:
    if infer_metadata.get('artifact_collect_error'):
        return 'collect_failed'

    status = ' '.join(
        str(infer_metadata.get(key, ''))
        for key in ('agent_termination_reason', 'agent_exit_status', 'agent_execution_status')
        if infer_metadata.get(key)
    ).upper()
    if 'TIMEOUT' in status:
        return 'agent_timeout'
    if 'TRUNCATION' in status or 'CONTEXT_LENGTH' in status or 'LENGTH_EXCEEDED' in status:
        return 'response_truncated'
    if 'MAX_STEPS' in status or 'MAX_ITERATIONS' in status:
        return 'max_steps'
    if 'FORMAT_ERROR' in status or 'TOOL_ERROR' in status:
        return 'format_error'
    if any(token in status for token in ('SUCCESS', 'FINISHED', 'SUBMITTED', 'ENV_DONE')):
        return 'no_files_written'
    return 'unknown'


def artifact_incomplete_cause(infer_metadata: dict[str, Any]) -> str:
    cause = artifact_empty_cause(infer_metadata)
    if cause in {'agent_timeout', 'max_steps', 'response_truncated', 'format_error'}:
        return cause
    return ''


def _normalized_artifact_paths(file_list: list[str]) -> list[str]:
    paths: list[str] = []
    for path in file_list:
        clean = path.strip()
        while clean.startswith('./'):
            clean = clean[2:]
        clean = clean.strip('/')
        if clean:
            paths.append(clean)
    split_paths = [path.split('/', 1) for path in paths if '/' in path]
    if split_paths and len(split_paths) == len(paths):
        first_dir = split_paths[0][0]
        if all(parts[0] == first_dir for parts in split_paths):
            return [parts[1] for parts in split_paths if parts[1]]
    return paths


def primary_error(judge_response: dict[str, Any]) -> dict[str, Any]:
    primary = judge_response.get('primary_error')
    return primary if isinstance(primary, dict) else {}


def primary_error_code(judge_response: dict[str, Any]) -> str:
    primary = primary_error(judge_response)
    code = primary.get('code') or judge_response.get('error_code') or ''
    return str(code) if code else ''


def compact_evaluator_errors(judge_response: dict[str, Any], limit: int = 16) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    errors = judge_response.get('errors')
    if not isinstance(errors, list):
        return compact
    for error in errors:
        if not isinstance(error, dict):
            continue
        item = {
            key: error[key]
            for key in (
                'code',
                'legacy_code',
                'phase',
                'origin',
                'evaluator_id',
                'config_index',
                'type',
                'model_alias',
                'model_id',
                'dimensions',
                'dimension',
                'rubric_index',
                'retryable',
            )
            if key in error
        }
        message = str(error.get('message') or '')
        if message:
            item['message'] = message[:500]
        compact.append(item)
        if len(compact) >= limit:
            break
    return compact


def build_reward_extra_info(
    *,
    request_id: str,
    judge_code: int,
    judge_response: dict[str, Any],
    term_reason: str = '',
    artifact_empty_cause: str = '',
    artifact_incomplete_cause: str = '',
    infer_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    infer_metadata = infer_metadata or {}
    _artifact = judge_response.get('artifact')
    artifact: dict[str, Any] = _artifact if isinstance(_artifact, dict) else {}
    primary = primary_error(judge_response)
    extra: dict[str, Any] = {
        'request_id': request_id,
        'judge_code': judge_code,
        'error_code': primary_error_code(judge_response),
        'primary_error_code': primary_error_code(judge_response),
        'primary_error_legacy_code': primary.get('legacy_code', judge_code) if primary else judge_code,
        'training_action': judge_response.get('training_action') or Judge(judge_code).action,
        'outcome': judge_response.get('outcome') or Judge(judge_code).outcome,
        'weighted_scores': judge_response.get('weighted_scores', {}),
        'extra_reward_metrics': build_judge_metrics(
            judge_code,
            term_reason,
            artifact_empty_cause,
        ),
    }
    if term_reason:
        extra['termination_reason'] = term_reason
    if primary:
        extra['primary_error'] = primary
    evaluator_errors = compact_evaluator_errors(judge_response)
    if evaluator_errors:
        extra['evaluator_errors'] = evaluator_errors
    if 'deploy' in judge_response:
        extra['deploy'] = judge_response['deploy']
    if artifact or infer_metadata:
        extra['artifact'] = {
            **artifact,
            'file_count': artifact.get('file_count', infer_metadata.get('artifact_file_count', 0)),
            'archive_bytes': artifact.get('archive_bytes', infer_metadata.get('artifact_archive_bytes', 0)),
        }
    if artifact_empty_cause:
        extra['artifact_empty_cause'] = artifact_empty_cause
    if artifact_incomplete_cause:
        # 纯诊断：1005 并入 ROLLOUT_UNFINISHED 后没有 artifact/incomplete/* 指标了，
        # 但读 trajectory dump 时这个 cause 仍能说明 rollout 为什么被中断。
        extra['artifact_incomplete_cause'] = artifact_incomplete_cause
    # agent_execution_status carries 'exit_code=N' for installed agents -- forwarding it
    # is what makes an agent crash distinguishable from a clean no-op in the RL dump.
    for key in (
        'artifact_file_count',
        'artifact_archive_bytes',
        'agent_termination_reason',
        'agent_exit_status',
        'agent_execution_status',
    ):
        if key in infer_metadata:
            extra[key] = infer_metadata[key]
    return extra
