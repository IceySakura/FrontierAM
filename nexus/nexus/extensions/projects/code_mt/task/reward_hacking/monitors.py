"""Monitor entry point for the code_mt reward-hacking / baseline plug-in.

Reuses the scaffold-agnostic SWE detectors (they take OpenAI-chat messages,
which is exactly what the code_mt loops stash under
ctx.tool_state['_code_mt_messages']), plus the code_mt baseline monitors.

Every detector FLAGS ONLY — none of them mutate the score here. The result dict
is merged into task_output.extra_fields and consumed by penalties.py.

All calls are individually fail-open: a monitor that raises leaves its key out
of the result rather than aborting the whole run.
"""

from __future__ import annotations

import logging

from typing import Any

from nexus.extensions.projects.code_mt.task.reward_hacking._vendored_swe import detect_reward_hacking
from nexus.extensions.projects.code_mt.task.reward_hacking._vendored_swe.monitors import (
    detect_codeswitch,
    detect_output_format_bad_patterns,
)
from nexus.utils.reasoning_repetition import detect_let_me_loop

from .baseline import run_baseline_monitors
from .config import RewardHackingConfig
from .str_replace_channels import detect_str_replace_noop_by_channel
from .text_parse_bridge import normalize_text_parse_messages


logger = logging.getLogger(__name__)


def run_code_mt_monitors(
    messages: list[dict] | None,
    *,
    config: RewardHackingConfig,
    eval_result: dict[str, Any] | None = None,
    patch_text: str = '',
    instance_id: str = '',
    repo: str = '',
    score: float | None = None,
    answer: str = '',
    task_type: str = '',
    base_commit: str = '',
    history_truncated: bool | None = None,
) -> dict[str, Any]:
    """Run output monitors + anti-hack detection + baseline monitors.

    Returns a dict of detection metadata to merge into extra_fields. Never
    raises; each block is guarded so a failure degrades to fewer keys.

    Each monitor family runs only when it is selected by the parquet whitelist
    (``config.monitor_on(name)``; empty whitelist = ALL). A family not selected
    is neither flagged nor penalized. The output-quality families additionally
    require the global ``enable_output_monitoring`` switch; the anti-hack family
    the ``enable_reward_hacking_detection`` / ``enable_anti_hack`` switches.
    """
    extra: dict[str, Any] = {}
    messages = messages or []

    # Structured-only monitors (anti_hack via parse_nexus_messages, and baseline
    # trailing_tool_failure/submit_loop) read only native tool_calls / role='tool'
    # messages. A text_parse trajectory has neither — commands live in the
    # assistant NL as <keystrokes>/JSON and observations are role='user'. Rebuild
    # a tool_call-shaped view so those monitors are not silently inert on the
    # terminal/tbtrain rows. Returns None for a genuine tool_call trajectory (or
    # when no command is recoverable), in which case we keep the raw messages.
    struct_messages = normalize_text_parse_messages(messages) or messages

    # --- codeswitch (Chinese+English mix in assistant NL) ---
    if config.enable_output_monitoring and config.monitor_on('codeswitch'):
        try:
            extra['codeswitch'] = detect_codeswitch(messages)
        except Exception:
            logger.exception('detect_codeswitch failed')

    # --- output_format_bad_pattern (F6_7 think-tag / F7_2 + F7_3 leaked tool_calls) ---
    if config.enable_output_monitoring and config.monitor_on('output_format_bad_pattern'):
        try:
            extra['output_format_bad_pattern'] = detect_output_format_bad_patterns(messages)
        except Exception:
            logger.exception('detect_output_format_bad_patterns failed')

    # --- str_replace_noop (old==new no-op edits), split by tool_call / function_call ---
    # Emitted under the same key the penalty code consumes ('str_replace_old_eq_new'),
    # so the penalty math is unchanged; the tool_call/function_call channel breakdown
    # is carried alongside for the log-only report.
    if config.enable_output_monitoring and config.monitor_on('str_replace_noop'):
        try:
            extra['str_replace_old_eq_new'] = detect_str_replace_noop_by_channel(messages)
        except Exception:
            logger.exception('detect_str_replace_noop_by_channel failed')

    # --- let_me_loop (dense "let me" restatement in reasoning/content) ---
    if config.enable_output_monitoring and config.monitor_on('let_me_loop'):
        try:
            extra['let_me_loop'] = detect_let_me_loop(messages)
        except Exception:
            logger.exception('detect_let_me_loop failed')

    # --- anti_hack (answer lookup: network channels always, local/recall-first channels
    # for SWE only — see _vendored_swe/rules.py::_is_swe_style). Flags only here; the
    # CODE_REWARD_HACKING termination is applied later, gated by enable_anti_hack. ---
    if config.monitor_on('anti_hack') and (config.enable_reward_hacking_detection or config.enable_anti_hack):
        try:
            extra.update(
                detect_reward_hacking(
                    struct_messages,
                    instance_id or None,
                    repo or None,
                    patch_text or None,
                    float(score) if score is not None else 0.0,
                    task_type=task_type or None,
                    base_commit=base_commit or None,
                    history_truncated=history_truncated,
                )
            )
        except Exception:
            logger.exception('detect_reward_hacking failed')

    # --- baseline (flag-only) monitors ---
    # empty_final_answer / noop_edit read assistant TEXT (raw messages); the
    # trailing_tool_failure / submit_loop monitors read tool_calls / role='tool'
    # (struct_messages). run_baseline_monitors takes both.
    if config.monitor_on('baseline'):
        try:
            extra.update(run_baseline_monitors(messages, answer, struct_messages=struct_messages))
        except Exception:
            logger.exception('run_baseline_monitors failed')

    return extra


__all__ = ['run_code_mt_monitors']
