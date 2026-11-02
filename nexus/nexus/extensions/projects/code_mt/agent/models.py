"""Data models for code_mt_agent."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from nexus.core import AgentRunInput, AgentRunOutput, WithMainTrajectory, WithUserPrompt, WithWorkingDir


class CodeMTExitStatus(str, Enum):
    """Canonical exit statuses + their HyTuner termination_reason mapping.

    Aligned with nexus main_agent (extensions/agents/main/agent.py MainExitStatus):
    a single enum owns both the status string and its termination_reason, instead
    of a scattered dict, so every loop and the task map identically. FINISHED /
    SUBMITTED are the two normal completions (submit tool vs bare-finish).
    """

    SUBMITTED = 'submitted'
    FINISHED = 'finished'
    MAX_ITERATIONS = 'max_iterations'
    TOOL_ERROR = 'tool_error'
    INFERENCE_ERROR = 'inference_error'
    TIMEOUT = 'timeout'
    TRUNCATION = 'truncation'
    ERROR = 'error'

    @property
    def termination_reason(self) -> str:
        return {
            CodeMTExitStatus.SUBMITTED: 'ENV_DONE',
            CodeMTExitStatus.FINISHED: 'ENV_DONE',
            CodeMTExitStatus.MAX_ITERATIONS: 'MAX_STEPS',
            CodeMTExitStatus.TOOL_ERROR: 'FORMAT_ERROR',
            CodeMTExitStatus.INFERENCE_ERROR: 'UNHANDLED_ERROR',
            CodeMTExitStatus.TIMEOUT: 'TASK_TIMEOUT',
            CodeMTExitStatus.TRUNCATION: 'TRUNCATION',
            CodeMTExitStatus.ERROR: 'UNHANDLED_ERROR',
        }[self]

    @classmethod
    def termination_reason_for(cls, value: str) -> str:
        try:
            return cls(value).termination_reason
        except ValueError:
            return 'UNKNOWN'


class CodeMTAgentInput(AgentRunInput, WithUserPrompt, WithWorkingDir):
    """Unified agent input supporting both tool_call and text_parse modes."""

    # Mode
    call_mode: Literal['tool_call', 'text_parse'] = 'tool_call'
    max_iterations: int = 100
    # Turn-0 system + user prompt are passed VERBATIM by the task (run_agent kwargs).
    # code_mt_agent does NO prompt assembly — no template render, no file read, no
    # injection. The task owns all prompt composition (aligns main_agent).
    system_prompt: str = ''

    # === Turn-0 env setup (import_path string) ===
    # Dataset-declared `module::func` strategy run ONCE on the first entry to the
    # agent loop (never on MT retry turns). Owns env prep — image switch,
    # dependency pre-install, /tmp patch helpers, repo exploration — and returns
    # inject_vars merged into ctx.task_config for the agent's prompt render. This
    # is what keeps code_mt_task agent-agnostic (setup moved task -> agent).
    setup_strategy: str = ''

    # === tool_call mode ===
    tools: list[str] = ['code_mt/bash', 'code_mt/str_replace_editor', 'code_mt/submit']
    mcp_servers: list[dict[str, Any]] | None = None
    # When set, forwarded to chat.completions.create as `parallel_tool_calls`.
    # Default None = don't transmit (let the model/proxy decide). Datasets that
    # want to forbid parallel tool_calls set this to False explicitly.
    parallel_tool_calls: bool | None = None
    # Deprecated legacy alias. Generators historically set `single_tool_call: True`
    # but the loop did not enforce it. If parallel_tool_calls is unset and this is
    # True, we coerce parallel_tool_calls=False at the start of the loop.
    single_tool_call: bool = False
    completion_retry_on_tool_error: int = 0
    # "No tool call" completion semantics (aligns with main_agent, which treats a
    # bare assistant message as FINISHED / ENV_DONE). Default False preserves the
    # historical swe_agent behavior: no tool call -> tool_error -> FORMAT_ERROR,
    # so existing FORMAT_ERROR statistics are unchanged unless a dataset opts in.
    # When True: a turn with content but no tool_calls ends the loop as
    # 'finished' (-> ENV_DONE) instead of a format error. Genuine format errors
    # (unparseable/invalid tool-call arguments) ALWAYS remain tool_error
    # regardless of this flag — this only reclassifies the "model chose to answer
    # in prose instead of calling a tool" case. Datasets whose contract is "always
    # finish via a submit-family tool" leave this False.
    no_tool_call_finish: bool = False
    # Mirror swe_agent.SWEAgentInput.early_stop (default False). When True, a tool
    # execution error (generic Exception branch) terminates the loop immediately
    # with exit_status='tool_error' (→ FORMAT_ERROR) instead of appending the error
    # observation and continuing. RuntimeUnhealthyError / TimeoutError always
    # terminate regardless of this flag (see tool_call_loop observation section).
    early_stop: bool = False
    observation_template: str = 'OBSERVATION:\n{output}'
    no_output_message: str = 'Your last command ran successfully and did not produce any output.'
    max_observation_length: int = 50000
    two_stage_submit: bool = False
    tool_schema_overrides: list[dict[str, Any]] | str | None = None
    tool_description_language: Literal['default', 'zh-CN'] = 'default'

    # === text_parse mode ===
    parser_name: Literal['json', 'xml'] = 'json'
    tmux_pane_width: int = 160
    tmux_pane_height: int = 40
    enable_summarize: bool = False
    proactive_summarization_threshold: int = 1_000_000
    model_name: str = ''
    context_limit: int | None = None
    two_stage_completion: bool = True
    max_terminal_output: int = 10000

    # === Common timeouts ===
    observation_time_timeout: float | None = None
    per_observation_timeout: float | None = None
    inference_time_timeout: float | None = None
    # Bash single-command hard kill (seconds). tool_call mode only — text_parse
    # drives tmux directly and never calls code_mt/bash. Dataset/operator decides
    # (e.g. builder=900); model cannot self-select. Default 300 = backward compat.
    bash_timeout: int = 300

    # === Hooks ===
    post_iteration_hook: str | None = None


class CodeMTAgentOutput(AgentRunOutput, WithMainTrajectory):
    """Unified agent output.

    Carries WithMainTrajectory (like main_agent's MainAgentOutput) so any task —
    including harbor_bridge_task, which reads agent_output.main_trajectory to
    persist the RepoQA deliverable — can extract the answer/transcript. Without
    it, harbor_bridge's deliverable persistence degrades to empty for code_mt.
    """

    exit_status: str = ''
    iterations: int = 0
