# macOSWorld paper-like GPT CUA configuration

Use `docs/examples/configs/mac_os_world/mac_os_world_gptcua_paper_devcloud.yaml` with the same
published macOSWorld task dataset. This is an opt-in protocol for GPT-5.4, not a
reproduction of the paper's `computer-use-preview-2025-03-11` model.

The profile changes the following settings:

- `max_iterations: 30` and `max_screenshots: 15`: stop when either budget is
  exhausted. Count each new screenshot delivered to the model, including the
  initial observation; resent context and local diagnostic/final captures do not
  count again. Pending outputs from a group of tool calls are not partially sent
  when they would exceed the remaining budget. Budget exhaustion retains the
  `max_iterations` exit status so the task still runs the official evaluator.
- `max_recent_screenshots: 3`: resend explicit native Responses history, replacing
  older images with black placeholders. Preserve all text, native computer calls,
  call IDs, outputs, and encrypted reasoning. Request encrypted reasoning and do
  not use `previous_response_id`; the client must also have
  `stateful_previous_response=false` (its default). Missing native output or
  encrypted reasoning fails explicitly rather than silently dropping context.
- `prompt_style: macosworld_paper`: use the Appendix G.4 CUA system prompt with the
  actual runtime account and configured password. Honor the paper's fenced
  `DONE`/`FAIL` signals on action-free turns.
- `task_config.include_task_context: false`: send the original task instruction
  without Nexus-specific folder-selection or read-only-verification guidance.

The default GPT CUA behavior remains unchanged when these options are unset.

## Differences retained

The DevCloud provider currently has no display-resolution or OS-language option
in its sandbox-create contract. Keep the actual 1920×1080 desktop and automatic
locale detection for now. Merely setting `screen_width`, `screen_height`, or
`environment_language=en` does not change the desktop or switch the OS language.
Changing OS language may require relogin and revalidation of localized templates
and GUI-service permissions. Image resizing alone would not reproduce the paper's
1024×768 UI layout and would also require coordinate mapping.

Keep GPT-5.4 `reasoning_effort=high`; there is no directly equivalent setting for
that paper model. The shared Responses client removes sampling controls when a
reasoning configuration is present, so setting `top_p=0.9` alone would not provide
paper-equivalent sampling. Native GPT-5.4 computer-call action batches and the
existing clipboard input implementation also remain in place.

The dataset, initialization commands and binary official evaluator remain the
same. This profile does not add an initial-state grading pass or change scores.

## Validation and platform rollout

Unit coverage exercises image-window retention, native reasoning preservation,
call/output pairing, observation-budget boundaries, paper completion markers,
and the opt-out of added task guidance. The 263 relevant unit tests and repository
format, lint and type checks pass. After an initial HTTP 401 on 2026-10-22, a
subsequent authentication check and a fresh local single-case run succeeded.
The Dictionary About case completed in 10 rounds with 10 new screenshots,
including native-history replay beyond the three-image window. The disposable
Mac was released after the run. This verifies the integration on one case;
full-suite reliability still needs evaluation.

For the corresponding full evaluation, set platform concurrency to 12 using the
task's `crawl_para_num`, not an agent setting or per-case thread count.

References: [paper setup](https://arxiv.org/html/2506.04135v4#S5.SS1),
[CUA protocol](https://arxiv.org/html/2506.04135v4#A7.SS4).
