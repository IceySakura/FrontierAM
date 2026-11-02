# search_s2t_rl_yb_glue

S2T (speech-to-text) Search Agentic RL that **reuses the `search_s2t_rl` agent +
search tool** but **replaces the reward with the `examples/yuanbao_search`
(`agts`) design**, and supports **mixed speech / text** training.

It glues:

| Piece | From | What is kept |
| --- | --- | --- |
| Agent loop + search tool | `search_s2t_rl` | multi-turn OpenAI function-calling agent; a single vendored PromptSearch tool (canonical `search_s2t_yb_glue/search`, exposed to the model as `web_search`) |
| Reward design + scoring | `agts` gates + Reward_system | required-search pipeline: format check → valid-search gate → AudioGRMDS 三档评分 → query cost |
| Speech input support | `examples/audio_baseline` | `RLHFDatasetAudio` + `<audio>` patches embedded as `{"type":"audio",...}` content parts |

> Only the single PromptSearch backend is used (surfaced as `web_search`); the
> yuanbao `picture_search` / `video_search` / `finance_search` backends are **not**.

## Reward (required-search variant of yuanbao `agts_closed_qa`)

All rows in this project are treated as requiring search. The reward pipeline is:

```text
rule_score = rule_based_judge(final_answer, extra_info)
if rule_score.planner_rule_check == -1:      # format failure
    score = -1.0
elif valid_search_count == 0:                # no search with a non-empty query
    score = 0.0                              # skip the GRM judge
else:
    closed_qa_score = llm_judge(...)         # agts 三档 judge，经 Reward_system → {0, .5, 1}
    query_cost = min(0.05, 0.01 * max(0, subquery_count - search_count))
    score = max(0.0, min(1.0, closed_qa_score - query_cost))
```

- **`reward/llm_judge.py`** — preserves the old `closed_qa_score` output
  contract **and the agts three-tier judge prompt verbatim**（``完全匹配 / 部分匹配 /
  不匹配`` → ``1.0 / 0.5 / 0.0``）; only the *transport* is swapped. Instead of a
  Taiji/alex-oss OpenAI judge, it sends the agts `PROMPT_TEMPLATE` through
  Reward_system via `reward/grm_transport.py::request_grm_raw_text` (an
  independent, hytuner-free port — see "Reward dependencies" below), using a
  json_schema `override` that constrains the GRM to emit the flat
  `{judgment, confidence, reason}` structure the agts parser expects, then parses
  and validates the raw completion client-side. Non-object JSON or invalid
  judgment/confidence/reason fields enter the bounded parse-retry loop instead
  of crashing with a type error. It scores the model answer against
  `reward_model.ground_truth` given the question **transcription**
  (`extra_info.question`) + `history`, so audio input turns don't change the
  reward path.
- **`reward/rule_based_judge.py`** — same field names / gating semantics as
  `agts` (`planner_rule_check == -1` ⇒ format failure). Adapted to the
  **structured** function-calling trajectory: instead of parsing DeepSeek/Qwen
  inline tool-call text, it fails `planner` when the final answer leaks tool-call
  markers, when the tool-call count exceeds the hard cap, or when any tool call
  had unparseable JSON arguments (`malformed_tool_calls > 0`). Media / picture /
  video checks are ported unchanged (inert for this search-only speech data). The
  `<result>`/`<answer>` wrapping requirement was **dropped** (sp.md outputs pure
  spoken text); the tags are still honored for extraction if present.
- **`reward/total_reward.py`** — retains the planner gate, three-tier semantic
  judge, and lightweight query cost, but requires at least one search containing
  a non-empty query. A trajectory without a valid search receives `0.0` without calling the GRM; searched
  trajectories are no longer multiplied by a tool-call discount. The search
  decision comes from the structured trajectory's `extra_info['valid_search_count']`;
  whitespace-only queries return a tool observation but do not unlock semantic reward.

## Agent (`search_s2t_agent_yb_glue`)

Same loop as `search_s2t_rl`'s agent (policy = `main` client), with a hard
project protocol of at most 2 model iterations, 1 search, and 3 subqueries per
search. Legacy rows carrying the previous 10/5 limits are clamped at runtime.

- Builds its starting conversation with `conversation.build_agent_messages`,
  which **preserves multi-modal content lists** and expands each `<audio>`
  placeholder into a `{"type": "audio", <patch>}` content part (in order across
  the whole multi-turn conversation). The patch dict is JSON-safe and is loaded
  lazily by `process_multi_modal_info` (`load_audio_from_patch`) with the
  processor's sampling rate — the exact path `RLHFDatasetAudio` uses — so both
  the vLLM rollout and the training-side token expansion see the audio.
- Exposes only the PromptSearch tool (model-visible name `web_search`, a single
  `query: array<string>` param per sp.md) and surfaces `tool_call_count` /
  `search_count` / `valid_search_count` / `final_answer` for the reward. A
  search becomes valid only when the returned ToolResponse reports at least one
  cleaned, non-empty subquery; backend failures and zero-result searches remain
  valid when the submitted query itself was non-empty.
- Restores `minItems=1` / `maxItems=3` on the model-visible query schema,
  enforces the same limit before execution, and records attempted/executed/
  over-limit subquery counts.
- Uses the HyTuner client's exact tokenizer/chat template token counter to cap
  the sum of all tool observations at 10,000 tokens. Oversized observations are
  truncated before entering history. The model-visible tool schema remains
  stable for the whole trajectory to preserve HyTuner's incremental-token cache;
  once a search/observation hard limit is reached, Agent-side execution gating
  rejects subsequent tool calls without executing them. Rejected calls are
  recorded in `rejected_tool_calls` / `overlimit_search_calls` and fail the
  planner reward gate, so a final answer cannot hide a repeated search request.
- The `role=tool` payload fed back to the policy is rendered by
  `tools.py::_render_web_search_result`, a port of `agent_loop.py`: each kept doc
  is a `[web_doc N begin] … [web_doc N end]` block with search query / title /
  site / **authority level** / **publish time** / content, under a global
  10000-char content budget (first-doc-overflow ⇒ truncate to 10000 + `......`);
  an empty or 3×-failed search feeds the fixed text `未查询到对应搜索结果`
  (the upstream error is never surfaced to the model).

Text rows (`audios=None`, no `<audio>`) build a plain-text conversation — one
code path handles mixed speech/text batches.

**Why the schema stays model-visible after the budget is exhausted:** HyTuner
incrementally tokenizes a trajectory and caches the rendered prefix after every
completion. Tool schemas participate in that render, so switching `tools` from a
schema list to `None` mid-trajectory rewrites the cached prefix and invalidates
its token alignment. The schema is therefore picked once and kept stable for the
whole trajectory, while the per-iteration hard limits independently gate tool
execution. Regression coverage lives in
`tests/unit/extensions/projects/search_s2t_rl_yb_glue/test_search_s2t_yb_glue_agent.py`
(`TestBudgetViolation` and `TestSpanReporting`).

## Task (`search_s2t_task_yb_glue`)

Runs the agent, extracts the final answer + question + history + valid-search count
from the trajectory, and calls the GRM-backed reward. Returns `WithScore.score`,
which HyTuner places on the last response token as the GRPO reward. Requires
`reward_model.enable=False` and
`actor_rollout_ref.rollout.agent.handle_nexus_none_reward_score=true`.

For Agentic OPD training, set `task_config.reward_mode` to `"zero"`. The task
still runs the agent and returns the complete trajectory/tool statistics, but
does not load or call any reward implementation and returns `score=0.0` with
`reward_skipped=true`. The default is `"compute"`. HyTuner marks validation
runs through `task_config.is_validation`; validation always computes the real
reward even when the requested mode is `"zero"`.

## Data contract

Per-row `nexus.task_config`:

```jsonc
{
  "data_source": "agts_closed_qa",
  "ground_truth": "<reward_model.ground_truth>",
  "reward_style": "audio-grm",
  "reward_system_user_token": "${oc.env:NEXUS_REWARD_SYSTEM_TOKEN,}",
  "reward_system_exp_id": "${oc.env:NEXUS_REWARD_SYSTEM_EXP_ID,hy3.0_a20b_main_rl_v12}",
  "reward_system_router_id": "${oc.env:NEXUS_REWARD_SYSTEM_ROUTER_ID,AudioGRMDS-SimpleFusion-0}",
  "reward_system_timeout": 1800,
  "reward_system_max_retries": 3,
  "reward_mode": "compute", // use "zero" for Agentic OPD training
  "messages": [ {"role": "system", ...}, {"role": "user", "content": "<audio>"}, ... ],
  "audios":   [ {"patch": "/abs/x.wav", "audio_length": ..., "sample_rate": ...}, ... ],
  "extra_info": { "question": "...", "history": [...], "图片等级": "none", "视频等级": "none", "是否调用文本搜": "", ... },
  "max_iterations": 2,
  "max_search_calls": 1,
  "max_subqueries_per_call": 3,
  "max_total_observation_tokens": 10000
}
```

- `messages` — multi-turn conversation ending at the target user turn; audio
  turns carry `<audio>`.
- `audios` — one patch per `<audio>` placeholder, in order (`null` for text rows).
- Reward_system fields are resolved from env vars in the generated task config.

Build such rows from `dataset/s2t_searchrl_0707_full` with
`examples/nexus_search_s2t_yb_glue/build_s2t_yb_glue_dataset.py` (that script
lives in the **hytuner** repo, not here).

## Reward dependencies (hytuner-free)

`reward/llm_judge.py` and `reward/gate_judge.py` reach the GRM through
`reward/grm_transport.py`, `reward/gates.py`, and `reward/grm_rubrics.py` — all
three are independent ports of the corresponding hytuner /
`recipe.audio.speech.reward` modules (see
`docs/developing/search-s2t-rl-yb-glue-reward-migration.md` for what was ported
and why). This extension no longer imports `hytuner` or `recipe` anywhere, at
module scope or lazily; `polaris-cpp-py` (declared under the `host` extra,
mirroring `nexus/extensions/tools/yb_search`) is the only extra runtime
dependency, needed for the Reward_system Polaris service discovery that
`grm_transport.py` shares with `nexus/utils/reward_system.py`.

## Tests

```
uv run pytest tests/unit/extensions/projects/search_s2t_rl_yb_glue -v
```

## Registration

`imports` in the per-row nexus config should include:

```
nexus.extensions.projects.search_s2t_rl_yb_glue.task
nexus.extensions.projects.search_s2t_rl_yb_glue.agent
```

(`agent` side-imports `tools`, so the search tool is registered too.)
