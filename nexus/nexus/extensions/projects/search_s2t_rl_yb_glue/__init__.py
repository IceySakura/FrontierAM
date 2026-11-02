"""S2T Search Agentic RL project with yuanbao reward gates and AudioGRMDS.

This project glues together two existing HyTuner components:

1. The **agent + search tool** from ``search_s2t_rl`` — a multi-turn Nexus agent
   that talks to the trainable policy through OpenAI-style function calling and
   uses a single vendored PromptSearch tool (canonical key
   ``search_s2t_yb_glue/search``, exposed to the model as ``web_search`` per the
   sp.md contract). No other backends (``picture_search`` / ``video_search`` /
   ``finance_search`` from the ``agts`` project) are used.

2. The **reward gates** from ``examples/yuanbao_search`` (the ``agts`` project,
   ``data_source='agts_closed_qa'``), with the semantic score produced by
   Reward_system AudioGRMDS:

       format_check → (fail ⇒ -1.0)
       no valid search → 0.0
       valid search    → max(0.0, closed_qa_score - small query cost)

   where ``closed_qa_score`` uses the three tiers ``0.0 / 0.5 / 1.0``.

The task supports **mixed speech / text** training: audio user turns carry a
``<audio>`` placeholder plus a patch reference that the agent embeds as an
``{"type": "audio", ...}`` content part, while text turns keep plain text. The
GRM judge is text-based (it scores the model answer against the
reference using the question transcription + history).
"""
