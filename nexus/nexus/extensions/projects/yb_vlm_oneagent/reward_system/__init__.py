"""Reward functions specific to the ``yb_vlm_oneagent`` task.

Currently exports:

* ``yb_rubric_reward`` — async wrapper around the rubric-based
  scoring framework (rubric checklist + LLM judge + multi-round mean).
  Registered under ``reward_func='yb_rubric'``. The actual rubric
  prompts / weighting live in topic modules such as ``shiwu_topic`` and
  ``medical_topic``; the wrapper dispatches by ``rubric_topic``.
* ``tool_use_label_reward`` — async reward that compares the trajectory's
  actual tool calls (``web_search`` / ``vision_search``) against the
  ground-truth ``txt2txt_search`` / ``img2img_search`` labels carried in
  ``target_answer``. Registered under ``reward_func='tool_use_label'``.
"""

from nexus.extensions.projects.yb_vlm_oneagent.reward_system.tool_use_label import (
    tool_use_label_reward,
)
from nexus.extensions.projects.yb_vlm_oneagent.reward_system.yb_rubric import (
    yb_rubric_reward,
)


__all__ = ['tool_use_label_reward', 'yb_rubric_reward']
