"""code_mt project: harbor_bridge + MT loop + host-side judge.

Self-contained project (agent + task deeply coupled). Agent loops live in
``.agent`` (tool_call / text_parse), the harbor-bridge-backed task + scoring in
``.task``. Registry names (code_mt / code_mt_task / code_mt/*) are unchanged from
the pre-move layout; only import paths moved under projects.code_mt.{agent,task}.
"""
