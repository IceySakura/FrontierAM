"""SWE-Cosmos bugmaker pipeline task package.

Runs the SWE-Cosmos bugmaker data-generation pipeline (bugmaker → collect → validate → gather → evaluate → issuegen) inside a single "root" sandbox built from the SWE-Cosmos image `mirrors.tencent.com/hunyuan_yanguan/petrichorqi.swe-cosmos:latest`. The pipeline itself spawns serveral child Gongfeng sandboxes for per-instance repo-clone/bug-validation/task-evaluation, while LLM calls are forwarded back to the host via an in-sandbox OpenAI-compatible proxy.

See ``task.py`` for the host-side orchestration and ``runtime_setup.py`` for the sandbox-side target executed via ``runtime.invoke``.
"""
