"""LocalOneAgentDirectClient — 复用 one_agent_infer 直连实现的本地模式 client。

``local_one_agent_infer`` 与 ``one_agent_infer`` 的直连逻辑几乎完全一致，故此处不再
复制一份实现，而是直接继承 ``one_agent_infer.direct_llm`` 的 ``OneAgentDirectClient``，
仅换一个注册名 ``local_one_agent_direct``（在 config 的 ``openai_clients.main.name``
引用），避免两个包在同进程内争抢同一注册名。
"""

from __future__ import annotations

from nexus.core import openai_client_registry
from nexus.extensions.tasks.one_agent_infer.direct_llm.client import OneAgentDirectClient


@openai_client_registry.register('local_one_agent_direct')
class LocalOneAgentDirectClient(OneAgentDirectClient):
    """本地模式直连 client，行为与 ``OneAgentDirectClient`` 完全一致。"""
