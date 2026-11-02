"""direct_llm — one_agent_infer 直连 CSIG / Taiji 的 OpenAI 兼容 client 包。

导入本包即注册 ``one_agent_direct`` client（见 client.py）。
"""

from __future__ import annotations

from .client import OneAgentDirectClient


__all__ = ['OneAgentDirectClient']
