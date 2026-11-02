"""direct_llm — local_one_agent_infer 的直连 CSIG / Taiji client 包。

本地模式复用 ``one_agent_infer.direct_llm`` 的全部实现（路由表、CSIG/Taiji 转发、
prompt cache 一致性哈希路由等），此处仅以一个薄子类继承
``OneAgentDirectClient``，并以独立注册名 ``local_one_agent_direct`` 登记，避免与
``one_agent_infer`` 的 ``one_agent_direct`` 在同进程内冲突。

导入本包即注册 ``local_one_agent_direct`` client（见 client.py）。
"""

from __future__ import annotations

from .client import LocalOneAgentDirectClient


__all__ = ['LocalOneAgentDirectClient']
