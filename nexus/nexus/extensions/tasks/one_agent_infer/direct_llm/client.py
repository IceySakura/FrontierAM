"""OneAgentDirectClient — 直连 CSIG / Taiji 的 OpenAI 兼容 client。

替代 litellm + 独立 router server 的方案：把原 litellm_router 中的路由
（router.resolve）与各平台转发逻辑（backend_csig / backend_taiji）内联到
nexus 进程内，agent loop 调用 ``create_completion`` 时直接 httpx 打到平台
真实 URL，无需再起 router server、无需再维护 litellm_router 仓库。

注册名：``one_agent_direct``（在 config 的 ``openai_clients.main.name`` 引用）。

仅支持 run-batch 的 **非流式**（``create_completion``）路径；web demo 的流式
路径（``client.client.chat.completions.create``）不在本 client 范围内。
"""

from __future__ import annotations

import json
import logging
import os

from pathlib import Path
from typing import Any

import httpx

from nexus.core import OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.context import require_context
from nexus.core.types import ExtendedChatCompletion

from . import backend_csig, backend_taiji, config
from .router import resolve


logger = logging.getLogger(__name__)


@openai_client_registry.register('one_agent_direct')
class OneAgentDirectClient(OpenAICompatibleClient):
    """直连 CSIG / Taiji 平台的 OpenAI 兼容 client。"""

    def __init__(self, conf: OpenAIConfig) -> None:
        super().__init__(conf)
        # 复用原 proxy_server 的关键 httpx 设置：
        #   proxy=None / trust_env=False —— 禁用代理，避免内网请求被 star-proxy 拦截
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=config.HTTP_MAX_CONNECTIONS,
                max_keepalive_connections=config.HTTP_MAX_KEEPALIVE,
            ),
            timeout=httpx.Timeout(360.0, connect=config.HTTP_CONNECT_TIMEOUT),
            proxy=None,
            trust_env=False,
        )

    # ── prompt cache 一致性哈希路由 ──────────────────────────────
    def _resolve_session_id(self, configured: str) -> str:
        """SessionID：显式配置优先，否则用当前 run_id/trace_id 兜底（按会话稳定）。"""
        session_id = configured
        if not session_id:
            try:
                ctx = require_context()
                session_id = ctx.run_id or ctx.trace_id or ''
            except Exception:
                session_id = ''
        return session_id

    def _taiji_routing(self) -> tuple[str, str]:
        """计算 Taiji 的 StrategyType / SessionID。

        SessionID 留空时用当前 run_id 兜底：generator 里 run_id=yb_bench_{session}
        天然按会话稳定，使同一会话多轮请求经 ConsistentHash 落到同一推理实例，
        命中该实例的 prefix/KV cache（即"打开 Taiji prompt cache"）。
        """
        return config.TAIJI_STRATEGY, self._resolve_session_id(config.TAIJI_SESSION_ID)

    def _csig_routing(self) -> tuple[str, str]:
        """计算 CSIG 的粘连路由 key（session_id）。

        session_id 会被 backend_csig.forward 写入请求体 ``user`` 字段：CSIG 上游
        litellm proxy 据此做一致性哈希，把同会话/同前缀的请求粘连到同一推理实例，
        命中该实例上由 cache_control 创建的 Anthropic prefix cache（实测 ~7/8）。
        返回的 strategy 仅作无害冗余 header（CSIG 不识别 Taiji 式 StrategyType）。
        """
        return config.CSIG_STRATEGY, self._resolve_session_id(config.CSIG_SESSION_ID)

    @staticmethod
    def _normalize_response(resp: dict) -> dict:
        """兜底规整，保证能通过 ExtendedChatCompletion 校验：
        - 补齐必填字段 id / created / object；
        - 保证每个 choice 的 finish_reason 非空。
        """
        resp.setdefault('id', '')
        resp.setdefault('created', 0)
        resp.setdefault('object', 'chat.completion')
        for choice in resp.get('choices', []):
            if choice.get('finish_reason'):
                continue
            msg = choice.get('message', {}) or {}
            choice['finish_reason'] = 'tool_calls' if msg.get('tool_calls') else 'stop'
        return resp

    @staticmethod
    def _append_jsonl(file_path: str, payload: dict) -> None:
        """将一条记录以 raw JSON 字符串追加写入 jsonl 文件。"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a+', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    async def _create_completion(
        self,
        messages: list[Any],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        model = model or self.config.request_args.get('model')
        if not model:
            raise ValueError('model must be specified either in call arguments or in request_args')
        route = resolve(model)

        # 组装请求体：request_args（去掉 model）+ 调用方 kwargs（tools 等）
        request_args = {k: v for k, v in self.config.request_args.items() if k != 'model'}
        body: dict = {
            'model': route.model,
            'messages': list(messages),
            **request_args,
            **kwargs,
        }
        # tools=None 时不要传，避免后端报错
        if body.get('tools') is None:
            body.pop('tools', None)

        if route.backend == 'csig':
            strategy, session_id = self._csig_routing()
            assembled = await backend_csig.forward(
                self._client,
                body,
                route,
                strategy_type=strategy,
                session_id=session_id,
            )
        elif route.backend == 'taiji':
            strategy, session_id = self._taiji_routing()
            assembled = await backend_taiji.forward(
                self._client,
                body,
                route,
                strategy_type=strategy,
                session_id=session_id,
            )
        else:
            raise ValueError(
                f"one_agent_direct: unsupported backend '{route.backend}' for model "
                f"'{model}'. Supported: {backend_supported()}"
            )

        # 还原对外 model 名 + 规整
        assembled['model'] = model
        assembled = self._normalize_response(assembled)

        response_file_path = os.getenv('OPENAI_RESPONSE_FILE_PATH', None)
        if response_file_path:
            try:
                self._append_jsonl(
                    response_file_path,
                    {
                        'request': body,
                        'response': assembled,
                    },
                )
            except Exception:
                logger.exception('Failed to append OpenAI request/response to %s', response_file_path)

        return ExtendedChatCompletion.model_validate(assembled)


def backend_supported() -> tuple[str, ...]:
    from .router import SUPPORTED_BACKENDS

    return SUPPORTED_BACKENDS
