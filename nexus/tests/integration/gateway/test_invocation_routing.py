"""Route ordinary and summary requests through two local HTTP model services."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from aiohttp import web

from nexus.core.openai_client import OpenAIConfig
from nexus.extensions.openai_clients.openai import OpenAIClient
from nexus.gateway.client import GatewayOpenAIClient
from nexus.gateway.policies.hycode import COMPACT_TRIGGER_MARKER
from nexus.gateway.session_manager import session_manager


async def test_invocation_routes_record_both_http_models_and_clean_session() -> None:
    requests: dict[str, list[dict[str, Any]]] = {'main': [], 'summary': []}
    models = []
    trajectory_id = 'http-invocation-routing'
    async with AsyncExitStack() as stack:
        for model_id, model_name in [('main', 'Qwen3-32B'), ('summary', 'Qwen3-30B-A3B')]:

            async def complete(request: web.Request, model_id: str = model_id) -> web.Response:
                requests[model_id].append(await request.json())
                return web.json_response({
                    'id': f'{model_id}-{len(requests[model_id])}',
                    'object': 'chat.completion',
                    'created': 1,
                    'model': 'server-model',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': '19' if model_id == 'main' else '12 + 7 = 19'},
                            'finish_reason': 'stop',
                        }
                    ],
                })

            app = web.Application()
            app.router.add_post('/v1/chat/completions', complete)
            runner = web.AppRunner(app)
            await runner.setup()
            stack.push_async_callback(runner.cleanup)
            await web.TCPSite(runner, '127.0.0.1', 0).start()
            service = OpenAIClient(
                OpenAIConfig(
                    client_args={
                        'base_url': f'http://127.0.0.1:{runner.addresses[0][1]}/v1',
                        'api_key': 'local-test',
                        'max_retries': 0,
                    },
                    request_args={'model': model_name},
                )
            )
            stack.push_async_callback(service.client.close)
            models.append({'id': model_id, 'openai_client': service})

        client = GatewayOpenAIClient(
            OpenAIConfig(
                client_args={
                    'trajectory_id': trajectory_id,
                    'scaffold_policy': 'claude_code',
                    'models': models,
                    'default_model_id': 'main',
                    'compaction_model_id': 'summary',
                },
                request_args={'max_tokens': 100},
            )
        )
        try:
            history = [{'role': 'user', 'content': 'What is 12 + 7?'}]
            for prompt, expected in [
                (None, '19'),
                (COMPACT_TRIGGER_MARKER + '\nSummarize the calculation.', '12 + 7 = 19'),
                ('Continue with the final answer.', '19'),
            ]:
                if prompt is not None:
                    history.append({'role': 'user', 'content': prompt})
                response = await client.create_completion(history)
                assert response.choices[0].message.content == expected, 'Use the endpoint selected by invocation'
                history.append(response.choices[0].message.model_dump(exclude_none=True))
            trajectory = session_manager.end_session(trajectory_id)
        finally:
            if session_manager.has_session(trajectory_id):
                await session_manager.abort_in_flight(trajectory_id)
                session_manager.end_session(trajectory_id)

    nodes = [node for node in trajectory.iter_nodes() if node.generated_by == 'llm']
    assert [node.provider_facts.model_id for node in nodes] == ['main', 'summary', 'main'], 'Record registry identities'
    assert {node.provider_facts.model for node in nodes} == {'server-model'}, 'Keep response model separately'
    assert [request['model'] for request in requests['main']] == ['Qwen3-32B', 'Qwen3-32B'], 'Use main wire defaults'
    assert [request['model'] for request in requests['summary']] == ['Qwen3-30B-A3B'], 'Use summary wire defaults'
    assert not session_manager.has_session(trajectory_id), 'Release the shared session'
