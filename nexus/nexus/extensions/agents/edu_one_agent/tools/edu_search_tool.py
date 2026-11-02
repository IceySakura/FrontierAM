"""edu_search_tool for edu_one_agent"""

from __future__ import annotations

import asyncio
import os

import aiohttp

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


EDU_SEARCH_TOOL_ENDPOINT = os.environ.get('EDU_SEARCH_TOOL_ENDPOINT', 'http://11.141.185.157:8003/batch_tool_call')
EDU_SEARCH_TOOL_TIMEOUT = float(os.environ.get('EDU_SEARCH_TOOL_TIMEOUT', '30'))


class EduSearchInput(BaseModel):
    query: str = Field(description='用户输入或 OCR 提取出的题目文本')
    image_id: str | None = Field(
        description='可选。当本次检索需结合用户上传的图片时填写，值为该图片的序号（数字字符串），'
        '取自上下文中的 [user_image N] 标签。例如 [user_image 1] 则填写 "1"。'
        '不要填写图片 URL、图片内容或 user_image 前缀。每次调用最多引用一张图片。',
        default=None,
    )


@tool_registry.register(
    'edu_one_agent/edu_search',
    meta={
        'name': 'edu_search',
        'description': (
            '教育知识库原题搜索工具。用于在教育题库中检索与用户问题（题目）匹配的原题及其答案、解析。'
            '识别到"题目+求解/求答案/求解析"意图时，优先调用本工具而非通用搜索工具。'
            '题库的答案权威性和解析质量高于网页搜索结果。'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def edu_search_tool(params: EduSearchInput, context: ToolContext) -> ToolResponse:
    image_refs = context.tool_state.get('image_refs')

    timeout = aiohttp.ClientTimeout(total=EDU_SEARCH_TOOL_TIMEOUT)

    if image_refs is not None and params.image_id is not None:
        payload = {
            'calls': [
                {
                    'call_id': 'nexus-edu-one-agent',
                    'name': 'edu_search',
                    'version': 'v1.0',
                    'arguments': {
                        'query': params.query,
                        'image_id': params.image_id,
                    },
                }
            ],
            'options': {'is_vlm': True},
            'context_in': {
                'user': {'user_id': 'nexus-edu-one-agent-trainer'},
                'session': {'round': 1, 'history': [], 'raw_query': '', 'image_refs': image_refs},
                'caller': {'call_source': 'training'},
            },
        }
    else:
        payload = {
            'calls': [
                {
                    'call_id': 'nexus-edu-one-agent',
                    'name': 'edu_search',
                    'version': 'v1.0',
                    'arguments': {
                        'query': params.query,
                        'image_id': None,
                    },
                }
            ],
            'options': {'is_vlm': True},
            'context_in': {
                'user': {'user_id': 'nexus-edu-one-agent-trainer'},
                'session': {
                    'round': 1,
                    'history': [],
                    'raw_query': '',
                },
                'caller': {'call_source': 'training'},
            },
        }
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            EDU_SEARCH_TOOL_ENDPOINT,
            json=payload,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'nexus-edu-one-agent',
            },
        ) as response:
            response.raise_for_status()
            data = await response.json(content_type=None)

    results = data.get('results', []) if isinstance(data, dict) else []
    texts: list[str] = []
    if results:
        chunks = results[0].get('for_model', []) or []
        texts = [c.get('content', '') for c in chunks if c.get('type') == 'text' and c.get('content')]
    text = '\n'.join(texts)
    # assert len(text) > 0
    # TODO: 注意 edu-search 必有非空 text; 此处仅为**临时兜底**
    text = text if text else '[未找到相关结果]'

    return ToolResponse(output=[{'type': 'text', 'text': text}])


if __name__ == '__main__':
    import asyncio

    async def _main() -> int:
        def print_resp(resp):
            print('-' * 60)
            for part in resp.output:
                if isinstance(part, dict) and part.get('type') == 'text':
                    print(part.get('text', ''))
            print('-' * 60)

        query = '一元二次方程的求根公式是什么？'

        print(f'[edu_search_tool] query={query}')
        print(f'[edu_search_tool] endpoint={EDU_SEARCH_TOOL_ENDPOINT}')

        print_resp(await edu_search_tool(EduSearchInput(query=query), ToolContext()))

        print(f'[edu_search_tool] query={query}')
        print(f'[edu_search_tool] endpoint={EDU_SEARCH_TOOL_ENDPOINT}')

        url = 'https://yuanbao-model-engin-1258344699.cos.ap-beijing-fsi.myqcloud.com/edu_search/test_edu_one_agent_demo_image.jpg?q-sign-algorithm=sha1&q-ak=AKIDqoriNtFQRFtZ3KOHxQhDXfZHoGcmkYAc&q-sign-time=1781244184;1781330584&q-key-time=1781244184;1781330584&q-header-list=host&q-url-param-list=&q-signature=f600dcd1edcd02b17c80f0f92a922d3b24f70882'
        image_refs = {
            'url': url,
            'resized_url': url,
            'width': 567,
            'height': 877,
        }
        print_resp(
            await edu_search_tool(
                EduSearchInput(query=query, image_id='1'), ToolContext(tool_state={'user_image 1': image_refs})
            )
        )
        return 0

    raise SystemExit(asyncio.run(_main()))
