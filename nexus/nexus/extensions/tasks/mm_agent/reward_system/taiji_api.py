import asyncio
import base64
import io
import logging
import random

from functools import cache
from typing import Any

import httpx
import openai

from PIL import Image

from nexus.core import OpenAIConfig
from nexus.extensions.openai_clients.taiji import TaijiClient


headers = {'Content-Type': 'application/json'}
DEFAULT_NAMESPACE = 'Production'
DEFAULT_PORT = 8021


class _RewardTaijiClient(TaijiClient):
    """Taiji client variant for the reward model's legacy AIDE endpoint."""

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.reward_port = int(config.client_args.get('reward_port', DEFAULT_PORT))
        # Reuse one httpx.AsyncClient across every request for this cached
        # client. The parent _create_completion re-discovers the host and
        # calls _create_openai_client on *each* request, so building a fresh
        # httpx.AsyncClient here would leak a socket pool per judge call and
        # eventually exhaust fds under batch evaluation. Bound lazily to the
        # running event loop on first use.
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(trust_env=False, timeout=self.timeout)
        return self._http_client

    def _create_openai_client(self, host: str, port: int) -> openai.AsyncOpenAI:
        del port
        # The AsyncOpenAI wrapper is recreated per request because base_url
        # tracks the freshly discovered host, but it shares the long-lived
        # http_client below. openai-python does not take ownership of an
        # injected http_client, so discarding the wrapper won't close it.
        return openai.AsyncOpenAI(
            base_url=f'http://{host}:{self.reward_port}/v1',
            api_key='dummy',
            timeout=self.timeout,
            default_headers=headers,
            http_client=self._get_http_client(),
        )


@cache
def _get_reward_taiji_client(model_name: str, request_timeout: float) -> _RewardTaijiClient:
    return _RewardTaijiClient(
        OpenAIConfig(
            client_args={
                'model': model_name,
                'namespace': DEFAULT_NAMESPACE,
                'service': f'{model_name}_AIDE',
                'timeout': request_timeout,
                # Keep model-level retries in this module so retries can rotate
                # across the configured judge model list.
                'max_retries': 1,
                'url_pattern': 'v1',
                'reward_port': DEFAULT_PORT,
            },
            request_args={
                'temperature': 0.8,
                'top_p': 1.0,
            },
        )
    )


def _normalize_models(model: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if model is None:
        raise ValueError('get_model_response: MODEL must not be None')
    if isinstance(model, str):
        model = [model]
    models = [name for name in model if name]
    if not models:
        raise ValueError('get_model_response: MODEL must not be empty')
    return models


def image2byte(image):
    """
    图片转byte
    image: 必须是PIL格式
    image_bytes: 二进制
    """
    # 创建一个字节流管道
    img_bytes = io.BytesIO()
    # 将图片数据存入字节流管道， format可以按照具体文件的格式填写
    image.save(img_bytes, format='JPEG')
    # 从字节流管道中获取二进制
    return img_bytes.getvalue()


def _build_messages(img_path: str | None, input_text: str, system_text: str | None) -> list[dict[str, Any]]:
    image_str = None
    if img_path is not None:
        image = Image.open(img_path).convert('RGB')
        image_str = 'data:image;base64,{}'.format(base64.b64encode(image2byte(image)).decode('ascii'))

    chat_messages: list[dict[str, Any]] = []
    if system_text is not None and str(system_text).strip() != '':
        chat_messages.append({'role': 'system', 'content': system_text})

    content: list[dict[str, Any]] = []
    if image_str is not None:
        content.append({
            'type': 'image_url',
            'image_url': {'url': image_str},
        })
    content.append({'type': 'text', 'text': input_text})
    chat_messages.append({'role': 'user', 'content': content})
    return chat_messages


async def get_model_response(
    img_path,
    input_text,
    max_token=4096,
    max_retry=2,
    MODEL=None,  # noqa: N803 - keep compatibility with the legacy public API.
    system_text=None,
    request_timeout=500,
    temperature=0.8,
    top_p=1.0,
    top_k=None,
    enable_thinking=None,
):
    models = _normalize_models(MODEL)
    chat_messages = _build_messages(img_path, input_text, system_text)

    for attempt in range(max_retry):
        resp = None
        model_name_for_request = random.choice(models)
        try:
            client = _get_reward_taiji_client(model_name_for_request, float(request_timeout))
            completion_kwargs: dict[str, Any] = {
                'model': model_name_for_request,
                'temperature': temperature,
                'top_p': top_p,
                'max_tokens': max_token,
                'max_retries': 1,
            }
            if top_k is not None:
                completion_kwargs['top_k'] = top_k
            if enable_thinking is not None:
                completion_kwargs['enable_thinking'] = enable_thinking
            resp = await client.create_completion(
                chat_messages,
                **completion_kwargs,
            )
            answer = resp.choices[0].message.content
            if answer is None or len(answer.strip()) == 0:
                if attempt + 1 < max_retry:
                    print('Received None answer, retrying...')
                continue
            return answer
        except Exception as e:
            if attempt + 1 < max_retry:
                print('Received None answer, retrying...')
            print(f'call_gpt error: {e}')
            logging.error(f'call_gpt error: {e}, response: {getattr(resp, "text", "N/A")}')

    return None


if __name__ == '__main__':

    async def _main():
        img_path = (
            '/apdcephfs_cq8/share_1611098/shitaoxiao/rl/Angel-RL/my_scripts/tmp_data/'
            '1d87a9c7cb37daea76374ccd7995c1d5.png'
        )
        input_text = '简单描述图片内容。'
        answer = await get_model_response(img_path, input_text, MODEL='Qwen2.5-VL-72B-Instruct')
        print('Final answer: ', answer)

        input_text = '1+1=？'
        answer = await get_model_response(
            None,
            input_text,
            MODEL=['rm_gpt-oss_nj2', 'rm_gpt-oss_bj', 'rm_gpt-oss_sh', 'rm_gpt-oss_gy', 'rm_gpt-oss_nj_public'],
        )
        print('Final answer: ', answer)

    asyncio.run(_main())
