"""AIHub Bedrock Invoke backend, registered only inside the V2 host worker."""

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import time
import uuid

from dataclasses import replace
from pathlib import Path

from nexus.core import OpenAIConfig
from nexus.extensions.openai_clients.anthropic_messages import AnthropicMessagesClient, AnthropicMessagesRequestError


class AWSThirdRequestError(RuntimeError):
    """Expose only the status to the upstream retry classifier, never credentials."""

    def __init__(self, status):
        self.status_code = status
        super().__init__(f'V2 aws_third Bedrock Invoke request failed (status={status})')


def register():
    # The pinned OSWorld source is available only after the worker loads it.
    from desktop_env.evaluators.backends import ModelBackend, register_backend
    from desktop_env.image_utils import anthropic_image_file_to_mime_and_base64

    @register_backend('aws_third')
    class AWSThirdBackend(ModelBackend):
        def __init__(self, config):
            # The official base owns retries. Do not multiply them by SDK retries.
            super().__init__(
                replace(
                    config,
                    retry_attempts=max(1, min(config.retry_attempts, 3)),
                    retry_delay=max(0, min(config.retry_delay, 10)),
                )
            )
            timeout = float(os.environ.get('OSWORLD_AWS_THIRD_TIMEOUT', '120'))
            if not math.isfinite(timeout) or not 0 < timeout <= 300:
                raise ValueError('OSWORLD_AWS_THIRD_TIMEOUT must be in (0, 300] seconds')
            if not config.base_url:
                raise ValueError('V2 aws_third requires an explicit AIHub base_url')
            if config.max_tokens <= 0:
                raise ValueError('V2 aws_third max_tokens must be positive')
            self.timeout = timeout
            request_args = {'model': config.model, 'max_tokens': config.max_tokens}
            if config.temperature is not None:
                request_args['temperature'] = config.temperature
            self.client = AnthropicMessagesClient(
                OpenAIConfig(
                    client_args={
                        'base_url': config.base_url,
                        'api_key': config.api_key,
                        'api_style': 'bedrock_invoke',
                        'provider': 'aws_third',
                        'auth_style': 'bearer',
                        'timeout': timeout,
                        'max_retries': 1,
                    },
                    request_args=request_args,
                )
            )

        def _bounded(self, function, *, image_paths=()):
            from desktop_env.evaluators.backends.base import is_transient_error

            call_id = uuid.uuid4().hex
            images = [
                {'name': Path(p).name, 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in image_paths
            ]
            for attempt in range(1, self.config.retry_attempts + 1):
                start = time.monotonic()
                record = {
                    'call_id': call_id,
                    'attempt': attempt,
                    'max_attempts': self.config.retry_attempts,
                    'model': self.config.model,
                    'images': images,
                }
                try:
                    result = function()
                    record.update(ok=True, output_length=len(result))
                    return result
                except Exception as exc:
                    retry = attempt < self.config.retry_attempts and is_transient_error(exc)
                    delay = min(10, self.config.retry_delay * 2 ** (attempt - 1))
                    delay = min(10, delay + random.uniform(0, min(1, delay / 4))) if delay else 0
                    record.update(
                        ok=False,
                        error_type=type(exc).__name__,
                        status=getattr(exc, 'status_code', None),
                        retry=retry,
                        retry_delay=delay if retry else 0,
                    )
                    if not retry:
                        raise
                finally:
                    record['elapsed_s'] = round(time.monotonic() - start, 3)
                    directory = os.environ.get('OSWORLD_AUX_DIAGNOSTIC_DIR')
                    if directory:
                        try:
                            with (Path(directory) / 'auxiliary-model-calls.jsonl').open('a') as stream:
                                stream.write(json.dumps(record) + '\n')
                        except OSError:
                            logging.getLogger(__name__).warning('Cannot save auxiliary model diagnostics')
                time.sleep(delay)
            raise RuntimeError('Auxiliary model retry budget exhausted')

        def generate(self, prompt, images, *, system=None):
            return self._bounded(lambda: self._generate_once(prompt, images, system=system), image_paths=images)

        def chat(self, messages):
            return self._bounded(lambda: self._chat_once(messages))

        def _generate_once(self, prompt, images, *, system=None):
            content = []
            for path in images:
                mime, encoded = anthropic_image_file_to_mime_and_base64(path)
                content.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}})
            content.append({'type': 'text', 'text': prompt})
            messages = [{'role': 'system', 'content': system}] if system else []
            messages.append({'role': 'user', 'content': content})
            return self._chat_once(messages)

        def _chat_once(self, messages):
            # Official evaluators and user simulators are synchronous; they run
            # in the dedicated worker, not the host runner's asyncio event loop.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError('V2 aws_third must run in the synchronous OSWorld worker')

            async def request():
                try:
                    result = await asyncio.wait_for(self.client._create_completion(messages), timeout=self.timeout + 5)
                except AnthropicMessagesRequestError as exc:
                    raise AWSThirdRequestError(exc.status) from exc
                if not result.choices:
                    raise RuntimeError('V2 aws_third returned no choices')
                choice = result.choices[0]
                if choice.finish_reason == 'length':
                    raise RuntimeError('V2 aws_third response truncated; increase auxiliary model max_tokens')
                text = choice.message.content
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError('V2 aws_third returned an empty text response')
                return text.strip()

            return asyncio.run(request())

    return AWSThirdBackend
