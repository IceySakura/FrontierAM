import logging
import traceback

from typing import Any

import aiohttp


logger = logging.getLogger(__name__)


class AsyncLean4Client:
    def __init__(
        self,
        polaris_url: str = 'http://lean4-server-test-v2.production.polaris:8080',
        retry_num: int = 3,
        repo_path: str = '/miniF2F-lean4',
        timeout: int = 600,
    ) -> None:
        self.polaris_url = polaris_url.replace('http://', '').replace('https://', '')
        self.repo_path = repo_path
        self.timeout = timeout
        self.retry_num = retry_num

    async def async_trace(
        self,
        statement: str,
        tactic_list: list[str] | None = None,
        deps: list[str] | None = None,
        defs: list[str] | None = None,
    ) -> dict[str, Any]:
        if tactic_list is None:
            tactic_list = []
        if deps is None:
            deps = []
        if defs is None:
            defs = []
        req_data = {
            'repo_path': self.repo_path,
            'statement': statement,
            'tactics': tactic_list,
            'deps': deps,
            'defs': defs,
        }

        error = 'not healthy'
        for _ in range(self.retry_num):
            url = f'http://{self.polaris_url}/proof_trace'
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=req_data, proxy=None, timeout=self.timeout) as response:
                        json_res = await response.json()
                        if json_res:
                            return json_res
            except Exception as e:
                traceback.print_exc()
                error = str(e)
        return {'error': error}

    async def async_check_whole(self, cmd: str) -> dict[str, Any]:
        req_data = {'repo_path': self.repo_path, 'cmd': cmd, 'timeout': self.timeout, 'return_repl_res': True}

        exception = 'not healthy'
        for attempt in range(1, self.retry_num + 1):
            url = f'http://{self.polaris_url}/check_whole'
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=req_data, proxy=None, timeout=self.timeout) as response:
                        json_res = await response.json()
                        if response.status == 200 and json_res:
                            if json_res.get('exception'):
                                exception = json_res.get('exception')
                                continue
                            return json_res
            except Exception as exc:
                exception = str(exc)
                logger.warning(
                    'async_check_whole attempt %d/%d failed: %r',
                    attempt,
                    self.retry_num,
                    exc,
                )
                continue
        return {'is_ok': False, 'exception': exception}
