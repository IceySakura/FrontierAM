"""Web access for development containers that may use the public internet (blackbox-cubature v3.4, user decision
2026-11-26): an HTTP proxy on a Unix socket, bridged into the container by socat; the container itself keeps no network.

Allowed: any public destination over HTTPS (CONNECT to port 443) or plain HTTP (port 80). Refused:
* GitHub and its mirrors, as whole hostnames: the benchmark's repository (task generators, earlier solutions, reviews)
  is public there, and HTTPS shows the proxy only the hostname; the Mathlib documentation site stays reachable;
* package indexes: the submission is graded offline with the installed packages only;
* hosted language-model services: an attempt measures one model;
* other benchmarks whose problems a task reuses (OptBound): their sites may carry solutions or discussions;
* internal names, and every address in the private, carrier-grade NAT, link-local, multicast or internal ranges (the
  AIHub gateway, git.woa.com and the other internal services), checked on the address actually connected to.
Every request is one JSON line in the log.

Usage: python3 -m arena.netproxy SOCKET LOG    (prints "ready" once listening)
"""

import asyncio
import ipaddress
import json
import os
import socket
import sys
import time
import urllib.parse

BLOCKED_NETWORKS = [ipaddress.ip_network(n) for n in (
    '0.0.0.0/8', '9.0.0.0/8', '10.0.0.0/8', '11.0.0.0/8', '21.0.0.0/8', '30.0.0.0/8', '100.64.0.0/10', '127.0.0.0/8',
    '169.254.0.0/16', '172.16.0.0/12', '192.168.0.0/16', '224.0.0.0/3')]
ALLOWED = {'leanprover-community.github.io'}
MIRROR_SUBSTRINGS = ('github', 'ghproxy', 'fastgit', 'gitclone')
BENCHMARK_SUBSTRINGS = ('optbound',)
DENIED = {
    'mirror': ('gitee.com', 'softwareheritage.org', 'sourcegraph.com'),
    'package_index': ('pypi.org', 'pythonhosted.org', 'pypi.python.org', 'anaconda.org', 'anaconda.com',
                      'conda-forge.org', 'npmjs.org', 'npmjs.com', 'yarnpkg.com', 'crates.io', 'rubygems.org',
                      'proxy.golang.org', 'repo1.maven.org', 'download.pytorch.org', 'huggingface.co', 'hf.co'),
    'language_model': ('openai.com', 'chatgpt.com', 'anthropic.com', 'claude.ai', 'gemini.google.com',
                       'generativelanguage.googleapis.com', 'aiplatform.googleapis.com', 'deepseek.com',
                       'openrouter.ai', 'together.ai', 'together.xyz', 'groq.com', 'mistral.ai', 'cohere.ai',
                       'cohere.com', 'perplexity.ai', 'x.ai', 'grok.com', 'poe.com', 'moonshot.cn', 'kimi.ai',
                       'kimi.com', 'bigmodel.cn', 'zhipuai.cn', 'chatglm.cn', 'dashscope.aliyuncs.com', 'qwen.ai',
                       'tongyi.aliyun.com', 'qianfan.baidubce.com', 'yiyan.baidu.com', 'minimax.chat', 'minimaxi.com',
                       'stepfun.com', 'volces.com', 'doubao.com', 'siliconflow.cn', 'siliconflow.com', 'deepinfra.com',
                       'fireworks.ai', 'replicate.com', 'cerebras.ai', 'sambanova.ai', 'novita.ai', 'lingyiwanwu.com',
                       '01.ai', 'phind.com', 'you.com', 'aihubmix.com'),
    'internal': ('woa.com', 'oa.com'),
}
CONNECTIONS = 64
IDLE_SECONDS = 300


class Refused(Exception):
    pass


def internal(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return ip.version != 4 or any(ip in network for network in BLOCKED_NETWORKS)


def refusal(host: str) -> str | None:
    """Why a hostname is refused, or None."""
    host = host.lower().rstrip('.')
    if host in ALLOWED:
        return None
    if any(part in host for part in MIRROR_SUBSTRINGS):
        return 'github_or_mirror'
    if any(part in host for part in BENCHMARK_SUBSTRINGS):
        return 'benchmark'
    for reason, domains in DENIED.items():
        if any(host == domain or host.endswith('.' + domain) for domain in domains):
            return reason
    try:
        return 'internal_address' if internal(host) else None
    except ValueError:
        return None


async def resolve(host: str, port: int) -> str:
    """A public IPv4 address of host; refused if any of its addresses is internal."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, family=socket.AF_INET,
                                                              type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise Refused('unresolved') from error
    addresses = sorted({info[4][0] for info in infos})
    if not addresses:
        raise Refused('unresolved')
    if any(internal(address) for address in addresses):
        raise Refused('internal_address')
    return addresses[0]


async def pipe(source: asyncio.StreamReader, sink: asyncio.StreamWriter, counts: dict, key: str) -> None:
    try:
        while data := await asyncio.wait_for(source.read(65536), IDLE_SECONDS):
            counts[key] += len(data)
            sink.write(data)
            await sink.drain()
    except (asyncio.TimeoutError, ConnectionError, OSError):
        pass
    finally:
        try:
            sink.write_eof()
        except (OSError, RuntimeError):
            pass


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, log, slots: asyncio.Semaphore) -> None:
    start = time.monotonic()
    record = {'t': round(time.time(), 3)}
    upstream = None
    try:
        if slots.locked():
            raise Refused('too_many_connections')
        async with slots:
            head = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 30)
            request, *headers = head.decode('latin-1').split('\r\n')
            method, target, version = request.split(' ', 2)
            if method == 'CONNECT':
                host, _, port = target.rpartition(':')
                host, port = host.strip('[]'), int(port)
                record.update(method=method, host=host, port=port)
                if port != 443:
                    raise Refused('port')
            else:
                url = urllib.parse.urlsplit(target)
                host, port = url.hostname or '', url.port or 80
                record.update(method=method, host=host, port=port, path=url.path[:200])
                if url.scheme != 'http' or not host or port != 80:
                    raise Refused('not_an_http_proxy_request')
            if (reason := refusal(host)) is not None:
                raise Refused(reason)
            address = await resolve(host, port)
            record['ip'] = address
            upstream = await asyncio.wait_for(asyncio.open_connection(address, port), 30)
            if method == 'CONNECT':
                writer.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            else:
                kept = [h for h in headers if h and not h.lower().startswith(('proxy-', 'connection:', 'keep-alive:'))]
                path = (url.path or '/') + (f'?{url.query}' if url.query else '')
                upstream[1].write(f'{method} {path} {version}\r\n'.encode('latin-1')
                                  + '\r\n'.join([*kept, 'Connection: close']).encode('latin-1') + b'\r\n\r\n')
            await writer.drain()
            counts = {'up': 0, 'down': 0}
            await asyncio.gather(pipe(reader, upstream[1], counts, 'up'), pipe(upstream[0], writer, counts, 'down'))
            record.update(decision='allow', **counts)
    except Refused as error:
        record['decision'] = f'deny:{error}'
        writer.write(f'HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n'
                     f'Blocked by the evaluation network policy: {error}\n'.encode())
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError, OSError) as error:
        record['decision'] = f'error:{type(error).__name__}'
        if 'ip' in record and upstream is None:
            writer.write(b'HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n')
    finally:
        record['seconds'] = round(time.monotonic() - start, 3)
        log.write(json.dumps(record) + '\n')
        for stream in ([writer] + ([upstream[1]] if upstream else [])):
            try:
                stream.close()
            except OSError:
                pass


async def main(socket_path: str, log_path: str) -> None:
    log = open(log_path, 'a', buffering=1)
    slots = asyncio.Semaphore(CONNECTIONS)
    server = await asyncio.start_unix_server(lambda r, w: handle(r, w, log, slots), path=socket_path)
    os.chmod(socket_path, 0o666)
    print('ready', flush=True)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], sys.argv[2]))
