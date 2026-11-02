"""SmartRun macOS Runtime 使用示例。

运行：SMARTRUN_TOKEN=<token> uv run python docs/examples/scripts/test_smartrun_macos.py
可选：--device-id（或 SMARTRUN_DEVICE）、--model、--output-dir、--duration。

通过 SmartRunRuntime 的 run_command / gui_* 验证 TextEdit 输入、保存、截图和文件回读。
Provider 负责申请和释放设备；释放后额外查询 MCP 会话状态核对清理结果。
设备需已登录桌面并允许远程 GUI 操作。文件上传下载、滚动不在本示例范围内。
输出保存到 output/smartrun_macos_runtime_<timestamp>/，不包含 API token。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import html
import json
import os
import shlex

from datetime import datetime
from pathlib import Path
from typing import Any

from nexus.runtime.providers.smartrun.client import SmartRunRuntime
from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider
from nexus.utils.log import setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='SmartRun macOS Runtime smoke test')
    parser.add_argument('--device-id', default=os.getenv('SMARTRUN_DEVICE'))
    parser.add_argument('--model', help='例如 Virtual Mac 或 Macmini9,1；默认自动分配')
    parser.add_argument('--api-base-url', default='https://smartrun.woa.com')
    parser.add_argument('--duration', type=int, default=600)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    args = parser.parse_args()
    if not os.getenv('SMARTRUN_TOKEN'):
        parser.error('请通过 SMARTRUN_TOKEN 提供凭据')
    return args


async def demo_macos(runtime: SmartRunRuntime, run_dir: Path, steps: list[dict[str, Any]]) -> None:
    """通过 Runtime 验证 shell 和 GUI 输入，以文件回读确认真实效果。"""

    async def snapshot(label: str, detail: str = '') -> None:
        entry = {'number': len(steps) + 1, 'label': label, 'detail': detail}
        steps.append(entry)
        print(f'[{len(steps):02d}] {label}', flush=True)
        await asyncio.sleep(0.3)
        shot = await runtime.gui_screenshot()
        if not shot.success or not shot.image:
            entry['error'] = str(shot.error)
            raise RuntimeError(f'screenshot failed: {shot.error}')
        filename = f'{len(steps):02d}.jpg'
        (run_dir / filename).write_bytes(base64.b64decode(shot.image))
        entry['image'] = filename
        (run_dir / 'steps.json').write_text(json.dumps(steps, ensure_ascii=False, indent=2) + '\n')

    async def command(text: str, label: str) -> str:
        result = await runtime.run_command(text, timeout=30)
        await snapshot(label, f'{text}\n\nreturn_code={result.return_code}\n{result.stdout or ""}')
        if result.return_code != 0:
            raise RuntimeError(f'command failed: {result}')
        return result.stdout or ''

    async def check(awaitable: Any, label: str) -> None:
        result = await awaitable
        await snapshot(label, f'success={result.success}')
        if not result.success:
            raise RuntimeError(str(result))

    await snapshot('申请设备并连接 SmartRunRuntime')
    system = await command('uname -s && uname -m && sw_vers', 'run_command · 检查系统')
    if 'Darwin' not in system:
        raise RuntimeError(f'Unexpected system: {system}')
    print(system)
    remote_dir = (await command('mktemp -d /tmp/nexus-smartrun-runtime.XXXXXX', 'run_command · 创建临时目录')).strip()
    if not remote_dir.startswith('/tmp/nexus-smartrun-runtime.') or '\n' in remote_dir:
        raise RuntimeError('Unexpected temporary directory')
    document = shlex.quote(remote_dir + '/demo.txt')
    try:
        await command(f'touch {document}', 'run_command · 创建空白文档')
        await check(runtime.gui_launch_app('com.apple.TextEdit'), 'gui_launch_app · 启动 TextEdit')
        await command(f'open -a TextEdit {document}', 'run_command · 打开测试文档')
        await asyncio.sleep(2)
        foreground = await command(
            '/usr/bin/lsappinfo info -only bundleid "$(/usr/bin/lsappinfo front)"', 'run_command · 检查前台应用'
        )
        if '"com.apple.TextEdit"' not in foreground:
            raise RuntimeError(f'TextEdit not in foreground: {foreground}')
        await check(runtime.gui_type('Nexus SmartRunRuntime macOS'), 'gui_type · 输入第一行')
        await check(runtime.gui_press_key('enter'), 'gui_press_key · Enter 换行')
        await check(runtime.gui_type('Typed through runtime executors'), 'gui_type · 输入第二行')
        await check(runtime.gui_hotkey(['cmd', 's']), 'gui_hotkey · Cmd+S 保存')
        await asyncio.sleep(1)
        saved = await command(f'cat {document}', 'run_command · 回读并核对保存内容')
        expected = 'Nexus SmartRunRuntime macOS\nTyped through runtime executors'
        if saved.rstrip('\n') != expected:
            raise RuntimeError(f'Saved GUI text mismatch: {saved!r}')
        (run_dir / 'saved_text.txt').write_text(saved)
        await check(runtime.gui_hotkey(['cmd', 'w']), 'gui_hotkey · Cmd+W 关闭测试文档')
    finally:
        quoted_dir = shlex.quote(remote_dir)
        await command(f'rm -rf -- {quoted_dir} && test ! -e {quoted_dir}', 'run_command · 清理并确认临时目录不存在')


def write_report(run_dir: Path, result: dict[str, Any]) -> None:
    """生成可离线查看的逐步截图报告；释放后不能再截图。"""
    cards = []
    for entry in result['steps']:
        picture = ''
        if entry.get('image'):
            encoded = base64.b64encode((run_dir / entry['image']).read_bytes()).decode()
            picture = f'<img loading="lazy" src="data:image/jpeg;base64,{encoded}" alt="操作后截图">'
        cards.append(
            f'<section id="step-{entry["number"]}"><h2>{entry["number"]:02d} · {html.escape(entry["label"])}</h2>'
            f'<details><summary>命令与返回值</summary><pre>{html.escape(entry["detail"])}</pre></details>'
            f'{picture}<p>{html.escape(entry.get("error", ""))}</p></section>'
        )
    passed = result['success'] and result['released']
    links = ''.join(
        f'<a href="#step-{entry["number"]}">{entry["number"]:02d} {html.escape(entry["label"])}</a>'
        for entry in result['steps']
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SmartRun macOS · 完整操作过程</title>
<style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eff4f5;color:#172e3c;font:16px/1.6 system-ui,sans-serif}
header,main,footer{max-width:1120px;margin:auto;padding:28px 22px}header{padding-top:48px}h1{font-size:38px;line-height:1.2}
h2{font-size:20px;margin:0 0 14px}p{color:#617582}.badge{display:inline-block;padding:5px 15px;background:#daeee5;color:#086e4c;border-radius:30px}
section{background:white;border:1px solid #d6e1e5;border-radius:16px;margin-bottom:28px;padding:22px;scroll-margin-top:15px}
img{width:100%;height:auto;display:block;border-radius:10px;margin-top:16px;cursor:zoom-in}pre{overflow:auto;background:#152f3e;color:#deeee9;padding:16px;border-radius:8px;font-size:13px}
summary{cursor:pointer;color:#447080}nav{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}nav a{font-size:12px;color:#285368;text-decoration:none;padding:5px 10px;background:#fff;border-radius:6px}
footer{font-size:13px}dialog{max-width:98vw;max-height:96vh;border:0;border-radius:12px;padding:12px}dialog img{width:1920px;max-width:none;margin:0}dialog button{position:sticky;top:0;left:0;padding:8px 20px;cursor:pointer}
@media(max-width:600px){h1{font-size:28px}section{padding:14px}main{padding:14px}}
</style><header><small>NEXUS / SMARTRUN RUNTIME</small><h1>macOS 完整操作过程</h1>
"""
    page += (
        f'<span class="badge">{"测试通过 · 会话已释放" if passed else "测试未完整通过"}</span>'
        f'<p>运行目录：{html.escape(run_dir.name)}<br>每次操作后均通过 SmartRunRuntime.gui_screenshot 保存截图。'
        'shell 操作通常不会改变桌面，因此相邻截图可能相同。点击图片可放大。</p>'
        f'<nav>{links}</nav></header><main>{"".join(cards)}'
        '<section><h2>最后一步 · 释放设备</h2>'
        f'<p>服务端会话结束确认：{result["released"]}。释放后不再访问设备，不生成伪造的释放后截图。</p>'
        f'<p>{html.escape(result.get("error", ""))}</p></section></main>'
        '<footer>本轮验证 shell、启动应用、输入、Enter、Cmd+S、文件回读、Cmd+W 和清理；'
        '不覆盖滚动、文件上传下载或 iOS 真机回归。图片内嵌，可离线打开。背景中可能保留此前测试窗口。</footer>'
        '<dialog id="viewer"><button onclick="this.parentElement.close()">关闭</button><img alt="放大截图"></dialog>'
        '<script>document.querySelectorAll("section img").forEach(img=>img.onclick=()=>{'
        'const d=document.getElementById("viewer");d.querySelector("img").src=img.src;d.showModal()});</script></html>'
    )
    (run_dir / 'report.html').write_text(page, encoding='utf-8')


async def main(args: argparse.Namespace) -> None:
    setup_logging()
    run_dir = args.output_dir / datetime.now().strftime('smartrun_macos_runtime_%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'输出目录: {run_dir}', flush=True)
    provider = SmartRunRuntimeProvider(
        platform='macos',
        model=args.model,
        duration=args.duration,
        api_base_url=args.api_base_url,
    )
    result: dict[str, Any] = {'success': False, 'released': False, 'platform': 'macos'}
    steps: list[dict[str, Any]] = []
    runtime: SmartRunRuntime | None = None
    try:
        params = {'device_id': args.device_id} if args.device_id else {}
        runtime = await provider.acquire('main', purpose='macOS runtime usage demo', **params)
        if not isinstance(runtime, SmartRunRuntime):
            raise TypeError(f'Expected SmartRunRuntime, got {type(runtime).__name__}')
        result['runtime_class'] = type(runtime).__name__
        if runtime.platform != 'macos':
            raise RuntimeError(f'Unexpected platform: {runtime.platform}')
        await demo_macos(runtime, run_dir, steps)
        result['success'] = True
    except Exception as exc:
        result['error'] = str(exc).replace(os.environ['SMARTRUN_TOKEN'], '<redacted>')
        raise
    finally:
        try:
            await provider.release_all()
            if runtime is not None:
                # Lifecycle audit only; all tested device operations use public Runtime methods.
                status = await runtime._call(
                    'get_session_status',
                    {
                        'session_id': runtime.session_id,
                        'json_output': True,
                    },
                )
                result['released'] = status.get('status') == 2
                if not result['released']:
                    raise RuntimeError(f'Session not released: {status}')
        finally:
            result['steps'] = steps
            write_report(run_dir, result)
            (run_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'测试通过，设备已释放；结果见 {run_dir}', flush=True)


if __name__ == '__main__':
    asyncio.run(main(_parse_args()))
