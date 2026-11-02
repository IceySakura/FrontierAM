"""Exercise a real AGRT Windows GPU sandbox without an LLM.

Set AGRT_API_KEY in the environment, then run:
    uv run python docs/examples/scripts/test_agrt_windows_gpu.py

The script loads the example YAML, saves result.json and a screenshot, and
always releases the instance. Never pass credentials on the command line.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import struct
import sys
import uuid

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nexus.runtime.providers.agrt import AGRTRuntimeProvider
from nexus.runtime.providers.ags.waa import require_raw_waa_runtime
from nexus.runtime.providers.registry import runtime_provider_registry
from nexus.utils.async_http import http_client


REPO_ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def _record_check(report: dict[str, Any], name: str):
    try:
        yield
    except Exception as exc:
        report.setdefault('errors', {})[name] = f'{type(exc).__name__}: {exc}'


async def run(config_path: Path, output_dir: Path) -> bool:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict) or config.get('runtime_provider_name') != 'agrt':
        raise ValueError('Expected an agrt runtime provider configuration')
    provider = runtime_provider_registry.get('agrt')(**config['runtime_provider_config'])
    assert isinstance(provider, AGRTRuntimeProvider)
    for name, params in config.get('runtime_configs', {}).items():
        provider.configure(name, **params)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {'success': False, 'checks': {}, 'cleanup': {'confirmed': False}}
    instance_id = None
    api_key = str(provider.get_params('main').get('api_key') or provider.api_key)
    stage = 'acquire'
    try:
        print('Creating AGRT Windows GPU instance...', flush=True)
        runtime = await provider.acquire('main')
        instance_id = provider.get_runtime_id('main')
        report['instance_id'] = instance_id
        print('Instance ready; checking Windows, GPU, files and screenshot.', flush=True)
        raw = require_raw_waa_runtime(runtime)
        nexus = raw.to_nexus()

        details: dict[str, Any] = {}
        with _record_check(report, 'platform'):
            platform = await raw.platform()
            report['checks']['platform'] = platform
            assert 'Windows' in platform, f'Unexpected platform: {platform!r}'

        with _record_check(report, 'command'):
            command = await nexus.run_command(
                'python -c "import json,platform,tempfile; '
                "print(json.dumps({'system':platform.system(),'temp':tempfile.gettempdir()}))\"",
                timeout=30,
            )
            report['checks']['command'] = command.model_dump(mode='json')
            assert command.return_code == 0, 'Windows command failed'
            assert command.stdout, 'Windows command returned no output'
            details = json.loads(command.stdout)
            assert details['system'] == 'Windows'

        with _record_check(report, 'file_roundtrip'):
            remote_path = details['temp'].rstrip('\\/') + f'\\nexus_agrt_{uuid.uuid4().hex}.txt'
            expected = 'Nexus AGRT Windows GPU file roundtrip: 中文\n'
            # WAA setup_create_file uses the guest's default encoding. Seed an ASCII
            # file through that API, then explicitly write UTF-8 through Nexus.
            await raw.setup_create_file(remote_path, 'initial')
            assert await raw.get_file(remote_path) == b'initial'
            ps_path = remote_path.replace("'", "''")
            ps_content = expected.replace("'", "''")
            script = (
                '$utf8 = New-Object System.Text.UTF8Encoding($false); '
                f"[System.IO.File]::WriteAllText('{ps_path}', '{ps_content}', $utf8)"
            )
            encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
            written = await nexus.run_command(
                f'powershell.exe -NoProfile -NonInteractive -EncodedCommand {encoded}', timeout=30
            )
            assert written.return_code == 0, 'Explicit UTF-8 file write failed'
            downloaded = await raw.get_file(remote_path)
            (output_dir / 'roundtrip.txt').write_bytes(downloaded)
            assert downloaded.decode('utf-8-sig').replace('\r\n', '\n') == expected, 'File content mismatch'
            report['checks']['file_roundtrip'] = {'matched': True, 'bytes': len(downloaded)}

        with _record_check(report, 'gui'):
            size = await nexus.gui_get_screen_size()
            report['checks']['screen_size'] = size.model_dump(mode='json')
            assert (size.width or 0) > 0
            assert (size.height or 0) > 0
            shot = await nexus.gui_screenshot()
            assert shot.success, 'GUI screenshot failed'
            assert shot.image, 'GUI screenshot returned no image'
            png = base64.b64decode(shot.image)
            assert png.startswith(b'\x89PNG\r\n\x1a\n'), 'Screenshot is not PNG'
            width, height = struct.unpack('>II', png[16:24])
            assert width > 0
            assert height > 0
            (output_dir / 'desktop.png').write_bytes(png)
            report['checks']['screenshot'] = {'width': width, 'height': height, 'bytes': len(png)}

        with _record_check(report, 'renew_and_refresh'):
            await provider.renew('main', '30m')
            await provider.refresh_token('main', 1800)
            assert 'Windows' in await raw.platform()
            report['checks']['renew_and_refresh'] = True

        with _record_check(report, 'gpu'):
            # WAA readiness does not guarantee that the GPU driver has finished starting.
            report['checks']['gpu_attempts'] = []
            for attempt in range(4):
                gpu = await nexus.run_command(
                    'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader', timeout=30
                )
                report['checks']['gpu_attempts'].append(gpu.model_dump(mode='json'))
                if gpu.return_code == 0:
                    break
                if attempt < 3:
                    await asyncio.sleep(10)
            report['checks']['gpu'] = gpu.model_dump(mode='json')
            if gpu.return_code != 0:
                diagnostics = await nexus.run_command(
                    'powershell.exe -NoProfile -NonInteractive -Command '
                    '"Get-CimInstance Win32_VideoController | '
                    'Select-Object Name,PNPDeviceID,Status,ConfigManagerErrorCode,DriverVersion | '
                    'ConvertTo-Json -Compress"',
                    timeout=30,
                )
                report['checks']['gpu_diagnostics'] = diagnostics.model_dump(mode='json')
            assert gpu.return_code == 0, 'nvidia-smi failed after readiness retries; see gpu_diagnostics'
            assert (gpu.stdout or '').strip(), 'nvidia-smi did not report a GPU'
        report['success'] = not report.get('errors')
    except Exception as exc:
        report['failed_stage'] = stage
        report['error'] = f'{type(exc).__name__}: {exc}'.replace(api_key, '<redacted>')
    finally:
        instance_id = instance_id or provider.get_runtime_id('main')
        report['instance_id'] = instance_id
        try:
            await provider.release('main')
            if instance_id:
                # A successful stop request can mean STOPPING; read back the final state.
                deadline = asyncio.get_running_loop().time() + 180
                while True:
                    state = await provider._api('GET', f'/v1/instances/{instance_id}', api_key=api_key)
                    status = state.get('status')
                    report['cleanup']['status'] = status
                    if status in {'STOPPED', 'EXPIRED'}:
                        break
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError(f'Stop did not finish: status={status}')
                    await asyncio.sleep(5)
            report['cleanup']['confirmed'] = True
        except Exception as exc:
            report['success'] = False
            report['cleanup']['error'] = f'{type(exc).__name__}: {exc}'.replace(api_key, '<redacted>')
        await http_client.close()
        serialized = json.dumps(report, indent=2, ensure_ascii=False).replace(api_key, '<redacted>')
        (output_dir / 'result.json').write_text(serialized + '\n')
        print(serialized, flush=True)
        print(f'Results: {output_dir.resolve()}', flush=True)
    return report['success']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'docs/examples/configs/agrt_windows_gpu.yaml')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'outputs/agrt_windows_gpu')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(0 if asyncio.run(run(args.config, args.output_dir)) else 1)


if __name__ == '__main__':
    main()
