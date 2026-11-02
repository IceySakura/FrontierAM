from __future__ import annotations

import base64
import json
import logging

from pathlib import Path
from urllib.parse import quote

import aiofiles
import aiohttp


APK_DOWNLOAD_BASE_URL = 'https://agentsandbox-1251707795.cos.ap-guangzhou.myqcloud.com/repo/apk'

logger = logging.getLogger(__name__)

APP_CONFIGS = {
    'yyb': {
        'name': 'App Store',
        'package': 'com.tencent.android.qqdownloader',
        'activity': 'com.tencent.assistantv2.activity.MainActivity',
        'apk_name': '应用宝_0302.apk',
        'remote_path': '/data/local/tmp/yyb.apk',
        'permissions': [
            'android.permission.ACCESS_FINE_LOCATION',
            'android.permission.ACCESS_COARSE_LOCATION',
            'android.permission.READ_EXTERNAL_STORAGE',
            'android.permission.WRITE_EXTERNAL_STORAGE',
        ],
    }
}


def save_base64_png(image_base64: str, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    padding = len(image_base64) % 4
    if padding:
        image_base64 += '=' * (4 - padding)
    image_data = base64.b64decode(image_base64)
    output_path.write_bytes(image_data)
    return str(output_path)


async def download_apk(apk_name: str, save_path: Path) -> bool:
    remote_name = apk_name.replace('.apk', '.ap')
    download_url = f'{APK_DOWNLOAD_BASE_URL}/{quote(remote_name)}'

    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug('Downloading AGR mobile APK: %s -> %s', download_url, save_path)

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(download_url) as response:
            response.raise_for_status()
            async with aiofiles.open(save_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    if chunk:
                        await f.write(chunk)
    return True


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return str(path)


def write_text(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return str(path)


def build_run_dir(output_dir: str | None, run_id: str | None) -> Path | None:
    if not output_dir:
        return None
    run_dir = Path(output_dir) / str(run_id or 'unknown_run')
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def ensure_apk_local(apk_dir: str | None, app_config: dict[str, object]) -> tuple[Path, bool]:
    apk_root = Path(apk_dir) if apk_dir else Path(__file__).resolve().parents[4] / 'tests' / 'e2e' / 'apk'
    apk_root.mkdir(parents=True, exist_ok=True)

    apk_name = str(app_config['apk_name'])
    apk_local_path = apk_root / apk_name
    apk_downloaded = False
    if not apk_local_path.exists():
        await download_apk(apk_name, apk_local_path)
        apk_downloaded = True
    else:
        logger.debug('Reusing cached AGR mobile APK: %s', apk_local_path)

    return apk_local_path, apk_downloaded
