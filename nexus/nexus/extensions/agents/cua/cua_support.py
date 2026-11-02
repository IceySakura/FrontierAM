"""Model-independent CUA screenshots and trajectory persistence helpers."""

from __future__ import annotations

import base64
import logging
import os

from typing import Any


logger = logging.getLogger(__name__)


def _get(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def _instruction_text(user_prompt: Any) -> str:
    if isinstance(user_prompt, str):
        return user_prompt
    if isinstance(user_prompt, list):
        parts = []
        for p in user_prompt:
            if _get(p, 'type') == 'text':
                parts.append(_get(p, 'text', ''))
        return '\n'.join(x for x in parts if x)
    return str(user_prompt)


async def _screenshot_bytes(runtime: Any) -> bytes:
    resp = await runtime.gui_screenshot()
    image_b64 = _get(resp, 'image')
    if not image_b64:
        raise RuntimeError(f'gui_screenshot returned no image: {_get(resp, "error")}')
    return base64.b64decode(image_b64)


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode('utf-8')


def _traj_dir(agent_input: Any) -> str | None:
    # An explicit per-run directory wins over the root: a batch runs many tasks
    # concurrently in one process, and they all share one root.
    explicit = str(getattr(agent_input, 'trajectory_dir', '') or '').strip()
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    return None


def _dump_json(traj_dir: str | None, name: str, obj: Any) -> None:
    if not traj_dir:
        return
    try:
        import json as _json

        with open(os.path.join(traj_dir, name), 'w', encoding='utf-8') as f:
            _json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.warning('traj dump json failed: %s', exc)


def _init_traj_dir(traj_dir: str | None) -> None:
    if not traj_dir:
        return
    os.makedirs(os.path.join(traj_dir, 'screenshots'), exist_ok=True)


def _save_screenshot(traj_dir: str | None, name: str, png_bytes: bytes) -> None:
    if not traj_dir:
        return
    try:
        path = os.path.join(traj_dir, 'screenshots', name)
        with open(path, 'wb') as f:
            f.write(png_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.warning('traj save screenshot failed: %s', exc)


def _append_jsonl(traj_dir: str | None, filename: str, obj: Any) -> None:
    if not traj_dir:
        return
    try:
        import json as _json

        with open(os.path.join(traj_dir, filename), 'a', encoding='utf-8') as f:
            f.write(_json.dumps(obj, ensure_ascii=False, default=str) + '\n')
    except Exception as exc:  # noqa: BLE001
        logger.warning('traj append jsonl failed: %s', exc)


def _save_result(traj_dir: str | None, result_data: dict) -> None:
    if not traj_dir:
        return
    _dump_json(traj_dir, 'result.json', result_data)
