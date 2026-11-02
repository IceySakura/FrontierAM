"""把一个 Nexus Dataset AIDATA 实例物化为可运行的 case 包。

已发布的 AIDATA 数据集把每个 case 存为一条 JSON 实例，包含 ``files[]`` 记录。
AIDATA 运行器则期望一个位于磁盘上的 ``cases/<case_id>/`` 目录树。 本适配器
只负责在这两种本地形态之间搭桥；获取实例与运行 case 仍是独立的 CLI 步骤。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import shutil
import sys

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaterializedCase:
    """从一个本地 ``instance.json`` 恢复出来的 case 。"""

    instance_id: str
    case_id: str
    output_dir: Path
    cases_root: Path
    case_dir: Path
    instance_path: Path

    def run_env(self) -> dict[str, str]:
        return {
            'AIDATA_CASES_ROOT': str(self.cases_root),
            'AIDATA_CASE_IDS': self.case_id,
        }


_RUN_GENERATOR = '''\
"""Generated AIDATA batch generator for this materialized Nexus Dataset instance."""

from __future__ import annotations

import os

from pathlib import Path


BUNDLE_DIR = Path(__file__).resolve().parent
os.environ['AIDATA_CASES_ROOT'] = str(BUNDLE_DIR / 'cases')
os.environ['AIDATA_CASE_IDS'] = (BUNDLE_DIR / 'case_id.txt').read_text(encoding='utf-8').strip()

from nexus.extensions.tasks.aidata.generator import generate_configs, get_batch_metadata, get_total_count  # noqa: E402,F401
'''


def _safe_relative_path(raw: Any, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f'{field} must be a non-empty relative path string')

    rel = PurePosixPath(raw)
    if rel.is_absolute() or any(part in ('', '.', '..') for part in rel.parts):
        raise ValueError(f'unsafe {field}: {raw!r}')
    return Path(*rel.parts)


def _safe_join(root: Path, raw: Any, *, field: str) -> Path:
    root_resolved = root.resolve()
    out = (root / _safe_relative_path(raw, field=field)).resolve()
    if not out.is_relative_to(root_resolved):
        raise ValueError(f'unsafe {field}: {raw!r}')
    return out


def _decode_file_content(file_record: dict[str, Any]) -> bytes:
    encoding = (file_record.get('encoding') or 'utf-8').lower().replace('_', '-')
    content = file_record.get('content', '')

    if encoding in ('utf-8', 'utf8'):
        if not isinstance(content, str):
            raise ValueError(f'{file_record.get("path")}: utf-8 content must be a string')
        return content.encode('utf-8')

    if encoding == 'base64':
        if not isinstance(content, str):
            raise ValueError(f'{file_record.get("path")}: base64 content must be a string')
        return base64.b64decode(content.encode('ascii'), validate=True)

    raise ValueError(f'{file_record.get("path")}: unsupported encoding {encoding!r}')


def _chmod_if_requested(path: Path, mode: Any) -> None:
    if mode in (None, ''):
        return
    try:
        numeric_mode = mode if isinstance(mode, int) else int(str(mode), 8)
        path.chmod(numeric_mode)
    except (OSError, ValueError) as exc:
        logger.warning('aidata dataset adapter: failed to chmod %s to %r: %s', path, mode, exc)


def _case_id_from_record(record: dict[str, Any]) -> str:
    case_id = record.get('case_id') or record.get('instance_id')
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError('instance must contain case_id or instance_id')
    return case_id.strip()


def _instance_id_from_record(record: dict[str, Any]) -> str:
    instance_id = record.get('instance_id') or record.get('case_id')
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError('instance must contain instance_id or case_id')
    return instance_id.strip()


def _write_instance_json(record: dict[str, Any], instance_path: Path) -> None:
    instance_path.parent.mkdir(parents=True, exist_ok=True)
    instance_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def materialize_instance_record(
    record: dict[str, Any],
    output_dir: str | Path,
    *,
    instance_path: str | Path | None = None,
    overwrite: bool = True,
) -> MaterializedCase:
    """把一条数据集实例恢复到 ``<output_dir>/cases/<case_id>`` 。"""

    if not isinstance(record, dict):
        raise ValueError('instance record must be a JSON object')

    files = record.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('instance record must contain a non-empty files[] list')

    out_dir = Path(output_dir)
    cases_root = out_dir / 'cases'
    case_id = _case_id_from_record(record)
    instance_id = _instance_id_from_record(record)
    case_dir = _safe_join(cases_root, case_id, field='case_id')

    if overwrite and case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    for file_record in files:
        if not isinstance(file_record, dict):
            raise ValueError('files[] entries must be JSON objects')
        out_file = _safe_join(case_dir, file_record.get('path'), field='files[].path')
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(_decode_file_content(file_record))
        _chmod_if_requested(out_file, file_record.get('mode'))

    raw_instance_path = Path(instance_path) if instance_path is not None else out_dir / 'instance.json'
    _write_instance_json(record, raw_instance_path)

    return MaterializedCase(
        instance_id=instance_id,
        case_id=case_id,
        output_dir=out_dir,
        cases_root=cases_root,
        case_dir=case_dir,
        instance_path=raw_instance_path,
    )


def materialize_instance_file(
    instance_json: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = True,
) -> MaterializedCase:
    """把一个本地 ``instance.json`` 恢复到 ``<output_dir>/cases`` 。"""

    src = Path(instance_json)
    record = json.loads(src.read_text(encoding='utf-8'))
    return materialize_instance_record(
        record,
        output_dir,
        instance_path=Path(output_dir) / 'instance.json',
        overwrite=overwrite,
    )


def _write_case_id_file(case_id: str, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(case_id + '\n', encoding='utf-8')
    return out


def write_run_generator(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_RUN_GENERATOR, encoding='utf-8')
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Materialize a Nexus Dataset AIDATA instance.json into a runnable cases/ tree.',
    )
    parser.add_argument('instance_json', help='Path to the instance.json produced by "nexus dataset get".')
    parser.add_argument('output_dir', help='Directory that will receive instance.json, case_id.txt, and cases/.')
    parser.add_argument(
        '--case-id-file',
        default=None,
        help='Where to write the restored case_id. Defaults to <output_dir>/case_id.txt.',
    )
    parser.add_argument(
        '--no-case-id-file',
        action='store_true',
        help='Do not write a case_id file.',
    )
    parser.add_argument(
        '--generator-file',
        default=None,
        help='Where to write the run-batch generator shim. Defaults to <output_dir>/generator.py.',
    )
    parser.add_argument(
        '--no-generator-file',
        action='store_true',
        help='Do not write a run-batch generator shim.',
    )
    parser.add_argument(
        '--no-overwrite',
        action='store_true',
        help='Do not remove an existing restored case directory before materializing.',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only print the restored case_id.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    restored = materialize_instance_file(
        args.instance_json,
        args.output_dir,
        overwrite=not args.no_overwrite,
    )

    case_id_file: Path | None = None
    if not args.no_case_id_file:
        case_id_file = _write_case_id_file(
            restored.case_id,
            args.case_id_file or (restored.output_dir / 'case_id.txt'),
        )

    generator_file: Path | None = None
    if not args.no_generator_file:
        generator_file = write_run_generator(args.generator_file or (restored.output_dir / 'generator.py'))

    if args.quiet:
        print(restored.case_id)
    else:
        print(f'case_id={restored.case_id}')
        print(f'cases_root={restored.cases_root}')
        if case_id_file is not None:
            print(f'case_id_file={case_id_file}')
        if generator_file is not None:
            print(f'generator_file={generator_file}')
    return 0


if __name__ == '__main__':
    sys.exit(main())


__all__ = [
    'MaterializedCase',
    'build_arg_parser',
    'main',
    'materialize_instance_file',
    'materialize_instance_record',
    'write_run_generator',
]
