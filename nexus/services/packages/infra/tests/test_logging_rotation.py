"""日志滚动落盘测试：stdout 与文件双写、按小时切分、失败降级。"""

from __future__ import annotations

import json
import logging
import time

from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytest

from infra.logging import configure_logging


@pytest.fixture(autouse=True)
def _restore_root_logger() -> object:
    """用例间隔离 root logger，避免 handler 互相污染。"""
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        handler.close()
    root.handlers = saved
    root.setLevel(saved_level)


def _file_handler() -> TimedRotatingFileHandler | None:
    for handler in logging.getLogger().handlers:
        if isinstance(handler, TimedRotatingFileHandler):
            return handler
    return None


def test_no_file_handler_without_log_file(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging('svc')
    logging.getLogger('t').info('only stdout')

    assert 'only stdout' in capsys.readouterr().out
    assert _file_handler() is None


def test_writes_to_both_stdout_and_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """双写：stdout 保证 kubectl logs 可用，文件供进容器排查。"""
    log_file = tmp_path / 'app.log'
    configure_logging('svc', log_file=log_file)
    logging.getLogger('t').info('dual write')

    assert 'dual write' in capsys.readouterr().out
    content = log_file.read_text(encoding='utf-8')
    assert 'dual write' in content
    # 文件内同为 JSON 格式，便于机器解析
    assert json.loads(content.strip())['message'] == 'dual write'


def test_creates_parent_directory(tmp_path: Path) -> None:
    log_file = tmp_path / 'nested' / 'deep' / 'app.log'
    configure_logging('svc', log_file=log_file)
    logging.getLogger('t').info('mkdir')

    assert log_file.exists()


def test_rotates_hourly(tmp_path: Path) -> None:
    """按小时滚动：when='H'、interval=3600 秒。"""
    configure_logging('svc', log_file=tmp_path / 'app.log')

    handler = _file_handler()
    assert handler is not None
    assert handler.when == 'H'
    assert handler.interval == 3600


def test_rotated_suffix_is_hourly(tmp_path: Path) -> None:
    """切分文件后缀精确到小时，便于按时间点定位问题。"""
    log_file = tmp_path / 'app.log'
    configure_logging('svc', log_file=log_file)

    handler = _file_handler()
    assert handler is not None
    assert handler.suffix == '%Y-%m-%d_%H'

    logging.getLogger('t').info('before rotate')
    # 把翻转时间点提前到过去，下一条日志即触发切分
    handler.rolloverAt = int(time.time()) - 1
    logging.getLogger('t').info('after rotate')

    rotated = [p.name for p in tmp_path.glob('app.log.*')]
    assert rotated, '应产生带时间后缀的切分文件'
    # 后缀形如 2026-07-28_15，共 13 个字符
    suffix = rotated[0].removeprefix('app.log.')
    assert len(suffix) == 13, f'后缀应精确到小时: {suffix}'
    # 切分后新日志写入当前文件
    assert 'after rotate' in log_file.read_text(encoding='utf-8')


def test_backup_count_configurable(tmp_path: Path) -> None:
    """保留份数可配，用于按磁盘容量调整保留时长。"""
    configure_logging('svc', log_file=tmp_path / 'app.log', backup_count=24)

    handler = _file_handler()
    assert handler is not None
    assert handler.backupCount == 24


def test_default_backup_count_is_seven_days(tmp_path: Path) -> None:
    configure_logging('svc', log_file=tmp_path / 'app.log')

    handler = _file_handler()
    assert handler is not None
    assert handler.backupCount == 168


def test_falls_back_to_stdout_when_file_unwritable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """落盘失败不应阻断服务：stdout 仍可用，且有告警。"""
    blocker = tmp_path / 'blocker'
    blocker.write_text('not a directory', encoding='utf-8')
    # 把文件当目录用，触发 OSError
    configure_logging('svc', log_file=blocker / 'app.log')

    logging.getLogger('t').info('still works')
    captured = capsys.readouterr().out
    assert 'still works' in captured
    assert 'file logging disabled' in captured
    assert _file_handler() is None


def test_text_format_also_written_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / 'app.log'
    configure_logging('svc', json_output=False, log_file=log_file)
    logging.getLogger('t').info('plain text')

    content = log_file.read_text(encoding='utf-8')
    assert 'plain text' in content
    # 文本模式不是 JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(content.strip())
