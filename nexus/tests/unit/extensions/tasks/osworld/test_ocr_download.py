"""OSWorld ocr download regressions."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
import zipfile

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nexus.extensions.tasks.os_world.evaluation import ocr
from nexus.extensions.tasks.os_world.evaluation.errors import EvaluatorInfrastructureError


@pytest.fixture
def model():
    return {
        'filename': 'test.pth',
        'url': 'https://unused.invalid/model.zip',
        'md5sum': hashlib.md5(b'model').hexdigest(),
    }


def test_cached_model_never_downloads(monkeypatch, tmp_path, model):
    (tmp_path / model['filename']).write_bytes(b'model')
    download = Mock(side_effect=AssertionError('must not download'))
    monkeypatch.setattr(ocr.subprocess, 'run', download)
    ocr._ensure_model(model, tmp_path)
    download.assert_not_called()


def test_retry_then_publish_validated_model(monkeypatch, tmp_path, model):
    calls = []
    (tmp_path / model['filename']).write_bytes(b'corrupt')

    def run(command, **kwargs):
        calls.append(kwargs)
        assert (tmp_path / model['filename']).read_bytes() == b'corrupt'
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        (Path(command[-1]) / 'model.pth').write_bytes(b'model')

    monkeypatch.setattr(ocr.subprocess, 'run', run)
    monkeypatch.setattr(ocr, 'RETRY_DELAY', 0)
    ocr._ensure_model(model, tmp_path)
    assert len(calls) == 2
    assert calls[0]['timeout'] == 600
    assert (tmp_path / model['filename']).read_bytes() == b'model'
    assert not list(tmp_path.glob('.ocr-download-*'))


@pytest.mark.parametrize('failure', ['timeout', 'exit', 'checksum'])
def test_exhausted_retries_are_infrastructure_errors(monkeypatch, tmp_path, model, failure):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        (Path(command[-1]) / 'model.pth').write_bytes(b'partial')
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        if failure == 'exit':
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(ocr.subprocess, 'run', run)
    monkeypatch.setattr(ocr, 'RETRY_DELAY', 0)
    with pytest.raises(EvaluatorInfrastructureError, match='3 attempts, 600s'):
        ocr._ensure_model(model, tmp_path)
    assert len(calls) == 3
    assert not (tmp_path / model['filename']).exists()
    assert not list(tmp_path.glob('.ocr-download-*'))


def test_concurrent_cases_share_one_download(monkeypatch, tmp_path, model):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        entered.set()
        assert release.wait(5)
        (Path(command[-1]) / 'model.pth').write_bytes(b'model')

    monkeypatch.setattr(ocr.subprocess, 'run', run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(ocr._ensure_model, model, tmp_path)
        try:
            assert entered.wait(5)
            second = pool.submit(ocr._ensure_model, model, tmp_path)
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert len(calls) == 1


def test_real_subprocess_download_and_checksum(tmp_path, model):
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr(model['filename'], b'model')
    model['url'] = archive.as_uri()
    ocr._ensure_model(model, tmp_path / 'cache')
    assert (tmp_path / 'cache' / model['filename']).read_bytes() == b'model'


def test_real_timeout_kills_downloader_and_removes_partial_files(monkeypatch, tmp_path, model):
    pidfile = tmp_path / 'pid'
    monkeypatch.setattr(
        ocr,
        '_DOWNLOAD_SCRIPT',
        f'import os,time,pathlib; pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)',
    )
    monkeypatch.setattr(ocr, 'DOWNLOAD_TIMEOUT', 0.5)
    monkeypatch.setattr(ocr, 'DOWNLOAD_ATTEMPTS', 1)
    started = time.monotonic()
    with pytest.raises(EvaluatorInfrastructureError, match='TimeoutExpired'):
        ocr._ensure_model(model, tmp_path / 'cache')
    assert time.monotonic() - started < 5
    assert pidfile.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
    assert not list((tmp_path / 'cache').glob('.ocr-download-*'))


def test_reader_keeps_defaults_but_disables_unbounded_download(monkeypatch, tmp_path, model):
    # Import no real torch/easyocr: exercise the model selection contract separately.
    monkeypatch.setitem(
        sys.modules,
        'easyocr.config',
        SimpleNamespace(
            MODULE_PATH=str(tmp_path),
            detection_models={'craft': model},
            recognition_models={'gen2': {'english_g2': {**model, 'filename': 'english.pth'}}},
        ),
    )
    prepare = Mock()
    monkeypatch.setattr(ocr, '_ensure_model', prepare)
    reader = Mock(return_value='reader')
    assert ocr.create_english_reader(SimpleNamespace(Reader=reader)) == 'reader'
    assert prepare.call_count == 2
    reader.assert_called_once_with(['en'], model_storage_directory=str(tmp_path / 'model'), download_enabled=False)


@pytest.mark.asyncio
async def test_download_failure_is_not_swallowed_as_zero(monkeypatch, tmp_path):
    from nexus.extensions.tasks.os_world.evaluation.evaluator import _call_metric
    from nexus.extensions.tasks.os_world.metrics.official import docs

    image = tmp_path / 'test.png'
    image.write_bytes(b'image')
    monkeypatch.setattr(docs, '_image', lambda: object())
    monkeypatch.setattr(docs, '_easyocr', lambda: object())
    monkeypatch.setattr(docs.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='PNG image'))
    monkeypatch.setattr(
        docs, 'create_english_reader', Mock(side_effect=EvaluatorInfrastructureError('OCR model download failed'))
    )
    with pytest.raises(EvaluatorInfrastructureError, match='OCR model download failed'):
        await _call_metric(
            'compare_image_text', docs.compare_image_text, str(image), None, {'rule': {'type': 'text', 'text': 'hello'}}
        )


@pytest.mark.parametrize(('text', 'score'), [('hello world', 1), ('other', 0)])
def test_ocr_scoring_unchanged(monkeypatch, tmp_path, text, score):
    from nexus.extensions.tasks.os_world.metrics.official import docs

    image = tmp_path / 'test.png'
    image.write_bytes(b'image')
    monkeypatch.setattr(docs, '_image', lambda: object())
    monkeypatch.setattr(docs, '_easyocr', lambda: object())
    monkeypatch.setattr(docs.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='PNG image'))
    monkeypatch.setattr(
        docs, 'create_english_reader', lambda _: SimpleNamespace(readtext=lambda path: [(None, text, 1.0)])
    )
    assert docs.compare_image_text(str(image), {'type': 'text', 'text': 'hello'}) == score
