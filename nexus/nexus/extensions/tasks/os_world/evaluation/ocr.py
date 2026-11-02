"""Bound EasyOCR model downloads without changing Reader or scoring defaults."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import time

from pathlib import Path

from filelock import FileLock, Timeout

from .errors import EvaluatorInfrastructureError


logger = logging.getLogger(__name__)
DOWNLOAD_TIMEOUT = 600.0
DOWNLOAD_ATTEMPTS = 3
RETRY_DELAY = 2.0

# A subprocess gives DNS, connect, read and unzip a real wall-clock deadline.
# Cancelling a thread/urlretrieve future would leave its download running.
_DOWNLOAD_SCRIPT = """
import pathlib, sys, urllib.request, zipfile
url, filename, directory = sys.argv[1:]
root = pathlib.Path(directory)
urllib.request.urlretrieve(url, root / 'model.zip')
with zipfile.ZipFile(root / 'model.zip') as archive:
    with archive.open(filename) as source, (root / 'model.pth').open('wb') as dest:
        import shutil
        shutil.copyfileobj(source, dest)
"""


def _valid_model(path: Path, expected_md5: str) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == expected_md5


def _ensure_model(model: dict, directory: Path) -> None:
    filename = model['filename']
    target = directory / filename
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if _valid_model(target, model['md5sum']):
            return
        # Concurrent cases share only a fully validated, atomically published file.
        lock_timeout = DOWNLOAD_ATTEMPTS * DOWNLOAD_TIMEOUT + RETRY_DELAY * (DOWNLOAD_ATTEMPTS - 1) + 10
        with FileLock(str(target) + '.lock', timeout=lock_timeout):
            if _valid_model(target, model['md5sum']):
                return
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                logger.info(
                    'OSWorld OCR model download: file=%s attempt=%d/%d deadline=%.1fs',
                    filename,
                    attempt,
                    DOWNLOAD_ATTEMPTS,
                    DOWNLOAD_TIMEOUT,
                )
                try:
                    with tempfile.TemporaryDirectory(prefix='.ocr-download-', dir=directory) as temp:
                        # No model/agent subprocess is restarted. Only this read-only download is retried.
                        subprocess.run(
                            [sys.executable, '-I', '-S', '-c', _DOWNLOAD_SCRIPT, model['url'], filename, temp],
                            timeout=DOWNLOAD_TIMEOUT,
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        downloaded = Path(temp) / 'model.pth'
                        if not _valid_model(downloaded, model['md5sum']):
                            raise ValueError('OCR model checksum mismatch')
                        os.replace(downloaded, target)
                    logger.info('OSWorld OCR model download completed: file=%s attempt=%d', filename, attempt)
                    return
                except (subprocess.SubprocessError, OSError, ValueError) as exc:
                    logger.warning(
                        'OSWorld OCR model download failed: file=%s attempt=%d/%d error=%s',
                        filename,
                        attempt,
                        DOWNLOAD_ATTEMPTS,
                        type(exc).__name__,
                    )
                    if attempt == DOWNLOAD_ATTEMPTS:
                        raise EvaluatorInfrastructureError(
                            f'OSWorld OCR model download failed: {filename}; '
                            f'{DOWNLOAD_ATTEMPTS} attempts, {DOWNLOAD_TIMEOUT:g}s deadline per attempt '
                            f'({type(exc).__name__})'
                        ) from exc
                    time.sleep(RETRY_DELAY)
    except (OSError, Timeout) as exc:
        raise EvaluatorInfrastructureError(
            f'OSWorld OCR model cache unavailable: {filename} ({type(exc).__name__})'
        ) from exc


def create_english_reader(easyocr):
    """Prepare exactly the models used by Reader(['en']), then forbid implicit downloads."""
    from easyocr.config import MODULE_PATH, detection_models, recognition_models

    directory = Path(MODULE_PATH) / 'model'
    _ensure_model(detection_models['craft'], directory)
    _ensure_model(recognition_models['gen2']['english_g2'], directory)
    try:
        return easyocr.Reader(['en'], model_storage_directory=str(directory), download_enabled=False)
    except FileNotFoundError as exc:
        # A deleted/corrupted cache after preparation is infrastructure failure too.
        raise EvaluatorInfrastructureError('OSWorld OCR model cache changed before Reader initialization') from exc
