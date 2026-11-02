"""Task-local, secret-free evidence for the visual evaluator."""

import hashlib
import json
import logging
import time

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def record_evaluation(task_id, cache):
    started = time.monotonic()
    record = {'task_id': task_id}
    try:
        yield record
        record['ok'] = True
    except BaseException as exc:
        record.update(ok=False, error_type=type(exc).__name__)
        raise
    finally:
        if task_id == '078':
            record['elapsed_s'] = round(time.monotonic() - started, 3)
            record['images'] = {}
            for name in ('task078_interact_default.png', 'task078_interact_rotated.png', 'task078_actual_default.png'):
                path = Path(cache) / name
                if path.is_file():
                    record['images'][name] = {
                        'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
            images = record['images']
            first, second = images.get('task078_interact_default.png'), images.get('task078_interact_rotated.png')
            record['interaction_images_identical'] = first['sha256'] == second['sha256'] if first and second else None
            try:
                with (Path(cache) / 'evaluation-diagnostics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
            except (OSError, ValueError):
                logging.getLogger(__name__).warning('Cannot save V2 evaluator diagnostics')
