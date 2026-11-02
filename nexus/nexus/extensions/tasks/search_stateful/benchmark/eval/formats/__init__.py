"""Dataset format specs and loaders for benchmark evaluation."""

from __future__ import annotations

import importlib
import json
import os

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, model_validator


QUESTION_KEYS = (
    'origin_question',
    'question',
    'prompt',
    'input',
    'query',
)
ANSWER_KEYS = (
    'ref_answer',
    'answer',
    'output',
    'response',
    'completion',
)


class BaseRecord(BaseModel):
    question: str
    answer: str
    category: str | None = None
    raw_subject: str | None = None
    answer_type: str | None = None
    id: str | None = None
    file_names: list[str] | None = None
    file_paths: list[str] | None = None
    author_name: str | None = None
    canary: str | None = None

    model_config = {'extra': 'allow'}

    @model_validator(mode='before')
    @classmethod
    def _normalize(cls, data):
        if not isinstance(data, dict):
            raise ValueError('Invalid record: expected object.')

        question = None
        for key in QUESTION_KEYS:
            value = data.get(key)
            if value is not None:
                question = value
                break

        answer = None
        for key in ANSWER_KEYS:
            value = data.get(key)
            if value is not None:
                answer = value
                break

        if question is None or answer is None:
            raise ValueError(
                'Missing question/answer in dataset record. '
                f'Question keys: {", ".join(QUESTION_KEYS)}; '
                f'answer keys: {", ".join(ANSWER_KEYS)}.'
            )

        normalized = dict(data)
        normalized.setdefault('question', question)
        normalized.setdefault('answer', answer)

        subject = data.get('subject') or data.get('category')
        if subject:
            normalized.setdefault('raw_subject', subject)
            normalized.setdefault('category', subject)

        return normalized


class HLERecord(BaseRecord):
    """Native HLE schema (parquet)."""


class NaiveRecord(BaseRecord):
    """Minimal JSONL schema (question/answer aliases)."""


class Format(ABC):
    name: str
    model: type[BaseRecord]

    @abstractmethod
    def load(self, path: Path) -> list[BaseRecord]:
        raise NotImplementedError

    def _parse(self, records: list[dict]) -> list[BaseRecord]:
        return [self.model.model_validate(record) for record in records]


class HLEFormat(Format):
    name = 'hle'
    model = HLERecord

    def load(self, path: Path) -> list[BaseRecord]:
        paths: list[Path]
        if path.is_dir():
            data_dir = path / 'data'
            paths = sorted(data_dir.glob('test-*.parquet')) if data_dir.exists() else []
            if not paths:
                paths = sorted(path.glob('*.parquet'))
        else:
            paths = [path]

        if not paths:
            raise FileNotFoundError(f'No parquet files found in {path}.')

        return self._parse(_load_parquet(paths))


class NaiveFormat(Format):
    name = 'naive'
    model = NaiveRecord

    def load(self, path: Path) -> list[BaseRecord]:
        paths = sorted(path.glob('*.jsonl')) if path.is_dir() else [path]
        if not paths:
            raise FileNotFoundError(f'No jsonl files found in {path}.')
        return self._parse(_load_jsonl(paths))


FORMATS = {
    'hle': HLEFormat(),
    'naive': NaiveFormat(),
}


def load_records(path: str | Path, format_name: str | None = None) -> list[BaseRecord]:
    fmt = (format_name or os.getenv('HLE_DATASET_FORMAT') or 'hle').lower()
    if fmt not in FORMATS:
        raise ValueError(f'Unsupported dataset format: {fmt}. Use one of: {", ".join(FORMATS)}.')
    return FORMATS[fmt].load(Path(path))


def _load_parquet(paths: list[Path]) -> list[dict]:
    pd = importlib.import_module('pandas')

    frames = []
    for path in paths:
        frames.append(pd.read_parquet(path))
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    return df.to_dict(orient='records')


def _load_jsonl(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open('r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSONL line {idx + 1} in {path}: {exc}') from exc
                if not isinstance(data, dict):
                    raise ValueError(f'Invalid JSONL line {idx + 1} in {path}: expected object.')
                records.append(data)
    return records


__all__ = [
    'BaseRecord',
    'HLERecord',
    'NaiveRecord',
    'load_records',
]
