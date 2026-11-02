"""Source loading utilities for stem_mas math benchmarks."""

from __future__ import annotations

import importlib
import json

from pathlib import Path
from typing import Any
from urllib.request import urlopen


class MathSourceLoader:
    """Load normalized row dictionaries without applying benchmark semantics."""

    @staticmethod
    def load_url_records(source_url: str) -> list[dict[str, Any]]:
        with urlopen(source_url, timeout=60) as response:
            text = response.read().decode('utf-8').strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON from {source_url}: {exc}') from exc
        return MathSourceLoader.require_json_array(source_url, loaded)

    @classmethod
    def load_hf_file_records(
        cls,
        repo_id: str,
        filename: str,
        revision: str | None,
    ) -> list[dict[str, Any]]:
        try:
            hf_hub_download = importlib.import_module('huggingface_hub').hf_hub_download
        except ImportError as exc:
            raise ImportError('Loading Hugging Face files requires `huggingface_hub`.') from exc

        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type='dataset',
                revision=revision,
            )
        )
        return cls.load_source_path(path)

    @classmethod
    def load_source_records(cls, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f'Math dataset path must be a JSON or JSONL file: {path}')
        records = cls.load_source_path(path)
        if not records:
            raise FileNotFoundError(f'No dataset records found in {path}.')
        return records

    @classmethod
    def load_source_path(cls, path: Path) -> list[dict[str, Any]]:
        if path.suffix == '.jsonl':
            return cls.load_jsonl_text(str(path), path.read_text(encoding='utf-8'))
        if path.suffix == '.json':
            return cls.load_json_array(path)
        raise ValueError(f'Unsupported math dataset file: {path}')

    @staticmethod
    def load_json_array(path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding='utf-8').strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON in {path}: {exc}') from exc
        return MathSourceLoader.require_json_array(str(path), loaded)

    @staticmethod
    def load_jsonl_text(path: str, text: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL line {line_number} in {path}: {exc}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Invalid JSONL line {line_number} in {path}: expected object.')
            records.append(row)
        return records

    @staticmethod
    def require_json_array(path: str, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            raise ValueError(f'Invalid JSON payload in {path}: expected an array.')
        records: list[dict[str, Any]] = []
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f'Invalid JSON item {index} in {path}: expected object.')
            records.append(row)
        return records


__all__ = ['MathSourceLoader']
