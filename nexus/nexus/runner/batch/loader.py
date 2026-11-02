"""Config generator loader for batch runner."""

from __future__ import annotations

import importlib.util
import sys

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nexus.runner.batch.batch_config import BatchItem


class GeneratorLoader:
    """Loads config generator module dynamically."""

    def __init__(self, generator_path: Path):
        """Initialize loader with generator file path.

        Args:
            generator_path: Path to the Python file containing generator functions
        """
        self.path = generator_path
        self.module = self._load_module()

    def _load_module(self) -> Any:
        """Dynamically load Python module from file path."""
        if not self.path.exists():
            raise FileNotFoundError(f'Generator file not found: {self.path}')

        spec = importlib.util.spec_from_file_location('config_generator', self.path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load module from {self.path}')

        module = importlib.util.module_from_spec(spec)
        sys.modules['config_generator'] = module
        spec.loader.exec_module(module)
        return module

    def get_generate_fn(self) -> Callable:
        """Get the generate_configs function from module.

        Returns:
            The generate_configs function

        Raises:
            AttributeError: If generate_configs is not defined in module
        """
        if not hasattr(self.module, 'generate_configs'):
            raise AttributeError(f'Module {self.path} must define generate_configs() function')
        return self.module.generate_configs

    def get_total_count(self) -> int | None:
        """Get total count of configs (optional).

        Returns:
            Total count if get_total_count() is defined, None otherwise
        """
        if hasattr(self.module, 'get_total_count'):
            return self.module.get_total_count()
        return None

    def get_batch_metadata(self) -> dict[str, Any]:
        """Get batch metadata (optional).

        Returns:
            Batch metadata dict if get_batch_metadata() is defined, empty dict otherwise
        """
        if hasattr(self.module, 'get_batch_metadata'):
            return self.module.get_batch_metadata()
        return {}

    def generate(self) -> Iterator[BatchItem]:
        """Generate batch items as iterator.

        Automatically handles both list and iterator return types.

        Returns:
            Iterator of BatchItem instances
        """
        generate_fn = self.get_generate_fn()
        result = generate_fn()

        if isinstance(result, list):
            return iter(result)
        return result


def serialize_processor_result(result: Any) -> Any:
    """Serialize processor result to JSON-compatible format.

    Args:
        result: Processor result (could be dict, list, or pydantic model)

    Returns:
        JSON-compatible representation
    """
    if isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, dict):
        return {k: serialize_processor_result(v) for k, v in result.items()}
    if isinstance(result, list):
        return [serialize_processor_result(v) for v in result]
    return result


__all__ = ['GeneratorLoader', 'serialize_processor_result']
