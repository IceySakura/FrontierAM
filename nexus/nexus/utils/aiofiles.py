import json

from typing import TypeVar

import aiofiles

from pydantic import BaseModel, ValidationError


ModelT = TypeVar('ModelT', bound=BaseModel)


async def read_file(path: str) -> str:
    """
    Read file content asynchronously.

    Args:
        path: The file path to read

    Returns:
        The file content as string
    """
    async with aiofiles.open(path, encoding='utf-8', errors='replace') as f:
        return await f.read()


async def read_file_or_empty(path: str) -> str:
    """
    Read file content asynchronously, returning an empty string if absent.

    Args:
        path: The file path to read

    Returns:
        The file content as string, or '' when the file does not exist.
    """
    try:
        return await read_file(path)
    except FileNotFoundError:
        return ''


async def read_json_model_or_none(path: str, model: type[ModelT]) -> ModelT | None:
    """
    Read a JSON file and validate it as a Pydantic model.

    Args:
        path: The JSON file path to read
        model: The Pydantic model class to validate against

    Returns:
        A validated model instance, or None when the file is missing, malformed,
        or fails Pydantic validation.
    """
    try:
        return model.model_validate(json.loads(await read_file(path)))
    except (FileNotFoundError, json.JSONDecodeError, ValidationError):
        return None


async def write_file(path: str, content: str) -> None:
    """
    Write content to file asynchronously.

    Args:
        path: The file path to write
        content: The content to write
    """
    async with aiofiles.open(path, mode='w') as f:
        await f.write(content)
