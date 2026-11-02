import base64

from pathlib import Path

from nexus.runtime.executors import FileItem


def read_local_file_to_file_item(
    local_file: Path,
    container_path: str,
) -> FileItem:
    if not local_file.exists():
        raise FileNotFoundError(f'Local file not found: {local_file}')

    if not local_file.is_file():
        raise ValueError(f'Path is not a file: {local_file}')

    try:
        content = local_file.read_text(encoding='utf-8')
        encoding = 'utf8'
    except (UnicodeDecodeError, OSError):
        content_bytes = local_file.read_bytes()
        content = base64.b64encode(content_bytes).decode('utf-8')
        encoding = 'base64'

    return FileItem(path=container_path, content=content, encoding=encoding)


def read_local_directory_to_file_items(
    local_dir: Path,
    container_base_path: str,
) -> list[FileItem]:
    file_items = []

    if not local_dir.exists():
        raise FileNotFoundError(f'Local directory not found: {local_dir}')

    if not local_dir.is_dir():
        raise NotADirectoryError(f'Path is not a directory: {local_dir}')

    for file_path in local_dir.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_dir)
            container_path = f'{container_base_path.rstrip("/")}/{relative_path.as_posix()}'
            file_items.append(read_local_file_to_file_item(file_path, container_path))

    return file_items
