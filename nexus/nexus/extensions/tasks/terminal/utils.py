import base64

from pathlib import Path

from nexus.runtime.executors import FileItem


__all__ = ['read_local_directory_to_file_items', 'read_local_file_to_file_item']


def read_local_file_to_file_item(
    local_file: Path,
    container_path: str,
) -> FileItem:
    """读取本地单个文件，返回 FileItem。

    先尝试以 UTF-8 文本方式读取，如果失败则以 base64 编码读取二进制内容。

    Args:
        local_file: 本地文件路径
        container_path: 容器内的目标路径

    Returns:
        FileItem 对象
    """
    if not local_file.exists():
        raise FileNotFoundError(f'Local file not found: {local_file}')

    if not local_file.is_file():
        raise ValueError(f'Path is not a file: {local_file}')

    # 先尝试以 UTF-8 文本方式读取
    try:
        content = local_file.read_text(encoding='utf-8')
        encoding = 'utf8'
    except UnicodeDecodeError:
        # 如果是二进制文件，使用 base64 编码
        content_bytes = local_file.read_bytes()
        content = base64.b64encode(content_bytes).decode('utf-8')
        encoding = 'base64'

    return FileItem(path=container_path, content=content, encoding=encoding)


def read_local_directory_to_file_items(
    local_dir: Path,
    container_base_path: str,
) -> list[FileItem]:
    """递归读取本地文件夹，返回 FileItem 列表用于上传到容器。

    Args:
        local_dir: 本地文件夹路径
        container_base_path: 容器内的目标基础路径

    Returns:
        FileItem 列表，保持原有的目录结构
    """
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
