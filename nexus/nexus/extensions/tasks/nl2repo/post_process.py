"""推理后处理函数 — 通过 invoke 在 Runtime 中执行。

包含：
- post_process_and_zip: 删除包管理文件、测试文件，并打包 workspace
- extract_zip: 解压 zip 文件到指定目录（不依赖系统 unzip 命令）
- PACKAGE_FILES: 需要删除的包管理文件列表常量
"""

# 注意：PACKAGE_FILES 常量在 Host 端使用，传参给 invoke。
# post_process_and_zip 函数在 Runtime 端执行，所有 import 必须在函数体内完成。

PACKAGE_FILES: list[str] = [
    'setup.py',
    'pyproject.toml',
    'setup.cfg',
    'requirements.txt',
    'requirements-dev.txt',
    'requirements-test.txt',
    'tox.ini',
    'pytest.ini',
    'poetry.lock',
    'Pipfile',
    'Pipfile.lock',
    'environment.yml',
    'conda-env.yaml',
    'manifest.in',
    'MANIFEST.in',
]


def post_process_and_zip(
    workspace_dir: str,
    test_files_list: list[str],
    package_files_list: list[str],
) -> str:
    """在 Runtime 中执行后处理：删除包管理文件、测试文件，并打包 workspace。

    该函数通过 invoke 在 infer_runtime 中执行，
    因此所有 import 必须在函数体内完成。

    Args:
        workspace_dir: workspace 目录路径（如 /workspace）
        test_files_list: 需要删除的测试文件/目录列表
        package_files_list: 需要删除的包管理文件名列表

    Returns:
        zip 文件路径（如 /tmp/workspace.zip）
    """
    import os
    import shutil
    import zipfile

    # 1. 删除包管理文件（递归查找 workspace 中所有匹配的文件）
    package_files_set = set(package_files_list)
    removed_package = 0
    for root, dirs, files in os.walk(workspace_dir):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        for fname in files:
            if fname in package_files_set:
                fpath = os.path.join(root, fname)
                try:
                    os.remove(fpath)
                    removed_package += 1
                except OSError:
                    pass

    # 2. 删除测试文件/目录
    removed_test = 0
    for test_item in test_files_list:
        target_path = os.path.join(workspace_dir, test_item)
        try:
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)
                removed_test += 1
            elif os.path.isfile(target_path):
                os.remove(target_path)
                removed_test += 1
        except OSError:
            pass  # 文件不存在时静默跳过

    # 3. 使用 Python zipfile 打包 workspace（不依赖系统 zip 命令）
    zip_path = '/tmp/workspace.zip'
    # 先删除可能存在的旧 zip 文件
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(workspace_dir):
            dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
            for fname in files:
                abs_path = os.path.join(root, fname)
                if os.path.islink(abs_path):
                    continue
                try:
                    arc_name = os.path.relpath(abs_path, workspace_dir)
                    zf.write(abs_path, arc_name)
                except (OSError, FileNotFoundError):
                    pass

    return zip_path


def extract_zip(zip_path: str, target_dir: str) -> str:
    """在 Runtime 中解压 zip 文件到指定目录（不依赖系统 unzip 命令）。

    该函数通过 invoke 在 eval_runtime 中执行，
    因此所有 import 必须在函数体内完成。

    Args:
        zip_path: zip 文件路径（如 /tmp/workspace.zip）
        target_dir: 解压目标目录（如 /workspace）

    Returns:
        解压目标目录路径
    """
    import os
    import zipfile

    os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)

    return target_dir


__all__ = ['PACKAGE_FILES', 'extract_zip', 'post_process_and_zip']
