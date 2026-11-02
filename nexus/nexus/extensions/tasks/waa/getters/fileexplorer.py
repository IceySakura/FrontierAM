"""File Explorer getters for WAA evaluation.

These getters check file/folder states on the Windows VM — existence, visibility,
sorting, permissions, etc.

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import io
import json
import logging
import os

from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.fileexplorer')


async def _check_ok(runtime: AGSRawWindowsAgentArenaRuntime, method, *args) -> bool:
    """Call a runtime method and return True on success (HTTP 200), False otherwise.

    The WAA server uses HTTP 200 to indicate a positive result (exists/true)
    and non-200 for negative (not exists/false). The runtime's request layer
    may raise on non-200 or return an error dict.
    """
    try:
        result = await method(*args)
        # If result is a dict with 'error' key, treat as failure
        return not (isinstance(result, dict) and 'error' in result)
    except Exception:
        return False


@getter_registry.register()
async def get_vm_folder_exists_in_documents(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a folder exists in the Documents directory."""
    documents_path_resp = await runtime.documents_path()
    documents_path = documents_path_resp['documents_path']
    folder_name = config['folder_name']
    return await _check_ok(runtime, runtime.folder_exists, folder_name, documents_path)


@getter_registry.register()
async def get_vm_file_exists_in_desktop(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a file exists on the Desktop."""
    desktop_path_resp = await runtime.desktop_path()
    desktop_path = desktop_path_resp['desktop_path']
    file_name = config['file_name']
    return await _check_ok(runtime, runtime.file_exists, file_name, desktop_path)


@getter_registry.register()
async def get_are_files_sorted_by_modified_time(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if files in a directory are sorted by modified time."""
    if not config['directory']:
        return False
    return await _check_ok(runtime, runtime.are_files_sorted_by_modified_time, config['directory'])


@getter_registry.register()
async def get_is_details_view(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if file explorer is in details view for the given folder."""
    if not config['folder_path']:
        return False
    return await _check_ok(runtime, runtime.is_details_view, config['folder_path'])


@getter_registry.register()
async def get_all_png_file_names(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if the text file lists all PNG files in the folder.

    Executes a Python script on the VM to compare the file content with actual PNG files.
    """
    file_path = config['file_path']
    folder_path = config['folder_path']

    script = f"""
import os
from sys import exit
def read_file(file_path):
    with open(file_path, 'r') as f:
        return set(f.read().splitlines())
def get_png_files(folder_path):
    return set([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])

if not os.path.exists(r'{file_path}'):
    print("File {file_path} does not exist")
    exit()

file_content = read_file(r'{file_path}')
png_files = get_png_files(r'{folder_path}')
print("File content:", repr(list(file_content)))
print("PNG files:", repr(list(png_files)))
print("Result:", "true" if file_content == png_files else "false")
"""
    response = await runtime.execute(['python', '-c', script], shell=False)

    if response.get('returncode') != 0:
        logger.error(f'Failed to execute script: {response}')
        return 'false'

    output_lines = response['output'].strip().split('\n')
    if output_lines[0] == f'File {file_path} does not exist':
        logger.error('Text file does not exist')
        return 'false'

    return output_lines[2].split(':', 1)[1].strip()


@getter_registry.register()
async def get_zipped_folder_in_desktop(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a password-protected 7z archive contains expected files."""
    import asyncio

    import py7zr

    folder_name = config['filename']
    zipped_file = await runtime.get_file(folder_name)

    def _check_7z(data: bytes, password: str) -> str:
        with py7zr.SevenZipFile(io.BytesIO(data), mode='r', password=password) as z:
            if not z.needs_password():
                return 'false'
            if 'OldProjects/example.txt' in z.getnames():
                return 'true'
        return 'false'

    return await asyncio.to_thread(_check_7z, zipped_file, config['password'])


@getter_registry.register()
async def get_is_all_docx_in_archive(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if all .docx files have been moved to an Archive subfolder."""
    folder_path = config['folder_path']
    if not folder_path:
        return 'false'

    # Check if folder exists
    folder_exists = await runtime.execute(
        ['python', '-c', f"import os; print(os.path.exists('{folder_path}'))"], shell=False
    )
    if folder_exists['output'].strip().lower() != 'true':
        return 'false'

    # Check if "Archive" folder exists
    archive_path = os.path.join(folder_path, 'Archive')
    archive_exists = await runtime.execute(
        ['python', '-c', f"import os; print(os.path.exists('{archive_path}'))"], shell=False
    )
    if archive_exists['output'].strip().lower() != 'true':
        return 'false'

    # Check if all files in Archive folder are .docx files
    list_archive = await runtime.execute(
        [
            'python',
            '-c',
            f"import os; print(','.join([f for f in os.listdir('{archive_path}') if os.path.isfile(os.path.join('{archive_path}', f))]))",
        ],
        shell=False,
    )
    archive_files = list_archive['output'].strip().split(',')
    if not all(file.lower().endswith('.docx') for file in archive_files):
        return 'false'

    # Check if no .docx files exist directly under folder_path
    list_folder = await runtime.execute(
        [
            'python',
            '-c',
            f"import os; print(','.join([f for f in os.listdir('{folder_path}') if os.path.isfile(os.path.join('{folder_path}', f))]))",
        ],
        shell=False,
    )
    folder_files = list_folder['output'].strip().split(',')
    if any(file.lower().endswith('.docx') for file in folder_files):
        return 'false'

    return 'true'


@getter_registry.register()
async def get_is_file_hidden(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a file is hidden (starts with '.' or has Windows hidden attribute)."""
    if not config or 'file_path' not in config:
        return 'false'
    file_path = config['file_path']
    command = f"import os; print(1 if os.path.basename(r'{file_path}').startswith('.') or (os.name == 'nt' and bool(os.stat(r'{file_path}').st_file_attributes & 2)) else 0)"
    response = await runtime.execute(['python', '-c', command], shell=False)
    if response.get('returncode') == 0:
        try:
            result = int(response['output'].strip())
            return 'true' if result == 1 else 'false'
        except ValueError:
            return 'false'
    return 'false'


@getter_registry.register()
async def get_vm_active_window_title(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get the title of the currently active window on the VM."""
    response = await runtime.execute(
        ['python', '-c', 'import pyautogui; print(pyautogui.getActiveWindowTitle())'], shell=False
    )
    return response['output'].strip()


@getter_registry.register()
async def get_is_directory_read_only_for_user(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a directory is read-only for a specific user."""
    if not config['directory']:
        raise ValueError('directory is required')
    if not config['user']:
        raise ValueError('user is required')
    return await _check_ok(runtime, runtime.is_directory_read_only_for_user, config['directory'], config['user'])


@getter_registry.register()
async def get_are_all_images_tagged(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if all images in a directory are tagged with a specific tag."""
    if not config['directory']:
        raise ValueError('directory is required')
    if not config['tag']:
        raise ValueError('tag is required')
    return await _check_ok(runtime, runtime.are_all_images_tagged, config['directory'], config['tag'])


@getter_registry.register()
async def get_is_file_desktop(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a file exists on the desktop."""
    desktop_path_resp = await runtime.desktop_path()
    desktop_path = desktop_path_resp['desktop_path']
    file_name = config['filename']
    file_content = await runtime.get_file(desktop_path + '\\' + file_name)
    if file_content is not None:
        return 'true'
    return 'false'


@getter_registry.register()
async def get_vm_library_folders(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Get the library folders for a given library name."""
    if 'library_name' not in config:
        raise ValueError('library_name is required')
    result = await runtime.library_folders(config['library_name'])
    return result.get('output') if isinstance(result, dict) else result


@getter_registry.register()
async def get_is_files_moved_downloads(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if files were moved from Downloads to OldDownloads."""
    file_name = config['file_name']
    folder_name = config['folder_name']

    downloads_resp = await runtime.list_directory(r'C:\Users\Docker\Downloads')
    downloads_directory = json.dumps(downloads_resp.get('directory_tree', {}))
    old_downloads_resp = await runtime.list_directory(r'C:\Users\Docker\OldDownloads')
    old_downloads_directory = old_downloads_resp.get('directory_tree', {})

    if file_name in downloads_directory and folder_name in downloads_directory:
        return 'false'

    for key in old_downloads_directory:
        if isinstance(old_downloads_directory[key], list):
            for i in old_downloads_directory[key]:
                if i['type'] == 'directory' and i['name'] == folder_name:
                    for j in i['children']:
                        if j['name'] == file_name:
                            return 'true'

    return 'false'


@getter_registry.register()
async def get_is_file_saved_desktop(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a file on the desktop contains specific text content."""
    desktop_path_resp = await runtime.desktop_path()
    desktop_path = desktop_path_resp['desktop_path']
    file_name = config['filename']
    file_content = await runtime.get_file(desktop_path + '\\' + file_name)
    if file_content is not None and config['textcontent'] in file_content.decode('utf-8', errors='replace'):
        return 'true'
    return 'false'
