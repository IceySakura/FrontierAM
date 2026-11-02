"""Sandbox-related CLI commands."""

from __future__ import annotations

import argparse
import asyncio
import sys


def sandbox_exec(args: argparse.Namespace) -> int:
    """Execute a command in a held sandbox via RemoteRuntime."""
    from nexus.runtime.client.remote import RemoteRuntime

    async def _exec() -> int:
        from nexus.utils.async_http import AsyncHTTP

        runtime = RemoteRuntime(base_url=args.base_url, auto_connect_ws=False, request_timeout=args.timeout + 30)
        try:
            result = await runtime.run_command(
                command=args.exec_command,
                cwd=args.cwd,
                timeout=args.timeout,
            )
            if result.stdout:
                sys.stdout.write(result.stdout)
                if not result.stdout.endswith('\n'):
                    sys.stdout.write('\n')
            if result.stderr:
                sys.stderr.write(result.stderr)
                if not result.stderr.endswith('\n'):
                    sys.stderr.write('\n')
            return result.return_code if result.return_code is not None else 1
        except Exception as e:
            import aiohttp

            if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                sys.stderr.write(f'error: command timed out after {args.timeout}s\n')
            elif isinstance(e, (aiohttp.ClientError, OSError)):
                sys.stderr.write(f'error: cannot reach sandbox at {args.base_url}: {e}\n')
            else:
                sys.stderr.write(f'error: {type(e).__name__}: {e}\n')
            return 1
        finally:
            await AsyncHTTP.close_all()

    return asyncio.run(_exec())


def sandbox_upload(args: argparse.Namespace) -> int:
    """Upload a local file to a held sandbox via RemoteRuntime."""
    from nexus.runtime.client.remote import RemoteRuntime

    async def _upload() -> int:
        import base64

        from pathlib import Path

        from nexus.runtime.executors.models.file import FileItem
        from nexus.utils.async_http import AsyncHTTP

        runtime = RemoteRuntime(base_url=args.base_url, auto_connect_ws=False, request_timeout=args.timeout + 30)
        try:
            local_path = Path(args.local_path)
            if not local_path.exists():
                sys.stderr.write(f'error: local file not found: {args.local_path}\n')
                return 1
            if not local_path.is_file():
                sys.stderr.write(f'error: not a regular file: {args.local_path}\n')
                return 1

            content_bytes = local_path.read_bytes()
            if len(content_bytes) > 10 * 1024 * 1024:
                sys.stderr.write('warning: uploading file larger than 10MB may take a while\n')

            try:
                content = content_bytes.decode('utf-8')
                encoding = 'utf8'
            except UnicodeDecodeError:
                content = base64.b64encode(content_bytes).decode('ascii')
                encoding = 'base64'

            item = FileItem(path=args.remote_path, content=content, encoding=encoding)
            response = await runtime.upload_file(files=[item])
            if not response.results or not response.results[0].success:
                message = response.results[0].message if response.results else 'upload failed'
                sys.stderr.write(f'error: {message}\n')
                return 1

            sys.stdout.write(f'uploaded: {args.local_path} -> {args.remote_path}\n')
            return 0
        except Exception as e:
            import aiohttp

            if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                sys.stderr.write(f'error: request timed out after {args.timeout}s\n')
            elif isinstance(e, (aiohttp.ClientError, OSError)):
                sys.stderr.write(f'error: cannot reach sandbox at {args.base_url}: {e}\n')
            else:
                sys.stderr.write(f'error: {type(e).__name__}: {e}\n')
            return 1
        finally:
            await AsyncHTTP.close_all()

    return asyncio.run(_upload())


def sandbox_download(args: argparse.Namespace) -> int:
    """Download a file from a held sandbox via RemoteRuntime."""
    from nexus.runtime.client.remote import RemoteRuntime

    async def _download() -> int:
        import base64

        from pathlib import Path

        from nexus.runtime.executors.models.file import FileItem
        from nexus.utils.async_http import AsyncHTTP

        runtime = RemoteRuntime(base_url=args.base_url, auto_connect_ws=False, request_timeout=args.timeout + 30)
        try:
            item = FileItem(path=args.remote_path, encoding='base64')
            response = await runtime.download_file(files=[item])
            if response.errors or not response.files:
                message = response.errors[0].message if response.errors else 'download failed'
                sys.stderr.write(f'error: {message}\n')
                return 1

            file_item = response.files[0]
            data = base64.b64decode(file_item.content or '')

            if args.local_path == '-':
                try:
                    sys.stdout.write(data.decode('utf-8'))
                except UnicodeDecodeError:
                    sys.stdout.buffer.write(data)
            else:
                local_path = Path(args.local_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(data)
                sys.stdout.write(f'downloaded: {args.remote_path} -> {args.local_path}\n')

            return 0
        except Exception as e:
            import aiohttp

            if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                sys.stderr.write(f'error: request timed out after {args.timeout}s\n')
            elif isinstance(e, (aiohttp.ClientError, OSError)):
                sys.stderr.write(f'error: cannot reach sandbox at {args.base_url}: {e}\n')
            else:
                sys.stderr.write(f'error: {type(e).__name__}: {e}\n')
            return 1
        finally:
            await AsyncHTTP.close_all()

    return asyncio.run(_download())


def register_sandbox_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the sandbox subcommand group."""
    sandbox_parser = subparsers.add_parser(
        'sandbox',
        help='Interact with held sandboxes',
    )
    sandbox_subparsers = sandbox_parser.add_subparsers(dest='sandbox_command')

    exec_parser = sandbox_subparsers.add_parser(
        'exec',
        help='Execute a command in a held sandbox',
        description='Run a shell command inside a sandbox that is being held (via --hold). '
        'The sandbox must be running and reachable at the given base_url.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Check installed packages
  nexus sandbox exec http://10.0.0.1:32000 "pip list"

  # Inspect files in a specific directory
  nexus sandbox exec http://10.0.0.1:32000 "ls -la /app" --cwd /app

  # Run a long diagnostic with extended timeout
  nexus sandbox exec http://10.0.0.1:32000 "pytest tests/ -x" --timeout 600

  # Check environment variables
  nexus sandbox exec http://10.0.0.1:32000 "env | grep PATH"

workflow:
  1. Run a task with --hold:  nexus run-single config.yaml --hold 3600
  2. The hold banner prints the sandbox base_url
  3. Use this command to inspect the sandbox
  4. Send SIGINT (Ctrl+C) to the held process to release the sandbox
""",
    )
    exec_parser.add_argument(
        'base_url',
        help='Base URL of the held sandbox runtime server (printed in the hold banner)',
    )
    exec_parser.add_argument(
        'exec_command',
        metavar='command',
        help='Shell command to execute inside the sandbox',
    )
    exec_parser.add_argument(
        '--cwd',
        default=None,
        help='Working directory for the command inside the sandbox (default: sandbox root)',
    )
    exec_parser.add_argument(
        '--timeout',
        type=float,
        default=180,
        help='Maximum seconds to wait for command completion (default: 180). '
        'The HTTP request timeout is set to this value + 30s to avoid client-side premature cutoff.',
    )

    upload_parser = sandbox_subparsers.add_parser(
        'upload',
        help='Upload a local file to a held sandbox',
        description='Upload a local file into a sandbox that is being held (via --hold). '
        'The sandbox must be running and reachable at the given base_url.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Upload a script into the sandbox
  nexus sandbox upload http://10.0.0.1:32000 ./script.sh /workspace/script.sh

  # Upload a binary artifact
  nexus sandbox upload http://10.0.0.1:32000 ./image.png /workspace/image.png

workflow:
  1. Run a task with --hold:  nexus run-single config.yaml --hold 3600
  2. The hold banner prints the sandbox base_url
  3. Use this command to copy files into the sandbox
  4. Send SIGINT (Ctrl+C) to the held process to release the sandbox
""",
    )
    upload_parser.add_argument(
        'base_url',
        help='Base URL of the held sandbox runtime server (printed in the hold banner)',
    )
    upload_parser.add_argument(
        'local_path',
        help='Local file path to upload',
    )
    upload_parser.add_argument(
        'remote_path',
        help='Destination file path inside the sandbox',
    )
    upload_parser.add_argument(
        '--timeout',
        type=float,
        default=180,
        help='Maximum seconds to wait for upload completion (default: 180). '
        'The HTTP request timeout is set to this value + 30s to avoid client-side premature cutoff.',
    )

    download_parser = sandbox_subparsers.add_parser(
        'download',
        help='Download a file from a held sandbox',
        description='Download a file from a sandbox that is being held (via --hold). '
        'The sandbox must be running and reachable at the given base_url.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Download a sandbox output file
  nexus sandbox download http://10.0.0.1:32000 /workspace/output.txt ./output.txt

  # Print a remote log to stdout
  nexus sandbox download http://10.0.0.1:32000 /workspace/log.txt -

workflow:
  1. Run a task with --hold:  nexus run-single config.yaml --hold 3600
  2. The hold banner prints the sandbox base_url
  3. Use this command to copy files out of the sandbox
  4. Send SIGINT (Ctrl+C) to the held process to release the sandbox
""",
    )
    download_parser.add_argument(
        'base_url',
        help='Base URL of the held sandbox runtime server (printed in the hold banner)',
    )
    download_parser.add_argument(
        'remote_path',
        help='Source file path inside the sandbox',
    )
    download_parser.add_argument(
        'local_path',
        help="Local file path to write, or '-' to write to stdout",
    )
    download_parser.add_argument(
        '--timeout',
        type=float,
        default=180,
        help='Maximum seconds to wait for download completion (default: 180). '
        'The HTTP request timeout is set to this value + 30s to avoid client-side premature cutoff.',
    )
