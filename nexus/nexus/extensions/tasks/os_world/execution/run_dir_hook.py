"""Record source identity for an optional local result directory."""

from __future__ import annotations

import subprocess

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nexus.runner.hooks.base import RunHook, run_hook_registry


if TYPE_CHECKING:
    from nexus.core.context import ExecutionContext


COMMIT_STAMP = 'source_commit.txt'


def source_commit(repo: Path) -> str:
    probe = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, check=False)
    return probe.stdout.strip() or 'unknown'


@run_hook_registry.register('osworld_run_dir')
class OSWorldRunDirHook(RunHook):
    """Create the result directory and record the source commit."""

    def __init__(self, *, run_dir: str) -> None:
        self.run_dir = Path(run_dir)
        self.commit = ''

    async def on_context_ready(self, ctx: ExecutionContext) -> None:
        # Concurrent tasks in a batch all land here; the directory is shared on
        # purpose, so an existing one is the normal case, not a conflict.
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.commit = source_commit(Path(__file__).resolve().parents[5])
        stamp = self.run_dir / COMMIT_STAMP
        recorded = stamp.read_text(encoding='utf-8').strip() if stamp.is_file() else ''
        if recorded and recorded != self.commit:
            raise RuntimeError(
                f'{self.run_dir} holds results from commit {recorded}; '
                f'this checkout is {self.commit}. Use a fresh result directory.'
            )
        if not recorded:
            stamp.write_text(f'{self.commit}\n', encoding='utf-8')

    def get_result(self) -> dict[str, Any]:
        return {'run_dir': str(self.run_dir), 'source_commit': self.commit}
