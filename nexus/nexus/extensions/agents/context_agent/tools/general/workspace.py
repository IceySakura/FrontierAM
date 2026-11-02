"""Per-rollout workspace for HyGeneralToolSystem.

The workspace IS the rollout log directory, so trajectory artefacts land
alongside whatever the rollout produced. ``workspace/`` is the agent's own
folder inside it: the task's local files are copied in before the loop starts
(see :func:`..state.stage_declared_files`), and the agent adds to it with
``curl`` and by writing its conversions there. It is empty only for a task
whose inputs are all URLs, or when staging is switched off.

``cwd`` is the log directory, so the model addresses whatever it downloaded by
the relative path ``./workspace/<name>``, exactly as the system prompt says.
"""

from __future__ import annotations

import logging
import re
import tempfile

from pathlib import Path


logger = logging.getLogger(__name__)

# Unicode word characters, not just ASCII ones. An ASCII-only class turns
# ``财报.md`` into ``md`` — every character before the dot is replaced, then the
# leading separators are stripped — which loses the name the agent has to cite
# and collides with any other non-ASCII name in the same task. Path separators,
# whitespace and punctuation are not word characters, so they are still removed.
_NAME_SANITIZE = re.compile(r'[^\w.\-]+')


def sanitize_filename(name: str, fallback: str = 'download') -> str:
    """Return a filesystem-safe filename derived from *name*."""
    base = _NAME_SANITIZE.sub('_', (name or '').strip()).strip('._')
    return base or fallback


class Workspace:
    """Rollout log directory plus the (initially empty) download folder.

    ``docs_dir`` / ``cwd`` may be pointed somewhere else, which is what lets a
    rollout work *in place* in a directory it does not own — under Harbor the
    graded ``/workdir``, which the container pre-populates and the verifier
    reads back. Tool trajectories are persisted by Nexus on the host; this class does not
    create a tool-call log directory inside the model workspace.
    """

    DOCS_SUBDIR = 'workspace'

    def __init__(
        self,
        root: Path,
        docs_dir: Path | str | None = None,
        cwd: Path | str | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        if docs_dir is None:
            self.docs_dir = self.root / self.DOCS_SUBDIR
            # Relative, because cwd is the log dir and the system prompt tells
            # the model to address files as ./workspace/<name>.
            self._docs_prefix = f'./{self.DOCS_SUBDIR}'
        else:
            self.docs_dir = Path(docs_dir)
            # Absolute: an external docs dir is not under cwd in general, and
            # bash / python get the string verbatim.
            self._docs_prefix = str(self.docs_dir).rstrip('/')
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self._cwd = Path(cwd) if cwd is not None else self.root

    @property
    def cwd(self) -> str:
        """Working directory for the bash / python tools (the log folder)."""
        return str(self._cwd)

    def path_for(self, name: str) -> str:
        """Relative path a downloaded file should be referred to by."""
        return f'{self._docs_prefix}/{name}'


def prepare_workspace(
    log_dir: Path | None,
    docs_dir: Path | str | None = None,
    cwd: Path | str | None = None,
) -> Workspace:
    """Create a :class:`Workspace` rooted at *log_dir* (temp dir when ``None``).

    Pass *docs_dir* to work in an existing directory instead of the log dir's
    own ``workspace/`` (see :class:`Workspace`); *cwd* defaults to *docs_dir*
    in that case, so the shell tools start where the task's files are.
    """
    if log_dir is None:
        log_dir = Path(tempfile.mkdtemp(prefix='hygeneral_'))
    if docs_dir is not None and cwd is None:
        cwd = docs_dir
    return Workspace(log_dir, docs_dir=docs_dir, cwd=cwd)
