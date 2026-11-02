"""Anti-hack PROMPT notice for code_mt (the prevention half of the defense).

This is the notice appended to the turn-0 instruction when the dataset sets
``enable_anti_hack_prompt`` in task_config. It is the PREVENTION half; the
ENFORCEMENT half is the trajectory detector (task.py::scan_trajectory_for_anti_hack,
gated by the separate ``enable_anti_hack`` flag) which flags a run so the trainer
can zero/mask it. Stating the rule up front is what makes that enforcement fair —
a sample that is then masked or penalized violated a constraint it was told about.

Self-contained on purpose: this file is a byte-for-byte copy of the notice text
from nexus-main swe/task.py (ANTI_HACK_PROMPT_TEMPLATE + the two consequence
lines), kept LOCAL so the code_mt project has NO import dependency on the swe
task extension. The two must stay reviewable together, so if the swe version
changes, mirror it here.

The line the notice draws is "anything newer than this checkout", not "everything
except the local history": telling the agent which reads are sanctioned points it
at history as a place to look for the answer, which is the behaviour the notice
exists to suppress. Plain reads of the checked-out history remain unflagged.
"""

from __future__ import annotations


# Appended to the problem statement when ``enable_anti_hack_prompt`` is set. SWE-bench style tasks
# check the repository out at the commit *before* the fix, but the fix itself is public: it lives in
# an upstream GitHub PR, in a newer release of the package on PyPI/npm, or in a web-indexed issue
# thread. Whenever the sandbox has network access, the cheapest way to earn reward is to copy that
# patch rather than derive it, which makes the gradient reward answer lookup instead of debugging.
ANTI_HACK_PROMPT_TEMPLATE = """<IMPORTANT>
You must derive the fix from the checked-out repository yourself: read the local code,
run tests, add diagnostics, and edit the work tree. Do NOT obtain the answer from any
source that may already contain it -- remote history that is not present in this
checkout, a newer release, an installed copy, a code host, or the web. Every state of
this project newer than the checkout you were given is out of bounds, however you reach it.

Normal inspection of the current checkout's own history is allowed. You may use commands
such as `git log`, `git show`, `git diff`, and `git blame` when they stay within HEAD's
ancestor history, or when they read commits that came from that history.

The following are forbidden:

- Cloning, fetching, pulling, listing, or updating any remote git repository, including
  through a remote this checkout already has configured, which requires no URL to be named.
- Downloading a package's source distribution, tarball, archive, or source file from a
  registry or CDN, e.g. `pip download`, `uv pip download`, `npm pack`, `pnpm pack`,
  `yarn pack`, or `gem fetch`.
- Downloading source, patch, or diff files from a code host over HTTP, whether with
  curl/wget or with a script that issues the request itself.
- Using the GitHub CLI or a code host's HTTP API to look up upstream pull requests,
  issues, commits, patches, or files.
- Web searches for the fix.
- Reading git objects outside the current checkout's HEAD history: walking all refs with
  commands such as `git log --all` or `git rev-list --all`, reading the reflog, reading
  remote-tracking refs such as `origin/*` or `upstream/*`, or running `git show`,
  `git diff`, `git checkout`, `git cherry-pick`, `git format-patch`, or `git log` on a
  commit that was not observed from the allowed HEAD history.
- Reading the source of the project under fix from an installed package tree such as
  site-packages or dist-packages, or installing a pinned release of it.

{consequence}
</IMPORTANT>"""

# The consequence wording is FIXED to "detected and penalized". How a flagged sample is actually
# disposed of (mask vs penalize) is an RL-side decision (trainer.additional_terminate_reason), not
# a dataset one, so a per-row wording toggle could only be set out of sync. And it would not matter:
# a MASKED sample carries no gradient, so the only samples on which this notice can shape learning
# are the ones that DO train — the penalized (reward-zeroed) ones. Hence "penalized" is always the
# correct wording, and there is no knob to get wrong.
_CONSEQUENCE_PENALIZED = 'Submissions that rely on any of the above are detected and penalized.'


def anti_hack_prompt() -> str:
    """The anti-hack notice (consequence fixed to 'detected and penalized')."""
    return ANTI_HACK_PROMPT_TEMPLATE.format(consequence=_CONSEQUENCE_PENALIZED)


def append_anti_hack_notice(prompt: str) -> str:
    """Append the notice to the END of an already-rendered turn-0 prompt.

    Idempotent: if the notice sentinel is already present, it is NOT appended
    twice. Returns prompt unchanged when it is empty.
    """
    if not prompt:
        return prompt
    if '<IMPORTANT>' in prompt and 'out of bounds' in prompt:
        return prompt
    sep = '' if prompt.endswith('\n') else '\n'
    return f'{prompt}{sep}\n{anti_hack_prompt()}'
