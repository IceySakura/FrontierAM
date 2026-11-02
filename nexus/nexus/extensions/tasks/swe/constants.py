from enum import Enum

from typing_extensions import NotRequired, TypedDict


class SWEErrorCode(str, Enum):
    """Error codes for SWE-bench task execution."""

    PATCH_APPLY_FAILED = 'PATCH_APPLY_FAILED'
    TEST_EXECUTION_TIMEOUT = 'TEST_EXECUTION_TIMEOUT'
    EVAL_SCRIPT_FAILED = 'EVAL_SCRIPT_FAILED'
    EVAL_REPORT_FAILED = 'EVAL_REPORT_FAILED'
    TEST_OUTPUT_READ_FAILED = 'TEST_OUTPUT_READ_FAILED'


# Constants - Task Instance Class
class SWEbenchInstance(TypedDict, total=False):
    # Required fields (commonly present)
    dataset_type: str
    repo: str
    instance_id: str

    # Optional fields - may be missing or have alternative names in some datasets
    problem_statement: NotRequired[str]  # May be derived from dataset handler, default will get from instance
    base_commit: NotRequired[str]  # May fallback to base.sha
    runtime_base_commit: NotRequired[str]  # Override base_commit for runtime initialization
    patch: NotRequired[str]  # Golden patch
    test_patch: NotRequired[str]
    hints_text: NotRequired[str]
    version: NotRequired[str]
    FAIL_TO_PASS: NotRequired[str | list[str]]  # Can be JSON string or list
    PASS_TO_PASS: NotRequired[str | list[str]]  # Can be JSON string or list
    environment_setup_commit: NotRequired[str]
    spec_dict: NotRequired[dict | None]
    language: NotRequired[str]  # May be derived from repo via dataset handler
    workdir: NotRequired[str]  # default to /testbed
    image: NotRequired[str]
    image_name: NotRequired[str]  # Alternative name for image (used by SWE-smith)
    # Whether references stay within the base commit's ancestry; None means unverified.
    history_truncated: NotRequired[bool | None]


# Filter the agent-facing instance while preserving the original data for grading.
GOLD_FIELDS: frozenset[str] = frozenset({
    'patch',
    'fix_patch',
    'bug_patch',
    'test_patch',
    'FAIL_TO_PASS',
    'PASS_TO_PASS',
    'spec_dict',
})


# Constants - Dataset Types
DATASET_SWE_BENCH = 'swe-bench'
DATASET_SWE_BENCH_TURING = 'swe-bench-turing'
DATASET_SWE_GYM = 'swe-gym'
DATASET_T3BUILD_SWE_BENCH_LIVE = 'T3Build_swe_bench_live'
DATASET_SWE_REBENCH = 'swe-rebench'
DATASET_SWE_SMITH = 'swe-smith'
DATASET_SWE_BENCH_PRO = 'swe-bench_pro'
DATASET_MULTI_SWE_BENCH = 'multi-swe-bench'
DATASET_SWE_BENCH_PLUS_PLUS = 'swe-bench-plus-plus'
DATASET_SWE_BENCH_MULTILINGUAL = 'swe-bench-multilingual'
DATASET_ABAKA = 'abaka'
DATASET_T3BUILD_REPOLAUNCH_MIDTRAIN = 'T3Build_RepoLaunch_Midtrain'
DATASET_T3BUILD_SWEBUILDER_MIDTRAIN = 'T3Build_SWEBuilder_Midtrain'
DATASET_SCALE_SWE = 'scale-swe'
DATASET_SWE_REBENCH_V2 = 'swe-rebench-v2'
DATASET_SWE_COSMOS = 'swe-cosmos'
DATASET_SWE_ENV = 'swe-env'
DATASET_SWE_ANY = 'swe-any'

KEY_PREDICTION = 'model_patch'
KEY_INSTANCE_ID = 'instance_id'


# Limit repository history to BASE_SHA before task execution.
# Disable dereferencing for each reference update to avoid following symbolic references.
# Expire reflogs and prune unreachable commits even when no references need changing.
GIT_TRUNCATE_SCRIPT = r"""
set -e
future=$(comm -23 <(git rev-list --all | sort -u) <(git rev-list "$BASE_SHA" | sort -u))
if [ -z "$future" ]; then
    echo "INIT: Nothing to do";
else
    echo "INIT: Found $(echo "$future" | wc -l) future commits"
    declare -A F
    while read -r c; do
        [[ -n "$c" ]] && F["$c"]=1
    done <<< "$future"
    {
        while read -r ref tip deref deref_type; do
            commit="${deref:-$tip}"
            # for-each-ref peels only one level; nested annotated tags need a complete peel.
            if [[ "$deref_type" == tag ]]; then
                commit=$(git rev-parse --verify "$ref^{commit}" 2>/dev/null) || continue
            fi
            [[ -n "$commit" && -v F["$commit"] ]] || continue
            if [[ "$ref" == refs/heads/* ]]; then
                printf 'option no-deref\nupdate %s %s\n' "$ref" "$BASE_SHA"
                echo "INIT: Updated $ref to $BASE_SHA" >&2
            else
                printf 'option no-deref\ndelete %s\n' "$ref"
            fi
        done <<< "$(git for-each-ref --format="%(refname) %(objectname) %(*objectname) %(*objecttype)" refs/heads refs/tags refs/remotes)"
    } | git update-ref --stdin
    git stash clear 2>/dev/null || true
    git for-each-ref --format="%(refname)" refs/notes refs/replace refs/worktree | xargs -r git update-ref -d 2>/dev/null || true
    echo "Truncated at $BASE_SHA"
fi

# Expire reflogs and prune unreachable commits even when no references changed.
git stash clear 2>/dev/null || true
git reflog expire --expire=all --all 2>/dev/null || true
# Report collection failures without aborting repository initialization.
if ! git gc --prune=now 2>/tmp/.nexus_gc_err; then
    echo "INIT: GC_FAILED (unreachable objects may remain readable via git show / cat-file)"
    echo "INIT: GC_STDERR: $(tr '\n' ' ' < /tmp/.nexus_gc_err | head -c 500)"
fi
rm -f /tmp/.nexus_gc_err 2>/dev/null || true
# Count stored commits that no remaining reference can reach.
echo "INIT: DANGLING_OBJECTS=$( { git cat-file --batch-all-objects --batch-check='%(objectname) %(objecttype)' 2>/dev/null \
    | awk '$2=="commit"{print $1}' | sort -u > /tmp/.nexus_all_c; \
    git rev-list --all 2>/dev/null | sort -u > /tmp/.nexus_reach_c; \
    comm -23 /tmp/.nexus_all_c /tmp/.nexus_reach_c | wc -l; } 2>/dev/null || echo -1 )"
rm -f /tmp/.nexus_all_c /tmp/.nexus_reach_c 2>/dev/null || true
"""

GIT_REINIT_SCRIPT = """
set -e

find . -name .git -exec rm -rf {} +
find . -name .gitmodules -type f -exec rm -f {} +
git config --global init.defaultBranch main
git config --global --add safe.directory "$(pwd)"

git init
git config --local user.name "nexus"
git config --local user.email "nexus@anonymous.com"
git config --local commit.gpgsign false
git add -A
git commit --no-verify --allow-empty -m "Initial commit"
"""


# Shared task notice rendered before agent execution.
ANTI_HACK_PROMPT = """<IMPORTANT>
You must derive the fix from the checked-out repository yourself: read the local code, run tests,
add diagnostics, and edit the work tree. Do NOT obtain the answer from any source that may already
contain it -- remote history that is not present in this checkout, a newer release, an installed
copy, a code host, the web, or the evaluation harness itself. Every state of this project newer
than the checkout you were given is out of bounds, however you reach it.

Normal inspection of the current checkout's own history is allowed. You may use commands such as
`git log`, `git show`, `git diff`, and `git blame` when they stay within HEAD's ancestor history,
or when they read commits that came from that history.

The following are forbidden:

- Cloning, fetching, pulling, or listing any remote git repository, including through a remote this
  checkout already has configured, which requires no URL to be named.
- Downloading a package's source distribution, tarball, archive, or source file from a registry or
  CDN, e.g. `pip download`, `uv pip download`, `npm pack`, `pnpm pack`, `yarn pack`, or `gem fetch`.
- Downloading source, patch, or diff files from a code host over HTTP, whether with curl/wget or
  with a script that issues the request itself.
- Using the GitHub CLI or a code host's HTTP API to look up upstream pull requests, issues, commits,
  patches, or files.
- Web searches for the fix.
- Reading git objects outside the current checkout's HEAD history: walking all refs with commands
  such as `git log --all` or `git rev-list --all`, reading the reflog, reading remote-tracking refs
  such as `origin/*` or `upstream/*`, or running `git show`, `git diff`, `git checkout`,
  `git cherry-pick`, `git format-patch`, or `git log` on a commit that was not observed from the
  allowed HEAD history.
- Reading the source of the project under fix from an installed package tree such as site-packages
  or dist-packages, or installing a pinned release of it.

Submissions that rely on any of the above are invalid and are discarded.
</IMPORTANT>"""


# Restrict reading framework history when the shell backend retains command output.
TMUX_HISTORY_CLAUSE = """
- Mining the harness's own bookkeeping for task content: the framework's command and editor history
  under `/tmp/nexus/`, and captured tool output. If you encounter such a file, do not read it for
  the fix. Finding the answer lying around does not make it yours to use."""


def anti_hack_prompt_for_backend(backend: str | None = None) -> str:
    """Render the shared notice for the selected shell backend."""
    import os as _os

    if backend is None:
        backend = _os.getenv('SWE_AGENT_BASH_BACKEND', 'terminal')
    notice = ANTI_HACK_PROMPT
    tail = '\n\nSubmissions that rely on any of the above are invalid and are discarded.'
    if tail not in notice:  # pragma: no cover - guards against future rewording
        raise RuntimeError('ANTI_HACK_PROMPT closing line changed; update anti_hack_prompt_for_backend')
    if (backend or '').strip().lower() in {'terminal', 'tmux'}:
        notice = notice.replace(tail, TMUX_HISTORY_CLAUSE + tail, 1)
    return notice
