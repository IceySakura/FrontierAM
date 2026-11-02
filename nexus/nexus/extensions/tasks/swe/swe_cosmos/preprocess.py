"""Preprocess for SWE Cosmos: rewrite problem_statement with tier/anti-hack for nexus input.

Reads a raw JSONL dataset, wraps each record's ``problem_statement`` with the chosen prompt tier, and writes the result to ``--output-path``.

Tier reference:

    --tier bare                  -> no wrapping
    --tier bare --anti-hack      -> bare + <IMPORTANT> block
    --tier template              -> <uploaded_files>/<pr_description>
    --tier template --anti-hack  -> template + <IMPORTANT> block
    --tier guided                -> template + step-by-step guidance
    --tier guided --anti-hack    -> guided + <IMPORTANT> block
    --tier random --anti-hack    -> sample one of the above tiers + <IMPORTANT> block

Usage:
    uv run python -m nexus.extensions.tasks.swe.swe_cosmos.preprocess -i input.jsonl -o output.jsonl --tier [bare|template|guided|random] --anti-hack
"""

from __future__ import annotations

import argparse
import json
import random
import sys


# --- Prompt templates (verbatim from rewrite_problem.py) ---------------------

_WRAP_HEAD = """<uploaded_files>
{working_dir}
</uploaded_files>
I've uploaded a code repository in the directory {working_dir}. Consider the following PR description:

<pr_description>
{problem_statement}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {working_dir} directory to ensure the <pr_description> is satisfied."""

_GUIDANCE = """
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
2. Create a script to reproduce the error and execute it using the bash tool, to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long."""

_ANTI_HACK = """

<IMPORTANT>

You must derive the fix from the code in {working_dir} yourself. Do NOT copy the answer from a future version of the code.

In particular, the following are forbidden:

- Cloning or fetching from the upstream github repository
- Downloading a newer version of the project's package (`npm pack pkg@<ver>`, `pip download <pkg>`, etc.) and reading its source
- Fetching specific files from github via `curl`/`wget` (blob, raw, releases URLs)
- Using `gh` CLI or the github API to look up upstream PRs / issues / commits
- Web searches for the fix

Reading {working_dir}'s local git history (`git log`, `git show <commit>`, `git diff`, `git blame`) is allowed — these don't touch any remote.

Submissions that rely on the forbidden channels are invalid regardless of test outcome.
</IMPORTANT>"""


TIERS = ('bare', 'template', 'guided', 'random')


def build_prompt(tier: str, problem_statement: str, working_dir: str, anti_hack: bool = False) -> str:
    """Build the wrapped problem_statement for the given tier and anti-hack flag."""
    if tier == 'random':
        tier = random.choice(TIERS[:-1])
    if tier == 'bare':
        if anti_hack:
            return problem_statement + _ANTI_HACK.format(working_dir=working_dir)
        return problem_statement
    text = _WRAP_HEAD
    if tier == 'guided':
        text += _GUIDANCE
    if anti_hack:
        text += _ANTI_HACK
    return text.format(working_dir=working_dir, problem_statement=problem_statement)


def _bool_arg(value: str | bool | None) -> bool:
    """Parse --anti-hack whether it arrives bare, or with a value."""
    if value is None or isinstance(value, bool):
        return True if value is None or value is True else value
    return str(value).strip().lower() in {'1', 'true', 't', 'yes', 'y', 'on'}


def rewrite(src: str, dst: str, tier: str, anti_hack: bool, working_dir_override: str | None) -> tuple[int, int]:
    """Read *src* JSONL, rewrite each record's problem_statement, write to *dst*.

    Returns ``(in_count, out_count)``; raises ``SystemExit(2)`` on invalid input.
    """
    in_count = 0
    out_count = 0
    with open(src, encoding='utf-8') as fin, open(dst, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            in_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f'Line {line_num} is not valid JSON: {exc}', file=sys.stderr)
                raise SystemExit(2) from exc
            if 'instance_id' not in row:
                print(f'Line {line_num} missing instance_id', file=sys.stderr)
                raise SystemExit(2)
            statement = row.get('problem_statement', row.get('problem', '')) or 'No problem statement provided.'
            workdir = working_dir_override or row.get('workdir') or '/testbed'
            row['problem_statement'] = build_prompt(tier, statement, workdir, anti_hack)
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            out_count += 1
    return in_count, out_count


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    ap = argparse.ArgumentParser(
        prog='nexus-preprocess',
        description='Rewrite problem_statement with tier/anti-hack for nexus input.',
    )
    ap.add_argument('-i', '--input', required=True, help='Input raw .jsonl file.')
    ap.add_argument('-o', '--output', required=True, help='Output preprocessed .jsonl file.')
    ap.add_argument(
        '--tier',
        required=True,
        choices=TIERS,
        help='Prompt tier: bare (no wrap), template (uploaded_files), guided (template + steps), random (sample one of the above).',
    )
    ap.add_argument(
        '--anti-hack',
        '--anti_hack',
        dest='anti_hack',
        nargs='?',
        const=True,
        default=False,
        type=_bool_arg,
        help='Append <IMPORTANT> block disabling external sources. Accepts --anti-hack, --anti-hack true, --anti_hack, etc.',
    )
    ap.add_argument(
        '--working-dir',
        '--working_dir',
        dest='working_dir',
        default=None,
        help="Override working_dir (default: each record's workdir field, or /testbed).",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point for rewriting problem_statement with tier/anti-hack for nexus input."""
    ap = build_parser()
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    in_count, out_count = rewrite(args.input, args.output, args.tier, args.anti_hack, args.working_dir)
    print(json.dumps({'in_count': in_count, 'out_count': out_count}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
