"""Agentless support for SWE-bench tasks.

This module provides:
- AgentlessOutput: Structured model output with search/replace blocks
- AgentlessAgentInput/Output: Agent input/output types
- run_agentless_agent: Agent that applies pre-computed edits without LLM inference
"""

import ast
import logging
import re

from collections import OrderedDict

from pydantic import BaseModel, Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    agent_registry,
    require_context,
)
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


# ============== Data Models ==============


class AgentlessOutput(BaseModel):
    """Structured output from agentless-style model inference."""

    raw_output: str
    """Raw model output containing search/replace blocks."""

    found_files: list[str]
    """List of file paths that the model edited."""


class AgentlessAgentInput(AgentRunInput):
    """Input for agentless agent."""

    agentless_output: AgentlessOutput = Field(description='Pre-computed agentless output with search/replace edits')


class AgentlessAgentOutput(AgentRunOutput):
    """Output from agentless agent."""

    success: bool = Field(default=False, description='Whether edits were applied successfully')
    written_files: dict[str, str] = Field(default_factory=dict, description='Map of file paths to their new contents')


# ============== Parser Functions ==============


def extract_python_blocks(text: str) -> list[str]:
    """Extract code blocks from markdown-style ```python blocks."""
    pattern = r'```python\n(.*?)\n```'
    return re.findall(pattern, text, re.DOTALL)


def split_edit_multifile_commands(commands: list[str]) -> dict[str, list[str]]:
    """Split commands based on edited files.

    Parses search/replace blocks and groups them by file path.
    """
    file_to_commands: dict[str, list[str]] = OrderedDict()

    for command in commands:
        file_name = None
        for subcommand in command.split('>>>>>>> REPLACE')[:-1]:
            subcommand = subcommand.strip()
            if '<<<<<<< SEARCH' in subcommand:
                fn = subcommand.split('<<<<<<< SEARCH')[0].lstrip('#').strip()
                if fn:
                    file_name = fn

            if len(subcommand.split('<<<<<<< SEARCH')) != 2:
                continue

            converted_command = '<<<<<<< SEARCH' + subcommand.split('<<<<<<< SEARCH')[1] + '\n' + '>>>>>>> REPLACE'

            # Deduplicate
            if file_name is not None and (
                file_name not in file_to_commands or converted_command not in file_to_commands[file_name]
            ):
                file_to_commands.setdefault(file_name, []).append(converted_command)

    return file_to_commands


def parse_diff_edit_commands(commands: list[str], content: str) -> str:
    """Apply search/replace commands to file content."""
    replaced = False
    can_apply = []

    for subcommand in commands:
        if not subcommand.startswith('<<<<<<< SEARCH') and subcommand.endswith('>>>>>>> REPLACE'):
            continue

        subcommand = '\n'.join(subcommand.splitlines()[1:-1])
        if len(subcommand.split('\n=======\n')) != 2:
            continue

        original, replace = subcommand.split('\n=======\n')

        if original in content:
            can_apply.append(subcommand)

    for subcommand in can_apply:
        original, replace = subcommand.split('\n=======\n')
        content = content.replace(original, replace)
        replaced = True

    if not replaced:
        logger.debug('No replacements applied')

    return content


def check_syntax(code: str | list[str]) -> bool:
    """Check if Python code has valid syntax."""
    if not isinstance(code, list):
        code = [code]

    for c in code:
        if not c.strip():
            return False
        try:
            ast.parse(c)
        except SyntaxError:
            return False
    return True


def remove_empty_lines(code: str) -> str:
    """Remove empty lines from code."""
    lines = code.splitlines()
    filtered_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(filtered_lines)


def check_code_differ_by_just_empty_lines(
    codes: str | list[str],
    prev_codes: str | list[str],
) -> bool:
    """Check if two code snippets differ only by empty lines."""
    codes_list: list[str] = [codes] if isinstance(codes, str) else codes
    prev_codes_list: list[str] = [prev_codes] if isinstance(prev_codes, str) else prev_codes

    normalized_code1 = ''
    normalized_code2 = ''

    for code, prev_code in zip(codes_list, prev_codes_list, strict=False):
        normalized_code1 += remove_empty_lines(code)
        normalized_code2 += remove_empty_lines(prev_code)

    return normalized_code1 == normalized_code2


def apply_edits_to_files(
    raw_output_text: str,
    file_contents: dict[str, str],
) -> dict[str, str]:
    """Apply search/replace edits to file contents.

    Args:
        raw_output_text: Raw model output with search/replace blocks.
        file_contents: Dict mapping file paths to their original contents.

    Returns:
        Dict mapping file paths to their new contents (only files that changed).
    """
    edit_multifile_commands = extract_python_blocks(raw_output_text)
    file_to_commands = split_edit_multifile_commands(edit_multifile_commands)

    result: dict[str, str] = {}

    for edited_file, edit_commands in file_to_commands.items():
        if edited_file not in file_contents:
            logger.warning(f'File not found in contents: {edited_file}')
            continue

        old_content = file_contents[edited_file]
        new_content = parse_diff_edit_commands(edit_commands, old_content)

        if not new_content:
            continue

        # Skip if no actual changes
        if new_content == old_content:
            logger.debug(f'No changes for file: {edited_file}')
            continue

        # Validate syntax
        if not check_syntax(new_content):
            logger.warning(f'Syntax check failed for: {edited_file}')
            continue

        # Skip if only empty line differences
        if check_code_differ_by_just_empty_lines(new_content, old_content):
            logger.warning(f'Only empty line differences for: {edited_file}')
            continue

        result[edited_file] = new_content

    return result


# ============== Runtime Operations ==============


async def read_files_from_runtime(
    runtime: Runtime,
    file_paths: list[str],
    working_dir: str = '/testbed',
) -> dict[str, str]:
    """Read multiple files from runtime."""
    return await runtime.invoke(
        _read_files,
        file_paths,
        working_dir,
    )


def _read_files(
    file_paths: list[str],
    working_dir: str = '/testbed',
) -> dict[str, str]:
    """Read files from filesystem. Executed inside runtime."""
    from pathlib import Path

    result: dict[str, str] = {}
    base_path = Path(working_dir)

    for file_path in file_paths:
        full_path = base_path / file_path
        try:
            if full_path.exists():
                result[file_path] = full_path.read_text()
            else:
                logger.warning(f'File not found: {full_path}')
        except Exception as e:
            logger.warning(f'Failed to read {full_path}: {e}')

    return result


async def write_files_to_runtime(
    runtime: Runtime,
    file_contents: dict[str, str],
    working_dir: str = '/testbed',
) -> list[str]:
    """Write multiple files to runtime."""
    return await runtime.invoke(
        _write_files,
        file_contents,
        working_dir,
    )


def _write_files(
    file_contents: dict[str, str],
    working_dir: str = '/testbed',
) -> list[str]:
    """Write files to filesystem. Executed inside runtime."""
    from pathlib import Path

    written: list[str] = []
    base_path = Path(working_dir)

    for file_path, content in file_contents.items():
        full_path = base_path / file_path
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            written.append(file_path)
        except Exception as e:
            logger.warning(f'Failed to write {full_path}: {e}')

    return written


async def apply_agentless_edits(
    runtime: Runtime,
    agentless_output: AgentlessOutput,
) -> tuple[bool, dict[str, str]]:
    """Apply agentless edits to files in runtime.

    This function:
    1. Reads file contents from runtime
    2. Applies search/replace edits locally
    3. Writes modified files back to runtime

    Returns:
        Tuple of (success, written_files) where written_files maps paths to new contents.
    """
    ctx = require_context()
    working_dir = ctx.tool_state.get('working_dir', '/testbed')

    # Read file contents from runtime
    file_contents = await read_files_from_runtime(
        runtime,
        agentless_output.found_files,
        working_dir,
    )

    if not file_contents:
        logger.warning('No files could be read from runtime')
        return False, {}

    logger.debug(f'Read {len(file_contents)} files from runtime: {list(file_contents.keys())}')

    # Apply edits locally
    edited_files = apply_edits_to_files(
        agentless_output.raw_output,
        file_contents,
    )

    if not edited_files:
        logger.warning('No valid edits could be applied')
        return False, {}

    logger.debug(f'Applied edits to {len(edited_files)} files: {list(edited_files.keys())}')

    # Write modified files back to runtime
    written = await write_files_to_runtime(runtime, edited_files, working_dir)

    if not written:
        logger.warning('Failed to write any files to runtime')
        return False, {}

    logger.debug(f'Written {len(written)} files to runtime')
    # Return only the files that were actually written
    written_files = {path: edited_files[path] for path in written}
    return True, written_files


# ============== Agent ==============


@agent_registry.register('agentless')
async def run_agentless_agent(agent_input: AgentlessAgentInput) -> AgentlessAgentOutput:
    """Run agentless agent to apply pre-computed edits.

    This agent doesn't run any LLM inference. It simply:
    1. Reads the specified files from runtime
    2. Applies search/replace edits locally
    3. Writes modified files back to runtime

    The patch is then extracted by the task using get_patch_from_runtime.
    """
    ctx = require_context()

    # Get runtime from context (should be pre-configured by task)
    runtime = await ctx.get_runtime()

    logger.debug(f'Applying agentless edits to {len(agent_input.agentless_output.found_files)} files')

    success, written_files = await apply_agentless_edits(runtime, agent_input.agentless_output)

    if success:
        logger.debug('Agentless edits applied successfully')
    else:
        logger.warning('Failed to apply agentless edits')

    return AgentlessAgentOutput(success=success, written_files=written_files)


__all__ = [
    'AgentlessAgentInput',
    'AgentlessAgentOutput',
    'AgentlessOutput',
    'apply_agentless_edits',
    'run_agentless_agent',
]
