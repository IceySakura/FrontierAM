from __future__ import annotations

import os

from dataclasses import dataclass, field

import aiofiles
import aiofiles.os
import yaml


SKILLS_FILENAME = 'SKILL.md'
MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 1024


@dataclass
class SkillMetadata:
    name: str
    description: str
    path: str


@dataclass
class SkillParseError:
    path: str
    message: str


@dataclass
class SkillLoadResult:
    skills: list[SkillMetadata] = field(default_factory=list)
    errors: list[SkillParseError] = field(default_factory=list)


def extract_frontmatter(contents: str) -> str | None:
    """Extract YAML frontmatter from file contents."""
    lines = contents.split('\n')

    if not lines or lines[0].strip() != '---':
        return None

    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == '---':
            frontmatter_lines = lines[1:i]
            if not frontmatter_lines:
                return None
            return '\n'.join(frontmatter_lines)

    return None


def sanitize_single_line(raw: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    if not raw:
        return ''
    return ' '.join(raw.split())


def validate_field(value: str, max_len: int, field_name: str) -> None:
    """Validate that field is non-empty and within length limit."""
    if not value:
        raise ValueError(f'missing field `{field_name}`')
    if len(value) > max_len:
        raise ValueError(f'invalid {field_name}: exceeds {max_len} characters')


async def parse_skill_file(skill_path: str) -> SkillMetadata:
    """Parse a single SKILL.md file."""
    skill_path = os.path.abspath(skill_path)

    async with aiofiles.open(skill_path, encoding='utf-8') as f:
        contents = await f.read()

    frontmatter = extract_frontmatter(contents)
    if frontmatter is None:
        raise ValueError('missing YAML frontmatter delimited by ---')

    parsed = yaml.safe_load(frontmatter)
    if not isinstance(parsed, dict):
        raise ValueError('frontmatter must be a YAML mapping')

    name = sanitize_single_line(parsed.get('name', ''))
    description = sanitize_single_line(parsed.get('description', ''))

    validate_field(name, MAX_NAME_LEN, 'name')
    validate_field(description, MAX_DESCRIPTION_LEN, 'description')

    return SkillMetadata(
        name=name,
        description=description,
        path=skill_path,
    )


async def load_skills_from_directories(
    directories: list[str],
    max_depth: int = 6,
    max_dirs: int = 2000,
    follow_symlinks: bool = True,
) -> SkillLoadResult:
    """Scan directory list and return all skill metadata."""
    result = SkillLoadResult()
    seen_paths: set[str] = set()

    for root_dir in directories:
        root_dir = os.path.abspath(root_dir)
        if not await aiofiles.os.path.isdir(root_dir):
            continue

        visited_dirs: set[str] = {root_dir}
        queue: list[tuple[str, int]] = [(root_dir, 0)]
        dir_count = 0

        while queue and dir_count < max_dirs:
            current_dir, depth = queue.pop(0)
            dir_count += 1

            try:
                entries = await aiofiles.os.listdir(current_dir)
            except OSError:
                continue

            for entry_name in entries:
                if entry_name.startswith('.'):
                    continue

                entry_path = os.path.join(current_dir, entry_name)

                if os.path.islink(entry_path):
                    if not follow_symlinks:
                        continue
                    try:
                        real_path = os.path.realpath(entry_path)
                        if (
                            await aiofiles.os.path.isdir(real_path)
                            and depth < max_depth
                            and real_path not in visited_dirs
                        ):
                            visited_dirs.add(real_path)
                            queue.append((real_path, depth + 1))
                    except OSError:
                        pass
                    continue

                if await aiofiles.os.path.isdir(entry_path) and depth < max_depth:
                    real_path = os.path.realpath(entry_path)
                    if real_path not in visited_dirs:
                        visited_dirs.add(real_path)
                        queue.append((real_path, depth + 1))
                    continue

                if entry_name == SKILLS_FILENAME:
                    real_path = os.path.realpath(entry_path)
                    if real_path in seen_paths:
                        continue
                    seen_paths.add(real_path)

                    try:
                        skill = await parse_skill_file(entry_path)
                        result.skills.append(skill)
                    except Exception as e:
                        result.errors.append(
                            SkillParseError(
                                path=entry_path,
                                message=str(e),
                            )
                        )

    result.skills.sort(key=lambda s: s.name)

    return result
