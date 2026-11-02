"""Naming a run's worker skills, and placing them where its harness reads them.

The two halves are split like the user memory file: the generator reads the
directories, where they exist, and records what they install as; the backend declares
where its harness reads them and uploads them there. So what is pinned here is that
``with_worker_skills`` carries the directories and the command names and nothing about
placement, and that ``place_worker_skills`` resolves the home sandbox-side and refuses
to leave a failure quiet -- a worker whose skills never arrived still advertises them
through ``worker_commands``, so the user simulator asks for a command that does nothing.
"""

from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.tasks.hitl.actor.factory import backend_class, backends
from nexus.extensions.tasks.hitl.actor.input_spec import WORKER_COMMANDS_KEY
from nexus.extensions.tasks.hitl.actor.skills import CONFIG_KEY, place_worker_skills, skill_dirs
from nexus.extensions.tasks.hitl.utils.generator_util import with_worker_skills


def write_skill(parent: Path, dirname: str, *, description: str = 'what it does') -> Path:
    directory = parent / dirname
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'SKILL.md').write_text(
        f'---\nname: {dirname}\ndescription: {description}\n---\n\ndo the thing\n',
        encoding='utf-8',
    )
    return directory


class FakeResult:
    def __init__(self, path: str, *, success: bool = True, message: str = '') -> None:
        self.path = path
        self.success = success
        self.message = message


class FakeResponse:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results


class FakeRuntime:
    """A runtime that answers the home question and records what it was asked to upload."""

    def __init__(self, *, home: str = '/home/worker', fail: str = '') -> None:
        self._home = home
        self._fail = fail
        self.uploaded: list[Any] = []

    async def invoke(self, fn, **kwargs):
        del kwargs
        return self._home

    async def upload_file(self, *, files):
        self.uploaded = list(files)
        results = [
            FakeResult(item.path, success=not (self._fail and self._fail in item.path), message='denied')
            for item in files
        ]
        return FakeResponse(results)


# --- what the generator's half carries --------------------------------------


def test_the_directories_are_recorded_for_the_backend(tmp_path):
    write_skill(tmp_path, 'my-skill')

    config = with_worker_skills({}, [str(tmp_path / 'my-skill')])

    assert skill_dirs(config) == [str(tmp_path / 'my-skill')]


def test_the_generator_decides_nothing_about_placement(tmp_path):
    """No install script and no sandbox path: both are the backend's to own."""
    write_skill(tmp_path, 'my-skill')

    config = with_worker_skills({}, [str(tmp_path / 'my-skill')])

    assert 'pre_start_scripts' not in config
    assert not any('.claude' in str(value) for value in config.values())


def test_the_installed_names_are_recorded(tmp_path):
    write_skill(tmp_path, 'first', description='the first thing')
    write_skill(tmp_path, 'second', description='the second thing')

    config = with_worker_skills({}, [str(tmp_path / 'first'), str(tmp_path / 'second')])

    assert config[WORKER_COMMANDS_KEY] == {
        'first': 'the first thing',
        'second': 'the second thing',
    }


def test_the_argument_is_not_modified(tmp_path):
    write_skill(tmp_path, 'my-skill')
    original: dict = {'timeout': 60}

    with_worker_skills(original, [str(tmp_path / 'my-skill')])

    assert original == {'timeout': 60}


def test_an_existing_pre_start_script_is_kept(tmp_path):
    write_skill(tmp_path, 'my-skill')
    existing = {'interpreter': 'python', 'code': 'print("mine")'}

    config = with_worker_skills({'pre_start_scripts': [existing]}, [str(tmp_path / 'my-skill')])

    assert config['pre_start_scripts'] == [existing]


def test_a_directory_that_is_not_one_names_the_variable_that_set_it(tmp_path):
    with pytest.raises(ValueError, match='MY_SKILL_DIRS'):
        with_worker_skills({}, [str(tmp_path / 'absent')], env_name='MY_SKILL_DIRS')


# --- what the backend's half does -------------------------------------------


async def test_the_files_land_under_the_sandbox_home(tmp_path):
    """The home comes from the sandbox, so the upload path is absolute and needs no
    ``~`` expansion by a helper that does not do it."""
    write_skill(tmp_path, 'my-skill')
    config = with_worker_skills({}, [str(tmp_path / 'my-skill')])
    runtime = FakeRuntime(home='/home/worker')

    await place_worker_skills(runtime, config, skills_path='.claude/skills')

    assert [item.path for item in runtime.uploaded] == ['/home/worker/.claude/skills/my-skill/SKILL.md']


async def test_another_harness_gets_its_own_path(tmp_path):
    write_skill(tmp_path, 'my-skill')
    config = with_worker_skills({}, [str(tmp_path / 'my-skill')])
    runtime = FakeRuntime()

    await place_worker_skills(runtime, config, skills_path='.myagent/skills')

    assert runtime.uploaded[0].path == '/home/worker/.myagent/skills/my-skill/SKILL.md'


async def test_a_binary_file_travels_as_base64(tmp_path):
    """A skill may carry a font or an image. Reading every file as utf-8 raised on one."""
    directory = write_skill(tmp_path, 'my-skill')
    (directory / 'logo.png').write_bytes(b'\x89PNG\r\n\x1a\n\x00\xff')
    config = with_worker_skills({}, [str(directory)])
    runtime = FakeRuntime()

    await place_worker_skills(runtime, config, skills_path='.claude/skills')

    by_name = {Path(item.path).name: item for item in runtime.uploaded}
    assert by_name['logo.png'].encoding == 'base64'
    assert by_name['SKILL.md'].encoding == 'utf8'


async def test_a_harness_with_no_skills_path_places_nothing(tmp_path):
    write_skill(tmp_path, 'my-skill')
    config = with_worker_skills({}, [str(tmp_path / 'my-skill')])
    runtime = FakeRuntime()

    await place_worker_skills(runtime, config, skills_path='')

    assert runtime.uploaded == []


async def test_a_run_naming_no_skills_places_nothing():
    runtime = FakeRuntime()

    await place_worker_skills(runtime, {}, skills_path='.claude/skills')

    assert runtime.uploaded == []


async def test_a_failed_upload_raises(tmp_path):
    write_skill(tmp_path, 'my-skill')
    config = with_worker_skills({}, [str(tmp_path / 'my-skill')])
    runtime = FakeRuntime(fail='SKILL.md')

    with pytest.raises(RuntimeError, match='could not place'):
        await place_worker_skills(runtime, config, skills_path='.claude/skills')


# --- every backend answers the question -------------------------------------


def test_every_registered_backend_states_where_it_reads_skills():
    """A backend placing skills somewhere nothing reads them fails silently, so the
    attribute is read off the class rather than assumed."""
    for backend in backends():
        assert backend_class(backend).skills_path, backend


def test_the_config_key_is_the_one_the_backend_reads(tmp_path):
    write_skill(tmp_path, 'my-skill')

    config = with_worker_skills({}, [str(tmp_path / 'my-skill')])

    assert CONFIG_KEY in config
