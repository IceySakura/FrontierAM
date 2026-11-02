import hashlib
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import pytest


class _E2EHelper:
    TRUE_VALUES = {'1', 'true', 'yes', 'on'}
    RUNTIME_DIFF_PATHS = ['deployment', 'nexus', 'pyproject.toml', 'Makefile']

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._built_tags: set[str] = set()

    def ensure_runtime_image(self, config_path: Path) -> Path:
        base_tag = self._runtime_image_from_config(config_path)
        state_id = self._git_state_id()
        image_tag = self._temp_image_tag(base_tag, state_id)
        if image_tag in self._built_tags:
            return self._write_config_with_image(config_path, image_tag)

        force_rebuild = self._env_flag('NEXUS_E2E_RUNTIME_IMAGE_FORCE_REBUILD', default=False)
        skip_build = self._env_flag('NEXUS_E2E_RUNTIME_IMAGE_SKIP_BUILD', default=False)
        safe_name = self._tag_to_safe_name(image_tag)
        lock_path = self._repo_root / 'workspace' / f'.runtime_image_{safe_name}.lock'
        temp_prefix = self._temp_tag_prefix(base_tag)

        with self._file_lock(lock_path):
            if not force_rebuild and self._runtime_image_exists(image_tag):
                self._cleanup_temp_images(temp_prefix, image_tag)
                self._built_tags.add(image_tag)
                return self._write_config_with_image(config_path, image_tag)

            if skip_build:
                pytest.skip(
                    f'runtime image {image_tag} not found; auto-build disabled via NEXUS_E2E_RUNTIME_IMAGE_SKIP_BUILD=1'
                )

            self._build_runtime_image(image_tag)

            if not self._runtime_image_exists(image_tag):
                pytest.fail(f'runtime image {image_tag} not found after build')
            self._cleanup_temp_images(temp_prefix, image_tag)

        self._built_tags.add(image_tag)
        return self._write_config_with_image(config_path, image_tag)

    def build_env(self, tool_name: str | None = None) -> dict[str, str]:
        env = os.environ.copy()
        if tool_name:
            env['CODE_INTERPRETER_TOOL'] = tool_name
        llm_no_proxy = os.environ.get(
            'NEXUS_LLM_NO_PROXY',
            'api.model-eval.woa.com,localhost,127.0.0.1',
        )
        existing_no_proxy = env.get('NO_PROXY') or env.get('no_proxy')
        merged_no_proxy = self._merge_no_proxy(existing_no_proxy, llm_no_proxy)
        env['NO_PROXY'] = merged_no_proxy
        env['no_proxy'] = merged_no_proxy
        for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'DOCKER_CONFIG'):
            env.pop(key, None)
        return env

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in _E2EHelper.TRUE_VALUES

    @staticmethod
    def _runtime_image_from_config(config_path: Path) -> str:
        for line in config_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith('runtime_image:'):
                return stripped.split(':', 1)[1].strip().strip('"').strip("'")
        return 'nexus-runtime:dev'

    @staticmethod
    def _merge_no_proxy(existing: str | None, extra: str) -> str:
        parts = []
        if existing:
            parts.extend(p.strip() for p in existing.split(',') if p.strip())
        parts.extend(p.strip() for p in extra.split(',') if p.strip())
        seen = set()
        merged = []
        for item in parts:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return ','.join(merged)

    @staticmethod
    def _runtime_image_exists(image_tag: str) -> bool:
        inspect = subprocess.run(
            ['docker', 'image', 'inspect', image_tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return inspect.returncode == 0

    def _git_cmd(self, *args: str, paths: list[str] | None = None) -> str | None:
        cmd = ['git', '-C', str(self._repo_root), *args]
        if paths:
            cmd.extend(['--', *paths])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    def _git_state_id(self) -> str:
        head = self._git_cmd('rev-parse', 'HEAD')
        if not head:
            return 'unknown'
        status = self._git_cmd('status', '--porcelain', paths=_E2EHelper.RUNTIME_DIFF_PATHS)
        if status is None:
            return 'unknown'
        if status:
            diff = self._git_cmd('diff', '--binary', paths=_E2EHelper.RUNTIME_DIFF_PATHS) or ''
            diff_cached = (
                self._git_cmd(
                    'diff',
                    '--cached',
                    '--binary',
                    paths=_E2EHelper.RUNTIME_DIFF_PATHS,
                )
                or ''
            )
            digest = hashlib.sha256((diff + diff_cached).encode('utf-8')).hexdigest()[:12]
            return f'{head}-dirty-{digest}'
        return head

    @staticmethod
    def _temp_image_tag(base_tag: str, state_id: str) -> str:
        if ':' in base_tag:
            repo, tag = base_tag.split(':', 1)
        else:
            repo, tag = base_tag, 'latest'

        if state_id == 'unknown':
            suffix = 'unknown'
        elif '-dirty-' in state_id:
            head, dirty_hash = state_id.split('-dirty-', 1)
            suffix = f'{head[:7]}-dirty-{dirty_hash[:7]}'
        else:
            suffix = state_id[:7]

        return f'{repo}:{tag}-e2e-tmp-{suffix}'

    @staticmethod
    def _temp_tag_prefix(base_tag: str) -> str:
        if ':' in base_tag:
            repo, tag = base_tag.split(':', 1)
        else:
            repo, tag = base_tag, 'latest'
        return f'{repo}:{tag}-e2e-tmp-'

    @staticmethod
    def _list_temp_images(prefix: str) -> list[str]:
        result = subprocess.run(
            ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip().startswith(prefix)]

    def _cleanup_temp_images(self, prefix: str, keep: str) -> None:
        for tag in self._list_temp_images(prefix):
            if tag == keep:
                continue
            subprocess.run(['docker', 'rmi', '-f', tag], check=False)

    def _build_runtime_image(self, image_tag: str) -> None:
        result = subprocess.run(
            ['make', 'docker-runtime'],
            cwd=self._repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = result.stdout + '\n' + result.stderr
            pytest.fail(f'runtime image build failed via Makefile for {image_tag}:\n{details}')

        source_tag = 'nexus-runtime:latest'
        if not self._runtime_image_exists(source_tag) and self._runtime_image_exists('nexus-runtime'):
            source_tag = 'nexus-runtime'

        if image_tag != source_tag:
            tag_result = subprocess.run(
                ['docker', 'tag', source_tag, image_tag],
                cwd=self._repo_root,
                capture_output=True,
                text=True,
            )
            if tag_result.returncode != 0:
                details = tag_result.stdout + '\n' + tag_result.stderr
                pytest.fail(f'runtime image tag failed for {image_tag}:\n{details}')

    def _write_config_with_image(self, config_path: Path, image_tag: str) -> Path:
        output_dir = self._repo_root / 'workspace' / 'e2e'
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = image_tag.replace('/', '_').replace(':', '_')
        output_path = output_dir / f'{config_path.stem}.{safe_tag}.yaml'

        updated = []
        for line in config_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith('image:'):
                prefix = line.split('image:', 1)[0]
                updated.append(f'{prefix}image: "{image_tag}"')
                continue
            if stripped.startswith('runtime_image:'):
                prefix = line.split('runtime_image:', 1)[0]
                updated.append(f'{prefix}runtime_image: "{image_tag}"')
                continue
            updated.append(line)

        output_path.write_text('\n'.join(updated) + '\n')
        return output_path

    @staticmethod
    def _tag_to_safe_name(image_tag: str) -> str:
        return image_tag.replace('/', '_').replace(':', '_')

    @staticmethod
    @contextmanager
    def _file_lock(lock_path: Path) -> Iterator[None]:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, 'w') as lock_file:
            try:
                import fcntl  # linux/unix only

                fcntl.flock(lock_file, fcntl.LOCK_EX)
            except Exception:
                yield
                return
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


class _PerfLogger:
    _DURATION_PAIRS = {
        'runner_to_tool_call_s': ('runner_start', 'tool_call'),
        'runtime_acquire_s': ('runtime_acquire', 'runtime_ready'),
        'container_create_s': ('container_create', 'container_created'),
        'container_start_s': ('container_start', 'runtime_ready'),
        'tool_exec_s': ('call_tool_exec', 'call_tool_done'),
        'tool_call_to_output_s': ('tool_call', 'tool_output'),
        'task_finish_s': ('task_completed', 'runtime_released'),
        'log_total_s': ('runner_start', 'runtime_released'),
    }
    _SEARCH_TOOLS = (
        'search_stateful/search',
        'search_stateful/open',
        'search_stateful/find',
        'google_search',
        'hunyuan_search',
    )

    def __init__(self, repo_root: Path, tool_name: str, image_tag: str) -> None:
        self._repo_root = repo_root
        self._tool_name = tool_name
        self._image_tag = image_tag

    def write(
        self,
        combined: str,
        image_elapsed: float,
        env_elapsed: float,
        run_elapsed: float,
        write_elapsed: float,
        total_elapsed: float,
        returncode: int,
    ) -> None:
        events, counters = self._parse_events(combined)
        durations = self._collect_durations(events)
        llm_stats = self._llm_stats(events)
        search_stats = self._search_stats(events)
        perf_log = self._repo_root / 'workspace' / 'research' / 'e2e_perf.log'
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        fields = [
            f'time={timestamp}',
            f'tool={self._tool_name}',
            f'image={self._image_tag}',
            f'ensure_image_s={image_elapsed:.2f}',
            f'env_prep_s={env_elapsed:.2f}',
            f'run_single_s={run_elapsed:.2f}',
            f'log_write_s={write_elapsed:.2f}',
            f'total_s={total_elapsed:.2f}',
            f'output_bytes={len(combined.encode("utf-8"))}',
            f'tool_call_count={counters["tool_call_count"]}',
            f'model_output_count={counters["model_output_count"]}',
            f'agent_loop_end_count={counters["agent_loop_end_count"]}',
            *[f'{key}={value:.2f}' for key, value in durations.items()],
            *[f'{key}={value:.2f}' for key, value in llm_stats.items()],
            *[f'{key}={value:.2f}' for key, value in search_stats.items()],
            f'rc={returncode}',
        ]
        self._append(perf_log, ' '.join(fields))

    def _event_patterns(self) -> dict[str, tuple[str, ...]]:
        return {
            'runner_start': ('Starting runner for task',),
            'task_execute': ('Executing task:',),
            'tool_call': (f'Tool call: {self._tool_name}',),
            'runtime_acquire': ("Acquiring runtime 'main'",),
            'image_skip_build': ('already exists, skipping build',),
            'container_create': ('Creating runtime container',),
            'container_created': ('Runtime container', 'created'),
            'container_start': ('Starting container',),
            'runtime_ready': ('Runtime main ready',),
            'socket_connected': ('socketio: connected',),
            'call_tool_received': ('call_tool: received request',),
            'call_tool_exec': (f"call_tool: executing tool '{self._tool_name}'",),
            'call_tool_done': (f"call_tool: '{self._tool_name}' executed successfully",),
            'tool_output': ('Tool output:',),
            'model_output': ('Model output:',),
            'task_completed': ('Task completed:',),
            'runtime_stop': ('Stopping container',),
            'runtime_released': ('Runtime main released',),
        }

    def _parse_events(self, log_text: str) -> tuple[dict[str, list[datetime]], dict[str, int]]:
        patterns = self._event_patterns()
        events: dict[str, list[datetime]] = {}
        tool_call_count = 0
        model_output_count = 0
        agent_loop_end_count = 0
        for line in log_text.splitlines():
            parts = line.split(' | ', 3)
            if len(parts) < 4:
                continue
            ts_str, _, _, message = parts
            try:
                ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
            if 'Tool call:' in message:
                tool_call_count += 1
            if 'Model output:' in message:
                model_output_count += 1
            if 'No tool calls in response, stopping agent loop' in message:
                agent_loop_end_count += 1
            for name, needles in patterns.items():
                if all(needle in message for needle in needles):
                    events.setdefault(name, []).append(ts)
                    break
            self._capture_search_events(message, ts, events)
        counters = {
            'tool_call_count': tool_call_count,
            'model_output_count': model_output_count,
            'agent_loop_end_count': agent_loop_end_count,
        }
        return events, counters

    def _capture_search_events(self, message: str, ts: datetime, events: dict[str, list[datetime]]) -> None:
        if "call_tool: executing tool '" in message:
            tool = message.split("call_tool: executing tool '", 1)[1].split("'", 1)[0]
            if tool in self._SEARCH_TOOLS:
                events.setdefault(f'search_exec::{tool}', []).append(ts)
        if "call_tool: '" in message and ' executed successfully' in message:
            tool = message.split("call_tool: '", 1)[1].split("'", 1)[0]
            if tool in self._SEARCH_TOOLS:
                events.setdefault(f'search_done::{tool}', []).append(ts)
        if 'Tool call:' in message:
            tool = message.split('Tool call:', 1)[1].split('(', 1)[0].strip()
            if tool in self._SEARCH_TOOLS:
                events.setdefault('search_tool_call', []).append(ts)

    @classmethod
    def _collect_durations(cls, events: dict[str, list[datetime]]) -> dict[str, float]:
        durations: dict[str, float] = {}
        for key, (start, end) in cls._DURATION_PAIRS.items():
            if start not in events or end not in events:
                continue
            durations[key] = (events[end][0] - events[start][0]).total_seconds()
        return durations

    def _llm_stats(self, events: dict[str, list[datetime]]) -> dict[str, float]:
        model_outputs = events.get('model_output', [])
        tool_outputs = events.get('tool_output', [])
        llm_total = 0.0
        paired = 0
        tool_index = 0
        for model_ts in model_outputs:
            while tool_index < len(tool_outputs) and tool_outputs[tool_index] <= model_ts:
                tool_index += 1
            if tool_index == 0:
                continue
            tool_ts = tool_outputs[tool_index - 1]
            delta = (model_ts - tool_ts).total_seconds()
            if delta >= 0:
                llm_total += delta
                paired += 1
        stats: dict[str, float] = {
            'llm_response_count': float(len(model_outputs)),
            'llm_response_total_s': llm_total,
        }
        if paired:
            stats['llm_response_avg_s'] = llm_total / paired
        if 'task_execute' in events and model_outputs:
            stats['llm_first_response_s'] = (model_outputs[0] - events['task_execute'][0]).total_seconds()
        return stats

    def _search_stats(self, events: dict[str, list[datetime]]) -> dict[str, float]:
        call_count = float(len(events.get('search_tool_call', [])))
        total = 0.0
        count = 0
        for tool in self._SEARCH_TOOLS:
            exec_times = events.get(f'search_exec::{tool}', [])
            done_times = events.get(f'search_done::{tool}', [])
            for i in range(min(len(exec_times), len(done_times))):
                total += (done_times[i] - exec_times[i]).total_seconds()
                count += 1
        stats: dict[str, float] = {
            'search_tool_call_count': call_count,
            'search_exec_count': float(count),
            'search_exec_total_s': total,
        }
        if count:
            stats['search_exec_avg_s'] = total / count
        return stats

    @staticmethod
    def _append(perf_log: Path, message: str) -> None:
        perf_log.parent.mkdir(parents=True, exist_ok=True)
        with perf_log.open('a') as handle:
            handle.write(message + '\n')


@pytest.fixture(scope='session')
def _e2e_helper() -> _E2EHelper:
    repo_root = Path(__file__).resolve().parents[2]
    return _E2EHelper(repo_root)


@pytest.fixture(scope='session')
def ensure_runtime_image(_e2e_helper: _E2EHelper) -> Callable[[Path], Path]:
    return _e2e_helper.ensure_runtime_image


@pytest.fixture
def nexus_run_env(_e2e_helper: _E2EHelper) -> Callable[[str | None], dict[str, str]]:
    return _e2e_helper.build_env


@pytest.fixture
def perf_logger_factory() -> type['_PerfLogger']:
    return _PerfLogger
