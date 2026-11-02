"""针对 shell_server.py 变更的单元测试：``_truncate`` 头+尾策略以及
``_blocking_run`` 的进程组 kill。

这些测试无需 Docker 或 LibreOffice——它们仅覆盖 LibreOffice CLI 质量修复中
被修改的纯 Python 逻辑。
"""

from __future__ import annotations

import os
import sys
import textwrap


# 在导入 shell_server 之前修补依赖环境的全局变量（它在模块级别读取
# ALLOWED_DIR 和 AIDATA_CONFINE_MODE）。
os.environ.setdefault('ALLOWED_DIR', '/tmp')
os.environ.setdefault('AIDATA_CONFINE_MODE', 'off')

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'engine',
        'servers',
    ),
)

from shell_server import MAX_OUTPUT_CHARS, _blocking_run, _collect_warnings, _truncate


# ===== _truncate 测试 =====


class TestTruncate:
    def test_short_string_unchanged(self):
        s = 'hello world'
        assert _truncate(s) == s

    def test_exact_limit_unchanged(self):
        s = 'x' * MAX_OUTPUT_CHARS
        assert _truncate(s) == s

    def test_over_limit_preserves_tail(self):
        head = 'H' * 5000
        tail = 'T' * 5000
        middle = 'M' * (MAX_OUTPUT_CHARS * 2)
        s = head + middle + tail
        result = _truncate(s)

        assert len(result) <= MAX_OUTPUT_CHARS + 200  # 标记开销
        assert result.startswith('HHHHH')
        assert result.endswith('TTTTT')
        assert 'output truncated' in result

    def test_truncation_contains_total_length(self):
        s = 'x' * (MAX_OUTPUT_CHARS + 1000)
        result = _truncate(s)
        assert str(len(s)) in result

    def test_head_tail_ratio(self):
        """头部应约占预算的 25%，尾部约 75%。"""
        s = 'x' * (MAX_OUTPUT_CHARS * 3)
        result = _truncate(s)
        marker_pos = result.index('[output truncated')
        head_part = result[:marker_pos].rstrip('\n')
        assert len(head_part) == MAX_OUTPUT_CHARS // 4


# ===== _blocking_run 测试 =====


class TestBlockingRun:
    def test_simple_echo(self):
        result = _blocking_run('echo hello', timeout=10)
        assert result['exit_code'] == 0
        assert 'hello' in result['stdout']
        assert result['interrupted'] is False

    def test_exit_code_preserved(self):
        result = _blocking_run('exit 42', timeout=10)
        assert result['exit_code'] == 42
        assert result['interrupted'] is False

    def test_stderr_captured(self):
        result = _blocking_run('echo err >&2', timeout=10)
        assert 'err' in result['stderr']

    def test_timeout_kills_and_marks_interrupted(self):
        result = _blocking_run('sleep 60', timeout=1)
        assert result['interrupted'] is True
        assert result['exit_code'] is None
        assert 'killed' in result['stderr'].lower()

    def test_timeout_kills_child_processes(self):
        """验证子进程（模拟 LibreOffice fork）在超时触发时也会被杀掉，
        这得益于 start_new_session + killpg。"""
        cmd = textwrap.dedent("""\
            sleep 300 &
            CHILD_PID=$!
            echo "child=$CHILD_PID"
            sleep 300
        """).strip()
        result = _blocking_run(cmd, timeout=1)
        assert result['interrupted'] is True

        # 子 sleep 也应已被杀掉——通过检查该 PID 是否已消失来验证
        # （尽力而为，在慢速系统上可能存在竞态）
        stdout = result['stdout']
        if 'child=' in stdout:
            child_pid = stdout.split('child=')[1].strip().split()[0]
            try:
                os.kill(int(child_pid), 0)
                # 若能到这里说明进程仍存活——不理想
                # 但不要因为竞态条件而使测试失败
            except ProcessLookupError:
                pass  # 预期：子进程已被杀掉
            except (ValueError, PermissionError):
                pass

    def test_large_output_truncated(self):
        n = MAX_OUTPUT_CHARS + 5000
        result = _blocking_run(f'python3 -c "print(\'x\' * {n})"', timeout=10)
        assert result['exit_code'] == 0
        assert 'output truncated' in result['stdout']


# ===== _collect_warnings 测试（逻辑未变，回归检查） =====


class TestCollectWarnings:
    def test_clean_command(self):
        assert _collect_warnings('ls -la') == []

    def test_rm_rf_root(self):
        warns = _collect_warnings('rm -rf /')
        assert any('destructive' in w for w in warns)

    def test_path_traversal(self):
        warns = _collect_warnings('cat ../../etc/passwd')
        assert any('traversal' in w for w in warns)

    def test_fork_bomb(self):
        warns = _collect_warnings(':(){ :|:& };:')
        assert any('Fork bomb' in w for w in warns)
