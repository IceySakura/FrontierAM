"""针对 LibreOffice （soffice） confine 豁免的回归测试。

背景
----
``engine/confine/libconfine.c`` 通过 ``LD_PRELOAD`` 注入到每一次 shell-MCP
命令中， 并且在 ``execve`` 时**重新注入**， 以便子进程仍处于 confine 中。
但其对 libc 的插桩会确定性地让 LibreOffice 的多线程启动**触发 SIGSEGV**，
所以任何通过 ``skills/xlsx/scripts/recalc.py`` 重算 xlsx 的任务
（该脚本会驱动 ``soffice``） 都会失败并返回
``{"error": "LibreOffice macro not configured properly"}`` 。

修复方法是将 LibreOffice 启动链
（ ``soffice`` / ``soffice.bin`` / ``oosplash`` ） 从 ``LD_PRELOAD`` 的重注入
中豁免出去， 让它们不受 confine 约束运行。

这些测试在本地编译 ``libconfine.c`` （无需 Docker、 无需 LibreOffice） ，
并断言：

* exec **豁免**的 basename 时不再收到 ``LD_PRELOAD`` （不受 confine） ；
* exec **非豁免**的 basename 时仍会重新注入 ``LD_PRELOAD``
  （其他一切的 confine 保持完好） ；
* 路径 deny-list 仍会阻止被拒绝的读操作。

在没有 C 编译器或非 Linux 平台上会自动跳过。
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys

from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='LD_PRELOAD confine is Linux-only')

CONFINE_C = Path(__file__).resolve().parents[1] / 'engine' / 'confine' / 'libconfine.c'
_CC = shutil.which('cc') or shutil.which('gcc')


@pytest.fixture(scope='module')
def libconfine_so(tmp_path_factory) -> Path:
    """针对本地 glibc 编译 libconfine.c （与 compile.sh 保持一致） 。"""
    cc = _CC
    if cc is None:
        pytest.skip('no C compiler (cc/gcc) available to build libconfine.so')
    assert cc is not None  # 为类型检查器收窄类型； 上面的 skip() 已提前退出
    if not CONFINE_C.is_file():
        pytest.skip(f'libconfine.c not found at {CONFINE_C}')
    out = tmp_path_factory.mktemp('confine') / 'libconfine.so'
    res = subprocess.run(
        [cc, '-O2', '-fPIC', '-shared', '-o', str(out), str(CONFINE_C), '-ldl'],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        pytest.skip(f'libconfine.c failed to compile: {res.stderr[:400]}')
    return out


def _make_echo_script(dir_: Path, name: str) -> Path:
    """一个极简可执行脚本， 打印其启动时的 LD_PRELOAD 值。"""
    p = dir_ / name
    p.write_text('#!/bin/sh\nprintf "LD_PRELOAD=[%s]\\n" "$LD_PRELOAD"\n')
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


def _child_ld_preload(so: Path, target: Path) -> str:
    """在 libconfine 下运行 ``bash -c 'exec <target>'``（bash 本身受 confine），
    并返回被 exec 出来的子进程实际观察到的 LD_PRELOAD 值。"""
    env = {
        **os.environ,
        'LD_PRELOAD': str(so),
        'AIDATA_CONFINE_DENY': '/nonexistent-deny',
    }
    r = subprocess.run(
        ['bash', '-c', f'exec "{target}"'],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    m = re.search(r'LD_PRELOAD=\[(.*)\]', r.stdout)
    assert m is not None, f'child did not report LD_PRELOAD; stdout={r.stdout!r} stderr={r.stderr!r} rc={r.returncode}'
    return m.group(1)


@pytest.mark.parametrize('name', ['soffice', 'soffice.bin', 'oosplash'])
def test_soffice_chain_runs_unconfined(libconfine_so: Path, tmp_path: Path, name: str) -> None:
    """LibreOffice 启动链 exec 时必须**不带** LD_PRELOAD， 否则我们的
    libc 钩子会让它崩溃 （segfault -> 'macro not configured properly'） 。"""
    target = _make_echo_script(tmp_path, name)
    val = _child_ld_preload(libconfine_so, target)
    assert val == '', (
        f'{name!r} should run UNCONFINED (no LD_PRELOAD), but the child saw '
        f'LD_PRELOAD={val!r}. The soffice confine exemption regressed — '
        f'LibreOffice recalc will SIGSEGV again.'
    )


def test_nonexempt_exec_still_reinjects(libconfine_so: Path, tmp_path: Path) -> None:
    """任何**未**在允许列表中的 exec， 仍必须在 execve 时重新注入 LD_PRELOAD ——
    也就是说该豁免不能削弱整体 confinement。"""
    target = _make_echo_script(tmp_path, 'definitely-not-soffice')
    val = _child_ld_preload(libconfine_so, target)
    msg = f'non-exempt exec must keep LD_PRELOAD re-injected for confinement, but the child saw LD_PRELOAD={val!r}'
    assert val, msg
    assert 'libconfine' in val, msg


def test_deny_list_still_enforced(libconfine_so: Path, tmp_path: Path) -> None:
    """健壮性检查： 路径 deny-list 仍会阻止 confined 命令的读操作。"""
    secret_dir = tmp_path / 'secret'
    secret_dir.mkdir()
    secret = secret_dir / 'x'
    secret.write_text('TOPSECRET\n')
    env = {
        **os.environ,
        'LD_PRELOAD': str(libconfine_so),
        'AIDATA_CONFINE_DENY': str(secret_dir),
    }
    r = subprocess.run(
        ['bash', '-c', f'cat "{secret}"'],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert 'TOPSECRET' not in r.stdout, f'deny-list bypass: confined `cat` leaked the secret. stdout={r.stdout!r}'
    assert r.returncode != 0, 'denied read should fail (non-zero exit)'
