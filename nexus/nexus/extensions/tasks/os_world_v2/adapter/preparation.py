"""V2 guest prerequisites and bounded long-running preparation transport."""

import asyncio
import json
import logging
import uuid

from types import SimpleNamespace

from nexus.utils.timeout import async_timeout


async def prepare_task048(runtime, password, timeout=1200):
    # Official task service is After=graphical.target. Do not bypass that ordering
    # or disable GitLab; wait until the image's boot dependency chain completes.
    async with async_timeout(timeout):
        while True:
            result = await runtime.run_command(command='systemctl is-active graphical.target', timeout=20)
            if result.return_code == 0 and (result.stdout or '').strip() == 'active':
                break
            await asyncio.sleep(5)
    probe = await runtime.run_command(
        command='python3 -c \'import os;s=os.statvfs("/dev/shm");print(s.f_frsize*s.f_blocks)\'',
        timeout=20,
    )
    if probe.return_code != 0:
        raise RuntimeError('Cannot inspect Task048 shared memory')
    if int((probe.stdout or '').strip()) < 1024**3:
        result = await runtime.run_command(
            command="sudo -S -k -p '' mount -o remount,size=1G /dev/shm",
            stdin=password + '\n',
            timeout=30,
        )
        if result.return_code != 0:
            raise RuntimeError('Task048 needs 1 GiB /dev/shm for verified archive, extracted build and browser')
        verified = await runtime.run_command(
            command='python3 -c \'import os;s=os.statvfs("/dev/shm");assert s.f_frsize*s.f_blocks>=1024**3\'',
            timeout=20,
        )
        if verified.return_code != 0:
            raise RuntimeError('Task048 shared-memory remount did not take effect')


async def run_preparation(runtime, **kwargs):
    """Execute once, detached from the guest HTTP deadline; poll status instead of restarting installers."""
    directory = '/tmp/nexus-v2-prepare-' + uuid.uuid4().hex
    timeout = float(kwargs.get('timeout', 900))
    spec = {'command': kwargs['command'], 'stdin': kwargs.get('stdin'), 'timeout': timeout}
    created = await runtime.run_command(command=f'umask 077; mkdir {directory}', timeout=20)
    if created.return_code != 0:
        raise RuntimeError('Cannot create private preparation directory')
    source = r"""
import base64,json,os,pathlib,subprocess,signal
root=pathlib.Path(__file__).parent
spec=json.loads((root/"spec").read_text())
p=subprocess.Popen(["bash","-lc",spec["command"]],stdin=subprocess.PIPE,
 stdout=(root/"stdout").open("wb"),stderr=(root/"stderr").open("wb"),start_new_session=True)
(root/"pid").write_text(str(p.pid))
try:
 p.communicate(input=spec["stdin"].encode() if spec["stdin"] is not None else None,timeout=spec["timeout"])
 rc=p.returncode
except subprocess.TimeoutExpired:
 os.killpg(p.pid,signal.SIGKILL);p.wait();rc=124
result={"return_code":rc}
for key in ("stdout","stderr"):
 with (root/key).open("rb") as f:
  f.seek(0,2);f.seek(max(0,f.tell()-32768));result[key]=f.read().decode(errors="replace")
(root/"result.tmp").write_text(json.dumps(result))
os.replace(root/"result.tmp",root/"result")
"""
    finished = False
    try:
        for name, data in [('spec', json.dumps(spec).encode()), ('run.py', source.encode())]:
            response = await runtime.setup_upload_bytes(directory + '/' + name, data, filename=name)
            if response.status_code != 200:
                raise RuntimeError('Cannot upload preparation program')
        launched = await runtime.run_command(
            command=f'nohup python3 {directory}/run.py >{directory}/runner.log 2>&1 </dev/null &',
            timeout=20,
        )
        if launched.return_code != 0:
            raise RuntimeError('Cannot launch preparation program')
        async with async_timeout(timeout + 60):
            while True:
                check = await runtime.run_command(
                    command=f'test -f {directory}/result && cat {directory}/result', timeout=20
                )
                if check.return_code == 0:
                    result = json.loads(check.stdout or '')
                    finished = True
                    return SimpleNamespace(**result)
                await asyncio.sleep(3)
    finally:
        if finished:
            await runtime.run_command(command=f'rm -rf {directory}', timeout=20)
        else:
            # The wrapper enforces the original command's deadline even if HTTP
            # transport is lost. Do not unlink state underneath a running process.
            logging.getLogger(__name__).warning(
                'Preparation did not return; guest wrapper retains its own %.0fs deadline',
                timeout,
            )
