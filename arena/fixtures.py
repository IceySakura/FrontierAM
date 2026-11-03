"""Local protocol fixture: drives real Codex through the offline tool without any paid model call."""

from __future__ import annotations

import base64

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fixture_command(task: str) -> str:
    source = ROOT / 'fixtures' / task
    files = {
        str(p.relative_to(source)): base64.b64encode(p.read_bytes()).decode() for p in source.rglob('*') if p.is_file()
    }
    return (
        "python3 - <<'PY'\nimport base64,os,socket\nfrom pathlib import Path\n"
        "assert os.getuid()==1000 and not Path('/tests').exists()\n"
        "assert not Path('/var/run/docker.sock').exists()\n"
        "assert not any('API_KEY' in k or 'AIHUB' in k or 'TAIHU' in k for k in os.environ)\n"
        "s=socket.socket();s.settimeout(1);assert s.connect_ex(('1.1.1.1',53))!=0;s.close()\n"
        "assert len(os.sched_getaffinity(0))==1\n"
        "root=Path('/workspace/submission');root.mkdir(exist_ok=True)\n"
        f'files={files!r}\n'
        'for name,value in files.items():\n'
        ' p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value));p.chmod(0o755)\n'
        "print('FIXTURE_WRITE_OK\\nFIXTURE_ISOLATION_OK')\nPY\n"
    )
