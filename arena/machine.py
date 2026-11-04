"""Machine slot layout and kernel-enforced CPU isolation.

A cgroup v2 cpuset partition (``frontieram.slice``) owns every benchmark CPU, so the kernel keeps all other processes
- system services, the default dockerd, monitoring agents, editors - on the housekeeping CPUs.

Every slot is one whole physical core: the slot's container runs on one hardware thread and only that slot's host-side
process (the grading timer or the development controller) runs on the sibling. Attached container stdio is on the
exponential timing path, and a shared relay was measured to dominate concurrent interference (results/CONCURRENCY.md),
so each grading slot has its own containerd + dockerd on its sibling thread; development uses the default daemon.
Grading cores default to their own last-level cache domain.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

from pathlib import Path


CONFIG = Path(os.environ.get('FRONTIERAM_MACHINE', '/etc/frontieram/machine.json'))
STATE = Path(os.environ.get('FRONTIERAM_STATE', '/run/frontieram'))
CGROUP = Path('/sys/fs/cgroup')
SYSTEMD = Path('/etc/systemd/system')
PARTITION_SLICE = 'frontieram.slice'
SLICES = {'dev': 'frontieram-dev.slice', 'grade': 'frontieram-grade.slice', 'host': 'frontieram-host.slice'}
DAEMON_SLICE = 'frontieram-daemon.slice'
PARTITION_UNIT = 'frontieram-partition.service'
# The single shared grading daemon of schema 2; removed on setup because every slot's timed stdio crossed it.
LEGACY_GRADE_UNITS = ('frontieram-grade-docker.service', 'frontieram-grade-containerd.service')
DEV_DOCKER_HOST = 'unix:///var/run/docker.sock'
GRADE_RUN = Path('/run/frontieram-grade')
GRADE_DATA = Path('/data/frontieram-grade')


def grade_daemon(slot_id: str) -> dict:
    """One private containerd + dockerd per grading slot, so no two slots share a stdio relay."""
    run = GRADE_RUN / slot_id
    return {
        'containerd_unit': f'frontieram-grade-containerd-{slot_id}.service',
        'docker_unit': f'frontieram-grade-docker-{slot_id}.service',
        'containerd_socket': str(run / 'containerd.sock'),
        'docker_host': f'unix://{run}/docker.sock',
        'run': str(run),
        'data': str(GRADE_DATA / slot_id),
    }


def parse_cpus(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.strip().split(','):
        if not part:
            continue
        low, _, high = part.partition('-')
        cpus.extend(range(int(low), int(high or low) + 1))
    return sorted(set(cpus))


def format_cpus(cpus) -> str:
    cpus = sorted(set(cpus))
    ranges, start = [], None
    for index, cpu in enumerate(cpus):
        if start is None:
            start = cpu
        if index + 1 == len(cpus) or cpus[index + 1] != cpu + 1:
            ranges.append(str(start) if start == cpu else f'{start}-{cpu}')
            start = None
    return ','.join(ranges)


def topology() -> list[dict]:
    """Physical cores in CPU order, each with its hardware threads and last-level cache domain."""
    cores: dict[tuple[int, int], dict] = {}
    for path in sorted(Path('/sys/devices/system/cpu').glob('cpu[0-9]*'), key=lambda p: int(p.name[3:])):
        if (path / 'online').exists() and (path / 'online').read_text().strip() == '0':
            continue
        top = path / 'topology'
        key = (int((top / 'physical_package_id').read_text()), int((top / 'core_id').read_text()))
        l3 = path / 'cache/index3/shared_cpu_list'
        core = cores.setdefault(key, {'cpus': [], 'l3': l3.read_text().strip() if l3.exists() else 'unknown'})
        core['cpus'].append(int(path.name[3:]))
    return sorted(cores.values(), key=lambda core: core['cpus'][0])


def mem_total() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemTotal:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemTotal unavailable')


def _slot(kind: str, core: dict) -> dict:
    if len(core['cpus']) < 2:
        raise ValueError('Each slot needs a physical core with an SMT sibling for its host-side process')
    cpu, host_cpu = core['cpus'][:2]
    slot = {'id': f'{kind}-{cpu:02d}', 'kind': kind, 'cpu': cpu, 'host_cpu': host_cpu, 'l3': core['l3']}
    if kind == 'grade':
        slot['daemon'] = grade_daemon(slot['id'])
        slot['docker_host'] = slot['daemon']['docker_host']
    else:
        slot['docker_host'] = DEV_DOCKER_HOST
    return slot


def plan(
    *,
    housekeeping_cores: int = 2,
    grade_cores: int = 4,
    host_reserve_gib: float = 6.0,
    grade_placement: str = 'packed',
) -> dict:
    """packed: grade cores share the last L3 domains; spread: at most one grade core per L3 domain."""
    cores = topology()
    reserved = housekeeping_cores + grade_cores
    if min(housekeeping_cores, grade_cores) < 1 or reserved >= len(cores):
        raise ValueError(f'{len(cores)} cores cannot hold that split and still leave development cores')
    housekeeping = cores[:housekeeping_cores]
    rest = cores[housekeeping_cores:]
    if grade_placement == 'packed':
        grade = rest[len(rest) - grade_cores :]
    elif grade_placement == 'spread':
        domains: dict[str, list[dict]] = {}
        for core in rest:
            domains.setdefault(core['l3'], []).append(core)
        if grade_cores > len(domains):
            raise ValueError(f'spread placement needs one L3 domain per grade core; only {len(domains)} available')
        grade = [members[-1] for members in list(domains.values())[-grade_cores:]]
    else:
        raise ValueError('grade_placement must be packed or spread')
    dev = [core for core in rest if core not in grade]
    slots = [_slot('dev', core) for core in dev] + [_slot('grade', core) for core in grade]
    layout = {
        'schema_version': 3,
        'grade_placement': grade_placement,
        'housekeeping_cpus': format_cpus(cpu for core in housekeeping for cpu in core['cpus']),
        'partition_cpus': format_cpus(cpu for core in dev + grade for cpu in core['cpus']),
        'docker_hosts': {'dev': DEV_DOCKER_HOST, 'grade': [s['docker_host'] for s in slots if s['kind'] == 'grade']},
        'memory_budget_bytes': int(mem_total() - host_reserve_gib * 2**30),
        'slices': {'partition': PARTITION_SLICE, 'daemon': DAEMON_SLICE, **SLICES},
        'slots': slots,
        'topology': cores,
    }
    layout['layout_sha256'] = layout_hash(layout)
    return layout


def layout_hash(layout: dict) -> str:
    keys = ('housekeeping_cpus', 'partition_cpus', 'docker_hosts', 'slots', 'topology')
    return hashlib.sha256(json.dumps({k: layout[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def all_docker_hosts(layout: dict) -> list[str]:
    return [layout['docker_hosts']['dev'], *layout['docker_hosts']['grade']]


def load(path: Path = CONFIG) -> dict:
    if not path.exists():
        raise RuntimeError(f'No machine layout at {path}; run ./bench host setup first')
    layout = json.loads(path.read_text())
    if layout.get('schema_version') != 3:
        raise RuntimeError(f'{path} predates per-slot grading daemons; rerun ./bench host setup')
    if layout.get('layout_sha256') != layout_hash(layout):
        raise RuntimeError(f'{path} was edited without refreshing layout_sha256; rerun ./bench host setup')
    partition = set(parse_cpus(layout['partition_cpus']))
    seen: set[int] = set()
    for slot in layout['slots']:
        cpus = {slot['cpu'], slot['host_cpu']}
        if not cpus <= partition or cpus & seen:
            raise RuntimeError(f'Slot {slot["id"]} overlaps another slot or leaves the partition')
        seen |= cpus
    return layout


def _systemctl(*args: str) -> None:
    result = subprocess.run(['systemctl', *args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f'systemctl {" ".join(args)} failed: {result.stderr.strip()}')


ISOLATE_SCRIPT = Path('/etc/frontieram/isolate.sh')
ISOLATE_UNIT = 'frontieram-isolate.service'
ISOLATE_TIMER = 'frontieram-isolate.timer'
SYSTEMD_TOP_LEVEL = ('system.slice', 'user.slice')


def _isolate_script(layout: dict) -> str:
    managed = ' '.join([PARTITION_SLICE, *SYSTEMD_TOP_LEVEL])
    return (
        '#!/bin/sh\n# Generated by ./bench host setup. systemd slices carry AllowedCPUs; this covers the rest.\n'
        'set -u\n'
        'for dir in /sys/fs/cgroup/*/; do\n'
        '  name=$(basename "$dir")\n'
        f'  case " {managed} " in *" $name "*) continue ;; esac\n'
        '  [ -f "$dir/cpuset.cpus" ] || continue\n'
        f'  [ "$(cat "$dir/cpuset.cpus")" = "{layout["housekeeping_cpus"]}" ] && continue\n'
        f'  echo {layout["housekeeping_cpus"]} > "$dir/cpuset.cpus" || echo "cannot confine $name" >&2\n'
        'done\n'
    )


def _grade_containerd_unit(slot: dict) -> str:
    """A private containerd: without --containerd, dockerd silently attaches to the system instance."""
    daemon = slot['daemon']
    config = CONFIG.parent / slot['id'] / 'containerd.toml'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        'version = 2\n'
        f'root = "{daemon["data"]}/containerd"\n'
        f'state = "{daemon["run"]}/containerd-state"\n'
        'disabled_plugins = ["io.containerd.grpc.v1.cri"]\n\n'
        f'[grpc]\n  address = "{daemon["containerd_socket"]}"\n\n'
        f'[ttrpc]\n  address = "{daemon["containerd_socket"]}.ttrpc"\n\n'
        '[debug]\n  address = ""\n  level = "warn"\n\n'
        '[metrics]\n  address = ""\n'
    )
    return (
        f'[Unit]\nDescription=FrontierAM containerd for {slot["id"]}; its shims relay that slot\'s timed stdio\n'
        f'After={PARTITION_SLICE}\nRequires={PARTITION_SLICE}\n\n'
        '[Service]\nType=notify\n'
        f'Slice={DAEMON_SLICE}\nAllowedCPUs={slot["host_cpu"]}\n'
        f'ExecStartPre=/bin/mkdir -p {daemon["run"]}\n'
        f'ExecStart=/usr/bin/containerd --config {config}\n'
        'Restart=always\nDelegate=yes\nKillMode=process\nLimitNOFILE=infinity\nLimitNPROC=infinity\n'
        'TasksMax=infinity\nOOMScoreAdjust=-999\n\n[Install]\nWantedBy=multi-user.target\n'
    )


def _grade_docker_unit(slot: dict) -> str:
    daemon = slot['daemon']
    config = CONFIG.parent / slot['id'] / 'dockerd.json'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({}) + '\n')
    arguments = [
        '/usr/bin/dockerd',
        f'--config-file={config}',
        f'--containerd={daemon["containerd_socket"]}',
        f'--host={daemon["docker_host"]}',
        f'--data-root={daemon["data"]}/docker',
        f'--exec-root={daemon["run"]}/exec',
        f'--pidfile={daemon["run"]}/dockerd.pid',
        '--bridge=none',
        '--iptables=false',
        '--ip6tables=false',
        '--ip-masq=false',
        '--exec-opt=native.cgroupdriver=systemd',
        '--log-level=warn',
    ]
    return (
        f'[Unit]\nDescription=FrontierAM Docker daemon serving only {slot["id"]}\n'
        f'After=network-online.target {PARTITION_SLICE} {daemon["containerd_unit"]}\n'
        f'Requires={PARTITION_SLICE} {daemon["containerd_unit"]}\n\n'
        '[Service]\nType=notify\n'
        f'Slice={DAEMON_SLICE}\nAllowedCPUs={slot["host_cpu"]}\n'
        f'ExecStartPre=/bin/mkdir -p {daemon["run"]}\n'
        f'ExecStart={" ".join(arguments)}\n'
        'ExecReload=/bin/kill -s HUP $MAINPID\nTimeoutStartSec=120\nRestart=on-failure\n'
        'LimitNOFILE=infinity\nLimitNPROC=infinity\nTasksMax=infinity\nDelegate=yes\nKillMode=process\n'
        'OOMScoreAdjust=-500\n\n[Install]\nWantedBy=multi-user.target\n'
    )


def grade_slots(layout: dict) -> list[dict]:
    return [slot for slot in layout['slots'] if slot['kind'] == 'grade']


def install(layout: dict, path: Path = CONFIG) -> dict:
    """Persist the layout and kernel isolation; safe to rerun. Requires root, never restarts the default dockerd."""
    if os.geteuid() != 0:
        raise PermissionError('Host setup must run as root')
    if live_leases_present():
        raise RuntimeError('Slots are leased; finish running work before changing the machine layout')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layout, indent=2) + '\n')
    (SYSTEMD / PARTITION_SLICE).write_text(
        '[Unit]\nDescription=FrontierAM isolated benchmark CPUs\n\n'
        f'[Slice]\nAllowedCPUs={layout["partition_cpus"]}\n'
    )
    # A kernel cpuset partition would be simpler, but it removes the CPUs from the root cgroup and Docker validates
    # --cpuset-cpus against the root, so every top-level sibling is confined to housekeeping instead.
    partition_file = CGROUP / PARTITION_SLICE / 'cpuset.cpus.partition'
    if partition_file.exists() and partition_file.read_text().startswith('root'):
        partition_file.write_text('member')
    if (SYSTEMD / PARTITION_UNIT).exists():
        subprocess.run(['systemctl', 'disable', '--now', PARTITION_UNIT], capture_output=True)
        (SYSTEMD / PARTITION_UNIT).unlink()
    ISOLATE_SCRIPT.write_text(_isolate_script(layout))
    ISOLATE_SCRIPT.chmod(0o755)
    (SYSTEMD / ISOLATE_UNIT).write_text(
        '[Unit]\nDescription=Confine every cgroup outside frontieram.slice to the housekeeping CPUs\n'
        f'After={PARTITION_SLICE}\nRequires={PARTITION_SLICE}\n\n'
        f'[Service]\nType=oneshot\nExecStart={ISOLATE_SCRIPT}\n\n[Install]\nWantedBy=multi-user.target\n'
    )
    (SYSTEMD / ISOLATE_TIMER).write_text(
        '[Unit]\nDescription=Re-confine new top-level cgroups (monitoring agents create their own)\n\n'
        '[Timer]\nOnBootSec=10\nOnUnitActiveSec=30\nAccuracySec=1\n\n[Install]\nWantedBy=timers.target\n'
    )
    for unit in LEGACY_GRADE_UNITS:
        if (SYSTEMD / unit).exists():
            subprocess.run(['systemctl', 'disable', '--now', unit], capture_output=True)
            (SYSTEMD / unit).unlink()
    wanted = set()
    for slot in grade_slots(layout):
        daemon = slot['daemon']
        (SYSTEMD / daemon['containerd_unit']).write_text(_grade_containerd_unit(slot))
        (SYSTEMD / daemon['docker_unit']).write_text(_grade_docker_unit(slot))
        wanted |= {daemon['containerd_unit'], daemon['docker_unit']}
    # Daemons of grade slots that no longer exist in this layout.
    for path in SYSTEMD.glob('frontieram-grade-*-*.service'):
        if path.name not in wanted:
            subprocess.run(['systemctl', 'disable', '--now', path.name], capture_output=True)
            path.unlink()
    _systemctl('daemon-reload')
    _systemctl('start', PARTITION_SLICE)
    _systemctl('set-property', '--runtime', PARTITION_SLICE, f'AllowedCPUs={layout["partition_cpus"]}')
    for unit in SYSTEMD_TOP_LEVEL:
        _systemctl('set-property', unit, f'AllowedCPUs={layout["housekeeping_cpus"]}')
    _systemctl('enable', ISOLATE_UNIT, ISOLATE_TIMER, *sorted(wanted))
    _systemctl('start', ISOLATE_UNIT)
    _systemctl('restart', ISOLATE_TIMER)
    for slot in grade_slots(layout):
        _systemctl('restart', slot['daemon']['containerd_unit'])
        _systemctl('restart', slot['daemon']['docker_unit'])
    # The default dockerd relays development stdio; keep it responsive among housekeeping processes.
    for unit in ('docker.service', 'containerd.service'):
        _systemctl('set-property', unit, 'CPUWeight=1000')
    STATE.mkdir(parents=True, exist_ok=True)
    return status(layout)


def live_leases_present() -> bool:
    from .slots import Broker

    try:
        return bool(Broker(load()).leases())
    except RuntimeError:
        return False


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError as error:
        return f'unavailable:{type(error).__name__}'


def unit_cpus(unit: str) -> str:
    result = subprocess.run(
        ['systemctl', 'show', unit, '-p', 'ControlGroup', '--value'], capture_output=True, text=True, check=False
    )
    group = result.stdout.strip()
    return _read(CGROUP / group.lstrip('/') / 'cpuset.cpus.effective') if group else 'unavailable'


def docker_ok(host: str) -> bool:
    result = subprocess.run(['docker', '--host', host, 'version', '--format', '{{.Server.Version}}'],
                            capture_output=True, text=True, check=False, timeout=20)
    return result.returncode == 0


def _user_process(pid: str) -> bool:
    """Kernel threads have an empty cmdline and stay where the kernel puts them."""
    try:
        return bool(Path(f'/proc/{pid}/cmdline').read_bytes())
    except OSError:
        return False


def private_containerd(slot: dict) -> bool:
    """The slot's dockerd must use the slot's own containerd process, not the system one."""
    daemon = slot['daemon']
    result = subprocess.run(
        ['systemctl', 'show', daemon['containerd_unit'], '-p', 'MainPID', '--value'], capture_output=True, text=True
    )
    pid = result.stdout.strip()
    if not pid or pid == '0':
        return False
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except OSError:
        return False
    config = str(CONFIG.parent / slot['id'] / 'containerd.toml').encode()
    dockerd = subprocess.run(
        ['systemctl', 'show', daemon['docker_unit'], '-p', 'ExecStart', '--value'], capture_output=True, text=True
    ).stdout
    return config in command and f'--containerd={daemon["containerd_socket"]}' in dockerd


def status(layout: dict | None = None) -> dict:
    layout = layout or load()
    partition = CGROUP / PARTITION_SLICE
    effective = _read(partition / 'cpuset.cpus.effective')
    housekeeping = set(parse_cpus(layout['housekeeping_cpus']))
    outside = {unit: unit_cpus(unit) for unit in ('docker.service', 'containerd.service')}
    for directory in sorted(p for p in CGROUP.iterdir() if p.is_dir() and p.name != PARTITION_SLICE):
        if (directory / 'cpuset.cpus.effective').exists():
            outside[directory.name] = _read(directory / 'cpuset.cpus.effective')
    root_processes = [pid for pid in _read(CGROUP / 'cgroup.procs').split() if _user_process(pid)]

    def within(value: str, allowed: set[int]) -> bool:
        return not value.startswith('unavailable') and bool(value) and set(parse_cpus(value)) <= allowed

    daemons = {}
    for slot in grade_slots(layout):
        daemon = slot['daemon']
        daemons[slot['id']] = {
            'cpus': {unit: unit_cpus(daemon[unit]) for unit in ('containerd_unit', 'docker_unit')},
            'pinned_to_sibling': all(
                within(unit_cpus(daemon[unit]), {slot['host_cpu']}) for unit in ('containerd_unit', 'docker_unit')
            ),
            'private_containerd': private_containerd(slot),
            'up': docker_ok(slot['docker_host']),
        }
    checks = {
        'partition_cpus': not effective.startswith('unavailable')
        and parse_cpus(effective) == parse_cpus(layout['partition_cpus']),
        'others_on_housekeeping': all(within(value, housekeeping) for value in outside.values()),
        'no_user_processes_in_root_cgroup': not root_processes,
        'docker_sees_all_cpus': _read(CGROUP / 'cpuset.cpus.effective') == format_cpus(
            cpu for core in layout['topology'] for cpu in core['cpus']
        ),
        'grade_daemons_pinned_to_slot': all(d['pinned_to_sibling'] for d in daemons.values()),
        'grade_containerd_private': all(d['private_containerd'] for d in daemons.values()),
        'grade_daemons_up': all(d['up'] for d in daemons.values()),
        'dev_daemon_up': docker_ok(layout['docker_hosts']['dev']),
    }
    return {
        'ready': all(checks.values()),
        'checks': checks,
        'partition_effective': effective,
        'outside_partition': outside,
        'root_cgroup_user_processes': root_processes,
        'grade_daemons': daemons,
        'housekeeping_cpus': layout['housekeeping_cpus'],
        'slots': {kind: [s['id'] for s in layout['slots'] if s['kind'] == kind] for kind in ('dev', 'grade')},
        'memory_budget_gib': round(layout['memory_budget_bytes'] / 2**30, 1),
        'layout_sha256': layout['layout_sha256'],
    }


def require_ready(layout: dict) -> None:
    report = status(layout)
    if not report['ready']:
        failed = ', '.join(k for k, v in report['checks'].items() if not v)
        raise RuntimeError(f'Host isolation is not active ({failed}); run ./bench host setup')
