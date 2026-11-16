"""Grade submissions on prototype cases on one leased grading slot (prototype family measurement)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, '/data/workspace/FrontierAM')
import arena.tasks  # noqa: F401
from arena import machine
from arena.slots import Broker
from arena.operator_drivers.blackbox_cubature import grade, load_oracle

oracle_path, cases_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
subs = dict(a.split('=', 1) for a in sys.argv[4:])
cases = json.loads(cases_file.read_text())
oracle = load_oracle(oracle_path)
layout = machine.load(); broker = Broker(layout)
result = {}
with broker.acquire('grade', 3 * 2**30, 'cubature prototype family measurement') as lease:
    slot = next(s for s in layout['slots'] if s['id'] == lease.slot['id'])
    for name, sub in subs.items():
        t = time.time()
        result[name] = grade(Path(sub), cases, oracle, docker_host=slot['docker_host'], image='frontieram-blackbox-cubature:amd64-v4-r1',
                             cpuset=lease.cpuset, cgroup_parent=lease.cgroup_parent, scratch=out.parent / 'scratch', case_limit=30)
        result[name]['wall_total_s'] = time.time() - t
        print(name, result[name]['status'], round(result[name]['wall_total_s']), flush=True)
out.write_text(json.dumps(result, indent=1) + '\n')
