"""Run species-tree simulation jobs in containers on leased development slots (never on unleased cores).

Each line of the job file is one shell command run inside the species image with /sim, /refs, /out and the phylogenetics
tools mounted. Workers each hold one dev lease from the slot broker for as long as they run jobs.

Usage: PYTHONPATH=. FRONTIERAM_RUNTIME=... run_leased.py JOBFILE OUTDIR --workers N
"""

import argparse
import queue
import subprocess
import threading

from pathlib import Path

from arena.cli import context, placement_for

IMAGE = 'frontieram-species-tree:amd64-b1'
ROOT = Path(__file__).resolve().parents[4]


def worker(broker, layout, jobs: queue.Queue, out: Path, index: int) -> None:
    with broker.acquire('dev', 4 * 2**30, f'species-tree simulation worker {index}', timeout=7200) as lease:
        placement = placement_for(lease, layout)
        while True:
            try:
                n, command = jobs.get_nowait()
            except queue.Empty:
                return
            log = out / 'logs' / f'job-{n:04d}.log'
            subprocess.run(
                ['docker', 'run', '--rm', '--network', 'none', '--cpuset-cpus', placement.cpuset, '--cpus', '1',
                 '--cgroup-parent', placement.cgroup_parent, '--memory', '4g', '--memory-swap', '4g',
                 '-v', '/data/tools/phylo:/data/tools/phylo:ro', '-v', f'{ROOT}/tasks/operator/species-tree/sim:/sim:ro',
                 '-v', f'{ROOT}/.runtime-ops/species-tree/references:/refs:ro', '-v', f'{out}:/out',
                 '--entrypoint', 'sh', IMAGE, '-c', command],
                stdout=log.open('wb'), stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobfile', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    out = args.out.resolve()
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    jobs: queue.Queue = queue.Queue()
    for n, line in enumerate(l for l in args.jobfile.read_text().splitlines() if l.strip()):
        jobs.put((n, line))
    layout, broker, _ = context()
    threads = [threading.Thread(target=worker, args=(broker, layout, jobs, out, i)) for i in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
