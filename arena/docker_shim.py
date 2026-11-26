"""Docker CLI shim for an operator package's grader running in one grading slot.

The package's graders call plain `docker` with the identity of their calibration host: an ARM64 platform, a fixed CPU
index and ARM64 image IDs. For `create` and `run` this maps those onto the slot: the sealed amd64 image, the slot's
CPU and its cgroup slice. Every other argument and subcommand passes through unchanged, and DOCKER_HOST already names
the slot's daemon. Configuration comes from FRONTIERAM_DOCKER_SHIM (JSON) and FRONTIERAM_REAL_DOCKER.
"""

import json
import os
import sys


def rewrite(args: list[str], config: dict) -> list[str]:
    images = config['images']
    if not args or args[0] not in ('create', 'run'):
        return [images.get(a, a) for a in args]
    out, i, has_parent = [args[0]], 1, False
    while i < len(args):
        arg = args[i]
        if arg in ('--platform', '--cpuset-cpus') and i + 1 < len(args):
            out += [arg, config['platform'] if arg == '--platform' else config['cpuset']]
            i += 2
            continue
        if config.get('tmp_exec') and arg == '--tmpfs' and i + 1 < len(args) and args[i + 1].startswith('/tmp:') \
                and 'exec' not in args[i + 1].split(':', 1)[1].split(','):
            # Docker mounts a tmpfs noexec unless told otherwise; the development container's /tmp is executable.
            out += [arg, args[i + 1] + ',exec']
            i += 2
            continue
        if arg.startswith('--platform='):
            out.append('--platform=' + config['platform'])
        elif arg.startswith('--cpuset-cpus='):
            out.append('--cpuset-cpus=' + config['cpuset'])
        else:
            has_parent |= arg == '--cgroup-parent' or arg.startswith('--cgroup-parent=')
            out.append(images.get(arg, arg))
        i += 1
    if not has_parent and config.get('cgroup_parent'):
        out.insert(1, '--cgroup-parent=' + config['cgroup_parent'])
    return out


def main():
    config = json.loads(os.environ['FRONTIERAM_DOCKER_SHIM'])
    real = os.environ['FRONTIERAM_REAL_DOCKER']
    args = rewrite(sys.argv[1:], config)
    with open(config['log'], 'a') as log:
        log.write(json.dumps(args) + '\n')
    os.execv(real, [real, *args])


if __name__ == '__main__':
    main()
