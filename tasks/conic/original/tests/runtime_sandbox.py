"""Linux pre-exec restrictions for an already unprivileged candidate child.

Call child_setup(paths, mode) from a single-threaded supervisor's Popen
preexec_fn, after Popen has switched to UID/GID 1000 and cleared extra_groups.
paths contains absolute source, tmp, output directories and one input file
(input may be None for build). The caller owns path creation/permissions,
timeouts, descendant cleanup and the clean, single-thread environment.

Requires Landlock ABI >= 3 and libseccomp; setup errors must be infrastructure
errors, never a fallback to an unrestricted child. RLIMIT_NPROC is UID-scoped,
not a per-attempt cgroup PID limit. No aggregate tmp/output or memory quota is
implemented here. Landlock does not hide stat metadata or constrain inherited
FDs: use close_fds=True, no pass_fds, and only stdin/stdout/stderr as intended.

API sources: https://docs.kernel.org/userspace-api/landlock.html
https://docs.kernel.org/userspace-api/seccomp_filter.html
https://github.com/seccomp/libseccomp/blob/main/doc/man/man3/seccomp_rule_add.3
"""

import ctypes as C
import errno
import os
from pathlib import Path
import platform
import resource
import stat
import sys


class _Ruleset(C.Structure):
    _fields_ = [("handled_access_fs", C.c_uint64)]


class _PathRule(C.Structure):
    _pack_ = 1  # linux/landlock.h explicitly packs the uint64 + int32 ABI.
    _fields_ = [("allowed_access", C.c_uint64), ("parent_fd", C.c_int32)]


class _Comparison(C.Structure):
    _fields_ = [("arg", C.c_uint), ("op", C.c_int),
                ("datum_a", C.c_uint64), ("datum_b", C.c_uint64)]


_READ_FILE = 1 << 2
_READ = (1 << 0) | _READ_FILE | (1 << 3)  # execute, read file, read directory
_WRITE_FILE = (1 << 1) | (1 << 14)  # write and truncate (ABI 3)
_HANDLED = (1 << 15) - 1
# Regular files/directories/symlinks/FIFOs; never create devices or sockets.
_WRITE = _HANDLED & ~((1 << 6) | (1 << 9) | (1 << 11))
_RUNTIME = ("/usr", "/lib", "/lib64", "/bin", "/sbin",
            "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d",
            "/etc/localtime", "/etc/nsswitch.conf", "/etc/passwd", "/etc/group",
            "/etc/os-release", "/etc/debian_version", "/etc/lsb-release",
            "/proc/cpuinfo", "/proc/meminfo", "/sys/devices/system/cpu")
_DENIED = """
socket socketpair connect bind listen accept accept4 sendto sendmsg sendmmsg
recvfrom recvmsg recvmmsg shutdown setsockopt getsockopt
ptrace process_vm_readv process_vm_writev pidfd_open pidfd_getfd pidfd_send_signal
kcmp bpf perf_event_open userfaultfd io_uring_setup io_uring_enter io_uring_register
mount umount2 pivot_root chroot move_mount open_tree fsopen fsmount fsconfig fspick
mount_setattr setns unshare open_by_handle_at name_to_handle_at
init_module finit_module delete_module kexec_load kexec_file_load reboot
swapon swapoff sethostname setdomainname quotactl syslog
add_key request_key keyctl mknod mknodat
sched_setaffinity sched_setscheduler sched_setparam sched_setattr
setuid setgid setreuid setregid setresuid setresgid setfsuid setfsgid setgroups
""".split()


def _libraries():
    libc = C.CDLL(None, use_errno=True)
    libc.syscall.restype = C.c_long
    libc.prctl.argtypes = [C.c_int, C.c_ulong, C.c_ulong, C.c_ulong, C.c_ulong]
    libc.prctl.restype = C.c_int
    sec = C.CDLL("libseccomp.so.2", use_errno=True)
    signatures = {
        "seccomp_syscall_resolve_name": ([C.c_char_p], C.c_int),
        "seccomp_init": ([C.c_uint32], C.c_void_p),
        "seccomp_release": ([C.c_void_p], None),
        "seccomp_load": ([C.c_void_p], C.c_int),
        "seccomp_rule_add_array": ([C.c_void_p, C.c_uint32, C.c_int, C.c_uint,
                                     C.POINTER(_Comparison)], C.c_int),
    }
    for name, (args, result) in signatures.items():
        function = getattr(sec, name)
        function.argtypes, function.restype = args, result
    return libc, sec


def _syscall(libc, sec, name, *args):
    number = sec.seccomp_syscall_resolve_name(name.encode())
    if number < 0:
        raise RuntimeError(f"libseccomp cannot resolve required syscall {name}")
    result = libc.syscall(C.c_long(number), *args)
    if result < 0:
        code = C.get_errno()
        raise OSError(code, f"{name}: {os.strerror(code)}")
    return result


def _landlock(libc, sec, paths, mode):
    abi = _syscall(libc, sec, "landlock_create_ruleset", None, C.c_size_t(0), C.c_uint(1))
    if abi < 3:
        raise RuntimeError(f"Landlock ABI >= 3 required, got {abi}")
    ruleset = _Ruleset(_HANDLED)
    fd = _syscall(libc, sec, "landlock_create_ruleset", C.byref(ruleset),
                  C.c_size_t(C.sizeof(ruleset)), C.c_uint(0))
    try:
        entries = [(path, _READ) for path in _RUNTIME if os.path.exists(path)]
        entries += [("/dev/null", _READ_FILE | _WRITE_FILE),
                    ("/dev/zero", _READ_FILE | _WRITE_FILE),
                    ("/dev/urandom", _READ_FILE), ("/dev/random", _READ_FILE),
                    (paths["source"], _WRITE if mode == "build" else _READ),
                    (paths["tmp"], _WRITE), (paths["output"], _WRITE & ~(1 << 0))]
        if paths.get("input") is not None:
            entries.append((paths["input"], _READ_FILE))
        for path, access in entries:
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if not stat.S_ISDIR(os.fstat(parent).st_mode):
                    access &= (1 << 0) | _READ_FILE | _WRITE_FILE
                rule = _PathRule(access, parent)
                _syscall(libc, sec, "landlock_add_rule", C.c_int(fd), C.c_int(1),
                         C.byref(rule), C.c_uint(0))
            finally:
                os.close(parent)
        _syscall(libc, sec, "landlock_restrict_self", C.c_int(fd), C.c_uint(0))
    finally:
        os.close(fd)


def _seccomp(sec):
    context = sec.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW; native ABI only.
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        def deny(name, error=errno.EPERM, comparison=None):
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number == -1:
                raise RuntimeError(f"libseccomp does not know syscall {name}")
            if number < 0:  # Known pseudo-number: syscall absent on native ABI.
                return
            result = sec.seccomp_rule_add_array(
                context, 0x00050000 | error, number,
                int(comparison is not None), C.byref(comparison) if comparison is not None else None)
            if result < 0:
                raise OSError(-result, f"seccomp rule {name}")
        for name in _DENIED:
            deny(name)
        # clone3 arguments live behind a pointer, which seccomp cannot inspect.
        # ENOSYS lets libc fall back to clone for ordinary threads/processes.
        deny("clone3", errno.ENOSYS)
        for flag in (0x00000080, 0x00020000, 0x02000000, 0x04000000,
                     0x08000000, 0x10000000, 0x20000000, 0x40000000):
            deny("clone", comparison=_Comparison(0, 7, flag, flag))
        result = sec.seccomp_load(context)
        if result < 0:
            raise OSError(-result, "seccomp_load")
    finally:
        sec.seccomp_release(context)


def child_setup(paths, mode):
    """Restrict the calling UID-1000 child; errors must abort its exec."""
    if sys.platform != "linux" or platform.machine() not in {"aarch64", "x86_64"}:
        raise RuntimeError("Sandbox requires native Linux aarch64 or x86_64")
    if os.getresuid() != (1000, 1000, 1000) or os.getresgid() != (1000, 1000, 1000) or os.getgroups():
        raise RuntimeError("Popen must first set all UID/GID values to 1000 and clear groups")
    if mode not in {"build", "solve"}:
        raise ValueError("mode must be build or solve")
    normalized = {}
    for key in ("source", "tmp", "output", "input"):
        value = paths.get(key)
        if key == "input" and value is None and mode == "build":
            normalized[key] = None
            continue
        path = Path(value)
        if not path.is_absolute() or path == Path("/") or path.is_symlink():
            raise ValueError(f"Unsafe sandbox {key} path: {path}")
        info = path.stat()
        if not (stat.S_ISREG(info.st_mode) if key == "input" else stat.S_ISDIR(info.st_mode)):
            raise ValueError(f"Invalid sandbox {key} type")
        normalized[key] = str(path.resolve(strict=True))
    directories = [Path(normalized[key]) for key in ("source", "tmp", "output")]
    if any(a == b or a in b.parents or b in a.parents
           for index, a in enumerate(directories) for b in directories[index + 1:]):
        raise ValueError("Source, tmp and output must be disjoint directories")
    if normalized["input"] and any(Path(normalized["input"]).is_relative_to(p) for p in directories):
        raise ValueError("Input must be outside writable/source directories")
    if 0 not in os.sched_getaffinity(0):
        raise RuntimeError("CPU 0 is unavailable; no affinity fallback is permitted")
    os.sched_setaffinity(0, {0})
    for limit, maximum in ((resource.RLIMIT_CORE, 0), (resource.RLIMIT_FSIZE, 2**30),
                           (resource.RLIMIT_NOFILE, 256), (resource.RLIMIT_NPROC, 128)):
        resource.setrlimit(limit, (maximum, maximum))
    libc, sec = _libraries()
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(C.get_errno(), "PR_SET_NO_NEW_PRIVS")
    _landlock(libc, sec, normalized, mode)
    _seccomp(sec)
