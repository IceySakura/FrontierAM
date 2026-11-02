#!/usr/bin/env bash

set -euo pipefail

LOG=/var/log/dockerd-bootstrap.log
# /run must be created before /var/run: in some images /var/run is a symlink to
# a missing /run, and `mkdir -p /var/run` then fails on the dangling symlink.
mkdir -p /var/log /run /var/run

if [ -S /var/run/docker.sock ] && docker info >/dev/null 2>&1; then
    echo "dockerd already running"
    exit 0
fi

# ---------------------------------------------------------------------------
# Platform probes. Everything below only reads these.
# The two probes are deliberately independent: "has a cgroup v2 hierarchy" and
# "is a Kata sandbox" are orthogonal. A Kata guest can be booted on cgroup v1,
# and a Kata Docker data root is not always virtio-fs (tmpfs and real volumes
# are both in use), so neither probe may stand in for the other.
# ---------------------------------------------------------------------------
cgroup_root=/sys/fs/cgroup
guest_has_cgroup_v2=false
if [[ -f "$cgroup_root/cgroup.controllers" ]]; then
    guest_has_cgroup_v2=true
fi

# Kata presents a regular emptyDir to the guest through virtio-fs. In the
# current platform contract a virtio-fs Docker data root is what identifies a
# Kata sandbox; it selects both the MTU and the disk image conversion below.
DOCKER_DATA_ROOT=/var/lib/docker
mkdir -p "$DOCKER_DATA_ROOT"
# findmnt lists every mount stacked on the path, oldest first. Once the image
# below is mounted the last line is our own filesystem, so the platform's own
# filesystem — the one that says whether this is Kata — is the first line.
# Reading the last line here would make a re-run after dockerd stopped forget
# that it is on Kata and hand dockerd the gVisor MTU.
docker_data_fs="$(findmnt -n -o FSTYPE -T "$DOCKER_DATA_ROOT" 2>/dev/null | head -n 1 || true)"
data_root_mounted_fs="$(findmnt -n -o FSTYPE -T "$DOCKER_DATA_ROOT" 2>/dev/null | tail -n 1 || true)"
data_root_on_virtiofs=false
case "$docker_data_fs" in
    virtiofs|virtio-fs) data_root_on_virtiofs=true ;;
esac

echo "platform: cgroup_v2=$guest_has_cgroup_v2 data_fs=${docker_data_fs:-unknown} virtiofs=$data_root_on_virtiofs" >>"$LOG"

# ---------------------------------------------------------------------------
# 1. Nested cgroup v2 delegation.
# A nested cgroup v2 hierarchy needs an empty domain parent. Otherwise runc
# falls back to threaded cgroups, which have no per-container memory accounting.
# Keep the outer processes in a leaf, as in moby's hack/dind, before dockerd
# creates any children. gVisor's cgroup v1 mounts do not need this preparation.
# ---------------------------------------------------------------------------
if [[ "$guest_has_cgroup_v2" == true ]]; then
    # cgroup.type is absent on a hierarchy root, so its presence is what confirms
    # the sandbox has its own cgroup namespace and $cgroup_root is a delegated
    # subtree. Without that namespace every path below would address the real
    # root, so report that directly instead of failing on a missing file.
    if [[ ! -f "$cgroup_root/cgroup.type" ]]; then
        echo "FATAL: DinD requires the sandbox to have its own cgroup namespace; $cgroup_root is a hierarchy root" >&2
        exit 1
    fi
    cgroup_type="$(<"$cgroup_root/cgroup.type")"
    if [[ "$cgroup_type" != domain ]]; then
        echo "FATAL: DinD requires a cgroup v2 domain parent, found '$cgroup_type'; recreate the sandbox" >&2
        exit 1
    fi
    read -r -a cgroup_controllers < "$cgroup_root/cgroup.controllers"
    if [[ " ${cgroup_controllers[*]} " != *" memory "* ]]; then
        echo "FATAL: memory controller is not delegated to the DinD sandbox" >&2
        exit 1
    fi
    if [[ ! -w "$cgroup_root/cgroup.subtree_control" ]]; then
        if ! mount -o remount,rw "$cgroup_root"; then
            echo "FATAL: DinD requires a writable delegated cgroup v2 mount" >&2
            exit 1
        fi
    fi
    echo "cgroup v2 sandbox: self_ns=$(readlink /proc/self/ns/cgroup) pid1_ns=$(readlink /proc/1/ns/cgroup) pid1_cgroup=$(cut -d: -f3 /proc/1/cgroup) controllers=${cgroup_controllers[*]}" >>"$LOG"
    mkdir -p "$cgroup_root/init"
    cgroup_ready=false
    for ((cgroup_attempt = 0; cgroup_attempt < 20; cgroup_attempt++)); do
        # Read with shell builtins so the snapshot does not include a transient
        # cat/xargs process. Concurrent execs can still race with this migration.
        mapfile -t cgroup_pids < "$cgroup_root/cgroup.procs"
        for cgroup_pid in "${cgroup_pids[@]}"; do
            if ! { echo "$cgroup_pid" > "$cgroup_root/init/cgroup.procs"; } 2>>"$LOG"; then
                if [[ -d "/proc/$cgroup_pid" ]]; then
                    echo "FATAL: cannot move sandbox process $cgroup_pid into the DinD init cgroup" >&2
                    exit 1
                fi
            fi
        done
        if { echo "${cgroup_controllers[*]/#/+}" > "$cgroup_root/cgroup.subtree_control"; } 2>>"$LOG"; then
            cgroup_ready=true
            break
        fi
        sleep 0.1
    done
    if [[ "$cgroup_ready" != true ]]; then
        echo "FATAL: cannot enable nested cgroup v2 controllers after 20 attempts; see $LOG" >&2
        exit 1
    fi
    # Enabling a controller in the parent creates its interface files in every
    # child. Checking for them turns a partial delegation into a bootstrap
    # failure instead of inner containers silently reporting "0B / 0B".
    if [[ ! -f "$cgroup_root/init/memory.current" ]]; then
        echo "FATAL: memory controller was not delegated to nested cgroups; see $LOG" >&2
        exit 1
    fi
    echo "prepared cgroup v2 domain for DinD: subtree_control=$(<"$cgroup_root/cgroup.subtree_control")" >>"$LOG"
fi

# ---------------------------------------------------------------------------
# 2. Network: uplink discovery, forwarding and SNAT.
# Section 3 depends on $dev discovered here.
# ---------------------------------------------------------------------------
if ! iptables --version 2>/dev/null | grep -q legacy; then
    echo "FATAL: iptables must use the legacy backend for Docker-in-Docker under gVisor" >&2
    iptables --version 2>&1 | head -1 >&2 || true
    exit 1
fi

dev="$(ip route show default | awk '/default/ {print $5; exit}')"
if [[ -z "$dev" ]]; then
    echo "FATAL: cannot determine default network device for Docker-in-Docker SNAT" >&2
    exit 1
fi

addr="$(ip -4 addr show dev "$dev" | awk '/inet / {print $2; exit}' | cut -d/ -f1)"
if [[ -z "$addr" ]]; then
    echo "FATAL: cannot determine IPv4 address for device $dev" >&2
    exit 1
fi

# /proc/sys is in runc's default readonlyPaths, so a sandbox that is not
# privileged gets it read-only even when it holds every capability. Remount it
# here rather than rely on the pod being privileged; CAP_SYS_ADMIN is enough,
# and dockerd may want other sysctls later, not just ip_forward.
if [[ ! -w /proc/sys/net/ipv4/ip_forward ]]; then
    if ! mount -o remount,rw /proc/sys; then
        echo "FATAL: /proc/sys is read-only and cannot be remounted; the sandbox needs CAP_SYS_ADMIN" >&2
        exit 1
    fi
    echo "remounted /proc/sys read-write" >>"$LOG"
fi
ip_forward="$(</proc/sys/net/ipv4/ip_forward)"
if [[ "$ip_forward" != 1 ]]; then
    if ! { echo 1 > /proc/sys/net/ipv4/ip_forward; }; then
        echo "FATAL: cannot enable IPv4 forwarding (current value: $ip_forward)" >&2
        exit 1
    fi
fi
iptables -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p tcp 2>>"$LOG" || true
iptables -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p udp 2>>"$LOG" || true

# ---------------------------------------------------------------------------
# 3. Docker data root: MTU selection and storage preparation.
# Both steps are selected by $data_root_on_virtiofs. The MTU is decided here
# rather than with the other probes because it needs $dev from section 2.
# ---------------------------------------------------------------------------

# Preserve gVisor's existing 65549 default; only a Kata data root inherits the
# guest uplink MTU (1450 on the current kata-pvm cluster).
mtu=65549
mtu_limit=65549
mtu_source=gvisor-default
if [[ "$data_root_on_virtiofs" == true ]]; then
    mtu_file="/sys/class/net/$dev/mtu"
    if [[ ! -r "$mtu_file" ]]; then
        echo "FATAL: cannot read Kata guest uplink MTU for default network device $dev" >&2
        exit 1
    fi
    uplink_mtu="$(<"$mtu_file")"
    if [[ ! "$uplink_mtu" =~ ^[0-9]+$ ]] || (( uplink_mtu < 68 )); then
        echo "FATAL: invalid Kata guest uplink MTU '$uplink_mtu' for default network device $dev" >&2
        exit 1
    fi
    mtu="$uplink_mtu"
    mtu_limit="$uplink_mtu"
    mtu_source=kata-auto
fi
if [[ -n "${NEXUS_DIND_MTU:-}" ]]; then
    mtu="$NEXUS_DIND_MTU"
    mtu_source=override
fi
if [[ ! "$mtu" =~ ^[0-9]+$ ]] || (( mtu < 68 || mtu > mtu_limit )); then
    echo "FATAL: Docker MTU '$mtu' must be between 68 and '$mtu_limit' for data filesystem '$docker_data_fs'" >&2
    exit 1
fi
echo "Docker MTU: data_fs=$docker_data_fs uplink=$dev docker_mtu=$mtu limit=$mtu_limit source=$mtu_source" >>"$LOG"

# overlay2 cannot use virto-fs as its backing filesystem, so create a sparse
# image on that volume and mount it over /var/lib/docker. gVisor's tmpfs and
# native xfs volumes back overlay2 directly and need no conversion.
if [[ "$data_root_on_virtiofs" == true && "$data_root_mounted_fs" == xfs ]]; then
    echo "Docker data root already carries the XFS image; leaving it mounted" >>"$LOG"
elif [[ "$data_root_on_virtiofs" == true ]]; then
    disk_image="$DOCKER_DATA_ROOT/.nexus-dind.img"
    disk_size="${NEXUS_DIND_DISK_SIZE:-100G}"
    for required_command in truncate mkfs.xfs blkid mount mknod; do
        if ! command -v "$required_command" >/dev/null 2>&1; then
            echo "FATAL: $required_command is required to prepare Docker storage on virtio-fs" >&2
            exit 1
        fi
    done
    if [[ ! -e "$disk_image" ]]; then
        echo "preparing $disk_size XFS image for Docker storage on virtio-fs" >>"$LOG"
        if ! truncate -s "$disk_size" "$disk_image"; then
            echo "FATAL: failed to create $disk_size Docker XFS image at $disk_image" >&2
            exit 1
        fi
        # overlay2 refuses an XFS without d_type, which ftype=1 provides. No
        # -f: the file was absent a moment ago, so anything to overwrite here
        # means the assumption broke and mkfs should refuse rather than wipe.
        if ! mkfs.xfs -q -n ftype=1 "$disk_image" >>"$LOG" 2>&1; then
            echo "FATAL: failed to format Docker XFS image at $disk_image" >&2
            exit 1
        fi
    else
        # An unreadable superblock may still be repairable, and reformatting
        # would throw that away. Probe failures are for a human to resolve.
        image_fs="$(blkid -p -s TYPE -o value "$disk_image" 2>/dev/null || true)"
        if [[ "$image_fs" != xfs ]]; then
            echo "FATAL: existing Docker image $disk_image probes as '${image_fs:-unidentifiable}', refusing to reformat; inspect it with xfs_repair -n, or delete it if the data is expendable" >&2
            exit 1
        fi
        echo "reusing the existing XFS image at $disk_image" >>"$LOG"
    fi

    # The Kata guest kernel has loop support, but minimal guest images do
    # not necessarily run udev/mdev to populate the guest-local loop device
    # nodes. Creating them requires CAP_MKNOD and write access to /dev;
    # mounting the image also requires access to these guest-local devices.
    if [[ ! -e /dev/loop-control ]]; then
        if ! mknod -m 0600 /dev/loop-control c 10 237; then
            echo "FATAL: failed to create /dev/loop-control; check CAP_MKNOD, /dev write permissions and device-node creation restrictions" >&2
            exit 1
        fi
    fi
    for loop_minor in $(seq 0 15); do
        if [[ ! -e "/dev/loop$loop_minor" ]]; then
            if ! mknod -m 0600 "/dev/loop$loop_minor" b 7 "$loop_minor"; then
                echo "FATAL: failed to create /dev/loop$loop_minor; check CAP_MKNOD, /dev write permissions and device-node creation restrictions" >&2
                exit 1
            fi
        fi
    done

    if ! mount -t xfs -o loop "$disk_image" "$DOCKER_DATA_ROOT" >>"$LOG" 2>&1; then
        echo "FATAL: failed to mount Docker XFS image on $DOCKER_DATA_ROOT; check CAP_SYS_ADMIN, guest loop device availability and access, and XFS support; see $LOG for details" >&2
        exit 1
    fi
    data_root_mounted_fs="$(findmnt -n -o FSTYPE -T "$DOCKER_DATA_ROOT" 2>/dev/null | tail -n 1 || true)"
    if [[ "$data_root_mounted_fs" != xfs ]]; then
        echo "FATAL: Docker data root filesystem is '$data_root_mounted_fs' after XFS mount" >&2
        exit 1
    fi
    echo "mounted XFS loop filesystem for Docker data root at $DOCKER_DATA_ROOT" >>"$LOG"
fi

# Snapshot the outer sandbox's env whitelist for dind-runc to inject into inner
# containers. MUST run before the `export SSL_CERT_FILE` below — otherwise the
# snapshot would capture docker-bootstrap.sh's dockerd-specific CA path
# (/var/run/dind-ca-bundle.crt) instead of the original outer value
# (/etc/ssl/certs/ca-certificates.crt), and inner containers would get a
# SSL_CERT_FILE pointing to a file that doesn't exist inside them.
INJECT_ENV=/var/run/dind-inject.env
: >"$INJECT_ENV"
for k in \
    HTTP_PROXY HTTPS_PROXY FTP_PROXY NO_PROXY \
    http_proxy https_proxy ftp_proxy no_proxy \
    PIP_INDEX_URL PIP_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_INDEX_URL \
    UV_NATIVE_TLS CARGO_HTTP_MULTIPLEXING GRADLE_OPTS \
    SSL_CERT_FILE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS; do
    v="${!k:-}"; [[ -n "$v" ]] && printf '%s=%s\n' "$k" "$v" >>"$INJECT_ENV"
done

# Build a CA bundle for dockerd's TLS verification when pulling images.
# Outbound traffic goes through a squid proxy that does TLS interception, so the
# registry cert dockerd sees is re-signed by a custom squid CA — without it
# dockerd fails with "x509: certificate signed by unknown authority". The task
# image rootfs (e.g. ubuntu:24.04) may also lack ca-certificates entirely.
# We therefore concatenate a public-root bundle (best effort) with the custom
# squid CA into one file and point dockerd at it via SSL_CERT_FILE. This is
# registry-agnostic: it applies to every registry dockerd talks to.
DIND_CA_BUNDLE=/var/run/dind-ca-bundle.crt
SQUID_CA=/usr/local/share/ca-certificates/squid-proxy-ca.crt
: >"$DIND_CA_BUNDLE"
for base_ca in \
    /nix/runtime/nix-env/etc/ssl/certs/ca-bundle.crt \
    /etc/ssl/certs/ca-certificates.crt \
    /etc/ssl/certs/ca-bundle.crt \
    /nix/store/*-nss-cacert-*/etc/ssl/certs/ca-bundle.crt; do
    if [[ -e "$base_ca" ]]; then
        cat "$base_ca" >>"$DIND_CA_BUNDLE"
        echo "CA bundle: included base roots from $base_ca" >>"$LOG"
        break
    fi
done
if [[ -e "$SQUID_CA" ]]; then
    cat "$SQUID_CA" >>"$DIND_CA_BUNDLE"
    echo "CA bundle: included custom squid proxy CA from $SQUID_CA" >>"$LOG"
else
    echo "WARNING: squid proxy CA not found at $SQUID_CA; registry pulls via the TLS-intercepting proxy may fail" >&2
fi
if [[ -s "$DIND_CA_BUNDLE" ]]; then
    export SSL_CERT_FILE="$DIND_CA_BUNDLE"
    # Some Docker BuildKit registry-resolution paths run before any dind-runc
    # container hook/env injection and read the platform-provided SSL_CERT_FILE
    # path from the outer sandbox environment (usually
    # /etc/ssl/certs/ca-certificates.crt). User task images may not contain that
    # file, so point it at the merged bundle as well. This keeps BuildKit base
    # image metadata/token fetches working with ssl-bump.
    mkdir -p /etc/ssl/certs
    ln -sf "$DIND_CA_BUNDLE" /etc/ssl/certs/ca-certificates.crt 2>>"$LOG" || true
    echo "using SSL_CERT_FILE=$SSL_CERT_FILE for dockerd TLS" >>"$LOG"
else
    echo "WARNING: no CA material found; dockerd TLS verification to registries may fail" >&2
fi

# ---------------------------------------------------------------------------+
# Inner-container egress: a runc wrapper that injects proxy env
# (connectivity), the proxy CA (cert trust) and mirror/download-source env
# into every inner container's OCI spec. dockerd itself only wires TLS for
# its own registry pulls (above); inner `docker run` containers inherit
# neither the outer proxy env nor the CA, so we inject them here.
#
# Why a runc wrapper: it is the only official mechanism that applies to
# ALL inner containers by default (same trick as nvidia-container-runtime).
# PATH-based runc and moby libexec/docker/runc are routed through dind-runc by
# the nix profile override and runtime image build step, so daemon.json no
# longer needs a custom runtime/default-runtime entry.
#
# Self-adapting egress: proxy env is snapshotted from the outer sandbox.
# With sbx-proxy disabled the platform sets HTTP(S)_PROXY on the outer and
# dind-runc injects them into inner containers (connectivity via explicit
# proxy). dockerd itself is started with those vars emptied so it does not
# use the HTTP proxy. If sbx-proxy is ever enabled the platform strips those
# env, so nothing is injected and transparent hijacking is expected to
# provide connectivity instead — no code change needed to switch modes.
# (INJECT_ENV snapshot was moved above, before the SSL_CERT_FILE override.)

# dind-runc wrapper 由 nix flake 打包成 /nix/runtime/nix-env/bin/dind-runc（见 flake.nix）
DIND_RUNC=/nix/runtime/nix-env/bin/dind-runc
mkdir -p /etc/docker
if [[ ! -x "$DIND_RUNC" ]]; then
    echo "dind-runc: FATAL: $DIND_RUNC missing — flake package dind_runc not installed" >>"$LOG"
    exit 1
fi
echo "dind-runc: using nix-packaged wrapper $DIND_RUNC" >>"$LOG"

# Keep dockerd options in daemon.json so dockerd starts/restarts with NO CLI
# flags — one source of truth. Runtime interception is handled by the runc
# symlink overrides baked into the runtime image, so no custom runtime entries
# are needed here. Environment is fully controlled by us (nix-packaged runtime
# image), so we overwrite unconditionally.
cat >/etc/docker/daemon.json <<DAEMON_EOF
{
  "iptables": false,
  "ip6tables": false,
  "mtu": $mtu,
  "default-network-opts": {
    "bridge": {
      "com.docker.network.driver.mtu": "$mtu"
    }
  },
  "storage-driver": "overlay2",
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://dockerhub.woa.com",
    "https://csighub.woa.com"
  ]
}
DAEMON_EOF
echo "installed dockerd daemon.json; dind-runc is provided via runc symlink overrides; inject env: $(tr '\n' ' ' <"$INJECT_ENV")" >>"$LOG"

# Keep data-root at /var/lib/docker. The platform prepares that path for nested overlay2.
# dockerd must not manage host iptables under gVisor (iptables=false in daemon.json above);
# the docker package is linked against iptables-legacy (see flake.nix: docker.override
# { iptables = iptables-legacy; }). All dockerd options now live in daemon.json, so start
# with NO CLI flags — restarting dockerd later is just `dockerd` with no args.
# Empty HTTP(S)/NO_PROXY on dockerd only; the INJECT_ENV snapshot above is unchanged.
HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= NO_PROXY= no_proxy= nohup dockerd >> "$LOG" 2>&1 &

for _ in $(seq 1 60); do
    if docker version >/dev/null 2>&1; then
        echo "dockerd ready"
        exit 0
    fi
    sleep 0.5
done

echo "WARNING: dockerd not ready after 30s, see $LOG" >&2
exit 1
