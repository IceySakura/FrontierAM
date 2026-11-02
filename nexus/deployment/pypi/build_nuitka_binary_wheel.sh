#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_VERSION="${NEXUS_BINARY_PYTHON_VERSION:-3.12}"
IMAGE="${NEXUS_BINARY_MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}"
PLAT_TAG="${NEXUS_BINARY_PLAT_TAG:-manylinux_2_28_x86_64}"
OUT_ROOT="${NEXUS_BINARY_BUILD_ROOT:-$ROOT/build}"
WHEEL_DIR="${NEXUS_BINARY_WHEEL_DIR:-$ROOT/dist}"
WORKDIR_IN_CONTAINER="${NEXUS_BINARY_BUILD_WORKDIR:-/io}"
USE_SJTU_MIRRORS="${NEXUS_BINARY_USE_SJTU_MIRRORS:-1}"
PIP_INDEX_URL="${NEXUS_BINARY_PIP_INDEX_URL:-https://mirror.sjtu.edu.cn/pypi/web/simple}"
ALMALINUX_MIRROR_URL="${NEXUS_BINARY_ALMALINUX_MIRROR_URL:-https://mirrors.sjtug.sjtu.edu.cn/almalinux}"
EPEL_MIRROR_URL="${NEXUS_BINARY_EPEL_MIRROR_URL:-https://ftp.sjtu.edu.cn/epel}"
PACKAGE_NAME="${NEXUS_BINARY_PACKAGE_NAME:-}"
NUITKA_JOBS="${NUITKA_JOBS:-32}"
CCACHE_DIR_HOST="${NEXUS_BINARY_CCACHE_DIR:-$ROOT/.cache/nuitka-ccache}"
CCACHE_MAX_SIZE="${NEXUS_BINARY_CCACHE_MAX_SIZE:-5G}"
PIP_CACHE_DIR_HOST="${NEXUS_BINARY_PIP_CACHE_DIR:-$ROOT/.cache/pip}"

usage() {
  cat <<'EOF'
Usage: deployment/pypi/build_nuitka_binary_wheel.sh [OPTIONS]

Build a compiled Nexus wheel with Nuitka inside a manylinux container.

Options:
  --python VERSION_OR_TAG  Python version/tag, e.g. 3.12, 3.13, cp312-cp312.
  --image IMAGE           manylinux image. Default: quay.io/pypa/manylinux_2_28_x86_64.
  --plat TAG              wheel platform tag. Default: manylinux_2_28_x86_64.
  --out-dir DIR           output wheel directory. Default: dist.
  -h, --help              show this help.

Examples:
  deployment/pypi/build_nuitka_binary_wheel.sh --python 3.12
  deployment/pypi/build_nuitka_binary_wheel.sh --python cp313-cp313 --plat manylinux_2_28_x86_64

Environment:
  NEXUS_BINARY_USE_SJTU_MIRRORS=0   Disable SJTU yum/pip mirror setup.
  NEXUS_BINARY_PACKAGE_NAME=NAME     Override wheel distribution name.
  NUITKA_JOBS=8                     Override Nuitka parallel compile jobs.
  NEXUS_BINARY_CCACHE_DIR=PATH      Persistent ccache directory.
  NEXUS_BINARY_CCACHE_MAX_SIZE=SIZE  Persistent ccache max size. Default: 5G.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="${2:?missing value for --python}"
      shift 2
      ;;
    --image)
      IMAGE="${2:?missing value for --image}"
      shift 2
      ;;
    --plat)
      PLAT_TAG="${2:?missing value for --plat}"
      shift 2
      ;;
    --out-dir)
      WHEEL_DIR="${2:?missing value for --out-dir}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

python_tag_from_version() {
  case "$1" in
    cp*-cp*) printf '%s\n' "$1" ;;
    3.12|312|cp312) printf 'cp312-cp312\n' ;;
    3.13|313|cp313) printf 'cp313-cp313\n' ;;
    3.14|314|cp314) printf 'cp314-cp314\n' ;;
    *)
      echo "Unsupported Python version/tag: $1" >&2
      echo "Use one of 3.12, 3.13, 3.14, or cpXXX-cpXXX." >&2
      exit 2
      ;;
  esac
}

PYTHON_TAG="$(python_tag_from_version "$PYTHON_VERSION")"
mkdir -p "$WHEEL_DIR"
mkdir -p "$OUT_ROOT"
mkdir -p "$CCACHE_DIR_HOST" "$PIP_CACHE_DIR_HOST"
WHEEL_DIR="$(cd "$WHEEL_DIR" && pwd)"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
CCACHE_DIR_HOST="$(cd "$CCACHE_DIR_HOST" && pwd)"
PIP_CACHE_DIR_HOST="$(cd "$PIP_CACHE_DIR_HOST" && pwd)"

container_path() {
  case "$1" in
    "$ROOT") printf '%s\n' "$WORKDIR_IN_CONTAINER" ;;
    "$ROOT"/*) printf '%s/%s\n' "$WORKDIR_IN_CONTAINER" "${1#"$ROOT"/}" ;;
    *)
      echo "Path must be inside the repository so it can be mounted into Docker: $1" >&2
      exit 2
      ;;
  esac
}

CONTAINER_OUT_ROOT="$(container_path "$OUT_ROOT")"
CONTAINER_WHEEL_DIR="$(container_path "$WHEEL_DIR")"

docker run --rm \
  -e PYTHON_TAG="$PYTHON_TAG" \
  -e PLAT_TAG="$PLAT_TAG" \
  -e OUT_ROOT="$CONTAINER_OUT_ROOT" \
  -e WHEEL_DIR="$CONTAINER_WHEEL_DIR" \
  -e USE_SJTU_MIRRORS="$USE_SJTU_MIRRORS" \
  -e PIP_INDEX_URL="$PIP_INDEX_URL" \
  -e ALMALINUX_MIRROR_URL="$ALMALINUX_MIRROR_URL" \
  -e EPEL_MIRROR_URL="$EPEL_MIRROR_URL" \
  -e NEXUS_BINARY_PACKAGE_NAME="$PACKAGE_NAME" \
  -e NUITKA_JOBS="$NUITKA_JOBS" \
  -e CCACHE_DIR=/ccache \
  -e NEXUS_BINARY_CCACHE_MAX_SIZE="$CCACHE_MAX_SIZE" \
  -e PIP_CACHE_DIR=/pip-cache \
  -v "$CCACHE_DIR_HOST:/ccache" \
  -v "$PIP_CACHE_DIR_HOST:/pip-cache" \
  -v "$ROOT:$WORKDIR_IN_CONTAINER" \
  -w "$WORKDIR_IN_CONTAINER" \
  "$IMAGE" \
  bash -lc '
set -Eeuo pipefail

setup_sjtu_mirrors() {
  if [[ "$USE_SJTU_MIRRORS" != "1" ]]; then
    return
  fi

  echo "Configuring SJTU mirrors for manylinux build..."
  if compgen -G "/etc/yum.repos.d/*.repo" > /dev/null; then
    sed -i.bak \
      -e "s|^mirrorlist=|#mirrorlist=|g" \
      -e "s|^metalink=|#metalink=|g" \
      -e "s|^# baseurl=https://repo.almalinux.org/almalinux/|baseurl=${ALMALINUX_MIRROR_URL}/|g" \
      -e "s|^#baseurl=https://repo.almalinux.org/almalinux/|baseurl=${ALMALINUX_MIRROR_URL}/|g" \
      -e "s|^# baseurl=https://repo.almalinux.org/vault/|baseurl=${ALMALINUX_MIRROR_URL}/|g" \
      -e "s|^#baseurl=https://repo.almalinux.org/vault/|baseurl=${ALMALINUX_MIRROR_URL}/|g" \
      -e "s|^#baseurl=https://download.example/pub/epel/|baseurl=${EPEL_MIRROR_URL}/|g" \
      -e "s|^#baseurl=https://download.fedoraproject.org/pub/epel/|baseurl=${EPEL_MIRROR_URL}/|g" \
      -e "s|^#baseurl=http://download.example/pub/epel/|baseurl=${EPEL_MIRROR_URL}/|g" \
      -e "s|^#baseurl=http://download.fedoraproject.org/pub/epel/|baseurl=${EPEL_MIRROR_URL}/|g" \
      /etc/yum.repos.d/*.repo || true
  fi

  if command -v dnf >/dev/null 2>&1; then
    dnf clean all >/dev/null || true
    dnf -y makecache --refresh >/dev/null || true
  elif command -v yum >/dev/null 2>&1; then
    yum clean all >/dev/null || true
    yum -y makecache fast >/dev/null || true
  fi
}

install_build_accelerators() {
  if ! command -v ccache >/dev/null 2>&1; then
    if command -v dnf >/dev/null 2>&1; then
      dnf install -y ccache >/dev/null || true
    elif command -v yum >/dev/null 2>&1; then
      yum install -y ccache >/dev/null || true
    fi
  fi

  if command -v ccache >/dev/null 2>&1; then
    ccache --set-config=max_size="${NEXUS_BINARY_CCACHE_MAX_SIZE:-5G}" >/dev/null || true
    ccache --zero-stats >/dev/null || true
  fi
}

pip_install() {
  if "$PYTHON" -m pip install "$@"; then
    return
  fi
  if [[ "$USE_SJTU_MIRRORS" != "1" ]]; then
    return 1
  fi
  echo "pip install via SJTU mirror failed; retrying with https://pypi.org/simple ..." >&2
  PIP_INDEX_URL=https://pypi.org/simple "$PYTHON" -m pip install "$@"
}

PYTHON="/opt/python/${PYTHON_TAG}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found: $PYTHON" >&2
  echo "Available interpreters:" >&2
  ls -1 /opt/python >&2 || true
  exit 2
fi

setup_sjtu_mirrors
install_build_accelerators

PY_ABI="${PYTHON_TAG%-*}"
ABI_TAG="${PYTHON_TAG#*-}"
OUT_DIR="$OUT_ROOT/nuitka-${PYTHON_TAG}"
STAGING_DIR="$OUT_ROOT/nexus-binary-wheel-${PYTHON_TAG}"
RAW_WHEEL_DIR="$OUT_ROOT/wheelhouse-raw-${PYTHON_TAG}"
MANIFEST_PATH="nexus/_compiled_module_manifest.py"

cleanup_manifest() {
  rm -f "$MANIFEST_PATH"
}
trap cleanup_manifest EXIT

"$PYTHON" --version
pip_install -q --upgrade pip setuptools wheel
pip_install -q -e ".[dev,runtime]" nuitka

"$PYTHON" deployment/pypi/generate_nuitka_module_manifest.py

"$PYTHON" -m nuitka \
  --module nexus \
  --include-package=nexus \
  --include-module=nexus._compiled_module_manifest \
  --jobs="${NUITKA_JOBS:-$(nproc)}" \
  --output-dir="$OUT_DIR" \
  --remove-output

rm -rf "$STAGING_DIR" "$RAW_WHEEL_DIR"
mkdir -p "$STAGING_DIR" "$RAW_WHEEL_DIR" "$WHEEL_DIR"

cp "$OUT_DIR"/nexus.*.so "$STAGING_DIR/"
if compgen -G "$OUT_DIR/nexus.pyi" > /dev/null; then
  cp "$OUT_DIR/nexus.pyi" "$STAGING_DIR/"
fi

"$PYTHON" deployment/pypi/stage_wheel_bundle.py --staging-dir "$STAGING_DIR"

STAGING_DIR="$STAGING_DIR" \
RAW_WHEEL_DIR="$RAW_WHEEL_DIR" \
PY_ABI="$PY_ABI" \
ABI_TAG="$ABI_TAG" \
"$PYTHON" - <<'"'"'PY'"'"'
from __future__ import annotations

import base64
import csv
import hashlib
import os
import re
import zipfile

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def metadata_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name)


def record_hash(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    return f"sha256={digest}", len(data)


staging_dir = Path(os.environ["STAGING_DIR"])
raw_wheel_dir = Path(os.environ["RAW_WHEEL_DIR"])
py_abi = os.environ["PY_ABI"]
abi_tag = os.environ["ABI_TAG"]
platform_tag = "linux_x86_64"

project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
name = os.environ.get("NEXUS_BINARY_PACKAGE_NAME", "").strip() or project["name"]
version = project["version"]
dist_name = normalize_name(name)
dist_info = staging_dir / f"{dist_name}-{version}.dist-info"
dist_info.mkdir(parents=True, exist_ok=True)

metadata_lines = [
    "Metadata-Version: 2.4",
    f"Name: {metadata_name(name)}",
    f"Version: {version}",
]
if project.get("description"):
    metadata_lines.append("Summary: {}".format(project["description"]))
if project.get("requires-python"):
    metadata_lines.append("Requires-Python: {}".format(project["requires-python"]))
for dep in project.get("dependencies", []):
    metadata_lines.append(f"Requires-Dist: {dep}")
for extra, deps in project.get("optional-dependencies", {}).items():
    metadata_lines.append(f"Provides-Extra: {extra}")
    for dep in deps:
        metadata_lines.append(f"Requires-Dist: {dep}; extra == \"{extra}\"")
(dist_info / "METADATA").write_text("\n".join(metadata_lines) + "\n")

(dist_info / "WHEEL").write_text(
    "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: nexus-nuitka-binary-wheel",
            "Root-Is-Purelib: false",
            f"Tag: {py_abi}-{abi_tag}-{platform_tag}",
            "",
        ]
    )
)
(dist_info / "entry_points.txt").write_text("[console_scripts]\nnexus = nexus.cli.main:main\n")
(dist_info / "top_level.txt").write_text("nexus\n")

record_path = dist_info / "RECORD"
rows = []
for path in sorted(p for p in staging_dir.rglob("*") if p.is_file() and p != record_path):
    rel = path.relative_to(staging_dir).as_posix()
    digest, size = record_hash(path)
    rows.append([rel, digest, str(size)])
rows.append([record_path.relative_to(staging_dir).as_posix(), "", ""])
with record_path.open("w", newline="") as f:
    csv.writer(f).writerows(rows)

wheel_name = f"{dist_name}-{version}-{py_abi}-{abi_tag}-{platform_tag}.whl"
wheel_path = raw_wheel_dir / wheel_name
with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(p for p in staging_dir.rglob("*") if p.is_file()):
        zf.write(path, path.relative_to(staging_dir).as_posix())
print(wheel_path)
PY

RAW_WHEEL="$(find "$RAW_WHEEL_DIR" -name "*.whl" -print -quit)"
if [[ -z "$RAW_WHEEL" ]]; then
  echo "No raw wheel found in $RAW_WHEEL_DIR" >&2
  exit 1
fi
auditwheel repair --plat "$PLAT_TAG" -w "$WHEEL_DIR" "$RAW_WHEEL"
if command -v ccache >/dev/null 2>&1; then
  ccache --show-stats || true
fi
echo "Wrote compiled wheel(s) to $WHEEL_DIR"
'
