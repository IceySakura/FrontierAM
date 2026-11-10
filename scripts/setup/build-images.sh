#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
platform=${PLATFORM:-linux/amd64}
docker buildx build --load --platform "$platform" --file "$root/docker/Dockerfile.exponential" --tag frontieram-exponential:port-v1 "$root/docker"
docker buildx build --load --platform "$platform" --file "$root/docker/Dockerfile.conic" --tag frontieram-conic:success-v2-port-v1 "$root/docker"
