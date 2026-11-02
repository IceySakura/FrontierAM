# Benchmark Docker

This image extends the latest Nexus runtime image and adds benchmark-only
dependencies via UV.

## Files

- `Dockerfile` — Extends the Nexus runtime image.
- `uv-requirements.txt` — Extra Python deps for benchmarks.
- `build.sh` — Local build helper.

## Build & Push

```bash
./build.sh
```

Defaults:
- Registry image: `mirrors.tencent.com/asi/nexus-eval:tmp-<LOCAL_IP>`
- Local tag: `nexus-benchmark:local`

Common overrides:

```bash
IMAGE_NAME=nexus-benchmark:latest \
REGISTRY_IMAGE=mirrors.tencent.com/asi/nexus-eval \
IMAGE_TAG=tmp-10.0.0.1 \
PUSH=true \
./build.sh
```

Auth for remote tag deletion (optional):

```bash
REGISTRY_USER=your_user REGISTRY_PASSWORD=your_password ./build.sh
```
