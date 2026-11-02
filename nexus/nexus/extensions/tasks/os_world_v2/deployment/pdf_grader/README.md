# Task025 evaluator-only PDF/OCR image

## Shared CAD+PDF tool

The shared `sdt-hl0en7ib` tool was updated on2026-10-21 to
`hy-llm.tencentcloudcr.com/osworld/cad-grader@sha256:0b1f1ddc0b62d82b9a85a6086de51104df0e5a03ddff3bddc1acf3d008507500`.
Both `cad_grader` and `pdf_grader` can set `install_dependencies=false`.
New instances inherit the image; running instances are unchanged.

Reproduce with `Dockerfile.combined`, which extends the old pinned CAD image,
preserves FreeCAD/system numpy/scipy, and installs PDF packages in a separate
`/opt/nexus-pdf-grader` venv. It inherits `/opt/cad-server.py` and port5000.
Do not substitute the PDF-only image below for this shared tool.

```bash
docker build -f nexus/extensions/tasks/os_world_v2/deployment/pdf_grader/Dockerfile.combined \
  -t <registry>/<repo>:<version> nexus/extensions/tasks/os_world_v2/deployment
```

## Optional PDF-only tool

Build from the **deployment directory** so the image can reuse the minimal
CAD grader HTTP transport:

```bash
docker build -f nexus/extensions/tasks/os_world_v2/deployment/pdf_grader/Dockerfile \
  -t <registry>/<repo>:<version> nexus/extensions/tasks/os_world_v2/deployment
```

Publish to an authorized registry and configure a new AGS custom tool with
the immutable image digest, port5000, HTTP `/health` probe, 4CPU/8GiB,
command `["/usr/bin/python3", "/opt/pdf-server.py"]`. Use token authentication.
Do not replace the CAD tool image: task103/104 still need their CAD dependencies.

No model credentials, PDFs or reference assets are embedded. At evaluation
time each run uses a fresh instance. Official scoring runs under a separate
unprivileged account in `unshare --net`; model calls are relayed to the host's
existing official auxiliary-model client. Inputs and trusted code are hash
checked. Never expose this control service without AGS authentication.

Set `task_config.v21.pdf_grader` to
`{"backend":"ags","tool_id":"<new tool>","install_dependencies":false,"timeout":1800}`.
Without a preinstalled tool, retain `install_dependencies=true`: a compatible
Ubuntu22.04 evaluator tool can install the pinned
PDF libraries and Tesseract in each fresh instance. This does not update the
underlying tool or affect other instances.
