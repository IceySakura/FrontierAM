## Python

### Available Images

| Image | Python | When to use |
|-------|--------|-------------|
| `ubuntu-git:22.04` | 3.10 | Default choice |
| `ubuntu-git:20.04` | 3.8 | Projects requiring Python 3.8 |
| `python-git:2.7-slim` | 2.7 (EOL) | Pinned version via `.python-version` / `pyproject.toml` `requires-python` |
| `python-git:3.5-slim` | 3.5 (EOL) | Same |
| `python-git:3.6-slim` | 3.6 (EOL) | Same |
| `python-git:3.7-slim` | 3.7 (EOL) | Same |
| `python-git:3.8-slim` | 3.8 | Same |
| `python-git:3.9-slim` | 3.9 | Same |
| `python-git:3.10-slim` | 3.10 | Same |
| `python-git:3.11-slim` | 3.11 | Same |
| `python-git:3.12-slim` | 3.12 | Same |
| `python-git:3.13-slim` | 3.13 | Same |
| `python-git:3.6-bullseye` ... `3.13-bullseye` | 3.6 -- 3.13 | Native C extensions that fail on `-slim` (includes gcc, cmake, etc.) |

### Phase 1 — Image Selection

1. Check `.python-version`, `pyproject.toml` (`requires-python`), `setup.cfg`, or `tox.ini` for a pinned Python version.
2. If a specific version is found, switch to the matching `python-git:X.Y-slim` image.
3. If the project has native C extensions (e.g. `*.c` / `*.pyx` in source, or build failures on `-slim`), use the `-bullseye` variant instead.
4. If no version is pinned, stay on the default `ubuntu-git:22.04` (Python 3.10).

### Phase 2 — Installation

Install directly into system Python (no venv needed — all base images allow global pip install).

```bash
#!/bin/bash
set -xo pipefail
echo "==[1/2] System packages=="
apt-get install -y libssl-dev libffi-dev build-essential 2>/dev/null || true
echo "==[2/2] pip install=="
pip install --upgrade pip -q
pip install -r requirements.txt || pip install -e ".[dev,test]" || { echo "FAILED pip"; exit 1; }
echo "==INSTALL COMPLETE=="
```

### Phase 3 — eval.sh

```bash
#!/bin/bash
set -xo pipefail
cd /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

python -m pytest tests/specific_test.py::TestClass::test_method -v
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

### Hints

- **ModuleNotFoundError** — run `pip install -e .` to install the project
- **pip mirror** — already set to Tencent in session; no need to re-configure
- **Django** — set `DJANGO_SETTINGS_MODULE` before running tests
