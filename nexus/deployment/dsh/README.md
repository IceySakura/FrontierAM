# DeepSeek Harness prebuilt npm tree

`build_prebuilt.py` accepts any exact DSH/ACP version pair, resolves it with
`npm install` (or accepts a previously materialized installation), verifies
`node-pty`, and produces this Generic-repository layout:

```text
v1/dsh-<version>/acp-<version>/<target>/
  manifest.json
  nexus-dsh-runtime-<version>-<target>.tar.gz
```

The archive is a normal npm prefix containing the generated `package.json`,
the resolution-time `package-lock.json`, and `node_modules`. The lock records
what was published but is not stored in the Nexus source tree. The archive
deliberately is not a DSH SEA binary: both the prebuilt and npm fallback paths
continue to launch the same `dsh --profile` entry and retain the same plugin
behavior.

Build on the oldest libc supported by the target, with the same Node major used
by Nexus runtime:

```bash
python3 deployment/dsh/build_prebuilt.py \
  --dsh-version 0.1.0-rc.6 \
  --acp-version 0.1.0-rc.6 \
  --output-dir /tmp/nexus-dsh-prebuilt
```

For a previously built npm prefix:

```bash
python3 deployment/dsh/build_prebuilt.py \
  --dsh-version 0.1.0-rc.6 \
  --acp-version 0.1.0-rc.6 \
  --source-install /path/to/install \
  --minimum-libc-version 2.34 \
  --output-dir /tmp/nexus-dsh-prebuilt
```

Upload both generated files without changing their relative path below `v1/`.
The Generic upload credential belongs in CI secrets; runtime downloads are
anonymous. Artifact paths are immutable. Rebuilds that change bytes must use a
new package version, target, or manifest schema rather than overwriting a
published archive.
