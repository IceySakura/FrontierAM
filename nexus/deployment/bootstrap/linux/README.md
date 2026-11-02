# Linux x86_64 bundle

Build on Linux x86_64 with Nix (flakes enabled), uv and bubblewrap installed:

```bash
make bundle BUNDLE_OS=linux
# Choose the base runtime interpreter (default 3.13):
uv run --no-sync python deployment/bootstrap/linux/main.py \
  --python-version 3.13 --output build/linux-release
make bundle-publish BUNDLE_OS=linux
```

The builder reuses `deployment/runtime/flake.nix` and its lock, including Chromium,
Node, the patched Bash, desktop tools and Python interpreters. As in
`Dockerfile.runtime`, core Python requirements use the exported uv lock, runtime
module requirements come from `nexus modules install --all --runtime`, and
Playwright's driver uses Nix Node. The bundle contains an installed Nexus wheel
and one base runtime venv, not an editable checkout. Other Python versions remain
available for worker venvs; dependencies unavailable in the matching base venv
may need downloading. `build.json` records the source, closure and archive hashes.

The release layout and publishing rules are the same as Windows:

```text
linux-x86_64/
  manifest.json
  manifest-<version>.json
  bundles/<version>.tar.gz
```

A release contains `entrypoint.py`, `links.json` and `nix/{store,runtime}`.
Nix's absolute and dangling symlinks cannot pass Bootstrap's release-contained
archive validation. The builder therefore archives their path/target metadata
in `links.json` instead of emitting tar symlink entries. After Bootstrap verifies
and extracts the archive, the trusted entrypoint restores these links strictly
under the release's `nix` directory, refusing existing files or symlink parents.
The metadata is covered by the archive SHA-256. Link targets, ELF references and
shebangs retain their original meaning. No global `/nix` is modified, and the
Bootstrap extractor's restrictions remain unchanged.

At launch, the entrypoint uses the **base image's Python 3.10+ and bwrap** to
construct a private mount namespace with the release read-only at `/nix`. It
reproduces the host's other top-level directories, retains the caller's user,
network and devices, and executes the bundled Python. The image must permit
unprivileged user namespaces. This filesystem layout is not an additional
security isolation boundary. The host's `/nix`, if any, is hidden only inside
this process tree. No root installation or Nix daemon is required.

State, worker venvs, uv cache and temporary files live outside the read-only
release. Temporary files use `/tmp` to keep Chromium socket paths short.
The default server port is 18888 (`NEXUS_RUNTIME_PORT` overrides it).
Worker callbacks use 127.0.0.1 while the server binds 0.0.0.0. The launcher keeps
the image's `DISPLAY`, `XAUTHORITY` and GPU environment; it does not start a second
desktop. GPU drivers and a working desktop session remain image responsibilities.
Device access does not automatically add the image's CUDA driver libraries to
Nix's dynamic loader search path. In the tested Ubuntu image, `nvidia-smi` works
and explicitly loading `/usr/lib/x86_64-linux-gnu/libcuda.so.1` from bundled Python
initializes CUDA, but loading `libcuda.so.1` by name alone fails. CUDA applications
must configure their driver lookup; this bundle does not package CUDA frameworks.

## Install Bootstrap

Build `nexus-bootstrap` from the sandbox repository for Linux x86_64 with a libc
baseline compatible with the target image (a static musl build avoids a host
glibc dependency). For a musl build, install the Rust target and an x86_64 musl
C compiler, then run in the sandbox repository:

```bash
cargo build --release --locked -p nexus-bootstrap --target x86_64-unknown-linux-musl
```

Package the resulting binary and this directory's `setup-bootstrap.sh` at the
archive root in `bootstrap.tar.gz`, and upload it to
`nexus-gui-bootstrap-bundle/linux-x86_64/bootstrap.tar.gz`. The tool archive is
separate from the versioned runtime releases. Extract it on the guest and run:


```bash
./setup-bootstrap.sh \
  https://mirrors.tencent.com/repository/generic/nexus-gui-bootstrap-bundle/linux-x86_64/manifest-<version>.json
systemctl --user status nexus-bootstrap.service
journalctl --user -u nexus-bootstrap.service
```

The installer enables and starts a user service and saves the current desktop
session environment. The user manager must be available; automatic desktop login
in the base image starts it after reboot. Session-specific Xauthority paths must
remain valid after reboot. Installation does not enable lingering or change
system permissions. The service uses the existing unsigned-manifest mode; use
only the configured trusted artifact source. Stop it with:

```bash
systemctl --user stop nexus-bootstrap.service
```

Publishing a new release does not restart an already running Bootstrap. Restart
the service to check its configured manifest again. Pin the full versioned URL
for compatibility with the host; Linux AGRT tests can pass it through the create
API's `env` field. The AGRT provider does not select the release or construct
the manifest URL.
