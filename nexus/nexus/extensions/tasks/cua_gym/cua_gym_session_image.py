"""CUA-Gym desktop session image support."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile

from collections.abc import Iterable
from pathlib import Path

from nexus.core import command_registry
from nexus.extensions.tasks.cua_gym.hub_manifest import DEFAULT_CUA_GYM_HUB_APPS, HUB_BUILD_PATCHES, HubBuildPatch


DEFAULT_CUA_GYM_SESSION_IMAGE = 'mirrors.tencent.com/hunyuan_yanguan/nexus-cua-gym-session:latest'
DEFAULT_UBUNTU_SESSION_IMAGE = 'nexus-ubuntu-session:latest'
CUA_GYM_SESSION_IMAGE_ENV = 'NEXUS_CUA_GYM_SESSION_IMAGE'
CUA_GYM_SESSION_IMAGE_FORCE_BUILD_ENV = 'NEXUS_CUA_GYM_SESSION_IMAGE_FORCE_BUILD'
CUA_GYM_HUB_ROOT = '/opt/cua-gym-hub'
PACKAGED_REWARD_JUDGE_PATH = Path(__file__).with_name('reward_judge.py')


def _docker(command: str, *, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', '-lc', f'{os.getenv("NEXUS_DOCKER_CMD", "docker")} {command}'],
        check=check,
        capture_output=capture_output,
        text=True,
    )


def _unique_hub_apps(hub_apps: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(app for app in hub_apps if app))


def _hub_patch_dockerfile(patch: HubBuildPatch) -> str:
    file_match = ' -o '.join(f'-name {shlex.quote(glob)}' for glob in patch.file_globs)
    sed_args = ' '.join(f'-e {shlex.quote(f"s/{source}/{target}/")}' for source, target in patch.replacements)
    return (
        f'      if [ "$app_name" = {shlex.quote(patch.app_name)} ]; then \\\n'
        f'        find "$app_dir/src" -type f \\( {file_match} \\) -exec sed -i {sed_args} {{}} +; \\\n'
        '      fi; \\'
    )


def _hub_patches_dockerfile() -> str:
    return '\n'.join(_hub_patch_dockerfile(patch) for patch in HUB_BUILD_PATCHES)


def _hub_prebuild_dockerfile(hub_apps: Iterable[str]) -> str:
    app_dirs = ' '.join(shlex.quote(f'{CUA_GYM_HUB_ROOT}/websites/{app}') for app in _unique_hub_apps(hub_apps))
    if not app_dirs:
        return f'RUN mkdir -p {CUA_GYM_HUB_ROOT} && chown -R user:user {CUA_GYM_HUB_ROOT}\n'
    patch_commands = _hub_patches_dockerfile()
    return f"""\
RUN set -eux; \\
    for app_dir in {app_dirs}; do \\
      test -f "$app_dir/package.json"; \\
      cd "$app_dir"; \\
      app_name="$(basename "$app_dir")"; \\
{patch_commands}
      npm install --include=dev --legacy-peer-deps --no-audit --no-fund; \\
      npm run build; \\
      rm -rf .mock-states .mock-files; \\
    done; \\
    chown -R user:user {CUA_GYM_HUB_ROOT}
"""


def build_cua_gym_session_dockerfile(
    base_image: str,
    *,
    include_hub: bool = False,
    hub_apps: Iterable[str] | None = None,
) -> str:
    """Build a Dockerfile that extends the shared GUI image with CUA-Gym dependencies."""
    selected_hub_apps = DEFAULT_CUA_GYM_HUB_APPS if hub_apps is None else _unique_hub_apps(hub_apps)
    hub_copy = (
        f'COPY --chown=user:user hub {CUA_GYM_HUB_ROOT}\n{_hub_prebuild_dockerfile(selected_hub_apps)}'
        if include_hub
        else ''
    )
    return f"""\
FROM {base_image}
ENV DEBIAN_FRONTEND=noninteractive
ENV NEXUS_CUA_GYM_HUB_ROOT={CUA_GYM_HUB_ROOT}
ENV PIP_BREAK_SYSTEM_PACKAGES=1
COPY --chown=user:user reward_judge.py /tmp/reward_judge.py
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
      build-essential \\
      ca-certificates \\
      curl \\
      eog \\
      evince \\
      ffmpeg \\
      gedit \\
      git \\
      imagemagick \\
      python3-tk \\
      python3-pip \\
      python3-uno \\
      python3-venv \\
      sudo \\
      unzip \\
      xdotool \\
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \\
    && apt-get install -y --no-install-recommends nodejs \\
    && npm --version >/dev/null \\
    && python3 -m pip install --break-system-packages --no-cache-dir \\
      beautifulsoup4 \\
      lxml \\
      opencv-python-headless \\
      odfpy \\
      openai \\
      openpyxl \\
      pandas \\
      pillow \\
      pikepdf \\
      pymupdf \\
      PyPDF2 \\
      pyautogui \\
      python-docx \\
      python-pptx \\
      requests \\
      scikit-image \\
      reportlab \\
      msoffcrypto-tool \\
      ezodf \\
      pyyaml \\
      virtualenv \\
      fpdf2 \\
      flask \\
      django \\
      sqlalchemy \\
      pytest \\
    && echo 'user:password' | chpasswd \\
    && usermod -aG sudo user \\
    && printf 'user ALL=(ALL) NOPASSWD:ALL\\n' > /etc/sudoers.d/nexus-user \\
    && chmod 0440 /etc/sudoers.d/nexus-user \\
    && mkdir -p {CUA_GYM_HUB_ROOT} \\
    && chown -R user:user {CUA_GYM_HUB_ROOT} \\
    && rm -rf /var/lib/apt/lists/*
ENV PIP_NO_INDEX=1
{hub_copy}"""


def resolve_cua_gym_session_image(image: str | None = None) -> str:
    """Resolve the CUA-Gym session image tag for GUI flows."""
    return image or os.getenv(CUA_GYM_SESSION_IMAGE_ENV, DEFAULT_CUA_GYM_SESSION_IMAGE)


def ensure_cua_gym_session_image(
    image: str | None = None,
    *,
    base_image: str | None = None,
    hub_source: str | None = None,
    hub_apps: Iterable[str] | None = None,
    force: bool = False,
    stream_build_output: bool = False,
) -> str:
    """Build the CUA-Gym session image if it is missing or explicitly forced."""
    resolved_image = resolve_cua_gym_session_image(image)
    should_force = force or os.getenv(CUA_GYM_SESSION_IMAGE_FORCE_BUILD_ENV) == '1'
    if (
        _docker(f'image inspect {shlex.quote(resolved_image)} >/dev/null 2>&1', check=False).returncode == 0
        and not should_force
    ):
        if stream_build_output:
            print(f'Image {resolved_image} already exists, skipping build')
        return resolved_image

    resolved_base = base_image or os.getenv('NEXUS_UBUNTU_SESSION_IMAGE', DEFAULT_UBUNTU_SESSION_IMAGE)
    with tempfile.TemporaryDirectory(prefix='nexus-cua-gym-session-') as tmp:
        context = Path(tmp)
        include_hub = bool(hub_source)
        shutil.copyfile(PACKAGED_REWARD_JUDGE_PATH, context / 'reward_judge.py')
        if hub_source:
            shutil.copytree(Path(hub_source), context / 'hub')
        (context / 'Dockerfile').write_text(
            build_cua_gym_session_dockerfile(resolved_base, include_hub=include_hub, hub_apps=hub_apps),
            encoding='utf-8',
        )
        if stream_build_output:
            print(f'Building image {resolved_image} from base {resolved_base}...')
        _docker(
            f'build -t {shlex.quote(resolved_image)} {shlex.quote(str(context))}',
            capture_output=not stream_build_output,
        )
    return resolved_image


@command_registry.register('build-cua-gym-session-image')
class BuildCuaGymSessionImageCommand:
    """CLI command for building the CUA-Gym desktop session Docker image."""

    @staticmethod
    def register_parser(subparsers, command_name: str) -> None:
        parser = subparsers.add_parser(command_name, help='Build the CUA-Gym desktop session Docker image')
        parser.add_argument('--image', '-i', help=f'Image tag to build (default: {DEFAULT_CUA_GYM_SESSION_IMAGE})')
        parser.add_argument('--base-image', help=f'Base GUI image (default: {DEFAULT_UBUNTU_SESSION_IMAGE})')
        parser.add_argument('--hub-source', help='Optional local CUA-Gym-Hub directory to bake into the image')
        parser.add_argument(
            '--hub-app',
            action='append',
            dest='hub_apps',
            help='CUA-Gym-Hub app directory to prebuild. May be repeated. Defaults to train mock apps.',
        )
        parser.add_argument('--force', '-f', action='store_true', help='Force rebuild even if the image exists')

    @staticmethod
    def handle_command(args) -> int:
        image = ensure_cua_gym_session_image(
            image=args.image,
            base_image=args.base_image,
            hub_source=args.hub_source,
            hub_apps=args.hub_apps,
            force=args.force,
            stream_build_output=True,
        )
        print(f'CUA-Gym session image ready: {image}')
        return 0


__all__ = [
    'BuildCuaGymSessionImageCommand',
    'CUA_GYM_HUB_ROOT',
    'CUA_GYM_SESSION_IMAGE_ENV',
    'CUA_GYM_SESSION_IMAGE_FORCE_BUILD_ENV',
    'DEFAULT_CUA_GYM_HUB_APPS',
    'DEFAULT_CUA_GYM_SESSION_IMAGE',
    'build_cua_gym_session_dockerfile',
    'ensure_cua_gym_session_image',
    'resolve_cua_gym_session_image',
]
