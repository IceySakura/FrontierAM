"""Opt-in compatibility shims for older WorkBuddy CLI releases."""

from __future__ import annotations

from typing import Any, Literal


DEFAULT_COMPAT_PERMISSION_MODE_PATCH = False

ReplacementRule = tuple[Literal['str', 'regex'], str, str]
VersionFilePatches = dict[str, list[ReplacementRule]]

# Versions whose minified shape the patch below has actually been checked against — not a
# claim that other releases are incompatible. The anchors carry mangled identifiers and a
# content-hashed filename, so a rebuild can move them; an anchor that silently stops matching
# looks exactly like the bug this patch exists to fix. Add a version only after re-confirming
# every anchor still matches exactly once.
PERMISSION_MODE_PATCH_SPECS: dict[str, VersionFilePatches] = {
    '5.7.0': {
        'lib/wb.js': [
            (
                'str',
                '"remote-debugging-port":{type:`string`}};',
                '"remote-debugging-port":{type:`string`},"permission-mode":{type:`string`}};',
            ),
            (
                'str',
                'json:!!n.json,...c?',
                'json:!!n.json,...n["permission-mode"]?{permissionMode:n["permission-mode"]}:{},...c?',
            ),
            (
                'str',
                '...d===void 0?{}:{remoteDebuggingPort:d},positionals:[...r]',
                '...n["permission-mode"]?{permissionMode:n["permission-mode"]}:{},...d===void 0?{}:{remoteDebuggingPort:d},positionals:[...r]',
            ),
            (
                'str',
                'runHeadless:r}=await Promise.resolve().then(()=>require("./headless-main-DvKAzMqY.js"));process.exit(await r({prompt:n,model:e.model,effort:e.effort,',
                'runHeadless:r}=await Promise.resolve().then(()=>require("./headless-main-DvKAzMqY.js"));process.exit(await r({prompt:n,permissionMode:e.permissionMode,model:e.model,effort:e.effort,',
            ),
        ],
        'lib/headless-main-DvKAzMqY.js': [
            (
                'str',
                'cwd:process.cwd(),model:en.model,',
                'cwd:process.cwd(),permissionMode:en.permissionMode,model:en.model,',
            ),
            (
                'str',
                'space:{type:`local`},...tn.model?',
                'space:{type:`local`},...tn.permissionMode?{permissionMode:tn.permissionMode}:{},...tn.model?',
            ),
        ],
    },
    '5.9.0': {
        'lib/wb.js': [
            (
                'str',
                '"remote-debugging-port":{type:`string`}};',
                '"remote-debugging-port":{type:`string`},"permission-mode":{type:`string`}};',
            ),
            (
                'str',
                'json:!!n.json,...c?',
                'json:!!n.json,...n["permission-mode"]?{permissionMode:n["permission-mode"]}:{},...c?',
            ),
            (
                'str',
                '...p===void 0?{}:{remoteDebuggingPort:p},positionals:[...r]',
                '...n["permission-mode"]?{permissionMode:n["permission-mode"]}:{},...p===void 0?{}:{remoteDebuggingPort:p},positionals:[...r]',
            ),
            (
                'str',
                'runHeadless:r}=await Promise.resolve().then(()=>require("./headless-main-BXJWea3y.js"));process.exit(await r({prompt:n,model:e.model,effort:e.effort,',
                'runHeadless:r}=await Promise.resolve().then(()=>require("./headless-main-BXJWea3y.js"));process.exit(await r({prompt:n,permissionMode:e.permissionMode,model:e.model,effort:e.effort,',
            ),
        ],
        'lib/headless-main-BXJWea3y.js': [
            (
                'str',
                'cwd:process.cwd(),model:r.model,',
                'cwd:process.cwd(),permissionMode:r.permissionMode,model:r.model,',
            ),
            (
                'str',
                'space:{type:`local`},...D.model?',
                'space:{type:`local`},...D.permissionMode?{permissionMode:D.permissionMode}:{},...D.model?',
            ),
        ],
    },
    # Same anchors as 5.9.0 apart from the content-hashed headless-main name.
    '5.11.0-hotfix.202610272104': {
        'lib/wb.js': [
            (
                'str',
                '"remote-debugging-port":{type:`string`}};',
                '"remote-debugging-port":{type:`string`},"permission-mode":{type:`string`}};',
            ),
            (
                'str',
                'json:!!n.json,...c?',
                'json:!!n.json,...n["permission-mode"]?{permissionMode:n["permission-mode"]}:{},...c?',
            ),
            (
                'str',
                '...p===void 0?{}:{remoteDebuggingPort:p},positionals:[...r]',
                '...n["permission-mode"]?{permissionMode:n["permission-mode"]}:{},...p===void 0?{}:{remoteDebuggingPort:p},positionals:[...r]',
            ),
            (
                'str',
                'runHeadless:r}=await Promise.resolve().then(()=>require("./headless-main-Cp4Kg6pD.js"));process.exit(await r({prompt:n,model:e.model,effort:e.effort,',
                'runHeadless:r}=await Promise.resolve().then(()=>require("./headless-main-Cp4Kg6pD.js"));process.exit(await r({prompt:n,permissionMode:e.permissionMode,model:e.model,effort:e.effort,',
            ),
        ],
        'lib/headless-main-Cp4Kg6pD.js': [
            (
                'str',
                'cwd:process.cwd(),model:r.model,',
                'cwd:process.cwd(),permissionMode:r.permissionMode,model:r.model,',
            ),
            (
                'str',
                'space:{type:`local`},...D.model?',
                'space:{type:`local`},...D.permissionMode?{permissionMode:D.permissionMode}:{},...D.model?',
            ),
        ],
    },
}
VERIFIED_PERMISSION_MODE_PATCH_VERSIONS = frozenset(PERMISSION_MODE_PATCH_SPECS)
SUPPORTED_PERMISSION_MODES = frozenset({
    'default',
    'acceptEdits',
    'plan',
    'auto',
    'dontAsk',
    'bypassPermissions',
    'fullAccess',
})


def _is_dist_tag(version: str) -> bool:
    return not version[:1].isdigit()


def _require_verified(version: str) -> None:
    """Reject an exact version that has no spec; a dist-tag is checked against the installed release instead."""
    if not _is_dist_tag(version) and version not in PERMISSION_MODE_PATCH_SPECS:
        raise ValueError(
            f'compat_permission_mode_patch has only been verified against WorkBuddy '
            f'{sorted(PERMISSION_MODE_PATCH_SPECS)}, got {version!r}; '
            f'pin WORKBUDDY_VERSION to one of them, or add this release to PERMISSION_MODE_PATCH_SPECS in compat.py'
        )


def _permission_mode_patch_script(version: str = '5.7.0') -> str:
    # The installer already pins an exact version. A dist-tag only names a release once
    # installed, so the script then picks the spec by package.json and refuses an unverified one.
    _require_verified(version)
    if _is_dist_tag(version):
        specs = PERMISSION_MODE_PATCH_SPECS
        installed = 'json.loads((root / "package.json").read_text()).get("version")'
    else:
        specs = {version: PERMISSION_MODE_PATCH_SPECS[version]}
        installed = repr(version)
    # No f-strings in the generated snippet: it runs under whatever interpreter the testbed image provides.
    return (
        'import json\n'
        'import re\n'
        'from pathlib import Path\n'
        'roots = [\n'
        '    Path("/tmp/nexus/agents/workbuddy/node_modules/@tencent/nexus-wb-cli"),\n'
        '    Path("/tmp/nexus/agents/workbuddy/package"),\n'
        ']\n'
        'root = next((candidate for candidate in roots if candidate.is_dir()), None)\n'
        'if root is None:\n'
        '    raise RuntimeError("WorkBuddy install not found under: %r" % (roots,))\n'
        f'installed = {installed}\n'
        f'specs = {specs!r}\n'
        f'requested = {version!r}\n'
        'if installed not in specs:\n'
        '    raise RuntimeError(\n'
        '        "compat_permission_mode_patch has no verified spec for installed WorkBuddy %r (requested %r); "\n'
        '        "pin WORKBUDDY_VERSION to one of %s, or add this release to PERMISSION_MODE_PATCH_SPECS in compat.py"\n'
        '        % (installed, requested, sorted(specs))\n'
        '    )\n'
        'file_patches = specs[installed]\n'
        'for relpath, rules in file_patches.items():\n'
        '    matches = sorted(root.glob(relpath)) if any(ch in relpath for ch in "*?[") else [root / relpath]\n'
        '    if len(matches) != 1 or not matches[0].is_file():\n'
        '        raise RuntimeError("WorkBuddy patch target %s matched %r" % (relpath, matches))\n'
        '    target = matches[0]\n'
        '    s = target.read_text()\n'
        '    for mode, old, new in rules:\n'
        '        if mode == "str":\n'
        '            count = s.count(old)\n'
        '            if count != 1:\n'
        '                raise RuntimeError("WorkBuddy %s shape changed; %d matches for: %s" % (relpath, count, old))\n'
        '            s = s.replace(old, new, 1)\n'
        '        elif mode == "regex":\n'
        '            s, count = re.subn(old, new, s)\n'
        '            if count != 1:\n'
        '                raise RuntimeError("WorkBuddy %s shape changed; %d regex matches for: %s" % (relpath, count, old))\n'
        '        else:\n'
        '            raise ValueError("Unsupported patch mode: %r" % (mode,))\n'
        '    target.write_text(s)\n'
    )


def build_compatibility_patches(version: str, permission_mode: str) -> dict[str, Any]:
    _require_verified(version)
    if permission_mode not in SUPPORTED_PERMISSION_MODES:
        raise ValueError(f'unsupported permission_mode: {permission_mode!r}')
    return {
        'extra_cli_args': ['--permission-mode', permission_mode],
        'config_file_patches': [
            {
                # User settings live in PathUtils.getHomeDir() = $CODEBUDDY_CONFIG_DIR, which the wb.js wrapper pins to $WORKBUDDY_CONFIG_DIR. The literal ~/.codebuddy is only the fallback for a bare agent process and is never read through the wrapper.
                'path': '~/.workbuddy/settings.json',
                'content': {
                    'permissions': {
                        'defaultMode': permission_mode,
                        'deny': ['WebSearch', 'AskUserQuestion'],
                    },
                    'autoUpdates': False,
                    'promptSuggestionEnabled': False,
                    'enableAllProjectMcpServers': False,
                },
            }
        ],
        'pre_start_scripts': [{'interpreter': 'python', 'code': _permission_mode_patch_script(version), 'timeout': 30}],
    }
