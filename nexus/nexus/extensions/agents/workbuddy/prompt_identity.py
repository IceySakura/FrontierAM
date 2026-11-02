"""Load the installed WorkBuddy prompt plugins and verify requests before inference."""

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any


WORKBUDDY_PREAMBLE = 'This conversation is powered by '
WORKBUDDY_TASK_MARKER = (
    "Your main goal is to follow the USER's instructions at each message, denoted by the <user_query> tag."
)
TITLE_PREAMBLE = 'Generate a concise, sentence-case title'
SUMMARY_MARKER = 'You are a helpful AI assistant tasked with summarizing conversations.'


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(
            block['text'] for block in value if isinstance(block, dict) and isinstance(block.get('text'), str)
        )
    return ''


def is_workbuddy_prompt(system: str) -> bool:
    return system.lstrip().startswith(WORKBUDDY_PREAMBLE) and WORKBUDDY_TASK_MARKER in system


def prepare_builtin_prompt_plugins(wb_js: str | Path, config_dir: Path, preset: str, mode: str) -> dict[str, Any]:
    """Register local resources; a missing plugin must not silently select CodeBuddy.

    The path comes from the installed entrypoint, covering both npm and tarball
    layouts. Other settings and unrelated plugins remain intact.
    """
    root = Path(wb_js).resolve().parent.parent / 'resources' / 'plugins' / 'workbuddy-builtin'
    manifest = json.loads((root / '.codebuddy-plugin' / 'marketplace.json').read_text(encoding='utf-8'))
    if manifest.get('name') != 'workbuddy-builtin':
        raise ValueError('The installed package has no workbuddy-builtin marketplace')
    catalog = {item['name']: item for item in manifest['plugins']}
    selected = (f'welcomemode-{preset}', f'interactionmode-{mode}', 'prompt-common')
    for name in selected:
        entry = catalog.get(name)
        if not entry or not isinstance(entry.get('source'), str):
            raise ValueError(f'WorkBuddy prompt plugin is missing: {name}')
        plugin_dir = (root / entry['source']).resolve()
        if not plugin_dir.is_relative_to(root.resolve()):
            raise ValueError(f'WorkBuddy prompt plugin is outside the installed package: {name}')
        plugin = json.loads((plugin_dir / '.workbuddy-plugin' / 'plugin.json').read_text(encoding='utf-8'))
        if plugin.get('name') != name:
            raise ValueError(f'WorkBuddy prompt plugin identity differs: {name}')
        if name.startswith('welcomemode-'):
            if not (plugin_dir / 'prompt.tpl').is_file():
                raise ValueError(f'WorkBuddy prompt template is missing: {name}')
            agents = plugin.get('agents') or []
            if not agents or any(not (plugin_dir / agent).is_file() for agent in agents):
                raise ValueError(f'WorkBuddy root agent is missing: {name}')

    path = config_dir / 'settings.json'
    settings = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(settings, dict):
        raise ValueError('WorkBuddy settings must be a JSON object')
    enabled = settings.setdefault('enabledPlugins', {})
    marketplaces = settings.setdefault('extraKnownMarketplaces', {})
    if not isinstance(enabled, dict) or not isinstance(marketplaces, dict):
        raise ValueError('WorkBuddy plugin settings must be JSON objects')
    # Only one welcome / interaction mode may supply the root prompt.
    for name in catalog:
        if name.startswith(('welcomemode-', 'interactionmode-')):
            enabled[f'{name}@workbuddy-builtin'] = name in selected
    enabled['prompt-common@workbuddy-builtin'] = True
    marketplaces['workbuddy-builtin'] = {'source': {'source': 'directory', 'path': str(root)}}
    config_dir.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix('.json.tmp')
    staged.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    staged.chmod(0o600)
    staged.replace(path)
    return {'marketplace': str(root), 'enabled_plugins': list(selected), 'preset': preset, 'mode': mode}


class PromptIdentityError(ValueError):
    """A request does not belong to the verified WorkBuddy root or its children."""


class WorkBuddyPromptGuard:
    """Do not infer main identity from a tool catalog shared with CodeBuddy.

    Every WorkBuddy root request must have the actual builtin template. Children
    are admitted only after a model response issued a matching Agent call; they
    may legitimately use CodeBuddy's own prompts. No prompt text leaves the audit.
    """

    def __init__(self) -> None:
        self.workbuddy_requests = 0
        self.sub_agent_requests = 0
        self.summary_requests = 0
        self.rejected_requests = 0
        self.prompt_sha256: set[str] = set()
        self.spawn_prompts: set[str] = set()
        self.failure_reason: str | None = None

    def inspect_response(self, payload: dict[str, Any]) -> None:
        for choice in payload.get('choices') or []:
            for call in (choice.get('message') or {}).get('tool_calls') or []:
                function = call.get('function') or {}
                if str(function.get('name', '')).lower() != 'agent':
                    continue
                args = function.get('arguments') or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        continue
                if isinstance(args, dict):
                    prompt = args.get('prompt') or args.get('description')
                    if isinstance(prompt, str) and prompt.strip():
                        self.spawn_prompts.add(prompt.strip())

    def check(self, body: dict[str, Any]) -> None:
        messages = body.get('messages') or []
        system = '\n'.join(
            text_content(message.get('content'))
            for message in messages
            if isinstance(message, dict) and message.get('role') in ('system', 'developer')
        ).lstrip()
        user = next(
            (
                text_content(message.get('content'))
                for message in messages
                if isinstance(message, dict) and message.get('role') == 'user'
            ),
            '',
        )
        if self.failure_reason:
            self._reject('A previous request failed WorkBuddy prompt identity verification')
        if system.startswith(TITLE_PREAMBLE):
            self._reject('Unexpected session-title request in WorkBuddy rollout mode')
        if is_workbuddy_prompt(system):
            self.workbuddy_requests += 1
            self.prompt_sha256.add(hashlib.sha256(system.encode()).hexdigest())
            return
        if self.workbuddy_requests and SUMMARY_MARKER in system:
            self.summary_requests += 1
            return
        if self.workbuddy_requests and any(prompt in user for prompt in self.spawn_prompts):
            self.sub_agent_requests += 1
            return
        self._reject('Request has no WorkBuddy main template or verified Agent-call parent')

    def _reject(self, reason: str) -> None:
        self.rejected_requests += 1
        self.failure_reason = self.failure_reason or reason
        raise PromptIdentityError(reason)

    def report(self) -> dict[str, Any]:
        return {
            'verified': self.workbuddy_requests > 0 and self.rejected_requests == 0,
            'workbuddy_requests': self.workbuddy_requests,
            'sub_agent_requests': self.sub_agent_requests,
            'summary_requests': self.summary_requests,
            'rejected_requests': self.rejected_requests,
            'system_sha256': sorted(self.prompt_sha256),
            'failure_reason': self.failure_reason,
        }
