"""Host-side cb-insiders agent registration and runtime invocation."""

from __future__ import annotations

import os

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator

from nexus.core import agent_registry
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)

from .installer import DEFAULT_VERSION, validate_version
from .runtime_setup import DEFAULT_MODEL_ID, validate_extra_args


class PreStartScript(BaseModel):
    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    timeout: float = Field(default=300.0, gt=0)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class CbInsidersInput(InstalledNodeCliInput):
    version: str = DEFAULT_VERSION
    npm_username: str | None = Field(default=None, repr=False)
    npm_token: SecretStr | None = Field(default=None, repr=False)
    package_path: str | None = None
    """Explicit development fallback: path to the verified deb inside the runtime."""

    auto_install_system_deps: bool = True
    model_id: str = DEFAULT_MODEL_ID
    effort: Literal['minimal', 'low', 'medium', 'high', 'xhigh', 'max'] | None = 'high'
    permission_mode: Literal['default', 'plan', 'acceptEdits', 'bypassPermissions', 'dontAsk'] = 'bypassPermissions'
    max_input_tokens: int = Field(default=200000, gt=0)
    max_output_tokens: int = Field(default=16384, gt=0)
    extra_env: dict[str, str] = Field(default_factory=dict)
    extra_cli_args: list[str] = Field(default_factory=list)
    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)

    @field_validator('version')
    @classmethod
    def exact_version(cls, value: str) -> str:
        return validate_version(value)

    @field_validator('extra_cli_args')
    @classmethod
    def managed_flags(cls, value: list[str]) -> list[str]:
        validate_extra_args(value)
        return value


class CbInsidersOutput(InstalledAgentOutput):
    status: str | None = None
    # Harbor and other task adapters read this standard field, not CLI status.
    exit_status: str | None = Field(default=None, validation_alias=AliasChoices('exit_status', 'status'))
    assistant_text: str | None = None
    session_id: str | None = None
    transcript_path: str | None = None
    permission_denials: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None


@agent_registry.register('cb-insiders')
async def run_cb_insiders(input_data: CbInsidersInput) -> CbInsidersOutput:
    from .runtime_setup import setup_and_run_cb_insiders

    username, token = None, None
    credentials = [
        (input_data.npm_username, input_data.npm_token.get_secret_value() if input_data.npm_token else None),
        (os.getenv('CB_INSIDERS_NPM_USERNAME'), os.getenv('CB_INSIDERS_NPM_TOKEN')),
        (os.getenv('WORKBUDDY_NPM_USERNAME'), os.getenv('WORKBUDDY_NPM_TOKEN')),
    ]
    for candidate_user, candidate_token in credentials:
        if candidate_user or candidate_token:
            if not candidate_user or not candidate_token:
                raise ValueError('npm_username and npm_token must both be set from the same configuration source')
            username, token = candidate_user, candidate_token
            break
    params = input_data.model_dump(exclude={'user_prompt', 'npm_username', 'npm_token'})
    result = await call_runtime_agent(
        setup_and_run_cb_insiders,
        task=coerce_user_prompt(input_data.user_prompt),
        npm_username=username,
        npm_token=token,
        **params,
    )
    # Bridge accounting fields remain in traces and are not part of CLI output.
    return CbInsidersOutput.model_validate({
        key: value for key, value in result.items() if key in CbInsidersOutput.model_fields
    })


__all__ = ['CbInsidersInput', 'CbInsidersOutput', 'run_cb_insiders']
