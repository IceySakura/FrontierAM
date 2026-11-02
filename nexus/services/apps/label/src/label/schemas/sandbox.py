from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from label.schemas.base import BaseResponse


class Platform(str, Enum):
    WINDOWS = 'windows'
    MACOS = 'macos'
    ANDROID = 'android'


class Provider(str, Enum):
    AGS = 'ags'
    DEVCLOUD = 'devcloud'
    SMARTRUN = 'smartrun'


_PROVIDER_PLATFORM = {
    Provider.AGS: Platform.WINDOWS,
    Provider.DEVCLOUD: Platform.MACOS,
    Provider.SMARTRUN: Platform.ANDROID,
}


class SandboxAcquireRequest(BaseModel):
    platform: Platform
    provider: Provider

    @model_validator(mode='after')
    def validate_provider_platform(self) -> SandboxAcquireRequest:
        expected = _PROVIDER_PLATFORM[self.provider]
        if self.platform != expected:
            raise ValueError(f'provider {self.provider.value} only supports platform {expected.value}')
        return self


class SandboxReleaseRequest(SandboxAcquireRequest):
    instance_id: str | None = None
    instance_ip: str | None = None
    vm_id: str | None = None
    device_id: str | None = None
    session_id: str | None = None

    @model_validator(mode='after')
    def validate_identifiers(self) -> SandboxReleaseRequest:
        required = {
            Provider.AGS: ('instance_id',),
            Provider.DEVCLOUD: ('instance_ip', 'vm_id'),
            Provider.SMARTRUN: ('device_id', 'session_id'),
        }[self.provider]
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(f'missing identifiers for {self.provider.value}: {", ".join(missing)}')
        return self


class SandboxData(BaseModel):
    model_config = ConfigDict(extra='allow')

    instance_id: str | None = None
    instance_ip: str | None = None
    vm_id: str | None = None
    device_id: str | None = None
    session_id: str | None = None
    remote_control_url: str


class SandboxAcquireResponse(BaseResponse):
    data: SandboxData


class SandboxReleaseResponse(BaseResponse):
    pass


class SandboxCallExecutorRequest(BaseModel):
    platform: Literal['windows']
    provider: Literal['ags']
    instance_id: str
    executor_name: str
    executor_args: Any


class SandboxCallExecutorResponse(BaseResponse):
    data: Any
