from __future__ import annotations

import os

from typing import Annotated, ClassVar

from pydantic import BaseModel, Field

from infra.config import ConfigHolder, Key, RainbowKey, bind, load_from_rainbow


class AgsConfig(BaseModel):
    secret_id: Annotated[str | None, Key('secretId')] = None
    secret_key: Annotated[str | None, Key('secretKey')] = None
    tool_id: Annotated[str | None, Key('toolId')] = None
    region: str = 'ap-guangzhou'
    domain: str | None = None


class DevCloudConfig(BaseModel):
    app_id: Annotated[str | None, Key('appId')] = None
    app_token: Annotated[str | None, Key('appToken')] = None
    sandbox_api_base_url: Annotated[str, Key('sandboxApiBaseUrl')] = 'http://21.6.221.82:18800'
    control_api_base_url: Annotated[str, Key('controlApiBaseUrl')] = 'http://ratel.devcloud.woa.com'
    project: str = ''
    pipeline_id: Annotated[str, Key('pipelineId')] = 'llm_t3'
    build_id: Annotated[str, Key('buildId')] = ''
    vm_seq_id: Annotated[str, Key('vmSeqId')] = ''


class SmartRunConfig(BaseModel):
    token: str | None = None
    api_base_url: Annotated[str, Key('apiBaseUrl')] = 'https://smartrun.woa.com'
    duration: int = 1800


class ProviderConfig(BaseModel):
    """sandbox provider 配置，对应七彩石中 `provider` 这份 YAML。

    嵌套 model 按字段名下钻：`ags` / `devcloud` / `smartrun` 对应 YAML 中的同名子树。
    """

    rainbow_key: ClassVar[RainbowKey] = RainbowKey(key='provider')

    ags: AgsConfig = Field(default_factory=AgsConfig)
    devcloud: DevCloudConfig = Field(default_factory=DevCloudConfig)
    smartrun: SmartRunConfig = Field(default_factory=SmartRunConfig)


async def load_provider(*, strict: bool = False) -> ProviderConfig:
    """从七彩石加载 provider 配置，同时用于启动加载与热更新重载。

    `strict=True` 时拉取失败抛 `RainbowError`（启动期用），
    否则降级为默认值（热更新用）。
    """
    return await load_from_rainbow(ProviderConfig, strict=strict)


class LabelConfig(BaseModel):
    """label 服务配置。

    进程启动期参数（host/port/log_level）直接读环境变量——它们在配置中心
    可用之前就要确定，无法自举；provider 业务配置来自七彩石。
    """

    service_name: str = 'label'
    host: str = '0.0.0.0'
    port: Annotated[int, Field(ge=1, le=65535)] = 8080
    log_level: str = 'INFO'
    # 日志落盘路径。留空则只输出 stdout；非空时 stdout 与文件双写，
    # 按小时滚动，最多保留 log_backup_count 份（默认 168 份即 7 天）。
    log_file: str | None = None
    log_backup_count: Annotated[int, Field(ge=0)] = 168
    provider: ProviderConfig = Field(default_factory=ProviderConfig)

    @classmethod
    def from_env(cls) -> LabelConfig:
        """只读取进程启动期参数（端口要先确定才能监听）。

        provider 业务配置由 lifespan 在七彩石客户端就绪后加载，
        因此这里是同步的、不触碰网络。
        """
        return bind(cls, _env_bootstrap())


# 承载 label 配置热更新的容器类型：仅 provider 字段参与重载，
# 由调用方直接 `ConfigHolder(config, field='provider', loader=load_provider)` 构造。
LabelConfigHolder = ConfigHolder[LabelConfig, ProviderConfig]


def _env_bootstrap() -> dict[str, str]:
    """读取启动期参数，未设置的键不出现，由 model 默认值兜底。

    值统一为字符串，交由 pydantic 校验转换（如 port 转 int）。
    """
    mapping = {
        'host': os.environ.get('NEXUS_LABEL_HOST'),
        'port': os.environ.get('NEXUS_LABEL_PORT'),
        'log_level': os.environ.get('NEXUS_LABEL_LOG_LEVEL'),
        'log_file': os.environ.get('NEXUS_LABEL_LOG_FILE'),
        'log_backup_count': os.environ.get('NEXUS_LABEL_LOG_BACKUP_COUNT'),
    }
    return {key: value for key, value in mapping.items() if value is not None}
