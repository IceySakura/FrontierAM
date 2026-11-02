# 共享包

`services/packages/` 存放可被多个独立服务复用、但不属于 Nexus 核心 SDK 的 Python 包。

每个共享包保留独立的 `pyproject.toml`，作为根 uv workspace 的成员，以获得清晰的依赖和导入边界；它不需要独立仓库、`uv.lock`、虚拟环境或部署流程。

共享代码至少被两个服务需要后再提取到这里。业务实现、数据库模型和服务专属配置应保留在对应的 `services/<name>/` 中。
