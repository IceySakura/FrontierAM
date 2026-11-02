# Nexus Services

`services/` 统一组织新建的可部署应用和内部共享包，现有 `nexus/distributed` 保持不动。

`services/apps/` 存放可独立运行、构建、发布和扩缩容的 FastAPI 应用。每个应用应包含独立的 `pyproject.toml`、源码、测试和 README；应用可以依赖根项目 `nexus-agent` 与 `services/packages/` 中的共享包，但禁止直接导入另一个应用的实现。

`services/packages/` 中的项目是 uv workspace 内的独立 Python 包，用于声明依赖、隔离导入边界和独立测试；它们不是可部署应用，也不需要独立仓库、锁文件或虚拟环境。

## 统一命令

```bash
make service-sync
make service-test SERVICE=label
make service-run SERVICE=label
make service-build SERVICE=label
```
