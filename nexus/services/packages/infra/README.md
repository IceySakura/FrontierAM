# Infra

供 `services/` 下微服务复用的基础设施包，不包含业务模型或服务专属配置。

## 日志

服务启动时统一配置结构化日志：

```python
from infra.logging import bind_log_context, configure_logging

configure_logging('example-service', level='INFO')

with bind_log_context(request_id='req-1', execution_id='exe-1'):
    logger.info('processing request')
```

默认输出 JSON 到标准输出；存在有效 OTel Span 时会自动写入 `trace_id` 和 `span_id`。本地开发可设置 `json_output=False` 使用文本格式。
