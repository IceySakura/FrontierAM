"""aidata engine —— 框架核心（不含 case 数据）。

从原本单体的 ``task.py`` 拆分出来，让单独的 case 包能够独立发布，无需拖着整个引擎一起走。
引擎由一小组单一职责的模块组成：

- :mod:`paths`         沙箱布局（agent desktop + 内部 root）
- :mod:`models`        ``AidataTaskInput`` / ``AidataTaskOutput`` pydantic
- :mod:`system_prompt` agent 的静态前置提示
- :mod:`case_spec`     ``cases/<id>/case.yaml`` 加载器 + 验证器
- :mod:`context`       host 侧上下文文件清单（agent 直接读取原始文件）
- :mod:`sandbox`       文件上传助手 + case 资源引导执行器
- :mod:`mcp_catalog`   默认 MCP schema + 按 case 合并 ``extra_mcp_servers``
- :mod:`state`         运行后 mock 状态下载 + 精简摘要
- :mod:`rubric`        LLM-as-judge 评估器
- :mod:`task`          薄壳 ``@task_registry.register('aidata')`` 入口，
                        用于编排以上模块

此处所有内容都与具体 case 无关。 case 数据位于 ``../cases/<id>/`` 下，
通过 ``case_spec`` 发现。每个 case 目录设计为自包含的 tarball，
可交付他人并仍可运行 —— 这也是引擎 / case 拆分的原因。
"""
