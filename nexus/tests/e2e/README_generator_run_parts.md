# Generator / Run Part 本地 E2E

在仓库根目录运行：

```bash
uv run pytest tests/e2e/test_generator_run_parts.py -m e2e -q
```

需要项目的 dev 依赖。测试不需要模型凭据、Docker 或 Cloud 集群，通过独立 CLI 进程
覆盖“生成/选择 → 合并配置 → worker → task → SQLite → 另一个 CLI 进程导出”的链路。
task 本身启动真实 Python 子进程计算并写出文件产物，不 mock Runner、任务结果或数据库。

| 场景 | 验证 |
|---|---|
| generator 与 data 等价 | 普通覆盖及环境变量 timeout 覆盖均得到相同有效配置、业务结果及 metadata |
| 双 worker + limit | 实际使用两个 worker，只生成前三条，关闭生成器，完整目录导出正确 |
| 首次失败再重试 | 真实失败后第二次成功，重试计数正确，覆盖配置保持不变 |
| 恢复锁定配置 | 修改 Run Part 并传 limit=0 后重跑，仍使用已有任务和配置，不再次生成 |
| 单／双 worker 恢复 | 覆盖文件删除、损坏，注册项不可用或覆盖表达式非法时，仍使用已锁定配置完成重跑和导出 |
| limit=0 单 worker | 不生成条目，不启动 task，不写任务产物，显式 JSONL 导出为空 |
| limit=0 双 worker | 同样的空批次语义 |

每个用例使用 pytest 临时目录。目录内保存 CLI 日志、generator 消费记录、任务尝试次数、
子进程计算产物、SQLite 数据库以及 JSONL/full 导出，失败信息会给出日志路径。
父 CLI 和任务子进程都设置超时，测试不会改生产配置或调用外部模型。
