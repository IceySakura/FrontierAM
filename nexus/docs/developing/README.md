# Developing Notes

Store commit-worthy development notes or design documents here only when they are intentionally part of the repository history.

- [Nexus Cloud 平台可观测性 MVP](distributed-platform-observability-mvp.md)：平台 P0 指标、OTel/智研上报、实时概览、状态快照、最长 30 天的 Metrics、按需业务统计 Tab 和性能验收。
- [导出内存与完整性设计](export-memory-fix-design.md)：SQLite 导出流式读取与原子写、执行期增量导出语义、孤立代理处理，以及 Cloud results 的免解析校验、按索引过滤与 strict 降级。
- [共享请求预算与 Client Policy 接入](shared-request-budget-policy-design.md)：每个 Manager 固定绑定一个请求策略，由 RequestBudgetControl 统一负责租约续期与本地限流，复用控制库连接协调共享预算；明确数据流、一致性与旧 Batch 联动的迁移边界。
