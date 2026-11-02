# HITL 开发约定

本文介绍 HITL 开发和扩展的相关信息和约束条件。

---

## 改动范围

HITL 默认改动应该限制在 `nexus/extensions/tasks/hitl/` 之内，尽可能不直接修改 nexus 的公共模块和其他扩展模块，最大程度保留与 nexus 主分支的兼容性。所需的定制化行为行为可以优先在 hitl 内部等价实现：继承上游的类、包一层 adapter，或把注入点挪到自己这一侧。对于必须修改上游才能得到的行为，应该直接和主分支维护者提出需求或合并请求。

`nexus/trace/`、`nexus/runtime/`、`nexus/core/`、`nexus/runner/` 与 `nexus/extensions/agents/` 是 HITL 共享代码，对他们的直接改动应该进行谨慎评估和讨论，以避免对其他 HITL 任务造成影响。recipe 特有的逻辑、资源和配置应该全部布局在recipe目录内部，同样避免感染其他 HITL 任务。

---

## 依赖方向

- `actor/` 与 `user_agent/` 为相互隔离的组件实现，两者共同依赖 `trace/`相关设施。
- `trace/`、`model/`、`utils/`、`mcp/` 为 HITL 基础设施，可由各个模块自行调用。
- `agents/` 与 `recipe/` 为组件的组装，agents 默认提供可隔离使用的 nexus agent，recipe 提供完整定义的nexus任务、他所依赖的所有资源和特殊实现以及相应的 agent 实现。
- `utils/generator_util/` 提供了供 generator 使用的 helper，可用于简化 generator的逻辑。

---

## 注释

HITL 内注释描述代码做什么，以及调用方必须知道的约定，需要尽可能避免将动机、取舍与缺陷历史直接放入文件中影响阅读。
