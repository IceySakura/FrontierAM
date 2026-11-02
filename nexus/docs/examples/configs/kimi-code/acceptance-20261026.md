# Kimi Code 2.3.2：Nexus 接入验收

更新：2026-10-26T09:28:45.653240+00:00。

**结论：可以接入 Nexus，Gongfeng 上的工具诊断及三道 SWE smoke 已全部通过。**

本次使用官方 TypeScript CLI `@moonshot-ai/kimi-code@2.3.2`，注册名 `kimi_code`，分支 `zkc/kimi-code`。
模型为 `HY4-T3-A49B-SFT1008-RL1009-Penalty-step460-r200`，high effort。
这验证的是接入与评测链路；3题不是全量能力分数。

## 结果

| case | 集合 | score | 调用 | 工具 | 输入 token | 缓存 token | 输出 token | patch B | Langfuse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| instance_protonmail__webclients-5e815cfa518b223a088fa9bb232a5fc90ab15691 | swe-bench_pro | 1.0 | 119 | 203 | 20,013,215 | 19,780,941 | 81,759 | 12,279 | [trace](https://langfuse-taiji.woa.com/project/cmvjo153501jrwc08tjkzezuy/traces/0f04e228597639b0aa84dc1ed0e19e4e) |
| nushell__nushell-12950 | swe-bench-multilingual | 1.0 | 91 | 97 | 7,516,160 | 7,398,317 | 47,536 | 3,869 | [trace](https://langfuse-taiji.woa.com/project/cmvjo153501jrwc08tjkzezuy/traces/2f7643a9f5f813b9d4434f82ecf35537) |
| preactjs__preact-3010 | swe-bench-multilingual | 1.0 | 76 | 122 | 8,518,168 | 8,370,741 | 45,179 | 5,296 | [trace](https://langfuse-taiji.woa.com/project/cmvjo153501jrwc08tjkzezuy/traces/1105914b2765f866f31526a4f903b645) |

三题均退出码0，官方测试判分成功，原始 anti-hack mask 均为false。
完整 provider request/response、thinking、token与cache、工具参数和返回已逐条与 Langfuse 核对。
独立从保存的官方测试日志重新判分，3/3与原始结果一致。

## 配置和协议

Nexus 基于 origin/main `cb769e34b7ef2fe6191d51143fa790bbd2e6005a`，新增文件只在 `nexus/extensions/agents/kimi_code/`、对应测试和示例文档。
公共代理及SWE判分未改。

```text
POST http://infer-proxy-yb-test.production.polaris:8000/openapi/chat/completions
Authorization: Bearer EMPTY
StrategyType: ConsistentHash
SessionID: 每个实验/阶段/任务固定且唯一
```

```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "max_tokens": 16384,
  "stream": false,
  "chat_template_kwargs": {
    "reasoning_effort": "high",
    "preserved_thinking": true,
    "interleaved_thinking": true
  }
}
```

`kimi --model nexus --output-format stream-json --prompt <task>`。
prompt模式内部启用auto权限，不能再传--auto/--yolo。
独立KIMI_CODE_HOME，禁止自动更新，关闭CLI自身遥测，Node≥22.19.0。
每题Gongfeng 4CPU/16GiB，worker venv上传本分支wheel，SWE每题使用自己的镜像。

## 已保留的限制和异常

- 镜像在2026-10-26返回latest=2.3.2、发布时间2026-10-25。节点直连GitHub和官方npm超时，版本结论来自发布包/镜像与实际CLI，不声称核对过GitHub最新commit。
- 首轮pilot的额外准备脚本漏转义find -exec花括号，3题在LLM前初始化失败。原批次保留，零调用核实后在pilot-config-fixed重跑。
- 2.3.2在退出后的文件历史清理报workspace is not materialized。四次成功运行均保留该日志；补丁、结束标记、工具结果和官方判分不受影响。这是仍需向上游排查的清理问题。
- 运行时遇到Ceph读写延迟和Socket.IO重连，原始耗时保留，不把它作为速度对比结论。
- 测试覆盖Linux x64 Gongfeng直接执行与OpenAI chat代理；其他架构、ACP、交互模式和更多模型需单独验证。

## 复现

可运行SOP与配置：`/apdcephfs_zwfy10/share_302593112/kechizhang/from_zwfy/mimo-nexus/nexus-kimi-code/docs/examples/configs/kimi-code/README.md`。
本实验目录：`/apdcephfs_zwfy10/share_302593112/kechizhang/from_zwfy/mimo-nexus/nexus/experiments/kimi_code_acceptance_20261026`。

先设置ignored私密环境文件中的Gongfeng和Langfuse值；固定模型参数、CLI版本和数据hash。

1. 工具诊断：driver.py telemetry → audit.py telemetry。独立读取sum.txt=36；5次LLM、6个工具核对通过。
2. SWE：固定Pro1+Multi2，最大并发3。driver.py pilot-config-fixed；使用独立stage，不覆盖旧DB。
3. audit.py pilot-config-fixed；grade_replay.py pilot-config-fixed。
4. resources.py --stop-owned-terminal，核对本实验归属，10/10已停止。
5. 校验实际CLI版本和SHA、所有原始补丁及判分记录后生成本报告。

新实验应建立新目录及新的manifest。这里的driver拒绝覆盖已存在stage，不要复制旧gate来绕过新验收。

## 证据位置

- `runtime_data/pilot-config-fixed/rollout.db.shards/`：原始结果与轨迹。
- `audit/pilot-config-fixed/`：逐条Langfuse核验。
- `grading_replay/pilot-config-fixed/`：独立判分复核。
- `analysis/resources.json`：10个沙箱全部停止。
- `metadata/`：包元数据、help、doctor和下载完整性。
- `preflight/`：配置预检、三题生成结果、首次失败恢复依据。

单元测试8项通过，ruff和ty检查通过。覆盖超时/取消子进程回收、Harbor超时分类、配置作用域和模型环境隔离。
所有DB、日志、缓存、下载和临时数据保存在Ceph实验目录，密钥不写入报告。
