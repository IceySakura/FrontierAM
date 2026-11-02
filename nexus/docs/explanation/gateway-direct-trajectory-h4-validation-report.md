# Gateway Direct-Trajectory H4 测试与验证总结

> 日期：2026-08-03（Asia/Hong_Kong）
> Nexus H4 code：`1ae27d8b09d67c8d4b46e21e273b672f804ed852`
> HyTuner H4 validation：`d5633441787e78ec853421eefb394a8e62bb2a2c`
> 功能基线：`patrickkklu/gateway-core@6312f04f62a3a45440aa975ca9f53594cd93f63f`

本文只总结 H4 代码及当时围绕 H4 执行的验证。后续 policy inspector/compiler、recognizer 或 declarative
policy revision 的测试结果不计入 H4 结论。

## 1. 结论

H4 的 Direct-Trajectory 主体已经通过本地完整 Gateway suite、两组 baseline differential characterization、静态架构
检查，以及启用 Gateway 和 Qwen legacy seam 的 compaction committed-golden bitwise CI。

可以确认的范围是：

- `TrajectoryAssembler` 直接写 live `Trajectory`，没有 `AssemblyTree`/`Materializer` 中间拓扑；
- `RuntimeState` 只有 NodeId-keyed flat ledger 与可重建 cache，没有第二套 parent/children/roots；
- freeze、chain、correlation、segment、token/mask 与 legacy projection 保持 baseline 语义；
- Generic/nexus_main_agent policy、shared-prefix ownership、compaction 与测试 policy 提供的 multi-agent back-edge facts 均被覆盖；
- HyTuner 仍然只训练 main segment，未启用 child training；
- public `Trajectory` schema 没有 scaffold/profile/framework 字段。

H4 的已知架构边界也没有被测试结果掩盖：自定义 scaffold 可以覆盖完整 `inspect_event()`，框架不会结构性强制其调用
Generic `super()`。这是 extension API 的 review 风险，不是本报告发现的 runtime regression。

## 2. 精确 revision 与 pin 关系

| 角色 | Revision | 说明 |
| --- | --- | --- |
| Nexus functional baseline | `6312f04f62a3a45440aa975ca9f53594cd93f63f` | `patrickkklu/gateway-core` 功能 oracle |
| Nexus H4 code | `1ae27d8b09d67c8d4b46e21e273b672f804ed852` | 本报告验证对象 |
| HyTuner H3 pin | `fea445a8c7e093eb9fb4b5e6a3cacde0aca0e272` | pin Nexus H4，但尚未启用 Qwen legacy seam |
| HyTuner H4 validation | `d5633441787e78ec853421eefb394a8e62bb2a2c` | 在 H3 上为 bitwise runtime env 增加 `GATEWAY_LEGACY_SEAM_COMPAT=1` |

因此，H4 bitwise 的有效组合是：

```text
HyTuner d5633441787e78ec853421eefb394a8e62bb2a2c
  third_party/nexus -> 1ae27d8b09d67c8d4b46e21e273b672f804ed852
```

## 3. 本地测试

### 3.1 Gateway full suite

H4 开发完成时保留的原始结果是：

```text
351 passed, 4 skipped
```

本次从 exact H4 clean worktree 重新执行：

```bash
pytest -q tests/unit/gateway
```

结果是：

```text
355 passed in 2.81s
```

两次收集到的测试总数相同。原始环境有 4 个 optional-dependency skip；本次环境具备相应依赖，所以 355 个全部执行。
这不是 H4 代码变化，也不代表新增了 4 个测试。

### 3.2 Baseline differential characterization

删除旧实现前先以 `6312f04f...` 生成行为 oracle，H4 完成后逐字段比较。保留的历史执行摘要记录了两组独立
characterization/differential collections：

```text
96 cases  — 0 divergence
493 cases — 0 divergence
```

比较面包括：

- `Trajectory.dump()`、dump round-trip 与 pickle；
- `to_segments()`、`to_legacy_segments()`、`to_legacy_hytuner()`；
- NodeId、physical parent/children、roots、chain ID/root、segment ID/order；
- spawn/join/correlation evidence、token IDs、response mask；
- active/final snapshot、cache rebuild、异常时机与 baseline 的 no-rollback 行为；
- shared prefix、chat-template 改写和外部 prefix 变化下的 creation-time ownership；
- compaction boundary、declared-chain 与 main-only HyTuner projection。

### 3.3 Multi-agent 与回边覆盖

H4 production 只启用 Generic 与 Nexus；Nexus `main_agent.py` 不隐式产生 spawn/join。跨事件 reducer 的 multi-agent
能力通过测试 policy 直接产生 typed facts 来验证，覆盖：

- inline、poll success、notify、launch ack、not-ready、wrong-agent 与 non-delivery；
- child-before-spawn、terminal spawn、duplicate occurrence、exact 优先于 prefix；
- parallel children 逆序返回、nested child、identical prompt 的 parent scope；
- rewrite multi-roots、intersection/union ambiguity、mixed/rejected branches；
- spawn-before-compaction、return-after-compaction；
- child leaf 不向 parent continuation 创建 physical merge edge。

固定 token oracle：

```text
Child segment:
  prompt_ids     [20, 21]
  response_ids   [22, 23]
  response_mask  [1, 1]

Main segment:
  response_ids   [12, 13, 14, 15, 16]
  response_mask  [1, 0, 0, 0, 1]
```

其中 child 自己生成的 token 为 mask 1；child 返回内容再次进入 parent prompt 时为 mask 0。Spawn 指向 parent node 1，
notify join 指向 parent node 2；它们都是 logical cross-reference，不是 physical parent edge。

## 4. 静态与代码质量门禁

H4 当时完成并通过：

| Gate | 结果 |
| --- | --- |
| Ruff lint | PASS |
| Changed-file format check | PASS |
| `compileall` | PASS |
| `git diff --check` | PASS |
| retired `AssemblyTree` / `Materializer` / tree policy symbol scan | PASS |
| RuntimeState adjacency、roots、runtime-node、message-copy scan | PASS |
| public Trajectory scaffold/profile/framework field scan | PASS |
| MR !920 ancestry：`1bbd2e8cf` 是 H4 ancestor | PASS |
| scoped code review | APPROVE，无遗留 finding |

这些门禁检查的是“旧树确实被删除且没有换名字重建”，而不只是 unit test 的最终输出。

## 5. Compaction bitwise CI

### 5.1 有效 H4 run

| 字段 | 证据 |
| --- | --- |
| Taiji task | `basic_train_petersheng_20260802122637_e4657716` |
| Instance | `8b2027419fb6e6fa019fc0ba14a302ec` |
| HyTuner | `d5633441787e78ec853421eefb394a8e62bb2a2c` |
| Nexus | `1ae27d8b09d67c8d4b46e21e273b672f804ed852` |
| Gateway | `USE_GATEWAY_AGENT_LOOP=1`；日志走 `SessionAgentLoop`，不走 legacy `NexusAgentLoop` |
| Qwen seam | `GATEWAY_LEGACY_SEAM_COMPAT=1` |
| Train | 3/3 steps，`original_acc` 与 `actor/grad_norm` 全部 exact committed golden |
| Validation | 2/2 `acc@2` 全部 exact committed golden |
| Golden | 日志出现 `bitwise test passes`，未更新或重生成 golden |
| Terminal | `SUCCEED` |

该 run 是 H4 的有效 distributed bitwise 证据。

### 5.2 缺失 seam 的早期 run

| 字段 | 证据 |
| --- | --- |
| Taiji task | `basic_train_petersheng_20260802062159_db0bea81` |
| Instance | `955360119fb6f2d8019fbf6bf72c0290` |
| HyTuner | `fea445a8c7e093eb9fb4b5e6a3cacde0aca0e272` |
| Nexus | `1ae27d8b09d67c8d4b46e21e273b672f804ed852` |
| 结果 | 三个 train step 与两个 val step 均完成，但 committed-golden compare FAILED |
| 首个差异 | `original_acc=0.5078125` expected，实际 `0.51171875`，即 1/256 sample |

早期 run 没有 `GATEWAY_LEGACY_SEAM_COMPAT=1`，所以它不能作为 H4 对 committed legacy golden 的失败证据。加入
seam 后，上述有效 run 恢复 exact golden。

## 6. Non-bitwise Ray 实验边界

当时三组 non-bitwise 实验都跑到 100/100 steps，但它们使用的是后续 Nexus/HyTuner revision，而不是本报告的 exact
H4 pair。因此：

- 它们可作为 Direct-Trajectory 路线的历史稳定性背景；
- 它们不计入 H4 revision 的 PASS 数量；
- 本报告不把其 Job ID、accuracy 或 throughput 归因给 H4。

这是有意的证据边界，避免把不同 immutable revision 的结果混在一起。

## 7. 可复现性与限制

- 本次 fresh full-suite 结果来自 exact H4 clean worktree，代码 revision 可复现。
- 96/493 differential 数量与原始 351/4 结果来自 H4 实施期间保留的 execution summary；原始临时报告随后继续被后续
  revision 更新，因此本文不伪造已经不存在的旧脚本路径或命令行。
- Taiji task/instance 与 full SHA 是 immutable evidence；认证信息未写入文档。
- 本报告证明 H4 已测到的功能和行为，不等于证明 H4 的 subclass policy extension surface 是最优 API。

## 8. 最终判定

对 H4 `1ae27d8b...` 可以给出以下判定：

1. **Direct-Trajectory runtime 与 baseline parity：PASS。**
2. **AssemblyTree/Materializer 删除且无 shadow topology：PASS。**
3. **shared-prefix、compaction、token/mask、legacy projection：PASS。**
4. **通用 multi-agent correlation/back-edge reducer：PASS。**
5. **Gateway compaction committed-golden bitwise：PASS。**
6. **H4 自身的 exact-revision non-bitwise 三实验：未执行，不做 PASS 声明。**
7. **Policy extension API：可扩展且功能正确，但 subclass override 边界偏宽，属于后续结构优化点。**
