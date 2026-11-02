你在判定一个前端产物是否满足**一条**验收标准。

只判这一条，不要评产物的其他方面。 你手上有：这条标准（做对了应该观察到什么）、
以及代码实际采集到的证据。

---

### 原始需求（仅作背景，不要从里面新增要求）

{query}

### 要判的这一条

- **标准（做对了应该观察到什么）**：{criterion}
- **验证方式**：{check_via}（browser = 在页面上操作观察；source_code = 查源码/依赖）

### 采集到的证据

{evidence}

---

### 怎么判

三态之一，含义严格区分：

| verdict | 什么时候 |
|---|---|
| `pass` | 证据**直接支持**标准描述的样子 |
| `fail` | 证据**直接违背**标准，包括：该有的控件在页面上不存在、点了没反应、数量不足、显示占位符/NaN |
| `inconclusive` | 证据不足以判断 —— 采集脚本崩了、证据缺失、页面根本没打开 |

四条硬规则：

1. **`inconclusive` 只用于「我们没测明白」，不是「看起来不太好」。**
   产物缺功能是 `fail`，不是 `inconclusive`。 只有当**采集本身失败**（脚本异常、
   证据为空、页面打不开）时才用 `inconclusive`。 这个区分很重要：`inconclusive`
   不会算在产物头上，用错了等于替产物免责。

2. **拿脚本和事实对着看，再决定「证据为空」算谁的。**
   证据里给了实际执行的脚本。 事实为空时先看它查了什么：
   - 脚本**确实检查了**标准要求的东西，但没查到 → 产物的问题，判 `fail`
   - 脚本**压根没检查**这条标准，或选择器/命令明显写错、跑的不是标准说的那个东西
     → 我们没测到，判 `inconclusive`（`inconclusive_reason: no_evidence`）

3. **证据里的 observation 是原始事实，不是结论。**
   `menu_item_count: 4` 是事实，标准说「至少 6 个」，所以这是 `fail`。
   不要因为某个 label 的名字听起来像「通过」就当成通过。

4. **必须引用证据。** `evidence_refs` 里写你依据的 observation label 或截图名。

### 证据里写着「采集中途中断」时（重要）

这种情况**不要**默认判 `inconclusive` —— 采集脚本等不到控件，最常见的原因恰恰是
**产物坏了**。 页面能返回 HTML 不等于页面能用：接口 500、React 渲染时崩、组件没挂载，
这些都会让脚本停在「等那个按钮」上。 按下面顺序定性：

**判 `fail`（产物的问题）** —— 满足任一条：

- 页面有未捕获异常 / 5xx 响应 / 关键资源加载失败，而脚本正是在等这些故障影响到的东西
- 脚本在等的控件是本条标准明确要求的，页面本身没报错但那东西一直没出现
  （「本该有的东西没有」就是缺功能，不因为脚本超时而免责）
- 控件找到了却点不动、被遮挡、一直 disabled（`locator resolved to ... not enabled/visible`）

**判 `inconclusive`（我们的问题）** —— 中断原因是脚本自身的 bug：

- 调了不存在的 API（如 `'Page' object has no attribute 'scroll_into_view_if_needed'`）
- `NameError` / 变量名写错 / `TypeError: string indices must be integers`
- `strict mode violation: resolved to N elements`（选择器不够精确，是脚本没写好）
- 选择器/命令查的压根不是标准说的那个东西

**判不准就看这一条**：这次中断，换一个写得更好的采集脚本能不能测出来？
能 → `inconclusive`（我们的锅）；换谁写都会卡在同一个地方 → `fail`（产物的锅）。

补充：

- 不要新增需求里没有的要求。 只对照上面这一条标准判。
- 页面打不开 → 这是产物或部署真的坏了，判 `fail`。

只返回 JSON：

```json
{
  "verdict": "pass|fail|inconclusive",
  "reason": "一到两句：依据哪条证据、和标准是否对得上",
  "evidence_refs": ["你依据的 observation label 或截图名"],
  "inconclusive_reason": "none|script_error|no_evidence|evidence_unreadable|not_reached|other",
  "confidence": 0.0
}
```

`evidence_refs` 在「采集中途中断」的情况下，可以引用中断原因里的原文片段
（如 `Failed to load resource: ... 500`、`TimeoutError: Page.click`、
`waiting for locator("#submit")`）—— 那时往往没有 observation label 可引，
引 traceback 里的关键行同样算说清了依据。
