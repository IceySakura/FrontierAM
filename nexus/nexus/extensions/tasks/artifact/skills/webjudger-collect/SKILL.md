---
name: webjudger-collect
description: 为网页产物编写 Playwright 或 bash 采集脚本时，避免那些会让整条 flow 报废、或让 judge 只看到空证据的写法。在写 flow2code 采集脚本时使用。
---

# webjudger-collect

采集脚本的职责是准确记录产物的可观察行为，不替产物补全未实现的逻辑。证据会送给 judge 判定。

以下每条都对应实测中真实发生过的失败。运行环境（harness）已提供 `page` 和采证函数，
签名如下，**不要加它们没有的参数**：

| 可用名称 | 签名 | 返回 |
|---|---|---|
| `page` | 已 goto 到目标页并等待加载完成 | Playwright async Page |
| `URL` | 当前页面地址（要重新导航时用它，不要自己拼） | str |
| `shot` | `await shot(name, label='')` | — |
| `text_of` | `await text_of(sel)` | str，无匹配返回 `None` |
| `count_of` | `await count_of(sel)` | int |
| `attr_of` | `await attr_of(sel, name)` | str，无匹配/无属性返回 `None` |
| `eval_js` | `await eval_js(expr)` | JS 返回值；**出错返回 `'eval_error: …'` 字符串** |
| `exists` | `await exists(sel)` | bool |
| `dom_state` | `await dom_state(sel='body')` | 无障碍树文本 |
| `appear` | `await appear(sel, timeout=5000)` | bool，不抛异常 |
| `record` | `record(label, value)` | — |

这些函数**没有 `label=` 参数**，也不存在 `query_text` / `query_all` —— 写了就是 `TypeError`
或 `NameError`，整条 flow 报废。

## 1. record 的两条硬约束

`record` 会静默丢弃两类调用，被丢弃的事实 judge 完全看不到：

- **所有 bool 值**，无论 label 叫什么
- label 命中判定词：`has` / `is` / `are` / `should` / `can` / `does` / `passed` / `failed` /
  `valid` / `correct` / `works` / `ok` / `success` / `matches` / `verified` 等（按 `_` 或首尾分词）

```python
# 错：值是 bool → 丢弃；label 带 failed → 丢弃
record('has_game_api', True)
record('click_failed', str(err))

# 对：记可核对的原始值，judge 自己下结论
record('window_game_type', await eval_js('typeof window.game'))
record('click_error_detail', str(err)[:200])
```

`text_of` / `count_of` / `attr_of` / `eval_js` / `dom_state` / `appear` 已自动记录
「查了什么、查到什么」，直接调用即可，不要再外面包一层 `record`。

## 2. 选择器取自源码

写代码前先 Read / Grep 定位目标控件，抄录源码里的真实写法。取到选择器即止，不通读全部代码。

```python
# 错：以契约措辞当文案
await page.click('button:has-text("Start Mission")')   # 源码里是 "PLAY"

# 对：源码中确认的结构
await page.click('#hero-cta, button.start-btn')
```

同一元素有多种写法时用逗号列候选：`'#submit, button[type=submit], .btn-submit'`。
`has-text` 是忽略大小写的子串匹配，取短核心词（`start`）而不是整句。

## 3. 选择器要落到唯一元素

匹配到多个元素时 Playwright 抛 `strict mode violation`，实测这是 flow 报废的第二大原因。
`label:has-text(...)` 这类跨容器写法一次命中十几个是常态。

```python
await page.locator('header button.primary').first.click()
```

`count_of` / `text_of` 内部已取 `.first`，不受影响；**原生 `page.click` / `fill` 要自己加**。

## 4. 内部状态先确认再取

`window.game` 一类全局是否暴露、字段名为何，源码里可直接查得。推断错了
`eval_js` 只会回 `None` 或 `'eval_error: …'`，judge 只见空值。

注意 `eval_js` 出错时返回的是**字符串**而不是 `None`：

```python
kind = await eval_js('typeof window.game')
if kind == 'object':
    await eval_js('window.game.score')
else:
    await text_of('[class*=score], [id*=score]')     # 退回界面文本
```

源码里不存在的接口走 UI 路径，不做白盒注入。

## 5. 页面取值都是字符串

`text_of` / `attr_of` / `inner_text` 返回 `str`。直接参与算术或比较会抛 `TypeError`
（`unsupported operand type(s)`、`'>' not supported between 'str' and 'int'`）。

```python
import re

def num(text):
    m = re.search(r'-?[\d.]+', (text or '').replace(',', ''))
    return float(m.group()) if m else None

before = num(await text_of('#score'))
after = num(await text_of('#score'))
if before is not None and after is not None:
    record('score_delta', after - before)
```

元素缺失时 `num()` 返回 `None`，比较前必须判空 —— `None > None` 同样抛 `TypeError`。
`eval_js` 返回的 JSON 同理：先确认是 dict 再索引，对字符串写 `data['key']` 抛
`string indices must be integers`。

## 6. Locator 是对象，且要 await 它的方法

```python
# 错
count = await page.locator('.item')()               # 'Locator' object is not callable
box = page.locator('canvas').first.bounding_box()   # 少 await → 拿到 coroutine

# 对
count = await page.locator('.item').count()
box = await page.locator('canvas').first.bounding_box()
```

Locator 没有 `is_in_viewport()` 一类方法，拿不准的属性用 `eval_js` 取。

## 7. 等条件，不等时钟

等状态变化用 `appear`（不抛异常、自动记录耗时），等不到时补一张截图 —— judge 要靠它区分
「产物没实现」和「机器慢」。

```python
if not await appear('.result', timeout=8000):
    await shot('result_timeout', '等 8s 结果区域仍未出现')
```

harness 已把动作默认超时设为 10s。单个控件明确可能不存在时可自己收紧
（`timeout=5000`），不要放大到 30s。

## 8. 触屏手势需要 hasTouch

`page.tap()` 在默认 context 下抛 `The page does not support tap`。移动端交互用 `click`
或键盘事件；只有源码显示产物仅监听 `touchstart` 时才需要模拟触屏。

## 9. 每个动作单独容错

这条决定失败时的损失范围。原生 `page.click` / `fill` / `goto` 超时会抛异常，未捕获则
脚本终止、**此前采到的截图一并丢失**。实测失败的 flow 里有三成完全没留下证据。

```python
await shot('initial', '初始状态')
clicked = False
try:
    await page.click('button:has-text("next")')
    clicked = True
except Exception as err:
    record('click_next_error', str(err)[:200])
    await shot('click_blocked', '点不到下一步控件')
if clicked:
    await shot('after', '点击后')
```

候选选择器全部未命中后记录并继续，不在首次失败处中止后续采集。

## 10. 不要覆盖 harness 的名字

生成的代码与 harness 共享作用域。给 `status`、`error`、`page`、`URL`、`observations`
赋值会破坏证据文件。局部变量另起名字。

## 11. 语法自检

脚本整体 exec，一处语法错误使整条 flow 报废、不留截图。

- 写 Python 不是 JS：注释 `#`、方法名 `snake_case`、筛选用
  `page.locator('.row').filter(has_text='x')`
- f-string 内不能有反斜杠：`f'{x.split("\n")}'` 报错，先赋值给变量
- 不写递归遍历 DOM（实测触发 `RecursionError`），用 `eval_js` 一次取列表
