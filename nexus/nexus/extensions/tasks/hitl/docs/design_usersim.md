# User simulator

HITL V2 的 User simulator 继承了 v1 的设计理念，即 user 本身也是一个完整的 agent loop，通过工具调用实现与 worker 的交互、自我验证以及需求的推进。在 v2 的设计中，我们额外以 persona 控制模型的行为特征，以 skill 扩展模型的主动能力。

---

## 1. user 主循环


我们认为 user simulation 需要维持 user 本身的主体性，由 user 来控制整个工作的流程，提出新的需求，审核 worker 工作， 要求返工，或是回答 user 问题。为此，我们将 user 设计为整个 HITL 的主循环，并为它配置了以下默认工具：


- `send_message` 触达 worker 的唯一方式
- `accept` 验收当前需求并推进队列
- `read` 阅读一个路径：文件、目录、图片
- `bash` 在常驻 shell 中执行命令

加载了 skill 的 run 还会拿到 `invoke_skill`，按名取回一份 skill 的全文。

### send_message

user 通过 send_message 向 worker 发送任务需求。工具包含三类参数：

- **`message`** —— worker 读到的内容。
- **`draft`** —— 继承 v1 的 private draft，user 自己的推理，包括它已经知道的命名、倾向的改法、尚未提出的边界情况，避免 user 在公开的 message 里把实现方案说得太细。
- **`kind`** —— 本轮的目标，用于区分用户的行为特点：
  - `goal` —— 开场提出需求，一次 run 的第一条消息。
  - `requirement` —— 追加一条约束，队列释放下一条需求以及补充新需求时使用。
  - `observation` —— 指出看到的观察、失败或任何用户的观测。
  - `validation` —— 要求 worker 做检查。
  - `pushback` —— 认为交回来的工作不对，要求退回重做。
  - `answer` —— 回答 worker 的提问，worker 阻塞在提问上时用。

### accept

accept 工具表达「当前状态满足 user 需求」的语义，用于推进任务进度。对于单轮的任务，accept 即代表任务完成，对于多轮需求，即代表环境可向 user 释放下一条需求。

### bash 和 read

用于为 user 提供 沙盒内独立观察与验证的能力，也可将 read 替换为 str_replace_editor从而支持 user 的编辑能力。


### 其他工具

skill 由 `invoke_skill` 取用，见 §6。

HITL 本身也计划支持为 user 和 worker 挂载 MCP server，可通过 `HITL_USER_MCP_SERVERS` 声明。

---

## 2. 需求队列

HITL v2 希望原生适配好单轮和多轮的需求，为此，我们定义了需求队列。单轮的 task 给一条 `user_prompt`，多轮的 task 给一串 `requirements`，两者进同一个队列，循环的形态不因此分叉。

队列由 user 自己拉取：它调用 `accept` 之后才会拿到下一条，而队列有多长、后面还有什么，环境都不告诉它。

---

## 3. system prompt

我们的 system prompt 主要包含以下设计组件：

### identity

定义 user 的用户模拟规范和通用行为约束。

* **整体角色定义** 一个 demanding 的 user，正在使用 agent 工具完成任务，结果不满足它的要求与工作偏好就不收下。
* **与 agent 的交互方式** 需要什么、什么算做完由 user 自己定，并按这个标准要求 worker；说要什么而不说怎么做，worker 拿来选择时它给出选择；结果不符合时打回并指名哪里不对；判断之前重读自己当初提的要求；不转发 worker 给的原文，用自己的话说自己的判断。
* **行为边界** 活由 worker 干，这一条的措辞取决于这次 run 有没有给 user 能写的工具：没有时要求它跑现成的东西、缺什么向 worker 要，有时允许它为了检查而写并要求在下一条消息里说明改了什么。同时不透露这是一次模拟，不提评分与验收标准。

### harness

定义用户模拟环境的运作方式，包括工具调用规范、`<system-reminder>` 含义、worker 返回值渲染原则等。

### usage_doc

定义一类工具怎么用，按用途分组 —— 与 worker 交谈、验收、自己动手查看各是一段，由声明这个用途的工具携带。一次 run 没有装某件工具时，关于它的说明也不出现在 prompt 里。

### persona

定义simulator的行为特点与动作集合的 persona，由文档进行声明

### 工作偏好

用于存放传递给 worker 的 CLAUDE.md/AGENTS.md 内容，即用户应该对齐偏好文档对 worker 模型的行为约束

### worker slash command

声明 worker 支持 slash command，例如 /{skillname}等，可以由 user 主动调用

### 运行环境

说明工作目录、worker harness 名称等环境相关信息

---

## 4. persona：系统单方面加的预设

persona 决定被模拟的 user 是什么样的人，例如需求提出和补充的方式、说话风格和验收时如何进行检查。Persnoa 由系统周期性提醒来维持。

### persona 定义

persona 可以像 skill 一样由一个 markdown 文档进行定义，正文是进 system prompt 的那段 prompt：

```markdown
---
name: default
tools: [send_message, accept, bash, read]
needs_tools: [bash, read]
reads_memory: true
recalls:
  - 'Asking: one or two points a message, where a point is one thing the worker can act on.'
on_handover:
  - 'Test the claim: use the thing that was failing, exactly as you first met it.'
---

You are a working engineer who needs a change to a codebase you rely on but do not
maintain, and you know what you need it to do.
```

- `name` 与文件名一致，`HITL_USER_PERSONA` 的标识符。
- `tools` 这个 persona 的工具集，默认为原先四件。
- `needs_tools` 正文依赖的工具。
- `reads_memory` 是否告知它写过 worker 的 CLAUDE.md 等偏好为你挡
- `recalls` 周期提醒的内容。
- `on_handover` worker 交付时插入到 user simulator的输入中

### 正文

正文进入 system prompt 的 `## Who you are`，定义模型的行为模式，包括角色遇到某种情形时怎么回应、主动做什么、明确不做什么。

默认 persona 正文首先这个人的身份与全局的说话风格：一次一到两个点、聊天的口吻、每个点不重复论述、不反复论证解释自己的观点。随后的内容分场景定义用户的行为参考，每个部分按照“一个人在这个场景里是什么样的、它会做什么、它明确不做的”进行组织。

* **Asking** 提需求。开场先说最影响使用的那一件，其余的留到它成为要紧的事时再说。
* **Interfaces** 回答 worker 拿来的选择：命名与形状、它要去项目外面、它说需求本身不对。
* **Reviewing** 阅读交回来的改动而不只是执行一遍，看它在哪里自行做了决定、实际跑的是什么、有没有做多余的事，发现了提出来而不自己改。
* **Accepting** 按自己真实的场景主动使用和验收，避免轻信 worker。


### recalls 与 on_handover

HITL 使用周期性的 reminder 维持 user 对于模拟指令和人设的遵循，主要分为 recall 与 handover：

* `recalls` 是正文的简略压缩版本，用于提醒模型遵循正文的约定，原则上应该和 persona 的主体内容对应。它分两级给出：完整的即为全部 `recalls` 与 skill 清单，在 user 向 worker 发完消息之后给出；为了防止过于频繁，为长 recall 设计了冷却，处于冷却时期，系统会用一条简略的消息代替完整文本。
* `on_handover` 在 worker 完成时作为系统提示插入，用于指导模型如何应对 worker 的反馈，例如核验、追问的行为方式。同一条提醒里列出全部选项，由 user 挑一条去做并说明挑了哪条。

---

## 5. 工作偏好文档

HITL 支持上传 CLAUDE.md 等偏好文档，由 harness 自动注入给 worker。为了对齐工作认知，我们将其内容包裹在 xml 标签哪呈现给 user 作为参考。
---

## 6. skill：模型主动取用的方法

skill 是可由 user 模型主动调用的工具，用于在特定任务上提供领域相关的额外能力。注意需要区分 user 可调用的skill与 worker 可调用的skill，两者在 HITL 哪需要分开定义。

---

## 7. 运行期注入的消息

运行期会向 user 的历史追加以下内容，用于提高模型的长程指令遵循能力。worker 回报是 `send_message` 的返回值，其余各条由环境插入，包在 `<system-reminder>` 里，让 user 区分它与对话本身。

| 消息 | 触发条件 |
| --- | --- |
| worker 回报 | 每次 `send_message`，与实际发出的原文一起返回 |
| 交付提醒 | worker 完成一轮且没有反问，带上 persona 的 `on_handover` |
| 周期提醒 | 见 §4 的 `recalls` |
| 无工具调用的督促 | 一整轮没有 tool call |
| 只查不说的督促 | 连续自己查看若干次而没有与 worker 说话，次数再高一档时换一条更强硬的措辞 |
| 下一条需求 | `accept` 之后队列释放的需求文本 |

每条需求的文本都附带一段任务边界，说明代码由 worker 写、worker 只知道工作目录与 user 说过的话、验收以任务全部完成为准。首条需求与后续需求各有一套开场要求：首条要求 user 在查看任何东西之前先用 `kind="goal"` 把需求发出去，后续的用 `kind="requirement"`，只提这一条需要的东西。

---

## 8. 历史压缩

HITL 提供了 user 的上下文压缩机制，使用 nexus 官方的 自我总结进行compact。当 history 的 token 数超过上下文窗口的一定比例时，压缩将出触发并根据以下方式提醒模型进行总结：

- **要求过什么** —— 提过的每一条，用当时的说法。
- **被告知了什么** —— worker 报告做了什么，以及它对结果的声明。
- **自己检查到了什么** —— 读过什么、跑过什么、实际看到什么。
- **有什么不满意** —— 退回过什么以及为什么，还有哪些说法没有验证过。
- **在等什么** —— 问了还没得到回答的，以及正在进行的工作。

压缩后 user 保留完整system prompt、所有的任务需求文本以及自我总结的内容。

