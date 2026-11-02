# user simulator 设计

user simulator 是 HITL 的核心模块，也是后续扩展的主要对象。在本文档中，我们将全面的介绍 user simulator的设计，涵盖用户建模、设计理念、实现细节以及扩展方式等诸多方向。

---

## 1. 总体设计

HITL 的 user simulation harness 为一个完整的 agent loop。user 模型通过特殊的工具与沙盒内的worker（如claude code）进行交互，并可以正常使用 bash 等工具实现完整的自我验证。在此基础上，为了让 simulator 能够真正模拟现实中多样化的用户，HITL 设计了结构化的 persona 用于定义 user simulator在合作中的行为，并总结了约束模型符合用户模拟角色的原则和方法。除此之外，HITL 也支持了 Skill、MCP Server 以及其他自定义工具，从而能够适应多样化 topic 的任务。

---

## 2. 用户建模与设计理念

user simulator 是对真实人类在进行特定任务下的特定行为特征的拟合，只能在一定程度上反应人类的特征。基于这个基本认知，我们总结了其设计理念：

**约束 LLM 的行为方式** LLM 有其自己的行为特点，且和真实人类存在系统性偏差，为了让模型能够抑制自己默认的行为趋势，并按照我们的期望模拟一个真实人类，我们需要**显式**的规定模型的行为特征，将模拟转化为一个指令遵循的过程。诚然，我们的指令不可能覆盖一切人类行为，但应该在统计意义上覆盖绝大部分人类常见的行为模式和特征。

**提供足够的上下文** LLM 在任务中能否做出合理的决策取决于其上下问是否完整。我们认为，一个合格的user simulator 应该拥有按照自己意愿读取上下文的方法，包括完整的 bash 执行命令、文件阅读、领域skill与mcp工具等权限。对于直接编辑文件的能力，我们认为需要谨慎考虑，因为 LLM 容易在编辑中偏离自己的角色定位，从而替代worker做过多的活。

**限制行为深度** user simulator 下，模型的工作会远比 one shot 任务长。因此，我们依旧需要限制 user simulator 的工作强度，避免其将大量时间花在自己的探索下。



---

## 3. User Harness 基础设计

### 3.1 工具集合

HITL 为 user 模型内置了以下工具：

| 工具　　　　　　　　 | 作用　　　　　　　　　　　　　　　　　　　　　 |
| ----------------------| ------------------------------------------------|
| `send_message`　　　 | 向 worker 发送消息　　　　　　　　　　　　　　 |
| `accept`　　　　　　 | 接受当前任务完成状态，对于单轮任务代表人物完成 |
| `read`　　　　　　　 | 阅读一个路径：文件、目录、图片　　　　　　　　 |
| `bash`　　　　　　　 | 执行 shell 命令　　　　　　　　　　　　　　　　|
| `str_replace_editor` | `read` 的能力和写入能力　　　　　　　　　　　　|
| `invoke_skill`　　　 | 调用 skill　　　　　　　　　　　　　　　　　　 |

其中，`str_replace_editor`不被开启，`invoke_skill` 应当只在在加载了 skill 时开启。

**`send_message`** 

send_message 包含以下字段：

- `message` 发送给 worker 的指令
- `draft` user 自己的推理，worker 不可见，用于存放它已经知道的命名、倾向的改法、尚未提出的边界情况。`draft` 为空的调用被拒绝。
- `kind` 本轮的指令的目标分类，目前有6个取值：`goal` 开场提出需求、`requirement` 追加一条需求或约束、`observation` 指出观察到的现象、`validation` 要求 worker 做检查、`pushback` 打回交回来的工作、`answer` 回答 worker 的提问。
- `to` 目标指向的 worker，目前均为单 worker 的情况，
- `rewrite` 这条消息的措辞是否允许在发出前被调整，目前暂时没有被使用，所有信息不回经过任何改写。

send_message 的调用结果为 worker 压缩后的中间轨迹与完整的最后一轮恢复内容，它的具体形态由对应的 worker manager 决定，并会渲染为结构化 xml 标签。

**`accept`** 

- `reason` 接受当前需求的原因，取值包括：`satisfied` 工作达到要求和`gave_up` 因为某种原因中途停止。
- `summary` 对收尾时任务状态的总结。
- `checked` 模型显式对给定任务所有需求点的总结，用于提醒模型是否真的完成了所有需求。

当 worker 还有 pending 提问时，不允许调用accept；当accept调用但没有后续的需求（例如，单需求任务第一次调用）时，系统会注入提示要求模型再次检查任务是否完全完成，并只在连续两次accept时才允许任务真正结束，从而防止模型不把任务完全做完。



**`read` 与 `bash`** 

提供沙箱内独立观察与验证的能力，与常规的 SWE Agent 完全一致；若希望 user 能够直接编辑文件，从而模拟 co-edit 的场景，可以将 read 换为 str_replace_editor。由于编辑权限容易导致 user 偏离人设且不好控制，默认不开启这个功能。

**其他**

所有的 HITL 工具本身描述尽可能保持了精简，其调用时机和约束由额外定义其在 system prompt 中的解释实现

### 3.2 需求队列

HITL 希望原生适配单轮与多轮的任务需求。为此，我们定义了需求队列：在任务开始时，向 user 模型注入第一条需求，随后每次 user 调用 accept 时，自动向 user 上下文注入后续需求，从而实现需求的渐进释放。对于最后一次需求，accept 会要求双重检查以防止 user 在任务没有完全完成时就过早宣布任务结束。

### 3.3 prompt 设计

system prompt 由多个段落构成：

**identity** 定义 user 模拟用户模拟的角色与通用行为约束。声明四件 user 模型的角色定位以及与 worker 模型的交互基本原则。并明确禁止透露任何用户模拟相关信息。

**harness** 声明 user simulationn harness 的基本信息，包括工具调用方式、系统提示的插入点、worker 工作结果的渲染方式。

**usage doc** 介绍所有 user 可用工具的使用时机和说明，区别于 tool schema，并会随着工具集合变化自动调整。

**persona** 定义 user 的表达、行为与动作特点，可以像 skill 一样灵活定义，用于描述多样化的 user simulator。

**工作偏好** 用于同步传入给 harness 的 CLAUDE.md/AGENTS.md 内容，由 xml标签包裹，避免直接成为 user 的指令。

**worker slash command** 声明 worker 支持的 slash command （cli内置的以及skill调用等），用于 user 主动调用 worker 特定能力。

**运行环境** 说明当前任务工作目录、worker 模型的实际 harness 以及称呼。

### 3.4 运行时提醒机制

User Simulation 的 harness 参考 claude code，使用 system reminder以压制 user 模型在长任务下的行为偏移。在 worker 返回、新需求释放、工具调用失败等时候会由 harness 自动插入。

HITL 允许通过 Persona 定义 reminder，这些 reminder 在 worker 返回时自动插入，分为完整与简洁两个版本。完整版 reminder 会列举 persona 的 recall、user skill 清单以及 worker 可用 slash command。两次提醒之间间隔轮次如果太短，会用简略版代替以节省上下文。


### 3.5 上下文压缩

user simulator 提供基于 nexus 的上下文压缩机制，默认在prompt token 超过窗口80%时触发。压缩采用 self- summarize的方式，压缩完成后保留完整 system prompt、完整的总结文本以及所有的过去需求说明。

### 3.6 工作偏好文档

HITL 支持上传 CLAUDE.md 等偏好文档，由 harness 注入给 worker。当配置了该文档时，user 也会同步收到相关的信息。

user 这一侧由 persona 的 `reads_memory` 控制，worker 那一侧不受它影响。设为 false 时这份文档不进 user 的 system prompt，用于没有前置合作关系的 user。

### 3.7 Skill、MCP 与 Slash Command

为了保证框架的通用性，HITL 支持了 Skill、MCP 与 Slash Command。

**Skill** HITL 支持按照目录为 user 和 worker 分别配置 skill，从而为模型提供相应的领域知识。

**MCP server** HITL 支持按照目录为 user 和 worker 分别配置 MCP，用于支持各类通用任务。

**worker 的 slash command。** HITL 允许 user 模型直接在文本中调用 worker harness 支持的 slash command，例如为 claude code 调用 skill 以及内置命令。User harness 会提前校验命令的合法性。

### 3.8 协作体验打分

HITL 正在测试引入 user 自打分机制：在 user 最后一次 accept后，向 user 模型注入用户体验相关的rubrics并要求他输出体验打分，从而评估 worker 模型在本次交互中的用户体感。rubrics 涵盖模型沟通是否通畅、是否有自作主张、是否有反复失败、谄媚以及不良的指令遵循

HITL 在 run 结束之后让 user 对这次协作打分。打分请求接在最后一次 accept 之后，沿用 user 自己的对话历史，由它带着全程的记忆评价。user 自己结束的 run 才打分，跑到轮次上限与丢了 worker session 的都不打。

维度共九个。前七个记录 worker 的失分项：沟通断档、声明拿不出证据、自行做的决定不说、该问的不问该做的不做、不照指令做、失败后不换路子、一味附和。另外两个记录这次协作里值得点出的亮点与失望。这些维度都取自 user 视角能看见的行为，worker 的思考过程与完整 tool 序列不到 user 手上，与之相关的维度不纳入。

每个维度是 1 到 10 的整数，两端的含义在请求里各自写明。回复是一个 JSON 对象。请求失败与解析失败都重试，共三次，三次都不成则返回未打分并带上最后一次回复。这条路径不抛异常。结果放在 `agent_output.experience`，`rated` 决定 `scores` 是否有意义。

---

## 4. Persona 设计

HITL 希望避免 user simulation 中的模版化、套路化等问题，为此，我们设计了 Persona，它可以像 Skill 一样由 markdown 文件定义，并动态的定义 user 模型在任务中的工作、交流方式以及工具集合。它包括 formatter 与 正文两部分：

| frontmatter    | 默认值　　　　　　　　　　　　　　　　　 | 内容　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| ----------------| ------------------------------------------| --------------------------------------------------------------------|
| `name`         | 文件名　　　　　　　　　　　　　　　　　 | persona 的标识符　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `tools`        | `send_message`、`accept`、`bash`、`read` | 该 persona 下 user 拿到的工具　　　　　　　　　　　　　　　　　　　|
| `needs_tools`  | 空　　　　　　　　　　　　　　　　　　　 | 正文要求 user 使用的工具，装配后检查它们都在工具集里，缺一件即报错 |
| `recalls`      | 空　　　　　　　　　　　　　　　　　　　 | 运行期提醒的内容，见 §4.5　　　　　　　　　　　　　　　　　　　　　|
| `reads_memory` | `true`　　　　　　　　　　　　　　　　　 | 是否告知它写过 worker 的工作偏好文档　　　　　　　　　　　　　　　 |

五项都可以不写，正文是唯一必填的部分。一个只有正文的 markdown 文件即一个可用的 persona，名字取自文件名，其余各项取默认值。

正文分四节，分别定义 user 怎么说话、看重什么、每一步做什么、以及随轮次如何变化。HITL 内置的 `default` 以开发者为蓝本，以下 §4.1 到 §4.4 逐节介绍。

### 4.1 交流方式与语气

为定义和约束 user simulator 说话方式，HITL 选择了通过观察用户日益与 总结早期实验轨迹，尽可能穷举模型不说人话的 pattern 并详细给出的举例。
这一节由规则和正反例规则，涵盖：

**写什么** 定义模型如何组织发送给 worker 消息的内容，例如内容的角度与不需要说的细节。包
**什么不能进消息** 禁止模型常见说话的 bad pattern，例如防御性话语、追加提醒等。
**怎么说** 涵盖语句大小、用词、段落结构、标点、粘贴等语句组织方式等，用于尽可能让模型等表达对齐真实用户。

### 4.2 工作偏好

工作偏好主要用于规定模型对特定任务的产物偏好，从而提高整体的多样性，他与 topic 强相关。我们对代码任务设计的偏好主要针对 Claude 模型编写代码的 Bad patten 与 ai味。
工作偏好推荐同时在 CLAUDE.md 等文件内定义，从而让 worker 的行为更符合规定的用户偏好。

- **注释与文档。** 注释和 docstring 简短地说明代码做什么，避免逻辑推理、长篇大论和防御性解释。
- **逻辑的写法。** 避免过度防御性编程与隐藏错误的代码写法，例如大量默认兜底行为。
- **代码的组织。** 要求工作按仓库既有的方式划分模块，新项目避免过度扁平与过度设计。
- **测试的写法。** 要求精简的测试，避免无意义与低价值的测试。

### 4.3 与 worker 交互方式

为了保证 user simulator 的行为合理，HITL 定义了其行为表。user 的所有与worker 的交互都应该有一个明确的分类，包括：

- **提出新需求。** 说明如何渐进的释放给定的需求给 worker
- **回答 worker 的提问。** 要求 user 模型根据自己的偏好做出 worker 选择，避免一直选择默认选项。
- **指出问题。** 说明 user 模型如何在发现问题时向 worker 呈现，包括需要说什么、避免说什么和模拟用户复制粘贴。
- **要求验证与澄清。** 说明 user 模型如何要求 worker 做额外自我验证。
- **打回。** 要求 user 模型不轻信 worker 的说法，从需求、偏好、边界条件等方面判断是否向 worker push back。也约束 user 在push back时提供的信息量。
- **验收当前工作。** 确认交给 user 的每一条需求都完整且正确地满足之后才验收。

### 4.4 跨轮次约束

为了模拟真实 user 在工作进行中的情绪等变化，我们额外设计了跨轮次约束，用于在 worker 反复失败、做有风险行为时改变 user 的措辞方式。

### 4.5 运行中注入信息

Persona 的 recalls 章节为运行时动态注入信息，适合根据按照 persona 正文的结构设计，并引用 persona 对应的模块。

### 4.6 其他

暂无

---

## 5. 后续扩展计划

### topic 适配

通用层的措辞不含任何工作类型，接入一个新 topic 需要的是它的需求文本与工具集。各领域自己的行为特征落成 persona 与 skill，是接入时的主要工作。

### 多样化用户建模

从用户日志分析真实用户如何提出需求、何时不满、打回时如何表述，并给予这些模式设计 Persona；针对不同 topic 的 task，也有特定的用户设计方法。

### 表达优化

尽管目前通过精细的 prompt 调控，已经能够较好的控制模型表述的长度，但其内容和真实 user 仍然有较大的差距。如何进一步优化口语化表达依旧是后续的重点方向。同时，我们也希望在此之后通过 hint 的形式同步优化 worker 模型的表述方式，通过降低其说 ai 黑话的频率来提高用户体验。


