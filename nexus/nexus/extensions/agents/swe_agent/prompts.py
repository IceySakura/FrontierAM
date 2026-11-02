"""Prompt sets for the SWE-agent loop.

`swe_agent` and `ch_swe_agent` run the same loop over the same tools; the only
thing that separates them is the language of what the loop says to the model.
Keeping both languages here is what lets the two agents share one implementation
instead of drifting as separate forks.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSet:
    """Everything the agent loop says to the model, in one language."""

    system: str
    next_step: str
    """Wraps a tool observation. Takes ``{observation}``."""
    next_step_no_output: str
    """Stands in for an observation when the command printed nothing."""
    instance_templates: Mapping[str, str]
    """Language of the task under repair -> instance prompt. Must hold a ``generic`` entry."""

    def instance_template(self, language: str) -> str:
        """Pick the instance prompt for a task language, falling back to the generic one."""
        return self.instance_templates.get(language, self.instance_templates['generic'])


# ============== English ==============

_EN_SYSTEM = """You are a helpful assistant that can interact with a computer to solve tasks."""

_EN_NEXT_STEP = """OBSERVATION:
{observation}"""

_EN_NEXT_STEP_NO_OUTPUT = """Your last command ran successfully and did not produce any output."""

_EN_PYTHON_INSTANCE = """<uploaded_files>
{working_dir}
</uploaded_files>
I've uploaded a python code repository in the directory {working_dir}. Consider the following PR description:

<pr_description>
{problem_statement}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {working_dir} directory to ensure the <pr_description> is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long.
"""

_EN_GENERIC_INSTANCE = """<uploaded_files>
{working_dir}
</uploaded_files>
I've uploaded a code repository in the directory {working_dir}. Consider the following PR description:

<pr_description>
{problem_statement}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {working_dir} directory to ensure the <pr_description> is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
2. Create a script to reproduce the error and execute it using the bash tool, to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long.
"""

ENGLISH = PromptSet(
    system=_EN_SYSTEM,
    next_step=_EN_NEXT_STEP,
    next_step_no_output=_EN_NEXT_STEP_NO_OUTPUT,
    instance_templates={'python': _EN_PYTHON_INSTANCE, 'generic': _EN_GENERIC_INSTANCE},
)


# ============== Chinese ==============

_ZH_SYSTEM = """你是一个能够通过工具与计算机交互并解决软件工程任务的助手。请优先使用中文进行分析和回答；代码、命令、文件路径、错误信息和 API 名称保持原文。"""

_ZH_NEXT_STEP = """观察结果：
{observation}"""

_ZH_NEXT_STEP_NO_OUTPUT = """上一个命令已成功执行，且没有产生输出。"""

_ZH_PYTHON_INSTANCE = """<uploaded_files>
{working_dir}
</uploaded_files>
我已经把一个 Python 代码仓库放在目录 {working_dir}。请阅读下面的 PR 描述：

<pr_description>
{problem_statement}
</pr_description>

请你在仓库中实现必要修改，使代码满足 <pr_description> 中的要求。
我已经处理好了 <pr_description> 中描述的所有测试文件变更。这意味着你不需要以任何方式修改测试逻辑或测试文件！
你的任务是在 {working_dir} 目录中对非测试文件做最小必要修改，确保 <pr_description> 被满足。
请按以下步骤解决问题：
1. 首先查找并阅读与 <pr_description> 相关的代码。
2. 编写一个复现脚本，并使用 bash 工具执行 `python <filename.py>` 来确认错误。
3. 修改仓库源码来解决问题。
4. 重新运行复现脚本，确认错误已经修复。
5. 思考边界情况，确保修复能覆盖这些情况。
你的思考可以很充分，较长也没有问题。
"""

_ZH_GENERIC_INSTANCE = """<uploaded_files>
{working_dir}
</uploaded_files>
我已经把一个代码仓库放在目录 {working_dir}。请阅读下面的 PR 描述：

<pr_description>
{problem_statement}
</pr_description>

请你在仓库中实现必要修改，使代码满足 <pr_description> 中的要求。
我已经处理好了 <pr_description> 中描述的所有测试文件变更。这意味着你不需要以任何方式修改测试逻辑或测试文件！
你的任务是在 {working_dir} 目录中对非测试文件做最小必要修改，确保 <pr_description> 被满足。
请按以下步骤解决问题：
1. 首先查找并阅读与 <pr_description> 相关的代码。
2. 编写一个复现脚本，并使用 bash 工具执行它来确认错误。
3. 修改仓库源码来解决问题。
4. 重新运行复现脚本，确认错误已经修复。
5. 思考边界情况，确保修复能覆盖这些情况。
你的思考可以很充分，较长也没有问题。
"""

CHINESE = PromptSet(
    system=_ZH_SYSTEM,
    next_step=_ZH_NEXT_STEP,
    next_step_no_output=_ZH_NEXT_STEP_NO_OUTPUT,
    instance_templates={'python': _ZH_PYTHON_INSTANCE, 'generic': _ZH_GENERIC_INSTANCE},
)


__all__ = ['CHINESE', 'ENGLISH', 'PromptSet']
