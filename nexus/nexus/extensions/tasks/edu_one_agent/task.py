import base64
import logging

from typing import Any

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, require_context, task_registry
from nexus.core.agent import WithMainTrajectory


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """
You are a senior subject teacher, and you provide an analysis of this question according to the following requirements.
You offer clear, logically rigorous explanations, providing correct analysis and answers.

Note: You must first think through the complete reasoning process in SIMPLIFIED CHINESE, and then give the final answer.
The final answer MUST be within "总结" †<answer goes here; replace the placeholder with the real answer>†.

Example1：
1+1+1=？2+2+2 = ？
### 解析：
……
### 总结：
†**$3$；$6$**†。

Example2：
A.B为同阶可逆矩阵，C是一个矩阵，其中c11=A. c12=0.c21=0.c22=B求C的伴随矩阵
### 解析：
……
### 总结：
†
\\[ \\text{adj}(C) = \\begin{pmatrix}
\\text{adj}(A) & 0 \\\\\n0 & \\text{adj}(B)\n\\end{pmatrix} \\]
†""".strip()


class EduOneAgentTaskInput(TaskRunInput, WithAgentRunner):
    """Input for EduOneAgentTask."""

    question: str
    answer: str
    image: str | None = None
    image_url: str | None = None
    image_width: int | None = None
    image_height: int | None = None

    tools: list[str] | None = None
    max_iterations: int = 5
    wrx_prompt_id: str | None = None

    def build_messages(self) -> tuple[str, list[dict]]:
        if self.wrx_prompt_id is None:
            if self.image is not None:
                with open(self.image, 'rb') as fh:
                    base64_data = base64.b64encode(fh.read()).decode('utf-8')
                return DEFAULT_SYSTEM_PROMPT, [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_data}'}},
                    {'type': 'text', 'text': self.question},
                ]
            return DEFAULT_SYSTEM_PROMPT, [
                {'type': 'text', 'text': self.question},
            ]

        try:
            from wrx_prompt import prompt
        except ImportError:
            raise ImportError('Not found wrx_prompt; please pip install wrx & wrx-prompt first.') from None

        if self.image is not None:
            with open(self.image, 'rb') as fh:
                image = fh.read()
            messages = prompt(self.wrx_prompt_id).render(question=self.question, image=image)
        else:
            messages = prompt(self.wrx_prompt_id).render(question=self.question)

        def make_dict(content: str | list) -> list[dict]:
            if isinstance(content, str):
                return [{'type': 'text', 'text': content}]
            return content

        if len(messages) == 1:
            if messages[0]['role'] != 'user':
                raise ValueError('no user message?')
            return DEFAULT_SYSTEM_PROMPT, make_dict(messages[0]['content'])

        if len(messages) == 2:
            if messages[0]['role'] != 'system':
                raise ValueError('no system message?')
            if messages[1]['role'] != 'user':
                raise ValueError('no user messages?')
            return messages[0]['content'], make_dict(messages[1]['content'])

        raise ValueError('task:edu_one_agent only support one or two messages via wrx-prompt.')


class EduOneAgentTaskOutput(TaskRunOutput):
    """Output containing both prediction and ground-truth for debugging."""

    prediction: str
    question: str
    answer: str
    trajectory: list[dict[str, Any]] | None = None


@task_registry.register('edu_one_agent')
async def edu_one_agent_task(task_input: EduOneAgentTaskInput) -> EduOneAgentTaskOutput:
    """Drive a single one-agent rollout."""
    question = task_input.question
    logger.debug('[edu_one_agent_task] started, question=%s', question)

    # --- Built Prompt First ------------
    system_prompt, user_prompts = task_input.build_messages()

    # --- Seed ctx.tool_state for tools + hook -------
    ctx = require_context()
    ctx.tool_state['image_refs'] = {
        'user_image 1': {
            'url': task_input.image_url,
            'resized_url': task_input.image_url,
            'width': task_input.image_width,
            'height': task_input.image_height,
        }
    }

    # --- Run the Agent -------------------
    agent_output = await task_input.run_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompts,
        max_iterations=task_input.max_iterations,
        tools=task_input.tools,
    )

    if not isinstance(agent_output, WithMainTrajectory):
        raise RuntimeError(f'Agent output invalid type: {type(agent_output)}')

    trajectory = agent_output.main_trajectory

    # exit_status = getattr(agent_output, 'exit_status', None)
    # exit_status_value = exit_status.value if hasattr(exit_status, 'value') else (exit_status or '')
    # TODO: exit_status_value in _ALWAYS_MASK_STATUSES
    # TODO: exit_status_value in _ALWAYS_ZERO_STATUSES

    last_msg = agent_output.main_trajectory[-1]
    last_content = last_msg.get('content') or last_msg.get('reasoning_content') or ''

    return EduOneAgentTaskOutput(
        prediction=last_content,
        question=task_input.question,
        answer=task_input.answer,
        trajectory=trajectory,
    )
