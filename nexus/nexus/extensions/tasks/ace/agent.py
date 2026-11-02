"""ACE agent — 使用官方 ACEBench prompt 和 decoder

直接导入官方模块，零重复代码！
"""

import asyncio
import importlib
import json
import logging
import os
import sys

from pathlib import Path
from typing import Any

from openai import OpenAI


# 尝试导入 tqdm，如果不存在则使用 fallback
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm_cls: Any

    # Fallback: 简单的计数器
    class TqdmFallback:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable: Any = iterable
            self.total = total or (len(iterable) if iterable else 0)
            self.desc = desc
            self.n = 0
            if desc:
                logger.debug(f'{desc}: 0/{self.total}')

        def __iter__(self):
            for item in self.iterable or ():
                yield item
                self.update()

        def update(self, n=1):
            self.n += n
            if self.desc and self.n % max(1, self.total // 10) == 0:
                logger.debug(f'{self.desc}: {self.n}/{self.total}')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            if self.desc:
                logger.debug(f'{self.desc}: {self.n}/{self.total} - Complete')

    tqdm_cls = TqdmFallback

else:
    tqdm_cls = tqdm

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ContextLengthExceededError,
    agent_registry,
    require_context,
)
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)


# ============ 导入 ACEBench 官方模块 ============


def _setup_acebench_path() -> tuple[Path, str | None]:
    candidates = []

    env_path = os.getenv('ACE_BENCH_PATH')
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        Path(__file__).resolve().parents[2] / 'tasks' / 'ace' / 'ACEBench',
        Path(__file__).resolve().parents[4] / 'ACEBench-upstream',
    ])

    for ace_path in candidates:
        if (ace_path / 'eval_main.py').exists() and (ace_path / 'generate.py').exists():
            if str(ace_path) not in sys.path:
                sys.path.insert(0, str(ace_path))
            return ace_path, os.getenv('ACE_BENCH_DATASET_PATH')

    searched = ', '.join(str(p) for p in candidates)
    raise RuntimeError(
        f'ACEBench source tree not found. Set ACE_BENCH_PATH or place the official repo at one of: {searched}'
    )


ACE_BENCH_PATH, ACE_BENCH_DATASET_PATH = _setup_acebench_path()

_multi_step_agent = importlib.import_module('model_inference.multi_step.APIModel_agent')
_multi_step_execution = importlib.import_module('model_inference.multi_step.execution_role_step')
_multi_step_scene = importlib.import_module('model_inference.multi_step.multi_step_scene')
_multi_turn_agent = importlib.import_module('model_inference.multi_turn.APIModel_agent')
_multi_turn_execution = importlib.import_module('model_inference.multi_turn.execution_role')
_multi_turn_scene = importlib.import_module('model_inference.multi_turn.multi_turn_scene')
_multi_turn_user = importlib.import_module('model_inference.multi_turn.APIModel_user')
_prompt_en = importlib.import_module('model_inference.prompt_en')
_prompt_zh = importlib.import_module('model_inference.prompt_zh')
_utils = importlib.import_module('model_inference.utils')

APIAgent_step = _multi_step_agent.APIAgent_step
APIAgent_turn = _multi_turn_agent.APIAgent_turn
APIUSER = _multi_turn_user.APIUSER
decode_ast = _utils.decode_ast
EXECUTION = _multi_turn_execution.EXECUTION
EXECUTION_STEP = _multi_step_execution.EXECUTION_STEP
Mulit_Step_Scene = _multi_step_scene.Mulit_Step_Scene
Scene = _multi_turn_scene.Scene
SYSTEM_PROMPT_FOR_NORMAL_DATA_EN = _prompt_en.SYSTEM_PROMPT_FOR_NORMAL_DATA_EN
SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH = _prompt_zh.SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH
SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN = _prompt_en.SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN
SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH = _prompt_zh.SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH
SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN = _prompt_en.SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN
SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH = _prompt_zh.SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH
USER_PROMPT_EN = _prompt_en.USER_PROMPT_EN
USER_PROMPT_ZH = _prompt_zh.USER_PROMPT_ZH


# ============ Agent Input/Output ============


class AceAgentInput(AgentRunInput):
    """ACE agent input"""

    category: str = 'normal'
    language: str = 'en'
    task_data: dict[str, Any] | None = None
    api_pool: list[dict[str, Any]] = []
    model_name: str = 'default'
    max_tokens: int = 4096
    temperature: float = 0.0


class AceAgentOutput(AgentRunOutput):
    """ACE agent output"""

    model_response: str = ''
    tool_calls: list[dict[str, Any]]
    category: str = ''
    task_id: str = ''
    termination_reason: str | None = None
    error: str | None = None


# ============ 辅助函数 ============


def _convert_int_keys_to_str(obj: Any) -> Any:
    """递归转换 dict 中的整数键为字符串，匹配 JSON ground_truth。"""
    if isinstance(obj, dict):
        return {str(k): _convert_int_keys_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_int_keys_to_str(item) for item in obj]
    return obj


def _extract_outermost_bracket_content(text: str) -> str | None:
    """提取最外层 [...] 内容（官方实现）"""
    start = -1
    depth = 0
    for i, char in enumerate(text):
        if char == '[':
            if depth == 0:
                start = i
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


def _build_prompts_official(
    category: str,
    language: str,
    task_data: dict[str, Any],
    api_pool: list[dict[str, Any]],
) -> tuple[str, str]:
    """使用官方 Prompt 模板"""

    # 格式化 API 定义
    function_str = json.dumps(api_pool, ensure_ascii=False, indent=2)

    # 提取问题、时间、profile
    question = task_data.get('question', '')
    time = task_data.get('time', '')
    profile = task_data.get('profile', '')  # ✅ 添加 profile 支持

    # 选择模板
    if language == 'zh':
        if 'special' in category:
            system_template = SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH
            system_prompt = system_template.format(function=function_str, time=time)
        elif 'preference' in category:
            system_template = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH
            system_prompt = system_template.format(profile=profile, function=function_str)
        else:
            system_template = SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH
            system_prompt = system_template.format(function=function_str, time=time)
        user_template = USER_PROMPT_ZH
    else:
        if 'special' in category:
            system_template = SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN
            system_prompt = system_template.format(function=function_str, time=time)
        elif 'preference' in category:
            system_template = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN
            system_prompt = system_template.format(profile=profile, function=function_str)
        else:
            system_template = SYSTEM_PROMPT_FOR_NORMAL_DATA_EN
            system_prompt = system_template.format(function=function_str, time=time)
        user_template = USER_PROMPT_EN

    # 格式化 user prompt
    user_prompt = user_template.format(question=question)

    return system_prompt, user_prompt


def _parse_tool_calls_official(response_text: str, model_name: str = 'default') -> list[dict[str, Any]]:
    """使用官方 AST decoder 解析工具调用"""

    try:
        # 提取 [...] 内容
        bracket_content = _extract_outermost_bracket_content(response_text)
        if not bracket_content:
            logger.warning('No bracket content found in response')
            return []

        # 使用官方 decoder
        tool_calls = decode_ast(model_name, bracket_content)
        logger.debug(f'Decoded {len(tool_calls)} tool calls')
        return tool_calls

    except Exception as e:
        logger.error(f'Failed to decode with official AST decoder: {e}')
        return []


def _report_termination_span(termination_reason: str | None) -> None:
    """Report termination to nexus span"""
    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={'_hytuner_termination_reason': termination_reason or 'UNKNOWN'},
        )


# ============ Agent 模拟器 ============


def _run_agent_multi_turn_sync(
    question: str,
    initial_config: dict[str, Any],
    functions: list[dict[str, Any]],
    involved_classes: list[str],
    test_id: str,
    time: str,
    model_name: str,
    language: str,
    max_dialog_turns: int = 40,
    user_model: str = 'gpt-4o',
    client: OpenAI = OpenAI(base_url='', api_key=''),
    user_client: OpenAI = OpenAI(base_url='', api_key=''),
) -> tuple[list[dict[str, Any]], list[str]]:
    """运行 agent_multi_turn 模拟（同步版本）

    Returns:
        (result_list, mile_stone): 类状态列表和函数调用历史
    """

    logger.debug(f'Starting agent_multi_turn simulation: test_id={test_id}, max_turns={max_dialog_turns}')
    logger.debug(f'Agent model: {model_name}, User model: {user_model}')

    # 创建角色
    logger.debug('Step 1/4: Creating agent, user, and execution roles...')
    agent = APIAgent_turn(
        client=client,
        model_name=model_name,
        time=time,
        functions=functions,
        involved_class=involved_classes,
        language=language,
    )

    user = APIUSER(client=user_client, model_name=user_model, involved_class=involved_classes, language=language)

    execution = EXECUTION(
        agent_model_name=model_name,
        initial_config=initial_config,
        involved_classes=involved_classes,
        test_id=test_id,
        language=language,
    )
    logger.debug('✓ All roles created successfully')

    # 初始化场景
    logger.debug('Step 2/4: Initializing scene...')

    async def inner():
        return await user.get_init_prompt(question)

    init_message = asyncio.run(inner())
    # init_message = await user.get_init_prompt(question)

    scene = Scene(
        initial_state=initial_config,
        functions=functions,
        agent_role=agent,
        user_role=user,
        init_message=init_message,
        language=language,
    )
    logger.debug('✓ Scene initialized')

    message_history = scene.dialogue_history
    result_instance_list = []
    mile_stone = []

    # 多轮对话循环
    logger.debug(f'Step 3/4: Starting multi-turn dialogue (max {max_dialog_turns} turns)...')

    with tqdm_cls(total=max_dialog_turns, desc='Multi-turn Dialogue', unit='turn', disable=not HAS_TQDM) as pbar:
        for index in range(max_dialog_turns):
            last_recipient = message_history[-1]['recipient']
            if last_recipient == 'user':
                # User 发给 Agent
                inference_message = scene.get_inference_message()
                user.step(message_history[-1]['message'])
                # current_message = user.respond()

                current_message = asyncio.run(user.respond())

            elif last_recipient == 'agent':
                # Agent 推理
                logger.debug(f'[Turn {index + 1}/{max_dialog_turns}] Agent thinking...')
                inference_message = scene.get_inference_message()
                # current_message = agent.respond(inference_message)

                current_message = asyncio.run(agent.respond(inference_message))

            else:
                # Execution 执行
                logger.debug(f'[Turn {index + 1}/{max_dialog_turns}] Executing action...')
                inference_message = scene.get_inference_message()
                mile_stone_message = message_history[-1]['message']
                mile_stone.append(mile_stone_message)
                current_message, result_instance = execution.respond(message_history)

                # 收集类实例
                if result_instance not in result_instance_list:
                    result_instance_list.append(result_instance)

            # 添加到对话历史
            scene.add_dialogue(current_message)

            # 检查是否结束
            if current_message.get('message') == 'FINISH':
                logger.info(f'✓ Conversation finished at turn {index + 1}')
                pbar.update(max_dialog_turns - index)  # 完成剩余进度
                break

            pbar.update(1)

    # 导出对话历史
    scene.write_message_history(test_id, model_name)

    # 提取最终状态
    saved_class = {
        'BaseApi': ['wifi', 'logged_in'],
        'MessageApi': ['inbox'],
        'ReminderApi': ['reminder_list'],
        'FoodPlatform': ['users', 'logged_in_users', 'orders'],
        'Finance': [
            'user_accounts',
            'is_logged_in',
            'deposit_history',
            'withdrawal_history',
            'loan_history',
            'orders',
            'holdings',
        ],
        'Travel': ['users', 'reservations'],
    }

    result_list = []
    for result_instance in result_instance_list:
        for name, instance in result_instance.items():
            item_dict = {}
            for item in instance.__dict__:
                if item in saved_class.get(name, []):
                    value = instance.__dict__[item]
                    value = _convert_int_keys_to_str(value)
                    item_dict[item] = value
            result_list.append({name: item_dict})

    logger.info(f'✓ agent_multi_turn finished: {len(result_list)} results, {len(mile_stone)} milestones')
    return result_list, mile_stone


def _run_agent_multi_step_sync(
    question: str,
    initial_config: dict[str, Any],
    functions: list[dict[str, Any]],
    involved_classes: list[str],
    test_id: str,
    model_name: str,
    language: str,
    max_dialog_turns: int = 40,
    client: OpenAI = OpenAI(base_url='', api_key=''),
    user_client: OpenAI = OpenAI(base_url='', api_key=''),
) -> tuple[list[dict[str, Any]], list[str]]:
    """运行 agent_multi_step 模拟（同步版本）

    Returns:
        (result_list, mile_stone): 实例列表和里程碑
    """

    logger.debug(f'Starting agent_multi_step simulation: test_id={test_id}')
    logger.debug(f'Model: {model_name}, Language: {language}, Functions: {len(functions)}')

    # 创建 agent
    logger.debug('Step 1/4: Creating APIAgent_step...')
    agent = APIAgent_step(
        client=client,
        model_name=model_name,
        time='',  # 官方代码使用空字符串
        functions=functions,
        language=language,
    )
    logger.debug('✓ Agent created successfully')

    # 创建场景
    logger.debug('Step 2/4: Creating multi-step scene...')
    scene = Mulit_Step_Scene(
        question=question, initial_state=initial_config, functions=functions, agent_role=agent, language=language
    )
    logger.debug('✓ Scene created successfully')

    # 创建 execution
    logger.debug('Step 3/4: Creating EXECUTION_STEP...')
    execution = EXECUTION_STEP(
        agent_model_name=model_name,
        initial_config=initial_config,
        involved_classes=involved_classes,
        test_id=test_id,
        language=language,
    )
    logger.debug('✓ Execution created successfully')

    message_history = scene.dialogue_history
    result_instance_list = []
    mile_stone = []

    # 执行 step-by-step 对话
    logger.debug(f'Step 4/4: Running step-by-step dialogue (max {max_dialog_turns} turns)...')
    with tqdm_cls(total=max_dialog_turns, desc='Multi-step Dialogue', unit='turn', disable=not HAS_TQDM) as pbar:
        for index in range(max_dialog_turns):
            last_sender = message_history[-1]['sender']

            if index == 0 or last_sender == 'execution':
                # Agent 轮
                logger.debug(f'[Turn {index + 1}/{max_dialog_turns}] Agent thinking...')
                inference_message = scene.get_inference_message()
                # current_message = await agent.respond(inference_message)

                current_message = asyncio.run(agent.respond(inference_message))
            else:
                # Execution 轮
                logger.debug(f'[Turn {index + 1}/{max_dialog_turns}] Executing action...')
                try:
                    current_message, result_instance = execution.respond(message_history)
                    mile_stone_message = message_history[-1]['message']
                    mile_stone.append(mile_stone_message)
                    if result_instance not in result_instance_list:
                        result_instance_list.append(result_instance)
                except Exception as e:
                    logger.error(f'❌ Execution failed at turn {index + 1}: {e}')
                    raise

            scene.add_dialogue(current_message)

            # 检查是否完成
            if index > 1 and 'finish conversation' in current_message['message']:
                logger.info(f'✓ Conversation finished at turn {index + 1}')
                pbar.update(max_dialog_turns - index)  # 跳过剩余进度
                break

            pbar.update(1)

    # 导出对话历史
    scene.write_message_history(test_id, model_name)

    # 提取最终状态
    # SAVED_CLASS 定义在 apimodel_inference.py
    saved_class = {
        'BaseApi': ['wifi', 'logged_in'],
        'MessageApi': ['inbox'],
        'ReminderApi': ['reminder_list'],
        'FoodPlatform': ['users', 'logged_in_users', 'orders'],
        'Finance': [
            'user_accounts',
            'is_logged_in',
            'deposit_history',
            'withdrawal_history',
            'loan_history',
            'orders',
            'holdings',
        ],
        'Travel': ['users', 'reservations'],
    }

    result_list = []
    for result_instance in result_instance_list:
        for name, instance in result_instance.items():
            item_dict = {}
            for item in instance.__dict__:
                if item in saved_class.get(name, []):
                    value = instance.__dict__[item]
                    value = _convert_int_keys_to_str(value)
                    item_dict[item] = value
            result_list.append({name: item_dict})

    logger.info(f'✓ agent_multi_step finished: {len(result_list)} results, {len(mile_stone)} milestones')
    return result_list, mile_stone


# ============ Agent 实现 ============


@agent_registry.register('ace_agent')
async def run_ace_agent(agent_input: AceAgentInput) -> AceAgentOutput:
    """ACE agent - 使用官方 prompt 和 decoder"""

    if agent_input.task_data is None:
        raise ValueError('task_data is required')

    task_id = agent_input.task_data.get('id', 'unknown')
    category = agent_input.category
    language = agent_input.language

    logger.debug(f'Running ACE agent: task_id={task_id}, category={category}, language={language}')

    ctx = require_context()
    client = ctx.openai_clients['main']
    user_client = ctx.openai_clients['user']

    # ✅ Agent 类别：使用官方模拟器
    if category.startswith('agent_'):
        logger.debug(f'Using official agent simulator for category: {category}')

        # 提取必需字段
        question = agent_input.task_data.get('question', '')
        initial_config = agent_input.task_data.get('initial_config', {})
        functions = agent_input.api_pool
        involved_classes = agent_input.task_data.get('involved_classes', [])
        time = agent_input.task_data.get('time', '')

        # ✅ User 模型配置（可独立于 Agent 模型）
        user_model = os.getenv('ACE_USER_MODEL', 'gpt-4o')
        os.getenv('ACE_USER_API_KEY', os.getenv('ACE_AGENT_API_KEY', ''))
        user_base_url = os.getenv('ACE_USER_BASE_URL', os.getenv('ACE_AGENT_BASE_URL', ''))

        logger.debug(f'Agent model: {agent_input.model_name}')
        logger.debug(f'User model: {user_model}')
        logger.debug(f'User API: {user_base_url}')

        try:
            if 'multi_turn' in category:
                # Multi-turn 模拟

                result_list, mile_stone = await asyncio.get_event_loop().run_in_executor(
                    None,
                    _run_agent_multi_turn_sync,
                    question,
                    initial_config,
                    functions,
                    involved_classes,
                    task_id,
                    time,
                    agent_input.model_name,
                    language,
                    40,  # max_dialog_turns
                    user_model,
                    client,
                    user_client,
                )

                # 最终结果是 mile_stone 的最后一个（函数调用历史）
                mile_stone[-1] if mile_stone else ''
            else:
                # Multi-step 模拟

                result_list, mile_stone = await asyncio.get_event_loop().run_in_executor(
                    None,
                    _run_agent_multi_step_sync,
                    question,
                    initial_config,
                    functions,
                    involved_classes,
                    task_id,
                    agent_input.model_name,
                    language,
                    40,  # max_dialog_turns
                    client,
                    user_client,
                )

                # 最终结果是 mile_stone 的最后一个（函数调用历史）
                mile_stone[-1] if mile_stone else ''

            # ❗ agent 类别用 result_list（类状态），不用 mile_stone（函数调用）
            model_response = result_list
            tool_calls = mile_stone  # ✅ 使用类状态，不是函数调用

            termination_reason = 'SUCCESS'
            error = None

        except Exception as e:
            model_response = ''
            tool_calls = []
            termination_reason = 'ERROR'
            error = str(e)

        tool_calls2 = [{str(idx): tool_call} for idx, tool_call in enumerate(tool_calls, start=1)]
        return AceAgentOutput(
            model_response=json.dumps(model_response, ensure_ascii=False),
            tool_calls=tool_calls2,
            category=category,
            task_id=task_id,
            termination_reason=termination_reason,
            error=error,
        )

    # Normal/Special/Preference: 单轮 API 调用
    logger.debug('Using single-turn API call')

    # 使用官方 Prompt 模板构建消息
    system_prompt, user_prompt = _build_prompts_official(
        category, language, agent_input.task_data, agent_input.api_pool
    )

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]

    # 初始化输出
    error = None
    termination_reason = None
    model_response = ''
    tool_calls = []

    try:
        # 调用 OpenAI API
        logger.debug(f'Calling API: model={agent_input.model_name}, max_tokens={agent_input.max_tokens}')
        logger.debug(f'System prompt: {len(system_prompt)} chars, User prompt: {len(user_prompt)} chars')

        response = await client.create_completion(
            model=agent_input.model_name,
            messages=messages,
            max_tokens=agent_input.max_tokens,
            temperature=agent_input.temperature,
        )

        logger.debug(f'Got API response: {len(response.choices)} choices')

        if response is None or not response.choices:
            raise RuntimeError('Empty response from model')

        # 提取响应
        message = response.choices[0].message
        model_response = message.content or ''

        # 使用官方 decoder 解析
        tool_calls = _parse_tool_calls_official(model_response, agent_input.model_name)

        termination_reason = 'SUCCESS'

    except ContextLengthExceededError as e:
        logger.warning(f'Context length exceeded: {e}')
        termination_reason = 'TRUNCATION'
        error = str(e)

    except Exception as e:
        logger.error(f'Agent error: {type(e).__name__}: {e}')
        termination_reason = 'ERROR'
        error = str(e)

    logger.info(f'Agent completed: termination={termination_reason}, tool_calls={len(tool_calls)}')
    _report_termination_span(termination_reason)

    return AceAgentOutput(
        model_response=model_response,
        tool_calls=tool_calls,
        category=category,
        task_id=str(task_id),
        termination_reason=termination_reason,
        error=error,
    )
