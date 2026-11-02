"""观测模块 — 从 WAA 运行时采集屏幕截图、无障碍树等环境信息。

该模块封装了一次完整的环境观测流程，对应原始 WAA 中 ``DesktopEnv._get_obs`` 的逻辑。
"""

import random

from dataclasses import dataclass
from typing import Any

from PIL import Image

from nexus.extensions.agents.navi.a11y import extract_a11y_entities
from nexus.extensions.agents.navi.som import draw_som
from nexus.extensions.agents.navi.utils import image_from_bytes
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


@dataclass
class NaviAgentObservation:
    """单次环境观测结果，包含截图、无障碍树和窗口元信息。"""

    screenshot: Image.Image
    """原始屏幕截图。"""

    a11y_tree: str
    """原始无障碍树 XML 字符串。"""

    a11y_entities: list[dict[str, Any]]
    """从无障碍树中提取并标注 label 后的可交互元素列表。"""

    annotated_screenshot: Image.Image
    """带有 Set-of-Marks 标注的屏幕截图。"""

    fg_window_title: str
    """当前前台窗口标题。"""

    fg_window_names: str
    """所有打开窗口名称的拼接字符串。"""

    human_input: str | None
    """用户额外输入（如有）。"""

    computer_clipboard: str | None
    """当前剪贴板内容（如有）。"""


async def observe(
    runtime: AGSRawWindowsAgentArenaRuntime,
    a11y_backend: str = 'uia',
) -> NaviAgentObservation:
    """从 WAA 运行时采集一次完整的环境观测。

    流程：
    1. 获取窗口元信息（标题、窗口列表、剪贴板等）
    2. 截取屏幕截图
    3. 获取无障碍树并提取可交互元素
    4. 对元素进行随机打乱并分配数字 label
    5. 在截图上绘制 Set-of-Marks 标注

    Args:
        runtime: WAA 原始运行时实例。
        a11y_backend: 无障碍树后端类型，默认 ``'uia'``。

    Returns:
        NaviAgentObservation: 包含截图、无障碍树和标注信息的观测结果。

    Raises:
        AssertionError: 当无障碍树为 None 或非字符串时抛出。
    """

    # 获取窗口元信息（标题、窗口列表、剪贴板等）
    obs = await runtime.obs_winagent()

    # 截取屏幕截图
    screenshot_bytes = await runtime.screenshot()
    screenshot = image_from_bytes(screenshot_bytes)

    # 获取无障碍树
    a11y_data = await runtime.accessibility_tree(backend=a11y_backend)
    assert isinstance(a11y_data, dict), f'accessibility response is not a dict: {a11y_data}'
    if a11y_backend == 'uia':
        actual_backend = a11y_data.get('backend')
        fallback_backend = a11y_data.get('fallback_backend')
        uia_error = a11y_data.get('uia_error')
        assert actual_backend in (None, 'uia'), (
            f'accessibility backend mismatch: requested=uia backend={actual_backend} data={a11y_data}'
        )
        assert not fallback_backend, (
            'accessibility returned fallback instead of UIA: '
            f'backend={actual_backend} fallback_backend={fallback_backend} uia_error={uia_error}'
        )
        assert not uia_error, (
            'UIA returned an error instead of UIA: '
            f'backend={actual_backend} fallback_backend={fallback_backend} uia_error={uia_error}'
        )
    a11y_tree = a11y_data.get('AT') if isinstance(a11y_data, dict) else None
    assert a11y_tree is not None, f'accessibility tree is None: {a11y_data}'
    assert isinstance(a11y_tree, str), f'accessibility tree is not a string: {a11y_data}'

    # 提取可交互元素并过滤无效区域（坐标为负或面积过小的元素）
    a11y_entities = [
        ent
        for ent in extract_a11y_entities(a11y_tree)
        if (
            ent['shape']['x'] >= 0
            and ent['shape']['y'] >= 0
            and ent['shape']['width'] > 0
            and ent['shape']['height'] > 0
            and ent['shape']['width'] * ent['shape']['height'] > 1
        )
    ]

    # 随机打乱元素顺序并分配数字 label，避免模型依赖固定顺序
    random.shuffle(a11y_entities)
    entities = [{**ent, 'label': str(i)} for i, ent in enumerate(a11y_entities)]

    return NaviAgentObservation(
        screenshot=screenshot,
        a11y_tree=a11y_tree,
        a11y_entities=entities,
        annotated_screenshot=draw_som(screenshot, entities),
        fg_window_title=obs.get('window_title', ''),
        fg_window_names=obs.get('window_names_str', ''),
        human_input=obs.get('human_input', None),
        computer_clipboard=obs.get('computer_clipboard', None),
    )
