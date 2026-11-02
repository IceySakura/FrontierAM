"""CUA-Gym prompt-caching strategy for sliding-window screenshot histories.

CUA-Gym 的 GUI agent 只保留最近 N 张截图，更早的图会被 elide 成占位文本
（见 ``main`` agent 的 ``_prepare_messages_for_inference``）。滑动窗口每轮把更靠前
的一张图改写成文本，因此通用的 ``last_n`` 策略（断点打在最近几条消息上）会落在改写
点之后：缓存前缀在 elide 边界分叉，被保留的图每轮都重新写缓存（cache_creation 飙升、
几乎没有 read）。

本策略把唯一的缓存断点打在 **首张图之前** —— elide 文本前缀的末尾。该前缀单调增长且
逐字节稳定（system + tools + 文本历史），能稳定命中；移动的图片尾部留在断点之后，按
普通输入计费，省去 1.25x 的缓存写惩罚。

只设置 ``cache_control`` 元数据，不改消息内容，模型输出（及 benchmark reward）不受影响。

通过 litellm 的可插拔策略机制接入（``cache_control_strategy`` 指向本函数的 import 路径），
litellm 公共代码不含任何 CUA-Gym 专有逻辑。
"""

from __future__ import annotations

import copy

from typing import Any


def _message_has_image(entry: dict[str, Any]) -> bool:
    content = entry.get('content')
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get('type') in {'image_url', 'image'} for part in content)


def _tag_message_cache_control(entry: dict[str, Any]) -> bool:
    """Tag a single message with cache_control. Returns True if successfully tagged."""
    content = entry.get('content')
    if content is None:
        return False
    if isinstance(content, str) and content.strip():
        entry['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        return True
    if isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict) and block.get('type') == 'text' and block.get('text', '').strip():
                block['cache_control'] = {'type': 'ephemeral'}
                return True
    return False


def image_window(messages: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
    """Place one cache breakpoint at the stable text prefix boundary before the first image.

    Signature matches litellm's pluggable strategy contract: ``(messages, **kwargs) -> messages``.
    Extra ``cache_control_*`` kwargs are accepted and ignored (this strategy is parameter-free).
    """
    new_messages = copy.deepcopy(messages)
    first_image_idx = next((i for i, m in enumerate(new_messages) if _message_has_image(m)), None)

    if first_image_idx is None:
        # No images yet: the whole history is stable — tag the last taggable message.
        for entry in reversed(new_messages):
            if _tag_message_cache_control(entry):
                break
        return new_messages

    # Tag the last stable (pre-image) message; fall back toward the front if needed.
    for idx in range(first_image_idx - 1, -1, -1):
        if _tag_message_cache_control(new_messages[idx]):
            break
    return new_messages
