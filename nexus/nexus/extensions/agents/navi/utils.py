"""Visual and accessibility helpers for the Nexus Navi agent."""

import base64
import io

from typing import Any

from PIL import Image


def coerce_prompt(value: str | list[Any]) -> str:
    """将用户提示统一转换为纯文本字符串。

    Navi 仅接受纯文本任务指令。当输入为多模态消息列表时，
    提取其中所有文本部分并用换行符拼接。

    Args:
        value: 纯文本字符串，或 OpenAI 多模态 content 列表。

    Returns:
        拼接后的纯文本指令。
    """
    if isinstance(value, str):
        return value

    # 遍历多模态消息列表，提取文本部分
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get('type') == 'text':
            parts.append(str(item.get('text', '')))
        else:
            parts.append(str(item))
    return '\n'.join(part for part in parts if part)


def image_from_bytes(value: bytes) -> Image.Image:
    """将原始图片字节解码为独立的 PIL Image 对象。

    使用 ``image.load()`` + ``image.copy()`` 确保返回的图片
    不依赖底层的 BytesIO 缓冲区，可安全在外部使用。

    Args:
        value: 原始图片字节数据（PNG / JPEG 等格式）。

    Returns:
        解码后的 PIL Image 对象。
    """
    with Image.open(io.BytesIO(value)) as image:
        image.load()  # 强制将像素数据加载到内存
        return image.copy()  # 返回独立副本，脱离底层 IO 缓冲区


def image_to_base64_png(image: Image.Image) -> str:
    """将 PIL Image 编码为 base64 PNG 字符串。

    Args:
        image: 待编码的 PIL Image 对象。

    Returns:
        base64 编码的 PNG 字符串。
    """
    with io.BytesIO() as buffer:
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


def image_to_data_url(image: Image.Image) -> str:
    """将 PIL Image 编码为内联 PNG data URL。

    生成的 URL 格式为 ``data:image/png;base64,...``，可直接嵌入
    OpenAI Vision API 的 ``image_url`` 字段。

    Args:
        image: 待编码的 PIL Image 对象。

    Returns:
        完整的 data URL 字符串。
    """
    return f'data:image/png;base64,{image_to_base64_png(image)}'


def resize_image(
    image: Image.Image,
    max_size: int = 2048,
    target_short_side: int = 768,
) -> Image.Image:
    """按视觉模型输入约束缩放图片。

    缩放策略（按顺序应用）：
    1. 若最长边超过 ``max_size``，等比缩小至最长边 = ``max_size``。
    2. 若最短边仍超过 ``target_short_side``，再次等比缩小至最短边 = ``target_short_side``。

    若图片已满足所有约束则直接返回原图，不做任何缩放。

    Args:
        image: 待缩放的 PIL Image 对象。
        max_size: 最长边允许的最大像素值，默认 2048。
        target_short_side: 最短边的目标像素值，默认 768。

    Returns:
        缩放后的 PIL Image 对象（或原图）。
    """
    width, height = image.size

    # 已满足所有约束，无需缩放
    if width <= max_size and height <= max_size and min(width, height) <= target_short_side:
        return image

    # 第一步：确保最长边不超过 max_size
    max_dim = max(width, height)
    if max_dim > max_size:
        scale = max_size / max_dim
        width = int(width * scale)
        height = int(height * scale)

    # 第二步：确保最短边不超过 target_short_side
    min_dim = min(width, height)
    if min_dim > target_short_side:
        scale = target_short_side / min_dim
        width = int(width * scale)
        height = int(height * scale)

    return image.resize((width, height))
