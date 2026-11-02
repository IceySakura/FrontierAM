"""
token_counter: 文本/图片 token 统计封装。

- 文本 token: 复用训练任务已加载的 HuggingFace tokenizer（由调用方在构造
  ``TokenCounter(tokenizer)`` 时显式传入），避免重复 load 5MB+ 的 tokenizer.json。
- 图片 token: 在已知图片宽高、resize 参数的情况下，按
  total = (H/P) * (W/P + 1) + 2 公式估算。
"""

import io
import math
import os

from typing import Any

import requests

from PIL import Image

from .image_resizer import resize_smart
from .image_utils import load_image_bytes


class TokenCounter:
    """文本 + 图片 token 统计器。

    文本统计直接调用外部传入的 HuggingFace tokenizer。
    """

    def __init__(self, tokenizer: Any, total_patch_size: int = 32, use_xrope: bool = True):
        if tokenizer is None:
            raise ValueError(
                'TokenCounter requires a non-None tokenizer; pass the training '
                'tokenizer explicitly when constructing TokenCounter(tokenizer).'
            )
        self.tokenizer = tokenizer
        self.total_patch_size = int(os.environ.get('IMG_RESIZE_FACTOR', total_patch_size))
        self.use_xrope = use_xrope

    # ───────── 文本 ─────────
    def _encode(self, text: str):
        return self.tokenizer(text, add_special_tokens=False)['input_ids']

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self._encode(text))
        except Exception:
            # 极端兜底：按字符数粗略估算
            return max(1, len(text) // 2)

    def truncate_text_by_tokens(self, text: str, max_tokens: int) -> tuple[str, int]:
        """按 token 数截断文本，返回 (截断后文本, 实际 token 数)。"""
        if not text or max_tokens <= 0:
            return '', 0
        ids = self._encode(text)
        if len(ids) <= max_tokens:
            return text, len(ids)
        ids = ids[:max_tokens]
        try:
            new_text = self.tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            new_text = text[: max_tokens * 2]
        return new_text, len(ids)

    # ───────── 图片 ─────────
    def count_image_tokens(self, w: int, h: int) -> int:
        """根据 patch 后的 (w, h) 计算图片占用的 token 数。

        公式：
          token 数 = (H/P) * (W/P + 1) + 2
        其中 +1 为行末 im_newline，+2 为首尾 image_token 标记。
        """
        patch_size = self.total_patch_size
        if w <= 0 or h <= 0:
            return 0
        cols = max(1, w // patch_size)
        rows = max(1, h // patch_size)
        # +1 为行末 im_newline，+2 为首尾 image_token 标记
        return rows * (cols + 1) + 2

    def estimate_image_tokens_after_resize(
        self, width: int, height: int, min_pixels: int, max_pixels: int, max_ratio: int = 200
    ) -> tuple[int, tuple[int, int]]:
        """对给定原图尺寸，按 resize_smart 计算 resize 后的尺寸和 token 数。"""
        try:
            new_w, new_h = resize_smart(
                width=width,
                height=height,
                factor=self.total_patch_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_ratio=max_ratio,
            )
        except AssertionError:
            # 宽高比超限，按 max_pixels 等比兜底
            scale = math.sqrt(max_pixels / max(width * height, 1))
            new_w = max(self.total_patch_size, (int(width * scale) // self.total_patch_size) * self.total_patch_size)
            new_h = max(self.total_patch_size, (int(height * scale) // self.total_patch_size) * self.total_patch_size)
        return self.count_image_tokens(new_w, new_h), (new_w, new_h)


# ───────── 工具：从图片源读取图片尺寸 ─────────
def get_image_size(image_source: Any, timeout: int = 10) -> tuple[int, int]:
    """读取图片宽高，支持本地路径、URL 与 patch dict。失败时抛出 IOError。"""
    try:
        if isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
            resp = requests.get(image_source, timeout=timeout)
            resp.raise_for_status()
            with Image.open(io.BytesIO(resp.content)) as img:
                return img.size  # (w, h)
        if isinstance(image_source, str) and os.path.exists(image_source):
            with Image.open(image_source) as img:
                return img.size
        if isinstance(image_source, dict):
            with Image.open(io.BytesIO(load_image_bytes(image_source))) as img:
                return img.size
        raise ValueError(f'无效的图片源: {image_source}')
    except Exception as e:
        raise OSError(f'读取图片尺寸失败: {e}') from e
