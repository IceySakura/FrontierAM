"""
image_resizer: 图片 resize 工具，与训练数据预处理侧保持一致。

包含：
  - resize_smart: 单图按 [min_pixels, max_pixels] 区间和 factor 对齐计算目标尺寸。
  - interleave_resize: 多图模式下基于剩余 token 配额动态计算每张图的 max_pixels；
    超过限制则退化为 interleave 最小分辨率。
"""

import math


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def resize_smart(
    width: int, height: int, factor: int, min_pixels: int, max_pixels: int, max_ratio: int = 200
) -> tuple[int, int]:
    """计算 resize 后的目标尺寸。

    返回 (new_width, new_height)，均为 factor 的整数倍。
    """
    if min(height, width) <= 0:
        raise AssertionError('invalid image size')
    aspect = max(height, width) / min(height, width)
    assert aspect < max_ratio, f'图片宽高比超出限制 {max_ratio}，实际为 {aspect:.2f}'

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    h_bar = max(factor, h_bar)
    w_bar = max(factor, w_bar)
    return w_bar, h_bar


def interleave_resize(
    image_count: int,
    reserved_pixel_tokens: int,
    factor: int = 32,
    base_max_pixels: int = 2048 * 2048,
    base_min_pixels: int = 256 * 256,
    multi_min_pixels: int = 256 * 256,
    interleave_max_size: int = 256,
    multi_nums_limit: int = 40,
    redundancy_nums: int = 50,
) -> tuple[int, int]:
    """根据图片数量和总像素 token 预算，动态计算每张图的 (min_pixels, max_pixels)。

    协议约定：所有返回的 ``min_pixels`` 都固定为 ``base_min_pixels``（默认 256*256），
    ``max_pixels`` 才是按图片数量与预算变化的部分；``multi_min_pixels`` 仅用作
    "多图退化阈值"，不参与最终 min_pixels 的取值。

    Args:
        image_count: 该 message 中图片数量。
        reserved_pixel_tokens: 留给所有图片合计可用的 token 数（已扣除文本占用）。
        factor: patch size。
        base_max_pixels: 单图理论最大像素上限。
        base_min_pixels: 单图理论最小像素下限，**所有分支统一返回这个值**。
        multi_min_pixels: 多图模式下退化判定阈值；若按预算分到的单图像素低于此值则退化。
        interleave_max_size: 退化模式下每张图最大边长。
        multi_nums_limit: 超过此数量直接退化为 interleave 模式。
        redundancy_nums: 多图占位冗余 token 数。

    Returns:
        (min_pixels, max_pixels)：每张图传入 resize_smart 时使用。
    """
    if image_count <= 0:
        return base_min_pixels, base_max_pixels

    if image_count == 1:
        # 单图：用全量预算约束 max_pixels（不超过 base_max_pixels）
        budget_pixels = max(0, reserved_pixel_tokens) * factor * factor
        max_pixels = min(base_max_pixels, max(multi_min_pixels, budget_pixels))
        return base_min_pixels, int(max_pixels)

    if image_count >= multi_nums_limit:
        # 退化路径 A：图片数过多，所有图统一压到 interleave_max_size²
        deg = interleave_max_size * interleave_max_size
        return base_min_pixels, deg

    # 多图：先估算每张图大概占用的 patch token，再分配剩余预算
    multi_images_max_size = int(math.sqrt(base_max_pixels))
    image_token_occupy = 4 * image_count + int(multi_images_max_size / factor) * image_count + redundancy_nums
    available_token = max(0, reserved_pixel_tokens - image_token_occupy)
    multi_images_total_pixels = available_token * factor * factor

    per_image_pixels = multi_images_total_pixels / image_count
    if per_image_pixels < multi_min_pixels:
        # 退化路径 B：每图分到的预算低于 multi_min_pixels，所有图统一压到 interleave_max_size²
        deg = interleave_max_size * interleave_max_size
        return base_min_pixels, deg

    # max_pixels = max(min(base_max_pixels, total/N), multi_min_pixels)
    max_pixels = max(min(base_max_pixels, per_image_pixels), multi_min_pixels)
    return base_min_pixels, int(max_pixels)
