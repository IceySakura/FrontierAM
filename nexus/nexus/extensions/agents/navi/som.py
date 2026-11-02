"""Set-of-Marks (SoM) 标注渲染模块。

在屏幕截图上绘制 a11y 实体的边界框和标签，用于为 LLM 提供带有元素 ID 标注的
可视化截图，使模型能够通过 ID 引用屏幕上的 UI 元素。
"""

from __future__ import annotations

import random
import re

from PIL import Image, ImageDraw, ImageFont


# 默认调色板，用于随机着色边界框和标签
COLORS = (
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
)

# 边界框默认样式：随机颜色描边，2px 线宽
BOX_RANDOM_COLOR_STYLE: dict = {
    'outlineWidth': 2,
    'stroke': 'rand(0)',
    'padding': '2 5',
}
# 标签默认样式：白色文字，随机背景色，放置在边界框右下角
LABEL_RANDOM_COLOR_STYLE: dict = {
    'anchor': 'top_right',
    'position': 'bottom_right',
    'stroke': 'white',
    'background': 'rand(0)',
    'padding': '2 5',
}

# 标签放置候选位置列表，按优先级排列 (position, anchor)
# 依次尝试外侧四角和内侧四角，选择第一个不与已有标签重叠的位置
LABEL_PLACEMENTS: tuple[tuple[str, str], ...] = (
    # Outer corners
    ('bottom_right', 'top_right'),
    ('bottom_left', 'top_left'),
    ('top_left', 'bottom_left'),
    ('top_right', 'bottom_right'),
    # Inner corners
    ('bottom_right', 'bottom_right'),
    ('bottom_left', 'bottom_left'),
    ('top_left', 'top_left'),
    ('top_right', 'top_right'),
)
# 锚点名称到归一化坐标的映射，用于定位标签相对于边界框的位置
ANCHORS: dict[str, tuple[float, float]] = {
    'center': (0.5, 0.5),
    'top_left': (0, 0),
    'top_right': (1, 0),
    'bottom_left': (0, 1),
    'bottom_right': (1, 1),
}


def draw_som(
    image: Image.Image,
    entities: list[dict] | tuple[dict, ...],
    color_mapping: dict[str, str] | None = None,
) -> Image.Image:
    """在屏幕截图上绘制 Set-of-Marks 标注（边界框 + 标签）。

    Args:
        image: 原始屏幕截图。
        entities: 实体列表，每个实体包含 ``label``、``type``、``shape`` 字段。
        color_mapping: 可选的实体类型到颜色的映射；为 ``None`` 时使用随机颜色。

    Returns:
        标注后的截图副本（不修改原图）。
    """

    image = image.copy()
    if not color_mapping:
        image = draw_bboxes(image, entities, default_style=BOX_RANDOM_COLOR_STYLE)
        return draw_labels(image, entities, default_style=LABEL_RANDOM_COLOR_STYLE)

    bbox_styles, label_styles = _build_styles(color_mapping)
    image = draw_bboxes(image, entities, styles=bbox_styles)
    return draw_labels(image, entities, styles=label_styles)


def _build_styles(color_mapping: dict[str, str]) -> tuple[dict[str, dict], dict[str, dict]]:
    """根据实体类型到颜色的映射，构建边界框和标签的样式字典。

    Args:
        color_mapping: 实体类型名称到颜色值的映射，如 ``{'button': '#ff0000'}``。

    Returns:
        (bbox_styles, label_styles) 两个字典，键为实体类型，值为对应的样式配置。
    """
    bbox_styles: dict[str, dict] = {}
    label_styles: dict[str, dict] = {}

    for entity_type, color in color_mapping.items():
        bbox_styles[entity_type] = {
            **BOX_RANDOM_COLOR_STYLE,
            'stroke': color,
        }
        label_styles[entity_type] = {
            **LABEL_RANDOM_COLOR_STYLE,
            'background': color,
        }

    return bbox_styles, label_styles


def draw_bboxes(
    image: Image.Image,
    entities: list[dict] | tuple[dict, ...],
    styles: dict[str, dict] | None = None,
    default_style: dict | None = None,
) -> Image.Image:
    """在图像上绘制所有实体的边界框。

    Args:
        image: 目标图像。
        entities: 实体列表。
        styles: 按实体类型区分的样式字典。
        default_style: 未匹配类型时的默认样式。

    Returns:
        绘制了边界框的图像。
    """

    styles = styles or {}
    for entity in entities:
        style = _style_for(entity, styles, default_style)
        if style is not None:
            image = draw_bbox(image, bbox=entity['shape'], style=style)
    return image


def draw_labels(
    image: Image.Image,
    entities: list[dict] | tuple[dict, ...],
    styles: dict[str, dict] | None = None,
    default_style: dict | None = None,
) -> Image.Image:
    """在实体边界框附近绘制标签，自动避免与文本实体和已有标签重叠。

    对每个实体，按 ``LABEL_PLACEMENTS`` 中的候选位置依次尝试，
    选择第一个不与已绘制矩形重叠的位置放置标签。

    Args:
        image: 目标图像。
        entities: 实体列表。
        styles: 按实体类型区分的样式字典。
        default_style: 未匹配类型时的默认样式。

    Returns:
        绘制了标签的图像。
    """

    styles = styles or {}
    drawn_rects = [_text_entity_rect(entity) for entity in entities if entity['type'] == 'text']

    for entity in entities:
        style = _style_for(entity, styles, default_style)
        if style is None:
            continue

        label = str(entity['label'])
        for index, (position, anchor) in enumerate(LABEL_PLACEMENTS):
            label_style = {
                **style,
                'position': position,
                'anchor': anchor,
            }
            rect = calc_label(bbox=entity['shape'], label=label, style=label_style)
            is_last_candidate = index == len(LABEL_PLACEMENTS) - 1

            if is_last_candidate or not any(check_overlap(rect, drawn_rect) for drawn_rect in drawn_rects):
                image = draw_label(image=image, bbox=entity['shape'], label=label, style=label_style)
                drawn_rects.append(rect)
                break

    return image


def _style_for(
    entity: dict,
    styles: dict[str, dict],
    default_style: dict | None,
) -> dict | None:
    """根据实体类型查找对应的样式，找不到则返回默认样式。

    Args:
        entity: 包含 ``type`` 字段的实体字典。
        styles: 实体类型到样式的映射。
        default_style: 未匹配到类型时使用的默认样式。

    Returns:
        匹配的样式字典，或 ``None`` 表示跳过该实体。
    """
    entity_type = str(entity['type'])
    if entity_type in styles:
        return styles[entity_type]
    return default_style


def _text_entity_rect(entity: dict) -> tuple[float, float, float, float]:
    """获取文本类型实体的矩形区域（内缩 1px），用于标签重叠检测。

    Args:
        entity: 包含 ``shape`` 字段的实体字典。

    Returns:
        ``(x, y, width, height)`` 元组。
    """
    shape = entity['shape']
    return (
        shape['x'] + 1,
        shape['y'] + 1,
        shape['width'] - 1,
        shape['height'] - 1,
    )


def read_color(style: dict, key: str, default: str | None, seed: int | float | None = None) -> str | None:
    """从样式字典中读取颜色值，支持 ``rand(offset)`` 随机颜色语法。"""
    if seed:
        random.seed(seed)
    color = style.get(key, default)
    if not color or color == 'transparent':
        return None
    if not isinstance(color, str):
        return str(color)
    if 'rand(' not in color:
        return color

    match = re.search(r'rand\((\d+)\)', color)
    offset = int(match.group(1)) if match else 0
    return COLORS[(offset + random.randint(0, len(COLORS))) % len(COLORS)]


def read_padding(style: dict, key: str, default: str) -> list[int]:
    """解析 CSS 风格的 padding 简写为 ``[top, right, bottom, left]`` 列表。

    支持 1~4 个值的简写形式，规则与 CSS padding 一致。

    Args:
        style: 样式字典。
        key: padding 属性名。
        default: 属性不存在时的默认值字符串。

    Returns:
        长度为 4 的整数列表 ``[top, right, bottom, left]``。

    Raises:
        ValueError: padding 值个数超过 4 个。
    """

    padding_values = [int(value) for value in str(style.get(key, default)).split()]
    if len(padding_values) == 1:
        return padding_values * 4
    if len(padding_values) == 2:
        return padding_values * 2
    if len(padding_values) == 3:
        return padding_values + padding_values[:1]
    if len(padding_values) == 4:
        return padding_values
    raise ValueError('Invalid padding value')


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载默认字体，用于标签文字渲染。"""
    return ImageFont.load_default()


def draw_bbox(image: Image.Image, bbox: dict, style: dict) -> Image.Image:
    """在图像上绘制单个边界框。

    Args:
        image: 目标图像。
        bbox: 边界框字典，包含 ``x``、``y``、``width``、``height``。
        style: 样式配置，支持 ``stroke``、``background``、``outlineWidth``。

    Returns:
        绘制了边界框的图像。
    """

    draw = ImageDraw.Draw(image)
    x, y = bbox['x'], bbox['y']
    width, height = bbox.get('width', 0), bbox.get('height', 0)
    seed = x + y + width + height

    draw.rectangle(
        [x, y, x + width, y + height],
        outline=read_color(style, 'stroke', 'red', seed=seed),
        fill=read_color(style, 'background', None, seed=seed),
        width=int(style.get('outlineWidth', 1)),
    )
    return image


def get_label_rect(
    label: str,
    bbox: dict,
    style: dict,
    padding: list[int] | tuple[int, ...],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[float, float, float, float]:
    """计算标签文字的渲染矩形区域。

    根据边界框位置、锚点设置和 padding 计算标签应放置的 ``(x, y, width, height)``。

    Args:
        label: 标签文本内容。
        bbox: 关联的边界框字典。
        style: 样式配置，包含 ``position`` 和 ``anchor``。
        padding: ``[top, right, bottom, left]`` 内边距。
        font: 用于测量文字尺寸的字体对象。

    Returns:
        ``(x, y, width, height)`` 元组。
    """

    x, y = bbox['x'], bbox['y']
    width, height = bbox.get('width', 0), bbox.get('height', 0)

    text_bbox = font.getbbox(label)
    text_width = text_bbox[2] - text_bbox[0] + padding[1] + padding[3]
    text_height = text_bbox[3] - text_bbox[1] + padding[0] + padding[2]

    position = str(style.get('position', 'center'))
    anchor = str(style.get('anchor', 'center'))

    label_x = x + width * ANCHORS[position][0] - text_width * ANCHORS[anchor][0]
    label_y = y + height * ANCHORS[position][1] - text_height * ANCHORS[anchor][1]

    return label_x, label_y, text_width, text_height


def check_overlap(rect1: tuple[float, float, float, float], rect2: tuple[float, float, float, float]) -> bool:
    """检查两个矩形是否重叠。

    Args:
        rect1: 第一个矩形 ``(x, y, width, height)``。
        rect2: 第二个矩形 ``(x, y, width, height)``。

    Returns:
        ``True`` 表示两个矩形存在重叠区域。
    """

    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def calc_label(bbox: dict, label: str, style: dict | None = None) -> tuple[float, float, float, float]:
    """计算单个实体标签占据的矩形区域。

    Args:
        bbox: 边界框字典。
        label: 标签文本。
        style: 可选的样式配置。

    Returns:
        ``(x, y, width, height)`` 元组。
    """

    style: dict = style or {}
    padding = read_padding(style, 'padding', '0 0 0 0')
    font = _load_font()
    return get_label_rect(label, bbox, style, padding, font)


def draw_label(image: Image.Image, bbox: dict, label: str, style: dict | None = None) -> Image.Image:
    """在图像上绘制单个标签（背景矩形 + 文字）。

    Args:
        image: 目标图像。
        bbox: 关联的边界框字典，用于定位和生成随机颜色种子。
        label: 标签文本。
        style: 可选的样式配置。

    Returns:
        绘制了标签的图像。
    """

    style: dict = style or {}
    draw = ImageDraw.Draw(image)
    seed = bbox['x'] + bbox['y'] + bbox.get('width', 0) + bbox.get('height', 0)

    stroke = read_color(style, 'stroke', None, seed=seed)
    background = read_color(style, 'background', None, seed=seed)
    padding = read_padding(style, 'padding', '0 0 0 0')
    font = _load_font()

    x, y, text_width, text_height = get_label_rect(label, bbox, style, padding, font)

    if background:
        draw.rectangle([x, y, x + text_width, y + text_height], fill=background)
    if stroke:
        draw.text((x + padding[3], y + (padding[0] / 2)), label, font=font, fill=stroke)

    return image
