"""无障碍树（Accessibility Tree）解析与实体提取工具。

从 AT-SPI 格式的 XML 无障碍树中提取可交互的 UI 元素，
用于 Navi agent 的 Set-of-Marks 标注和动作规划。
"""

import xml.etree.ElementTree as ET


def is_leaf_node(node):
    """判断节点是否为叶子节点（无子元素）。"""
    return len(list(node)) == 0


def is_visible(node):
    """判断节点在 AT-SPI 状态中是否标记为可见。"""
    visible = node.get('{uri:deskat:state.at-spi.gnome.org}visible', 'false')
    return visible == 'true'


def is_enabled(node):
    """判断节点在 AT-SPI 状态中是否标记为启用（可交互）。"""
    visible = node.get('{uri:deskat:state.at-spi.gnome.org}enabled', 'false')
    return visible == 'true'


def has_coords(node):
    """判断节点是否包含屏幕坐标和尺寸信息。"""
    position = node.get('{uri:deskat:component.at-spi.gnome.org}screencoord', '')
    size = node.get('{uri:deskat:component.at-spi.gnome.org}size', '')
    return bool(position) and bool(size)


def center_of_bbox(position, size):
    """计算边界框的中心点坐标。

    Args:
        position: 左上角坐标 ``[x, y]``。
        size: 宽高 ``[width, height]``。

    Returns:
        中心点坐标 ``(center_x, center_y)``。
    """
    center_x = position[0] + size[0] // 2
    center_y = position[1] + size[1] // 2
    return center_x, center_y


def rect_overlaps(center, bbox_position, bbox_size):
    """判断一个点是否落在矩形区域内。

    Args:
        center: 待检测的点坐标 ``(x, y)``。
        bbox_position: 矩形左上角坐标 ``[x, y]``。
        bbox_size: 矩形宽高 ``[width, height]``。

    Returns:
        点在矩形内则返回 ``True``。
    """
    x, y = center
    bbox_left = bbox_position[0]
    bbox_top = bbox_position[1]
    bbox_right = bbox_left + bbox_size[0]
    bbox_bottom = bbox_top + bbox_size[1]
    return bbox_left <= x <= bbox_right and bbox_top <= y <= bbox_bottom


def overlap_ratio(bbox1_position, bbox1_size, bbox2_position, bbox2_size):
    """计算两个矩形的重叠比率（Dice 系数）。

    返回值为 ``2 * 交集面积 / (面积1 + 面积2)``，范围 ``[0, 1]``。
    任一矩形面积为 0 时返回 0。

    Args:
        bbox1_position: 第一个矩形左上角坐标。
        bbox1_size: 第一个矩形宽高。
        bbox2_position: 第二个矩形左上角坐标。
        bbox2_size: 第二个矩形宽高。
    """
    bbox1_area = bbox1_size[0] * bbox1_size[1]
    bbox2_area = bbox2_size[0] * bbox2_size[1]
    if bbox1_area == 0 or bbox2_area == 0:
        return 0

    x1, y1 = bbox1_position
    x2, y2 = bbox2_position
    x_overlap = max(0, min(x1 + bbox1_size[0], x2 + bbox2_size[0]) - max(x1, x2))
    y_overlap = max(0, min(y1 + bbox1_size[1], y2 + bbox2_size[1]) - max(y1, y2))
    overlap_area = x_overlap * y_overlap
    return overlap_area * 2 / (bbox1_area + bbox2_area)


def filter_redundant_entities(entities, th=0.8):
    """过滤高度重叠的冗余实体。

    逐对比较实体的重叠比率，移除超过阈值 ``th`` 的重复项。

    Args:
        entities: 实体列表，每个实体包含 ``shape`` 字段。
        th: 重叠比率阈值，超过此值则视为冗余，默认 0.8。

    Returns:
        去重后的实体列表。
    """
    # 逐对比较，移除重叠比率超过阈值的实体
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if (
                overlap_ratio(
                    (entities[i]['shape']['x'], entities[i]['shape']['y']),
                    (entities[i]['shape']['width'], entities[i]['shape']['height']),
                    (entities[j]['shape']['x'], entities[j]['shape']['y']),
                    (entities[j]['shape']['width'], entities[j]['shape']['height']),
                )
                > th
            ):
                entities.pop(j)
                break
    return entities


def reverse_a11y_windows(accessibility_tree: str) -> str:
    """反转无障碍树中顶层窗口的顺序。

    将窗口列表反转，使最前面的窗口（Z-order 最高）排在最后处理，
    从而在实体提取时实现正确的遮挡关系。

    Args:
        accessibility_tree: AT-SPI 格式的 XML 无障碍树字符串。

    Returns:
        窗口顺序反转后的 XML 字符串。
    """
    desktop = ET.fromstring(accessibility_tree)

    if desktop is not None:
        children = list(desktop)[::-1]
        desktop.clear()
        for child in children:
            desktop.append(child)

    return ET.tostring(desktop, encoding='unicode')


def extract_a11y_entities(accessibility_tree: str | None) -> list[dict]:
    """从 AT-SPI XML 无障碍树中提取可交互的 UI 实体。

    处理流程：
    1. 反转窗口顺序以正确处理遮挡关系。
    2. 按窗口逐层遍历，提取可见、启用且有坐标的叶子节点。
    3. 每处理一个新窗口时，移除被该窗口遮挡的已有实体。
    4. 最后过滤高度重叠的冗余实体。

    Args:
        accessibility_tree: AT-SPI 格式的 XML 字符串，为 ``None`` 时返回空列表。

    Returns:
        实体字典列表，每个实体包含 ``from``、``type``、``shape``、``text`` 字段。
    """
    if not accessibility_tree:
        return []

    entities = []
    a11y_tree = reverse_a11y_windows(accessibility_tree)
    desktop = ET.fromstring(a11y_tree)

    def iter_tree(node):
        """递归遍历节点树，收集可见且可交互的叶子节点。"""
        if is_visible(node):
            if is_leaf_node(node) and is_visible(node) and is_enabled(node) and has_coords(node):
                name = node.attrib.get('name', '')
                position = node.get('{uri:deskat:component.at-spi.gnome.org}screencoord', '')
                size = node.get('{uri:deskat:component.at-spi.gnome.org}size', '')

                if position and size:
                    position = [int(coord.strip('()')) for coord in position.split(',')]
                    size = [int(dim.strip('()')) for dim in size.split(',')]
                    entities.append((
                        center_of_bbox(position, size),
                        {
                            'from': 'a11y',
                            'type': 'a11y',
                            'shape': {'x': position[0], 'y': position[1], 'width': size[0], 'height': size[1]},
                            'text': name,
                        },
                    ))
            for child in node:
                iter_tree(child)

    for window in desktop:
        if not is_visible(window):
            continue
        window_position = window.get('{uri:deskat:component.at-spi.gnome.org}screencoord', '')
        window_size = window.get('{uri:deskat:component.at-spi.gnome.org}size', '')
        window_position = [int(coord.strip('()')) for coord in window_position.split(',')]
        window_size = [int(dim.strip('()')) for dim in window_size.split(',')]

        # Remove all entities that this window overlaps
        entities = [(xy, e) for xy, e in entities if not rect_overlaps(xy, window_position, window_size)]

        # Add entities from the window
        iter_tree(window)

    entities = [ent for _, ent in entities]
    entities = filter_redundant_entities(entities, th=0.8)
    return entities
