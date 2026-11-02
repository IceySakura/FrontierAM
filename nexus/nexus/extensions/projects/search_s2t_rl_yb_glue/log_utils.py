"""统一的日志工具模块（对齐 agts 项目）。

提供带有统一格式的日志函数，方便 grep 和定位。
"""

from datetime import datetime


def log_trace(tag: str, traj_id: str, phase: str, detail: str = ''):
    """统一的追踪日志格式。

    输出格式: [时间][{TAG}][{traj_id}] {phase} | {detail}

    Args:
        tag: 日志标签，如 'TASK', 'REWARD', 'AGENT'
        traj_id: 轨迹/任务唯一标识
        phase: 阶段名称，如 'START', 'DONE'
        detail: 可选的详细信息
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    detail_str = f' | {detail}' if detail else ''
    print(f'[{timestamp}][{tag}][{traj_id}] {phase}{detail_str}')
