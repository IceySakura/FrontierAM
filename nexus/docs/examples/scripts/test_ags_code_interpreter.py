"""End-to-end demo for the AGS CodeInterpreter provider.

模拟一个真实的代码执行任务：在沙盒里生成一份 CSV 数据、用 Python (pandas/标准库)
做聚合统计、把结果写回文件再读出来验证。 不接 LLM，纯脚本驱动，演示 run_command
（nexus interface）和原始 /execute（raw interface）两条路径在真实数据处理任务里怎么用。

Usage:
    # native 模式（E2B 模板）：
    uv run python docs/examples/scripts/test_ags_code_interpreter.py \
        --api-key <key> --template <code-interpreter-template> \
        [--domain ap-guangzhou.tencentags.com] \
        [--interface nexus|raw]

    # control-plane 模式（腾讯云控制面，复用已有内置 code-interpreter 工具）：
    uv run python docs/examples/scripts/test_ags_code_interpreter.py \
        --mode control-plane \
        --secret-id <SecretId> --secret-key <SecretKey> --tool-id <ToolId> \
        [--region ap-guangzhou] [--interface nexus|raw]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from pathlib import Path
from typing import Any

from nexus.utils.log import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_FILE = '/tmp/nexus_sales.csv'
RESULT_FILE = '/tmp/nexus_sales_summary.txt'

# 一段真实的数据处理脚本：造 CSV -> 按地区聚合销售额 -> 写结果文件
DATA_TASK = f"""
import csv, os

rows = [
    ('North', 'widget', 120),
    ('North', 'gadget', 80),
    ('South', 'widget', 200),
    ('South', 'gadget', 50),
    ('East', 'widget', 90),
]
with open({DATA_FILE!r}, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['region', 'product', 'amount'])
    w.writerows(rows)

# 按地区汇总
totals = {{}}
with open({DATA_FILE!r}) as f:
    for r in csv.DictReader(f):
        totals[r['region']] = totals.get(r['region'], 0) + int(r['amount'])

lines = [f'{{region}}: {{amount}}' for region, amount in sorted(totals.items())]
summary = 'Sales by region:\\n' + '\\n'.join(lines) + f'\\nTOTAL: {{sum(totals.values())}}'
with open({RESULT_FILE!r}, 'w') as f:
    f.write(summary)
print(summary)
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='AGS CodeInterpreter end-to-end demo')
    p.add_argument('--template', default=None, help='AGS code-interpreter template name')
    p.add_argument('--domain', default='ap-guangzhou.tencentags.com')
    p.add_argument('--region', default='ap-guangzhou', help='Tencent Cloud region (control-plane 模式)')
    p.add_argument('--interface', choices=['nexus', 'raw'], default='nexus')
    p.add_argument(
        '--mode',
        choices=['native', 'control-plane'],
        default='native',
        help='native=E2B 模板原生 CodeInterpreter; control-plane=腾讯云控制面内置 code-interpreter 工具',
    )
    p.add_argument('--api-key', default=None, help='AGS API key (native 模式必填)')
    p.add_argument('--secret-id', default=None, help='Tencent Cloud SecretId (control-plane 模式必填)')
    p.add_argument('--secret-key', default=None, help='Tencent Cloud SecretKey (control-plane 模式必填)')
    p.add_argument('--tool-id', default=None, help='已有内置 code-interpreter 工具的 ToolId (control-plane 模式必填)')
    return p.parse_args()


async def _test_nexus_native(runtime: Any) -> None:
    """nexus interface：用 run_command 跑 Python 数据处理脚本，再读回结果文件。"""
    # 1. 执行数据处理脚本（python3 -c 接收整段代码）
    import shlex

    res = await runtime.call_executor(
        'run_command',
        {
            'command': f'python3 -c {shlex.quote(DATA_TASK)}',
            'timeout': 60.0,
        },
    )
    print(f'  [1] data task status={res.get("status")} exit={res.get("return_code")}')
    print('      stdout:\n      ' + res.get('stdout', '').strip().replace('\n', '\n      '))
    assert res.get('return_code') == 0, f'data task failed: {res}'

    # 2. 用 shell 独立读回结果文件，验证落盘成功
    read = await runtime.call_executor(
        'run_command',
        {
            'command': f'cat {RESULT_FILE}',
            'timeout': 15.0,
        },
    )
    print('  [2] result file content:\n      ' + read.get('stdout', '').strip().replace('\n', '\n      '))

    # 3. 清理
    await runtime.call_executor(
        'run_command',
        {
            'command': f'rm -f {DATA_FILE} {RESULT_FILE}',
            'timeout': 15.0,
        },
    )
    print('  [done] cleaned up data files')


async def _test_raw_native(runtime: Any) -> None:
    """raw interface：直接把 Python 代码发给 /execute，无需 run_command 包装。"""
    # 1. 直接执行整段数据处理代码
    result = await runtime.execute(DATA_TASK, language='python')
    print(f'  [1] execute result: {result}')

    # 2. 再发一段代码读回结果文件，演示多次 /execute 共享文件系统
    read_code = f'print(open({RESULT_FILE!r}).read())'
    result2 = await runtime.execute(read_code, language='python')
    print(f'  [2] read-back via execute: {result2}')

    # 3. 用 execute_raw 跑清理（演示原始 payload 形态）
    cleanup = await runtime.execute_raw(
        {
            'code': f'import os; [os.remove(p) for p in ({DATA_FILE!r}, {RESULT_FILE!r}) if os.path.exists(p)]',
            'language': 'python',
        }
    )
    print(f'  [done] cleanup: {cleanup}')


async def main() -> int:
    setup_logging()
    args = _parse_args()

    from nexus.runtime.providers.ags import AGSRuntimeProvider

    if args.mode == 'control-plane':
        if not (args.secret_id and args.secret_key):
            print('ERROR: control-plane 模式需要 --secret-id 和 --secret-key', file=sys.stderr)
            return 1
        if not args.tool_id:
            print('ERROR: control-plane 模式需要 --tool-id（已有内置 code-interpreter 工具的 ToolId）', file=sys.stderr)
            return 1
        provider = AGSRuntimeProvider(
            secret_id=args.secret_id,
            secret_key=args.secret_key,
            region=args.region,
            domain=args.domain,
            interface_mode=args.interface,
            request_timeout=300.0,
            instance_ready_timeout=300.0,
        )
        # control_plane：用 tool_id 直接复用已有内置 code-interpreter 工具，provider 不按名查找、不新建、不删除。
        # launch_mode=native + tool_type=code-interpreter 走 AGSCodeInterpreterRuntime。
        acquire_params: dict[str, Any] = {
            'launch_mode': 'native',
            'runtime_kind': 'code',
            'tool_type': 'code-interpreter',
            'tool_id': args.tool_id,
        }
    else:
        api_key = args.api_key
        if not api_key:
            print('ERROR: native 模式需要 --api-key', file=sys.stderr)
            return 1
        if not args.template:
            print('ERROR: --template required for native mode', file=sys.stderr)
            return 1
        provider = AGSRuntimeProvider(
            api_key=api_key,
            domain=args.domain,
            interface_mode=args.interface,
            request_timeout=300.0,
            instance_ready_timeout=300.0,
        )
        acquire_params = {
            # launch_mode=native 显式声明走原生 code-interpreter（/execute on 49999）。
            # template 模式下 provider 也会自动推断成 native，这里写出来更直观。
            'launch_mode': 'native',
            'runtime_kind': 'code',
            'template': args.template,
        }

    print(f'[ags-code-interpreter] mode={args.mode} interface={args.interface}')
    runtime = None
    try:
        runtime = await provider.acquire('main', **acquire_params)
        print('[ags-code-interpreter] runtime acquired')

        if args.interface == 'nexus':
            await _test_nexus_native(runtime)
        else:
            await _test_raw_native(runtime)

        print('[ags-code-interpreter] ALL PASSED')
        return 0
    except Exception as e:
        print(f'[ags-code-interpreter] FAILED: {e}', file=sys.stderr)
        return 1
    finally:
        await provider.release_all()


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
