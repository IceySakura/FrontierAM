"""``script_probe`` 原语——运行脚本并比对 stdout / exit_code。"""

import re

from typing import Any

from ..config import JunitLookupContext, RubricLimits, VerifierResult


async def _eval_script_probe(
    cfg: dict,
    state: dict,
    runtime: Any = None,
    junit_ctx: JunitLookupContext | None = None,
    *,
    limits: RubricLimits | None = None,
) -> VerifierResult:
    """assert.type: script_probe——运行脚本，比对 stdout / exit_code。

    Config：
      script:      str（必填）——内联 shell/python
      service:     str（默认 'main'）
      timeout_sec: int（默认 15）
      expect_exit: int（可选；默认 0）
      contains:    str（可选）——子串必须出现在 stdout 中
      regex:       str（可选）——正则必须匹配 stdout

    仅当 exit code 等于 expect_exit 且所有提供的文本谓词都匹配时，判定为 Y。
    若 runtime 缺失，返回 UNAVAILABLE。
    """
    from ..fetch import Fetcher, ProbeSpec

    script = cfg.get('script')
    if not script:
        return False, 'INVALID', 'script_probe missing "script"', ''

    fetcher = state.get('_fetcher') or Fetcher(state, runtime=runtime, limits=limits)
    stdout, exit_code = await fetcher.probe(
        ProbeSpec(
            script=script,
            service=cfg.get('service', 'main'),
            timeout_sec=int(cfg.get('timeout_sec', 15)),
        )
    )

    if exit_code == -1:
        return False, 'UNAVAILABLE', 'no runtime for script_probe', stdout[:120]
    if exit_code == -2:
        return False, 'TIMEOUT', f'script_probe timed out ({cfg.get("timeout_sec", 15)}s)', stdout[:120]
    if exit_code == -3:
        return False, 'ERROR', 'script_probe exception', stdout[:120]

    reasons = []
    expect_exit = cfg.get('expect_exit', 0)
    if int(exit_code) != int(expect_exit):
        reasons.append(f'exit_code={exit_code} != expected {expect_exit}')
    if 'contains' in cfg and cfg['contains'] not in stdout:
        reasons.append(f'stdout missing substring {cfg["contains"]!r}')
    if 'regex' in cfg and not re.search(cfg['regex'], stdout):
        reasons.append(f'stdout does not match /{cfg["regex"]}/')

    passed = not reasons
    detail = f'exit_code={exit_code}' + (' ; ' + ' ; '.join(reasons) if reasons else '')
    return passed, 'Y' if passed else 'N', detail, stdout[:200]
