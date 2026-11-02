"""aidata 测试结果契约的最简 junit.xml 解析器。

aidata cases 通过一个固定协议来暴露其 agent 运行后的校验：

- `cases/<id>/run-test.sh` 会在 agent 完成后由 engine 调用一次。
  它唯一的职责就是在 `/workspace/tests/report.junit.xml`
  留下一个 junit-xml 文件。
- rubric 通过稳定的 check.id 来引用具体的 testcase 结果
  （见 engine/check_map.py ， 了解该 id 如何映射到 junit 的 testcase 名字）。

nexus 的其余部分只处理下面的规范化结构， 从不接触
原始 XML 。 我们刻意让解析器保持精简且宽容： 未知
属性会被忽略， `<testsuite>` / `<testsuites>` 的嵌套会被
展平， 结构不合法的 XML 会以 ``JunitParseError`` 抛出，
消息内容让 case 作者可据此排查。
"""

from __future__ import annotations

import logging

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


logger = logging.getLogger(__name__)


class JunitParseError(RuntimeError):
    """当 report.junit.xml 缺失或格式错误时抛出。"""


@dataclass
class TestCaseResult:
    """单个 junit testcase 的规范化记录。"""

    name: str
    classname: str = ''
    passed: bool = True
    duration_s: float = 0.0
    message: str = ''
    failure_type: str = ''

    @property
    def fqname(self) -> str:
        """命令行 pytest 使用的完全限定名： classname.name 。"""
        return f'{self.classname}.{self.name}' if self.classname else self.name


@dataclass
class JunitReport:
    """解析后的 junit 报告， 以 testcase 名字为 key 。"""

    cases: list[TestCaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.cases if not c.passed)

    def find(self, *candidate_names: str) -> TestCaseResult | None:
        """返回 name / fqname 匹配任一候选项的第一条 testcase 。

        候选项按顺序尝试。 对于 testsuite-classname 前缀不同的测试，
        仍可通过裸 ``name`` 匹配， 以适配常见的 pytest
        输出（例如 ``tests.test_outputs.test_foo`` 与查询
        ``test_foo`` ）。
        """
        by_name: dict[str, TestCaseResult] = {}
        for c in self.cases:
            by_name.setdefault(c.name, c)
            by_name.setdefault(c.fqname, c)
        for cand in candidate_names:
            if cand in by_name:
                return by_name[cand]
        return None

    def all_names(self) -> list[str]:
        """去重后的 testcase 名字扁平列表（用于 debug 日志）。"""
        seen: set[str] = set()
        out: list[str] = []
        for c in self.cases:
            if c.name not in seen:
                seen.add(c.name)
                out.append(c.name)
        return out


def parse_junit_xml(xml_text: str) -> JunitReport:
    """将 junit-xml 字符串解析为 :class:`JunitReport` 。

    接受两种常见的顶层根节点： ``<testsuites>`` 和单个
    ``<testsuite>`` 。 任何 ``<failure>`` / ``<error>`` 子节点都会将
    ``passed`` 翻转为 False ； ``<skipped>`` 保持 ``passed`` 为 True （对
    严格 rubric 不理想， 但符合 CI 惯例 —— skipped != failed ）。
    """
    if not xml_text or not xml_text.strip():
        raise JunitParseError('junit report is empty')
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise JunitParseError(f'junit report is not valid XML: {exc}') from exc

    suites: Iterable[ET.Element]
    if root.tag == 'testsuites':
        suites = root.findall('testsuite')
    elif root.tag == 'testsuite':
        suites = [root]
    else:
        # 部分框架会输出 <report> / <results> 包裹层； 扫描其子节点。
        suites = root.findall('.//testsuite')

    cases: list[TestCaseResult] = []
    for suite in suites:
        for tc in suite.findall('testcase'):
            cases.append(_parse_testcase(tc))
    return JunitReport(cases=cases)


def parse_junit_file(path: Path) -> JunitReport:
    if not path.exists():
        raise JunitParseError(f'junit report not found at {path}')
    try:
        return parse_junit_xml(path.read_text(encoding='utf-8', errors='replace'))
    except JunitParseError:
        raise
    except Exception as exc:
        raise JunitParseError(f'cannot read junit report {path}: {exc}') from exc


def _parse_testcase(tc: ET.Element) -> TestCaseResult:
    name = tc.attrib.get('name', '') or ''
    classname = tc.attrib.get('classname', '') or ''
    try:
        duration = float(tc.attrib.get('time', '0') or '0')
    except ValueError:
        duration = 0.0

    # failure / error 子节点会翻转判定。 有些生产者会同时嵌套两者，
    # 也没关系 —— 任一个都视作 failure 。
    failure = tc.find('failure')
    error = tc.find('error')
    if failure is not None or error is not None:
        source = failure if failure is not None else error
        ftype = source.attrib.get('type', '') if source is not None else ''
        # 优先使用子节点的 message 属性； 回退到其 text 主体。
        msg = ''
        if source is not None:
            msg = source.attrib.get('message', '') or (source.text or '').strip()
        return TestCaseResult(
            name=name,
            classname=classname,
            passed=False,
            duration_s=duration,
            message=msg[:1000],
            failure_type=ftype,
        )
    return TestCaseResult(
        name=name,
        classname=classname,
        passed=True,
        duration_s=duration,
    )


__all__ = [
    'JunitParseError',
    'JunitReport',
    'TestCaseResult',
    'parse_junit_file',
    'parse_junit_xml',
]
