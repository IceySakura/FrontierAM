"""OSWorld upstream alignment regressions."""

from __future__ import annotations

import contextlib

from types import SimpleNamespace

import openpyxl
import pytest

from nexus.extensions.tasks.os_world.evaluation.errors import EvaluatorInfrastructureError
from nexus.extensions.tasks.os_world.evaluation.evaluator import evaluate_task
from nexus.extensions.tasks.os_world.execution import setup
from nexus.extensions.tasks.os_world.getters import chrome
from nexus.extensions.tasks.os_world.metrics.official.table import compare_table


def awaitable(fn):
    async def call(*args, **kwargs):
        return fn(*args, **kwargs)

    return call


def write_book(path, values):
    book = openpyxl.Workbook()
    book.active.title = 'Sheet1'
    book.active.append(values)
    book.save(path)
    return str(path)


@pytest.mark.parametrize(
    ('kind', 'actual', 'expected', 'extra', 'score'),
    [
        ('phone_match', '+1 (212) 555-0100', '2125550100', {}, 1),
        ('phone_match', '2125550100.0', '2125550100', {}, 1),
        ('phone_match', '9995550100', '2125550100', {}, 0),
        ('phone_match', None, '2125550100', {}, 0),
        ('phone_match', '2125550100', None, {'allow_missing_expected': True}, 1),
        ('phone_match', '2125550100', None, {}, 0),
        ('domain_match', 'HTTPS://WWW.Example.COM:443/path', 'example.com', {}, 1),
        ('domain_match', 'booking.example.com', 'example.com', {}, 1),
        ('domain_match', 'wrong.org', 'example.com', {}, 0),
        ('domain_match', None, 'example.com', {}, 0),
    ],
)
def test_sheet_fuzzy_rules_do_not_reuse_previous_result(tmp_path, kind, actual, expected, extra, score):
    result = write_book(tmp_path / 'result.xlsx', ['same', actual])
    gold = write_book(tmp_path / 'gold.xlsx', ['same', expected])
    rules = [
        {
            'type': 'sheet_fuzzy',
            'sheet_idx0': 'RNSheet1',
            'sheet_idx1': 'ENSheet1',
            'rules': [
                {'type': 'exact_match', 'range': ['A1']},
                {'type': kind, 'range': ['B1'], **extra},
            ],
        }
    ]
    assert compare_table(result, gold, rules=rules) == score


def test_phone_rule_can_be_first_rule(tmp_path):
    result = write_book(tmp_path / 'result.xlsx', ['+1 (212) 555-0100'])
    gold = write_book(tmp_path / 'gold.xlsx', ['2125550100'])
    assert (
        compare_table(
            result,
            gold,
            rules=[
                {
                    'type': 'sheet_fuzzy',
                    'sheet_idx0': 'RNSheet1',
                    'sheet_idx1': 'ENSheet1',
                    'rules': [{'type': 'phone_match', 'range': ['A1']}],
                }
            ],
        )
        == 1
    )


def test_unknown_sheet_rule_does_not_inherit_pass(tmp_path):
    result = write_book(tmp_path / 'result.xlsx', ['same'])
    with pytest.raises(NotImplementedError, match='Unsupported sheet_fuzzy'):
        compare_table(
            result,
            result,
            rules=[
                {
                    'type': 'sheet_fuzzy',
                    'sheet_idx0': 'RNSheet1',
                    'sheet_idx1': 'ENSheet1',
                    'rules': [{'type': 'exact_match', 'range': ['A1']}, {'type': 'unknown', 'range': ['A1']}],
                }
            ],
        )


@pytest.mark.asyncio
async def test_recreation_failure_propagates_through_evaluator(monkeypatch, tmp_path):
    counts = {'loads': 0, 'launches': 0}

    async def fail(*args):
        counts['loads'] += 1
        raise TimeoutError('simulated navigation timeout')

    async def launch(runtime):
        counts['launches'] += 1

    monkeypatch.setattr(chrome, '_load_recreation_expected_content', fail)
    monkeypatch.setattr(chrome, '_launch_chrome_for_cdp', launch)
    config = {
        'func': 'check_direct_json_object',
        'result': {'type': 'rule', 'rules': {}},
        'expected': {'type': 'gotoRecreationPage_and_get_html_content', 'selector': 'class', 'class': 'header'},
    }
    with pytest.raises(EvaluatorInfrastructureError):
        await evaluate_task(object(), config, None, cache_dir=tmp_path)
    assert counts == {'loads': 2, 'launches': 1}


@pytest.mark.asyncio
@pytest.mark.parametrize('texts', [[], [''], ['   ']])
async def test_recreation_missing_or_empty_expected_is_not_zero(texts):
    page = SimpleNamespace(
        query_selector_all=awaitable(
            lambda selector: [SimpleNamespace(text_content=awaitable(lambda text=text: text)) for text in texts]
        )
    )
    with pytest.raises(EvaluatorInfrastructureError):
        await chrome._recreation_expected_from_page(page, {'selector': 'class', 'class': 'header', 'order': 0})


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_recreation_cleans_its_context_before_disconnect(monkeypatch, failure):
    events = []
    page = SimpleNamespace(set_default_timeout=lambda x: None, set_extra_http_headers=awaitable(lambda x: None))
    context = SimpleNamespace(new_page=awaitable(lambda: page), close=awaitable(lambda: events.append('context')))
    browser = SimpleNamespace(new_context=awaitable(lambda: context))

    @contextlib.asynccontextmanager
    async def connect(runtime):
        try:
            yield browser
        finally:
            events.append('disconnect')

    monkeypatch.setattr(chrome, 'connect_chromium', connect)

    async def navigate(page):
        if failure:
            raise TimeoutError('navigation')
        return page

    monkeypatch.setattr(chrome, '_goto_recreation_diamond_page', navigate)
    monkeypatch.setattr(chrome, '_click_next_available_and_wait_for_header', awaitable(lambda *a, **kw: 'Sep 30'))
    monkeypatch.setattr(
        chrome, '_recreation_expected_from_page', awaitable(lambda *a: {'expected': {'header': 'Sep 30'}})
    )
    if failure:
        with pytest.raises(TimeoutError):
            await chrome._load_recreation_expected_content(object(), {'class': 'header'})
    else:
        assert (await chrome._load_recreation_expected_content(object(), {'class': 'header'}))['expected']
    assert events == ['context', 'disconnect']


class Page:
    def __init__(self, url, context):
        self.url = url
        self.context = context
        self.closed = False
        self.on_wait = lambda: None

    def is_closed(self):
        return self.closed

    async def wait_for_timeout(self, value):
        self.on_wait()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['same', 'popup', 'blank_popup', 'stale_only'])
async def test_recreation_navigation_clicks_once_and_ignores_stale_tabs(mode):
    url = 'https://www.recreation.gov/camping/campgrounds/233869'
    context = SimpleNamespace(pages=[])
    source = Page('https://www.recreation.gov/search', context)
    stale = Page(url, context)
    context.pages = [source, stale]
    clicks = []

    def click(**kwargs):
        clicks.append(1)
        if mode == 'same':
            source.url = url
        elif mode != 'stale_only':
            popup = Page('about:blank' if mode == 'blank_popup' else url, context)
            context.pages.append(popup)
            source.on_wait = lambda: setattr(popup, 'url', url)

    target = SimpleNamespace(click=awaitable(click))
    if mode == 'stale_only':
        with pytest.raises(TimeoutError):
            await chrome._click_and_resolve_navigation_page(source, target, timeout_ms=2, poll_interval_ms=1)
    else:
        page = await chrome._click_and_resolve_navigation_page(source, target, timeout_ms=2, poll_interval_ms=1)
        assert page is not stale
        assert page.url == url
    assert len(clicks) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [True, False])
async def test_next_available_requires_changed_header(changes):
    value = ['Sep 15']
    header = SimpleNamespace(inner_text=awaitable(lambda **kw: value[0]))
    clicks = []
    button = SimpleNamespace(wait_for=awaitable(lambda **kw: None), click=awaitable(lambda **kw: clicks.append(1)))

    def role(role, **kwargs):
        assert role == 'button'
        assert kwargs == {'name': 'Next Available', 'exact': True}
        return button

    page = SimpleNamespace(
        locator=lambda x: SimpleNamespace(nth=lambda x: header),
        get_by_role=role,
        wait_for_timeout=awaitable(lambda x: value.__setitem__(0, 'Sep 30' if changes else 'Sep 15')),
    )
    if changes:
        assert await chrome._click_next_available_and_wait_for_header(page, timeout_ms=2, poll_interval_ms=1) == 'Sep30'
    else:
        with pytest.raises(TimeoutError):
            await chrome._click_next_available_and_wait_for_header(page, timeout_ms=2, poll_interval_ms=1)
    assert len(clicks) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(('wait', 'success'), [(True, True), (False, True), (True, False)])
async def test_launch_wait_for_cdp_is_opt_in_and_checks_launch(monkeypatch, wait, success):
    called = []

    async def command(*args, **kwargs):
        return {'success': success}

    async def ready(runtime, operation):
        called.append('ready')
        await operation(SimpleNamespace(contexts=[]))

    async def screen_size(runtime):
        return 1920, 1080

    monkeypatch.setattr(setup, '_screen_size', screen_size)
    monkeypatch.setattr(setup, '_run_command', command)
    monkeypatch.setattr(setup, '_run_chrome_cdp_setup', ready)
    params = {'command': ['google-chrome', '--remote-debugging-port=1337'], 'wait_for_cdp': wait}
    if not success:
        with pytest.raises(RuntimeError, match='launch failed'):
            await setup.run_setup_action(object(), {'type': 'launch', 'parameters': params})
        assert not called
    else:
        await setup.run_setup_action(object(), {'type': 'launch', 'parameters': params})
        assert called == (['ready'] if wait else [])
