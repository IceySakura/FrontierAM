from __future__ import annotations

import json
import sqlite3

import pytest

from nexus.core import ToolContext
from nexus.extensions.tasks.aidata.engine.base_tools import AskUserInput, ask_user_tool
from nexus.extensions.tasks.aidata.engine.rubric import _gather_source, run_rubric_async


@pytest.mark.asyncio
async def test_text_contains_accepts_pattern_alias() -> None:
    state = {'agent_answer': '已推荐 2 款精华液，并说明价格与卷翘效果。'}
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'answer_mentions_price',
                'description': 'pattern alias works for text_contains',
                'dimension': 'functional',
                'weight': 1,
                'assert': {
                    'type': 'text_contains',
                    'from': 'agent_answer',
                    'pattern': r'(?s)价格.*卷翘',
                },
            }
        ],
    }

    [row] = await run_rubric_async(state, rubric)

    assert row['passed'] is True
    assert row['verdict'] == 'Y'
    assert 'pattern=match' in row['detail']


@pytest.mark.asyncio
async def test_text_contains_accepts_source_alias() -> None:
    state = {'agent_answer': '任务完成，输出文件位于 /workspace/desktop/output/report.md'}
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'source_alias',
                'description': 'source alias works for text_contains',
                'dimension': 'functional',
                'weight': 1,
                'assert': {
                    'type': 'text_contains',
                    'source': 'agent_answer',
                    'contains_all': ['任务完成', 'report.md'],
                },
            }
        ],
    }

    [row] = await run_rubric_async(state, rubric)

    assert row['passed'] is True
    assert row['verdict'] == 'Y'


@pytest.mark.asyncio
async def test_state_equals_sql_uses_declared_db_path(tmp_path) -> None:
    db_path = tmp_path / 'rollout.db'
    with sqlite3.connect(db_path) as conn:
        conn.execute('CREATE TABLE orders(order_id TEXT)')
        conn.execute("INSERT INTO orders(order_id) VALUES ('TV2-001')")

    state = {'_rubric_sqlite_path': str(db_path)}
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'sql_probe',
                'description': 'SQL probe reads the declared rubric sqlite path',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'state_equals',
                    'from': "sql:SELECT COUNT(*) FROM orders WHERE order_id LIKE 'TV2-%'",
                    'equals': 1,
                },
            }
        ],
    }

    [row] = await run_rubric_async(state, rubric)

    assert row['passed'] is True
    assert row['verdict'] == 'Y'


def test_all_source_omits_internal_rubric_routing_keys() -> None:
    text = _gather_source(
        {
            'emails': [{'to': ['hr@example.com']}],
            '_rubric_sqlite_path': '/tmp/private.db',
            '_rubric_sqlite_service': 'service_lifegym_mocks',
            'trajectory': [],
        },
        'all',
        None,
    )

    assert 'hr@example.com' in text
    assert '/tmp/private.db' not in text
    assert 'service_lifegym_mocks' not in text


def test_docs_selector_collects_document_slices_generically() -> None:
    text = _gather_source(
        {
            'custom_docs_mcp': {
                'docs': [
                    {'title': 'analysis.xlsx', 'type': 'excel', 'content': 'revenue table'},
                    {'title': 'summary.md', 'type': 'markdown', 'content': 'summary'},
                ],
            },
            'documents': [
                {'filename': 'report.docx', 'content': 'word report'},
            ],
        },
        'docs:excel',
        None,
    )

    assert 'analysis.xlsx' in text
    assert 'revenue table' in text
    assert 'summary.md' not in text
    assert 'report.docx' not in text


def test_dotted_source_selector_reads_namespaced_state() -> None:
    state = {
        'taobao': {
            'orders': [
                {'order_id': 'TV2-001', 'status': 'paid'},
            ],
        },
        'custom_docs_mcp': {
            'docs': [
                {'title': 'analysis.xlsx', 'type': 'excel', 'content': 'revenue table'},
                {'title': 'summary.md', 'type': 'markdown', 'content': 'summary'},
            ],
        },
    }

    orders = _gather_source(state, 'taobao.orders', None)
    excel_docs = _gather_source(state, 'custom_docs_mcp.docs', {'type': 'excel'})

    assert 'TV2-001' in orders
    assert 'analysis.xlsx' in excel_docs
    assert 'summary.md' not in excel_docs


@pytest.mark.asyncio
async def test_text_contains_generated_aliases() -> None:
    state = {
        'agent_answer': '订单已送至北京市海淀区中关村。',
        'trajectory': [{'role': 'assistant', 'content': '搜索了贴膏商品'}],
    }
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'text_alias',
                'description': 'text alias works',
                'dimension': 'functional',
                'weight': 1,
                'assert': {
                    'type': 'text_contains',
                    'from': 'agent_answer',
                    'text': '北京市海淀区中关村',
                },
            },
            {
                'id': 'keywords_alias',
                'description': 'keywords alias works',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'text_contains',
                    'from': 'trajectory',
                    'keywords': ['贴膏', '凝胶'],
                },
            },
        ],
    }

    rows = await run_rubric_async(state, rubric)

    assert [row['passed'] for row in rows] == [True, True]


@pytest.mark.asyncio
async def test_text_not_contains_and_trajectory_source_aliases() -> None:
    state = {
        'agent_answer': '触发止盈提醒，未输出心跳。',
        'trajectory': [{'role': 'assistant', 'content': '查询足球 L 码商品'}],
    }
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'negative_text',
                'description': 'text_not_contains works',
                'dimension': 'robustness',
                'weight': 1,
                'assert': {
                    'type': 'text_not_contains',
                    'from': 'agent_answer',
                    'contains': 'HEARTBEAT_OK',
                },
            },
            {
                'id': 'agent_trajectory_alias',
                'description': 'agent_trajectory maps to trajectory',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'text_contains',
                    'from': 'agent_trajectory',
                    'contains_all': ['足球', 'L 码'],
                },
            },
        ],
    }

    rows = await run_rubric_async(state, rubric)

    assert [row['passed'] for row in rows] == [True, True]


@pytest.mark.asyncio
async def test_text_contains_resolves_to_namespace_not_flatten() -> None:
    """``from: <mock_name>`` 必须解析到该 mock 的 namespaced bucket。

    这是针对旧的 ``'notes': 'memos'`` source alias 的回归测试， 该 alias
    会强制把 ``from: notes`` 降级到 flatten 层。 自从 state.py 开始
    同时发布 namespaced bucket 与扁平 slice 之后， 那个 alias 就变错了：
    如果一个 rubric 作者按 README 推荐的 mock-namespace 风格写
    ``from: notes`` ， 现在应当直接读取完整的 dump_state payload， 而不是
    遗留的扁平 slice。

    关键点在于： 收集到的文本头部必须引用 mock 名字 （ ``## notes`` ） ，
    而不是扁平 key （ ``## memos`` ） ， 这样 judge / regex 输出仍归属于
    该 mock， 而不是被归属到 flatten 层。
    """
    state = {
        # state.py 会为每个 mock 同时写入两处 —— namespaced bucket 是规范
        # 路径； 扁平 slice 只是留给旧 rubric 的兼容层。
        'notes': {
            'memos': {
                'memo_001': {
                    'memo_id': 'memo_001',
                    'entries': [
                        {'entry_id': 'memo_001_e001', 'data': {'file': 'auth.py', 'score': 4}},
                        {'entry_id': 'memo_001_e003', 'data': {'file': 'parser.py', 'score': 3}},
                    ],
                },
            },
            'seq': 1,
        },
        'memos': {  # 遗留 flatten —— 当 from=notes 时绝不应命中此处
            'memo_001': {'memo_id': 'memo_001', 'entries': []},
        },
        'seq': 1,
        'agent_answer': '',
        'trajectory': [],
    }
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'finds_entry_in_namespace',
                'description': 'rubric should see e003 via namespaced bucket',
                'dimension': 'functional',
                'weight': 1,
                'assert': {
                    'type': 'text_contains',
                    'from': 'notes',
                    'contains': 'memo_001_e003',
                },
            },
        ],
    }
    rows = await run_rubric_async(state, rubric)
    # 断言为 True 表示 rubric 命中的是 namespaced bucket （其中含有该
    # entry） ， 而不是 flatten 层 （在这里为空） 。
    assert rows[0]['passed'] is True, rows


@pytest.mark.asyncio
async def test_tool_called_scope_splits_seed_and_live_trajectory() -> None:
    state = {
        '_initial_trajectory_len': 2,
        'trajectory': [
            {'role': 'system', 'content': 'system'},
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'function': {'name': 'wx_agent_hub_list_installed', 'arguments': '{}'},
                    }
                ],
            },
            {'role': 'tool', 'name': 'wx_agent_hub_list_installed', 'content': '{"installed":[]}'},
            {'role': 'user', 'content': 'live task'},
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'function': {'name': 'wx_agent_hub_install_skill', 'arguments': '{"skill":"ride_app"}'},
                    }
                ],
            },
        ],
    }
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'seed_only',
                'description': 'seed scope sees seeded tool call',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {'type': 'tool_called', 'tool': 'list_installed', 'scope': 'seed'},
            },
            {
                'id': 'live_only',
                'description': 'live scope excludes seeded tool call',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'tool_called',
                    'tool': 'install_skill',
                    'scope': 'live',
                    'with_text': 'ride_app',
                },
            },
            {
                'id': 'seed_excluded_from_live',
                'description': 'live scope does not see seeded call',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {'type': 'tool_called', 'tool': 'list_installed', 'scope': 'live'},
            },
        ],
    }

    rows = await run_rubric_async(state, rubric)

    assert [row['passed'] for row in rows] == [True, True, False]


@pytest.mark.asyncio
async def test_tool_called_with_text_matches_ascii_escaped_cjk_args() -> None:
    # 工具调用参数以 ASCII 转义的 JSON 存储 （ ``ensure_ascii`` ） ， 因此
    # 直接给出中文 ``with_text`` 永远不会是原始 blob 的子串
    # （ blob 里是 ``\uXXXX`` ） 。 匹配器还必须查看解码后的 args。
    escaped_args = json.dumps({'keyword': '洗衣液 2L 4瓶', 'limit': 15})
    assert '\\u6d17' in escaped_args  # 保护性检查： args 确实是被转义了
    state = {
        'trajectory': [
            {'role': 'system', 'content': 'system'},
            {
                'role': 'assistant',
                'tool_calls': [
                    {'function': {'name': 'pdd_search_products', 'arguments': escaped_args}},
                ],
            },
        ],
    }
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'pdd_searched_cjk',
                'description': '拼多多搜索了洗衣液',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'tool_called',
                    'tool': 'pdd_search_products',
                    'with_text': '洗衣',
                },
            },
            {
                'id': 'pdd_searched_absent',
                'description': '未搜索的关键词不应误判命中',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'tool_called',
                    'tool': 'pdd_search_products',
                    'with_text': '纸巾',
                },
            },
        ],
    }

    rows = await run_rubric_async(state, rubric)

    assert [row['passed'] for row in rows] == [True, False]


@pytest.mark.asyncio
async def test_tool_called_with_text_equals_is_enforced() -> None:
    state = {
        'trajectory': [
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'function': {
                            'name': 'fliggy_hotel_book',
                            'arguments': json.dumps(
                                {
                                    'hotel_id': 'AJ_COUNTY_CONF_999',
                                    'room_type': '百人团建会议整包',
                                    'check_in': '2027-05-01',
                                    'check_out': '2027-05-02',
                                    'guest_name': '陈然',
                                },
                                ensure_ascii=False,
                            ),
                        },
                    },
                ],
            },
        ],
    }
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'hotel_id_exact_match_fails',
                'description': 'hotel_id must be exact, not silently ignored',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'tool_called',
                    'tool': 'fliggy_hotel_book',
                    'with_text': {'field': 'arguments.hotel_id', 'equals': 'AJ_COUNTY_CONF_168'},
                },
            },
            {
                'id': 'check_in_exact_match_passes',
                'description': 'equals passes when the selected field exactly matches',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'tool_called',
                    'tool': 'fliggy_hotel_book',
                    'with_text': {'field': 'arguments.check_in', 'equals': '2027-05-01'},
                },
            },
        ],
    }

    rows = await run_rubric_async(state, rubric)

    assert [row['passed'] for row in rows] == [False, True]


@pytest.mark.asyncio
async def test_tool_called_max_count_is_enforced() -> None:
    state = {
        'trajectory': [
            {
                'role': 'assistant',
                'tool_calls': [
                    {'function': {'name': 'email_send', 'arguments': '{"to":"a@example.com"}'}},
                    {'function': {'name': 'email_send', 'arguments': '{"to":"b@example.com"}'}},
                ],
            },
        ],
    }
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'at_most_one_fails',
                'description': 'two calls exceed max_count=1',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {'type': 'tool_called', 'tool': 'email_send', 'max_count': 1},
            },
            {
                'id': 'bounded_range_passes',
                'description': 'two calls satisfy 1..2',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {'type': 'tool_called', 'tool': 'email_send', 'min_count': 1, 'max_count': 2},
            },
            {
                'id': 'max_zero_without_min_means_absent',
                'description': 'max_count=0 alone is a valid absence check',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {'type': 'tool_called', 'tool': 'sms_send', 'max_count': 0},
            },
        ],
    }

    rows = await run_rubric_async(state, rubric)

    assert [row['passed'] for row in rows] == [False, True, True]
    assert 'max=1' in rows[0]['detail']


@pytest.mark.asyncio
async def test_tool_called_acceptable_tools_are_or_aliases() -> None:
    state = {
        'trajectory': [
            {
                'role': 'assistant',
                'tool_calls': [
                    {'function': {'name': 'wecom_im_history', 'arguments': '{}'}},
                    {'function': {'name': 'wecom_im_search', 'arguments': '{}'}},
                ],
            },
        ],
    }
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'alternate_tool_matches',
                'description': 'acceptable_tools is an OR-list with tool',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'tool_called',
                    'tool': 'wecom_im_message_get',
                    'acceptable_tools': ['wecom_im_history'],
                },
            },
            {
                'id': 'slash_alias_matches',
                'description': 'slash namespace aliases match underscore tool names',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'tool_called',
                    'acceptable_tools': ['wecom/im_search'],
                },
            },
            {
                'id': 'dot_alias_matches_sequence_step',
                'description': 'tool_called_sequence reuses acceptable_tools matching',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'tool_called_sequence',
                    'steps': [
                        {'acceptable_tools': ['wecom.im_history']},
                        {'acceptable_tools': ['im_search']},
                    ],
                },
            },
            {
                'id': 'unmatched_alias_fails',
                'description': 'acceptable_tools must still match at least one call',
                'dimension': 'correctness',
                'weight': 1,
                'assert': {
                    'type': 'tool_called',
                    'acceptable_tools': ['wecom_im_thread_list'],
                },
            },
        ],
    }

    rows = await run_rubric_async(state, rubric)

    assert [row['passed'] for row in rows] == [True, True, True, False]


@pytest.mark.asyncio
async def test_ask_user_tool_uses_case_script(tmp_path) -> None:
    script = tmp_path / 'ask_user.py'
    script.write_text(
        "def ask_user(ask_user_msg, trajectory=None):\n    return 'answered: ' + ask_user_msg\n",
        encoding='utf-8',
    )

    response = await ask_user_tool(
        AskUserInput(question='车型？'),
        ToolContext(tool_configs={'base/ask_user': {'script_path': str(script)}}),
    )

    assert response.output == [{'type': 'text', 'text': 'answered: 车型？'}]
