
from __future__ import annotations

import asyncio
import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nexus.extensions.tasks.ppt_rubric import (
    judge,  # noqa: E402
    ppt_metrics,  # noqa: E402
    scoring,  # noqa: E402
)
from nexus.extensions.tasks.ppt_rubric import task as ppt_task  # noqa: E402


DECLARED_NUMERIC = {key for key, value in judge.stable_columns().items() if isinstance(value, (int, float))}


def numeric_keys(columns: dict) -> set[str]:
    return {key for key, value in columns.items() if isinstance(value, (int, float))}


def assert_columns_subset(columns: dict, label: str) -> None:
    mapped = {judge.detail_column(key): value for key, value in columns.items()}
    extra = numeric_keys(mapped) - DECLARED_NUMERIC
    assert not extra, f'{label}: undeclared numeric columns {sorted(extra)}'


def _mock_post(response: dict):
    async def fake(url, payload, headers):
        return response

    return fake


def _judge_call(response: dict) -> tuple[float | None, dict]:
    judge.post_to_reward_platform = _mock_post(response)
    import os

    os.environ.setdefault('NEXUS_REWARD_SYSTEM_TOKEN', 'test-token')
    return asyncio.run(
        judge.subjective_score(
            prompt_str='reproduce this slide',
            response='',
            router_id='MultimodalReward-PPTJudgeScoring-0',
            ref_texts=['title'],
            ref_code='',
        )
    )


class TestColumnSetInvariant:
    def test_render_failure_path(self):
        output = ppt_task._failed_output('soffice produced no page', unrenderable=True)
        assert_columns_subset(output.extra_fields, 'render failure')
        columns = {**judge.stable_columns(), **output.extra_fields}
        assert numeric_keys(columns) == DECLARED_NUMERIC, 'render failure changes the declared set'

    def test_scoring_invoke_failure_path(self):
        output = ppt_task._failed_output('scoring invoke failed: Boom', unrenderable=False, evaluator_failed=True)
        assert_columns_subset(output.extra_fields, 'invoke failure')
        assert output.score == -100.0

    def test_paste_gate_host_path(self):
        diagnostics = {
            'paste_gate': 'no_gold_text_chars',
            'paste_code_chars': 0.0,
            'paste_pdf_chars': 0.0,
        }
        columns = judge.paste_gate_columns(diagnostics, 'from pptx import Presentation')
        assert_columns_subset(columns, 'paste gate host re-decision')

    def test_judge_three_dimensions_path(self):
        score, detail = _judge_call({
            'status': 0,
            'data': {
                'response_datas': [
                    {
                        'class': 'MultimodalRewardPPTJudgeScoring',
                        'error_code': 0,
                        'data': {
                            'final_score': 0.6,
                            'details': {
                                'subj_accuracy': 0.5,
                                'subj_quality': 0.75,
                                'subj_aesthetics': 0.5,
                            },
                        },
                    }
                ]
            },
        })
        assert score is not None
        assert_columns_subset(detail, 'judge three dimensions')

    def test_judge_bare_scalar_path(self):
        score, detail = _judge_call({
            'status': 0,
            'data': {
                'response_datas': [
                    {'class': 'MultimodalRewardPPTJudgeScoring', 'error_code': 0, 'data': {'reward_score': 0.25}}
                ]
            },
        })
        assert score is not None
        assert abs(score - 0.25) < 1e-9
        assert_columns_subset(detail, 'judge bare scalar')

    def test_judge_absent_path(self):
        score, detail = _judge_call({
            'status': 0,
            'data': {'response_datas': [{'class': 'MultimodalRewardPPTJudgeScoring', 'error_code': 500, 'data': {}}]},
        })
        assert score is None
        assert_columns_subset(detail, 'judge outage')

    def test_stable_columns_constructs_without_collision(self):
        cols = judge.stable_columns()
        assert len(cols) >= 65
        neutral = {'ppt_baked_gate'}
        for key, value in cols.items():
            if isinstance(value, (int, float)):
                if value == 1.0:
                    assert key in neutral, f'{key} has an undocumented 1.0 default'
                else:
                    assert value in (-1.0, 0.0), f'{key} has an undocumented numeric default {value!r}'

    def test_not_measured_and_judge_sentinel_are_distinct_names(self):
        assert judge.NOT_MEASURED == judge.JUDGE_NOT_JUDGEABLE == -1.0
        cols = judge.stable_columns()
        assert all(v != judge.JUDGE_NOT_JUDGEABLE or v == judge.NOT_MEASURED for v in cols.values())

    def test_mirror_filters_non_numeric(self):
        extra_fields = {**judge.stable_columns(), 'ppt_scoring_diagnostics': {'soffice': '7.5'}}
        judge.mirror_into_reward_extra_info(extra_fields)
        mirrored = extra_fields['reward_extra_info']
        assert 'ppt_scoring_diagnostics' not in mirrored
        assert all(isinstance(v, float) for v in mirrored.values())
        assert set(mirrored) == DECLARED_NUMERIC


def test_host_gate_redecision_declares_model_floor():
    assert 'ppt_model_floor' in judge.stable_columns()
    diagnostics = {'paste_gate': 'no_gold_text_chars', 'paste_code_chars': -1.0, 'paste_pdf_chars': 120.0}
    long_ref = 'title = "' + 'A' * 60 + '"'
    columns = judge.paste_gate_columns(diagnostics, long_ref)
    assert columns['ppt_text_gate'] == 'host_ok'
    assert 'ppt_model_floor' in columns


def test_v2_judge_accuracy_mode_is_coded_and_declared():
    payload = {
        'subj_accuracy': 0.75,
        'accuracy_mode': 'structure_listing',
        'accuracy_prompt_chars': 26760.0,
        'ref_listing_chars': 14155.0,
        'cand_listing_chars': 17081.0,
    }
    _dims, detail = judge._parse_judge_payload(payload)
    assert detail['judge_accuracy_mode'] == judge.ACCURACY_MODE_CODES['structure_listing']
    assert detail['judge_accuracy_prompt_chars'] == 26760.0
    declared = judge.stable_columns()
    for key in detail:
        if key.startswith('judge_'):
            assert judge.detail_column(key) in declared, f'{key} emitted but not declared'


def test_v2_judge_unknown_accuracy_mode_is_flagged_not_dropped():
    _dims, detail = judge._parse_judge_payload({'subj_accuracy': 0.5, 'accuracy_mode': 'renamed_calibre'})
    assert detail['judge_accuracy_mode'] == judge.UNKNOWN_ACCURACY_MODE


def test_gold_relative_deducts_only_for_quantities_worse_than_the_reference():
    gold = {
        'text_density': {'ink_ratio': 0.30},
        'imbalance': {'offset_norm': [0.02, 0.03]},
        'text_overflow': {'worst_ratio': 1.4},
        'whitespace': {'content_ratio': 0.50},
        'collision': {'collision_density': 1.2},
        'contrast': {'worst_normalised': 0.9},
    }
    scores, info = ppt_metrics.relative_objective_scores(gold, gold)
    assert info['n_used'] == 6
    assert all(abs(v - 1.0) < 1e-9 for v in scores.values())

    sparse = {
        'text_density': {'ink_ratio': 0.03},
        'imbalance': {'offset_norm': [0.02, 0.03]},
        'text_overflow': {'worst_ratio': 1.4},
        'whitespace': {'content_ratio': 0.05},
        'collision': {'collision_density': 0.0},
        'contrast': {'worst_normalised': 0.9},
    }
    thin, _ = ppt_metrics.relative_objective_scores(gold, sparse)
    assert thin['text_density'] == 1.0
    assert thin['collision'] == 1.0
    assert thin['whitespace'] < 0.5, thin['whitespace']

    worse = {
        'text_density': {'ink_ratio': 0.90},
        'imbalance': {'offset_norm': [0.30, 0.20]},
        'text_overflow': {'worst_ratio': 4.2},
        'whitespace': {'content_ratio': 0.50},
        'collision': {'collision_density': 3.6},
        'contrast': {'worst_normalised': 0.9},
    }
    degraded, _ = ppt_metrics.relative_objective_scores(gold, worse)
    assert degraded['text_density'] < 0.5, degraded['text_density']
    assert degraded['imbalance'] < 0.5, degraded['imbalance']
    assert degraded['text_overflow'] < 0.5, degraded['text_overflow']
    assert degraded['collision'] < 0.5, degraded['collision']


def test_gold_relative_omits_metrics_neither_side_measured():
    scores, info = ppt_metrics.relative_objective_scores({'collision': {'collision_density': 1.0}}, {'collision': {}})
    assert scores['collision'] is None
    assert info['n_used'] == 0


def test_element_iou_factors_are_declared():
    declared = judge.stable_columns()
    for key in scoring.ELEMENT_IOU_FACTORS:
        assert f'ppt_element_{key}' in declared, key
    assert tuple(judge._ELEMENT_IOU_FACTORS) == tuple(scoring.ELEMENT_IOU_FACTORS)
