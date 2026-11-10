import math

import arena.tasks  # noqa: F401  (operator_tasks is imported through arena.tasks)

from arena import feedback
from arena.operator_tasks import score_100
from arena.public_v63 import galaxy_score, mapped, raw_error


def row(case, category, status='passed', accuracy=None, seconds=1.0, **extra):
    return {'id': case, 'category': category, 'status': status, 'accuracy': accuracy, 'seconds': seconds, **extra}


def test_status_mapping():
    assert feedback.status_of('ok') == 'passed'
    assert feedback.status_of('ok', correct=False) == 'wrong'
    assert feedback.status_of('cpu_timeout') == feedback.status_of('wall_timeout') == 'timeout'
    assert feedback.status_of('nonzero_exit') == feedback.status_of('oom') == feedback.status_of(None) == 'crash'
    assert feedback.status_of('invalid_output') == 'wrong'


def test_report_has_categories_ratios_and_no_case_ids():
    mine = [row('a', 'x', accuracy=2.0, seconds=2.0), row('b', 'x', 'timeout'), row('c', 'y', accuracy=1.0, seconds=1.0)]
    ref = [row('a', 'x', accuracy=1.0, seconds=1.0), row('b', 'x', accuracy=1.0, seconds=1.0), row('c', 'y', accuracy=1.0, seconds=4.0)]
    report = feedback.build('t', mine, ref, score=42.0, categorized=True, accuracy_name='error', timed=True,
                            category_score=lambda m, r: 10.0 * len(m))
    x, y = report['categories']['x'], report['categories']['y']
    assert (x['cases'], x['passed'], x['timeout'], x['score']) == (2, 1, 1, 20.0)
    assert math.isclose(y['speed_ratio'], 0.25) and math.isclose(y['accuracy_ratio'], 1.0)
    assert report['total']['cases'] == 3 and report['score'] == 42.0
    text = feedback.render(report)
    assert '42.0' in text and '| x |' in text and '| y |' in text
    assert all(f'| {case} |' not in text for case in 'abc')


def test_uncategorized_report_has_total_only():
    mine = [row('a', 'all', accuracy=0.2), row('b', 'all', accuracy=0.4)]
    report = feedback.build('t', mine, mine, score=50.0, categorized=False, accuracy_name='RMSE', timed=False)
    assert report['categories'] == {}
    text = feedback.render(report)
    assert '**total**' in text and 'time:' not in text


def test_invalid_and_unmeasured_render():
    assert 'Invalid submission' in feedback.render(feedback.invalid('t', 'no solve'))
    assert 'Not measured' in feedback.render(feedback.unmeasured('t', 'slot lost'))


def test_galaxy_log_mapping_anchors():
    assert galaxy_score(1e-3, 1e-3, 1e-7) == 0.0
    assert math.isclose(galaxy_score(1e-7, 1e-3, 1e-7), 50.0)
    assert math.isclose(galaxy_score(1e-5, 1e-3, 1e-7), 25.0)
    assert math.isclose(galaxy_score(1e-15, 1e-3, 1e-7), 100.0)
    assert math.isclose(galaxy_score(1e-11, 1e-3, 1e-7), 75.0)


def test_raw_error_aggregates_like_the_tasks():
    galaxy = [row('1', 'disc', accuracy=1.0), row('2', 'disc', accuracy=3.0), row('3', 'jaffe', accuracy=6.0)]
    assert raw_error('galaxy-gravity-v2', galaxy) == 4.0
    assert raw_error('galaxy-gravity-v2', galaxy + [row('4', 'disc', 'timeout')]) is None
    traj = [row('1', 'ar', accuracy=1.0, length=10), row('2', 'ar', accuracy=3.0, length=10),
            row('3', 'ar', accuracy=5.0, length=20)]
    assert raw_error('trajectory-covariance', traj) == 3.5
    assert raw_error('species-tree-v2', [row('1', 'all', accuracy=0.1), row('2', 'all', 'crash', accuracy=1.0)]) == 0.55


def test_mapped_uses_weak_and_strong_references():
    mine = [row('1', 'all', accuracy=0.15)]
    weak, strong = [row('1', 'all', accuracy=0.2)], [row('1', 'all', accuracy=0.1)]
    assert math.isclose(mapped('species-tree-v2', mine, weak, strong), score_100(0.15, 0.2, 0.1))
    assert mapped('polarization-demosaicking', [row('1', 'all', 'timeout')], weak, strong) == 0.0
