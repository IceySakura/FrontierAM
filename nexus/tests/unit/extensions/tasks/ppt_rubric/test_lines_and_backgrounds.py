import pytest

from nexus.extensions.tasks.ppt_rubric.ppt_fidelity import FidelityConfig, metric_bounds_fidelity, metric_element_iou
from nexus.extensions.tasks.ppt_rubric.ppt_metrics import (
    PPTRubricConfig,
    SlideElement,
    SlideGeometry,
    compute_objective_metrics,
    metric_collision,
    metric_imbalance,
    relative_objective_scores,
)


LAYOUT = PPTRubricConfig()
FIDELITY = FidelityConfig()


def _block():
    return SlideElement(x=48, y=96, width=384, height=480, kind='shape')


def _title():
    return SlideElement(x=67, y=125, width=326, height=77, kind='text', text='Left heavy', text_len=10)


def _backdrop():
    return SlideElement(x=0, y=0, width=1280, height=720, kind='shape', is_decorative=True)


def _card():
    return SlideElement(x=96, y=96, width=576, height=480, kind='shape')


def _card_title():
    return SlideElement(x=125, y=115, width=480, height=58, kind='text', text='Title', text_len=5)


def _rule():
    return SlideElement(x=125, y=287, width=499, height=1.0, kind='line')


def _geometry(*elements):
    return SlideGeometry(elements=list(elements))


def test_a_backdrop_counts_in_imbalance_and_pulls_the_centroid_to_centre():
    without, detail_without = metric_imbalance(_geometry(_block(), _title()), LAYOUT)
    with_backdrop, detail_with = metric_imbalance(_geometry(_backdrop(), _block(), _title()), LAYOUT)
    assert without == 0.0
    assert with_backdrop > 0.5, with_backdrop
    assert detail_with['nodes_used'] == detail_without['nodes_used'] + 1


def test_a_decorative_panel_below_the_area_cap_counts_in_collision():
    panel = SlideElement(x=96, y=48, width=1000, height=600, kind='shape', is_decorative=True)
    icon = SlideElement(x=200, y=200, width=120, height=120, kind='shape')
    score, detail = metric_collision(_geometry(panel, icon), LAYOUT)
    assert score is not None, detail
    assert score < 1.0, detail


def test_a_full_bleed_backdrop_is_held_out_of_collision_by_the_area_cap():
    left = SlideElement(x=100, y=100, width=120, height=120, kind='shape')
    right = SlideElement(x=900, y=100, width=120, height=120, kind='shape')
    score, detail = metric_collision(_geometry(_backdrop(), left, right), LAYOUT)
    assert score == 1.0, detail


def test_bounds_fidelity_does_not_let_a_backdrop_dilute_an_overflow():
    overflowing = SlideElement(x=1100, y=100, width=360, height=80, kind='text', text='x', text_len=1)
    alone, _ = metric_bounds_fidelity(_geometry(overflowing), FIDELITY)
    with_backdrop, _ = metric_bounds_fidelity(_geometry(_backdrop(), overflowing), FIDELITY)
    assert alone < 1.0
    assert with_backdrop == alone


def test_element_iou_counts_a_missing_backdrop_as_missed_content():
    reference = _geometry(_backdrop(), _block(), _title())
    candidate = _geometry(_block(), _title())
    _score, detail = metric_element_iou(candidate, reference, FIDELITY)
    assert detail['missed_from_ref'] == 1, detail
    assert detail['coverage'] < 1.0, detail


def test_a_rule_inside_a_card_collides_under_the_absolute_calibre():
    plain, plain_detail = metric_collision(_geometry(_card(), _card_title()), LAYOUT)
    ruled, ruled_detail = metric_collision(_geometry(_card(), _card_title(), _rule()), LAYOUT)
    assert plain == 1.0, plain_detail
    assert ruled < plain, ruled_detail


def test_the_relative_calibre_cancels_a_rule_present_on_both_sides():
    deck = _geometry(_card(), _card_title(), _rule())
    _, gold = compute_objective_metrics(deck, LAYOUT)
    _, candidate = compute_objective_metrics(_geometry(_card(), _card_title(), _rule()), LAYOUT)
    scores, _ = relative_objective_scores(gold, candidate, LAYOUT)
    assert scores['collision'] == 1.0, scores


def test_lines_take_part_in_element_iou():
    reference = _geometry(_card(), _rule())
    with_rule, detail_with = metric_element_iou(_geometry(_card(), _rule()), reference, FIDELITY)
    without_rule, detail_without = metric_element_iou(_geometry(_card()), reference, FIDELITY)
    assert detail_with['missed_from_ref'] == 0, detail_with
    assert detail_without['missed_from_ref'] == 1, detail_without
    assert without_rule <= with_rule


def test_a_full_bleed_rectangle_is_flagged_decorative_and_lifts_imbalance(tmp_path):
    pytest.importorskip('pptx')
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches

    from nexus.extensions.tasks.ppt_rubric.ppt_render import geometry_from_pptx

    def build(path, with_backdrop):
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        if with_backdrop:
            slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1), Inches(4), Inches(5))
        box = slide.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(3.4), Inches(0.8))
        box.text_frame.text = 'Left heavy'
        prs.save(path)
        return path

    plain, _ = geometry_from_pptx(build(str(tmp_path / 'plain.pptx'), False))
    backed, _ = geometry_from_pptx(build(str(tmp_path / 'backed.pptx'), True))
    assert sum(1 for e in backed.elements if e.is_decorative) == 1
    assert metric_imbalance(plain, LAYOUT)[0] == 0.0
    assert metric_imbalance(backed, LAYOUT)[0] > 0.5
