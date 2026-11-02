from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.extensions.tasks.ppt_rubric.ppt_metrics import PPTRubricConfig
from nexus.extensions.tasks.ppt_rubric.ppt_structure import DeckElement, DeckStructure, render_structure_listing


def test_codex_system_prompt_is_staged_as_model_instructions_file():
    import asyncio

    import nexus.extensions.tasks.ppt_rubric.task as ppt_task

    captured = {}

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)

        class _Output:
            final_response = '{}'

        return _Output()

    task_input = ppt_task.PPTTaskInput(
        run_agent=fake_run_agent,
        system_prompt='be precise',
        question='q',
        instance_id='i',
        codex_image_inject_mode='path',
    )

    uploaded_paths = []

    async def fake_upload_file(items, overwrite=True):
        uploaded_paths.extend(item.path for item in items)

        class _Result:
            success = True
            message = ''

        class _Response:
            results = [_Result()]

        return _Response()

    fake_runtime = MagicMock()
    fake_runtime.upload_file = AsyncMock(side_effect=fake_upload_file)

    fake_ctx = MagicMock()
    fake_ctx.agent_name = 'codex'
    fake_ctx.get_runtime = AsyncMock(return_value=fake_runtime)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ppt_task, 'require_context', lambda: fake_ctx)
        patch.setattr(ppt_task, '_find_generator', AsyncMock(side_effect=RuntimeError('stop')))
        with pytest.raises(RuntimeError):
            asyncio.run(ppt_task.run_ppt_topic(task_input))

    assert captured.get('developer_instructions') is None
    staged = captured.get('model_instructions_file') or ''
    assert staged.startswith('/tmp/ppt_sysp_')
    assert staged.endswith('.txt')
    assert uploaded_paths == [staged]
    assert 'extra_config_toml' not in captured


def test_codex_system_prompt_degrades_to_builtin_instructions_when_staging_fails():
    import asyncio

    import nexus.extensions.tasks.ppt_rubric.task as ppt_task

    captured = {}

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)

        class _Output:
            final_response = '{}'

        return _Output()

    task_input = ppt_task.PPTTaskInput(
        run_agent=fake_run_agent,
        system_prompt='be precise',
        question='q',
        instance_id='i',
        codex_image_inject_mode='path',
    )

    fake_runtime = MagicMock()
    fake_runtime.upload_file = AsyncMock(side_effect=OSError('sandbox gone'))

    fake_ctx = MagicMock()
    fake_ctx.agent_name = 'codex'
    fake_ctx.get_runtime = AsyncMock(return_value=fake_runtime)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ppt_task, 'require_context', lambda: fake_ctx)
        patch.setattr(ppt_task, '_find_generator', AsyncMock(side_effect=RuntimeError('stop')))
        with pytest.raises(RuntimeError):
            asyncio.run(ppt_task.run_ppt_topic(task_input))

    assert 'model_instructions_file' not in captured
    assert 'extra_config_toml' not in captured


def test_zero_height_line_reports_its_stroke_and_dash():
    line = DeckElement(
        kind='line', x0=0.1, y0=0.5, x1=0.4, y1=0.5, stroke_pt=2.2, dash='dash', stroke_color=(84, 170, 67)
    )
    listing = render_structure_listing(DeckStructure(elements=[line]))
    assert 'stroke2.2pt' in listing
    assert 'dash:dash' in listing
    assert 'line#54AA43' in listing


def test_inherited_stroke_width_is_omitted_from_the_listing():
    line = DeckElement(kind='line', x0=0.1, y0=0.5, x1=0.4, y1=0.5, stroke_pt=None)
    listing = render_structure_listing(DeckStructure(elements=[line]))
    assert 'stroke' not in listing


def test_listing_header_declares_rendered_visible_characters():
    text = DeckElement(kind='text', x0=0.1, y0=0.1, x1=0.5, y1=0.2, text='hello')
    listing = render_structure_listing(DeckStructure(elements=[text]), visible_chars=69)
    header = listing.splitlines()[0]
    assert header.startswith('# rendered visible characters: 69')
    assert 'still listed' in header


def test_listing_without_visible_chars_has_no_header():
    text = DeckElement(kind='text', x0=0.1, y0=0.1, x1=0.5, y1=0.2, text='hello')
    listing = render_structure_listing(DeckStructure(elements=[text]))
    assert not listing.startswith('#')


def test_layout_weights_are_one_except_the_retired_whitespace():
    cfg = PPTRubricConfig()
    weights = cfg.objective_weights()
    assert weights['whitespace'] == 0.0
    for name in ('text_density', 'imbalance', 'text_overflow', 'collision', 'contrast'):
        assert weights[name] == 1.0, (name, weights[name])


@pytest.mark.parametrize(
    ('has_ref', 'reference_parsed', 'reason'),
    [
        (False, False, 'no reference deck'),
        (True, False, 'the reference deck could not be parsed'),
        (True, True, 'no layout metric was measured on both decks'),
    ],
)
def test_gold_relative_falls_back_to_absolute_and_states_why(has_ref, reference_parsed, reason):
    from nexus.extensions.tasks.ppt_rubric.scoring import _select_objective_calibre

    calibre = _select_objective_calibre('gold_relative', None, has_ref=has_ref, reference_parsed=reference_parsed)
    assert calibre == ('absolute', reason)


def _score_a_ruled_card(tmp_path, monkeypatch, rubric_config, *, with_reference):
    from nexus.extensions.tasks.ppt_rubric import scoring
    from nexus.extensions.tasks.ppt_rubric.ppt_metrics import SlideElement, SlideGeometry

    def ruled_card():
        elements = [
            SlideElement(x=96, y=96, width=576, height=480, kind='shape'),
            SlideElement(x=125, y=115, width=480, height=58, kind='text', text='Title', text_len=5),
            SlideElement(x=125, y=287, width=499, height=1.0, kind='line'),
        ]
        return SlideGeometry(elements=elements, slide_aspect=16 / 9)

    monkeypatch.setattr(scoring, 'render_environment', lambda: {})
    monkeypatch.setattr(scoring, '_build_geometry', lambda path, **kwargs: (ruled_card(), {'n_slides': 1}))
    monkeypatch.setattr(scoring, 'extract_elements', lambda path: DeckStructure())
    deck = tmp_path / 'card.pptx'
    reference = tmp_path / 'reference.pptx'
    deck.write_bytes(b'')
    reference.write_bytes(b'')
    return scoring.score_pptx(
        str(deck),
        [1280, 720],
        {'enable_pdf_measure': False, **rubric_config},
        ref_pptx_path=str(reference) if with_reference else '',
        want_judge_png=False,
    )


def test_score_pptx_scores_a_row_that_names_no_calibre_on_gold_relative(tmp_path, monkeypatch):
    result = _score_a_ruled_card(tmp_path, monkeypatch, {'weights': {'w_objective': 0.15}}, with_reference=True)
    assert result['completed'], result['error']
    assert result['diagnostics']['objective_mode'] == 'gold_relative'
    assert 'objective_mode_fallback' not in result['diagnostics']
    assert result['objective'] == result['objective_relative'] == pytest.approx(1.0)


def test_score_pptx_records_why_it_fell_back_to_absolute(tmp_path, monkeypatch):
    result = _score_a_ruled_card(tmp_path, monkeypatch, {}, with_reference=False)
    assert result['completed'], result['error']
    assert result['diagnostics']['objective_mode'] == 'absolute'
    assert result['diagnostics']['objective_mode_fallback'] == 'no reference deck'
    assert result['objective_relative'] is None
    assert result['objective'] < 1.0


def test_score_pptx_keeps_an_explicit_absolute_row_absolute(tmp_path, monkeypatch):
    absolute = {'weights': {'objective_mode': 'absolute'}}
    result = _score_a_ruled_card(tmp_path, monkeypatch, absolute, with_reference=True)
    assert result['completed'], result['error']
    assert result['diagnostics']['objective_mode'] == 'absolute'
    assert 'objective_mode_fallback' not in result['diagnostics']
    assert result['objective'] < result['objective_relative'] == pytest.approx(1.0)


def _score_with_sandbox_result(diagnostics):
    import asyncio

    from types import SimpleNamespace

    import nexus.extensions.tasks.ppt_rubric.task as ppt_task

    async def fake_run_agent(**kwargs):
        return SimpleNamespace(final_response='{}')

    runtime = MagicMock()
    runtime.run_command = AsyncMock(return_value=SimpleNamespace(stdout='', exit_code=1))
    runtime.invoke = AsyncMock(
        return_value={'completed': True, 'objective': 0.5, 'fidelity': 0.8, 'metrics': {}, 'diagnostics': diagnostics}
    )
    fake_ctx = MagicMock()
    fake_ctx.agent_name = ''
    fake_ctx.get_runtime = AsyncMock(return_value=runtime)

    task_input = ppt_task.PPTTaskInput(
        run_agent=fake_run_agent, instance_id='row-7', user_prompt='q', source_size=[1280, 720]
    )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ppt_task, 'require_context', lambda: fake_ctx)
        return asyncio.run(ppt_task.run_ppt_topic(task_input))


def test_the_host_warns_only_when_the_calibre_fell_back(caplog):
    caplog.set_level('WARNING', logger='nexus.extensions.tasks.ppt_rubric.task')

    fallback = {'objective_mode': 'absolute', 'objective_mode_fallback': 'no reference deck'}
    assert _score_with_sandbox_result(fallback).completed
    warnings = [r.getMessage() for r in caplog.records if 'gold_relative objective unavailable' in r.getMessage()]
    assert len(warnings) == 1, warnings
    assert 'row-7' in warnings[0]
    assert 'no reference deck' in warnings[0]

    caplog.clear()
    assert _score_with_sandbox_result({'objective_mode': 'gold_relative'}).completed
    assert not [r for r in caplog.records if 'gold_relative objective unavailable' in r.getMessage()]


def test_inherited_width_still_gets_a_real_stroke_box(tmp_path):
    pytest.importorskip('pptx')
    from pptx import Presentation
    from pptx.enum.shapes import MSO_CONNECTOR
    from pptx.util import Inches

    from nexus.extensions.tasks.ppt_rubric.ppt_render import geometry_from_pptx

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1), Inches(1), Inches(3), Inches(1))
    path = str(tmp_path / 'connector_inherited.pptx')
    prs.save(path)

    geo, diag = geometry_from_pptx(path)
    assert geo is not None, diag
    thin = [e for e in geo.elements if e.height < 2.0 and e.width > 10]
    assert thin, 'the connector must survive the area filter'
    assert all(e.height > 0 for e in thin), [(e.height, e.width) for e in geo.elements]


def _card_deck(path, divider=False, on_edge=False):
    from pptx import Presentation
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.util import Inches

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1), Inches(1), Inches(6), Inches(5))
    box = slide.shapes.add_textbox(Inches(1.3), Inches(1.2), Inches(5), Inches(0.6))
    box.text_frame.text = 'Title'
    if divider:
        y = Inches(1) if on_edge else Inches(3)
        slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.3), y, Inches(6.5), y)
    prs.save(path)
    return path


def test_connectors_are_classified_as_lines(tmp_path):
    pytest.importorskip('pptx')
    from nexus.extensions.tasks.ppt_rubric.ppt_render import geometry_from_pptx

    geo, diag = geometry_from_pptx(_card_deck(str(tmp_path / 'card.pptx'), divider=True))
    assert geo is not None, diag
    assert [e.kind for e in geo.elements].count('line') == 1


def test_a_divider_counts_as_a_collision_under_the_absolute_calibre(tmp_path):
    pytest.importorskip('pptx')
    from nexus.extensions.tasks.ppt_rubric.ppt_metrics import metric_collision
    from nexus.extensions.tasks.ppt_rubric.ppt_render import geometry_from_pptx

    cfg = PPTRubricConfig()
    plain, _ = metric_collision(geometry_from_pptx(_card_deck(str(tmp_path / 'plain.pptx')))[0], cfg)
    ruled, _ = metric_collision(geometry_from_pptx(_card_deck(str(tmp_path / 'ruled.pptx'), divider=True))[0], cfg)
    assert plain == 1.0
    assert ruled < plain, (plain, ruled)


def test_relative_calibre_cancels_rules_present_on_both_sides():
    from nexus.extensions.tasks.ppt_rubric.ppt_metrics import relative_objective_scores

    cfg = PPTRubricConfig()
    gold = {'collision': {'collision_density': 0.6}}
    same = {'collision': {'collision_density': 0.6}}
    scores, _ = relative_objective_scores(gold, same, cfg)
    assert scores['collision'] == 1.0


def test_claude_code_rejects_a_system_prompt_it_cannot_receive():
    import asyncio

    import nexus.extensions.tasks.ppt_rubric.task as ppt_task

    async def unreachable_run_agent(**kwargs):
        raise AssertionError('claude_code must reject system_prompt before running the agent')

    task_input = ppt_task.PPTTaskInput(run_agent=unreachable_run_agent, system_prompt='be precise', instance_id='i')

    fake_ctx = MagicMock()
    fake_ctx.agent_name = 'claude_code'
    fake_ctx.get_runtime = AsyncMock(return_value=MagicMock())

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ppt_task, 'require_context', lambda: fake_ctx)
        with pytest.raises(ValueError, match='system_prompt'):
            asyncio.run(ppt_task.run_ppt_topic(task_input))


def test_decorative_inkless_column_is_declared():
    from nexus.extensions.tasks.ppt_rubric import judge

    assert judge.stable_columns()['ppt_decorative_inkless'] == judge.NOT_MEASURED


def _background_deck(path, background):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if background is not None:
        rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        rect.fill.solid()
        rect.fill.fore_color.rgb = RGBColor(*background)
        rect.line.fill.background()
    slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1), Inches(4), Inches(5))
    box = slide.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(3.4), Inches(0.8))
    box.text_frame.text = 'Left heavy'
    prs.save(path)
    return path


@pytest.mark.skipif(
    not (__import__('shutil').which('soffice') or __import__('shutil').which('libreoffice')),
    reason='needs LibreOffice to render the deck',
)
def test_invisible_full_bleed_rectangle_is_counted_as_inkless(tmp_path):
    pytest.importorskip('pptx')
    pytest.importorskip('fitz')
    from nexus.extensions.tasks.ppt_rubric.ppt_pdf_layout import pptx_to_pdf
    from nexus.extensions.tasks.ppt_rubric.ppt_render import count_inkless_decorations, geometry_from_pptx

    counts = {}
    for name, background in (('none', None), ('white', (255, 255, 255)), ('blue', (30, 60, 120))):
        path = _background_deck(str(tmp_path / f'{name}.pptx'), background)
        pdf, diag = pptx_to_pdf(path, str(tmp_path), timeout=180)
        assert pdf, diag
        geo, _ = geometry_from_pptx(path)
        counts[name], _ = count_inkless_decorations(path, geo, pdf)
    assert counts == {'none': 0, 'white': 1, 'blue': 0}, counts


def _mutation_bait_deck(path):
    from pptx import Presentation
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.oxml.ns import qn
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1), Inches(1), Inches(3), Inches(1))
    hidden = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1), Inches(2), Inches(3), Inches(2))
    hidden.line.fill.background()
    box = slide.shapes.add_textbox(Inches(1), Inches(3), Inches(3), Inches(1))
    box.text_frame.text = 'plain run'
    bare = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(5), Inches(1), Inches(2), Inches(1))
    bare._element.remove(bare._element.find(qn('p:txBody')))
    table = slide.shapes.add_table(1, 2, Inches(5), Inches(3), Inches(3), Inches(1)).table
    cell = table.cell(0, 1)
    cell._tc.remove(cell._tc.find(qn('a:txBody')))
    prs.save(path)
    return path


def test_reading_a_deck_never_rewrites_its_xml(tmp_path, monkeypatch):
    pytest.importorskip('pptx')
    import pptx

    from lxml import etree

    from nexus.extensions.tasks.ppt_rubric.ppt_render import geometry_from_pptx
    from nexus.extensions.tasks.ppt_rubric.ppt_structure import extract_elements

    path = _mutation_bait_deck(str(tmp_path / 'bait.pptx'))
    pristine = etree.tostring(pptx.Presentation(path).slides[0]._element)

    opened = []
    real_presentation = pptx.Presentation

    def spy(source):
        prs = real_presentation(source)
        opened.append(prs)
        return prs

    monkeypatch.setattr(pptx, 'Presentation', spy)
    geometry_from_pptx(path)
    extract_elements(path)

    assert len(opened) == 2
    for prs in opened:
        assert etree.tostring(prs.slides[0]._element) == pristine


def test_a_no_fill_line_reports_no_colour_and_stays_no_fill(tmp_path):
    pytest.importorskip('pptx')
    from pptx import Presentation
    from pptx.oxml.ns import qn

    from nexus.extensions.tasks.ppt_rubric.ppt_structure import _line_props

    prs = Presentation(_mutation_bait_deck(str(tmp_path / 'bait.pptx')))
    hidden = prs.slides[0].shapes[1]
    _width, _dash, colour = _line_props(hidden)
    line_element = hidden._element.find(qn('p:spPr')).find(qn('a:ln'))
    assert colour is None
    assert line_element.find(qn('a:noFill')) is not None
    assert line_element.find(qn('a:solidFill')) is None
