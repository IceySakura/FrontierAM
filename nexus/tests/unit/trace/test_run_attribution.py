from nexus.trace.types import RUN_ID_ATTRIBUTE, RUN_KIND_ATTRIBUTE, RUN_SPAN_ID_ATTRIBUTE, run_attribution_attributes


def test_run_attribution_attributes_uses_one_canonical_key_set() -> None:
    assert run_attribution_attributes(
        run_id='run-1',
        run_span_id='run-span-1',
        run_kind='single',
    ) == {
        RUN_ID_ATTRIBUTE: 'run-1',
        RUN_SPAN_ID_ATTRIBUTE: 'run-span-1',
        RUN_KIND_ATTRIBUTE: 'single',
    }


def test_run_attribution_attributes_omits_missing_values() -> None:
    assert run_attribution_attributes(run_id='run-1') == {RUN_ID_ATTRIBUTE: 'run-1'}
