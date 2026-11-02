from nexus.extensions.tasks.aidata.engine.rubric import _snapshot_state_sql_scalar


def test_state_sql_materializes_empty_table_from_payload_columns() -> None:
    state = {
        'pdd': {
            'tables': {
                'order_notes': {
                    'count': 0,
                    'rows': [],
                    'columns': ['row_id', 'order_id', 'note', 'updated_by', 'ts'],
                },
            },
        },
    }

    got, err = _snapshot_state_sql_scalar(
        state,
        'pdd',
        "SELECT CASE WHEN NOT EXISTS (SELECT 1 FROM order_notes WHERE order_id = 'ORD-1') THEN 1 ELSE 0 END",
    )

    assert err is None
    assert got == 1


def test_state_sql_materializes_empty_table_from_table_schemas() -> None:
    state = {
        'pdd': {
            'tables': {
                'order_notes': {
                    'count': 0,
                    'rows': [],
                },
            },
            'tableSchemas': {
                'order_notes': ['row_id', 'order_id', 'note', 'updated_by', 'ts'],
            },
        },
    }

    got, err = _snapshot_state_sql_scalar(
        state,
        'pdd',
        "SELECT CASE WHEN NOT EXISTS (SELECT 1 FROM order_notes WHERE order_id = 'ORD-1') THEN 1 ELSE 0 END",
    )

    assert err is None
    assert got == 1


def test_state_sql_keeps_missing_table_as_sql_error() -> None:
    state = {
        'pdd': {
            'tables': {
                'orders': {
                    'count': 1,
                    'rows': [{'order_id': 'ORD-1'}],
                },
            },
            'tableSchemas': {
                'order_notes': ['row_id', 'order_id', 'note', 'updated_by', 'ts'],
            },
        },
    }

    got, err = _snapshot_state_sql_scalar(
        state,
        'pdd',
        "SELECT CASE WHEN NOT EXISTS (SELECT 1 FROM order_notes WHERE order_id = 'ORD-1') THEN 1 ELSE 0 END",
    )

    assert got is None
    assert err is not None
    assert 'no such table: order_notes' in err
