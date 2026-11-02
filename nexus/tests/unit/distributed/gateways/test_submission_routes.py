from nexus.distributed.gateways.control.routes import submissions


def test_batch_web_ui_targets_batch_detail_page() -> None:
    urls = submissions._submission_web_ui_urls(
        console_base_url='https://console.example/',
        execution_id='exe_batch',
        spec_type='nexus_batch',
    )

    assert urls == {
        'task_web_ui': 'https://console.example/v1/console/batches/exe_batch',
        'environment_web_ui': 'https://console.example/v1/console/my-environments',
        'tracedb_web_ui': 'https://console.example/v1/console/my-environments',
    }
