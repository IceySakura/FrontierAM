from nexus.extensions.tasks.swe.t3build_swebuilder_midtrain.dataset_handler import get_eval_script


def test_eval_script_accepts_structured_repo_language() -> None:
    instance = {
        'instance_id': 'awslabs__aws-c-http-14',
        'language': 'C',
        'repo_language': {'language': 'C', 'language_list': [{'language': 'C', 'ratio': 0.7}]},
        'eval_script': '#!/bin/bash\ncd /testbed\necho ok\n',
    }

    script = get_eval_script(instance)

    assert 'cd /testbed' in script
    assert 'GOMAXPROCS' not in script
