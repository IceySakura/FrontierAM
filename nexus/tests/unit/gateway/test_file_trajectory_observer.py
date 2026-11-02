import gzip
import json

from concurrent.futures import ThreadPoolExecutor

import pytest

from nexus.gateway.observers import create_trajectory_observers
from nexus.gateway.observers.file import FileTrajectoryObserver, visualization_dump
from nexus.gateway.observers.hub import TrajectoryObservationHub
from nexus.gateway.trajectory import (
    Chain,
    JoinLink,
    ProviderFacts,
    SegmentBoundary,
    SpawnLink,
    TokenAccount,
    Trajectory,
    TrajectoryNode,
)
from nexus.gateway.trajectory_store import NodePayloadRef


def export_trajectory() -> Trajectory:
    return Trajectory(
        session_id='test / file-export',
        roots=[0],
        nodes={
            0: TrajectoryNode(
                node_id=0, message_id=0, parent=None, children=[1], generated_by='client',
                original_message={'role': 'user', 'content': 'Input'},
            ),
            1: TrajectoryNode(
                node_id=1, message_id=1, parent=0, children=[2, 3], generated_by='llm',
                original_message={'role': 'assistant', 'content': 'Delegate'},
            ),
            2: TrajectoryNode(
                node_id=2, message_id=2, parent=1, children=[], generated_by='llm',
                original_message={'role': 'assistant', 'content': 'Result'},
                provider_facts=ProviderFacts(model='test-model', usage={'input_tokens': 4, 'output_tokens': 2}),
                segment_boundary=SegmentBoundary(
                    starts_new_segment=True, kind='compaction', causes=('context_compaction',),
                ),
                selection='selected',
            ),
            3: TrajectoryNode(
                node_id=3, message_id=3, parent=1, children=[], generated_by='observation',
                original_message={'role': 'tool', 'content': 'Joined'},
            ),
        },
        chains={
            'main': Chain(chain_id='main', roots=[0]),
            'worker': Chain(
                chain_id='worker', kind='subagent', roots=[2], parent_chain_id='main',
                spawn=SpawnLink(node=1), join=JoinLink(node=3, kind='poll'),
            ),
        },
        final_reward=1.0,
        termination_reason='completed',
    )


def read_artifact(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


@pytest.mark.parametrize('compress', [True, False])
def test_final_export_roundtrips_semantic_facts(tmp_path, compress):
    original = export_trajectory()
    expected = original.dump()
    observer = FileTrajectoryObserver(str(tmp_path), compress=compress)
    hub = TrajectoryObservationHub([observer])
    assert hub.reserve_update(original.session_id) is None
    assert hub.publish_final(original)
    assert not hub.publish_final(original)
    files = list(tmp_path.glob('*.trajectory.json*'))
    assert len(files) == 1
    artifact = read_artifact(files[0])
    assert artifact['trajectory'] == expected
    assert 'artifact_version' not in artifact
    restored = Trajectory.from_dump(artifact['trajectory'])
    assert restored.dump() == expected
    summary = json.loads(next(tmp_path.glob('*.summary.json')).read_text())
    assert 'artifact_version' not in summary
    assert summary['file'] == files[0].name
    assert summary['session_id'] == original.session_id
    assert not list(tmp_path.glob('*.tmp'))


def test_visualization_export_does_not_copy_or_mutate_training_header():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise AssertionError('Training tensors must not be copied')

    original = export_trajectory()
    header = {'segment': Uncopyable()}
    original.segment_training_data = header
    before_nodes = original.nodes
    result = visualization_dump(original)
    assert original.segment_training_data is header
    assert original.nodes is before_nodes
    assert result['nodes'][0]['original_message'] == original.nodes[0].original_message
    result['nodes'][0]['original_message']['content'] = 'changed'
    assert original.nodes[0].original_message['content'] != 'changed'
    assert 'segment_training_data' not in result


def test_explicit_config_and_registration_is_discovered(tmp_path):
    observers = create_trajectory_observers(
        {
            'file': {
                'name': 'trajectory_file',
                'args': {'output_dir': str(tmp_path / 'explicit'), 'compress': False},
            },
        }
    )
    assert observers['file'].output_dir == tmp_path / 'explicit'
    assert observers['file'].compress is False


def test_output_dir_is_required_even_when_legacy_env_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv('NEXUS_TRAJECTORY_OUTPUT_DIR', str(tmp_path / 'env'))
    with pytest.raises(TypeError, match='output_dir'):
        create_trajectory_observers({'file': {'name': 'trajectory_file', 'args': {}}})
    assert FileTrajectoryObserver(str(tmp_path / 'explicit')).output_dir == tmp_path / 'explicit'
    assert not (tmp_path / 'env').exists()


@pytest.mark.parametrize('output_dir', ['', '   '])
def test_empty_output_dir_is_rejected(output_dir):
    with pytest.raises(ValueError, match='non-empty output_dir'):
        FileTrajectoryObserver(output_dir)


def test_object_store_output_dir_is_rejected():
    with pytest.raises(ValueError, match='mounted Ceph'):
        FileTrajectoryObserver('s3://bucket/run')


def test_concurrent_same_session_exports_publish_one_complete_artifact(tmp_path):
    """Artifacts are keyed by session_id, so repeated exports of one session converge.

    ``on_final`` names the artifact after ``trajectory.session_id`` to keep it aligned
    with the HyTuner trajectory_id used for cross-search, which means concurrent
    exports of the same session target the same path by design. What must hold is that
    the atomic publish never leaves a torn or half-written file behind, and that unsafe
    characters in the id cannot escape ``output_dir``.
    """
    original = export_trajectory()
    original.session_id = '../../outside / user-supplied-session'
    observer = FileTrajectoryObserver(str(tmp_path))
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda _: observer.on_final(original), range(12)))

    files = list(tmp_path.glob('*.trajectory.json.gz'))
    assert len(files) == 1
    assert files[0].name == '.._.._outside___user-supplied-session.trajectory.json.gz'
    # The traversal-unsafe id stayed inside output_dir and created no extra directories.
    assert [path for path in tmp_path.iterdir() if path.is_dir()] == []
    assert not list(tmp_path.glob('.trajectory-*.tmp')), 'temporary files must not survive'
    summaries = list(tmp_path.glob('*.summary.json'))
    assert len(summaries) == 1
    # Readable end to end, i.e. the surviving artifact is a complete gzip/JSON document.
    assert read_artifact(files[0])['session_id'] == original.session_id
    assert json.loads(summaries[0].read_text())['file'] == files[0].name


def test_failed_publish_leaves_no_discoverable_partial_artifact(tmp_path, monkeypatch, caplog):
    import nexus.gateway.observers.file as module

    def fail(*args):
        raise OSError('mount unavailable')

    monkeypatch.setattr(module.os, 'replace', fail)
    hub = TrajectoryObservationHub([FileTrajectoryObserver(str(tmp_path))])
    assert hub.publish_final(export_trajectory())
    assert not list(tmp_path.iterdir())
    assert 'mount unavailable' in caplog.text


def test_failed_trajectory_is_exported_with_recorded_state(tmp_path):
    trajectory = export_trajectory()
    trajectory.state = 'failed'
    trajectory.termination_reason = 'cancelled'
    FileTrajectoryObserver(str(tmp_path)).on_final(trajectory)
    data = read_artifact(next(tmp_path.glob('*.trajectory.json.gz')))
    assert data['trajectory']['state'] == 'failed'
    assert data['trajectory']['termination_reason'] == 'cancelled'


def test_external_payload_refs_are_omitted_without_touching_source():
    original = export_trajectory()
    ref = NodePayloadRef()
    original.nodes[2].token_account = TokenAccount(payload_ref=ref, gen_len=7)
    exported = visualization_dump(original)
    account = next(node for node in exported['nodes'] if node['node_id'] == 2)['token_account']
    assert 'payload_ref' not in account
    assert account['gen_len'] == 7
    assert original.nodes[2].token_account.payload_ref is ref


def test_empty_final_trajectory_exports(tmp_path):
    FileTrajectoryObserver(str(tmp_path)).on_final(Trajectory(session_id='empty', state='failed'))
    data = read_artifact(next(tmp_path.glob('*.trajectory.json.gz')))
    assert data['trajectory']['nodes'] == []
    assert data['trajectory']['chains'] == []
