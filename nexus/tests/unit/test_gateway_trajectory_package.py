import subprocess
import sys


def test_trajectory_pickle_uses_gateway_module_path():
    script = """
import pickle
from nexus.gateway.trajectory import Trajectory

assert Trajectory.__module__ == 'nexus.gateway.trajectory'
trajectory = Trajectory(session_id='gateway')
restored = pickle.loads(pickle.dumps(trajectory))
assert restored.session_id == 'gateway'
assert restored.to_segment_node_lists() == ()
"""

    subprocess.run([sys.executable, '-c', script], check=True)
