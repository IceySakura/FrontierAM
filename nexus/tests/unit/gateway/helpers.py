def freeze_assembler(assembler):
    return assembler.freeze(session_id='test', state='completed')


def latest_node_id(assembler):
    return max(assembler.trajectory.nodes)
