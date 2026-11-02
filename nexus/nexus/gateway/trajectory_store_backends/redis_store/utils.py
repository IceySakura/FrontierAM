"""Shared Redis backend helpers."""


def get_ray_head_ip() -> str:
    import ray

    if not ray.is_initialized():
        raise RuntimeError(
            'Address discovery requires an initialized Ray session; connect to Ray or pass explicit addresses'
        )
    for node in ray.nodes():
        if node['Alive'] and 'node:__internal_head__' in node['Resources']:
            return node['NodeManagerAddress']
    raise RuntimeError('Cannot find an alive Ray head node for address discovery')
