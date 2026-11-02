class DependencyBuildStatus:
    REQUESTED = 'requested'
    BUILDING = 'building'
    READY = 'ready'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


__all__ = ['DependencyBuildStatus']
