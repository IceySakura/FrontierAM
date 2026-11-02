"""Errors raised while compiling a native Compose task."""


class ComposeConfigError(ValueError):
    """The task's Nexus Compose extension is invalid."""


class ComposeDefinitionError(ValueError):
    """The locked Compose definition is missing or malformed."""


class ComposeAdmissionError(ValueError):
    """The locked Compose topology uses an unsupported or unsafe feature."""


__all__ = ['ComposeAdmissionError', 'ComposeConfigError', 'ComposeDefinitionError']
