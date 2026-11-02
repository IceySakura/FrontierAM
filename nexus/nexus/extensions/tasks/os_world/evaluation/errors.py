"""Failures of the evaluator machinery, rather than negative task scores."""


class EvaluatorInfrastructureError(RuntimeError):
    """The sandbox state could not be reliably observed or scored."""
