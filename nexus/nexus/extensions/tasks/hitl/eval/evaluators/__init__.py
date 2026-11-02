"""One module per evaluator.

Each takes what it alone needs and returns an :class:`EvaluationResult`. Signatures
differ: the unit tests need a prepared sandbox, the rubrics judge needs a model
client. The recipe that assembles those inputs is what decides which evaluators run.

Combining what they return is :mod:`..aggregate`, which is independent of which ones
ran.
"""
