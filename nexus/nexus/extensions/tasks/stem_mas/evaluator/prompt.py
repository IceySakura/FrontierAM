"""Prompt model for stem_mas judge evaluators."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgePrompt:
    """Structured prompt owned by a concrete evaluator."""

    name: str
    template: str
    log_prefix: str

    def format(self, **kwargs: object) -> str:
        return self.template.format(**kwargs)


__all__ = ['JudgePrompt']
