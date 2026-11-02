"""Judge result models for stem_mas evaluators."""


class JudgeResult:
    """Judge evaluation result with detailed information."""

    def __init__(
        self,
        score: float,
        judge_prompt: str,
        judge_response: str,
        parsed_result: dict | None,
        error: str | None = None,
    ):
        self.score = score
        self.judge_prompt = judge_prompt
        self.judge_response = judge_response
        self.parsed_result = parsed_result
        self.error = error

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'score': self.score,
            'judge_prompt': self.judge_prompt,
            'judge_response': self.judge_response,
            'parsed_result': self.parsed_result,
            'error': self.error,
        }
