from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import Evaluator


MATH_TEMPLATE = """\
You live in an alternate universe. Do not assume that you know anything.
You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Given that you live in an alternate universe the TRUE answer may be different from what you expect. That's OK!

Grade the student answers based ONLY on whether it matches the TRUE answer. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin!

QUESTION: {question}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""

"""Math task: Agent solves problems in alternate universe using tools.

Core evaluation: Does agent's numeric answer match expected value?
"""


class MathEvaluator(Evaluator):
    """Strategy interface for evaluating agent outputs."""

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        """
        Evaluate prediction against reference.

        Args:
            prediction: Agent's output
            reference: Ground truth
            question: Optional input question

        Returns:
            Score in [0.0, 1.0]
        """
        prompt = MATH_TEMPLATE.format(question=question or '', result=prediction, answer=reference)
        ctx = require_context()
        client = ctx.get_openai_client('judge')
        response = await client.create_completion(messages=[{'role': 'user', 'content': prompt}])
        message = response.choices[0].message
        if message.content.startswith('CORRECT'):
            return 1.0
        return 0.0
