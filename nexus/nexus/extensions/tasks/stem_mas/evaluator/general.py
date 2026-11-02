"""General LLM judge evaluator for stem_mas."""

import json
import logging

from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import Evaluator
from nexus.extensions.tasks.stem_mas.evaluator.common import log_judge_input, strip_think_blocks
from nexus.extensions.tasks.stem_mas.evaluator.prompt import JudgePrompt
from nexus.extensions.tasks.stem_mas.evaluator.result import JudgeResult


logger = logging.getLogger(__name__)


GENERAL_PROMPT = JudgePrompt(
    name='general',
    log_prefix='General Judge',
    template="""
给你一个问题的两个答案（可能不完整,最终答案一般在最后），对于一个问题的每个子问题（可能一个，可能多个），你需要抽取两个答案的精简回答res1和res2，根据以下规则做一些转换，然后依据[答案一致性判断规则]比较两个答案是否一致，并给出简短的判断原因。
0.问题拆分
0.1 若问题为多选题，答案1为多个选项，不要将多选题拆分为多个子问题。若问题是选择题但答案不是选项，精简答案应该为非选项答案对应到题目中的选项。
0.2 若答案为范围或者区间，整个范围或者区间是一个完整的答案（子问题类型为公式），不要拆分成多个子问题。
1. 子问题类型为公式、数值、选项、简答其中之一。化学表达式算是公式的一种。
2. 若回答的精简内容为选项，精简答案的形式为选项的拼接（字符串形式，多个选项用,分割）。举个例子，若答案为B和C，res为"B,C"。
3. 若回答的精简内容为公式，需要将精简答案转化为latex格式，即res1和res2为\\boxed{{...}}的格式。注意res1和res2不要带单位。
4. 若回答的精简内容为数值，需要考虑单位情况，将单位统一后二者的数值分别放到res1和res2。注意res1和res2不要带单位。
4.1 若两个答案都不带单位或者都带相同的单位，直接抽取。
4.2 若两个答案都带单位，但单位不同，需要按照答案1的单位，对答案2的结果进行单位转换，将答案1和答案2的单位统一。
4.3 若答案1没带单位，默认是题目给定的单位；若答案2没带单位，可以结合完整的答案。然后结合4.2看是否需要转化。
5. 若答案2的最终答案在之前被四舍五入过，则抽取四舍五入之前的那个答案，即精度选更高的一个答案
6. 若回答的精简内容为数值，必须将其转化为latex格式，即res1和res2为\\boxed{{...}}的格式。若其中包含分数、根式、指数、对数、科学计数法等，不要计算为小数，也不要进行化简约分，直接放\\boxed{{...}}里面。若原来为小数或者整数，也放\\boxed{{...}}里，数值不要改变。
7. 若回答为空或者无意义内容，或者结尾处被截断，没有有效答案，精简内容为空字符串
8. 若题目中有小问是证明题，抽取答案1和答案2中对应证明题的小问答案，如果没有则精简内容为空字符串，统一打分为0
9. 依据下面给出的[答案一致性判断规则]，比较res1和res2是否一致，若一致分数为1，否则为0，并给出简短的判断原因。
**注意**：这里的一致，指的是res1和res2在实际含义上的一致性，而不是字面上的完全一致。

[答案一致性判断规则]
- 若答案类型为"选项"，则比较两个答案的选项内容是否一致，如果是多选，则忽略选项的顺序
- 如果其中一个答案类型为"数值"，需要根据题目要求，判断进行一致性比较时，是否需要考虑精度
  - 若题目中有明确的精度要求，则需要比较两个数值，或数值与公式的计算结果在给定精度上是否完全一致
    - 比较时，若答案1和答案2的单位不同，则需要先将两者的单位统一，然后再进行精度比较
  - 若题目中没有明确的精度要求，则只需要比较两个数值，或数值与公式的计算结果是否一致时，可以允许一定程度的精度差异，如"2.34"和"2.342"可以认为是相同的，只要符合四舍五入的规则即可。但是要注意，如果答案的有效数字小于3位，不能直接进行四舍五入判断，例如0.6和1，二者差距过大，不能认为是相同的
- 若答案类型为"数值"或"公式"。
  - 需要在单位统一的前提下，比较两个答案的数值或公式内容是否等价，即数值是否相等，公式之间是否可以相互转换。如果一个是数值，另一个是公式，则公式的计算结果是否等于数值
  - 若答案1和答案2只有正负号不一样，根据题意判断正负号是否影响正确性，例如如果题目要求的是距离、长度等非负量，则正负号不影响正确性，认为两者一致；如果题目要求的是位移、功等可能为负的量，则正负号影响正确性，认为两者不一致。
  - 若答案1和答案2都是公式，且答案2是答案1的通解形式，在题目没有要求必须使用特殊解的情况下，则认为两者一致。
- 若答案类型为"简答"，则比较两个答案的内容在语义上是否一致，忽略字面上的差异
- 如果答案属于"化学表达式"，则需要依据化学知识判断它们是否等价，比如化学反应方程式的平衡、化学物质的状态等
- 若答案是范围或者区间，需要按照数学中的区间正确性来判定

你的输出需要是以下json列表[{{"类型": 类型, "res1": res1, "res2": res2, "单位": 单位(可以为空字符串), "分数": score, "判断原因": xxx}}, ...]的格式，其中每一个json表示一个子问题。

问题：
{question}


答案1：
{ref_answer}


答案2：
{answer}
""",
)


class GeneralEvaluator(Evaluator):
    """General evaluator with detailed Chinese rules for sub-question analysis."""

    last_judge_result: JudgeResult | None = None

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        clean_prediction = strip_think_blocks(prediction)
        prompt = GENERAL_PROMPT.format(question=question, ref_answer=reference, answer=clean_prediction)

        log_judge_input(GENERAL_PROMPT.log_prefix, question, clean_prediction, reference, prompt)

        ctx = require_context()
        client = ctx.get_openai_client('judge')
        response = await client.create_completion(messages=[{'role': 'user', 'content': prompt}])
        message = response.choices[0].message
        output = message.content

        logger.debug(f'[General Judge] Response:\n{output}')

        try:
            try:
                ans_check_output = json.loads(output)
            except json.JSONDecodeError:
                ans_check_output = json.loads(output.replace('\\', '\\\\'))

            if isinstance(ans_check_output, list) and len(ans_check_output) > 0:
                score = sum(x['分数'] for x in ans_check_output) / len(ans_check_output)
                logger.debug(f'[General Judge] Parsed: {ans_check_output}')
                logger.info(f'[General Judge] Score: {score}')
                logger.debug('=' * 80)
                self.last_judge_result = JudgeResult(
                    score=score,
                    judge_prompt=prompt,
                    judge_response=output,
                    parsed_result={'sub_questions': ans_check_output},
                )
                return score
            raise ValueError('Invalid output format: expected non-empty list')

        except Exception as e:
            logger.warning(f'[General Judge] Failed to parse output: {e}, returning 0.0')
            logger.debug('=' * 80)
            self.last_judge_result = JudgeResult(
                score=0.0,
                judge_prompt=prompt,
                judge_response=output,
                parsed_result=None,
                error=f'Failed to parse judge output: {str(e)}',
            )
            return 0.0

    def get_last_judge_result(self) -> dict | None:
        if self.last_judge_result is None:
            return None
        return self.last_judge_result.to_dict()
