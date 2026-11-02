import re

from typing import Any, Literal

from nexus.core import require_context
from nexus.extensions.agents.base.logger import logger
from nexus.extensions.agents.base.schema import ToolChoice


FINAL_ANSWER_PROMPT = """\
<role>
You are a Precision Answer Extractor. Your task is to extract the **exact, minimal answer** from the agent's execution logs. The agent's logs often mix the actual answer together with reasoning, filtering tables, rejected candidates, hedging, and framing text. Your job is to output ONLY the answer set the question asks for, and nothing else.
</role>

<objective>
Extract the most direct and **exact** answer to the user's question. The answer must be:
1. **Concise (unless a comprehensive report is requested)**: For fact-based questions, provide only the essential answer, no explanations, no additional context. For deep research questions, provide the full analysis but remove conversational filler.
2. **Precise**: The **exact** entity, number, phrase, or structured data that directly answers the question.
3. **Grounded**: Derived directly from the tool outputs with no speculation or ambiguity.
</objective>

<extraction_rules>
- Keep only the entities that are the actual answer. Remove every rejected candidate, runner-up, "also possible", and "X or Y" hedge.
- Remove all working: filtering tables, elimination steps, intermediate lists, "here's how I narrowed it down".
- Strip annotations attached to answer items — counts, dates, vote totals, scores, parentheticals, source notes, reference links. (e.g., Output `Missouri`, not `Missouri — 9,980 votes`. Output `The Open University`, not `The Open University (largest by enrolment)`).
- Remove all preamble, framing, and decorative language (e.g., "Based on the data...", "The answer is...", "根据...", "具体来说...", "Here is the report...").
- Remove all markdown formatting (e.g., **bold**, ## headings, bullet points) ONLY IF the question asks for simple entities. For Deep Research reports or WideSearch tables, preserve the necessary markdown structure.
- Do not add, infer, or pad anything not present as a committed answer. If the agent explicitly states it does not know / found no answer, output exactly: I don't know
- Preserve each answer entity's surface name exactly as the agent committed to it (do not rename, translate, or reformat the entity itself).
- If the question asks "who", extract only the person's **name**. If "where", only the **location name**. If "when", only the **date/time**. If "how many", only the **number**.
- Conflicting information → if the agent explicitly concludes that sources disagree and no single answer is definitive, output the agent's synthesized conclusion briefly, rather than picking one side arbitrarily.
- Deep Research / Comprehensive Report → preserve the full synthesized report, document structure, and detailed analysis, but still strip the agent's conversational preamble and postamble.
</extraction_rules>

<incomplete_logs>
The agent may have stopped before fully concluding (for example, it ran out of its iteration budget). Even when the logs are partial:
- Still return the single best, most concise answer supported by the tool outputs gathered so far.
- Never return an empty answer or a refusal. If the evidence only partially supports an answer, output the most likely concise answer grounded in what was actually found.
</incomplete_logs>

<output_requirements>
First, provide your step-by-step analysis inside `<thought>` tags.
Then, provide the cleaned final answer inside `<answer>` tags.

- Single-answer question → one line inside `<answer>`: the answer entity, by itself.
- Set-answer question → one entity per line inside `<answer>`, no numbering, no commentary, no blank lines.
- Boolean question (Yes/No) → one line inside `<answer>`: exactly "Yes" or "No" (or the equivalent in the requested language).
- Quantitative question → preserve the unit of measurement if provided (e.g., "330 meters", not just "330").
- Multi-part question → output each requested attribute on a new line.
- Table/Structured Data question (WideSearch) → ONLY IF the question explicitly requests a markdown table format without arbitrary omissions, output a complete, well-formed Markdown table wrapped in a markdown code block. Ensure all requested columns and rows are present.
- Comprehensive Report (Deep Research) → output the full structured report, removing only the conversational framing.

<examples>
<example>
<description>Single Answer</description>
<question>一位官员在担任柏林办事处首任处长之前，曾在哪个办事处工作？</question>
<bad_output>根据搜索到的资料，这位官员是黄继儿。在担任柏林经贸办首任处长之前，黄继儿曾在**香港驻布鲁塞尔经济贸易办事处**工作。</bad_output>
<good_output>香港驻布鲁塞尔经济贸易办事处</good_output>
</example>

<example>
<description>Set Answer</description>
<question>哪些国家是G7成员国？</question>
<bad_output>G7成员国包括：1. 美国 2. 英国 3. 法国...</bad_output>
<good_output>美国
英国
法国
德国
日本
意大利
加拿大</good_output>
</example>

<example>
<description>WideSearch / Table</description>
<question>List the top 3 tallest buildings and their heights. Don't ask me any questions, just output the results according to the column without omitting cells arbitrarily. The output format is ```markdown
[data_content]
```.</question>
<bad_output>Here is the table you requested:
| Building | Height |
|---|---|
| Burj Khalifa | 828m |</bad_output>
<good_output>```markdown
| Building | Height |
|---|---|
| Burj Khalifa | 828m |
| Merdeka 118 | 679m |
| Shanghai Tower | 632m |
```</good_output>
</example>

<example>
<description>Deep Research / Report</description>
<question>Write a comprehensive report on the causes of the 2008 financial crisis.</question>
<bad_output>I have conducted a deep search. Here is the report:

# The 2008 Financial Crisis
The crisis was caused by...

Let me know if you need more details!</bad_output>
<good_output># The 2008 Financial Crisis
The crisis was caused by...</good_output>
</example>
</examples>
</output_requirements>

Now, extract the cleaned final answer from the execution trajectory above.
"""


async def generate_final_answer(
    trajectory: list[dict],
    return_type: Literal['dict', 'str'] = 'str',
) -> dict[str, Any] | str:
    """
    Analyzes the agent's execution trajectory and generates a final structured response.

    Args:
        trajectory (List[dict]): The history of the conversation, including tool calls and results.
        return_type (Literal["dict", "str"]): Control return format.
            - "dict": Return full dictionary with 'thought' and 'answer'.
            - "str": Return only the 'answer' string.
            Default is "str".

    Returns:
        Union[Dict[str, Any], str]:
            - If return_type="dict": A dictionary containing 'thought' and 'answer'.
            - If return_type="str": The final answer string only.
    """
    # A truncated trajectory may end on an assistant message with unanswered
    # tool_calls; appending a user turn after it violates the chat API's
    # tool_call/response pairing, so drop that dangling message first.
    safe_trajectory = list(trajectory)
    if safe_trajectory and safe_trajectory[-1].get('role') == 'assistant' and safe_trajectory[-1].get('tool_calls'):
        safe_trajectory = safe_trajectory[:-1]

    # The trajectory already begins with the agent's own system message, so the
    # extraction instructions are appended as the final user turn instead of a
    # second (conflicting) system message. Tools are disabled to force a pure
    # text (XML) response, mirroring the search_agent summarization path.
    messages = [*safe_trajectory, {'role': 'user', 'content': FINAL_ANSWER_PROMPT}]

    ctx = require_context()
    client = ctx.get_openai_client('main')
    response = await client.create_completion(
        messages=messages,
        tools=[],
        tool_choice=ToolChoice.NONE,
    )
    content = response.choices[0].message.content or ''

    # Parse XML
    thought_match = re.search(r'<thought>(.*?)</thought>', content, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)

    thought = thought_match.group(1).strip() if thought_match else 'No thought provided.'
    answer = answer_match.group(1).strip() if answer_match else content.strip()

    if not answer_match:
        logger.warning('Model returned raw text instead of <answer> tags. Using raw text as answer.')

    result = {
        'thought': thought,
        'answer': answer,
    }
    return result['answer'] if return_type == 'str' else result
