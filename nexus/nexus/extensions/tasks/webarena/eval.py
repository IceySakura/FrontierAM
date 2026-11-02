"""WebArena evaluation functions.

This module contains evaluation functions for WebArena tasks.
These functions are designed to be serialized and executed in the runtime environment.

Reference: https://github.com/web-arena-x/webarena/blob/main/evaluation_harness/evaluators.py
"""

import asyncio
import collections
import html
import logging
import urllib.parse

from typing import Any


logger = logging.getLogger(__name__)


# ============================================================================
# String Evaluator Helpers
# ============================================================================


def clean_answer(answer: str) -> str:
    """Clean and normalize answer string.

    Matches original WebArena StringEvaluator.clean_answer.
    """
    answer = answer.strip()
    if answer.startswith("'") and answer.endswith("'") or answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    return answer.lower()


def exact_match(ref: str, pred: str) -> float:
    """Check exact match between reference and prediction."""
    return float(clean_answer(pred) == clean_answer(ref))


def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
    """Check if reference is included in prediction.

    Args:
        ref: Reference string that must be included
        pred: Prediction string to check
        tokenize: If True and ref is single char, tokenize pred first

    Returns:
        1.0 if ref is in pred, 0.0 otherwise
    """
    clean_ref = clean_answer(ref)
    clean_pred = clean_answer(pred)

    # Tokenize if ref is single character (prevent false positives like "0")
    if tokenize and len(clean_ref) == 1:
        # Simple word tokenization (split on whitespace and punctuation)
        import re

        tok_pred = re.findall(r'\b\w+\b', clean_pred)
        return float(clean_ref in tok_pred)
    return float(clean_ref in clean_pred)


async def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with LLM.

    Uses the exact prompt from original WebArena evaluators.py.

    Args:
        pred: Student/agent's answer
        reference: Reference answer
        question: The original question/intent

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    from nexus.core import require_context

    # Construct the prompt (exact copy from original WebArena)
    message = (
        'Help a teacher to grade the answer of a student given a question. '
        'Keep in mind that the student may use different phrasing or wording to answer the question. '
        'The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n'
    )
    message += f'question: {question}\n'
    message += f'reference answer: {reference}\n'
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f'student answer: {pred}\n'
    message += 'Conclude the judgement by correct/incorrect/partially correct.'

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': message},
    ]

    ctx = require_context()
    client = ctx.get_openai_client('judge')
    response = await client.create_completion(messages=messages)
    response_text = response.choices[0].message.content.lower()

    if 'partially correct' in response_text or 'incorrect' in response_text:
        return 0.0
    # Should contain "correct"
    return 1.0


async def llm_ua_match(pred: str, reference: str, question: str) -> float:
    """Check whether the unachievable reason matches using LLM.

    Uses the exact prompt from original WebArena evaluators.py.

    Args:
        pred: Reported unachievable reason
        reference: Actual unachievable reason
        question: The original task/question

    Returns:
        1.0 if reasons align, 0.0 otherwise
    """
    from nexus.core import require_context

    # Construct the prompt (exact copy from original WebArena)
    message = f'task: {question}\n'
    message += f'actual unachievable reason: {reference}\n'
    message += f'reported unachievable reason: {pred}\n'
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        'An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, '
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        'Determine if the reported reason aligns with the actual reason, even if implicitly. '
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': message},
    ]

    ctx = require_context()
    client = ctx.get_openai_client('judge')
    response = await client.create_completion(messages=messages)
    response_text = response.choices[0].message.content.lower()

    if 'different' in response_text:
        return 0.0
    # Should contain "same"
    return 1.0


# ============================================================================
# WebArena Evaluators
# ============================================================================


async def eval_string_match(config: dict[str, Any], answer: str) -> float:
    """Evaluate using string matching.

    Matches original WebArena StringEvaluator logic.

    Args:
        config: Task configuration containing eval.reference_answers
        answer: Agent's answer string

    Returns:
        Score (1.0 for match, 0.0 for no match)
    """
    eval_config = config.get('eval', {})
    reference_answers = eval_config.get('reference_answers', {})

    # Normalize answer
    pred = clean_answer(answer)

    score = 1.0
    for approach, value in reference_answers.items():
        if approach == 'exact_match':
            score *= exact_match(ref=value, pred=pred)

        elif approach == 'must_include':
            assert isinstance(value, list), f'must_include value should be a list, got {type(value)}'
            for must_value in value:
                score *= must_include(
                    ref=must_value,
                    pred=pred,
                    tokenize=(len(value) == 1),
                )

        elif approach == 'fuzzy_match':
            intent = config.get('intent', '')
            if value == 'N/A':
                # If the instruction only asks the model to generate N/A
                # when encountering an unachievable task
                score *= exact_match(ref=value, pred=pred)
                # If exact match fails, check if reason aligns
                if score != 1:
                    string_note = eval_config.get('string_note', '')
                    score = 1.0 * await llm_ua_match(
                        pred=pred,
                        reference=string_note,
                        question=intent,
                    )
            else:
                assert isinstance(value, list), f'fuzzy_match value should be a list, got {type(value)}'
                for ref in value:
                    score *= await llm_fuzzy_match(
                        pred=pred,
                        reference=ref,
                        question=intent,
                    )

    return score


async def eval_url_match(config: dict[str, Any], current_url: str) -> float:
    """Evaluate by checking current URL.

    Matches original WebArena URLEvaluator logic.

    Args:
        config: Task configuration containing eval.reference_url
        current_url: Current page URL

    Returns:
        Score (1.0 for match, 0.0 for no match)
    """
    eval_config = config.get('eval', {})
    reference_url = eval_config.get('reference_url', '')
    matching_rule = eval_config.get('url_note', 'GOLD in PRED')

    def clean_url(url: str) -> str:
        url = str(url)
        return url.rstrip('/')

    def parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
        """Parse a URL into its base path and query components."""
        parsed_url = urllib.parse.urlparse(url)
        # Strip trailing slashes from base_path to avoid mismatch when
        # reference URL has a trailing slash before query params (e.g., /issues/?sort=...)
        # but the predicted URL doesn't (e.g., /issues)
        base_path = (parsed_url.netloc + parsed_url.path).rstrip('/')
        query = urllib.parse.parse_qs(parsed_url.query)
        return base_path, query

    def parse_urls(urls: list[str]) -> tuple[list[str], dict[str, set[str]]]:
        """Parse a list of URLs."""
        base_paths = []
        queries: dict[str, set[str]] = collections.defaultdict(set)
        for url in urls:
            base_path, query = parse_url(url)
            base_paths.append(base_path)
            for k, v in query.items():
                queries[k].update(v)
        return base_paths, queries

    pred = clean_url(current_url)
    ref_urls = reference_url.split(' |OR| ')
    ref_urls = [clean_url(url) for url in ref_urls]

    if matching_rule == 'GOLD in PRED':
        ref_base_paths, ref_queries = parse_urls(ref_urls)
        pred_base_path, pred_query = parse_url(pred)

        # Check if any reference base path is in the predicted base path
        base_score = float(any(ref_base_path in pred_base_path for ref_base_path in ref_base_paths))

        # Check query parameters
        query_score = 1.0
        for k, possible_values in ref_queries.items():
            query_score *= float(
                any(possible_ref_value in pred_query.get(k, []) for possible_ref_value in possible_values)
            )

        score = base_score * query_score
    else:
        raise ValueError(f'Unknown matching rule: {matching_rule}')

    return score


async def eval_program_html(config: dict[str, Any], session_id: str) -> float:
    """Evaluate by checking HTML content on page(s).

    Matches original WebArena HTMLContentEvaluator logic.
    Uses browser executor API instead of direct Page access.

    Args:
        config: Task configuration containing eval.program_html
        session_id: Browser session ID

    Returns:
        Score (1.0 if all checks pass, 0.0 otherwise)
    """
    from nexus.runtime.executors.browser import (
        browser_eval_js,
        browser_get_content,
        browser_get_url,
        browser_goto,
    )
    from nexus.runtime.executors.models.browser import (
        EvalJsRequest,
        GetContentRequest,
        GetUrlRequest,
        GotoRequest,
    )

    eval_config = config.get('eval', {})
    targets = eval_config.get('program_html', [])

    if not targets:
        return 0.0

    # Get current URL once for func: expressions
    url_resp = await browser_get_url(GetUrlRequest(session_id=session_id))
    current_url = url_resp.url

    score = 1.0
    for target in targets:
        target_url: str = target.get('url', 'last')
        locator: str = target.get('locator', '')
        required_contents = target.get('required_contents', {})

        # Handle func: URL patterns
        if target_url.startswith('func'):
            func_expr = target_url.split('func:')[1]
            func_expr = func_expr.replace('__last_url__', current_url)
            try:
                target_url = eval(func_expr)
            except Exception as e:
                logger.error(f'Failed to evaluate URL function: {e}')
                score *= 0.0
                continue

        # Navigate to target URL if not "last"
        if target_url != 'last':
            try:
                await browser_goto(GotoRequest(session_id=session_id, url=target_url))
                await asyncio.sleep(3)  # Wait for page load
            except Exception as e:
                logger.error(f'Failed to navigate to {target_url}: {e}')
                score *= 0.0
                continue

        # Get selected element based on locator
        selected_element = ''

        if not locator.strip():
            # Empty locator - use full page content
            content_resp = await browser_get_content(GetContentRequest(session_id=session_id))
            selected_element = content_resp.content

        elif locator.startswith('document.') or locator.startswith('[...document.'):
            # JavaScript locator
            if 'prep_actions' in target:
                try:
                    for prep_action in target['prep_actions']:
                        await browser_eval_js(
                            EvalJsRequest(
                                session_id=session_id,
                                expression=f'() => {prep_action}',
                            )
                        )
                except Exception:
                    pass

            try:
                eval_resp = await browser_eval_js(
                    EvalJsRequest(
                        session_id=session_id,
                        expression=f'() => {locator}',
                    )
                )
                selected_element = str(eval_resp.value) if eval_resp.value else ''
            except Exception as e:
                logger.error(f'Failed to evaluate locator {locator}: {e}')
                selected_element = ''

        elif locator.startswith('func:'):
            # Helper function locator
            func_expr = locator.split('func:')[1]
            func_expr = func_expr.replace('__page__', 'page')
            try:
                selected_element = await _evaluate_helper_function(func_expr, session_id)
            except Exception as e:
                logger.error(f'Failed to evaluate helper function {func_expr}: {e}')
                selected_element = ''

        else:
            logger.error(f'Unknown locator type: {locator}')
            score *= 0.0
            continue

        # Unescape HTML entities
        selected_element = html.unescape(selected_element)

        # Check required contents
        if 'exact_match' in required_contents:
            ref_content = required_contents['exact_match']
            cur_score = exact_match(ref=ref_content, pred=selected_element)
            score *= float(cur_score)

        elif 'must_include' in required_contents:
            ref_contents = required_contents['must_include']
            assert isinstance(ref_contents, list), f'must_include should be a list, got {type(ref_contents)}'
            for content in ref_contents:
                # Support |OR| in content
                content_or = content.split(' |OR| ')
                cur_score = any(must_include(ref=c, pred=selected_element, tokenize=False) for c in content_or)
                score *= float(cur_score)

        else:
            logger.error(f'Unknown required_contents keys: {required_contents.keys()}')
            score *= 0.0

    return score


async def _evaluate_helper_function(func_expr: str, session_id: str) -> str:
    """Evaluate a helper function expression.

    Supports common patterns used in WebArena evaluations.
    """
    from nexus.runtime.executors.browser import browser_get_url
    from nexus.runtime.executors.models.browser import GetUrlRequest

    # Common helper function patterns
    if 'shopping_get_latest_order_url' in func_expr:
        # This would need API access to shopping backend
        logger.warning('shopping_get_latest_order_url not fully implemented')
        return ''

    if 'reddit_get_post_url' in func_expr:
        url_resp = await browser_get_url(GetUrlRequest(session_id=session_id))
        return _reddit_get_post_url(url_resp.url)

    if 'gitlab_get_project_memeber_role' in func_expr:
        # Extract account name from expression
        import re

        match = re.search(r"gitlab_get_project_memeber_role\(page,\s*['\"](\w+)['\"]\)", func_expr)
        if match:
            account_name = match.group(1)
            return await _gitlab_get_project_member_role(session_id, account_name)
        return ''

    logger.warning(f'Unsupported helper function: {func_expr}')
    return ''


def _reddit_get_post_url(url: str) -> str:
    """Get the post URL from a Reddit-style URL."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    domain = parsed.netloc
    tok_url = parsed.path.split('/')

    # Not a valid post/comment URL
    if len(tok_url) < 4:
        return url
    if tok_url[1] != 'f':
        return url

    subreddit = tok_url[2]
    post_id = tok_url[3]
    scheme = parsed.scheme

    return f'{scheme}://{domain}/f/{subreddit}/{post_id}/'


async def _gitlab_get_project_member_role(session_id: str, account_name: str) -> str:
    """Get the role of a project member in GitLab."""
    from nexus.runtime.executors.browser import browser_eval_js
    from nexus.runtime.executors.models.browser import EvalJsRequest

    try:
        account_resp = await browser_eval_js(
            EvalJsRequest(
                session_id=session_id,
                expression=f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()""",
            )
        )
        account_idx = account_resp.value

        if account_idx is None or account_idx < 0:
            return ''

        role_resp = await browser_eval_js(
            EvalJsRequest(
                session_id=session_id,
                expression=f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()""",
            )
        )
        return str(role_resp.value) if role_resp.value else ''
    except Exception:
        return ''
