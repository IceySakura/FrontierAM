"""Custom Web task implementation for browser-based automation.

This task supports:
- Custom Docker-based websites (e.g., delivery app)
- Multi-type evaluation via verifier module (state, retrieval, navigation, action)
- Trajectory collection for RL training

Task format follows delivery.jsonl structure:
{
    "ID": "001",
    "task": "任务描述",
    "intentions": ["意图1", "意图2"],
    "classes": ["页面导航类", "状态调整类"],
    "groundtruth": [
        {"type": "state", "expected_states": {...}, "options": {...}},
        {"type": "retrieval", "expected_content": "...", "options": {...}}
    ]
}
"""

import contextlib
import logging

from typing import Any

from pydantic import BaseModel, Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Default custom web service configuration
DEFAULT_SERVICE_CONFIG = {
    'image': 'mirrors.tencent.com/hunyuan_yanguan/delivery-app:latest',
    'port': 80,
    'startup_wait_seconds': 30,
    'health_check_path': '/',
}


# ============================================================================
# Helper Functions
# ============================================================================


def _determine_runtime_config(service_config: dict[str, Any]) -> tuple[str, list[int], dict[str, int]]:
    """Determine runtime configuration based on service config.

    Args:
        service_config: Service configuration dictionary

    Returns:
        Tuple of (image, exposed_ports, service_port_map):
        - image: Docker image to use
        - exposed_ports: List of internal ports to expose
        - service_port_map: Mapping from service name to internal port
    """
    image = service_config.get('image', DEFAULT_SERVICE_CONFIG['image'])
    port = service_config.get('port', DEFAULT_SERVICE_CONFIG['port'])

    exposed_ports = [port]
    service_port_map = {'main': port}

    return image, exposed_ports, service_port_map


def _substitute_urls(text: str, urls: dict[str, str]) -> str:
    """Substitute URL placeholders in text with actual URLs.

    Args:
        text: Text containing placeholders like __SERVICE__
        urls: Mapping from URL keys to actual URLs

    Returns:
        Text with substituted URLs
    """
    result = text
    for pattern, url in urls.items():
        placeholder = f'__{pattern.upper()}__'
        result = result.replace(placeholder, url.rstrip('/'))
    return result


# ============================================================================
# Helper Models for invoke
# ============================================================================


class SetupCustomWebTaskResult(BaseModel):
    """Result from setup_custom_web_task function."""

    success: bool
    goal: str
    error_message: str | None = None


class EvaluateCustomWebResult(BaseModel):
    """Result from evaluate_custom_web function."""

    score: float
    success: bool
    details: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


# ============================================================================
# Data Models
# ============================================================================


class TaskData(BaseModel):
    """Task data following delivery.jsonl format.

    Attributes:
        ID: Unique task identifier
        task: Task description/instruction for the agent
        intentions: List of expected user intentions
        classes: Task classification tags
        groundtruth: List of groundtruth items for evaluation
    """

    ID: str
    task: str
    intentions: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    groundtruth: list[dict[str, Any]] = Field(default_factory=list)


class CustomWebTaskInput(TaskRunInput, WithAgentRunner):
    """Custom web task input.

    All task data is embedded directly in the input, no external loader required.

    Attributes:
        task_data: Task data following delivery.jsonl format
        service_config: Docker service configuration (image, port, etc.)
        start_url: Starting URL for the task (may contain __SERVICE__ placeholder)
        runtime_name: Runtime name to use
        timeout: Timeout in seconds
        with_na_hint: Whether to include N/A hint in instruction
        judge_client_name: Name of OpenAI client for LLM judge (uses ctx.get_openai_client)
    """

    task_data: TaskData
    service_config: dict[str, Any] = Field(default_factory=lambda: DEFAULT_SERVICE_CONFIG.copy())

    # URL configuration
    start_url: str = '__SERVICE__/'

    # Runtime configuration
    runtime_name: str = 'main'
    timeout: int = 600  # 10 minutes per task

    # Task options
    with_na_hint: bool = False

    # LLM judge client name for retrieval verification (uses ctx.get_openai_client)
    judge_client_name: str = 'judge'


class CustomWebTaskOutput(TaskRunOutput, WithScore):
    """Custom web task output.

    Attributes:
        task_id: Task identifier
        score: Score (0.0 or 1.0)
        is_success: Whether task was completed successfully
        agent_answer: The agent's final answer
        error_message: Error message if task failed
        trajectory_length: Number of agent iterations
        evaluation_details: Additional evaluation information
    """

    task_id: str
    score: float
    is_success: bool = False
    agent_answer: str | None = None
    error_message: str | None = None
    trajectory_length: int = 0
    evaluation_details: dict[str, Any] = Field(default_factory=dict)


@task_registry.register('custom_web')
async def run_custom_web_task(task_input: CustomWebTaskInput) -> CustomWebTaskOutput:
    """Run custom web task with Docker-based website.

    Workflow:
    1. Determine required services from service_config
    2. Configure runtime provider with image and exposed ports
    3. Acquire runtime and wait for services to start
    4. Substitute URL placeholders with actual service URLs
    5. Create browser session using executor
    6. Setup task (navigation) using executor
    7. Run agent with browser tools (session_id in tool_state)
    8. Extract answer from trajectory
    9. Evaluate using verifier
    10. Destroy browser session
    """
    task_data = task_input.task_data
    logger.debug(f'Running custom web task {task_data.ID}')

    ctx = require_context()
    assert ctx.runtime_provider is not None, 'custom_web requires runtime provider'

    # Determine image and ports based on service_config
    service_config = task_input.service_config
    image, exposed_ports, service_port_map = _determine_runtime_config(service_config)

    # Configure runtime provider with image and exposed ports
    ctx.runtime_provider.configure('main', image=image, exposed_ports=exposed_ports)
    runtime = await ctx.get_runtime(task_input.runtime_name)

    browser_session_id = None

    try:
        # Step 1: Start custom web services
        from nexus.extensions.tasks.custom_web.service_setup import setup_custom_web_services

        logger.debug('Starting custom web services...')
        service_urls = await setup_custom_web_services(runtime, service_config, service_port_map)
        logger.debug(f'Services started: {service_urls}')

        # Step 2: Build complete URL mapping
        custom_web_urls = {'service': service_urls.get('main', '')}

        # Step 3: Substitute URL placeholders in start_url
        start_url = _substitute_urls(task_input.start_url, custom_web_urls)

        # Step 4: Create browser session
        logger.debug('Creating browser session...')
        create_result = await runtime.create_browser_session(
            headless=True, viewport_width=1280, viewport_height=720, timeout=10000
        )
        browser_session_id = create_result.session_id
        logger.debug(f'Browser session created: {browser_session_id}')
        ctx.tool_state['browser_session_id'] = browser_session_id

        # Step 5: Setup custom web task (navigation)
        logger.debug('Setting up custom web task...')
        setup_result = await runtime.invoke(
            _setup_custom_web_task_impl,
            session_id=browser_session_id,
            start_url=start_url,
            task_description=task_data.task,
        )

        if not setup_result.success:
            raise RuntimeError(f'Failed to setup custom web task: {setup_result.error_message}')

        goal = setup_result.goal

        # Add optional hints
        if task_input.with_na_hint:
            goal += '\n\n如果你认为任务无法完成，请回答 "N/A"。'

        logger.debug(f'Task goal: {goal[:200]}...')

        # Step 6: Run agent with browser tools (session_id in initial_tool_state)
        logger.debug('Running agent...')
        agent_output = await task_input.run_agent(
            user_prompt=goal,
            timeout=task_input.timeout,
        )

        # Step 7: Extract answer from trajectory
        if not isinstance(agent_output, WithMainTrajectory):
            raise RuntimeError('custom_web task requires agent output to be a subclass of WithMainTrajectory')

        agent_answer = _extract_answer(agent_output.main_trajectory)
        trajectory_length = len(agent_output.main_trajectory)

        logger.debug(f'Agent answer: {agent_answer}')

        # Step 8: Evaluate task completion using verifier
        logger.debug('Evaluating task completion...')

        groundtruth = task_data.groundtruth
        if not groundtruth:
            logger.warning('No groundtruth specified, returning success')
            return CustomWebTaskOutput(
                task_id=task_data.ID,
                score=1.0,
                is_success=True,
                agent_answer=agent_answer,
                trajectory_length=trajectory_length,
                evaluation_details={'message': 'No groundtruth to verify'},
            )

        # print(agent_output.main_trajectory)
        # Extract actions and pages from trajectory
        executed_actions, executed_pages = _extract_actions_and_pages(agent_output.main_trajectory)

        # Get current application state from page
        current_state = await _get_application_state_via_runtime(runtime, browser_session_id)
        logger.debug(f'Current state keys: {list(current_state.keys())}')

        # Run evaluation via verifier
        from nexus.extensions.tasks.custom_web.evaluator import evaluate_with_verifier

        verification_result = await evaluate_with_verifier(
            groundtruth=groundtruth,
            agent_answer=agent_answer,
            executed_actions=executed_actions,
            executed_pages=executed_pages,
            current_state=current_state,
            question=task_data.task,  # 传递任务题目
        )

        score = verification_result.score
        is_success = verification_result.success

        logger.info(f'Task {task_data.ID} - Score: {score}, Success: {is_success}')

        return CustomWebTaskOutput(
            task_id=task_data.ID,
            score=score,
            is_success=is_success,
            agent_answer=agent_answer,
            trajectory_length=trajectory_length,
            evaluation_details=verification_result.to_dict(),
        )

    except Exception as e:
        logger.error(f'Task {task_data.ID} failed: {e}', exc_info=True)
        return CustomWebTaskOutput(
            task_id=task_data.ID,
            score=0.0,
            is_success=False,
            agent_answer=None,
            error_message=str(e),
        )

    finally:
        # Step 9: Cleanup - destroy browser session
        if browser_session_id:
            try:
                logger.debug(f'Destroying browser session: {browser_session_id}')
                await runtime.destroy_browser_session(session_id=browser_session_id)
            except Exception as e:
                logger.error(f'Failed to destroy browser session: {e}')


def _extract_answer(trajectory: list[dict]) -> str | None:
    """Extract the final answer from agent's trajectory.

    Looks for submit_result tool calls or the last assistant message.
    """
    # First, look for submit_result tool calls
    for msg in reversed(trajectory):
        if msg.get('role') == 'assistant':
            content = msg.get('content', [])
            if isinstance(content, list):
                for block in content:
                    if block.get('type') == 'tool_use' and block.get('name') == 'submit_result':
                        # Extract message from tool input
                        tool_input = block.get('input', {})
                        return tool_input.get('message', '')

    # Fallback: look for last assistant text message
    for msg in reversed(trajectory):
        if msg.get('role') == 'assistant':
            content = msg.get('content', [])
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if block.get('type') == 'text':
                        return block.get('text', '')

    return None


def _extract_actions_and_pages(trajectory: list[dict]) -> tuple[list[str], list[str]]:
    """Extract executed actions and visited pages from agent's trajectory.

    Extracts pages from:
    1. User messages containing observation with URL (captures all navigation including click-triggered)
    2. browser/goto actions (explicit navigation)

    Returns:
        Tuple of (executed_actions, executed_pages)
    """
    import re

    executed_actions = []
    executed_pages = []
    seen_pages = set()  # Track unique pages to avoid duplicates

    def extract_path(url: str) -> str:
        """Extract path from URL."""
        if '://' in url:
            url = url.split('://', 1)[1]
            if '/' in url:
                return '/' + url.split('/', 1)[1]
            return '/'
        return url if url.startswith('/') else '/' + url

    def add_page(url: str):
        """Add page to list if not duplicate."""
        path = extract_path(url)
        # Remove query string for deduplication but keep in list
        path_base = path.split('?')[0]
        if path_base not in seen_pages:
            seen_pages.add(path_base)
            executed_pages.append(path)

    for msg in trajectory:
        role = msg.get('role', '')

        # Extract URL from user messages (observation)
        # This captures ALL page changes including click-triggered navigation
        if role == 'user':
            content = msg.get('content', '')
            if isinstance(content, str):
                # Match patterns like "URL: http://..." or "[URL] http://..."
                url_match = re.search(r'(?:URL[:\s]*|url[:\s]*)(\S+)', content)
                if url_match:
                    add_page(url_match.group(1))
            elif isinstance(content, list):
                # Handle structured content format
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text = block.get('text', '')
                        url_match = re.search(r'(?:URL[:\s]*|url[:\s]*)(\S+)', text)
                        if url_match:
                            add_page(url_match.group(1))

        # Extract actions from assistant messages
        elif role == 'assistant':
            content = msg.get('content', [])
            tool_calls = msg.get('tool_calls', [])

            # Process tool_calls format (OpenAI style)
            for tool_call in tool_calls:
                func = tool_call.get('function', {})
                tool_name = func.get('name', '')
                executed_actions.append(tool_name)

            # Process content format (Anthropic style)
            if isinstance(content, list):
                for block in content:
                    if block.get('type') == 'tool_use':
                        tool_name = block.get('name', '')
                        executed_actions.append(tool_name)

    return executed_actions, executed_pages


# ============================================================================
# Helper functions for custom web setup and evaluation
# ============================================================================


async def _setup_custom_web_task_impl(
    session_id: str, start_url: str, task_description: str
) -> SetupCustomWebTaskResult:
    """Implementation of custom web task setup to be called via invoke.

    This function will be executed in the runtime environment using invoke.
    Uses browser executor API (handle-based, no direct BROWSER_SESSIONS access).

    Args:
        session_id: Browser session ID
        start_url: Starting URL for the task
        task_description: Task description/goal

    Returns:
        SetupCustomWebTaskResult with success status and goal
    """
    try:
        if start_url:
            from nexus.runtime.executors.browser import browser_goto
            from nexus.runtime.executors.models.browser import GotoRequest

            await browser_goto(GotoRequest(session_id=session_id, url=start_url))

        return SetupCustomWebTaskResult(success=True, goal=task_description)

    except Exception as e:
        logger.error(f'Setup custom web task failed: {e}', exc_info=True)
        return SetupCustomWebTaskResult(success=False, goal='', error_message=str(e))


async def _get_application_state_via_runtime(runtime, session_id: str) -> dict[str, Any]:
    """Get initial and current application state from the /finish page.

    The /finish page provides both initial state and current state,
    allowing us to compare state changes during task execution.

    Args:
        runtime: Runtime instance
        session_id: Browser session ID

    Returns:
        Dictionary with 'initial_state' and 'current_state' keys
    """
    try:
        result = await runtime.invoke(
            _extract_page_state_impl,
            session_id=session_id,
        )
        return result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning(f'Failed to get application state: {e}')
        return {}


async def _extract_page_state_impl(session_id: str) -> dict[str, Any]:
    """Extract initial and current application state by visiting /finish page.

    The /finish page displays both initial state and current state in <pre> tags,
    allowing comparison of state changes during task execution.

    Uses browser executor API (handle-based, no direct BROWSER_SESSIONS access).

    This function:
    1. Saves the current URL
    2. Navigates to /finish page
    3. Extracts initial_state and current_state from <pre> tags

    Returns:
        Dictionary with 'initial_state' and 'current_state' keys
    """
    import asyncio

    from nexus.runtime.executors.browser import (
        browser_eval_js,
        browser_get_url,
        browser_goto,
        browser_wait_for_selector,
    )
    from nexus.runtime.executors.models.browser import (
        EvalJsRequest,
        GetUrlRequest,
        GotoRequest,
        WaitForSelectorRequest,
    )

    try:
        # Save current URL to return later
        url_resp = await browser_get_url(GetUrlRequest(session_id=session_id))
        current_url = url_resp.url

        # Extract base URL for /finish endpoint
        # e.g., http://172.17.0.1:33174/merchant/xxx -> http://172.17.0.1:33174/finish
        if '://' in current_url:
            parts = current_url.split('://', 1)
            protocol = parts[0]
            rest = parts[1]
            host = rest.split('/')[0]
            finish_url = f'{protocol}://{host}/finish'
        else:
            finish_url = '/finish'

        # Navigate to /finish page
        await browser_goto(GotoRequest(session_id=session_id, url=finish_url))
        # Wait for page to fully render - Next.js apps may need more time
        await asyncio.sleep(3)

        # Wait for pre tags to appear (with timeout)
        with contextlib.suppress(Exception):
            await browser_wait_for_selector(
                WaitForSelectorRequest(
                    session_id=session_id,
                    selector='pre',
                    timeout=5000,
                )
            )

        # Extract states from <pre> tags
        # The /finish page has <p> labels ("初始状态" / "当前状态") followed by <pre> tags
        eval_resp = await browser_eval_js(
            EvalJsRequest(
                session_id=session_id,
                expression="""() => {
            const result = {
                initial_state: null,
                current_state: null,
                pre_count: 0,
                page_text: document.body.innerText.substring(0, 500),
                debug_labels: [],
                debug_pre_contents: []
            };

            const pres = document.querySelectorAll('pre');
            result.pre_count = pres.length;

            // Debug: collect first 100 chars of each pre tag
            for (let i = 0; i < pres.length; i++) {
                result.debug_pre_contents.push(pres[i].textContent.substring(0, 100));
            }

            // Find all <p> tags that might be labels
            const allElements = document.querySelectorAll('p, pre');
            let currentLabel = null;

            for (const el of allElements) {
                if (el.tagName === 'P') {
                    const text = el.textContent.trim();
                    result.debug_labels.push(text.substring(0, 50));
                    if (text.includes('初始状态')) {
                        currentLabel = 'initial';
                    } else if (text.includes('当前状态')) {
                        currentLabel = 'current';
                    }
                } else if (el.tagName === 'PRE') {
                    try {
                        const parsed = JSON.parse(el.textContent);
                        if (currentLabel === 'initial') {
                            result.initial_state = parsed;
                            result.found_initial_label = true;
                            currentLabel = null;
                        } else if (currentLabel === 'current') {
                            result.current_state = parsed;
                            result.found_current_label = true;
                            currentLabel = null;
                        }
                    } catch (e) {
                        result.parse_error = e.message;
                    }
                }
            }

            // Fallback: if labels not found, try old logic (first pre = initial, second = current)
            if (result.initial_state === null && result.current_state === null) {
                if (pres.length >= 2) {
                    try {
                        result.initial_state = JSON.parse(pres[0].textContent);
                        result.current_state = JSON.parse(pres[1].textContent);
                        result.fallback_method = 'index_order';
                    } catch (e) {
                        result.parse_error = e.message;
                    }
                } else if (pres.length === 1) {
                    try {
                        result.current_state = JSON.parse(pres[0].textContent);
                        result.fallback_method = 'single_pre';
                    } catch (e) {
                        result.parse_error = e.message;
                    }
                }
            }

            return result;
        }""",
            )
        )
        states = eval_resp.value

        # Debug: log what labels were found
        logger.debug(f'State extraction debug - labels found: {states.get("debug_labels", [])}')
        logger.debug(f'State extraction debug - pre_contents: {states.get("debug_pre_contents", [])}')
        logger.debug(f'State extraction debug - found_initial_label: {states.get("found_initial_label", False)}')
        logger.debug(f'State extraction debug - found_current_label: {states.get("found_current_label", False)}')
        logger.debug(f'State extraction debug - fallback_method: {states.get("fallback_method", "none")}')
        logger.debug(f'State extraction debug - pre_count: {states.get("pre_count", 0)}')
        logger.info(f'State extraction debug - parse_error: {states.get("parse_error", "none")}')
        logger.debug(f'State extraction debug - initial_state type: {type(states.get("initial_state")).__name__}')
        logger.debug(f'State extraction debug - current_state type: {type(states.get("current_state")).__name__}')

        # If no states found from pre tags, try localStorage as fallback
        if not states.get('initial_state') and not states.get('current_state'):
            fallback_resp = await browser_eval_js(
                EvalJsRequest(
                    session_id=session_id,
                    expression="""() => {
                const state = localStorage.getItem('delivery-app-state');
                if (state) {
                    try {
                        return JSON.parse(state);
                    } catch (e) {
                        return null;
                    }
                }
                return null;
            }""",
                )
            )
            if fallback_resp.value:
                states['current_state'] = fallback_resp.value
                states['fallback'] = 'localStorage'

        return states if states else {}

    except Exception as e:
        # Fallback: try to extract state from localStorage on current page
        try:
            fallback_resp = await browser_eval_js(
                EvalJsRequest(
                    session_id=session_id,
                    expression="""() => {
                const state = localStorage.getItem('delivery-app-state');
                if (state) {
                    return {
                        current_state: JSON.parse(state),
                        initial_state: null,
                        fallback: 'localStorage_error'
                    };
                }
                return {error: 'no_state_found'};
            }""",
                )
            )
            return fallback_resp.value or {'error': str(e)}
        except Exception:
            return {'error': str(e)}
