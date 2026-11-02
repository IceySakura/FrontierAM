"""WebArena benchmark task implementation.

WebArena is a benchmark for evaluating AI agents on realistic web navigation tasks.
"""

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

# Default WebArena instance URLs (used when webarena_urls is not provided)
# Note: wikipedia and map services are static read-only services hosted externally
DEFAULT_WEBARENA_URLS = {
    'gitlab': '__GITLAB__',
    'reddit': '__REDDIT__',
    'shopping': '__SHOPPING__',
    'shopping_admin': '__SHOPPING_ADMIN__',
    'wikipedia': 'http://webarena-wikipedia.fengluan.woa.com:8080',  # Static read-only service
    'map': 'http://21.67.190.13:3000',  # Static read-only service (tile server)
}


# ============================================================================
# Helper Functions
# ============================================================================


def _determine_runtime_config(sites: list[str]) -> dict[str, dict[str, Any]]:
    """Determine runtime configuration based on required sites.

    Supports multi-image tasks: when different services need different Docker images
    (e.g., gitlab + reddit), they are grouped into separate named runtimes.

    Services sharing the same image (e.g., shopping + shopping_admin) share a runtime.

    Args:
        sites: List of site names required by the task (e.g., ['shopping', 'gitlab'])

    Returns:
        Dict mapping runtime_name -> runtime config dict:
        {
            'service_gitlab': {
                'image': '...gitlab:latest',
                'exposed_ports': [8023],
                'service_port_map': {'gitlab': 8023},
                'sites': ['gitlab'],
            },
            'service_reddit': {
                'image': '...reddit:latest',
                'exposed_ports': [80],
                'service_port_map': {'reddit': 80},
                'sites': ['reddit'],
            },
        }
        Returns empty dict if no dynamic services needed.
    """
    # Service to image mapping (only dynamic services that need Docker containers)
    service_images = {
        'shopping': 'mirrors.tencent.com/hunyuan_yanguan/webarena.onestopshop:latest',
        'shopping_admin': 'mirrors.tencent.com/hunyuan_yanguan/webarena.cms:latest',
        'reddit': 'mirrors.tencent.com/hunyuan_yanguan/webarena.reddit:latest',
        'gitlab': 'mirrors.tencent.com/hunyuan_yanguan/webarena.gitlab:latest',
    }

    # Default port for each service
    service_ports = {
        'shopping': 80,
        'shopping_admin': 80,
        'reddit': 80,
        'gitlab': 8023,
    }

    # Filter to only dynamic services (exclude static services like wikipedia, map)
    dynamic_sites = [s for s in sites if s in service_images]

    if not dynamic_sites:
        return {}

    # Group services by image to minimize runtimes
    # Services with the same image share a runtime
    image_to_sites: dict[str, list[str]] = {}
    for site in dynamic_sites:
        image = service_images[site]
        if image not in image_to_sites:
            image_to_sites[image] = []
        image_to_sites[image].append(site)

    # Build runtime configs - one per unique image
    runtime_configs: dict[str, dict[str, Any]] = {}

    for image, image_sites in image_to_sites.items():
        # Use first site name as runtime identifier
        runtime_name = f'service_{image_sites[0]}'

        exposed_ports = []
        service_port_map = {}
        for site in image_sites:
            port = service_ports[site]
            if port not in exposed_ports:
                exposed_ports.append(port)
            service_port_map[site] = port

        runtime_configs[runtime_name] = {
            'image': image,
            'exposed_ports': exposed_ports,
            'service_port_map': service_port_map,
            'sites': image_sites,
        }

    return runtime_configs


def _substitute_urls(task_config: dict[str, Any], urls: dict[str, str]) -> dict[str, Any]:
    """Substitute URL placeholders in task_config with actual URLs.

    Recursively traverses the task_config dictionary and replaces URL placeholders
    like __GITLAB__, __SHOPPING__, etc. with actual URLs from the urls mapping.

    Args:
        task_config: Task configuration dictionary (may contain placeholders)
        urls: Mapping from URL keys to actual URLs

    Returns:
        New task_config dictionary with substituted URLs
    """
    import json

    # Convert to JSON string for easy substitution
    config_str = json.dumps(task_config)

    # IMPORTANT: Replace __SHOPPING_ADMIN__ BEFORE __SHOPPING__ to avoid partial match.
    # __SHOPPING_ADMIN__ contains __SHOPPING__ as prefix, so if __SHOPPING__ is replaced
    # first, it would corrupt __SHOPPING_ADMIN__ into "<url>_ADMIN__".
    if 'shopping_admin' in urls:
        # In sandbox, shopping_admin URL is the same base as shopping (same port/image),
        # but the Magento admin panel lives under /admin. Original WebArena had a
        # dedicated port with /admin baked into the URL, so __SHOPPING_ADMIN__ references
        # in test.raw.json assume paths like __SHOPPING_ADMIN__/sales/order/ which should
        # resolve to <base>/admin/sales/order/.
        admin_url = urls['shopping_admin'].rstrip('/')
        if not admin_url.endswith('/admin'):
            admin_url += '/admin'
        config_str = config_str.replace('__SHOPPING_ADMIN__', admin_url)

    # Substitute each URL placeholder
    for pattern, url_key in {
        '__GITLAB__': 'gitlab',
        '__REDDIT__': 'reddit',
        '__SHOPPING__': 'shopping',
        '__WIKIPEDIA__': 'wikipedia',
        '__MAP__': 'map',
    }.items():
        if url_key in urls:
            config_str = config_str.replace(pattern, urls[url_key])

    # Also substitute hardcoded WebArena URLs from test.raw.json
    # These appear in reference_answers.must_include and other eval fields
    for hardcoded_url, url_key in {
        'http://www.reddit.com': 'reddit',
        'http://reddit.com': 'reddit',
        'http://www.gitlab.com': 'gitlab',
        'http://gitlab.com': 'gitlab',
    }.items():
        if url_key in urls:
            config_str = config_str.replace(hardcoded_url, urls[url_key])

    return json.loads(config_str)


# ============================================================================
# Helper Models for invoke
# ============================================================================


class SetupWebArenaTaskResult(BaseModel):
    """Result from setup_webarena_task function."""

    success: bool
    goal: str
    error_message: str | None = None


class EvaluateWebArenaResult(BaseModel):
    """Result from evaluate_webarena function."""

    score: float
    success: bool
    error_message: str | None = None


class WebArenaTaskInput(TaskRunInput, WithAgentRunner):
    """WebArena task input.

    All task data is embedded directly in the input, no external loader required.

    Attributes:
        task_id: WebArena task ID (from the benchmark dataset)
        task_config: Full task configuration dictionary containing:
            - sites: List of sites to authenticate
            - start_url: Starting URL(s) for the task (may contain __PLACEHOLDER__ URLs)
            - intent: Task goal/instruction
            - eval: Evaluation configuration
            - geolocation: Optional geolocation settings
            - require_login: Whether authentication is required
            - storage_state: Path to auth state file (if needed)
        webarena_urls: Optional WebArena instance URLs mapping (gitlab, reddit, shopping, etc.)
                       Used to substitute __PLACEHOLDER__ URLs in task_config at runtime.
                       If not provided, uses default placeholder URLs.
        runtime_name: Runtime name to use
        timeout: Timeout in seconds
        with_na_hint: Whether to include N/A hint in instruction
        with_homepage_hint: Whether to include homepage hint
    """

    task_id: int
    task_config: dict[str, Any]
    webarena_urls: dict[str, str] | None = None

    # Runtime configuration
    runtime_name: str = 'main'
    timeout: int = 600  # 10 minutes per task

    # Task options
    with_na_hint: bool = False
    with_homepage_hint: bool = False


class WebArenaTaskOutput(TaskRunOutput, WithScore):
    """WebArena task output.

    Attributes:
        task_id: Task identifier
        score: Score (0.0 or 1.0)
        is_success: Whether task was completed successfully
        agent_answer: The agent's final answer
        error_message: Error message if task failed
        trajectory_length: Number of agent iterations
        evaluation_details: Additional evaluation information
    """

    task_id: int
    score: float
    is_success: bool = False
    agent_answer: str | None = None
    error_message: str | None = None
    trajectory_length: int = 0
    evaluation_details: dict[str, Any] = Field(default_factory=dict)


@task_registry.register('webarena')
async def run_webarena_task(task_input: WebArenaTaskInput) -> WebArenaTaskOutput:
    """Run WebArena benchmark task.

    Supports multi-image tasks: when a task requires services with different Docker images
    (e.g., gitlab + reddit), separate runtimes are created for each image. The browser
    session runs on a dedicated 'main' runtime that accesses services via external URLs.

    Workflow:
    1. Determine required services and group by image
    2. Configure and acquire runtimes (one per unique image + one for browser)
    3. Start services in their respective runtimes
    4. Substitute URL placeholders with actual service URLs
    5. Create browser session on the main runtime
    6. Setup task (authentication, navigation)
    7. Run agent with browser tools
    8. Extract answer from trajectory
    9. Evaluate task completion
    10. Cleanup browser session
    """
    logger.debug(f'Running WebArena task {task_input.task_id}')

    ctx = require_context()
    assert ctx.runtime_provider is not None, 'webarena requires runtime provider'

    # Determine runtime configs based on task_config sites
    sites = task_input.task_config.get('sites', [])
    service_runtime_configs = _determine_runtime_config(sites)

    # Determine if we need multiple runtimes (multi-image scenario)
    is_multi_runtime = len(service_runtime_configs) > 1

    if is_multi_runtime:
        # Multi-image: configure separate runtimes for each service group,
        # plus a dedicated 'main' runtime for the browser
        logger.debug(f'Multi-image task detected: {len(service_runtime_configs)} service runtimes needed')

        for rt_name, rt_config in service_runtime_configs.items():
            ctx.runtime_provider.configure(
                rt_name,
                image=rt_config['image'],
                exposed_ports=rt_config['exposed_ports'],
            )

        # Browser runtime uses a lightweight image (no services)
        default_image = 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest'
        ctx.runtime_provider.configure('main', image=default_image)

    elif service_runtime_configs:
        # Single-image: use 'main' runtime for both services and browser (original behavior)
        rt_config = next(iter(service_runtime_configs.values()))
        ctx.runtime_provider.configure(
            'main',
            image=rt_config['image'],
            exposed_ports=rt_config['exposed_ports'],
        )
    else:
        # No dynamic services - just need browser runtime
        default_image = 'mirrors.tencent.com/hunyuan_yanguan/ubuntu:24.04'
        ctx.runtime_provider.configure('main', image=default_image)

    # Acquire the main (browser) runtime
    browser_runtime = await ctx.get_runtime('main')

    browser_session_id = None

    try:
        # Step 1: Start required WebArena services
        service_urls = {}
        if service_runtime_configs:
            from nexus.extensions.tasks.webarena.service_setup import setup_webarena_services

            if is_multi_runtime:
                # Multi-image: acquire each service runtime and start services on it
                for rt_name, rt_config in service_runtime_configs.items():
                    logger.debug(f'Acquiring service runtime {rt_name} (image={rt_config["image"]})...')
                    service_runtime = await ctx.get_runtime(rt_name)

                    logger.debug(f'Starting services {rt_config["sites"]} on runtime {rt_name}...')
                    urls = await setup_webarena_services(
                        service_runtime, task_input.task_config, rt_config['service_port_map']
                    )
                    service_urls.update(urls)
                    logger.debug(f'Services on {rt_name} started: {urls}')
            else:
                # Single-image: start all services on the main runtime
                rt_config = next(iter(service_runtime_configs.values()))
                logger.debug('Starting WebArena services on main runtime...')
                service_urls = await setup_webarena_services(
                    browser_runtime, task_input.task_config, rt_config['service_port_map']
                )
                logger.debug(f'Services started: {service_urls}')
        else:
            logger.debug('No dynamic services needed, using static URLs only')

        # Step 2: Build complete URL mapping (services + static URLs)
        webarena_urls = task_input.webarena_urls or DEFAULT_WEBARENA_URLS.copy()
        webarena_urls.update(service_urls)  # Override with actual service URLs

        # Step 3: Substitute URL placeholders in task_config
        substituted_task_config = _substitute_urls(task_input.task_config, webarena_urls)
        logger.debug(f'WebArena URLs: {webarena_urls}')
        logger.debug(f'Substituted start_url: {substituted_task_config.get("start_url")}')

        # Step 4: Create browser session on the main runtime
        logger.debug('Creating browser session...')
        create_result = await browser_runtime.create_browser_session(
            headless=True, viewport_width=1280, viewport_height=720, timeout=10000
        )
        browser_session_id = create_result.session_id
        logger.debug(f'Browser session created: {browser_session_id}')
        ctx.tool_state['browser_session_id'] = browser_session_id
        ctx.tool_state['webarena_urls'] = webarena_urls

        # Step 5: Setup WebArena task (authentication, navigation)
        logger.debug('Setting up WebArena task...')
        setup_result = await browser_runtime.invoke(
            _setup_webarena_task_impl,
            session_id=browser_session_id,
            task_config=substituted_task_config,
            urls=webarena_urls,
        )

        if not setup_result.success:
            raise RuntimeError(f'Failed to setup WebArena task: {setup_result.error_message}')

        goal = setup_result.goal

        # Add optional hints
        if task_input.with_homepage_hint:
            home_url = webarena_urls.get('homepage', 'http://homepage.example.com')
            goal += f'\n\n(Note: if you want to visit other websites, check out the homepage at {home_url}. It has a list of websites you can visit. {home_url}/password.html lists all the account name and password for the websites. You can use them to log in to the websites.)'

        if task_input.with_na_hint:
            goal += '\n\nIf you believe the task is impossible to complete, provide the answer "N/A".'

        logger.debug(f'Task goal: {goal[:200]}...')

        # Step 6: Run agent with browser tools
        logger.debug('Running agent...')
        agent_output = await task_input.run_agent(
            user_prompt=goal,
            timeout=task_input.timeout,
        )

        # Step 7: Extract answer from trajectory
        if not isinstance(agent_output, WithMainTrajectory):
            raise RuntimeError('WebArena task requires agent output to be a subclass of WithMainTrajectory')

        agent_answer = _extract_answer(agent_output.main_trajectory)
        trajectory_length = len(agent_output.main_trajectory)

        logger.debug(f'Agent answer: {agent_answer}')

        # Step 8: Evaluate task completion
        logger.debug('Evaluating task completion...')

        eval_types = substituted_task_config.get('eval', {}).get('eval_types', [])
        if not eval_types:
            raise RuntimeError('No eval_types specified in task config')

        from nexus.extensions.tasks.webarena.eval import eval_string_match, eval_url_match

        score = 1.0
        for eval_type in eval_types:
            if eval_type == 'string_match':
                cur_score = await eval_string_match(substituted_task_config, agent_answer or '')
                logger.info(f'string_match score: {cur_score}')

            elif eval_type == 'url_match':
                observation = await browser_runtime.browser_get_observation(session_id=browser_session_id)
                current_url = observation.url
                cur_score = await eval_url_match(substituted_task_config, current_url)
                logger.info(f'url_match score: {cur_score} (url: {current_url})')

            elif eval_type == 'program_html':
                eval_result = await browser_runtime.invoke(
                    _evaluate_program_html_impl,
                    session_id=browser_session_id,
                    task_config=substituted_task_config,
                )
                if eval_result.error_message:
                    logger.warning(f'program_html evaluation error: {eval_result.error_message}')
                cur_score = eval_result.score
                logger.info(f'program_html score: {cur_score}')

            else:
                raise RuntimeError(f'Unknown evaluator type: {eval_type}')

            score *= cur_score

        is_success = score == 1.0

        logger.info(f'Task {task_input.task_id} - Score: {score}, Success: {is_success}')

        return WebArenaTaskOutput(
            task_id=task_input.task_id,
            score=score,
            is_success=is_success,
            agent_answer=agent_answer,
            trajectory_length=trajectory_length,
            evaluation_details={'eval_types': eval_types, 'final_score': score},
        )

    except Exception as e:
        logger.error(f'Task {task_input.task_id} failed: {e}', exc_info=True)
        return WebArenaTaskOutput(
            task_id=task_input.task_id,
            score=0.0,
            is_success=False,
            agent_answer=None,
            error_message=str(e),
        )

    finally:
        # Cleanup - destroy browser session
        if browser_session_id:
            try:
                logger.debug(f'Destroying browser session: {browser_session_id}')
                await browser_runtime.destroy_browser_session(session_id=browser_session_id)
            except Exception as e:
                logger.error(f'Failed to destroy browser session: {e}')


# def _extract_answer(trajectory: list[dict]) -> str | None:
#     """Extract the final answer from agent's trajectory.

#     Looks for submit_result tool calls or the last assistant message.
#     """
#     # First, look for submit_result tool calls
#     for msg in reversed(trajectory):
#         if msg.get('role') == 'assistant':
#             content = msg.get('content', [])
#             if isinstance(content, list):
#                 for block in content:
#                     if block.get('type') == 'tool_use' and block.get('name') == 'submit_result':
#                         # Extract message from tool input
#                         tool_input = block.get('input', {})
#                         return tool_input.get('message', '')

#     # Fallback: look for last assistant text message
#     for msg in reversed(trajectory):
#         if msg.get('role') == 'assistant':
#             content = msg.get('content', [])
#             if isinstance(content, str):
#                 return content
#             if isinstance(content, list):
#                 for block in content:
#                     if block.get('type') == 'text':
#                         return block.get('text', '')


#     return None
def _extract_answer(trajectory: list[dict]) -> str | None:
    """Extract the final answer from agent's trajectory.

    Looks for submit_result tool calls or the last assistant message.
    Supports both OpenAI format (tool_calls) and Anthropic format (content with tool_use).
    """
    import json as json_module

    # First, look for submit_result tool calls
    for msg in reversed(trajectory):
        if msg.get('role') == 'assistant':
            # OpenAI format: tool_calls array
            tool_calls = msg.get('tool_calls', [])
            for tool_call in tool_calls:
                func = tool_call.get('function', {})
                if func.get('name') == 'submit_result':
                    # Parse arguments JSON string
                    args_str = func.get('arguments', '{}')
                    try:
                        args = json_module.loads(args_str)
                        return args.get('message', '')
                    except json_module.JSONDecodeError:
                        continue

            # Anthropic format: content array with tool_use blocks
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


# ============================================================================
# Helper functions for WebArena setup and evaluation
# ============================================================================


async def _setup_webarena_task_impl(
    session_id: str, task_config: dict[str, Any], urls: dict[str, str]
) -> SetupWebArenaTaskResult:
    """Implementation of WebArena task setup to be called via invoke.

    This function will be executed in the runtime environment using invoke.
    Uses browser executor API (handle-based, no direct BROWSER_SESSIONS access).

    Args:
        session_id: Browser session ID
        task_config: WebArena task configuration
        urls: WebArena instance URLs

    Returns:
        SetupWebArenaTaskResult with success status and goal
    """
    try:
        from nexus.extensions.tasks.webarena.auth import authenticate_site
        from nexus.runtime.executors.browser import (
            browser_get_url,
            browser_goto,
            browser_new_tab,
            browser_set_geolocation,
        )
        from nexus.runtime.executors.models.browser import (
            GetUrlRequest,
            GotoRequest,
            NewTabRequest,
            SetGeolocationRequest,
        )

        # Authenticate for required sites
        sites = task_config.get('sites', [])
        if len(sites) > 1:
            logger.debug(f'[Setup] Cross-site task detected, authenticating to {len(sites)} sites: {sites}')

        for i, site in enumerate(sites):
            logger.debug(f'[Setup] Authenticating site {i + 1}/{len(sites)}: {site}')
            try:
                await authenticate_site(session_id, site, urls)
            except Exception as auth_err:
                error_msg = f'Failed to setup WebArena task: {auth_err}'
                logger.error(f'[Setup] {error_msg}')
                return SetupWebArenaTaskResult(success=False, goal='', error_message=error_msg)

        # Set geolocation if specified
        if 'geolocation' in task_config:
            geo = task_config['geolocation']
            await browser_set_geolocation(
                SetGeolocationRequest(
                    session_id=session_id,
                    latitude=geo['latitude'],
                    longitude=geo['longitude'],
                    accuracy=geo.get('accuracy', 0),
                )
            )

        # Navigate to start URL(s)
        start_url = task_config.get('start_url', '')
        logger.debug(f'[Setup] Navigating to start_url: {start_url}')
        if start_url:
            start_urls = start_url.split(' |AND| ')
            for i, url in enumerate(start_urls):
                logger.debug(f'[Setup] goto: {url}')
                await browser_goto(GotoRequest(session_id=session_id, url=url))
                url_resp = await browser_get_url(GetUrlRequest(session_id=session_id))
                logger.debug(f'[Setup] Current page URL after goto: {url_resp.url}')
                if i < len(start_urls) - 1:
                    # Open new tab for additional URLs
                    await browser_new_tab(NewTabRequest(session_id=session_id))

        # Get goal
        goal = task_config.get('intent', '')

        return SetupWebArenaTaskResult(success=True, goal=goal)

    except Exception as e:
        logger.error(f'Setup WebArena task failed: {e}', exc_info=True)
        return SetupWebArenaTaskResult(success=False, goal='', error_message=str(e))


async def _evaluate_program_html_impl(session_id: str, task_config: dict[str, Any]) -> EvaluateWebArenaResult:
    """Evaluate program_html in runtime (requires browser for navigation and JS).

    This function will be executed in the runtime environment using invoke.
    Only handles program_html evaluation type.

    Args:
        session_id: Browser session ID
        task_config: WebArena task configuration

    Returns:
        EvaluateWebArenaResult with score and success status
    """
    try:
        from nexus.extensions.tasks.webarena.eval import eval_program_html

        score = await eval_program_html(task_config, session_id)

        return EvaluateWebArenaResult(score=score, success=score > 0)

    except Exception as e:
        logger.error(f'Evaluate program_html failed: {e}', exc_info=True)
        return EvaluateWebArenaResult(score=0.0, success=False, error_message=str(e))
