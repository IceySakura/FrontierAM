"""WebArena-Verified benchmark task implementation.

WebArena-Verified is a verified version of the WebArena benchmark with:
- Deterministic evaluation (no LLM-as-judge)
- Type-aware value normalization
- Structured comparison for nested objects/arrays
- 812 verified tasks with human-audited reference answers
"""

import json
import logging
import os
import re

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
    'wikipedia': 'http://webarena.eval.prod.polaris.polaris:8888',  # Static read-only service
    'map': 'http://webarena.eval.prod.polaris.polaris:3000',  # Static read-only service (tile server)
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
        Dict mapping runtime_name -> runtime config dict
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
    image_to_sites: dict[str, list[str]] = {}
    for site in dynamic_sites:
        image = service_images[site]
        if image not in image_to_sites:
            image_to_sites[image] = []
        image_to_sites[image].append(site)

    # Build runtime configs - one per unique image
    runtime_configs: dict[str, dict[str, Any]] = {}

    for image, image_sites in image_to_sites.items():
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


class NetworkEvent:
    """Represents a network event from HAR file.

    Contains HTTP method, URL, and POST data for evaluating mutate tasks.
    """

    def __init__(self, method: str, url: str, post_data: dict | str | None = None):
        self.method = method.upper()
        self.url = url
        self.post_data = post_data

    def __repr__(self) -> str:
        return f'NetworkEvent({self.method} {self.url})'


def _parse_har_for_network_events(har_path: str) -> list[NetworkEvent]:
    """Parse HAR file to extract network events with full details.

    WebArena-Verified uses network events to verify agent actions.
    This function extracts complete network event information including:
    - HTTP method (GET/POST/PUT/DELETE/PATCH)
    - URL
    - POST data (request body)

    This is essential for evaluating "mutate" type tasks where the agent
    needs to perform actions like creating, updating, or deleting resources.

    Args:
        har_path: Path to the HAR file

    Returns:
        List of NetworkEvent objects with method, url, and post_data
    """
    if not har_path or not os.path.exists(har_path):
        logger.warning(f'HAR file not found: {har_path}')
        return []

    try:
        with open(har_path, encoding='utf-8') as f:
            har_data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f'Failed to parse HAR file {har_path}: {e}')
        return []

    network_events: list[NetworkEvent] = []
    entries = har_data.get('log', {}).get('entries', [])

    for entry in entries:
        request = entry.get('request', {})
        url = request.get('url', '')
        method = request.get('method', 'GET').upper()

        # Extract headers for request type detection
        headers = {h.get('name', '').lower(): h.get('value', '') for h in request.get('headers', [])}

        # Extract POST data if present
        post_data: dict | str | None = None
        post_data_raw = request.get('postData', {})
        if post_data_raw:
            mime_type = post_data_raw.get('mimeType', '')
            text = post_data_raw.get('text', '')

            # Try to parse as JSON if it's JSON content type
            if text:
                if 'json' in mime_type.lower():
                    try:
                        post_data = json.loads(text)
                    except json.JSONDecodeError:
                        post_data = text
                elif 'multipart' in mime_type.lower():
                    # Parse multipart/form-data using boundary
                    boundary_match = re.search(r'boundary=([^\s;]+)', mime_type)
                    if boundary_match:
                        boundary = '--' + boundary_match.group(1)
                        parts = text.split(boundary)
                        result = {}
                        for part in parts:
                            name_match = re.search(
                                r'Content-Disposition:\s*form-data;\s*name="([^"]+)"', part, re.IGNORECASE
                            )
                            if name_match:
                                field_name = name_match.group(1)
                                for sep in ['\r\n\r\n', '\n\n']:
                                    idx = part.find(sep)
                                    if idx != -1:
                                        value = part[idx + len(sep) :].strip()
                                        if value.endswith('--'):
                                            value = value[:-2].strip()
                                        result[field_name] = value
                                        break
                        post_data = result if result else text
                    else:
                        post_data = text
                elif 'form' in mime_type.lower():
                    # Parse URL-encoded form data: param1=value1&param2=value2
                    try:
                        from urllib.parse import parse_qs

                        parsed = parse_qs(text, keep_blank_values=True)
                        # Simplify single-value params
                        post_data = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
                    except Exception:
                        post_data = text
                else:
                    post_data = text

            # Also check params array (alternative HAR format)
            params = post_data_raw.get('params', [])
            if params and not post_data:
                post_data = {p.get('name', ''): p.get('value', '') for p in params if p.get('name')}

        # Check request type to determine what events to capture
        sec_fetch_dest = headers.get('sec-fetch-dest', '')
        sec_fetch_mode = headers.get('sec-fetch-mode', '')
        headers.get('content-type', '')

        # Response content type
        response = entry.get('response', {})
        status = response.get('status', 0)
        response_headers = {h.get('name', '').lower(): h.get('value', '') for h in response.get('headers', [])}
        response_content_type = response_headers.get('content-type', '')

        # Determine if this is a relevant event:
        # 1. Navigation events (document loads)
        # 2. API calls (XHR/fetch) - important for mutate operations
        # 3. Form submissions
        # 4. Any mutation (POST/PUT/DELETE/PATCH)
        is_navigation = sec_fetch_dest == 'document' and sec_fetch_mode == 'navigate'
        is_api_call = sec_fetch_dest in ('empty', '') and sec_fetch_mode in ('cors', 'same-origin', 'no-cors')
        is_mutation = method in ('POST', 'PUT', 'DELETE', 'PATCH')
        is_form_submit = sec_fetch_dest == 'document' and method == 'POST'

        # Check if response is HTML (likely a page navigation)
        is_html_response = 'text/html' in response_content_type.lower()

        # Check response status
        is_successful = 200 <= status < 400

        # Skip static resources
        is_static_resource = any(
            ext in url.lower()
            for ext in [
                '.js',
                '.css',
                '.png',
                '.jpg',
                '.jpeg',
                '.gif',
                '.svg',
                '.woff',
                '.woff2',
                '.ttf',
                '.ico',
                '.webp',
                '.chunk',
            ]
        )

        # Include event if it's relevant
        should_include = (
            not is_static_resource
            and url
            and is_successful
            and (
                # Navigation events (with sec-fetch headers)
                is_navigation
                or
                # API calls that are mutations
                (is_api_call and is_mutation)
                or
                # Form submissions
                is_form_submit
                or
                # Any mutation that succeeded (important: catch-all for all POST/PUT/DELETE/PATCH)
                is_mutation
                or
                # Navigation by GET with sec-fetch headers
                (method == 'GET' and (sec_fetch_dest == 'document' or sec_fetch_mode == 'navigate'))
                or
                # GET request returning HTML (fallback when sec-fetch headers are missing)
                (method == 'GET' and is_html_response)
            )
        )

        if should_include and url:
            event = NetworkEvent(method=method, url=url, post_data=post_data)
            network_events.append(event)
            logger.debug(f'HAR network event: {event}')

    logger.debug(f'Parsed HAR file: {len(network_events)} network events found')
    return network_events


def _parse_har_for_visited_urls(har_path: str) -> list[str]:
    """Parse HAR file to extract navigation URLs (legacy interface).

    This is a simplified interface that returns only URLs.
    For full network event data including POST bodies, use _parse_har_for_network_events().

    Args:
        har_path: Path to the HAR file

    Returns:
        List of visited URLs
    """
    events = _parse_har_for_network_events(har_path)
    return [event.url for event in events]


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
    # Convert to JSON string for easy substitution
    config_str = json.dumps(task_config)

    # IMPORTANT: Replace __SHOPPING_ADMIN__ BEFORE __SHOPPING__ to avoid partial match
    if 'shopping_admin' in urls:
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
# Helper Models for call_function
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
    details: dict[str, Any] = Field(default_factory=dict)


class WebArenaVerifiedTaskInput(TaskRunInput, WithAgentRunner):
    """WebArena-Verified task input.

    All task data is embedded directly in the input, no external loader required.

    Attributes:
        task_id: WebArena task ID (from the benchmark dataset)
        task_config: Full task configuration dictionary containing:
            - sites: List of sites to authenticate
            - start_url/start_urls: Starting URL(s) for the task
            - intent: Task goal/instruction
            - eval: Evaluation configuration (verified format)
        webarena_urls: Optional WebArena instance URLs mapping
        runtime_name: Runtime name to use
        timeout: Timeout in seconds
        with_na_hint: Whether to include N/A hint in instruction
        with_homepage_hint: Whether to include homepage hint
        use_deterministic_eval: Whether to use deterministic evaluation (default: True)
        record_har: Whether to record HAR (network trace) file
        har_output_dir: Directory to save HAR files (default: outputs/har)
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

    # Evaluation options
    use_deterministic_eval: bool = True  # Use verified deterministic evaluation

    # HAR recording options
    record_har: bool = False  # Whether to record network trace (HAR file)
    har_output_dir: str = 'outputs/har'  # Directory to save HAR files


class WebArenaVerifiedTaskOutput(TaskRunOutput, WithScore):
    """WebArena-Verified task output.

    Attributes:
        task_id: Task identifier
        score: Score (0.0 or 1.0)
        is_success: Whether task was completed successfully
        agent_answer: The agent's final answer
        error_message: Error message if task failed
        trajectory_length: Number of agent iterations
        evaluation_details: Additional evaluation information
        har_path: Path to the recorded HAR file (if HAR recording was enabled)
    """

    task_id: int
    score: float
    is_success: bool = False
    agent_answer: str | None = None
    error_message: str | None = None
    trajectory_length: int = 0
    evaluation_details: dict[str, Any] = Field(default_factory=dict)
    har_path: str | None = None  # Path to recorded HAR file


@task_registry.register('webarena_verified')
async def run_webarena_verified_task(task_input: WebArenaVerifiedTaskInput) -> WebArenaVerifiedTaskOutput:
    """Run WebArena-Verified benchmark task.

    This task uses deterministic evaluation instead of LLM-as-judge:
    - Type-aware value normalization (numbers, currency, dates, etc.)
    - Structural comparison for nested objects and arrays
    - Alternative value support without LLM judgment

    Workflow:
    1. Determine required services and group by image
    2. Configure and acquire runtimes
    3. Start services in their respective runtimes
    4. Substitute URL placeholders with actual service URLs
    5. Create browser session on the main runtime
    6. Setup task (authentication, navigation)
    7. Run agent with browser tools
    8. Extract answer from trajectory
    9. Evaluate task completion using deterministic evaluators
    10. Cleanup browser session
    """
    logger.debug(f'Running WebArena-Verified task {task_input.task_id}')

    ctx = require_context()
    assert ctx.runtime_provider is not None, 'webarena_verified requires runtime provider'

    # Determine runtime configs based on task_config sites
    sites = task_input.task_config.get('sites', [])
    service_runtime_configs = _determine_runtime_config(sites)

    # Determine if we need multiple runtimes (multi-image scenario)
    is_multi_runtime = len(service_runtime_configs) > 1

    if is_multi_runtime:
        logger.debug(f'Multi-image task detected: {len(service_runtime_configs)} service runtimes needed')

        for rt_name, rt_config in service_runtime_configs.items():
            ctx.runtime_provider.configure(
                rt_name,
                image=rt_config['image'],
                exposed_ports=rt_config['exposed_ports'],
            )

        default_image = 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest'
        ctx.runtime_provider.configure('main', image=default_image)

    elif service_runtime_configs:
        rt_config = next(iter(service_runtime_configs.values()))
        ctx.runtime_provider.configure(
            'main',
            image=rt_config['image'],
            exposed_ports=rt_config['exposed_ports'],
        )
    else:
        default_image = 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest'
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
        webarena_urls.update(service_urls)

        # Step 3: Substitute URL placeholders in task_config
        substituted_task_config = _substitute_urls(task_input.task_config, webarena_urls)
        logger.debug(f'WebArena URLs: {webarena_urls}')

        # Handle both start_url and start_urls formats
        start_url = substituted_task_config.get('start_url') or substituted_task_config.get('start_urls', [''])[0]
        logger.debug(f'Substituted start_url: {start_url}')

        # Step 4: Create browser session on the main runtime
        logger.debug('Creating browser session...')

        # Setup HAR recording path if enabled
        # Use run_id to create a unique subdirectory for complete isolation in parallel execution
        har_path = None
        har_filename = None
        har_subdir = None
        if task_input.record_har:
            # Prefer run_id for uniqueness, fallback to task_id
            unique_id = ctx.run_id or f'task_{task_input.task_id}'
            # Create a unique subdirectory for this run to ensure complete isolation
            har_subdir = os.path.join(task_input.har_output_dir, unique_id)
            os.makedirs(har_subdir, exist_ok=True)
            har_filename = 'network.har'  # Simple fixed name since we have unique dir
            har_path = os.path.join(har_subdir, har_filename)
            logger.debug(f'HAR recording enabled, will save to: {har_path}')

        # Build browser session arguments
        browser_session_args: dict[str, Any] = {
            'headless': True,
            'viewport_width': 1280,
            'viewport_height': 720,
            'timeout': 10000,
        }
        # Only add HAR recording params if enabled (requires updated runtime image)
        if task_input.record_har and har_path:
            browser_session_args['record_har'] = True
            browser_session_args['har_path'] = har_path

        create_result = await browser_runtime.create_browser_session(**browser_session_args)
        browser_session_id = create_result.session_id
        logger.debug(f'Browser session created: {browser_session_id}')
        ctx.tool_state['browser_session_id'] = browser_session_id
        ctx.tool_state['webarena_urls'] = webarena_urls

        # Step 5: Setup WebArena task (authentication, navigation)
        logger.debug('Setting up WebArena-Verified task...')
        setup_result = await browser_runtime.invoke(
            _setup_webarena_verified_task_impl,
            session_id=browser_session_id,
            task_config=substituted_task_config,
            urls=webarena_urls,
        )

        if not setup_result.success:
            raise RuntimeError(f'Failed to setup WebArena-Verified task: {setup_result.error_message}')

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
            raise RuntimeError('WebArena-Verified task requires agent output to be a subclass of WithMainTrajectory')

        agent_answer = _extract_answer(agent_output.main_trajectory)
        trajectory_length = len(agent_output.main_trajectory)

        logger.debug(f'Agent answer: {agent_answer}')

        # Step 8: Evaluate task completion using deterministic evaluators
        logger.debug('Evaluating task completion (deterministic)...')

        # Extract network events from HAR file if HAR recording is enabled
        network_events: list[NetworkEvent] = []
        if task_input.record_har and har_path:
            # HAR is written on context.close(), so we need to close the session first to get the HAR
            # But we also need the current URL before closing, so get it now
            try:
                observation = await browser_runtime.browser_get_observation(session_id=browser_session_id)
                current_url = observation.url
                # Add current URL as a navigation event
                network_events.append(NetworkEvent(method='GET', url=current_url))
                logger.debug(f'Current URL before evaluation: {current_url}')
            except Exception as e:
                logger.warning(f'Failed to get current URL: {e}')

            # Close browser session to flush HAR file (HAR is written to runtime's filesystem)
            logger.debug('Closing browser session to flush HAR file...')
            remote_har_path = None
            try:
                destroy_result = await browser_runtime.destroy_browser_session(session_id=browser_session_id)
                if destroy_result.har_path:
                    remote_har_path = destroy_result.har_path
                    logger.debug(f'HAR file saved on runtime to: {remote_har_path}')
                browser_session_id = None  # Mark as closed to avoid double-close in finally
            except Exception as e:
                logger.error(f'Failed to destroy browser session for HAR flush: {e}')

            # Download HAR file from remote runtime to local filesystem
            if remote_har_path:
                try:
                    from nexus.runtime.executors.file import FileItem

                    logger.debug(f'Downloading HAR file from runtime: {remote_har_path}')
                    download_result = await browser_runtime.download_file(
                        files=[FileItem(path=remote_har_path, encoding='utf8')]
                    )

                    if download_result.files:
                        # Save HAR content to local file in the unique subdirectory
                        unique_id = ctx.run_id or f'task_{task_input.task_id}'
                        # Use har_subdir if already set, otherwise create it
                        if not har_subdir:
                            har_subdir = os.path.join(task_input.har_output_dir, unique_id)
                        os.makedirs(har_subdir, exist_ok=True)
                        # Use har_filename if already set, otherwise use default
                        if not har_filename:
                            har_filename = 'network.har'
                        local_har_path = os.path.join(har_subdir, har_filename)
                        har_content = download_result.files[0].content
                        if har_content is None:
                            raise ValueError('HAR file content is empty')
                        with open(local_har_path, 'w', encoding='utf-8') as f:
                            f.write(har_content)
                        har_path = local_har_path
                        logger.debug(f'HAR file downloaded to local: {har_path}')
                    else:
                        logger.warning(f'Failed to download HAR file: {download_result.errors}')
                except Exception as e:
                    logger.error(f'Failed to download HAR file from runtime: {e}')

            # Parse HAR file to extract full network events
            har_network_events = _parse_har_for_network_events(har_path)
            if har_network_events:
                network_events = har_network_events  # Use HAR events which are more complete
                logger.debug(f'Extracted {len(network_events)} network events from HAR')

        if task_input.use_deterministic_eval:
            # Use new deterministic evaluation
            from nexus.extensions.tasks.webarena_verified.evaluator import evaluate_task_deterministic

            score, eval_details = await evaluate_task_deterministic(
                task_config=substituted_task_config,
                agent_answer=agent_answer,
                page=None,  # TODO: Pass page for HTML evaluation if needed
                network_events=network_events,  # Pass full network events for NetworkEventEvaluator
            )
        else:
            # Fall back to legacy evaluation
            score, eval_details = await _legacy_evaluate(
                substituted_task_config, agent_answer, browser_runtime, browser_session_id or ''
            )

        is_success = score == 1.0

        logger.info(f'Task {task_input.task_id} - Score: {score}, Success: {is_success}')

        return WebArenaVerifiedTaskOutput(
            task_id=task_input.task_id,
            score=score,
            is_success=is_success,
            agent_answer=agent_answer,
            trajectory_length=trajectory_length,
            evaluation_details=eval_details,
            har_path=har_path,
        )

    except Exception as e:
        logger.error(f'Task {task_input.task_id} failed: {e}', exc_info=True)
        return WebArenaVerifiedTaskOutput(
            task_id=task_input.task_id,
            score=0.0,
            is_success=False,
            agent_answer=None,
            error_message=str(e),
            har_path=har_path if 'har_path' in locals() else None,
        )

    finally:
        # Cleanup - destroy browser session
        # Note: HAR file is automatically written when context.close() is called
        if browser_session_id:
            try:
                logger.debug(f'Destroying browser session: {browser_session_id}')
                destroy_result = await browser_runtime.destroy_browser_session(session_id=browser_session_id)
                if destroy_result.har_path:
                    logger.info(f'HAR file saved to: {destroy_result.har_path}')
            except Exception as e:
                logger.error(f'Failed to destroy browser session: {e}')


async def _legacy_evaluate(
    task_config: dict,
    agent_answer: str | None,
    browser_runtime: Any,
    browser_session_id: str,
) -> tuple[float, dict]:
    """Legacy evaluation using original WebArena evaluators."""
    eval_config = task_config.get('eval', {})
    eval_types = eval_config.get('eval_types', [])

    if not eval_types:
        # Try new-style eval format
        if isinstance(eval_config, list):
            # Use deterministic evaluation for list format
            from nexus.extensions.tasks.webarena_verified.evaluator import evaluate_task_deterministic

            return await evaluate_task_deterministic(task_config, agent_answer)
        raise RuntimeError('No eval_types specified in task config')

    from nexus.extensions.tasks.webarena.eval import eval_string_match, eval_url_match

    score = 1.0
    details = {'eval_types': eval_types}

    for eval_type in eval_types:
        if eval_type == 'string_match':
            cur_score = await eval_string_match(task_config, agent_answer or '')
            logger.info(f'string_match score: {cur_score}')

        elif eval_type == 'url_match':
            observation = await browser_runtime.browser_get_observation(session_id=browser_session_id)
            current_url = observation.url
            cur_score = await eval_url_match(task_config, current_url)
            logger.info(f'url_match score: {cur_score} (url: {current_url})')

        elif eval_type == 'program_html':
            # Import here to avoid circular imports
            from nexus.extensions.tasks.webarena.task import _evaluate_program_html_impl

            eval_result = await browser_runtime.call_function(
                _evaluate_program_html_impl,
                session_id=browser_session_id,
                task_config=task_config,
            )
            if eval_result.error_message:
                logger.warning(f'program_html evaluation error: {eval_result.error_message}')
            cur_score = eval_result.score
            logger.info(f'program_html score: {cur_score}')

        else:
            raise RuntimeError(f'Unknown evaluator type: {eval_type}')

        score *= cur_score

    details['final_score'] = score
    return score, details


def _extract_answer(trajectory: list[dict]) -> str | None:
    """Extract the final answer from agent's trajectory.

    Looks for submit_result tool calls or the last assistant message.
    Supports both OpenAI format (tool_calls) and Anthropic format (content with tool_use).
    """
    # First, look for submit_result tool calls
    for msg in reversed(trajectory):
        if msg.get('role') == 'assistant':
            # OpenAI format: tool_calls array
            tool_calls = msg.get('tool_calls', [])
            for tool_call in tool_calls:
                func = tool_call.get('function', {})
                if func.get('name') == 'submit_result':
                    args_str = func.get('arguments', '{}')
                    try:
                        args = json.loads(args_str)
                        return args.get('message', '')
                    except json.JSONDecodeError:
                        continue

            # Anthropic format: content array with tool_use blocks
            content = msg.get('content', [])
            if isinstance(content, list):
                for block in content:
                    if block.get('type') == 'tool_use' and block.get('name') == 'submit_result':
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
# Helper functions for WebArena setup (executed in runtime)
# ============================================================================


async def _setup_webarena_verified_task_impl(
    session_id: str, task_config: dict[str, Any], urls: dict[str, str]
) -> SetupWebArenaTaskResult:
    """Implementation of WebArena-Verified task setup to be called via call_function.

    This function will be executed in the runtime environment using call_function.
    It accesses BROWSER_SESSIONS from the runtime's executor context.

    Args:
        session_id: Browser session ID
        task_config: WebArena-Verified task configuration
        urls: WebArena instance URLs

    Returns:
        SetupWebArenaTaskResult with success status and goal
    """
    try:
        # Import runtime-side dependencies (only available in runtime)
        from nexus.extensions.tasks.webarena.auth import authenticate_site
        from nexus.runtime.executors.browser import BROWSER_SESSIONS

        session = BROWSER_SESSIONS.get(session_id)
        if not session:
            return SetupWebArenaTaskResult(success=False, goal='', error_message=f'Session not found: {session_id}')

        page = session.current_page

        # Authenticate for required sites
        sites = task_config.get('sites', [])
        if len(sites) > 1:
            logger.debug(f'[Setup] Cross-site task detected, authenticating to {len(sites)} sites: {sites}')

        for i, site in enumerate(sites):
            logger.debug(f'[Setup] Authenticating site {i + 1}/{len(sites)}: {site}')
            try:
                await authenticate_site(session_id, site, urls)
            except Exception as auth_err:
                error_msg = f'Failed to setup WebArena-Verified task: {auth_err}'
                logger.error(f'[Setup] {error_msg}')
                return SetupWebArenaTaskResult(success=False, goal='', error_message=error_msg)

        # Set geolocation if specified
        if 'geolocation' in task_config:
            geo = task_config['geolocation']
            await session.context.set_geolocation(geo)

        # Navigate to start URL(s) - handle both start_url and start_urls formats
        start_url = task_config.get('start_url', '')
        if not start_url:
            start_urls = task_config.get('start_urls', [])
            start_url = ' |AND| '.join(start_urls) if start_urls else ''

        logger.debug(f'[Setup] Navigating to start_url: {start_url}')
        if start_url:
            start_urls = start_url.split(' |AND| ')
            for i, url in enumerate(start_urls):
                logger.debug(f'[Setup] goto: {url}')
                await session.goto(url)
                logger.debug(f'[Setup] Current page URL after goto: {page.url}')
                if i < len(start_urls) - 1:
                    await session.new_tab()

        # Get goal
        goal = task_config.get('intent', '')

        return SetupWebArenaTaskResult(success=True, goal=goal)

    except Exception as e:
        logger.error(f'Setup WebArena-Verified task failed: {e}', exc_info=True)
        return SetupWebArenaTaskResult(success=False, goal='', error_message=str(e))
