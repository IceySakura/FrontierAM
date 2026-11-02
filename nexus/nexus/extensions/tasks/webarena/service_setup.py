"""WebArena service setup utilities.

This module provides functions for starting WebArena services (one-stop-shop, cms, reddit, gitlab)
in runtime containers before task execution.
"""

import asyncio
import logging

from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


async def start_onestop_shop_service(runtime: Runtime, port: int = 8023) -> str:
    """Start one-stop-shop service in runtime container.

    Based on setup instructions in browsergym_setup.md:
    1. Execute /docker-entrypoint.sh supervisord -n -j /supervisord.pid > /dev/null 2>&1 & (wait 60s)
    2. Execute /var/www/magento2/bin/magento setup:store-config:set --base-url="http://{ip:port}" (wait 10s)
    3. Execute mysql command to update core_config_data (wait 10s)
    4. Execute /var/www/magento2/bin/magento cache:flush (wait 90s)

    Args:
        runtime: Runtime instance with exposed port
        port: Internal port number for the service (must be in runtime.port_mappings)

    Returns:
        External URL for accessing the service

    Raises:
        ValueError: If port not found in runtime.port_mappings
        RuntimeError: If service startup fails
    """
    if port not in runtime.port_mappings:
        raise ValueError(f'Port {port} not found in runtime.port_mappings')

    url = runtime.port_mappings[port]
    logger.debug(f'Starting one-stop-shop service at {url}...')

    # Create terminal session
    session = await runtime.create_terminal_session()
    session_id = session.session_id

    # Step 1: Start supervisord (wait 60s)
    logger.debug('Starting supervisord...')
    await runtime.run_command_in_terminal_session(
        session_id=session_id,
        command='/docker-entrypoint.sh supervisord -n -j /supervisord.pid > /tmp/supervisord.log 2>&1 &',
    )
    await asyncio.sleep(60)

    # Step 2: Configure base URL (wait 10s)
    logger.debug(f'Configuring base URL to {url}...')
    await runtime.run_command_in_terminal_session(
        session_id=session_id,
        command=f'/var/www/magento2/bin/magento setup:store-config:set --base-url="{url}"',
    )
    await asyncio.sleep(30)

    # Step 3: Update database (wait 10s)
    logger.debug('Updating database configuration...')
    mysql_cmd = f'mysql -u magentouser -pMyPassword magentodb -e \'UPDATE core_config_data SET value="{url}/" WHERE path = "web/secure/base_url";\''
    await runtime.run_command_in_terminal_session(session_id=session_id, command=mysql_cmd)
    await asyncio.sleep(30)

    # Step 4: Flush cache (wait 90s)
    logger.debug('Flushing cache...')
    await runtime.run_command_in_terminal_session(
        session_id=session_id, command='/var/www/magento2/bin/magento cache:flush'
    )
    await asyncio.sleep(90)

    logger.debug(f'One-stop-shop service ready at {url}')
    return url


async def start_reddit_service(runtime: Runtime, port: int = 8023) -> str:
    """Start reddit service in runtime container.

    Based on setup instructions in browsergym_setup.md:
    1. Execute /docker-entrypoint.sh supervisord -n -j /supervisord.pid > /dev/null 2>&1 & (wait 90s)

    Args:
        runtime: Runtime instance with exposed port
        port: Internal port number for the service (must be in runtime.port_mappings)

    Returns:
        External URL for accessing the service

    Raises:
        ValueError: If port not found in runtime.port_mappings
        RuntimeError: If service startup fails
    """
    if port not in runtime.port_mappings:
        raise ValueError(f'Port {port} not found in runtime.port_mappings')

    url = runtime.port_mappings[port]
    logger.debug(f'Starting reddit service at {url}...')

    # Create terminal session
    session = await runtime.create_terminal_session()
    session_id = session.session_id

    # Start supervisord (wait 90s)
    logger.debug('Starting supervisord...')
    await runtime.run_command_in_terminal_session(
        session_id=session_id,
        command='/docker-entrypoint.sh supervisord -n -j /supervisord.pid > /dev/null 2>&1 &',
    )
    await asyncio.sleep(90)

    logger.debug(f'Reddit service ready at {url}')
    return url


async def start_gitlab_service(runtime: Runtime, port: int = 8023) -> str:
    """Start gitlab service in runtime container.

    Based on setup instructions in browsergym_setup.md:
    1. Execute /opt/gitlab/embedded/bin/runsvdir-start > /dev/null 2>&1 & (wait 30s)
    2. Configure gitlab.rb with external_url and nginx settings
    3. Execute gitlab-ctl reconfigure (wait 30s)

    Args:
        runtime: Runtime instance with exposed port
        port: Internal port number for the service (must be in runtime.port_mappings)

    Returns:
        External URL for accessing the service

    Raises:
        ValueError: If port not found in runtime.port_mappings
        RuntimeError: If service startup fails
    """
    if port not in runtime.port_mappings:
        raise ValueError(f'Port {port} not found in runtime.port_mappings')

    url = runtime.port_mappings[port]
    logger.debug(f'Starting gitlab service at {url}...')

    # Create terminal session
    session = await runtime.create_terminal_session()
    session_id = session.session_id

    # Step 1: Start runsvdir (wait 30s)
    logger.debug('Starting runsvdir...')
    await runtime.run_command_in_terminal_session(
        session_id=session_id,
        command='/opt/gitlab/embedded/bin/runsvdir-start > /dev/null 2>&1 &',
    )
    await asyncio.sleep(30)

    # Step 2: Configure gitlab.rb
    logger.debug(f'Configuring GitLab with external URL {url}...')

    # Update external_url
    await runtime.run_command_in_terminal_session(
        session_id=session_id,
        command=f'sed -i "s|^external_url.*|external_url \'{url}\'|" /etc/gitlab/gitlab.rb',
    )

    # Add nginx configuration
    await runtime.run_command_in_terminal_session(
        session_id=session_id, command='echo "nginx[\'enable\'] = true" >> /etc/gitlab/gitlab.rb'
    )
    await runtime.run_command_in_terminal_session(
        session_id=session_id, command=f'echo "nginx[\'listen_port\'] = {port}" >> /etc/gitlab/gitlab.rb'
    )
    await runtime.run_command_in_terminal_session(
        session_id=session_id,
        command="echo \"nginx['listen_addresses'] = ['0.0.0.0']\" >> /etc/gitlab/gitlab.rb",
    )
    await runtime.run_command_in_terminal_session(
        session_id=session_id, command='echo "nginx[\'listen_https\'] = false" >> /etc/gitlab/gitlab.rb'
    )
    await runtime.run_command_in_terminal_session(
        session_id=session_id,
        command='echo "nginx[\'redirect_http_to_https\'] = false" >> /etc/gitlab/gitlab.rb',
    )

    # Configure puma workers based on runtime.resource_spec.cpu
    # resource_spec is passed from GongfengRuntimeProvider when creating HttpRuntime
    resource_spec = getattr(runtime, 'resource_spec', {})
    cpu_limit = resource_spec.get('cpu')

    if cpu_limit:
        worker_count = max(1, int(cpu_limit))
        logger.debug(f'Configuring puma with {worker_count} workers (from runtime.resource_spec.cpu={cpu_limit})...')
    else:
        worker_count = 2
        logger.debug(f'runtime.resource_spec.cpu not set, using default {worker_count} workers')

    await runtime.run_command_in_terminal_session(
        session_id=session_id,
        command=f'echo "puma[\'worker_processes\'] = {worker_count}" >> /etc/gitlab/gitlab.rb',
    )

    # Step 3: Reconfigure GitLab (wait 30s)
    logger.debug('Running gitlab-ctl reconfigure...')
    await runtime.run_command_in_terminal_session(session_id=session_id, command='gitlab-ctl reconfigure')
    await asyncio.sleep(30)

    logger.debug(f'GitLab service ready at {url}')
    return url


# Service type to startup function mapping
SERVICE_STARTUP_FUNCTIONS = {
    'shopping': start_onestop_shop_service,
    'shopping_admin': start_onestop_shop_service,  # Same service, different URL
    'reddit': start_reddit_service,
    'gitlab': start_gitlab_service,
}


async def setup_webarena_services(
    runtime: Runtime, task_config: dict, service_port_map: dict[str, int]
) -> dict[str, str]:
    """Setup WebArena services based on task configuration.

    Args:
        runtime: Runtime instance with exposed ports
        task_config: WebArena task configuration
        service_port_map: Mapping from service name to internal port
                         Example: {'shopping': 80, 'gitlab': 8023}

    Returns:
        Dictionary mapping service names to their external URLs

    Note:
        - Service startup functions read CPU limit from runtime.resource_spec['cpu']
        - Services sharing the same startup function and port (e.g., shopping + shopping_admin)
          are only started once, with the URL shared across all their names.

    Example:
        service_urls = await setup_webarena_services(
            runtime,
            task_config={'sites': ['shopping', 'shopping_admin']},
            service_port_map={'shopping': 80, 'shopping_admin': 80}
        )
        # Returns: {'shopping': 'http://localhost:32768', 'shopping_admin': 'http://localhost:32768'}
    """
    sites = task_config.get('sites', [])
    service_urls = {}

    # Track which (startup_func, port) combos have already been started
    # to avoid starting the same service twice (e.g., shopping + shopping_admin)
    started_services: dict[tuple[int, int], str] = {}  # (func_id, port) -> url

    for site in sites:
        if site not in SERVICE_STARTUP_FUNCTIONS:
            logger.warning(f'No startup function for service: {site}')
            continue

        if site not in service_port_map:
            logger.warning(f'No port mapping for service: {site}')
            continue

        port = service_port_map[site]
        startup_func = SERVICE_STARTUP_FUNCTIONS[site]
        func_key = (id(startup_func), port)

        if func_key in started_services:
            # Service already started (e.g., shopping_admin reuses shopping's service)
            service_urls[site] = started_services[func_key]
            logger.debug(f'Reusing already-started service for {site}: {service_urls[site]}')
            continue

        try:
            url = await startup_func(runtime, port)
            service_urls[site] = url
            started_services[func_key] = url
        except Exception as e:
            logger.error(f'Failed to start service {site}: {e}', exc_info=True)
            raise

    return service_urls
