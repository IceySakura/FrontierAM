"""Service setup utilities for custom web tasks.

Provides functions to start and manage Docker-based web services
in runtime containers.

This module follows the same pattern as webarena/service_setup.py.
"""

import asyncio
import logging

from typing import Any

from pydantic import BaseModel

from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


# ============================================================================
# Main Service Setup Function (matches webarena pattern)
# ============================================================================


async def setup_custom_web_services(
    runtime: Runtime,
    service_config: dict[str, Any],
    service_port_map: dict[str, int],
) -> dict[str, str]:
    """Setup custom web services and return their URLs.

    Since nexus runtime uses its own start.sh script which overrides the Docker image's
    entrypoint, we need to manually start the web service inside the container.

    Args:
        runtime: Runtime instance with exposed ports
        service_config: Service configuration dictionary
        service_port_map: Mapping from service name to internal port

    Returns:
        Mapping from service name to external URL
    """
    service_urls = {}

    for service_name, port in service_port_map.items():
        if port not in runtime.port_mappings:
            logger.warning(
                f'Port {port} not found in runtime.port_mappings for service {service_name}. '
                f'Available ports: {list(runtime.port_mappings.keys())}'
            )
            continue

        url = runtime.port_mappings[port]
        service_urls[service_name] = url
        logger.debug(f'Service {service_name} mapped to {url}')

    # Get startup command from config
    # Since nexus runtime overrides the Docker entrypoint, we need to manually start the service
    startup_command = service_config.get('startup_command')
    app_dir = service_config.get('app_dir', '/app')

    if startup_command:
        logger.debug(f'Starting service with command: {startup_command}')
        try:
            session = await runtime.create_terminal_session()
            session_id = session.session_id

            # Run startup command in background using nohup and disown
            # This ensures the process continues running after the shell command returns
            # Pattern: nohup sh -c 'cd /app && npm start' > /tmp/service.log 2>&1 &
            full_command = f"nohup sh -c 'cd {app_dir} && {startup_command}' > /tmp/service.log 2>&1 &"
            logger.debug(f'Running: {full_command}')

            # Use a short timeout to prevent blocking forever
            # The command should return immediately since it's backgrounded
            try:
                await asyncio.wait_for(
                    runtime.run_command_in_terminal_session(
                        session_id=session_id,
                        command=full_command,
                    ),
                    timeout=10.0,  # 10 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning('Command timed out, but service may still be starting in background')

        except Exception as e:
            logger.error(f'Failed to start service: {e}')
            raise RuntimeError(f'Failed to start service: {e}') from e

    # Wait for services to become ready
    startup_wait = service_config.get('startup_wait_seconds', 30)
    skip_health_check = service_config.get('skip_health_check', True)

    logger.debug(f'Waiting {startup_wait}s for services to start...')
    await asyncio.sleep(startup_wait)

    # Debug: Check service log after waiting
    if startup_command:
        try:
            debug_session = await runtime.create_terminal_session()
            debug_result = await runtime.run_command_in_terminal_session(
                session_id=debug_session.session_id,
                command='cat /tmp/service.log 2>&1 | head -20',
            )
            logger.debug(f'Service log after startup: {debug_result.stdout[:500] if debug_result.stdout else "empty"}')

            # Also check if process is running
            ps_result = await runtime.run_command_in_terminal_session(
                session_id=debug_session.session_id,
                command='ps aux | grep -E "node|npm" | grep -v grep',
            )
            logger.debug(f'Running processes: {ps_result.stdout[:200] if ps_result.stdout else "none"}')
        except Exception as e:
            logger.warning(f'Failed to get debug info: {e}')

    # Optionally verify service health
    if not skip_health_check:
        health_check_path = service_config.get('health_check_path', '/')
        try:
            session = await runtime.create_terminal_session()
            session_id = session.session_id

            for service_name, url in service_urls.items():
                is_ready = await wait_for_service_ready(
                    runtime=runtime,
                    session_id=session_id,
                    url=url,
                    health_check_path=health_check_path,
                    timeout=60,
                    interval=5,
                )
                if not is_ready:
                    logger.warning(f'Service {service_name} may not be ready at {url}')
        except Exception as e:
            logger.warning(f'Health check failed, but continuing: {e}')

    return service_urls


# ============================================================================
# Data Models
# ============================================================================


class ServiceConfig(BaseModel):
    """Configuration for a Docker-based web service.

    Attributes:
        image: Docker image name
        port: Internal port the service listens on
        startup_command: Optional command to start the service
        startup_wait_seconds: Seconds to wait after startup
        health_check_path: HTTP path for health check
        environment: Environment variables
    """

    image: str
    port: int = 80
    startup_command: str | None = None
    startup_wait_seconds: int = 30
    health_check_path: str = '/'
    environment: dict[str, str] = {}


class ServiceStatus(BaseModel):
    """Status of a running service."""

    url: str
    is_ready: bool
    error_message: str | None = None


# ============================================================================
# Service Startup Functions
# ============================================================================


async def start_custom_service(runtime: Runtime, config: ServiceConfig) -> str:
    """Start a custom web service in runtime container.

    Args:
        runtime: Runtime instance with exposed port
        config: Service configuration

    Returns:
        External URL for accessing the service

    Raises:
        ValueError: If port not found in runtime.port_mappings
        RuntimeError: If service startup fails
    """
    port = config.port

    if port not in runtime.port_mappings:
        raise ValueError(
            f'Port {port} not found in runtime.port_mappings. Available ports: {list(runtime.port_mappings.keys())}'
        )

    url = runtime.port_mappings[port]
    logger.debug(f'Starting custom service at {url}...')

    # Create terminal session for running commands
    session = await runtime.create_terminal_session()
    session_id = session.session_id

    try:
        # Set environment variables if provided
        for key, value in config.environment.items():
            await runtime.run_command_in_terminal_session(
                session_id=session_id,
                command=f'export {key}="{value}"',
            )

        # Run startup command if provided
        if config.startup_command:
            logger.debug(f'Running startup command: {config.startup_command}')
            await runtime.run_command_in_terminal_session(
                session_id=session_id,
                command=config.startup_command,
            )

        # Wait for service to start
        logger.debug(f'Waiting {config.startup_wait_seconds}s for service to start...')
        await asyncio.sleep(config.startup_wait_seconds)

        # Verify service is ready
        is_ready = await wait_for_service_ready(
            runtime=runtime,
            session_id=session_id,
            url=url,
            health_check_path=config.health_check_path,
            timeout=60,
            interval=5,
        )

        if not is_ready:
            raise RuntimeError(f'Service failed to become ready at {url}')

        logger.debug(f'Custom service ready at {url}')
        return url

    except Exception as e:
        logger.error(f'Failed to start custom service: {e}')
        raise


async def wait_for_service_ready(
    runtime: Runtime,
    session_id: str,
    url: str,
    health_check_path: str = '/',
    timeout: int = 60,
    interval: int = 5,
) -> bool:
    """Wait for a service to become ready.

    Args:
        runtime: Runtime instance
        session_id: Terminal session ID
        url: Service URL
        health_check_path: Path for health check
        timeout: Maximum wait time in seconds
        interval: Check interval in seconds

    Returns:
        True if service is ready, False if timeout
    """
    health_url = url.rstrip('/') + health_check_path
    elapsed = 0

    while elapsed < timeout:
        try:
            # Use curl to check service health
            result = await runtime.run_command_in_terminal_session(
                session_id=session_id,
                command=f'curl -s -o /dev/null -w "%{{http_code}}" --max-time 5 {health_url}',
            )

            # Check if we got a successful response
            output = result.output.strip() if hasattr(result, 'output') and result.output else str(result)
            if output.startswith('2') or output.startswith('3'):
                logger.debug(f'Service health check passed: {health_url} -> {output}')
                return True

            logger.debug(f'Service not ready yet: {health_url} -> {output}')

        except Exception as e:
            logger.debug(f'Health check failed: {e}')

        await asyncio.sleep(interval)
        elapsed += interval

    logger.warning(f'Service health check timeout after {timeout}s')
    return False


async def start_generic_web_service(
    runtime: Runtime,
    port: int = 80,
    startup_command: str | None = None,
    wait_seconds: int = 30,
) -> str:
    """Start a generic web service with default settings.

    This is a simplified version for common use cases.

    Args:
        runtime: Runtime instance
        port: Internal port
        startup_command: Optional startup command
        wait_seconds: Seconds to wait after startup

    Returns:
        External URL for the service
    """
    config = ServiceConfig(
        image='',  # Image already configured in runtime
        port=port,
        startup_command=startup_command,
        startup_wait_seconds=wait_seconds,
    )
    return await start_custom_service(runtime, config)


# ============================================================================
# Common Service Templates
# ============================================================================


async def start_nginx_service(runtime: Runtime, port: int = 80) -> str:
    """Start nginx service."""
    config = ServiceConfig(
        image='nginx:latest',
        port=port,
        startup_command='nginx -g "daemon off;" &',
        startup_wait_seconds=5,
        health_check_path='/',
    )
    return await start_custom_service(runtime, config)


async def start_python_http_service(
    runtime: Runtime,
    port: int = 8000,
    directory: str = '/var/www',
) -> str:
    """Start Python HTTP server."""
    config = ServiceConfig(
        image='python:3.11',
        port=port,
        startup_command=f'cd {directory} && python -m http.server {port} &',
        startup_wait_seconds=5,
        health_check_path='/',
    )
    return await start_custom_service(runtime, config)


async def start_node_service(
    runtime: Runtime,
    port: int = 3000,
    app_dir: str = '/app',
    start_script: str = 'npm start',
) -> str:
    """Start Node.js application."""
    config = ServiceConfig(
        image='node:18',
        port=port,
        startup_command=f'cd {app_dir} && {start_script} &',
        startup_wait_seconds=30,
        health_check_path='/',
    )
    return await start_custom_service(runtime, config)


async def start_flask_service(
    runtime: Runtime,
    port: int = 5000,
    app_path: str = '/app/app.py',
) -> str:
    """Start Flask application."""
    config = ServiceConfig(
        image='python:3.11',
        port=port,
        startup_command=f'cd /app && FLASK_APP={app_path} flask run --host=0.0.0.0 --port={port} &',
        startup_wait_seconds=10,
        health_check_path='/',
        environment={'FLASK_ENV': 'production'},
    )
    return await start_custom_service(runtime, config)


async def start_django_service(
    runtime: Runtime,
    port: int = 8000,
    project_dir: str = '/app',
) -> str:
    """Start Django application."""
    config = ServiceConfig(
        image='python:3.11',
        port=port,
        startup_command=f'cd {project_dir} && python manage.py runserver 0.0.0.0:{port} &',
        startup_wait_seconds=15,
        health_check_path='/',
    )
    return await start_custom_service(runtime, config)


# ============================================================================
# Service Management
# ============================================================================


async def stop_service(runtime: Runtime, session_id: str, port: int) -> bool:
    """Stop a service running on the specified port.

    Args:
        runtime: Runtime instance
        session_id: Terminal session ID
        port: Port the service is running on

    Returns:
        True if service was stopped successfully
    """
    try:
        # Find and kill process using the port
        await runtime.run_command_in_terminal_session(
            session_id=session_id,
            command=f'fuser -k {port}/tcp 2>/dev/null || true',
        )
        logger.debug(f'Stopped service on port {port}')
        return True
    except Exception as e:
        logger.error(f'Failed to stop service on port {port}: {e}')
        return False


async def restart_service(
    runtime: Runtime,
    config: ServiceConfig,
    session_id: str | None = None,
) -> str:
    """Restart a service.

    Args:
        runtime: Runtime instance
        config: Service configuration
        session_id: Optional existing terminal session

    Returns:
        New service URL
    """
    if session_id:
        await stop_service(runtime, session_id, config.port)
        await asyncio.sleep(2)

    return await start_custom_service(runtime, config)
