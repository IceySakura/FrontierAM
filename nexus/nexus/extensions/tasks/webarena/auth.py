"""WebArena authentication helpers.

This module contains authentication functions for WebArena sites.
These functions are designed to be used within the runtime environment.
"""

import json
import logging


logger = logging.getLogger(__name__)


# WebArena site credentials (from WebArena benchmark)
WEBARENA_CREDENTIALS = {
    'gitlab': {'username': 'byteblaze', 'password': 'hello1234'},
    'shopping': {'username': 'emma.lopez@gmail.com', 'password': 'Password.123'},
    'shopping_admin': {'username': 'admin', 'password': 'admin1234'},
    'reddit': {'username': 'MarvelsGrantMan136', 'password': 'test1234'},
    'wikipedia': {},  # No auth needed
    'map': {},  # No auth needed
}


async def authenticate_site(session_id: str, site: str, urls: dict[str, str]) -> None:
    """Authenticate to a WebArena site using browser executor API.

    Args:
        session_id: Browser session ID (handle-based, no direct Page access)
        site: Site name (e.g., 'gitlab', 'shopping', 'reddit')
        urls: Dictionary mapping site names to base URLs

    Note:
        This function uses executor functions directly (per runtime coding guidelines).
        For sites with simple CSS selectors, browser_fill/browser_click are used.
        For reddit (label/role-based locators), browser_eval_js is used as escape hatch.
    """
    if site not in WEBARENA_CREDENTIALS:
        logger.warning(f'Unknown site: {site}')
        return

    creds = WEBARENA_CREDENTIALS[site]
    if not creds:
        return  # No authentication needed

    base_url = urls.get(site, '')
    if not base_url:
        logger.warning(f'No URL found for site: {site}')
        return

    from nexus.runtime.executors.browser import (
        browser_click,
        browser_eval_js,
        browser_fill,
        browser_goto,
        browser_wait_for_load_state,
    )
    from nexus.runtime.executors.models.browser import (
        ClickRequest,
        EvalJsRequest,
        FillRequest,
        GotoRequest,
        SelectorLocator,
        WaitForLoadStateRequest,
    )

    # Use longer timeout for login actions as they involve navigation which can be slow
    login_timeout = 60000  # 60 seconds for login operations

    if site == 'gitlab':
        await browser_goto(GotoRequest(session_id=session_id, url=f'{base_url}/users/sign_in', timeout=login_timeout))
        await browser_fill(
            FillRequest(
                session_id=session_id,
                locator=SelectorLocator(selector='#user_login'),
                value=creds['username'],
                timeout=login_timeout,
            )
        )
        await browser_fill(
            FillRequest(
                session_id=session_id,
                locator=SelectorLocator(selector='#user_password'),
                value=creds['password'],
                timeout=login_timeout,
            )
        )
        await browser_click(
            ClickRequest(
                session_id=session_id,
                locator=SelectorLocator(selector="button[type='submit']"),
                timeout=login_timeout,
            )
        )
        await browser_wait_for_load_state(
            WaitForLoadStateRequest(
                session_id=session_id,
                state='networkidle',
                timeout=login_timeout,
            )
        )

    elif site == 'shopping':
        await browser_goto(
            GotoRequest(
                session_id=session_id,
                url=f'{base_url}/customer/account/login',
                timeout=login_timeout,
            )
        )
        await browser_fill(
            FillRequest(
                session_id=session_id,
                locator=SelectorLocator(selector='#email'),
                value=creds['username'],
                timeout=login_timeout,
            )
        )
        await browser_fill(
            FillRequest(
                session_id=session_id,
                locator=SelectorLocator(selector='#pass'),
                value=creds['password'],
                timeout=login_timeout,
            )
        )
        await browser_click(
            ClickRequest(
                session_id=session_id,
                locator=SelectorLocator(selector='#send2'),
                timeout=login_timeout,
            )
        )
        await browser_wait_for_load_state(
            WaitForLoadStateRequest(
                session_id=session_id,
                state='networkidle',
                timeout=login_timeout,
            )
        )

    elif site == 'shopping_admin':
        await browser_goto(GotoRequest(session_id=session_id, url=f'{base_url}/admin', timeout=login_timeout))
        await browser_fill(
            FillRequest(
                session_id=session_id,
                locator=SelectorLocator(selector='#username'),
                value=creds['username'],
                timeout=login_timeout,
            )
        )
        await browser_fill(
            FillRequest(
                session_id=session_id,
                locator=SelectorLocator(selector='#login'),
                value=creds['password'],
                timeout=login_timeout,
            )
        )
        await browser_click(
            ClickRequest(
                session_id=session_id,
                locator=SelectorLocator(selector='.action-login'),
                timeout=login_timeout,
            )
        )
        await browser_wait_for_load_state(
            WaitForLoadStateRequest(
                session_id=session_id,
                state='networkidle',
                timeout=login_timeout,
            )
        )

    elif site == 'reddit':
        await browser_goto(GotoRequest(session_id=session_id, url=f'{base_url}/login', timeout=login_timeout))
        # Reddit uses label-based locators — use browser_eval_js as escape hatch
        username_json = json.dumps(creds['username'])
        password_json = json.dumps(creds['password'])
        await browser_eval_js(
            EvalJsRequest(
                session_id=session_id,
                expression=f"""() => {{
                function fillByLabel(labelText, value) {{
                    const labels = document.querySelectorAll('label');
                    for (const label of labels) {{
                        if (label.textContent.trim() === labelText) {{
                            const input = document.getElementById(label.htmlFor)
                                || label.querySelector('input');
                            if (input) {{
                                const setter = Object.getOwnPropertyDescriptor(
                                    HTMLInputElement.prototype, 'value').set;
                                setter.call(input, value);
                                input.dispatchEvent(new Event('input', {{bubbles: true}}));
                            }}
                        }}
                    }}
                }}
                fillByLabel('Username', {username_json});
                fillByLabel('Password', {password_json});
                const btn = [...document.querySelectorAll('button')]
                    .find(b => b.textContent.trim() === 'Log in');
                if (btn) btn.click();
            }}""",
            )
        )
        await browser_wait_for_load_state(
            WaitForLoadStateRequest(
                session_id=session_id,
                state='networkidle',
                timeout=login_timeout,
            )
        )

    logger.debug(f'Successfully authenticated to {site}')
