"""CUA-Gym-Hub app manifest and source compatibility patches."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CUA_GYM_HUB_APPS = (
    'slack_mock',
    'notion_mock',
    'gmail_mock',
    'salesforce_mock',
    'github_mock',
    'jira_mock',
    'linkedin_mock',
    'google_docs_mock',
    'instagram_mock',
    'hubspot_mock',
    'twitter_mock',
    'google_sheets_mock',
    'outlook_web_mock',
    'instacart_mock',
    'asana_mock',
    'gitlab_mock',
    'google_calendar_mock',
    'trello_mock',
    'pinterest_mock',
    'postman_mock',
    'uber_eats_mock',
    'google_drive_mock',
    'microsoft_teams_mock',
    'reddit_mock',
    'monday_mock',
    'shopify_admin_mock',
    'stripe_dashboard_mock',
    'docusign_mock',
    'discord_mock',
    'wechat_mock',
    'facebook_mock',
)


@dataclass(frozen=True)
class HubBuildPatch:
    """Build-time compatibility patch for upstream CUA-Gym-Hub app sources."""

    app_name: str
    reason: str
    replacements: tuple[tuple[str, str], ...]
    file_globs: tuple[str, ...] = ('*.js', '*.jsx', '*.ts', '*.tsx')


HUB_BUILD_PATCHES = (
    HubBuildPatch(
        app_name='github_mock',
        reason='github_mock imports XitHub from lucide-react, but lucide exports Github.',
        replacements=(
            ('{ XitHub,', '{ Github as XitHub,'),
            ('{XitHub,', '{ Github as XitHub,'),
        ),
    ),
    HubBuildPatch(
        app_name='trello_mock',
        reason='trello_mock imports Xrello from lucide-react, but lucide exports Trello.',
        replacements=(
            ('{ Xrello,', '{ Trello as Xrello,'),
            ('{Xrello,', '{ Trello as Xrello,'),
        ),
    ),
)


__all__ = ['DEFAULT_CUA_GYM_HUB_APPS', 'HUB_BUILD_PATCHES', 'HubBuildPatch']
