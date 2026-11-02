"""Site-Aware DOM Filtering Configuration.

This module provides site-specific filtering configurations based on
empirical analysis of filter performance across different websites.

Key findings from experiments:
- Reddit/Shopping Admin: Filtering is harmful, should be minimal or disabled
- Shopping/Map/Wikipedia: Filtering is beneficial, should be preserved

Usage:
    from site_aware_filter import get_site_filter_config, detect_site_from_url

    site = detect_site_from_url(current_url)
    config = get_site_filter_config(site)

    # Apply config to TaskAwareFilter
    filter = TaskAwareFilter(
        task_goal=task,
        enable_filter=config['enable_filter'],
        relevance_threshold=config['relevance_threshold'],
        min_elements=config['min_elements'],
    )
"""

import re

from typing import TypedDict


class SiteFilterConfig(TypedDict):
    """Configuration for site-specific filtering."""

    enable_filter: bool  # Whether to enable filtering at all
    relevance_threshold: float  # Threshold for relevance scoring (lower = keep more)
    min_elements: int  # Minimum elements to always keep
    preserve_navigation: bool  # Whether to preserve navigation elements
    preserve_tables: bool  # Whether to preserve table structures
    additional_keywords: list[str]  # Site-specific keywords to always consider relevant


# Site detection patterns
SITE_PATTERNS = {
    'shopping': [
        r'magento',
        r'onestopmarket',
        r'\.shopping\.',
        r'/customer/',
        r'/checkout/',
        r'/catalog/',
    ],
    'shopping_admin': [
        r'/admin/',
        r'admin\.',
        r'backend',
        r'/sales/order',
        r'/catalog/product',
        r'/customer/index',
    ],
    'reddit': [
        r'reddit',
        r'/r/',
        r'postmill',
        r'/f/',
        r'/forum/',
        r'/submit',
    ],
    'gitlab': [
        r'gitlab',
        r'/-/',
        r'/projects/',
        r'/issues/',
        r'/merge_requests/',
        r'/commits/',
    ],
    'map': [
        r'openstreetmap',
        r'osm',
        r'nominatim',
        r'/search\?',
        r'lat=',
        r'lon=',
        r'zoom=',
    ],
    'wikipedia': [
        r'wikipedia',
        r'wiki/',
        r'mediawiki',
        r'/wiki/',
        r'Special:',
        r'Category:',
    ],
}


# Site-specific filter configurations
# Based on experimental results:
# - Reddit (+4 net): Disable filtering
# - Shopping Admin (+1 net): Minimal filtering
# - Shopping (-3 net): Keep filtering
# - Map (-2 net): Keep filtering
# - Wikipedia (-2 net): Keep filtering
# - GitLab (-1 net): Moderate filtering

SITE_CONFIGS: dict[str, SiteFilterConfig] = {
    # === Sites where filtering is HARMFUL (disable or minimize) ===
    'reddit': {
        'enable_filter': False,  # Disable filtering entirely
        'relevance_threshold': 0.0,
        'min_elements': 100,
        'preserve_navigation': True,
        'preserve_tables': True,
        'additional_keywords': [
            'post',
            'comment',
            'vote',
            'upvote',
            'downvote',
            'submit',
            'reply',
            'thread',
            'forum',
            'community',
            'subscriber',
            'user',
            'karma',
            'award',
        ],
    },
    'shopping_admin': {
        'enable_filter': True,
        'relevance_threshold': 0.05,  # Very low threshold (keep more elements)
        'min_elements': 50,  # Keep more elements minimum
        'preserve_navigation': True,
        'preserve_tables': True,  # Critical for admin panels
        'additional_keywords': [
            'order',
            'customer',
            'product',
            'invoice',
            'shipment',
            'status',
            'pending',
            'processing',
            'complete',
            'canceled',
            'revenue',
            'sales',
            'report',
            'dashboard',
            'grid',
            'action',
            'edit',
            'view',
            'delete',
            'mass',
            'select',
            'filter',
            'search',
            'date',
            'total',
            'quantity',
            'price',
        ],
    },
    # === Sites where filtering is BENEFICIAL (keep enabled) ===
    'shopping': {
        'enable_filter': True,
        'relevance_threshold': 0.12,  # Standard threshold
        'min_elements': 25,
        'preserve_navigation': True,
        'preserve_tables': False,
        'additional_keywords': [
            'add to cart',
            'buy',
            'price',
            'checkout',
            'cart',
            'product',
            'review',
            'rating',
            'stock',
            'quantity',
            'size',
            'color',
            'option',
            'wishlist',
        ],
    },
    'map': {
        'enable_filter': True,
        'relevance_threshold': 0.15,  # Slightly higher (more aggressive filtering)
        'min_elements': 20,
        'preserve_navigation': True,
        'preserve_tables': False,
        'additional_keywords': [
            'search',
            'location',
            'address',
            'route',
            'direction',
            'zoom',
            'layer',
            'marker',
            'distance',
            'navigate',
            'street',
            'city',
            'country',
            'coordinates',
        ],
    },
    'wikipedia': {
        'enable_filter': True,
        'relevance_threshold': 0.15,  # Higher threshold (more aggressive)
        'min_elements': 20,
        'preserve_navigation': False,  # Wikipedia has lots of nav clutter
        'preserve_tables': True,  # Tables often contain important info
        'additional_keywords': [
            'article',
            'section',
            'reference',
            'citation',
            'edit',
            'history',
            'talk',
            'category',
            'portal',
            'contents',
        ],
    },
    'gitlab': {
        'enable_filter': True,
        'relevance_threshold': 0.08,  # Lower threshold (keep more)
        'min_elements': 35,
        'preserve_navigation': True,  # Important for navigation
        'preserve_tables': True,
        'additional_keywords': [
            'project',
            'repository',
            'branch',
            'commit',
            'merge',
            'issue',
            'pipeline',
            'ci',
            'cd',
            'deploy',
            'build',
            'member',
            'contributor',
            'activity',
            'file',
            'tree',
            'blame',
            'history',
            'tag',
            'release',
            'milestone',
            'label',
            'assignee',
            'author',
            'created',
            'updated',
        ],
    },
    # === Default configuration for unknown sites ===
    'default': {
        'enable_filter': True,
        'relevance_threshold': 0.10,  # Moderate threshold
        'min_elements': 25,
        'preserve_navigation': True,
        'preserve_tables': True,
        'additional_keywords': [],
    },
}


def detect_site_from_url(url: str) -> str:
    """Detect site type from URL.

    Args:
        url: The current page URL

    Returns:
        Site type string (e.g., 'reddit', 'shopping', etc.)
    """
    if not url:
        return 'default'

    url_lower = url.lower()

    # Check each site pattern
    for site, patterns in SITE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, url_lower):
                return site

    return 'default'


def detect_site_from_task(task_goal: str) -> str:
    """Detect site type from task goal description.

    Args:
        task_goal: The task goal text

    Returns:
        Site type string
    """
    if not task_goal:
        return 'default'

    task_lower = task_goal.lower()

    # Site keywords in task
    site_keywords = {
        'reddit': ['reddit', 'subreddit', 'forum', 'post', 'upvote'],
        'gitlab': ['gitlab', 'repository', 'commit', 'merge request', 'issue', 'pipeline'],
        'wikipedia': ['wikipedia', 'wiki', 'article', 'encyclopedia'],
        'shopping': ['shop', 'buy', 'purchase', 'cart', 'product', 'price', 'order'],
        'shopping_admin': ['admin', 'backend', 'manage', 'dashboard', 'magento'],
        'map': ['map', 'location', 'direction', 'route', 'address', 'openstreetmap'],
    }

    for site, keywords in site_keywords.items():
        for kw in keywords:
            if kw in task_lower:
                return site

    return 'default'


def get_site_filter_config(site: str) -> SiteFilterConfig:
    """Get filter configuration for a site.

    Args:
        site: Site type string

    Returns:
        SiteFilterConfig for the site
    """
    return SITE_CONFIGS.get(site, SITE_CONFIGS['default'])


def should_preserve_element(
    element_type: str,
    element_text: str,
    site: str,
    config: SiteFilterConfig,
) -> bool:
    """Check if an element should be preserved based on site config.

    Args:
        element_type: Type of the element (e.g., 'link', 'button')
        element_text: Text content of the element
        site: Site type
        config: Site filter configuration

    Returns:
        True if element should be preserved regardless of relevance score
    """
    element_type_lower = element_type.lower()
    element_text_lower = element_text.lower()

    # Preserve navigation elements if configured
    if config['preserve_navigation']:
        nav_types = {'navigation', 'menu', 'menuitem', 'menubar', 'nav'}
        if element_type_lower in nav_types:
            return True

        # Navigation-related text
        nav_texts = ['menu', 'nav', 'home', 'back', 'next', 'previous']
        if any(t in element_text_lower for t in nav_texts):
            return True

    # Preserve table elements if configured
    if config['preserve_tables']:
        table_types = {'table', 'row', 'cell', 'columnheader', 'rowheader', 'grid', 'gridcell'}
        if element_type_lower in table_types:
            return True

    # Check additional keywords
    return any(keyword.lower() in element_text_lower for keyword in config['additional_keywords'])


# Integration helper for agent.py
def create_site_aware_filter(
    task_goal: str,
    initial_url: str | None = None,
):
    """Create a TaskAwareFilter with site-specific configuration.

    This is a helper function to integrate site-aware filtering into the agent.

    Args:
        task_goal: The task goal
        initial_url: Initial URL (optional, for site detection)

    Returns:
        Tuple of (TaskAwareFilter instance or None, site_name, config)
    """
    from nexus.extensions.agents.browser_gym.task_aware_filter import TaskAwareFilter

    # Detect site
    site = 'default'
    if initial_url:
        site = detect_site_from_url(initial_url)
    if site == 'default':
        site = detect_site_from_task(task_goal)

    # Get config
    config = get_site_filter_config(site)

    # Create filter if enabled
    if not config['enable_filter']:
        return None, site, config

    # Extend task keywords with site-specific ones
    filter_instance = TaskAwareFilter(
        task_goal=task_goal,
        enable_filter=True,
        relevance_threshold=config['relevance_threshold'],
        min_elements=config['min_elements'],
    )

    # Add site-specific keywords
    filter_instance.update_task_context(config['additional_keywords'])

    return filter_instance, site, config
