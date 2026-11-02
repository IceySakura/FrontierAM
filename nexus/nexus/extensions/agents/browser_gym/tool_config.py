"""Tool subset configurations for BrowserGym agent.

Based on BrowserGym's ACTION_SUBSETS, defines which browser tools
are available for different benchmarks like WebArena.
"""

from typing import Literal


# Tool subset type
ToolSubset = Literal['webarena', 'full']


# Tool subsets - mapping benchmark names to lists of tool identifiers
TOOL_SUBSETS: dict[ToolSubset, list[str]] = {
    'webarena': [
        'browser/scroll',  # Scroll page up/down
        'browser/press_key',  # Press keyboard keys
        'browser/click',  # Click element by bid
        'browser/fill',  # Fill form field
        'browser/hover',  # Hover over element
        'browser/focus_tab',  # Switch tab focus
        'browser/new_tab',  # Open new tab
        'browser/go_back',  # Browser back
        'browser/go_forward',  # Browser forward
        'browser/goto',  # Navigate to URL
        'browser/close_tab',  # Close tab
        'browser/select_option',  # Select dropdown option
        'browser_gym/submit_result',  # Submit result (STOP action)
        # Note: 'report_infeasible' is sent via submit_result with specific message
    ],
    'full': [
        'browser/goto',
        'browser/click',
        'browser/fill',
        'browser/scroll',
        'browser/press_key',
        'browser/hover',
        'browser/select_option',
        'browser/new_tab',
        'browser/close_tab',
        'browser/focus_tab',
        'browser/go_back',
        'browser/go_forward',
        'browser/get_observation',
        'browser_gym/submit_result',
    ],
}


def get_tools_for_subset(subset: ToolSubset) -> list[str]:
    """Get the list of tool identifiers for a given subset.

    Args:
        subset: The subset name (e.g., 'webarena', 'full')

    Returns:
        List of tool identifiers (e.g., ['browser/click', 'browser/fill', ...])

    Raises:
        ValueError: If subset is not recognized
    """
    if subset not in TOOL_SUBSETS:
        raise ValueError(f'Unknown tool subset: {subset}. Available: {list(TOOL_SUBSETS.keys())}')
    return TOOL_SUBSETS[subset]
