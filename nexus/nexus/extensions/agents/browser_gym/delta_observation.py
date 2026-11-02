"""Delta Observation Module - Incremental DOM Updates.

This module computes the difference between consecutive observations,
significantly reducing redundant information in the context.

Key benefits:
1. Reduces token usage by 50-70% for stable pages
2. Highlights changes that are most relevant to actions
3. Maintains context about unchanged elements
4. Automatically resets on URL changes to avoid confusion
"""

import logging
import re

from dataclasses import dataclass
from difflib import SequenceMatcher

from nexus.extensions.agents.browser_gym.token_counting import (
    count_tokens_with_fallback,
)


logger = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    return count_tokens_with_fallback(text, chars_per_token=4)


def extract_url_from_observation(observation: str) -> str | None:
    """Extract URL from observation text.

    Args:
        observation: The observation text

    Returns:
        The URL if found, None otherwise
    """
    match = re.search(r'URL:\s*([^\n]+)', observation)
    if match:
        return match.group(1).strip()
    return None


def extract_title_from_observation(observation: str) -> str | None:
    """Extract title from observation text.

    Args:
        observation: The observation text

    Returns:
        The title if found, None otherwise
    """
    match = re.search(r'Title:\s*([^\n]+)', observation)
    if match:
        return match.group(1).strip()
    return None


@dataclass
class DeltaStats:
    """Statistics about delta computation."""

    original_tokens: int
    delta_tokens: int
    unchanged_elements: int
    changed_elements: int
    new_elements: int
    removed_elements: int
    reduction_ratio: float

    def __str__(self) -> str:
        return (
            f'Delta: {self.original_tokens:,} → {self.delta_tokens:,} tokens '
            f'({self.reduction_ratio:.1%} reduction), '
            f'unchanged={self.unchanged_elements}, changed={self.changed_elements}, '
            f'new={self.new_elements}, removed={self.removed_elements}'
        )


@dataclass
class ParsedElement:
    """A parsed accessibility tree element for comparison."""

    bid: str | None
    element_type: str
    text: str
    attributes: dict[str, str]
    indent: int
    original_line: str

    def get_key(self) -> str:
        """Get a unique key for matching elements across observations."""
        if self.bid:
            return f'bid:{self.bid}'
        # For elements without bid, use type + text hash
        text_hash = hash(self.text[:50]) if self.text else 0
        return f'type:{self.element_type}:{self.indent}:{text_hash}'

    def is_similar(self, other: 'ParsedElement') -> bool:
        """Check if two elements are similar (same element, possibly changed)."""
        if self.bid and other.bid:
            return self.bid == other.bid
        return (
            self.element_type == other.element_type
            and self.indent == other.indent
            and self.text[:30] == other.text[:30]
        )

    def get_changes(self, other: 'ParsedElement') -> list[str]:
        """Get list of changes compared to another element."""
        changes = []
        if self.text != other.text:
            changes.append(f"text: '{self.text[:30]}...' → '{other.text[:30]}...'")
        if self.attributes != other.attributes:
            for key in set(self.attributes.keys()) | set(other.attributes.keys()):
                old_val = self.attributes.get(key)
                new_val = other.attributes.get(key)
                if old_val != new_val:
                    changes.append(f'{key}: {old_val} → {new_val}')
        return changes


def parse_axtree_line(line: str) -> ParsedElement | None:
    """Parse a single line from the accessibility tree."""
    if not line.strip():
        return None

    # Calculate indent
    indent = 0
    stripped = line
    while stripped.startswith('\t'):
        indent += 1
        stripped = stripped[1:]

    stripped = stripped.strip()
    if not stripped:
        return None

    # Extract bid
    bid = None
    bid_match = re.match(r'\[(\d+)\]\s*', stripped)
    if bid_match:
        bid = bid_match.group(1)
        stripped = stripped[bid_match.end() :]

    # Extract element type
    type_match = re.match(r'(\w+)', stripped)
    if not type_match:
        return ParsedElement(
            bid=bid,
            element_type='unknown',
            text=stripped,
            attributes={},
            indent=indent,
            original_line=line,
        )

    element_type = type_match.group(1)
    rest = stripped[type_match.end() :].strip()

    # Extract text
    text = ''
    text_match = re.search(r"'([^']*)'|\"([^\"]*)\"", rest)
    if text_match:
        text = text_match.group(1) or text_match.group(2) or ''

    # Extract attributes
    attributes = {}
    attr_patterns = [
        (r"url='([^']*)'", 'url'),
        (r'\bfocused\b', 'focused'),
        (r'\bdisabled\b', 'disabled'),
        (r'\bchecked\b', 'checked'),
        (r'\bselected\b', 'selected'),
    ]
    for pattern, attr_name in attr_patterns:
        match = re.search(pattern, rest)
        if match:
            if match.lastindex:
                attributes[attr_name] = match.group(1)
            else:
                attributes[attr_name] = 'true'

    return ParsedElement(
        bid=bid,
        element_type=element_type,
        text=text,
        attributes=attributes,
        indent=indent,
        original_line=line,
    )


def parse_axtree(axtree: str) -> dict[str, ParsedElement]:
    """Parse accessibility tree into a dictionary keyed by element key."""
    elements = {}
    for line in axtree.strip().split('\n'):
        elem = parse_axtree_line(line)
        if elem:
            key = elem.get_key()
            elements[key] = elem
    return elements


def extract_axtree_section(observation: str) -> tuple[str, str, str]:
    """Extract header, axtree, and footer from observation.

    Returns:
        Tuple of (header, axtree, footer)
    """
    parts = observation.split('## Accessibility Tree')
    if len(parts) < 2:
        return observation, '', ''

    header = parts[0] + '## Accessibility Tree\n'
    rest = parts[1]

    # Check for action history
    if '## Action History' in rest:
        axtree_parts = rest.split('## Action History')
        axtree = axtree_parts[0].strip()
        footer = '\n## Action History' + axtree_parts[1]
    else:
        axtree = rest.strip()
        footer = ''

    return header, axtree, footer


def compute_delta(
    previous_observation: str | None,
    current_observation: str,
    max_unchanged_display: int = 10,  # Increased from 5 to 10
    include_context_summary: bool = True,
) -> tuple[str, DeltaStats | None]:
    """Compute delta between two observations.

    Args:
        previous_observation: Previous observation (None for first observation)
        current_observation: Current observation
        max_unchanged_display: Maximum unchanged elements to display explicitly
        include_context_summary: Whether to include a summary of unchanged context

    Returns:
        Tuple of (delta_observation, stats) or (current_observation, None) if no previous
    """
    if previous_observation is None:
        return current_observation, None

    # Extract axtree sections
    prev_header, prev_axtree, prev_footer = extract_axtree_section(previous_observation)
    curr_header, curr_axtree, curr_footer = extract_axtree_section(current_observation)

    if not prev_axtree or not curr_axtree:
        return current_observation, None

    # Parse both trees
    prev_elements = parse_axtree(prev_axtree)
    curr_elements = parse_axtree(curr_axtree)

    prev_keys = set(prev_elements.keys())
    curr_keys = set(curr_elements.keys())

    # Categorize elements
    unchanged_keys = []
    changed_elements = []
    new_elements = []
    removed_keys = prev_keys - curr_keys

    for key in curr_keys:
        if key in prev_keys:
            prev_elem = prev_elements[key]
            curr_elem = curr_elements[key]

            # Check if element changed
            if prev_elem.original_line.strip() == curr_elem.original_line.strip():
                unchanged_keys.append(key)
            else:
                changes = prev_elem.get_changes(curr_elem)
                changed_elements.append((curr_elem, changes))
        else:
            new_elements.append(curr_elements[key])

    # Build delta observation
    delta_parts = [curr_header]

    # Summary of unchanged elements
    if include_context_summary and unchanged_keys:
        # Count by type
        type_counts = {}
        for key in unchanged_keys:
            elem_type = curr_elements[key].element_type
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

        summary_items = [f'{count} {etype}' for etype, count in sorted(type_counts.items(), key=lambda x: -x[1])[:5]]
        delta_parts.append(f'\n[UNCHANGED: {len(unchanged_keys)} elements - {", ".join(summary_items)}]\n')

    # Show important unchanged elements (with bid) - EXPANDED list of important types
    important_element_types = {
        'button',
        'link',
        'textbox',
        'input',
        'searchbox',
        'combobox',
        'heading',
        'table',
        'form',
        'menuitem',
        'tab',
        'checkbox',
        'radio',
        'listbox',
        'option',
        'navigation',
        'main',
    }
    important_unchanged = [
        curr_elements[key]
        for key in unchanged_keys
        if curr_elements[key].bid and curr_elements[key].element_type.lower() in important_element_types
    ][:max_unchanged_display]

    if important_unchanged:
        delta_parts.append('\n[KEY ELEMENTS (unchanged):]')
        for elem in important_unchanged:
            delta_parts.append(elem.original_line.rstrip())

    # Show changed elements
    if changed_elements:
        delta_parts.append('\n\n[CHANGED:]')
        for elem, changes in changed_elements:
            delta_parts.append(elem.original_line.rstrip())
            if changes:
                for change in changes[:2]:  # Limit changes shown
                    delta_parts.append(f'  ↳ {change}')

    # Show new elements
    if new_elements:
        delta_parts.append('\n\n[NEW:]')
        for elem in new_elements:
            delta_parts.append(elem.original_line.rstrip())

    # Show removed elements (brief)
    if removed_keys:
        removed_bids = [prev_elements[key].bid for key in removed_keys if prev_elements[key].bid]
        if removed_bids:
            delta_parts.append(
                f'\n\n[REMOVED: {len(removed_keys)} elements, bids: {", ".join(str(bid) for bid in removed_bids[:10])}{"..." if len(removed_bids) > 10 else ""}]'
            )
        else:
            delta_parts.append(f'\n\n[REMOVED: {len(removed_keys)} elements]')

    # Add footer (action history)
    if curr_footer:
        delta_parts.append(curr_footer)

    delta_observation = '\n'.join(delta_parts)

    # Calculate stats
    original_tokens = count_tokens(current_observation)
    delta_tokens = count_tokens(delta_observation)
    reduction_ratio = 1 - (delta_tokens / original_tokens) if original_tokens > 0 else 0

    stats = DeltaStats(
        original_tokens=original_tokens,
        delta_tokens=delta_tokens,
        unchanged_elements=len(unchanged_keys),
        changed_elements=len(changed_elements),
        new_elements=len(new_elements),
        removed_elements=len(removed_keys),
        reduction_ratio=reduction_ratio,
    )

    logger.debug(f'📊 [delta_observation] {stats}')

    return delta_observation, stats


class ObservationTracker:
    """Tracks observations across steps to compute deltas.

    Features:
    - Computes incremental changes between observations
    - Automatically detects URL changes and resets delta mode
    - Tracks statistics about token savings

    Usage:
        tracker = ObservationTracker()

        # Step 1
        obs1 = get_observation(...)
        delta1, stats1 = tracker.update(obs1)  # Returns full observation

        # Step 2
        obs2 = get_observation(...)
        delta2, stats2 = tracker.update(obs2)  # Returns delta
    """

    def __init__(
        self,
        enable_delta: bool = True,
        min_similarity_for_delta: float = 0.3,
        max_unchanged_display: int = 10,  # Increased from 5
    ):
        """
        Args:
            enable_delta: Whether to compute deltas
            min_similarity_for_delta: Minimum similarity ratio to use delta mode
            max_unchanged_display: Max unchanged elements to show
        """
        self.enable_delta = enable_delta
        self.min_similarity_for_delta = min_similarity_for_delta
        self.max_unchanged_display = max_unchanged_display
        self.previous_observation: str | None = None
        self.previous_url: str | None = None
        self.previous_title: str | None = None
        self.total_tokens_saved: int = 0
        self.delta_count: int = 0
        self.url_change_count: int = 0

    def _check_page_change(self, current_observation: str) -> bool:
        """Check if the page has changed significantly (URL or title).

        Args:
            current_observation: Current observation text

        Returns:
            True if page has changed, False otherwise
        """
        current_url = extract_url_from_observation(current_observation)
        current_title = extract_title_from_observation(current_observation)

        # URL change detection
        if self.previous_url and current_url:
            # Normalize URLs for comparison (remove trailing slashes, query params variations)
            prev_base = self.previous_url.split('?')[0].rstrip('/')
            curr_base = current_url.split('?')[0].rstrip('/')

            if prev_base != curr_base:
                logger.debug(f'📊 [delta_observation] URL changed: {self.previous_url[:50]}... → {current_url[:50]}...')
                self.url_change_count += 1
                return True

        # Significant title change detection (for SPAs that don't change URL)
        if self.previous_title and current_title and self.previous_title != current_title:
            # Check if it's a significant change (not just minor update)
            similarity = SequenceMatcher(None, self.previous_title, current_title).ratio()
            if similarity < 0.5:
                logger.debug(
                    f'📊 [delta_observation] Title changed significantly: '
                    f"'{self.previous_title[:30]}...' → '{current_title[:30]}...'"
                )
                return True

        return False

    def update(self, current_observation: str) -> tuple[str, DeltaStats | None]:
        """Update tracker with new observation and return delta or full observation.

        Args:
            current_observation: The current observation

        Returns:
            Tuple of (observation_to_use, delta_stats)
        """
        # Extract current page info
        current_url = extract_url_from_observation(current_observation)
        current_title = extract_title_from_observation(current_observation)

        # Check if delta mode should be disabled
        if not self.enable_delta or self.previous_observation is None:
            self.previous_observation = current_observation
            self.previous_url = current_url
            self.previous_title = current_title
            return current_observation, None

        # Check for page change - reset delta if page changed
        if self._check_page_change(current_observation):
            logger.debug('📊 [delta_observation] Page changed, resetting tracker and using full observation')
            self.previous_observation = current_observation
            self.previous_url = current_url
            self.previous_title = current_title
            return current_observation, None

        # Check similarity to decide if delta is appropriate
        # Compare more of the observation for better accuracy
        compare_length = min(10000, len(self.previous_observation), len(current_observation))
        similarity = SequenceMatcher(
            None, self.previous_observation[:compare_length], current_observation[:compare_length]
        ).quick_ratio()

        if similarity < self.min_similarity_for_delta:
            # Pages are too different, send full observation
            logger.debug(f'📊 [delta_observation] Similarity {similarity:.2f} < threshold, using full observation')
            self.previous_observation = current_observation
            self.previous_url = current_url
            self.previous_title = current_title
            return current_observation, None

        # Compute delta
        delta_observation, stats = compute_delta(
            self.previous_observation,
            current_observation,
            max_unchanged_display=self.max_unchanged_display,
        )

        # Update state
        self.previous_observation = current_observation
        self.previous_url = current_url
        self.previous_title = current_title

        if stats and stats.reduction_ratio > 0.1:  # Only use delta if it saves >10%
            self.total_tokens_saved += stats.original_tokens - stats.delta_tokens
            self.delta_count += 1
            return delta_observation, stats
        return current_observation, None

    def reset(self) -> None:
        """Reset the tracker (e.g., after navigation to a new page)."""
        self.previous_observation = None
        self.previous_url = None
        self.previous_title = None

    def get_summary(self) -> dict:
        """Get summary of delta computation."""
        return {
            'delta_count': self.delta_count,
            'total_tokens_saved': self.total_tokens_saved,
            'url_change_count': self.url_change_count,
        }
