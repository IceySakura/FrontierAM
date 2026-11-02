"""DOM Tree (AXTree) Compression Module.

This module provides intelligent compression of accessibility trees to reduce
token usage while preserving critical information for web automation tasks.

Key strategies:
1. Remove non-interactive elements (none, InlineTextBox)
2. Collapse deep nesting levels
3. Deduplicate repeated static text
4. Prioritize elements with bid (browser IDs)
5. Truncate long text values
"""

import logging
import re

from dataclasses import dataclass
from typing import Literal

from nexus.extensions.agents.browser_gym.token_counting import (
    count_tokens_with_fallback,
)


logger = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    return count_tokens_with_fallback(text, chars_per_token=2)


@dataclass
class CompressionStats:
    """Statistics about DOM compression."""

    original_chars: int
    compressed_chars: int
    original_lines: int
    compressed_lines: int
    elements_removed: int
    compression_ratio: float
    original_tokens: int = 0
    compressed_tokens: int = 0

    def __str__(self) -> str:
        return (
            f'Compression: {self.original_chars:,} → {self.compressed_chars:,} chars '
            f'({self.compression_ratio:.1%} reduction), '
            f'{self.original_lines} → {self.compressed_lines} lines, '
            f'{self.original_tokens:,} → {self.compressed_tokens:,} tokens'
        )


# Element types that are essential for interaction
INTERACTIVE_ELEMENTS = {
    'button',
    'link',
    'textbox',
    'input',
    'checkbox',
    'combobox',
    'select',
    'searchbox',
    'spinbutton',
    'slider',
    'switch',
    'radio',
    'menuitem',
    'menuitemcheckbox',
    'menuitemradio',
    'tab',
    'option',
    'listbox',
}

# Element types that provide structural context
STRUCTURAL_ELEMENTS = {
    'heading',
    'navigation',
    'main',
    'banner',
    'contentinfo',
    'article',
    'section',
    'form',
    'dialog',
    'alertdialog',
    'menu',
    'menubar',
    'toolbar',
    'table',
    'row',
    'cell',
    'list',
    'listitem',
    'tree',
    'treeitem',
    'RootWebArea',
}

# Element types that can be removed safely
REMOVABLE_ELEMENTS = {
    'none',
    'InlineTextBox',
    'LineBreak',
    'paragraph',
    'separator',
}

# Element types to keep but simplify
SIMPLIFY_ELEMENTS = {
    'generic',
    'group',
    'region',
}


@dataclass
class ParsedElement:
    """A parsed accessibility tree element."""

    indent: int
    bid: str | None
    element_type: str
    text: str
    attributes: dict[str, str]
    original_line: str


def parse_axtree_line(line: str) -> ParsedElement | None:
    """Parse a single line from the accessibility tree."""
    if not line.strip():
        return None

    # Calculate indent (tab-based)
    indent = 0
    stripped = line
    while stripped.startswith('\t'):
        indent += 1
        stripped = stripped[1:]

    stripped = stripped.strip()
    if not stripped:
        return None

    # Extract bid if present: [123] or [bid=xxx]
    bid = None
    bid_match = re.match(r'\[(\d+)\]\s*', stripped)
    if bid_match:
        bid = bid_match.group(1)
        stripped = stripped[bid_match.end() :]
    else:
        bid_match = re.match(r'\[bid=([^\]]+)\]\s*', stripped)
        if bid_match:
            bid = bid_match.group(1)
            stripped = stripped[bid_match.end() :]

    # Extract element type (first word)
    type_match = re.match(r'(\w+)', stripped)
    if not type_match:
        return ParsedElement(
            indent=indent,
            bid=bid,
            element_type='unknown',
            text=stripped,
            attributes={},
            original_line=line,
        )

    element_type = type_match.group(1)
    rest = stripped[type_match.end() :].strip()

    # Extract quoted text
    text = ''
    text_match = re.search(r"'([^']*)'|\"([^\"]*)\"", rest)
    if text_match:
        text = text_match.group(1) or text_match.group(2) or ''

    # Extract attributes
    attributes = {}
    attr_patterns = [
        (r"url='([^']*)'", 'url'),
        (r'url="([^"]*)"', 'url'),
        (r'\bfocused\b', 'focused'),
        (r'\bdisabled\b', 'disabled'),
        (r'\bchecked\b', 'checked'),
        (r'\bselected\b', 'selected'),
        (r"orientation='([^']*)'", 'orientation'),
    ]
    for pattern, attr_name in attr_patterns:
        match = re.search(pattern, rest)
        if match:
            if match.lastindex:
                attributes[attr_name] = match.group(1)
            else:
                attributes[attr_name] = 'true'

    return ParsedElement(
        indent=indent,
        bid=bid,
        element_type=element_type,
        text=text,
        attributes=attributes,
        original_line=line,
    )


def compress_light(elements: list[ParsedElement], preserve_bids: bool) -> list[ParsedElement]:
    """Light compression: only remove noise elements."""
    result = []

    for elem in elements:
        # Remove clear noise elements
        if elem.element_type in REMOVABLE_ELEMENTS and not (preserve_bids and elem.bid):
            continue

        result.append(elem)

    return result


def compress_medium(elements: list[ParsedElement], preserve_bids: bool) -> list[ParsedElement]:
    """Medium compression: remove noise + simplify structure."""
    result = []
    seen_texts = set()  # For deduplication

    for elem in elements:
        # Remove noise elements
        if elem.element_type in REMOVABLE_ELEMENTS and not (preserve_bids and elem.bid):
            continue

        # Skip duplicate StaticText with same content
        if elem.element_type == 'StaticText':
            text_key = elem.text[:100]  # Use first 100 chars as key
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

        # Truncate long text values
        if len(elem.text) > 100:
            elem = ParsedElement(
                indent=elem.indent,
                bid=elem.bid,
                element_type=elem.element_type,
                text=elem.text[:100] + '...',
                attributes=elem.attributes,
                original_line=elem.original_line,
            )

        # Skip generic elements without bid at deep levels
        if elem.element_type in SIMPLIFY_ELEMENTS and elem.indent > 8 and not elem.bid:
            continue

        result.append(elem)

    return result


def compress_aggressive(elements: list[ParsedElement], preserve_bids: bool) -> list[ParsedElement]:
    """Aggressive compression: keep only essential interactive elements."""
    result = []
    seen_texts = set()

    # First pass: identify all interactive elements
    interactive_indices = set()
    for i, elem in enumerate(elements):
        if elem.element_type.lower() in INTERACTIVE_ELEMENTS or elem.bid:
            interactive_indices.add(i)

    # Second pass: keep interactive elements and their immediate parents
    for i, elem in enumerate(elements):
        # Always keep elements with bid
        if preserve_bids and elem.bid:
            result.append(simplify_element(elem))
            continue

        # Keep interactive elements
        if elem.element_type.lower() in INTERACTIVE_ELEMENTS:
            result.append(simplify_element(elem))
            continue

        # Keep structural elements (but simplified)
        if elem.element_type in STRUCTURAL_ELEMENTS:
            # Only keep if it has children that are interactive
            has_interactive_child = False
            for j in range(i + 1, min(i + 50, len(elements))):
                if elements[j].indent <= elem.indent:
                    break
                if j in interactive_indices:
                    has_interactive_child = True
                    break

            if has_interactive_child or elem.element_type == 'RootWebArea':
                result.append(simplify_element(elem))
            continue

        # Keep StaticText only if unique and short
        if elem.element_type == 'StaticText' and elem.text:
            text_key = elem.text[:50]
            if text_key not in seen_texts and len(elem.text) < 200:
                seen_texts.add(text_key)
                # Keep as context but truncate
                simplified = simplify_element(elem)
                if len(simplified.text) > 50:
                    simplified = ParsedElement(
                        indent=simplified.indent,
                        bid=simplified.bid,
                        element_type=simplified.element_type,
                        text=simplified.text[:50] + '...',
                        attributes=simplified.attributes,
                        original_line=simplified.original_line,
                    )
                result.append(simplified)

    return result


def simplify_element(elem: ParsedElement) -> ParsedElement:
    """Simplify an element by truncating text and removing less important attributes."""
    # Truncate long text
    text = elem.text
    if len(text) > 80:
        text = text[:80] + '...'

    # Keep only important attributes
    important_attrs = {'url', 'focused', 'disabled', 'checked', 'selected'}
    attrs = {k: v for k, v in elem.attributes.items() if k in important_attrs}

    return ParsedElement(
        indent=elem.indent,
        bid=elem.bid,
        element_type=elem.element_type,
        text=text,
        attributes=attrs,
        original_line=elem.original_line,
    )


def reconstruct_axtree(elements: list[ParsedElement]) -> str:
    """Reconstruct accessibility tree text from parsed elements."""
    lines = []

    for elem in elements:
        # Build the line
        indent = '\t' * elem.indent

        parts = []
        if elem.bid:
            parts.append(f'[{elem.bid}]')
        parts.append(elem.element_type)

        if elem.text:
            parts.append(f"'{elem.text}'")

        # Add important attributes
        if elem.attributes.get('focused'):
            parts.append('focused')
        if elem.attributes.get('disabled'):
            parts.append('disabled')
        if elem.attributes.get('checked'):
            parts.append('checked')
        if elem.attributes.get('selected'):
            parts.append('selected')
        if elem.attributes.get('url'):
            # Truncate URLs
            url = elem.attributes['url']
            if len(url) > 80:
                url = url[:77] + '...'
            parts.append(f"url='{url}'")

        line = indent + ' '.join(parts)
        lines.append(line)

    return '\n'.join(lines)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    lines = text.split('\n')

    # Binary search for the right cutoff
    low, high = 0, len(lines)

    while low < high:
        mid = (low + high + 1) // 2
        subset = '\n'.join(lines[:mid])
        if count_tokens(subset) <= max_tokens:
            low = mid
        else:
            high = mid - 1

    result = '\n'.join(lines[:low])
    if low < len(lines):
        result += f'\n[... {len(lines) - low} more elements truncated ...]'

    return result


def compress_axtree(
    axtree: str,
    max_tokens: int = 30000,
    compression_level: Literal['light', 'medium', 'aggressive'] = 'medium',
    preserve_all_bids: bool = True,
) -> tuple[str, CompressionStats]:
    """
    Compress an accessibility tree while preserving essential information.

    Args:
        axtree: The accessibility tree text to compress
        max_tokens: Maximum tokens for the compressed output
        compression_level: How aggressively to compress
            - 'light': Remove only noise elements (none, InlineTextBox)
            - 'medium': Also simplify deep nesting and truncate text
            - 'aggressive': Maximum compression, may lose some context
        preserve_all_bids: Always keep elements with bid identifiers

    Returns:
        Tuple of (compressed_axtree, compression_stats)
    """
    original_chars = len(axtree)
    original_lines = len(axtree.strip().split('\n'))
    original_tokens = count_tokens(axtree)

    lines = axtree.strip().split('\n')

    # Phase 1: Parse lines into structured elements
    parsed_lines = []
    for line in lines:
        parsed = parse_axtree_line(line)
        if parsed:
            parsed_lines.append(parsed)

    # Phase 2: Apply compression based on level
    if compression_level == 'light':
        compressed = compress_light(parsed_lines, preserve_all_bids)
    elif compression_level == 'medium':
        compressed = compress_medium(parsed_lines, preserve_all_bids)
    else:  # aggressive
        compressed = compress_aggressive(parsed_lines, preserve_all_bids)

    # Phase 3: Reconstruct the tree
    result = reconstruct_axtree(compressed)

    # Phase 4: Truncate if still over limit
    current_tokens = count_tokens(result)
    if current_tokens > max_tokens:
        result = truncate_to_tokens(result, max_tokens)

    # Calculate stats
    compressed_chars = len(result)
    compressed_lines = len(result.strip().split('\n'))
    compressed_tokens = count_tokens(result)
    elements_removed = original_lines - compressed_lines
    compression_ratio = 1 - (compressed_chars / original_chars) if original_chars > 0 else 0

    stats = CompressionStats(
        original_chars=original_chars,
        compressed_chars=compressed_chars,
        original_lines=original_lines,
        compressed_lines=compressed_lines,
        elements_removed=elements_removed,
        compression_ratio=compression_ratio,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
    )

    return result, stats


def compress_observation(
    observation_text: str,
    max_tokens: int = 30000,
    compression_level: Literal['light', 'medium', 'aggressive'] = 'medium',
) -> tuple[str, CompressionStats | None]:
    """
    Compress a full observation (including URL, title, and AXTree).

    Args:
        observation_text: The full observation text with URL, title, and accessibility tree
        max_tokens: Maximum tokens for the result
        compression_level: Compression aggressiveness

    Returns:
        Tuple of (compressed_observation, stats) or (original, None) if no compression needed
    """
    # Check if compression is needed
    current_tokens = count_tokens(observation_text)
    if current_tokens <= max_tokens:
        return observation_text, None

    # Split observation into parts
    parts = observation_text.split('## Accessibility Tree')
    if len(parts) < 2:
        # No AXTree found, just truncate
        return truncate_to_tokens(observation_text, max_tokens), None

    header = parts[0]
    axtree_section = parts[1]

    # Further split to get action history if present
    axtree_parts = axtree_section.split('## Action History')
    axtree = axtree_parts[0]
    action_history = axtree_parts[1] if len(axtree_parts) > 1 else ''

    # Calculate budget for AXTree (leave room for header and history)
    header_tokens = count_tokens(header)
    history_tokens = count_tokens(action_history) if action_history else 0
    axtree_budget = max_tokens - header_tokens - history_tokens - 500  # 500 token buffer

    if axtree_budget < 5000:
        # Very tight budget, use aggressive compression
        compression_level = 'aggressive'
        axtree_budget = max(5000, max_tokens // 2)

    # Compress the AXTree
    compressed_axtree, stats = compress_axtree(
        axtree,
        max_tokens=axtree_budget,
        compression_level=compression_level,
    )

    # Reconstruct the observation
    result_parts = [header, '## Accessibility Tree', compressed_axtree]
    if action_history:
        result_parts.extend(['## Action History', action_history])

    result = '\n'.join(result_parts)

    logger.debug(f'🗜️ [dom_compression] {stats}')

    return result, stats


def get_recommended_compression_level(
    current_tokens: int,
    max_tokens: int,
) -> Literal['light', 'medium', 'aggressive']:
    """Get recommended compression level based on token ratio."""
    ratio = current_tokens / max_tokens

    if ratio < 1.5:
        return 'light'
    if ratio < 3.0:
        return 'medium'
    return 'aggressive'


def should_compress(text: str, threshold_tokens: int = 20000) -> bool:
    """Check if text should be compressed."""
    return count_tokens(text) > threshold_tokens
