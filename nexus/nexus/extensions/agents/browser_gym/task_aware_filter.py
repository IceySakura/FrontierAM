"""Task-Aware DOM Filtering Module.

This module intelligently filters DOM elements based on the current task goal,
removing elements that are unlikely to be relevant to the task.

Key strategies:
1. Extract keywords from task goal
2. Score elements based on relevance to task
3. Preserve elements that match task-related patterns
4. Remove clearly irrelevant sections (ads, footers, etc.)
"""

import logging
import re

from dataclasses import dataclass

from nexus.extensions.agents.browser_gym.token_counting import (
    count_tokens_with_fallback,
)


logger = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    return count_tokens_with_fallback(text, chars_per_token=4)


@dataclass
class FilterStats:
    """Statistics about task-aware filtering."""

    original_elements: int
    filtered_elements: int
    original_tokens: int
    filtered_tokens: int
    task_keywords: list[str]
    matched_elements: int
    reduction_ratio: float

    def __str__(self) -> str:
        return (
            f'TaskFilter: {self.original_elements} → {self.filtered_elements} elements '
            f'({self.original_tokens:,} → {self.filtered_tokens:,} tokens, '
            f'{self.reduction_ratio:.1%} reduction), '
            f'keywords={self.task_keywords[:5]}, matched={self.matched_elements}'
        )


# Task type patterns and their associated element priorities
TASK_PATTERNS = {
    'search': {
        'keywords': ['search', 'find', 'look for', 'locate', 'query'],
        'priority_elements': ['searchbox', 'textbox', 'input', 'button', 'link'],
        'priority_texts': ['search', 'find', 'go', 'submit'],
    },
    'login': {
        'keywords': ['login', 'log in', 'sign in', 'authenticate', 'password', 'username'],
        'priority_elements': ['textbox', 'input', 'button', 'form', 'link'],
        'priority_texts': ['login', 'sign in', 'password', 'email', 'username', 'submit'],
    },
    'purchase': {
        'keywords': ['buy', 'purchase', 'add to cart', 'checkout', 'order', 'price'],
        'priority_elements': ['button', 'link', 'textbox', 'combobox', 'listbox'],
        'priority_texts': ['add to cart', 'buy', 'checkout', 'price', 'quantity', 'order'],
    },
    'navigate': {
        'keywords': ['go to', 'navigate', 'open', 'click', 'visit'],
        'priority_elements': ['link', 'button', 'navigation', 'menu', 'menuitem'],
        'priority_texts': [],  # Any navigation target
    },
    'fill_form': {
        'keywords': ['fill', 'enter', 'type', 'input', 'submit', 'form'],
        'priority_elements': ['textbox', 'input', 'combobox', 'checkbox', 'radio', 'button', 'form'],
        'priority_texts': ['submit', 'save', 'send', 'confirm'],
    },
    'read_info': {
        'keywords': ['what is', 'how many', 'find the', 'tell me', 'show me', 'list'],
        'priority_elements': ['heading', 'StaticText', 'link', 'table', 'cell', 'listitem'],
        'priority_texts': [],  # Depends on specific query
    },
    'settings': {
        'keywords': ['setting', 'configure', 'change', 'update', 'edit', 'modify'],
        'priority_elements': ['textbox', 'checkbox', 'switch', 'radio', 'combobox', 'button'],
        'priority_texts': ['save', 'apply', 'update', 'settings'],
    },
}

# Elements that are typically irrelevant regardless of task
IRRELEVANT_PATTERNS = [
    r'advertisement',
    r'sponsored',
    r'cookie\s*(policy|consent|notice)',
    r'privacy\s*policy',
    r'terms\s*(of\s*service|and\s*conditions)',
    r'newsletter',
    r'subscribe\s*(to\s*our)?',
    r'follow\s*us',
    r'social\s*media',
    r'facebook|twitter|instagram|linkedin',
    r'copyright\s*©',
    r'all\s*rights\s*reserved',
]

# Compiled irrelevant patterns
IRRELEVANT_REGEX = re.compile('|'.join(IRRELEVANT_PATTERNS), re.IGNORECASE)


@dataclass
class ParsedElement:
    """A parsed accessibility tree element."""

    indent: int
    bid: str | None
    element_type: str
    text: str
    attributes: dict[str, str]
    original_line: str
    relevance_score: float = 0.0


def parse_axtree_line(line: str) -> ParsedElement | None:
    """Parse a single line from the accessibility tree."""
    if not line.strip():
        return None

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
            indent=indent,
            bid=bid,
            element_type='unknown',
            text=stripped,
            attributes={},
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
            attributes[attr_name] = match.group(1) if match.lastindex else 'true'

    return ParsedElement(
        indent=indent,
        bid=bid,
        element_type=element_type,
        text=text,
        attributes=attributes,
        original_line=line,
    )


def extract_task_keywords(task_goal: str) -> list[str]:
    """Extract important keywords from task goal.

    Enhanced to extract:
    1. Quoted strings (highest priority)
    2. Proper nouns (capitalized words)
    3. Numbers and prices
    4. URL path keywords
    5. Regular keywords after stop word filtering

    Args:
        task_goal: The task description/goal

    Returns:
        List of relevant keywords
    """
    # Normalize and tokenize
    text = task_goal.lower()
    keywords = []

    # 1. Extract quoted strings as highest priority keywords
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', task_goal)
    for q1, q2 in quoted:
        keyword = q1 or q2
        if keyword and keyword.lower() not in keywords:
            keywords.insert(0, keyword.lower())  # Quoted strings are high priority

    # 2. Extract proper nouns (capitalized words that aren't at sentence start)
    # This catches product names, company names, etc.
    # Note: Use two separate patterns to avoid variable-width look-behind (not supported in Python 3.13+)
    proper_nouns_after_punct = re.findall(r'(?<=[.!?] )[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', task_goal)
    proper_nouns_after_space = re.findall(r'(?<= )[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', task_goal)
    proper_nouns = proper_nouns_after_punct + proper_nouns_after_space
    # Also get standalone capitalized words
    standalone_caps = re.findall(r'\b[A-Z][a-z]{2,}\b', task_goal)
    for noun in proper_nouns + standalone_caps:
        noun_lower = noun.lower()
        if noun_lower not in keywords and len(noun_lower) > 2:
            keywords.append(noun_lower)

    # 3. Extract numbers and prices (might be prices, quantities, IDs, etc.)
    prices = re.findall(r'\$\d+(?:\.\d+)?', task_goal)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', task_goal)
    keywords.extend(prices)
    keywords.extend([n for n in numbers if n not in keywords])

    # 4. Extract URL path keywords
    urls = re.findall(r'https?://[^\s]+', task_goal)
    for url in urls:
        # Extract meaningful path segments
        path_parts = url.split('/')
        for part in path_parts[-4:]:  # Last 4 segments
            # Clean up query params and anchors
            clean_part = part.split('?')[0].split('#')[0]
            if clean_part and len(clean_part) > 2 and clean_part.lower() not in keywords:
                keywords.append(clean_part.lower())

    # 5. Remove common stop words and extract remaining keywords
    stop_words = {
        'the',
        'a',
        'an',
        'is',
        'are',
        'was',
        'were',
        'be',
        'been',
        'being',
        'have',
        'has',
        'had',
        'do',
        'does',
        'did',
        'will',
        'would',
        'could',
        'should',
        'may',
        'might',
        'must',
        'shall',
        'can',
        'need',
        'dare',
        'ought',
        'used',
        'to',
        'of',
        'in',
        'for',
        'on',
        'with',
        'at',
        'by',
        'from',
        'as',
        'into',
        'through',
        'during',
        'before',
        'after',
        'above',
        'below',
        'between',
        'under',
        'again',
        'further',
        'then',
        'once',
        'here',
        'there',
        'when',
        'where',
        'why',
        'how',
        'all',
        'each',
        'few',
        'more',
        'most',
        'other',
        'some',
        'such',
        'no',
        'nor',
        'not',
        'only',
        'own',
        'same',
        'so',
        'than',
        'too',
        'very',
        'just',
        'and',
        'but',
        'if',
        'or',
        'because',
        'until',
        'while',
        'this',
        'that',
        'these',
        'those',
        'what',
        'which',
        'who',
        'whom',
        'i',
        'me',
        'my',
        'myself',
        'we',
        'our',
        'ours',
        'you',
        'your',
        'yours',
        'he',
        'him',
        'his',
        'she',
        'her',
        'hers',
        'it',
        'its',
        'they',
        'them',
        'their',
        'click',
        'page',
        'website',
        'site',
    }

    # Extract words
    words = re.findall(r'\b[a-z]+\b', text)

    # Filter stop words and short words, add to keywords
    for w in words:
        if w not in stop_words and len(w) > 2 and w not in keywords:
            keywords.append(w)

    # 6. Extract multi-word phrases (bigrams) that might be important
    # Look for patterns like "red dress", "shopping cart", etc.
    bigrams = re.findall(r'\b([a-z]+\s+[a-z]+)\b', text)
    for bigram in bigrams:
        words_in_bigram = bigram.split()
        if all(w not in stop_words for w in words_in_bigram) and bigram not in keywords:
            keywords.append(bigram)

    return keywords[:30]  # Increased from 20 to 30


def detect_task_type(task_goal: str) -> list[str]:
    """Detect task type(s) from the goal.

    Args:
        task_goal: The task description

    Returns:
        List of detected task types
    """
    text = task_goal.lower()
    detected = []

    for task_type, patterns in TASK_PATTERNS.items():
        for keyword in patterns['keywords']:
            if keyword in text:
                detected.append(task_type)
                break

    return detected if detected else ['navigate']  # Default to navigate


def score_element(
    element: ParsedElement,
    task_keywords: list[str],
    task_types: list[str],
    focused_element_bid: str | None = None,
) -> float:
    """Score an element's relevance to the task.

    Args:
        element: The element to score
        task_keywords: Keywords extracted from task
        task_types: Detected task types
        focused_element_bid: BID of the currently focused element

    Returns:
        Relevance score (0.0 - 1.0)
    """
    score = 0.0

    # Elements with bid are always somewhat important
    if element.bid:
        score += 0.3

        # Currently focused element is very important
        if focused_element_bid and element.bid == focused_element_bid:
            score += 0.5

    # Check element type against task priorities
    elem_type_lower = element.element_type.lower()
    for task_type in task_types:
        if task_type in TASK_PATTERNS:
            if elem_type_lower in TASK_PATTERNS[task_type]['priority_elements']:
                score += 0.2

            # Check text against priority texts
            elem_text_lower = element.text.lower()
            for priority_text in TASK_PATTERNS[task_type]['priority_texts']:
                if priority_text in elem_text_lower:
                    score += 0.3
                    break

    # Check against task keywords
    elem_text_lower = element.text.lower()
    for keyword in task_keywords:
        if keyword.lower() in elem_text_lower:
            score += 0.4
            break

    # Check URL for keywords (for links)
    if 'url' in element.attributes:
        url_lower = element.attributes['url'].lower()
        for keyword in task_keywords:
            if keyword.lower() in url_lower:
                score += 0.2
                break

    # Penalize clearly irrelevant content
    if IRRELEVANT_REGEX.search(element.text):
        score -= 0.5

    # Structural elements are somewhat important for context
    if element.element_type in {'heading', 'navigation', 'main', 'form', 'table'}:
        score += 0.1

    # Interactive elements without other score still get some relevance
    if elem_type_lower in {'button', 'link', 'textbox', 'input', 'checkbox', 'combobox'}:
        score = max(score, 0.2)

    return min(1.0, max(0.0, score))


def filter_axtree(
    axtree: str,
    task_goal: str,
    relevance_threshold: float = 0.10,  # Lowered from 0.15 to be less aggressive
    max_elements: int | None = None,
    preserve_structure: bool = True,
    focused_element_bid: str | None = None,
    min_elements: int = 20,  # NEW: Minimum elements to keep
) -> tuple[str, FilterStats]:
    """Filter accessibility tree based on task relevance.

    Args:
        axtree: The accessibility tree text
        task_goal: The task goal/description
        relevance_threshold: Minimum relevance score to keep element
        max_elements: Maximum elements to keep (None = no limit)
        preserve_structure: Keep structural context for relevant elements
        focused_element_bid: BID of currently focused element
        min_elements: Minimum elements to keep even if below threshold

    Returns:
        Tuple of (filtered_axtree, stats)
    """
    original_tokens = count_tokens(axtree)
    lines = axtree.strip().split('\n')
    original_elements = len(lines)

    # Extract task info
    task_keywords = extract_task_keywords(task_goal)
    task_types = detect_task_type(task_goal)

    logger.debug(f'🎯 [task_filter] Task types: {task_types}, keywords: {task_keywords[:10]}')

    # Parse and score elements
    elements: list[ParsedElement] = []
    for line in lines:
        elem = parse_axtree_line(line)
        if elem:
            elem.relevance_score = score_element(elem, task_keywords, task_types, focused_element_bid)
            elements.append(elem)

    # Filter elements
    filtered_elements: list[ParsedElement] = []
    matched_count = 0

    # Build a set of relevant element indices
    relevant_indices = set()
    for i, elem in enumerate(elements):
        if elem.relevance_score >= relevance_threshold:
            relevant_indices.add(i)
            matched_count += 1

            # If preserving structure, also keep parent elements
            if preserve_structure and elem.indent > 0:
                # Find parent elements (lower indent levels)
                for j in range(i - 1, -1, -1):
                    parent = elements[j]
                    if parent.indent < elem.indent:
                        relevant_indices.add(j)
                        if parent.indent == 0:
                            break

    # NEW: If too few elements matched, add more by lowering threshold dynamically
    if len(relevant_indices) < min_elements:
        # Sort elements by score and add top ones
        sorted_indices = sorted(range(len(elements)), key=lambda i: elements[i].relevance_score, reverse=True)
        for idx in sorted_indices[:min_elements]:
            relevant_indices.add(idx)
            # Also add parents for structure
            if preserve_structure and elements[idx].indent > 0:
                for j in range(idx - 1, -1, -1):
                    parent = elements[j]
                    if parent.indent < elements[idx].indent:
                        relevant_indices.add(j)
                        if parent.indent == 0:
                            break

    # Collect filtered elements
    for i, elem in enumerate(elements):
        if i in relevant_indices:
            filtered_elements.append(elem)

    # Apply max_elements limit if specified
    if max_elements and len(filtered_elements) > max_elements:
        # Sort by relevance and keep top elements
        filtered_elements.sort(key=lambda e: -e.relevance_score)
        filtered_elements = filtered_elements[:max_elements]
        # Re-sort by original order (indent-based)
        filtered_elements.sort(key=lambda e: (e.indent, elements.index(e) if e in elements else 0))

    # Reconstruct tree
    filtered_lines = [elem.original_line.rstrip() for elem in filtered_elements]
    filtered_axtree = '\n'.join(filtered_lines)

    filtered_tokens = count_tokens(filtered_axtree)
    reduction_ratio = 1 - (filtered_tokens / original_tokens) if original_tokens > 0 else 0

    stats = FilterStats(
        original_elements=original_elements,
        filtered_elements=len(filtered_elements),
        original_tokens=original_tokens,
        filtered_tokens=filtered_tokens,
        task_keywords=task_keywords,
        matched_elements=matched_count,
        reduction_ratio=reduction_ratio,
    )

    logger.debug(f'🎯 [task_filter] {stats}')

    return filtered_axtree, stats


def filter_observation(
    observation: str,
    task_goal: str,
    relevance_threshold: float = 0.15,
    max_elements: int | None = None,
    preserve_structure: bool = True,
) -> tuple[str, FilterStats | None]:
    """Filter a full observation based on task relevance.

    Args:
        observation: Full observation text (URL, title, axtree, etc.)
        task_goal: The task goal
        relevance_threshold: Minimum relevance score
        max_elements: Maximum elements to keep
        preserve_structure: Keep structural context

    Returns:
        Tuple of (filtered_observation, stats)
    """
    # Split observation into parts
    parts = observation.split('## Accessibility Tree')
    if len(parts) < 2:
        return observation, None

    header = parts[0]
    axtree_section = parts[1]

    # Check for action history
    if '## Action History' in axtree_section:
        axtree_parts = axtree_section.split('## Action History')
        axtree = axtree_parts[0].strip()
        action_history = '## Action History' + axtree_parts[1]
    else:
        axtree = axtree_section.strip()
        action_history = ''

    # Filter the axtree
    filtered_axtree, stats = filter_axtree(
        axtree,
        task_goal,
        relevance_threshold=relevance_threshold,
        max_elements=max_elements,
        preserve_structure=preserve_structure,
    )

    # Reconstruct observation
    result_parts = [
        header,
        '## Accessibility Tree\n',
        filtered_axtree,
    ]
    if action_history:
        result_parts.append('\n\n')
        result_parts.append(action_history)

    return ''.join(result_parts), stats


class TaskAwareFilter:
    """Manages task-aware filtering across an agent session.

    Features:
    - Extracts keywords from task goal for relevance scoring
    - Detects task type to prioritize relevant element types
    - Adaptive threshold adjustment based on filtering results
    - Minimum element guarantee to prevent over-filtering

    Usage:
        filter = TaskAwareFilter(task_goal="Find the cheapest laptop")

        # Each step
        filtered_obs, stats = filter.filter(observation)
    """

    def __init__(
        self,
        task_goal: str,
        enable_filter: bool = True,
        relevance_threshold: float = 0.10,  # Lowered from 0.15
        max_elements: int | None = None,
        adaptive_threshold: bool = True,
        min_elements: int = 25,  # NEW: Minimum elements to always keep
    ):
        """
        Args:
            task_goal: The task goal/description
            enable_filter: Whether filtering is enabled
            relevance_threshold: Base relevance threshold
            max_elements: Maximum elements to keep
            adaptive_threshold: Dynamically adjust threshold based on results
            min_elements: Minimum elements to keep per filter
        """
        self.task_goal = task_goal
        self.enable_filter = enable_filter
        self.relevance_threshold = relevance_threshold
        self.initial_threshold = relevance_threshold  # Store initial for reference
        self.max_elements = max_elements
        self.adaptive_threshold = adaptive_threshold
        self.min_elements = min_elements

        # Pre-compute task info
        self.task_keywords = extract_task_keywords(task_goal)
        self.task_types = detect_task_type(task_goal)

        # Statistics
        self.filter_count = 0
        self.total_tokens_saved = 0
        self.threshold_adjustments = 0

    def filter(
        self,
        observation: str,
        current_focus_bid: str | None = None,
    ) -> tuple[str, FilterStats | None]:
        """Filter an observation.

        Args:
            observation: The observation to filter
            current_focus_bid: BID of currently focused element

        Returns:
            Tuple of (filtered_observation, stats)
        """
        if not self.enable_filter:
            return observation, None

        filtered_obs, stats = filter_observation(
            observation,
            self.task_goal,
            relevance_threshold=self.relevance_threshold,
            max_elements=self.max_elements,
        )

        if stats:
            self.filter_count += 1
            self.total_tokens_saved += stats.original_tokens - stats.filtered_tokens

            # Adaptive threshold adjustment with improved logic
            if self.adaptive_threshold:
                # If too few elements after filtering (excluding min_elements guarantee)
                if stats.matched_elements < 15:
                    # Lower threshold more aggressively
                    old_threshold = self.relevance_threshold
                    self.relevance_threshold = max(0.03, self.relevance_threshold - 0.03)
                    if old_threshold != self.relevance_threshold:
                        self.threshold_adjustments += 1
                        logger.debug(
                            f'🎯 [task_filter] Lowered threshold: {old_threshold:.2f} → {self.relevance_threshold:.2f} '
                            f'(only {stats.matched_elements} matched elements)'
                        )
                elif stats.reduction_ratio < 0.05:
                    # Not enough reduction, raise threshold slightly
                    old_threshold = self.relevance_threshold
                    self.relevance_threshold = min(0.3, self.relevance_threshold + 0.02)
                    if old_threshold != self.relevance_threshold:
                        self.threshold_adjustments += 1
                        logger.debug(
                            f'🎯 [task_filter] Raised threshold: {old_threshold:.2f} → {self.relevance_threshold:.2f} '
                            f'(reduction ratio only {stats.reduction_ratio:.1%})'
                        )

        return filtered_obs, stats

    def update_task_context(self, additional_keywords: list[str]) -> None:
        """Add additional keywords discovered during task execution."""
        for kw in additional_keywords:
            if kw.lower() not in [k.lower() for k in self.task_keywords]:
                self.task_keywords.append(kw)

    def get_summary(self) -> dict:
        """Get filtering summary."""
        return {
            'filter_count': self.filter_count,
            'total_tokens_saved': self.total_tokens_saved,
            'initial_threshold': self.initial_threshold,
            'final_threshold': self.relevance_threshold,
            'threshold_adjustments': self.threshold_adjustments,
            'task_keywords': self.task_keywords,
            'task_types': self.task_types,
        }
