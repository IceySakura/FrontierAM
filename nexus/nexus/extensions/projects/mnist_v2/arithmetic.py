"""Provide exact integer arithmetic for the private calculator operator."""

from functools import reduce


def calculate_at(a: int, b: int) -> int:
    """Apply the calculator's private binary @ operator."""
    return a + (b if b % 2 == 0 else -b)


def calculate_at_chain(numbers: list[int]) -> int:
    """Evaluate a nonempty list left to right in a single tool invocation."""
    if not numbers:
        raise ValueError('Expected at least one integer')
    return reduce(calculate_at, numbers)
