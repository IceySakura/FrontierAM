"""Shared errors for base agents and their extension tools."""

NON_RETRYABLE_PROVIDER_ERROR_MARKERS = ('no_budget_error', '预算不足')


class NonRetryableProviderError(RuntimeError):
    """Provider rejected the request with an error that retries cannot fix."""


def is_non_retryable_provider_error(error_text: str) -> bool:
    """Return True for provider errors that should bypass retry loops."""
    return any(marker in error_text for marker in NON_RETRYABLE_PROVIDER_ERROR_MARKERS)


def is_non_retryable_provider_exception(exc: BaseException) -> bool:
    """Return True when an exception represents a non-retryable provider error."""
    return isinstance(exc, NonRetryableProviderError) or is_non_retryable_provider_error(str(exc))


def raise_for_non_retryable_provider_error(error_text: str) -> None:
    """Raise for provider errors that should bypass retry loops."""
    if is_non_retryable_provider_error(error_text):
        raise NonRetryableProviderError(f'Provider returned a non-retryable error: {error_text}')
