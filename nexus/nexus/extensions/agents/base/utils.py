import copy
import json

from typing import Any


def parse_func_arguments(arguments: str | dict | None) -> dict:
    """Parse function arguments to dict.

    Args:
        arguments: JSON string, dict, or None/empty.

    Returns:
        Parsed dict.

    Raises:
        ValueError: When parsing fails.
    """
    if not arguments:
        return {}
    if isinstance(arguments, dict):
        return arguments

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON: {e} (input: {arguments!r})') from e

    if not isinstance(parsed, dict):
        raise ValueError(f'Expected dict, got {type(parsed).__name__}: {arguments!r}')

    return parsed


def format_to_user_message(messages: list[dict], separator: str = '\n') -> dict:
    """
    Combines a list of message dictionaries into a single User Message.

    This function iterates through a list of message dictionaries, formats their content
    based on role (e.g., System, User, Assistant), and combines them into
    a single string separated by the specified separator. This string then
    becomes the content of a new User Message.

    Args:
        messages: The list of message dictionaries to be converted.
        separator: The separator to be placed between messages, defaults to two newlines.

    Returns:
        dict: A dictionary containing the combined content of all input messages.

    Raises:
        ValueError: If the message list is empty or contains invalid dictionary structure.

    Examples:
        >>> msgs = [
        ...     {'role': 'system', 'content': 'You are a helpful assistant'},
        ...     {'role': 'user', 'content': 'Hello'},
        ...     {'role': 'assistant', 'content': 'Hi there!'},
        ...     {'role': 'user', 'content': 'How are you?'}
        ... ]
        >>> user_msg = format_to_user_message(msgs)
        >>> print(user_msg['content'])
        # Output:
        # System: You are a helpful assistant
        #
        # User: Hello
        #
        # Assistant: Hi there!
        #
        # User: How are you?
    """
    if not messages:
        raise ValueError('Messages list cannot be empty')

    content_parts = []

    # Iterate through each message in the list
    for message in messages:
        # Validate that each item is a dictionary with 'role' and 'content' keys
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError(f'Each message must be a dictionary with "role" and "content" keys, got {type(message)}')

        # Get the role name and capitalize the first letter
        role_name = message['role'].capitalize()

        # Process the message content if it exists
        content = message['content']
        if content:
            if isinstance(content, str):
                # Simple text content
                content_parts.append(f'{role_name}: {content}')
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text' and 'text' in item:
                            text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)

                if text_parts:
                    content_parts.append(f'{role_name}: {" ".join(text_parts)}')
            else:
                # Other types of content, convert to a string
                content_parts.append(f'{role_name}: {str(content)}')

    # Check if any valid content was found
    if not content_parts:
        raise ValueError('No valid content found in messages')

    # Join all content parts with the specified separator
    combined_content = separator.join(content_parts)
    return {'role': 'user', 'content': combined_content}


def extract_text_from_content(content: Any) -> str:
    """Extract text string from MessageContent (str or list of content parts).

    Args:
        content: MessageContent which can be str or list[dict]

    Returns:
        Extracted text as string
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    text_parts.append(item['text'])
            elif isinstance(item, str):
                text_parts.append(item)
        return ' '.join(text_parts) if text_parts else ''
    return str(content)


def extract_reasoning_and_signature(message: Any) -> tuple[Any, str | None]:
    """Extract reasoning content and Claude signature from message."""
    reasoning_content = getattr(message, 'reasoning_content', None)
    signature = None
    for tb in getattr(message, 'thinking_blocks', None) or []:
        sig = getattr(tb, 'signature', None)
        if sig:
            signature = sig
            break
    if not signature:
        psf = getattr(message, 'provider_specific_fields', None) or {}
        reasoning_blocks = psf.get('reasoningContentBlocks', []) if isinstance(psf, dict) else []
        for block in reasoning_blocks:
            sig = block.get('reasoningText', {}).get('signature')
            if sig:
                signature = sig
                break
    return reasoning_content, signature


def prepare_messages_for_thinking(messages: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    is_anthropic_model = 'api_aws_third_anthropic' in model_name
    converted = copy.deepcopy(messages)

    if is_anthropic_model:
        converted = prepare_messages_for_thinking_anthropic(converted)

    return converted


def prepare_messages_for_thinking_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert assistant reasoning fields to thinking content blocks."""
    for message in messages:
        role = message.get('role')
        role_value = getattr(role, 'value', role)
        if role_value != 'assistant':
            continue
        reasoning_content = message.pop('reasoning_content', None)
        signature = message.pop('signature', None)
        if not reasoning_content:
            continue
        thinking_block: dict[str, Any] = {'type': 'thinking', 'thinking': reasoning_content}
        if signature:
            thinking_block['signature'] = signature
        content = message.get('content')
        blocks: list[dict[str, Any]] = [thinking_block]
        if isinstance(content, str) and content:
            blocks.append({'type': 'text', 'text': content})
        message['content'] = blocks
    return messages
