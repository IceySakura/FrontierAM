# /// script
# dependencies = []
#
# [project]
# name = "chat_tracker"
# authors = [
#     { name = "tarschen" },
#     { name = "edwardswu" },
# ]
#
# [tool.nexus]
# span_processors = { chat_tracker = "nexus.trace.processors.chat_tracker" }
# run_parts = { chat_tracker = "nexus.trace.processors.chat_tracker" }
# ///

from __future__ import annotations

import json
import re

from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.runner.compose import run_part
from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.types import Span
from nexus.utils.common import serialize_object


# Tools tracking mode: global_first records the first tools globally,
# branch_last records tools per branch (last one in each branch).
# Future extensions: global_last, branch_first, none
ToolsTrackingMode = Literal['global_first', 'branch_last']


class ChatNode(BaseModel):
    """Represents a node in the chat tree structure."""

    generated_by: Literal['unknown', 'user', 'env', 'llm']
    generation_detail: Any = None
    tags: list[str] = Field(default_factory=list)
    message: dict[str, Any]
    children: list[ChatNode] = Field(default_factory=list, exclude=True)
    parent: ChatNode | None = Field(default=None, exclude=True)

    model_config = {'arbitrary_types_allowed': True}


@run_part('chat_tracker')
def chat_tracker_run_part(params: dict[str, Any]) -> dict[str, Any]:
    if params:
        raise ValueError(f'Run part chat_tracker does not accept parameters: {sorted(params)}')
    return {
        'processors': {
            'chat': {
                'name': 'chat_tracker',
                'args': {'tools_mode': 'branch_last'},
            },
        },
    }


@span_processor_registry.register('chat_tracker')
class ChatTrackerProcessor(SpanProcessor):
    """A processor that tracks LLM chat calls and exports trajectories.

    This processor listens for 'openai_completion' spans and builds a tree
    structure of chat messages. This is useful for analyzing conversation
    flows and exporting training data.

    The processor tracks:
    - Input messages leading to each LLM call
    - Output responses from the LLM
    - Tools definition from input

    Example usage:
        tracker = ChatTrackerProcessor(tools_mode='global_first')
        add_span_processor(tracker)

        # ... run your agent ...

        # Get result (tools and trajectories)
        result = tracker.get_result()
    """

    def __init__(self, tools_mode: ToolsTrackingMode = 'global_first'):
        """Initialize the ChatTrackerProcessor.

        Args:
            tools_mode: How to track tools in trajectories.
                - 'global_first': Record the first tools globally, shared across all branches.
                - 'branch_last': Record tools per branch, using the last tools in each branch.
        """
        self.tools_mode = tools_mode
        self._global_tools: list[dict[str, Any]] | None = None
        self.root = ChatNode(
            generated_by='unknown',
            generation_detail=None,
            tags=[],
            message={'role': 'system', 'content': ''},
            children=[],
            parent=None,
        )

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        """Called when a span is ended."""
        if span.name != 'openai_completion':
            return

        inputs = span.attributes.get('inputs', {})
        outputs = span.attributes.get('outputs')

        if not inputs:
            return

        messages = inputs.get('messages', [])
        if not messages:
            return

        # Extract tools from input
        tools = inputs.get('kwargs', {}).get('tools')

        # For global_first mode, record tools only once
        if self.tools_mode == 'global_first' and self._global_tools is None and tools:
            self._global_tools = tools

        phase = span.attributes.get('_chat_phase')
        if phase:
            inputs = {**inputs, 'tags': [phase]}

        self._track_llm_call(inputs, outputs, tools)

    _CCH_PATTERN = re.compile(r'cch=[^;]*;')

    def _normalize_content(self, content: Any) -> Any:
        """Normalize content for equality comparison.

        Strips the volatile `cch=...;` token from Claude Code SDK's
        `x-anthropic-billing-header` line, which changes on every call and
        otherwise prevents system-message deduplication across branches.

        Also flattens a pure-text content block list (``[{'type': 'text',
        'text': ...}, ...]``) to its concatenated text so that the same message
        compares equal whether a provider sent it as a plain string or as text
        blocks. Without this, a prefix recorded as ``str`` on the first call and
        as text blocks on later calls forks the tree and yields a truncated
        trajectory. Lists holding any non-text block (image, document, ...) are
        returned as-is so structurally different content never compares equal.
        Comparison only -- stored messages keep their original shape.
        """
        if isinstance(content, str):
            return self._CCH_PATTERN.sub('', content)
        if (
            isinstance(content, list)
            and content
            and all(
                isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str)
                for block in content
            )
        ):
            return self._CCH_PATTERN.sub('', ''.join(block['text'] for block in content))
        return content

    def _normalize_tool_arguments(self, arguments: Any) -> Any:
        """Normalize function arguments that may be JSON strings."""
        arguments = serialize_object(arguments)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return arguments
            return self._normalize_tool_arguments(parsed)
        if isinstance(arguments, dict):
            return {key: self._normalize_tool_arguments(arguments[key]) for key in sorted(arguments)}
        if isinstance(arguments, list):
            return [self._normalize_tool_arguments(item) for item in arguments]
        return arguments

    def _normalize_tool_call(self, tool_call: Any) -> Any:
        """Normalize provider-specific tool call decoration for equality checks."""
        tool_call = serialize_object(tool_call)
        if not isinstance(tool_call, dict):
            return tool_call

        normalized: dict[str, Any] = {}
        for key, value in tool_call.items():
            if key in {'id', 'index'}:
                continue
            if key == 'function' and isinstance(value, dict):
                function = dict(value)
                if 'arguments' in function:
                    function['arguments'] = self._normalize_tool_arguments(function['arguments'])
                normalized[key] = function
            else:
                normalized[key] = value
        return normalized

    def _tool_call_identities(self, message: dict) -> tuple[tuple[str, str], ...] | None:
        """Return stable ``(id, function_name)`` identities when all are present."""
        tool_calls = serialize_object(message.get('tool_calls'))
        if not isinstance(tool_calls, list) or not tool_calls:
            return None

        identities: list[tuple[str, str]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                return None
            call_id = tool_call.get('id')
            function = tool_call.get('function')
            function_name = function.get('name') if isinstance(function, dict) else None
            if call_id in (None, '') or function_name in (None, ''):
                return None
            identities.append((str(call_id), str(function_name)))
        return tuple(identities)

    def _reasoning_identities(self, message: dict) -> tuple[str, ...] | None:
        """Return stable reasoning-item ids when all are present.

        Anchor for the Responses path (Codex + gpt-5.x), where the SAME model response is
        recorded twice in two different shapes:

        1. First seen as ``choices[0].message`` straight off the response — text only,
           ``reasoning_content: ''``, and **no** ``tool_calls`` yet.
        2. Seen again in the next request's ``input_messages`` when Codex replays history —
           now carrying the ``tool_calls`` that the Responses ``function_call`` item became.

        ``_tool_call_identities`` cannot bridge these: shape 1 has no tool_calls at all, so it
        returns None and the ID fast path is skipped. The semantic fallback then compares dicts
        of different arity (``{role, content}`` vs ``{role, content, tool_calls}``) and declares
        them unequal — forking the tree and leaving the shape-1 branch as a truncated leaf that
        ``dump_trajectories`` emits as a bogus extra trajectory (observed: an assistant turn with
        ``finish_reason='stop'`` and no tool_calls, though the model did call a tool).

        ``reasoning_items[].id`` survives both shapes unchanged (verified on
        R2_codex0114_evohard3: ``rs_095761da3dadebf0006a9848de2dbc8196a73ce5a60d9b0397`` present
        and identical in both), so it identifies the underlying response.

        Extends MR !1198's "reconnect replays via stable identity" approach to the case where one
        side has no tool_calls at all — that MR only covered both-sides-have-tool_calls with
        rewritten arguments.
        """
        items = serialize_object(message.get('reasoning_items'))
        if not isinstance(items, list) or not items:
            return None

        ids: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                return None
            item_id = item.get('id')
            if item_id in (None, ''):
                return None
            ids.append(str(item_id))
        return tuple(ids)

    def _normalize_message_for_compare(self, message: dict) -> dict[str, Any]:
        role = message.get('role')
        content = self._normalize_content(message.get('content'))
        tool_calls = message.get('tool_calls')

        if role == 'assistant' and tool_calls:
            if content in (None, ''):
                content = None
            return {
                'role': role,
                'content': content,
                'tool_calls': [self._normalize_tool_call(tool_call) for tool_call in tool_calls],
            }

        return {'role': role, 'content': content}

    def _message_equal(self, message1: dict, message2: dict) -> bool:
        """Compare two messages for equality."""
        normalized1 = self._normalize_message_for_compare(message1)
        normalized2 = self._normalize_message_for_compare(message2)

        identities1 = self._tool_call_identities(message1)
        identities2 = self._tool_call_identities(message2)
        if identities1 is not None and identities1 == identities2:
            # Claude Code preserves tool-call IDs while canonicalizing executed
            # inputs (for example, adding ``replace_all=false`` or removing a
            # redundant ``cd /workspace &&``). Matching identities therefore
            # prove these are two representations of the same logical call.
            return normalized1['role'] == normalized2['role'] and normalized1['content'] == normalized2['content']

        # IDs may be absent or provider-reformatted, so retain the semantic
        # comparison fallback for equivalent calls from different providers.
        if normalized1 == normalized2:
            return True

        # Responses path: the same response can appear with and without tool_calls (see
        # _reasoning_identities). Reasoning-item ids survive both shapes, so equal ids plus
        # equal role/content prove these are one response, not two. Checked last so it only
        # ever merges what the strict comparison already rejected.
        reasoning1 = self._reasoning_identities(message1)
        reasoning2 = self._reasoning_identities(message2)
        if reasoning1 is not None and reasoning1 == reasoning2:
            return normalized1['role'] == normalized2['role'] and normalized1['content'] == normalized2['content']

        return False

    @staticmethod
    def _is_bare_assistant_snapshot(message: dict) -> bool:
        """True for an assistant message that carries no tool_calls.

        On the Responses path this is the shape a response has when it is first recorded from
        ``choices[0].message``: the ``function_call`` item has not been turned into ``tool_calls``
        yet. Verified across R1-R4 exports: such messages appear ONLY as the last message of a
        trajectory (349 occurrences, 0 in the middle), so upgrading one in place can never rewrite
        a message that already has a recorded continuation.
        """
        return message.get('role') == 'assistant' and not message.get('tool_calls')

    def _find_upgradable_snapshot(self, parent: ChatNode, message: dict) -> ChatNode | None:
        """Find a childless bare-assistant sibling that ``message`` is the tool_calls-bearing form of.

        Completes the fix started by ``_reasoning_identities``: that one bridges the two shapes only
        when both carry reasoning-item ids, but the raw snapshot often has neither ``reasoning_items``
        nor ``tool_calls`` (observed at every divergence point of R4). Here the anchor is structural
        instead of field-based -- same parent, same role/content, one side a childless leaf missing
        exactly the tool_calls -- which is what makes them the same response rather than two turns.

        Requiring ``not child.children`` keeps this conservative: a snapshot that already grew a
        continuation is a real turn boundary and is left alone.
        """
        if not message.get('tool_calls') or message.get('role') != 'assistant':
            return None
        content = self._normalize_content(message.get('content'))
        for child in parent.children:
            if child.children or not self._is_bare_assistant_snapshot(child.message):
                continue
            if self._normalize_content(child.message.get('content')) == content:
                return child
        return None

    @staticmethod
    def _append_child(parent: ChatNode, child: ChatNode) -> None:
        """Append a child and release data that is only needed while an LLM node is a leaf."""
        if parent.generated_by == 'llm' and parent.generation_detail:
            parent.generation_detail.pop('input_messages', None)
            parent.generation_detail.pop('output_message', None)
            parent.generation_detail.pop('tools', None)
        parent.children.append(child)

    def _track_llm_call(
        self, call_input: dict[str, Any], call_output: Any, tools: list[dict[str, Any]] | None = None
    ) -> None:
        """Track an LLM call in the chat tree.

        Args:
            call_input: The input to the LLM call (messages, model, etc.)
            call_output: The output from the LLM call
            tools: The tools used in this LLM call (if any)
        """
        output_message, trajectory_output_message = self._extract_output_message(call_output)
        if trajectory_output_message is None:
            return

        input_messages: list[dict] = call_input.get('messages', [])
        current_node = self.root

        # Walk through existing tree or create new nodes for input messages
        for message in input_messages:
            # Search for existing child with matching message
            found = False
            for child in current_node.children:
                if self._message_equal(child.message, message):
                    current_node = child
                    found = True
                    break

            if not found:
                # Before forking, check whether a childless sibling is the same response recorded
                # before its tool_calls were restored (Responses path). If so, upgrade that node in
                # place instead of growing a duplicate branch: the snapshot would otherwise stay a
                # leaf and be exported as an extra trajectory, truncated right where the model
                # actually called a tool.
                snapshot = self._find_upgradable_snapshot(current_node, message)
                if snapshot is not None:
                    snapshot.message = message
                    # It stops being a leaf, so drop the leaf-only payload exactly as
                    # _append_child would have on the branch we are collapsing into it.
                    if snapshot.generated_by == 'llm' and snapshot.generation_detail:
                        snapshot.generation_detail.pop('input_messages', None)
                        snapshot.generation_detail.pop('output_message', None)
                        snapshot.generation_detail.pop('tools', None)
                    current_node = snapshot
                    continue

                # Create new node for this message
                new_node = ChatNode(
                    generated_by='unknown',
                    generation_detail=None,
                    tags=call_input.get('tags', []),
                    message=message,
                    children=[],
                    parent=current_node,
                )
                self._append_child(current_node, new_node)
                current_node = new_node

        # Add output as a new child node
        generation_detail: dict[str, Any] = {
            # The span dispatcher already gives processors a deep-copied, read-only
            # snapshot. Retain its message list directly while this node is a leaf;
            # _append_child releases it as soon as the conversation is extended.
            'input_messages': input_messages,
            'output_message': trajectory_output_message,
            'finish_reason': self._extract_finish_reason(call_output),
            'tool_choice': call_input.get('kwargs', {}).get('tool_choice'),
        }
        # Tools are leaf-only data in branch_last mode. _append_child drops them as
        # soon as this output is extended by a later call.
        if self.tools_mode == 'branch_last':
            generation_detail['tools'] = tools if tools else None

        output_node = ChatNode(
            generated_by='llm',
            generation_detail=generation_detail,
            tags=call_input.get('tags', []),
            message=output_message,
            children=[],
            parent=current_node,
        )
        if trajectory_output_message is not None:
            output_node.generation_detail['output_message'] = (
                output_node.message if output_message is trajectory_output_message else trajectory_output_message
            )
        self._append_child(current_node, output_node)

    def _extract_output_message(self, output: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Extract the assistant message from an LLM output."""
        message = None
        if isinstance(output, dict):
            choices = output.get('choices', [])
            if choices:
                message = choices[0].get('message')
        elif hasattr(output, 'choices') and output.choices:
            first_choice = output.choices[0]
            if hasattr(first_choice, 'message'):
                message = first_choice.message

        message = serialize_object(message)
        if isinstance(message, dict):
            trajectory_output_message = message if message else None
            if 'role' not in message:
                message = {**message, 'role': 'assistant'}
            return message, trajectory_output_message

        return {'role': 'assistant', 'content': str(output) if output else ''}, None

    def _extract_finish_reason(self, output: Any) -> Any:
        """Extract the first choice's finish reason without retaining the full response."""
        if isinstance(output, dict):
            choices = output.get('choices', [])
            if choices:
                return choices[0].get('finish_reason')
            return None
        if hasattr(output, 'choices') and output.choices:
            return getattr(output.choices[0], 'finish_reason', None)
        return None

    def dump_trajectories(self) -> list[list[dict[str, Any]]] | list[dict[str, Any]]:
        """Dump all chat trajectories from the tree.

        The return format depends on tools_mode:
        - 'global_first': Returns list of message lists (tools are in get_result at top level)
        - 'branch_last': Returns list of dicts with 'messages' and 'tools' keys

        Each trajectory represents the complete message history for one conversation
        path through the tree, ending at a leaf node.

        Returns:
            For global_first: List of trajectories, each being a list of messages.
            For branch_last: List of trajectory dicts, each containing 'messages' and 'tools'.
        """
        trajectories: list[Any] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(reversed(node.children))
                continue

            # Reached a leaf node, extract the full trajectory
            if node.generated_by == 'llm' and node.generation_detail:
                messages = list(node.generation_detail.get('input_messages', []))
                output_message = node.generation_detail.get('output_message')
                if output_message is not None:
                    # Omit null fields from the final assistant message only; inputs stay
                    # verbatim. Always keep ``content`` (as '') so tool-only turns match
                    # the SFT assistant schema (key present, not absent).
                    cleaned = {key: value for key, value in output_message.items() if value is not None}
                    if cleaned.get('role', 'assistant') == 'assistant' and 'content' not in cleaned:
                        cleaned['content'] = ''
                    messages.append(cleaned)

                if self.tools_mode == 'branch_last':
                    # Include tools per trajectory
                    tools = node.generation_detail.get('tools')
                    trajectory: dict[str, Any] = {'messages': messages, 'tools': tools}
                    role_node: ChatNode | None = node
                    while role_node is not None:
                        if role_node.tags:
                            trajectory['role'] = role_node.tags[0]
                            break
                        role_node = role_node.parent
                    # Per-turn finish_reason / tool_choice (response/request metadata,
                    # not message fields) recovered by walking the node path.
                    turns = self._collect_turns(node)
                    if turns:
                        trajectory['extra'] = {'turns': turns}
                    trajectories.append(trajectory)
                else:
                    # global_first: just return messages list
                    trajectories.append(messages)

        return trajectories

    def _collect_turns(self, leaf: ChatNode) -> list[dict[str, Any]]:
        """Walk root->leaf and collect per-LLM-turn metadata, aligned to messages via index."""
        path: list[ChatNode] = []
        node: ChatNode | None = leaf
        while node is not None and node is not self.root:
            path.append(node)
            node = node.parent
        path.reverse()

        turns: list[dict[str, Any]] = []
        for index, n in enumerate(path):
            detail = n.generation_detail
            if n.generated_by != 'llm' or not detail:
                continue
            turn: dict[str, Any] = {'index': index}
            finish_reason = detail.get('finish_reason')
            if finish_reason is not None:
                turn['response'] = {'finish_reason': finish_reason}
            tool_choice = detail.get('tool_choice')
            if tool_choice is not None:
                turn['request'] = {'tool_choice': tool_choice}
            if len(turn) > 1:
                turns.append(turn)
        return turns

    def get_result(self) -> dict[str, Any]:
        """Get the result of this processor.

        Returns:
            For global_first mode:
                {'trajectories': [[msg, ...], ...], 'tools': [...]}
            For branch_last mode:
                {'trajectories': [{'messages': [...], 'tools': [...]}, ...]}
        """
        result: dict[str, Any] = {'trajectories': self.dump_trajectories()}
        if self.tools_mode == 'global_first':
            result['tools'] = self._global_tools
        return result

    def snapshot_result(self) -> dict[str, Any]:
        """Get a read-only snapshot of current chat trajectories."""
        result: dict[str, Any] = {'trajectories': self.dump_trajectories()}
        if self.tools_mode == 'global_first':
            result['tools'] = self._global_tools
        return result
