"""Picklable trajectory independent of scaffold, model, and policy.

Gateway runtime writes topology and token attribution directly; this module
also exposes snapshot queries, serialization, and training segments.
"""

from __future__ import annotations

import base64 as _base64
import binascii as _binascii
import copy as _copy
import math as _math

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from nexus.gateway.trajectory_store import DumpRef, NodePayloadRecord, NodePayloadRef
from nexus.gateway.types import LOGPROB_DTYPE, TOKEN_ID_DTYPE


SCHEMA_VERSION = 'gateway-trajectory-v1'

NodeId = int
MessageId = int
ChainId = str

MAIN_CHAIN_ID: ChainId = 'main'

_TYPE_TAG = '__nexus_gateway_type__'

_TOKEN_ACCOUNT_STORE_PAYLOAD_FIELDS: tuple[str, ...] = (
    'token_ids',
    'logprobs',
    'topk_indices',
    'topk_values',
    'routed_experts',
    'multi_modal_items',
)


# For Json serialization safety.
def _encode_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not _math.isfinite(value):
            raise TypeError(f'{path}: non-finite floats are not supported')
        return value
    if isinstance(value, bytes):
        return {_TYPE_TAG: 'bytes', 'data': _base64.b64encode(value).decode('ascii')}
    if isinstance(value, tuple):
        return {
            _TYPE_TAG: 'tuple',
            'items': [_encode_value(item, f'{path}[{index}]') for index, item in enumerate(value)],
        }
    if isinstance(value, list):
        return [_encode_value(item, f'{path}[{index}]') for index, item in enumerate(value)]
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f'{path}: dictionary key {key!r} is not a string')
        items = {key: _encode_value(value[key], f'{path}.{key}') for key in sorted(value)}
        if _TYPE_TAG in items:
            return {_TYPE_TAG: 'dict', 'items': [[key, item] for key, item in items.items()]}
        return items
    value_type = type(value)
    if value_type.__module__.startswith('numpy') and value_type.__name__ == 'ndarray':
        if getattr(value.dtype, 'hasobject', False):
            raise TypeError(f'{path}: object-dtype ndarray is not supported')
        return {
            _TYPE_TAG: 'ndarray',
            'data': _base64.b64encode(value.tobytes(order='C')).decode('ascii'),
            'dtype': value.dtype.str,
            'shape': list(value.shape),
        }
    raise TypeError(f'{path}: unsupported value type {value_type.__module__}.{value_type.__qualname__}')


# For Json deserialization safety.
def _decode_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not _math.isfinite(value):
            raise ValueError(f'{path}: non-finite floats are not supported')
        return value
    if isinstance(value, list):
        return [_decode_value(item, f'{path}[{index}]') for index, item in enumerate(value)]
    if not isinstance(value, dict):
        raise TypeError(f'{path}: invalid decoded value type {type(value).__name__}')
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f'{path}: decoded dictionary keys must be strings')
    tag = value.get(_TYPE_TAG)
    if tag is None:
        return {key: _decode_value(item, f'{path}.{key}') for key, item in value.items()}
    if tag == 'bytes' and set(value) == {_TYPE_TAG, 'data'}:
        try:
            return _base64.b64decode(value['data'], validate=True)
        except (TypeError, ValueError, _binascii.Error) as exc:
            raise ValueError(f'{path}: invalid bytes payload') from exc
    if tag == 'tuple' and set(value) == {_TYPE_TAG, 'items'}:
        if not isinstance(value['items'], list):
            raise ValueError(f'{path}: invalid tuple payload')
        return tuple(_decode_value(item, f'{path}[{index}]') for index, item in enumerate(value['items']))
    if tag == 'dict' and set(value) == {_TYPE_TAG, 'items'}:
        if not isinstance(value['items'], list):
            raise ValueError(f'{path}: invalid dictionary payload')
        decoded: dict[str, Any] = {}
        for index, pair in enumerate(value['items']):
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                raise ValueError(f'{path}.items[{index}]: invalid dictionary entry')
            key, item = pair
            if key in decoded:
                raise ValueError(f'{path}: duplicate dictionary key {key!r}')
            decoded[key] = _decode_value(item, f'{path}.{key}')
        return decoded
    if tag == 'ndarray' and set(value) == {_TYPE_TAG, 'data', 'dtype', 'shape'}:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on the consumer environment
            raise RuntimeError(f'{path}: decoding ndarray requires numpy') from exc
        try:
            dtype = np.dtype(value['dtype'])
            if dtype.hasobject:
                raise ValueError('object dtype')
            shape = value['shape']
            if not isinstance(shape, list) or any(not isinstance(size, int) or size < 0 for size in shape):
                raise ValueError('invalid shape')
            raw = _base64.b64decode(value['data'], validate=True)
            return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
        except (TypeError, ValueError, _binascii.Error) as exc:
            raise ValueError(f'{path}: invalid ndarray payload') from exc
    raise ValueError(f'{path}: invalid or unknown tagged value {tag!r}')


# ---------------------------------------------------------------------------
# Recorded fact types
# ---------------------------------------------------------------------------


@dataclass
class ProviderFacts:
    """Per-call provider response facts (model routing / cost audit)."""

    model: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    inference_time: float | None = None
    wall_inference_time: float | None = None
    abort_wait_time: float | None = None
    num_preempted: int | None = None
    model_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        fields = {
            'model': self.model,
            'model_id': self.model_id,
            'response_id': self.response_id,
            'usage': self.usage,
            'finish_reason': self.finish_reason,
            'inference_time': self.inference_time,
            'wall_inference_time': self.wall_inference_time,
            'abort_wait_time': self.abort_wait_time,
            'num_preempted': self.num_preempted,
        }
        return {k: v for k, v in fields.items() if v is not None}


@dataclass
class InvocationFacts:
    """Declared-only request ordinals (white-box scaffolds declare via
    request_meta; black-box lanes stay None — the engine never infers)."""

    iteration: int | None = None
    retry: int | None = None

    def to_dict(self) -> dict[str, Any]:
        fields = {'iteration': self.iteration, 'retry': self.retry}
        return {k: v for k, v in fields.items() if v is not None}


@dataclass
class SegmentBoundary:
    """Boundary conclusion on the llm node that opens a segment.

    Mapped from the assembler's boundary metadata: kind <- segment_type,
    causes <- boundary_causes, evidence <- the rest.
    """

    starts_new_segment: bool = False
    kind: str | None = None
    causes: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {'starts_new_segment': self.starts_new_segment}
        if self.kind is not None:
            data['kind'] = self.kind
        if self.causes:
            data['causes'] = list(self.causes)
        if self.evidence:
            data['evidence'] = self.evidence
        return data


@dataclass
class TokenAccount:
    """Tokens owned by one templated message.

    ``gen_start`` and ``gen_len`` locate the sampled span. Tokens outside that
    span come from the tokenizer template or observations and remain mask-0.
    """

    payload_ref: NodePayloadRef | DumpRef | None = None
    token_ids: np.ndarray | None = None
    gen_start: int = 0
    gen_len: int = 0
    # NOTE(jerryhrzhou): A factory leaves old pickles without this metadata detectably incompatible.
    has_routed_experts: bool = field(default_factory=bool)

    raw_token_ids: np.ndarray | None = None
    num_images: int = 0
    multi_modal_items: dict[str, list[dict[str, Any]]] | None = None
    logprobs: np.ndarray | None = None
    topk_indices: np.ndarray | None = None
    topk_values: np.ndarray | None = None
    routed_experts: Any | None = None

    @property
    def rollout_token_ids(self) -> np.ndarray | None:
        """This message's tokens as the rollout engine saw them (raw where it has images)."""
        return self.raw_token_ids if self.raw_token_ids is not None else self.token_ids

    def __post_init__(self) -> None:
        if self.token_ids is None:
            return
        count = len(self.token_ids)
        if not 0 <= self.gen_start <= count:
            raise ValueError(f'gen_start {self.gen_start} outside token_ids length {count}')
        if not 0 <= self.gen_len <= count - self.gen_start:
            raise ValueError(f'gen_start {self.gen_start} + gen_len {self.gen_len} outside token_ids length {count}')

    @property
    def has_inline_store_payload(self) -> bool:
        """Check for Store fields that have not been discarded, including empty containers."""
        return any(getattr(self, name) is not None for name in _TOKEN_ACCOUNT_STORE_PAYLOAD_FIELDS)

    def hydrate_store_payload(self, payload: Mapping[str, Any]) -> None:
        """Restore the Store fields present in a Node payload."""
        for name in _TOKEN_ACCOUNT_STORE_PAYLOAD_FIELDS:
            if name in payload:
                setattr(self, name, payload[name])

    def discard_store_payload(self) -> None:
        """Drop fields recoverable from ``payload_ref`` while retaining compact token metadata."""
        for name in _TOKEN_ACCOUNT_STORE_PAYLOAD_FIELDS:
            setattr(self, name, None)

    def build_store_payload_record(self, trajectory_id: str, node_id: NodeId) -> NodePayloadRecord:
        """Build the immutable Store record for this token account."""
        return NodePayloadRecord(
            trajectory_id=trajectory_id,
            node_id=node_id,
            token_ids=self.token_ids if self.token_ids is not None else np.empty(0, dtype=TOKEN_ID_DTYPE),
            logprobs=self.logprobs,
            topk_indices=self.topk_indices,
            topk_values=self.topk_values,
            routed_experts=self.routed_experts,
            multi_modal_items=self.multi_modal_items,
        )

    @property
    def generated_ids(self) -> np.ndarray:
        """The sampled span (mask=1 domain; numeric arrays align here)."""
        if self.token_ids is None:
            return np.empty(0, dtype=TOKEN_ID_DTYPE)
        return self.token_ids[self.gen_start : self.gen_start + self.gen_len]

    @property
    def glue_ids(self) -> np.ndarray:
        """Prompt-side tokens before the sampled span."""
        if self.token_ids is None:
            return np.empty(0, dtype=TOKEN_ID_DTYPE)
        return self.token_ids[: self.gen_start]

    @property
    def trailing_ids(self) -> np.ndarray:
        """Mask-0 tokens after the sampled span."""
        if self.token_ids is None:
            return np.empty(0, dtype=TOKEN_ID_DTYPE)
        return self.token_ids[self.gen_start + self.gen_len :]

    def to_dict(self) -> dict[str, Any]:
        """Serialize account metadata and any hydrated Store payload fields."""
        data: dict[str, Any] = {'has_routed_experts': self.has_routed_experts}
        for name in _TOKEN_ACCOUNT_STORE_PAYLOAD_FIELDS:
            if (value := getattr(self, name)) is not None:
                data[name] = value
        if self.gen_start:
            data['gen_start'] = self.gen_start
        if self.gen_len:
            data['gen_len'] = self.gen_len
        if self.num_images:
            data['num_images'] = self.num_images
        return data


@dataclass
class SpawnLink:
    """Tool birthplace of a chain (None for client-side forks e.g. session_memory)."""

    node: NodeId | None = None
    tool_use_id: str | None = None
    candidates: list[str] = field(default_factory=list)


@dataclass
class JoinLink:
    """Re-integration of a chain's result into its parent (None = fire-and-forget)."""

    node: NodeId | None = None
    tool_use_id: str | None = None
    kind: str | None = None  # inline | poll | notify


@dataclass
class Chain:
    """One execution stream. 'main' is an ordinary entry (spawn/join/parent None).

    Identity rules: main='main'; declared chains pass their declared id
    through verbatim; inferred chains use the wire agent_id; identity-less side
    streams synthesize ``chain@<root NodeId>``. ``roots`` marks ownership
    entries whose chain cannot be inherited from a physical ancestor. Typed
    links contain resolved facts; evidence preserves ambiguous candidates.
    """

    chain_id: ChainId
    kind: str = 'main'
    parent_chain_id: ChainId | None = None
    roots: list[NodeId] = field(default_factory=list)
    spawn: SpawnLink | None = None
    join: JoinLink | None = None
    detector: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryNode:
    """One scaffold-agnostic wire message in the public trajectory contract.

    Three zones: wire facts / semantic annotations / reward.

    Chain membership and render continuity are derived — no chain_id, no
    render key. ``parent`` and ``children`` expose the public physical topology in both
    directions. Render variants share ``message_id`` but keep distinct
    ``node_id`` values; inference attempts never deduplicate on message
    identity.
    """

    node_id: NodeId
    message_id: MessageId
    parent: NodeId | None
    generated_by: Literal['llm', 'observation', 'client']
    children: list[NodeId] = field(kw_only=True)

    # -- zone 1: wire facts (recorded, append-only) --
    original_message: dict[str, Any] = field(default_factory=dict)
    templated_message: str | None = None  # Rendered by the tokenizer template.
    request_kwargs: dict[str, Any] | None = None  # llm only
    token_account: TokenAccount | None = None  # None on message lane, or when another node owns the rendered tokens.
    provider_facts: ProviderFacts | None = None  # llm only
    request_message_overrides: dict[NodeId, dict[str, Any]] = field(default_factory=dict)  # llm request variants
    timeline: list[dict[str, Any]] = field(default_factory=list)  # node-anchored events
    as_sent_variants: list[dict[str, Any]] = field(default_factory=list)  # audit-only wire occurrences

    # -- zone 2: trajectory semantics (plain data; reading needs no policy) --
    segment_boundary: SegmentBoundary | None = None
    selection: Literal['selected', 'rejected_sibling'] | None = None
    invocation: InvocationFacts | None = None

    # -- zone 3: training annotation --
    reward: float | None = None


def count_rendered_images(nodes: Iterable[TrajectoryNode]) -> int:
    return sum(account.num_images for node in nodes if (account := node.token_account) is not None)


# ---------------------------------------------------------------------------
# Segments (derived training view)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SegmentNodeList:
    """Canonical Node composition for one segment occurrence."""

    segment_id: str
    chain_id: ChainId
    nodes: tuple[TrajectoryNode, ...]
    first_response_node_index: int

    @property
    def response_nodes(self) -> tuple[TrajectoryNode, ...]:
        return tuple(node for node in self.nodes[self.first_response_node_index :] if node.generated_by == 'llm')

    @property
    def last_response_node(self) -> TrajectoryNode:
        return self.response_nodes[-1]

    @property
    def num_images(self) -> int:
        """Images this segment's model calls consumed; the next prompt owns anything after them."""
        return count_rendered_images(self.nodes[: self.nodes.index(self.last_response_node) + 1])


@dataclass(frozen=True, eq=False)
class SegmentTrainingData:
    """Immutable training header for one canonical segment occurrence."""

    node_list: SegmentNodeList
    prompt_length: int
    response_length: int
    generated_ranges: tuple[tuple[int, int], ...]
    training_enabled: bool


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class Trajectory:
    """Unified topology and compact training handles in one picklable object.

    Forest semantics; scaffold/policy agnostic. Large token payloads live in the
    injected record store and are addressed through Node refs. There is no
    synthetic root; an empty trajectory has roots == []. Gateway runtime mutates
    the live forest; freeze either copies it for an active snapshot or finalizes
    it in place when the session ends.
    """

    def __init__(
        self,
        *,
        session_id: str,
        state: str = 'completed',
        nodes: dict[NodeId, TrajectoryNode] | None = None,
        roots: list[NodeId] | None = None,
        chains: dict[ChainId, Chain] | None = None,
        timeline: list[dict[str, Any]] | None = None,
        final_reward: float | None = None,
        termination_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self.schema_version = SCHEMA_VERSION
        self.session_id = session_id
        self.state = state
        self.nodes: dict[NodeId, TrajectoryNode] = {} if nodes is None else nodes
        self.roots: list[NodeId] = [] if roots is None else roots
        self.chains: dict[ChainId, Chain] = {} if chains is None else chains
        self.timeline: list[dict[str, Any]] = [] if timeline is None else timeline  # non-node-anchored only
        self.final_reward = final_reward
        self.termination_reason = termination_reason
        self.metrics: dict[str, Any] = {} if metrics is None else metrics
        self.extra: dict[str, Any] = {} if extra is None else extra
        self.segment_training_data: dict[str, SegmentTrainingData] = {}
        self._invalidate_caches()

    @property
    def num_images(self) -> int:
        """Images this rollout's model calls consumed, over the whole node forest.

        Segment counts share prefix nodes, so this cannot be their sum.
        """
        return count_rendered_images(self.nodes.values())

    # -- pickle: semantic state only, caches rebuilt on load --

    _STATE_FIELDS = (
        'schema_version',
        'session_id',
        'state',
        'nodes',
        'roots',
        'chains',
        'timeline',
        'final_reward',
        'termination_reason',
        'metrics',
        'extra',
        'segment_training_data',
    )

    def __getstate__(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._STATE_FIELDS}

    def __setstate__(self, state: dict[str, Any]) -> None:
        version = state.get('schema_version')
        if version != SCHEMA_VERSION:
            raise ValueError(f'unsupported trajectory schema_version: {version!r} (expected {SCHEMA_VERSION})')
        for key in self._STATE_FIELDS:
            setattr(self, key, state[key])
        self._invalidate_caches()

    def _invalidate_caches(self) -> None:
        self._chain_index: dict[NodeId, ChainId] | None = None

    # ------------------------------------------------------------------
    # topology queries
    # ------------------------------------------------------------------

    def iter_nodes(self) -> Iterator[TrajectoryNode]:
        for node_id in sorted(self.nodes):  # node ids are assigned in assembly order
            yield self.nodes[node_id]

    def node_chain(self, node_id: NodeId) -> ChainId:
        """Derived membership: the chain whose root is the nearest ancestor
        chain-root; shared-prefix nodes belong to the parent chain."""
        index = self._chain_membership()
        return index.get(node_id, MAIN_CHAIN_ID)

    def _chain_membership(self) -> dict[NodeId, ChainId]:
        if self._chain_index is None:
            root_owner: dict[NodeId, ChainId] = {}
            for chain in self.chains.values():
                for root in chain.roots:
                    root_owner[root] = chain.chain_id
            # memoized upward walk: robust to any id order (restart-synthesized
            # branch parents carry LARGER ids than their children)
            index: dict[NodeId, ChainId] = {}
            for node_id in self.nodes:
                trail: list[NodeId] = []
                current: NodeId | None = node_id
                resolved = MAIN_CHAIN_ID
                while current is not None:
                    if current in index:
                        resolved = index[current]
                        break
                    if current in root_owner:
                        resolved = root_owner[current]
                        trail.append(current)
                        break
                    trail.append(current)
                    current = self.nodes[current].parent
                for nid in trail:
                    index[nid] = resolved
            self._chain_index = index
        return self._chain_index

    # ------------------------------------------------------------------
    # token reconstruction
    # ------------------------------------------------------------------

    def _path_to(self, node_id: NodeId) -> list[NodeId]:
        path: list[NodeId] = []
        current: NodeId | None = node_id
        while current is not None:
            path.append(current)
            current = self.nodes[current].parent
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # derived views
    # ------------------------------------------------------------------

    def _selected_paths(
        self,
        chain_ids: set[ChainId],
        roots: list[NodeId] | None = None,
    ) -> Iterator[tuple[ChainId, list[NodeId]]]:
        """Derive selected chain-local paths from frozen structure."""

        def chain_depth(chain_id: ChainId) -> int:
            depth = 0
            seen = {chain_id}
            chain = self.chains.get(chain_id)
            parent = chain.parent_chain_id if chain is not None else None
            while parent in self.chains and parent not in seen:
                seen.add(parent)
                depth += 1
                parent = self.chains[parent].parent_chain_id
            return depth

        entries = list(self.roots if roots is None else roots)
        if roots is None:
            entries.sort(
                key=lambda node_id: (
                    self.node_chain(node_id) == MAIN_CHAIN_ID,
                    -chain_depth(self.node_chain(node_id)),
                )
            )

        for root in entries:
            if self.nodes[root].selection == 'rejected_sibling':
                continue
            chain_id = self.node_chain(root)
            stack: list[tuple[str, NodeId, ChainId, NodeId, bool]] = [
                ('visit', root, chain_id, root, self.nodes[root].generated_by == 'llm')
            ]
            while stack:
                action, node_id, chain_id, branch_root, owns_llm = stack.pop()
                if action == 'emit':
                    if chain_id in chain_ids:
                        path = self._path_to(node_id)
                        yield chain_id, path[path.index(branch_root) :]
                    continue

                selected = [
                    child for child in self.nodes[node_id].children if self.nodes[child].selection != 'rejected_sibling'
                ]
                same = [child for child in selected if self.node_chain(child) == chain_id]
                foreign = [child for child in selected if self.node_chain(child) != chain_id]
                if not same and owns_llm:
                    stack.append(('emit', node_id, chain_id, branch_root, owns_llm))
                for child in reversed(same):
                    stack.append((
                        'visit',
                        child,
                        chain_id,
                        branch_root,
                        owns_llm or self.nodes[child].generated_by == 'llm',
                    ))
                for child in reversed(foreign):
                    child_chain = self.node_chain(child)
                    stack.append(('visit', child, child_chain, child, self.nodes[child].generated_by == 'llm'))

    def _belongs_to_previous_turn(self, node: TrajectoryNode) -> bool:
        return node.generated_by == 'observation' or node.original_message.get('role') in {'tool', 'function'}

    def to_messages(self, chain_id: ChainId = MAIN_CHAIN_ID) -> list[dict[str, Any]]:
        """Wire transcript of the chain's latest selected path."""
        paths = [path for _, path in self._selected_paths({chain_id})]
        if not paths:
            return []
        last_path = max(paths, key=lambda path: (self.nodes[path[-1]].message_id, path[-1]))
        request_node_id = next(
            (node_id for node_id in reversed(last_path) if self.nodes[node_id].generated_by == 'llm'),
            None,
        )
        if request_node_id is None:
            return [_copy.deepcopy(self.nodes[node_id].original_message) for node_id in last_path]
        return self._messages_for_nodes(last_path, request_node_id)

    # ------------------------------------------------------------------
    # Canonical training exit
    # ------------------------------------------------------------------

    def to_segment_node_lists(
        self,
        chains: str | list[ChainId] = 'all',
    ) -> tuple[SegmentNodeList, ...]:
        """Project canonical segment compositions without materializing token rows."""
        return tuple(self._project_segment_node_lists(self._selected_segment_chains(chains)))

    def _selected_segment_chains(self, chains: str | list[ChainId]) -> list[ChainId]:
        if isinstance(chains, str):
            if chains == 'all':
                return list(self.chains)
            if chains == 'main':
                return [MAIN_CHAIN_ID]
            raise ValueError(f"chains must be 'all', 'main', or a collection of chain ids; got {chains!r}")
        selected = list(chains)
        missing = [chain_id for chain_id in selected if chain_id not in self.chains]
        if missing:
            raise ValueError(f'unknown chain ids: {missing}')
        return selected

    def _project_segment_node_lists(
        self,
        selected: list[ChainId],
    ) -> list[SegmentNodeList]:
        projections: list[SegmentNodeList] = []
        exported: set[NodeId] = set()
        for chain_id, path in self._selected_paths(set(selected)):
            for group in self._semantic_groups(path):
                fresh = [node_id for node_id in group if node_id not in exported]
                exported.update(fresh)
                if not fresh:
                    continue
                for physical_group in self._frozen_physical_groups(fresh):
                    node_ids = self._group_message_nodes(path, physical_group)
                    first_response_node_index = node_ids.index(physical_group[0])
                    projections.append(
                        SegmentNodeList(
                            segment_id=f'{chain_id}@{physical_group[0]}',
                            chain_id=chain_id,
                            nodes=tuple(self.nodes[node_id] for node_id in node_ids),
                            first_response_node_index=first_response_node_index,
                        )
                    )
        chain_order: dict[ChainId, int] = {}
        for node_list in projections:
            chain_order.setdefault(node_list.chain_id, len(chain_order))
        projections.sort(
            key=lambda node_list: (
                chain_order[node_list.chain_id],
                node_list.response_nodes[0].message_id,
            )
        )
        return projections

    # ------------------------------------------------------------------
    # Segment-occurrence training tensors
    # ------------------------------------------------------------------

    # GATEWAY_KV_TRANSITION: REPLACE BY MR4 WITH MR3 ATOMIC STORE PUBLICATION.
    def prepare_segment_training_data(
        self,
        initializations: Iterable[tuple[SegmentNodeList, bool]],
    ) -> dict[str, SegmentTrainingData]:
        """Build one complete training-data publication without mutating the trajectory."""
        if self.state != 'completed':
            raise ValueError('segment training data requires a completed trajectory')
        if self.segment_training_data:
            raise ValueError('segment training data is already published')
        replacement: dict[str, SegmentTrainingData] = {}
        for node_list, training_enabled in initializations:
            if any(self.nodes.get(node.node_id) is not node for node in node_list.nodes):
                raise ValueError(f'segment {node_list.segment_id!r} contains nodes from another trajectory')
            if node_list.segment_id in replacement:
                raise KeyError(f'duplicate segment training data {node_list.segment_id!r}')
            first_account = node_list.response_nodes[0].token_account
            prompt_length = 0
            response_length = 0
            generated_ranges: list[tuple[int, int]] = []
            if first_account is not None:
                prompt_length = (
                    sum(
                        len(account.token_ids)
                        for node in node_list.nodes[: node_list.first_response_node_index]
                        if (account := node.token_account) is not None and account.token_ids is not None
                    )
                    + first_account.gen_start
                )
                last_response_node_index = node_list.nodes.index(node_list.last_response_node)
                for index, node in enumerate(
                    node_list.nodes[node_list.first_response_node_index : last_response_node_index + 1]
                ):
                    account = node.token_account
                    if account is None or account.token_ids is None:
                        continue
                    generated_start = response_length + (0 if index == 0 else account.gen_start)
                    generated_stop = generated_start + account.gen_len
                    if account.gen_len:
                        generated_ranges.append((generated_start, generated_stop))
                    response_length += len(account.token_ids) - (first_account.gen_start if index == 0 else 0)
            replacement[node_list.segment_id] = SegmentTrainingData(
                node_list=node_list,
                prompt_length=prompt_length,
                response_length=response_length,
                generated_ranges=tuple(generated_ranges),
                training_enabled=training_enabled,
            )
        return replacement

    @property
    def has_inline_store_payload(self) -> bool:
        """Check all nodes for inline Store payloads without modifying the trajectory."""
        return any(
            node.token_account is not None and node.token_account.has_inline_store_payload
            for node in self.nodes.values()
        )

    def discard_inline_payload(self) -> None:
        """Discard producer-local payload after scoring and any training preparation are complete."""
        if self.state != 'completed':
            raise RuntimeError('inline payload can only be released from a completed trajectory')
        for node in self.nodes.values():
            if node.token_account is not None:
                account = node.token_account
                account.discard_store_payload()
                account.raw_token_ids = None
            node.original_message.clear()
            node.templated_message = None
            node.request_kwargs = None
            node.request_message_overrides.clear()
            node.timeline.clear()
            node.as_sent_variants.clear()
        self.timeline.clear()

    def _semantic_groups(self, path: list[NodeId]) -> list[list[NodeId]]:
        """Split the path's llm nodes at segment_boundary.starts_new_segment."""
        groups: list[list[NodeId]] = []
        current: list[NodeId] = []
        for node_id in path:
            node = self.nodes[node_id]
            if node.generated_by != 'llm':
                continue
            boundary = node.segment_boundary
            starts = boundary is not None and boundary.starts_new_segment
            if current and starts:
                groups.append(current)
                current = []
            current.append(node_id)
        if current:
            groups.append(current)
        return groups

    def _frozen_physical_groups(self, nodes: list[NodeId]) -> list[list[NodeId]]:
        groups: list[list[NodeId]] = []
        current: list[NodeId] = []
        for node_id in nodes:
            boundary = self.nodes[node_id].segment_boundary
            if current and boundary is not None and self._physical_only(boundary):
                groups.append(current)
                current = []
            current.append(node_id)
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _physical_only(boundary: SegmentBoundary) -> bool:
        return boundary.evidence.get('physical_only') is True

    def _group_message_nodes(self, path: list[NodeId], group: list[NodeId]) -> list[NodeId]:
        """Message span of a segment: full structural context (walk parents to
        the forest root — sidechain leaf paths start at the chain root, but the
        model's real input includes the shared parent-chain prefix; legacy
        recovers it from the first llm's recorded input) through the group's
        trailing observations."""
        end = path.index(group[-1]) + 1
        while end < len(path) and self._belongs_to_previous_turn(self.nodes[path[end]]):
            end += 1
        # ancestors above the path head (for chain-rooted paths)
        above: list[NodeId] = []
        current = self.nodes[path[0]].parent
        while current is not None:
            above.append(current)
            current = self.nodes[current].parent
        above.reverse()
        return above + path[:end]

    def _messages_for_nodes(self, node_ids: list[NodeId], request_node_id: NodeId) -> list[dict[str, Any]]:
        overrides = self.nodes[request_node_id].request_message_overrides
        return [_copy.deepcopy(overrides.get(node_id, self.nodes[node_id].original_message)) for node_id in node_ids]

    # ------------------------------------------------------------------
    # dump / from_dump (versioned plain-data form; preserves provider-native facts)
    # ------------------------------------------------------------------

    def to_debug_dict(self) -> dict[str, Any]:
        """Export node facts and complete segment token sequences for caller-side decoding.

        Token accounts must already have hydrated token_ids. Each segment concatenates
        its canonical node composition, including shared context and trailing tool
        observations. Training records and non-token payloads are excluded. Metadata
        retains its original values; callers handle JSON conversion after decoding.
        """
        nodes: list[dict[str, Any]] = []
        tokens_by_node: dict[NodeId, list[int]] = {}
        for node_id in sorted(self.nodes):
            node = self.nodes[node_id]
            row = {
                'node_id': node.node_id,
                'message_id': node.message_id,
                'parent': node.parent,
                'children': list(node.children),
                'generated_by': node.generated_by,
                'selection': node.selection,
                'reward': node.reward,
            }
            for name in ('segment_boundary', 'invocation', 'provider_facts'):
                facts = getattr(node, name)
                if facts is not None:
                    row[name] = facts.to_dict()
            account = node.token_account
            if account is not None:
                assert account.token_ids is not None, f'Hydrate token_ids before exporting node {node_id} for debugging'
                token_ids = account.token_ids.tolist()
                tokens_by_node[node_id] = token_ids
                row.update(
                    token_ids=token_ids,
                    token_count=len(token_ids),
                    gen_start=account.gen_start,
                    gen_len=account.gen_len,
                    num_images=account.num_images,
                )
            nodes.append(row)

        segments = []
        for node_list in self.to_segment_node_lists():
            token_ids = [
                token_id
                for node in node_list.nodes
                if node.token_account is not None
                for token_id in tokens_by_node[node.node_id]
            ]
            segments.append({
                'segment_id': node_list.segment_id,
                'chain_id': node_list.chain_id,
                'node_ids': [node.node_id for node in node_list.nodes],
                'first_response_node_index': node_list.first_response_node_index,
                'token_ids': token_ids,
                'token_count': len(token_ids),
            })
        return {
            'session_id': self.session_id,
            'state': self.state,
            'final_reward': self.final_reward,
            'termination_reason': self.termination_reason,
            'roots': list(self.roots),
            'chains': [self._dump_chain(chain) for chain in self.chains.values()],
            'nodes': nodes,
            'segments': segments,
            'metrics': self.metrics,
            'extra': self.extra,
        }

    def dump(self) -> dict[str, Any]:
        """Serialize the trajectory, including any hydrated Store payloads."""
        if self.segment_training_data:
            raise TypeError(
                'trajectory contains segment training data; '
                'use pickle or Ray serialization instead of Trajectory.dump()'
            )
        data = {
            'schema_version': self.schema_version,
            'session_id': self.session_id,
            'state': self.state,
            'roots': list(self.roots),
            'nodes': [self._dump_node(self.nodes[node_id]) for node_id in sorted(self.nodes)],
            'chains': [self._dump_chain(chain) for chain in self.chains.values()],
            'timeline': self.timeline,
            'final_reward': self.final_reward,
            'termination_reason': self.termination_reason,
            'metrics': self.metrics,
            'extra': self.extra,
        }
        return _encode_value(data, '$')

    @staticmethod
    def _dump_node(node: TrajectoryNode) -> dict[str, Any]:
        data: dict[str, Any] = {
            'node_id': node.node_id,
            'message_id': node.message_id,
            'parent': node.parent,
            'children': list(node.children),
            'generated_by': node.generated_by,
            'original_message': node.original_message,
        }
        if node.templated_message is not None:
            data['templated_message'] = node.templated_message
        if node.request_kwargs is not None:
            data['request_kwargs'] = node.request_kwargs
        if node.token_account is not None:
            data['token_account'] = node.token_account.to_dict()
        if node.provider_facts is not None:
            data['provider_facts'] = node.provider_facts.to_dict()
        if node.request_message_overrides:
            data['request_message_overrides'] = [
                {'node_id': node_id, 'message': message} for node_id, message in node.request_message_overrides.items()
            ]
        if node.timeline:
            data['timeline'] = node.timeline
        if node.as_sent_variants:
            data['as_sent_variants'] = node.as_sent_variants
        if node.segment_boundary is not None:
            data['segment_boundary'] = node.segment_boundary.to_dict()
        if node.selection is not None:
            data['selection'] = node.selection
        if node.invocation is not None:
            data['invocation'] = node.invocation.to_dict()
        if node.reward is not None:
            data['reward'] = node.reward
        return data

    @staticmethod
    def _dump_chain(chain: Chain) -> dict[str, Any]:
        data: dict[str, Any] = {
            'chain_id': chain.chain_id,
            'kind': chain.kind,
            'parent_chain_id': chain.parent_chain_id,
            'roots': list(chain.roots),
        }
        if chain.spawn is not None:
            data['spawn'] = {
                'node': chain.spawn.node,
                'tool_use_id': chain.spawn.tool_use_id,
                'candidates': list(chain.spawn.candidates),
            }
        if chain.join is not None:
            data['join'] = {'node': chain.join.node, 'tool_use_id': chain.join.tool_use_id, 'kind': chain.join.kind}
        if chain.detector is not None:
            data['detector'] = chain.detector
        if chain.evidence:
            data['evidence'] = chain.evidence
        return data

    @classmethod
    def from_dump(cls, data: dict[str, Any]) -> Trajectory:
        data = _decode_value(data, '$')
        if not isinstance(data, dict):
            raise TypeError('trajectory dump must decode to a dictionary')
        version = data.get('schema_version')
        if version != SCHEMA_VERSION:
            raise ValueError(f'unsupported trajectory schema_version: {version!r} (expected {SCHEMA_VERSION})')
        kwargs_pool = data.get('kwargs_pool')
        if kwargs_pool is not None and not isinstance(kwargs_pool, list):
            raise TypeError('trajectory kwargs_pool must be a list')
        nodes: dict[NodeId, TrajectoryNode] = {}
        for item in data['nodes']:
            account_data = item.get('token_account')
            account = None
            if account_data is not None:
                token_ids = account_data.get('token_ids')
                logprobs = account_data.get('logprobs')
                topk_indices = account_data.get('topk_indices')
                topk_values = account_data.get('topk_values')
                if topk_indices is not None:
                    topk_indices = np.asarray(topk_indices, dtype=TOKEN_ID_DTYPE)
                    if topk_indices.shape == (0,):
                        topk_indices = topk_indices.reshape(0, 0)
                if topk_values is not None:
                    topk_values = np.asarray(topk_values, dtype=LOGPROB_DTYPE)
                    if topk_values.shape == (0,):
                        topk_values = topk_values.reshape(0, 0)
                account = TokenAccount(
                    token_ids=np.asarray(token_ids, dtype=TOKEN_ID_DTYPE) if token_ids is not None else None,
                    gen_start=account_data.get('gen_start', 0),
                    gen_len=account_data.get('gen_len', 0),
                    has_routed_experts=account_data['has_routed_experts'],
                    num_images=account_data.get('num_images', 0),
                    logprobs=np.asarray(logprobs, dtype=LOGPROB_DTYPE) if logprobs is not None else None,
                    topk_indices=topk_indices,
                    topk_values=topk_values,
                    routed_experts=account_data.get('routed_experts'),
                    multi_modal_items=account_data.get('multi_modal_items'),
                )
            boundary_data = item.get('segment_boundary')
            boundary = None
            if boundary_data is not None:
                boundary = SegmentBoundary(
                    starts_new_segment=boundary_data.get('starts_new_segment', False),
                    kind=boundary_data.get('kind'),
                    causes=tuple(boundary_data.get('causes', ())),
                    evidence=boundary_data.get('evidence', {}),
                )
            provider_data = item.get('provider_facts')
            invocation_data = item.get('invocation')
            kwargs_ref = item.get('request_kwargs_ref')
            if kwargs_ref is not None:
                if kwargs_pool is None or not isinstance(kwargs_ref, int) or not 0 <= kwargs_ref < len(kwargs_pool):
                    raise ValueError(f'invalid request_kwargs_ref: {kwargs_ref!r}')
                kwargs = _copy.deepcopy(kwargs_pool[kwargs_ref])
            else:
                kwargs = item.get('request_kwargs')
            if kwargs is not None and not isinstance(kwargs, dict):
                raise TypeError('trajectory node request_kwargs must be a dictionary')
            node = TrajectoryNode(
                node_id=item['node_id'],
                message_id=item['message_id'],
                parent=item.get('parent'),
                generated_by=item['generated_by'],
                children=list(item['children']),
                original_message=item['original_message'],
                templated_message=item.get('templated_message'),
                request_kwargs=kwargs,
                token_account=account,
                provider_facts=ProviderFacts(**provider_data) if provider_data else None,
                request_message_overrides={
                    override['node_id']: override['message'] for override in item.get('request_message_overrides', [])
                },
                timeline=item.get('timeline', []),
                as_sent_variants=item.get('as_sent_variants', []),
                segment_boundary=boundary,
                selection=item.get('selection'),
                invocation=InvocationFacts(**invocation_data) if invocation_data else None,
                reward=item.get('reward'),
            )
            if node.node_id in nodes:
                raise ValueError(f'duplicate node id {node.node_id}')
            nodes[node.node_id] = node
        chains: dict[ChainId, Chain] = {}
        for item in data['chains']:
            chain_id = item['chain_id']
            if chain_id in chains:
                raise ValueError(f'duplicate chain id {chain_id!r}')
            spawn_data, join_data = item.get('spawn'), item.get('join')
            chains[chain_id] = Chain(
                chain_id=chain_id,
                kind=item.get('kind', 'main'),
                parent_chain_id=item.get('parent_chain_id'),
                roots=list(item.get('roots', [])),
                spawn=SpawnLink(**spawn_data) if spawn_data else None,
                join=JoinLink(**join_data) if join_data else None,
                detector=item.get('detector'),
                evidence=item.get('evidence', {}),
            )
        for node in nodes.values():
            if node.parent is not None and node.parent not in nodes:
                raise ValueError(f'invalid trajectory node {node.node_id}: parent {node.parent} does not exist')
        acyclic: set[NodeId] = set()
        for node_id in nodes:
            trail: list[NodeId] = []
            current: NodeId | None = node_id
            while current is not None and current not in acyclic:
                if current in trail:
                    raise ValueError(f'parent cycle through node {current}')
                trail.append(current)
                current = nodes[current].parent
            acyclic.update(trail)
        declared_children: set[NodeId] = set()
        for node in nodes.values():
            for child_id in node.children:
                if child_id not in nodes:
                    raise ValueError(f'invalid trajectory node {node.node_id}: child {child_id} does not exist')
                if child_id in declared_children or nodes[child_id].parent != node.node_id:
                    raise ValueError(f'invalid trajectory node {node.node_id}: children do not match parent links')
                declared_children.add(child_id)
        for node in nodes.values():
            if node.parent is not None and node.node_id not in declared_children:
                raise ValueError(f'invalid trajectory node {node.node_id}: children do not match parent links')
        for node in nodes.values():
            ancestors = set()
            current = node.parent
            while current is not None:
                ancestors.add(current)
                current = nodes[current].parent
            invalid = set(node.request_message_overrides) - ancestors
            if invalid:
                raise ValueError(f'node {node.node_id} has request overrides outside its ancestors: {sorted(invalid)}')
        roots = list(data.get('roots', []))
        if len(roots) != len(set(roots)):
            raise ValueError('duplicate trajectory root')
        actual_roots = {node.node_id for node in nodes.values() if node.parent is None}
        if set(roots) != actual_roots:
            raise ValueError('trajectory roots do not match parentless nodes')
        for root in roots:
            if root not in nodes:
                raise ValueError(f'root {root} does not exist')
        chain_owners: dict[NodeId, ChainId] = {}
        for chain in chains.values():
            for root in chain.roots:
                if root not in nodes:
                    raise ValueError(f'chain {chain.chain_id!r} root {root} does not exist')
                owner = chain_owners.setdefault(root, chain.chain_id)
                if owner != chain.chain_id or chain.roots.count(root) > 1:
                    raise ValueError(f'chain root {root} belongs to more than one chain entry')
            for relation, link in (('spawn', chain.spawn), ('join', chain.join)):
                if link is not None and link.node is not None and link.node not in nodes:
                    raise ValueError(f'chain {chain.chain_id!r} {relation} node {link.node} does not exist')
        return cls(
            session_id=data['session_id'],
            state=data.get('state', 'completed'),
            nodes=nodes,
            roots=roots,
            chains=chains,
            timeline=data.get('timeline', []),
            final_reward=data.get('final_reward'),
            termination_reason=data.get('termination_reason'),
            metrics=data.get('metrics', {}),
            extra=data.get('extra', {}),
        )


def dump_trajectory_with_payloads(trajectory: Trajectory, node_payloads: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Export trajectory facts with self-contained Node payloads."""
    data = trajectory.dump()
    data['node_payloads'] = _encode_value(
        {str(node_id): payload for node_id, payload in node_payloads.items()}, '$.node_payloads'
    )
    return data


def load_trajectory_with_payloads(data: dict[str, Any]) -> tuple[Trajectory, dict[int, dict[str, Any]]]:
    """Restore a trajectory and its inlined payloads from ``dump_trajectory_with_payloads``."""
    payloads = _decode_value(data['node_payloads'], '$.node_payloads')
    return Trajectory.from_dump(data), {int(node_id): payload for node_id, payload in payloads.items()}
