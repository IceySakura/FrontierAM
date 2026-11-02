"""Direct event-fact policy for Nexus-native scaffolds."""

import json

from dataclasses import replace

from nexus.gateway.scaffold_policy import (
    BoundaryFact,
    DeclaredChainClaim,
    DeliveryReceipt,
    GenericScaffoldPolicy,
    InputMessage,
    ObservationEvent,
    OutputMessage,
    RequestEvent,
    ResponseEvent,
    RetrievalIntent,
    SidechainMembership,
    SpawnBinding,
    SpawnIntent,
)


class NexusMainAgentScaffoldPolicy(GenericScaffoldPolicy):
    """Declared-chain attribution, subagent spawn/wait links, and explicit/marker compaction boundaries."""

    def __init__(self):
        super().__init__(policy_id='nexus_main_agent')

    def inspect_event(self, event):
        facts = super().inspect_event(event)
        request = event.request if isinstance(event, ResponseEvent) else event
        if isinstance(request, ObservationEvent):
            messages = request.parent_messages or ()
            meta = request.request_meta or {}
        elif isinstance(request, RequestEvent):
            messages = request.messages
            meta = request.request_meta or {}
        else:
            return facts

        declared = meta.get('declared_chain')
        if not isinstance(declared, dict) or declared.get('chain_id') is None:
            declared_claim = None
            membership = None
            chain_id = 'main'
        else:
            chain_id = str(declared['chain_id'])
            parent_chain_id = None if declared.get('parent_chain_id') is None else str(declared['parent_chain_id'])
            kind = str(declared.get('kind') or 'sidechain')
            membership_attributes = []
            if parent_chain_id is not None:
                membership_attributes.append(('parent_chain_id', parent_chain_id))
            declared_claim = DeclaredChainClaim(chain_id, kind, parent_chain_id)
            membership = (
                None
                if kind == 'main'
                else SidechainMembership(
                    chain_id=chain_id,
                    kind=kind,
                    parent_chain_id=parent_chain_id,
                    detector='declared_chain',
                    attributes=tuple(membership_attributes),
                )
            )

        if isinstance(event, RequestEvent) and messages:
            boundary_facts = self._boundary_facts(messages, meta)
        elif isinstance(event, ResponseEvent) and not messages:
            boundary_facts = self._empty_request_boundary_facts(meta)
        else:
            boundary_facts = ()
        if isinstance(event, ResponseEvent):
            links = self._subagent_links([(OutputMessage(), event.message)])
        elif isinstance(event, RequestEvent):
            links = self._subagent_links((InputMessage(index), message) for index, message in enumerate(messages))
        else:
            links = {}
        timeline = dict(facts.timeline)
        if timeline:
            timeline['is_sidechain'] = membership is not None
        return replace(
            facts,
            context_key=('nexus_main_agent', chain_id, membership is not None),
            membership=membership,
            declared_claim=declared_claim,
            parent_claim=None,
            boundary_facts=boundary_facts,
            timeline=tuple(timeline.items()),
            **links,
        )

    @staticmethod
    def _subagent_links(anchored_messages):
        """Link facts for the main agent's spawn_agent/wait_agent calls and their JSON results.

        A wait result joins only the agents it reports as completed or failed; agents left pending
        by ``return_when='any'`` or a timeout are joined by a later wait.
        """
        links = {'spawn_intents': [], 'spawn_bindings': [], 'retrieval_intents': [], 'delivery_receipts': []}
        tool_names = {}
        for anchor, message in anchored_messages:
            for call in message.get('tool_calls') or ():
                function = call.get('function') or {}
                tool_names[call.get('id')] = name = function.get('name')
                if name == 'spawn_agent' and (prompt := _json_dict(function.get('arguments')).get('prompt')):
                    links['spawn_intents'].append(SpawnIntent(anchor, call.get('id'), prompt))
            call_id = message.get('tool_call_id')
            tool_name = tool_names.get(call_id) if message.get('role') == 'tool' else None
            result = _json_dict(message.get('content')) if tool_name else {}
            if tool_name == 'spawn_agent' and 'agent_id' in result:
                links['spawn_bindings'].append(SpawnBinding(anchor, call_id, result['agent_id']))
            elif tool_name == 'wait_agent':
                joined = [item['agent_id'] for item in (*result.get('completed', ()), *result.get('failed', ()))]
                links['retrieval_intents'] += [RetrievalIntent(anchor, call_id, agent_id) for agent_id in joined]
                if joined:
                    links['delivery_receipts'].append(DeliveryReceipt(anchor, call_id, True))
        return {key: tuple(value) for key, value in links.items()}

    @staticmethod
    def _boundary_facts(messages, meta):
        facts = []
        invocation_kind = meta.get('invocation_kind')
        if invocation_kind in ('summary_inference', 'compaction_resume'):
            facts.append(
                BoundaryFact(
                    'nexus_summary_inference'
                    if invocation_kind == 'summary_inference'
                    else 'nexus_compaction_resume_explicit',
                    InputMessage(len(messages) - 1),
                    invocation_kind,
                    'compaction',
                    from_request_context=True,
                    at_request_edge=True,
                )
            )
        compacted_index = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].get('_is_compacted_summary')),
            None,
        )
        if compacted_index is not None:
            facts.append(
                BoundaryFact(
                    'nexus_compaction_resume',
                    InputMessage(compacted_index),
                    'compaction_resume',
                    'compaction',
                    at_request_edge=compacted_index == len(messages) - 1,
                )
            )
        return tuple(facts)

    @staticmethod
    def _empty_request_boundary_facts(meta):
        invocation_kind = meta.get('invocation_kind')
        if invocation_kind not in ('summary_inference', 'compaction_resume'):
            return ()
        return (
            BoundaryFact(
                'nexus_summary_inference'
                if invocation_kind == 'summary_inference'
                else 'nexus_compaction_resume_explicit',
                OutputMessage(),
                invocation_kind,
                'compaction',
                from_request_context=True,
                at_request_edge=True,
            ),
        )


def _json_dict(text):
    """Parse a JSON object; model-malformed arguments and plain-text tool errors carry no link facts."""
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


NEXUS_MAIN_AGENT_POLICY = NexusMainAgentScaffoldPolicy()
