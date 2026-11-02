"""Validated private scenario cards and their material-delivery plan."""

from __future__ import annotations

import re

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .simulator import DEFAULT_EVENTS


class ScenarioModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ScenarioMeta(ScenarioModel):
    model_config = ConfigDict(extra='allow')

    script_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
    topic_id: str = Field(min_length=1)
    expected_turns: tuple[int, int] | None = None

    @field_validator('expected_turns')
    @classmethod
    def valid_turn_range(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and not 1 <= value[0] <= value[1]:
            raise ValueError('expected_turns must be a positive [minimum, maximum] range')
        return value


class ScenarioUser(ScenarioModel):
    background: str = Field(min_length=1)
    style: Literal['A', 'B']
    disclosure: str | None = None
    habits: list[str] = Field(default_factory=list)


class ScenarioMaterial(ScenarioModel):
    """One thing the user holds: spoken text, a file whose text goes into the sandbox, or a public URL."""

    name: str = Field(min_length=1)
    via: Literal['口述', '文件'] = '口述'
    content: str | None = None
    url: str | None = None

    @model_validator(mode='after')
    def require_material(self) -> ScenarioMaterial:
        has_content = bool(self.content and self.content.strip())
        if self.url:
            if not self.url.startswith(('https://', 'http://')) or has_content:
                raise ValueError('URL material must contain only a public http(s) URL, not mixed local content')
            return self
        if not has_content:
            raise ValueError(
                f'material {self.name!r} needs its text in content (a file is written into the sandbox from it) '
                'or a public url'
            )
        return self


class ScenarioContext(ScenarioModel):
    initial_state: Any = None
    materials: list[ScenarioMaterial] = Field(default_factory=list)

    @field_validator('materials', mode='before')
    @classmethod
    def name_inline_materials(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [
            {'name': f'材料{index + 1}', 'via': '口述', 'content': item} if isinstance(item, str) else item
            for index, item in enumerate(value)
        ]


NODE_WHEN = {
    'D1': '开场',
    'D2': '开场',
    'D3': '在做时',
    'D4': '在做时',
    'D5': '做完后',
    'D6': '停了以后',
    'D8': '做完后',
    'D9': '做完后',
    'C1': '在做时',
    'C2': '做完后',
    'Q1': '等久了',
    'Q2': '做完后',
    'O1': '做完后',
    'N2': '在做时',
}
OPTIONAL_TYPES = {'S1', 'S2', 'N1'}
EVENT_TYPES = {'question': {'I1', 'I2', 'I3'}, 'permission': {'P1', 'P2', 'P3', 'P4'}, 'failure': {'D7'}}
DEFAULT_EVENT_RESPONSES = DEFAULT_EVENTS


class ScenarioNode(ScenarioModel):
    type: str | None = None
    when: str = Field(min_length=1)
    action: str = Field(min_length=1)


class ScenarioInsertion(ScenarioModel):
    type: str | None = None
    when: str = '空档'
    action: str = Field(min_length=1)


class ScenarioEvent(ScenarioModel):
    types: list[str] = Field(default_factory=list)
    stance: Literal['retry', 'switch', 'accept', 'giveup'] | None = None
    response: str = Field(min_length=1)


class ScenarioEnding(ScenarioModel):
    normal: str = Field(min_length=1)
    early: str | None = None


class ScenarioCard(ScenarioModel):
    """The format in 剧本格式及字段定义.docx; old scenario rows are intentionally rejected."""

    meta: ScenarioMeta
    topic: str = Field(min_length=1)
    user: ScenarioUser
    goal: str = Field(min_length=1)
    context: ScenarioContext = Field(default_factory=ScenarioContext)
    required_nodes: list[ScenarioNode] = Field(default_factory=list)
    optional_insertions: list[ScenarioInsertion] = Field(default_factory=list)
    event_plan: dict[Literal['question', 'permission', 'failure'], ScenarioEvent] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    ending: ScenarioEnding

    @model_validator(mode='after')
    def validate_behavior_tags(self) -> ScenarioCard:
        for index, node in enumerate(self.required_nodes):
            if node.type is None:
                continue
            if node.type not in NODE_WHEN:
                raise ValueError(f'required_nodes[{index}].type must be a required-node behavior')
            # The type is a coverage tag; concrete timing comes from the authored when condition.
            if node.type in {'D1', 'D2'} and not node.when.startswith('开场'):
                raise ValueError('D1/D2 must use an opening condition')
            if node.type in {'D1', 'D2'} and index != 0:
                raise ValueError('opening node must be first')
            if node.type in {'D9', 'C2'} and index != len(self.required_nodes) - 1:
                raise ValueError('D9 and C2 may only be the last required node')
        for node in self.optional_insertions:
            if node.type is not None and node.type not in OPTIONAL_TYPES:
                raise ValueError('optional_insertions.type must be S1, S2, or N1')
        for event, strategy in self.event_plan.items():
            if set(strategy.types) - EVENT_TYPES[event]:
                raise ValueError(f'event_plan.{event}.types has an incompatible behavior tag')
            if event != 'failure' and strategy.stance is not None:
                raise ValueError('stance is only used for event_plan.failure')
        return self

    def assert_fresh_start(self) -> None:
        state = self.context.initial_state
        if state is None or state == '' or state == [] or state == {} or state in ('无', '无任务', '无已有任务'):
            return
        raise ValueError(
            'context.initial_state is nonempty: this fresh-episode runner needs a real restored task/session fixture; '
            'it cannot turn a prose description into existing background tasks'
        )

    def event_responses(self) -> dict[str, str]:
        responses = dict(DEFAULT_EVENT_RESPONSES)
        for name, strategy in self.event_plan.items():
            responses[name] = strategy.response
        return responses


def oral_chunks(text: str, limit: int = 52) -> list[str]:
    """Preserve text in order, allowing long oral materials to fit natural short turns."""
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        candidates = [match.end() for match in re.finditer(r'[。！？；，、,.!?; ]', text[:limit])]
        cut = candidates[-1] if candidates else limit
        chunks.append(text[:cut])
        text = text[cut:]
    return chunks


__all__ = ['DEFAULT_EVENT_RESPONSES', 'ScenarioCard', 'oral_chunks']
