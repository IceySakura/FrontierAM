"""Configuration contract for Harbor tool-call bad-pattern scoring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CHINESE_CODE_MT_AGENT = 'ch_code_mt'
CHINESE_FINAL_LANGUAGE_RULE = 'final_turn_language_ch'
DEFAULT_FINAL_LANGUAGE_RULE = 'final_turn_language'


class BadPatternRulePolicy(BaseModel):
    """Control detection and penalty eligibility for one rule."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    penalize: bool = True
    # Optional rule-local settlement overrides. Unset values inherit the
    # policy-level fixed penalty and reward threshold.
    penalty: float | None = Field(default=None, ge=0.0)
    reward_threshold: float | None = None
    max_thinking_chars: int | None = Field(default=None, gt=0, strict=True)


class LegacyDeferredPenaltyPolicy(BaseModel):
    """One legacy detector family settled later by the HyTuner trainer."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    penalty: float = Field(default=0.0, ge=0.0)
    reward_threshold: float = 1.0


class ToolCallBadPatternPolicy(BaseModel):
    """Versioned, opt-in policy for Host-side trajectory checks."""

    model_config = ConfigDict(extra='forbid')

    id: str
    patterns: dict[str, BadPatternRulePolicy] = Field(default_factory=dict)
    penalty: float = Field(default=0.0, ge=0.0)
    reward_threshold: float = 1.0
    max_hits: int = Field(default=100, ge=0)
    # Backward-compatible field name: this also makes any tool/think protocol
    # finding on a truncated final assistant turn monitor-only, even when the
    # visible protocol tags happen to be balanced.
    ignore_incomplete_final_assistant: bool = True

    @field_validator('id')
    @classmethod
    def _require_non_empty_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError('must be a non-empty string')
        return normalized

    @model_validator(mode='after')
    def _validate_patterns(self) -> ToolCallBadPatternPolicy:
        from nexus.extensions.harbor_bridge.bad_pattern_operator.rules import RULE_TITLES

        unknown = set(self.patterns) - set(RULE_TITLES)
        if unknown:
            raise ValueError(f'unknown bad patterns: {", ".join(sorted(unknown))}')
        for rule_id, config in self.patterns.items():
            if config.max_thinking_chars is not None and rule_id != 'single_turn_thinking_too_long':
                raise ValueError('max_thinking_chars is only supported by single_turn_thinking_too_long')
        if not any(config.enabled for config in self.patterns.values()):
            raise ValueError('patterns must enable at least one bad-pattern rule')
        return self

    def enabled_rules(self) -> set[str]:
        return {rule_id for rule_id, config in self.patterns.items() if config.enabled}

    def penalized_rules(self) -> set[str]:
        return {rule_id for rule_id, config in self.patterns.items() if config.enabled and config.penalize}

    def penalty_for(self, rule_id: str) -> float:
        config = self.patterns.get(rule_id)
        return self.penalty if config is None or config.penalty is None else config.penalty

    def reward_threshold_for(self, rule_id: str) -> float:
        config = self.patterns.get(rule_id)
        return self.reward_threshold if config is None or config.reward_threshold is None else config.reward_threshold

    def enabled_rules_for_agent(self, agent_name: str | None) -> set[str]:
        enabled_rules = self.enabled_rules()
        # NOTE(speedzhu): The ratio-based Chinese rule replaces, rather than stacks
        # with, final_turn_language and is intentionally scoped to ch_code_mt.
        if str(agent_name or '') == CHINESE_CODE_MT_AGENT:
            enabled_rules.discard(DEFAULT_FINAL_LANGUAGE_RULE)
        else:
            enabled_rules.discard(CHINESE_FINAL_LANGUAGE_RULE)
        return enabled_rules

    def detection_kwargs(self, *, agent_name: str | None = None) -> dict[str, Any]:
        enabled_rules = self.enabled_rules_for_agent(agent_name)
        kwargs: dict[str, Any] = {
            'enabled_rules': enabled_rules,
            'max_hits': self.max_hits,
            'ignore_incomplete_final_assistant': self.ignore_incomplete_final_assistant,
        }
        thinking = self.patterns.get('single_turn_thinking_too_long')
        if (
            'single_turn_thinking_too_long' in enabled_rules
            and thinking is not None
            and thinking.max_thinking_chars is not None
        ):
            kwargs['max_thinking_chars'] = thinking.max_thinking_chars
        return kwargs

    def enabled_legacy_deferred(self) -> set[str]:
        """Compatibility shim for the existing task call site; v2 has no legacy families."""
        return set()


__all__ = ['BadPatternRulePolicy', 'LegacyDeferredPenaltyPolicy', 'ToolCallBadPatternPolicy']
