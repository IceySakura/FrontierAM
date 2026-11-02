"""Prompt profiles for official formal-math benchmark styles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LeanPromptProfile:
    """Build the single-shot Lean proof prompt expected by a benchmark verifier."""

    name: str
    theorem_suffix: str = ':= by sorry'
    proof_instruction: str = (
        'The verifier will replace the final `sorry` placeholder with the tag content. '
        'If the theorem ends with `:= by sorry`, output only the tactic body without a leading `by`. '
        'If the theorem ends with `:= sorry`, output a complete Lean proof term; tactic proofs should start with `by`.'
    )

    def system_prompt(self) -> str:
        return (
            'You are a Lean 4 formal theorem prover. Output only one visible '
            '<final_proof></final_proof> block containing Lean proof code. Do not write natural-language reasoning, '
            'analysis, markdown, imports, theorem statements, namespaces, or any text outside the tag.'
        )

    def build_prompt(self, *, lean_version: str, header: str, statement: str) -> str:
        theorem = self.normalize_theorem(statement)
        return (
            f'Prove the following Lean 4({lean_version}) theorem using the official {self.name} verifier style:\n\n'
            f'{header}\n\n{theorem}\n\n'
            'Put exactly one visible final answer in <final_proof></final_proof>. '
            f'{self.proof_instruction} '
            'Do not include imports, theorem statements, namespaces, markdown fences, or explanatory text. '
            'Do not leave the final answer empty. Even if you are unsure, output your best Lean proof candidate '
            'inside <final_proof></final_proof>.'
        )

    def normalize_theorem(self, statement: str) -> str:
        theorem = statement.strip()
        if theorem.endswith(':= by'):
            return f'{theorem} sorry'
        if self.has_sorry_placeholder(theorem):
            return theorem
        return f'{theorem} {self.theorem_suffix}'

    @staticmethod
    def has_sorry_placeholder(theorem: str) -> bool:
        return ':= by sorry' in theorem or ':= sorry' in theorem or theorem.endswith('sorry')


@dataclass(frozen=True)
class OfficialFullCodePromptProfile(LeanPromptProfile):
    """Prompt for benchmarks whose generation contract returns complete Lean code."""

    include_natural_language_proof: bool = False

    def system_prompt(self) -> str:
        return (
            'You are an expert in Lean 4 and mathematics. You may reason briefly, but your final answer must contain '
            'exactly one Lean code block fenced with ```lean4 and ```. The code block should be the complete Lean 4 '
            'file or declaration needed to prove the given statement. Do not use sorry or admit.'
        )

    def build_prompt(self, *, lean_version: str, header: str, statement: str) -> str:
        theorem = self.normalize_theorem(statement)
        nl_instruction = (
            ' You may write a concise natural-language proof before the Lean code.'
            if self.include_natural_language_proof
            else ''
        )
        header_text = f'\n\nContext/header:\n```lean4\n{header.strip()}\n```' if header.strip() else ''
        return (
            f'Complete the following Lean 4({lean_version}) theorem using the official {self.name} benchmark style.'
            f'{nl_instruction}\n'
            'Do not change the theorem statement. Copy the final statement to prove exactly in your final Lean code. '
            'The target proof must be self-contained: do not reference new helper declarations, because verification '
            'extracts only the target proof body. Include only necessary imports and namespaces. '
            'Put the complete Lean code in the last ```lean4 fenced block. Do not use sorry or admit.'
            f'{header_text}\n\nStatement to complete:\n```lean4\n{theorem}\n```'
        )


class MiniF2FPromptProfile(LeanPromptProfile):
    """miniF2F official tasks mostly ask for tactic bodies under `:= by`."""

    def __init__(self) -> None:
        super().__init__(
            name='miniF2F',
            proof_instruction=(
                'miniF2F official verification expects a proof body for the theorem stub. '
                'For `:= by sorry`, output only the tactic body without a leading `by`. '
                'For the rare `:= sorry` stub, output a complete proof term; tactic proofs should start with `by`.'
            ),
        )


class PutnamBenchPromptProfile(OfficialFullCodePromptProfile):
    """Use the strict full-file contract validated by PutnamBench experiments."""

    def __init__(self) -> None:
        super().__init__(
            name='PutnamBench',
            theorem_suffix=':= sorry',
            include_natural_language_proof=False,
            proof_instruction=(
                'PutnamBench official verification accepts a full Lean file, but rejects non-allowlisted imports '
                'and official-forbidden tokens such as set_option, axiom, constant, and unsafe.'
            ),
        )

    def normalize_theorem(self, statement: str) -> str:
        theorem = statement.strip()
        theorem = theorem.replace(':= by\n  sorry', ':= by\n  -- REPLACE_WITH_PROOF')
        theorem = theorem.replace(':= by sorry', ':= by\n  -- REPLACE_WITH_PROOF')
        theorem = theorem.replace(':=\n  sorry', ':= by\n  -- REPLACE_WITH_PROOF')
        return theorem.replace(':= sorry', ':= REPLACE_WITH_DEFINITION')

    def build_prompt(self, *, lean_version: str, header: str, statement: str) -> str:
        del header
        theorem = self.normalize_theorem(statement)
        return (
            f'Complete the following Lean 4({lean_version}) template using the official {self.name} benchmark style.\n'
            'Your final answer must contain exactly one final ```lean4 fenced block with a complete Lean file. '
            'Preserve theorem names, types, binders, and intended statements, but replace every placeholder with valid Lean. '
            'Official PutnamBench verification rejects candidates containing forbidden boilerplate. Follow these hard rules:\n'
            '- Start the Lean block with exactly one import line: `import Mathlib`.\n'
            '- Do not include any other import line.\n'
            '- Never write `import Aesop`, `set_option`, `axiom`, `constant`, or `unsafe` anywhere in the Lean block.\n'
            '- Never include `sorry` or `admit`.\n'
            '- Do not add helper declarations referenced by the target proof; verification extracts only its proof body.\n'
            '- Never copy `REPLACE_WITH_PROOF`, `REPLACE_WITH_DEFINITION`, or comments from the template.\n'
            '- Do not copy benchmark scaffolding/header lines; they are intentionally omitted from this prompt.\n'
            '- Return only the final complete Lean file; do not include commentary outside that final code block.\n'
            'If you are unsure, still provide the best complete Lean proof that obeys these restrictions.\n\n'
            f'Template to complete:\n```lean4\n{theorem}\n```'
        )


class FateEvalPromptProfile(OfficialFullCodePromptProfile):
    """Use FATE-Eval full-code generation and proof-body verification."""

    def __init__(self) -> None:
        super().__init__(
            name='FATE-Eval',
            include_natural_language_proof=True,
            proof_instruction=(
                'FATE-Eval official verification ultimately replaces the target `sorry`, but generation should '
                'produce a complete Lean code block. The evaluator extracts the target proof body before scoring.'
            ),
        )


PROMPT_PROFILES: dict[str, Callable[[], LeanPromptProfile]] = {
    'minif2f': MiniF2FPromptProfile,
    'putnam_full_code': PutnamBenchPromptProfile,
    'fate_full_code': FateEvalPromptProfile,
}


def build_math_prompt_profile(profile_name: str, benchmark_name: str) -> LeanPromptProfile:
    if profile_name == 'proof_body':
        return LeanPromptProfile(name=benchmark_name)
    try:
        profile_cls = PROMPT_PROFILES[profile_name]
    except KeyError as exc:
        known = ', '.join(['proof_body', *sorted(PROMPT_PROFILES)])
        raise ValueError(f'Unknown math prompt profile {profile_name!r}. Known profiles: {known}.') from exc
    return profile_cls()


__all__ = [
    'FateEvalPromptProfile',
    'LeanPromptProfile',
    'MiniF2FPromptProfile',
    'OfficialFullCodePromptProfile',
    'PROMPT_PROFILES',
    'PutnamBenchPromptProfile',
    'build_math_prompt_profile',
]
