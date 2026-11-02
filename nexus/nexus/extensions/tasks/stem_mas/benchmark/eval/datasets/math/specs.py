"""Static benchmark specs for stem_mas math datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PromptProfileName = Literal['proof_body', 'minif2f', 'fate_full_code', 'putnam_full_code']
CandidatePolicyName = Literal['proof_body', 'target_declaration_proof']


PUTNAM_2025_PROBLEM_NAMES = (
    'putnam_2025_a1',
    'putnam_2025_a2',
    'putnam_2025_a3',
    'putnam_2025_a4',
    'putnam_2025_a5',
    'putnam_2025_a6',
    'putnam_2025_b1',
    'putnam_2025_b2',
    'putnam_2025_b3',
    'putnam_2025_b4',
    'putnam_2025_b5',
    'putnam_2025_b6',
)


@dataclass(frozen=True)
class MathBenchmarkSpec:
    name: str
    task_id_field: str
    source_fields: tuple[str, ...]
    prompt_profile: PromptProfileName
    candidate_policy: CandidatePolicyName
    lean_version: str
    lean4_verifier_benchmark: str
    lean4_verifier_manifest: str
    header_field: str | None = None
    metadata_fields: tuple[str, ...] = ()
    lean4_verifier_task_id_zero_pad_width: int | None = None
    hf_id: str | None = None
    hf_file: str | None = None
    hf_revision: str | None = None
    included_task_ids: frozenset[str] | None = None
    source_url: str | None = None


MATH_BENCHMARK_SPECS: dict[str, MathBenchmarkSpec] = {
    spec.name: spec
    for spec in (
        MathBenchmarkSpec(
            name='minif2f',
            task_id_field='name',
            source_fields=(
                'formal_proof',
                'formal_statement',
                'header',
                'informal_proof',
                'informal_statement',
                'name',
                'split',
            ),
            header_field='header',
            metadata_fields=('split',),
            prompt_profile='minif2f',
            candidate_policy='proof_body',
            lean_version='v4.30.0',
            lean4_verifier_benchmark='minif2f',
            lean4_verifier_manifest='minif2f-v2-905eca84',
            hf_id='roozbeh-yz/miniF2F_v2',
            hf_file='miniF2F_v2c.jsonl',
            hf_revision='065cb8057ccfc6e70f4afdd4d8542306c0f7cdb2',
        ),
        MathBenchmarkSpec(
            name='putnam672',
            task_id_field='problem_name',
            source_fields=(
                'formal_statement',
                'informal_solution',
                'informal_statement',
                'problem_name',
                'split',
                'tags',
            ),
            metadata_fields=('split', 'tags'),
            prompt_profile='putnam_full_code',
            candidate_policy='target_declaration_proof',
            lean_version='v4.27.0',
            lean4_verifier_benchmark='putnam672',
            lean4_verifier_manifest='putnambench-c8c8fa28',
            hf_id='ChristianZ97/PutnamBench-lean4',
            hf_file='putnam_bench_lean4.jsonl',
            hf_revision='f753021f5fd7e40da05e137c0d0cb8624790f227',
        ),
        MathBenchmarkSpec(
            name='putnam-2025',
            task_id_field='problem_name',
            source_fields=(
                'formal_statement',
                'informal_solution',
                'informal_statement',
                'problem_name',
                'split',
                'tags',
            ),
            metadata_fields=('split', 'tags'),
            prompt_profile='putnam_full_code',
            candidate_policy='target_declaration_proof',
            lean_version='v4.27.0',
            lean4_verifier_benchmark='putnam672',
            lean4_verifier_manifest='putnambench-c8c8fa28',
            hf_id='ChristianZ97/PutnamBench-lean4',
            hf_file='putnam_bench_lean4.jsonl',
            hf_revision='f753021f5fd7e40da05e137c0d0cb8624790f227',
            included_task_ids=frozenset(PUTNAM_2025_PROBLEM_NAMES),
        ),
        MathBenchmarkSpec(
            name='fatem',
            task_id_field='id',
            source_fields=('formal_statement', 'id', 'informal_statement', 'source'),
            metadata_fields=('source',),
            prompt_profile='fate_full_code',
            candidate_policy='target_declaration_proof',
            lean_version='v4.16.0',
            lean4_verifier_benchmark='fatem',
            lean4_verifier_manifest='fate-eval-fe358fb5',
            lean4_verifier_task_id_zero_pad_width=3,
            source_url=(
                'https://raw.githubusercontent.com/frenzymath/FATE-Eval/'
                'fe358fb5b0a53fdafaaf156540a2ecbf9b5b98a4/data/FATE-M.json'
            ),
        ),
        MathBenchmarkSpec(
            name='fateh',
            task_id_field='id',
            source_fields=('formal_statement', 'id', 'informal_statement', 'source', 'tag'),
            metadata_fields=('source', 'tag'),
            prompt_profile='fate_full_code',
            candidate_policy='target_declaration_proof',
            lean_version='v4.16.0',
            lean4_verifier_benchmark='fateh',
            lean4_verifier_manifest='fate-eval-fe358fb5',
            lean4_verifier_task_id_zero_pad_width=3,
            source_url=(
                'https://raw.githubusercontent.com/frenzymath/FATE-Eval/'
                'fe358fb5b0a53fdafaaf156540a2ecbf9b5b98a4/data/FATE-H.json'
            ),
        ),
        MathBenchmarkSpec(
            name='fatex',
            task_id_field='id',
            source_fields=('formal_statement', 'id', 'informal_statement', 'source', 'tag'),
            metadata_fields=('source', 'tag'),
            prompt_profile='fate_full_code',
            candidate_policy='target_declaration_proof',
            lean_version='v4.16.0',
            lean4_verifier_benchmark='fatex',
            lean4_verifier_manifest='fate-eval-fe358fb5',
            lean4_verifier_task_id_zero_pad_width=3,
            source_url=(
                'https://raw.githubusercontent.com/frenzymath/FATE-Eval/'
                'fe358fb5b0a53fdafaaf156540a2ecbf9b5b98a4/data/FATE-X.json'
            ),
        ),
    )
}


__all__ = ['CandidatePolicyName', 'MATH_BENCHMARK_SPECS', 'MathBenchmarkSpec', 'PromptProfileName']
