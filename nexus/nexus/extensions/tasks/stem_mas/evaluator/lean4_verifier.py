"""Lean4 verifier evaluator for formal stem_mas math benchmarks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.error
import urllib.request

from typing import Any, Literal

from nexus.extensions.agents.base.evaluation import Evaluator
from nexus.extensions.tasks.stem_mas.evaluator.common import strip_think_blocks


logger = logging.getLogger(__name__)

_FINAL_PROOF_RE = re.compile(r'<final_proof>\s*(.*?)\s*</final_proof>', re.DOTALL | re.IGNORECASE)
_FINAL_PROOF_OPEN_RE = re.compile(r'<final_proof>\s*', re.IGNORECASE)
_FINAL_PROOF_CLOSE_RE = re.compile(r'</final_proof>', re.IGNORECASE)
_LEAN_FENCE_RE = re.compile(r'```(?:lean4|lean)?\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)

CandidateKind = Literal['proof_body']
CandidatePolicyName = Literal['proof_body', 'target_declaration_proof']


class Lean4CandidatePolicy:
    """Normalize model output into the candidate shape expected by lean4-verifier."""

    def extract_candidate_text(self, prediction: Any) -> str:
        text = strip_think_blocks(str(prediction or '')).strip()
        final_matches = [match.group(1) for match in _FINAL_PROOF_RE.finditer(text) if match.group(1).strip()]
        if final_matches:
            return self.strip_markdown_fence(final_matches[-1])

        open_matches = list(_FINAL_PROOF_OPEN_RE.finditer(text))
        if open_matches:
            text = text[open_matches[-1].end() :]
            close_match = _FINAL_PROOF_CLOSE_RE.search(text)
            if close_match:
                text = text[: close_match.start()]
            return self.strip_markdown_fence(text)

        fences = _LEAN_FENCE_RE.findall(text)
        if fences:
            return self.strip_markdown_fence(fences[-1])

        return self.strip_markdown_fence(text)

    def normalize_candidate_text(self, text: str, question: str | None = None) -> str:
        return text

    @staticmethod
    def strip_markdown_fence(text: str) -> str:
        text = text.strip()
        if text.startswith('```') and text.endswith('```'):
            lines = text.splitlines()
            if len(lines) >= 2:
                return '\n'.join(lines[1:-1]).strip()
        return text


class ProofBodyCandidatePolicy(Lean4CandidatePolicy):
    """Normalize proof bodies according to the theorem's final placeholder."""

    def normalize_candidate_text(self, text: str, question: str | None = None) -> str:
        if self.placeholder_expects_tactic_body(question):
            return self.strip_leading_by(text)
        return text

    @staticmethod
    def placeholder_expects_tactic_body(question: str | None) -> bool:
        if not question:
            return False
        stripped = question.strip()
        return bool(re.search(r':=\s*by\s+sorry\b', stripped)) or bool(re.search(r':=\s*by\s*$', stripped))

    @staticmethod
    def strip_leading_by(text: str) -> str:
        stripped = text.strip()
        if stripped == 'by':
            return ''
        if stripped.startswith('by\n'):
            return stripped[3:].lstrip()
        if stripped.startswith('by '):
            return stripped[3:].lstrip()
        return text


class FullCodeToProofBodyPolicy(ProofBodyCandidatePolicy):
    """Extract the benchmark target proof from an official-style Lean file."""

    _DECL_RE = re.compile(
        r'(?m)^(?:noncomputable\s+)?(?:private\s+)?(?:theorem|lemma|example)'
        r'(?:\s+(?P<name>[^\s(:{]+))?\b'
    )
    _TOP_LEVEL_RE = re.compile(
        r'(?m)^(?:noncomputable\s+)?(?:private\s+)?'
        r'(?:theorem|lemma|example|def|abbrev|instance)\b'
    )

    def normalize_candidate_text(self, text: str, question: str | None = None) -> str:
        candidate = self.extract_proof_body_from_full_code(text, question)
        return super().normalize_candidate_text(candidate, question)

    def extract_proof_body_from_full_code(self, text: str, question: str | None = None) -> str:
        stripped = text.strip()
        if not self.looks_like_full_code(stripped):
            return stripped

        segment = self.target_declaration_segment(stripped, question)

        proof = self.proof_rhs(segment)
        if not proof:
            return stripped
        return self.strip_trailing_namespace_ends(proof)

    @classmethod
    def looks_like_full_code(cls, text: str) -> bool:
        return bool(re.search(r'(?m)^\s*import\s+', text)) or bool(cls._DECL_RE.search(text))

    @classmethod
    def target_declaration_segment(cls, text: str, question: str | None) -> str:
        declarations = list(cls._DECL_RE.finditer(text))
        if not declarations:
            raise CandidateFormatError(
                'Complete Lean output does not contain a theorem, lemma, or example declaration.'
            )

        target_name = cls.target_declaration_name(question)
        if not target_name:
            raise CandidateFormatError('Benchmark question does not expose a named target declaration.')
        target = next(
            (match for match in reversed(declarations) if cls.same_declaration_name(match.group('name'), target_name)),
            None,
        )
        if target is None:
            raise CandidateFormatError(f'Complete Lean output does not contain target declaration {target_name!r}.')
        following = cls._TOP_LEVEL_RE.search(text, target.end())
        end = following.start() if following else len(text)
        return text[target.start() : end].strip()

    @classmethod
    def target_declaration_name(cls, question: str | None) -> str | None:
        if not question:
            return None
        declarations = list(cls._DECL_RE.finditer(question))
        for match in reversed(declarations):
            if match.group('name'):
                return match.group('name')
        return None

    @staticmethod
    def same_declaration_name(candidate: str | None, target: str) -> bool:
        if not candidate:
            return False
        return candidate == target or candidate.rsplit('.', 1)[-1] == target.rsplit('.', 1)[-1]

    @staticmethod
    def proof_rhs(declaration: str) -> str:
        marker = declaration.find(':=')
        if marker < 0:
            return ''
        return declaration[marker + 2 :].strip()

    @staticmethod
    def strip_trailing_namespace_ends(text: str) -> str:
        lines = text.strip().splitlines()
        while lines and re.fullmatch(r"\s*end(?:\s+[A-Za-z0-9_'.]+)?\s*", lines[-1]):
            lines.pop()
        return '\n'.join(lines).strip()


class CandidateFormatError(ValueError):
    """The model output cannot be normalized into the benchmark candidate contract."""


CANDIDATE_POLICIES: dict[CandidatePolicyName, type[Lean4CandidatePolicy]] = {
    'proof_body': ProofBodyCandidatePolicy,
    'target_declaration_proof': FullCodeToProofBodyPolicy,
}


def build_candidate_policy(policy_name: CandidatePolicyName) -> Lean4CandidatePolicy:
    try:
        policy_cls = CANDIDATE_POLICIES[policy_name]
    except KeyError as exc:
        known = ', '.join(sorted(CANDIDATE_POLICIES))
        raise ValueError(f'Unknown Lean candidate policy {policy_name!r}. Known policies: {known}.') from exc
    return policy_cls()


class Lean4VerifierError(RuntimeError):
    """The verifier request failed before a benchmark correctness result was produced."""


class Lean4VerifierEvaluator(Evaluator):
    """Evaluate a Lean proof body with the external lean4-verifier service."""

    last_judge_result: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        url: str | None = None,
        benchmark: str,
        task_id: str,
        expected_manifest_version: str | None = None,
        timeout_seconds: int = 300,
        candidate_kind: CandidateKind = 'proof_body',
        candidate_policy: CandidatePolicyName = 'proof_body',
        forbid_sorry: bool = True,
    ) -> None:
        self.url = (
            url or os.getenv('LEAN4_VERIFIER_URL') or 'http://lean4-verifier-v0d2d0.production.polaris:8080'
        ).rstrip('/')
        self.benchmark = benchmark
        self.task_id = task_id
        self.expected_manifest_version = expected_manifest_version
        self.timeout_seconds = timeout_seconds
        self.candidate_kind = candidate_kind
        self.candidate_policy_name = candidate_policy
        self.candidate_policy = build_candidate_policy(candidate_policy)
        self.forbid_sorry = forbid_sorry

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        try:
            candidate = self.candidate_policy.normalize_candidate_text(
                self.candidate_policy.extract_candidate_text(prediction),
                question,
            )
        except CandidateFormatError as exc:
            self.last_judge_result = {
                'request': self.request_details(),
                'candidate_preview': '',
                'response': {
                    'request_ok': True,
                    'correct': False,
                    'score': 0.0,
                    'failure_type': 'invalid_candidate',
                    'error': str(exc),
                },
            }
            return 0.0
        payload = {
            'mode': 'official',
            'benchmark': self.benchmark,
            'task_id': self.task_id,
            'candidate': {
                'kind': self.candidate_kind,
                'text': candidate,
            },
            'timeout_seconds': self.timeout_seconds,
            'forbid_sorry': self.forbid_sorry,
        }
        self.last_judge_result = {
            'request': self.request_details(),
            'candidate_preview': candidate[:500],
            'response': None,
        }
        try:
            response = await asyncio.to_thread(self.post_json, '/verify/lean4', payload)
        except urllib.error.HTTPError as exc:
            self.last_judge_result['response'] = {
                'request_ok': False,
                'correct': False,
                'score': 0.0,
                'failure_type': 'http_error',
                'status': exc.code,
                'reason': exc.reason,
            }
            raise Lean4VerifierError(f'Lean4 verifier HTTP {exc.code}: {exc.reason}') from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.last_judge_result['response'] = {
                'request_ok': False,
                'correct': False,
                'failure_type': 'transport_error',
                'error': str(exc),
            }
            raise Lean4VerifierError(f'Lean4 verifier transport error: {exc}') from exc
        except json.JSONDecodeError as exc:
            self.last_judge_result['response'] = {
                'request_ok': False,
                'correct': False,
                'failure_type': 'protocol_error',
                'error': str(exc),
            }
            raise Lean4VerifierError(f'Lean4 verifier returned invalid JSON: {exc}') from exc
        if not isinstance(response, dict):
            raise Lean4VerifierError(f'Lean4 verifier returned {type(response).__name__}, expected an object')
        self.last_judge_result['response'] = response
        if response.get('request_ok') is not True:
            failure_type = response.get('failure_type') or 'request_failed'
            error = response.get('error') or 'no error detail'
            raise Lean4VerifierError(f'Lean4 verifier {failure_type}: {error}')
        if response.get('failure_type') == 'timeout':
            raise Lean4VerifierError('Lean4 verifier timed out before producing a correctness result')
        self.validate_manifest_version(response)
        score = response.get('score')
        if isinstance(score, int | float):
            return float(score)
        return 1.0 if response.get('correct') is True else 0.0

    def request_details(self) -> dict[str, Any]:
        return {
            'url': self.url,
            'benchmark': self.benchmark,
            'task_id': self.task_id,
            'expected_manifest_version': self.expected_manifest_version,
            'candidate_kind': self.candidate_kind,
            'candidate_policy': self.candidate_policy_name,
            'timeout_seconds': self.timeout_seconds,
            'forbid_sorry': self.forbid_sorry,
        }

    def validate_manifest_version(self, response: dict[str, Any]) -> None:
        if not self.expected_manifest_version:
            return
        task = response.get('task')
        actual = task.get('manifest_version') if isinstance(task, dict) else None
        if actual != self.expected_manifest_version:
            raise Lean4VerifierError(
                f'Lean4 verifier manifest mismatch: expected {self.expected_manifest_version!r}, received {actual!r}'
            )

    def get_last_judge_result(self) -> dict | None:
        return self.last_judge_result

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            f'{self.url}{path}',
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds + 30) as response:
                raw = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            logger.error('Lean4 verifier HTTP error: status=%s body=%s', exc.code, body[:1000])
            raise
        return json.loads(raw)


__all__ = ['Lean4VerifierError', 'Lean4VerifierEvaluator']
