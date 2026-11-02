"""Generate verifiable flow contracts from task requirements.

Use the direct OpenAI gateway; FLOW_GATEWAY_MODEL overrides the default model.
Contract quality issues are reported separately from generation failures.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

from collections import Counter
from pathlib import Path
from typing import Any


def _prompt() -> str:
    """Reload the contract-generation prompt on each call."""
    from nexus.extensions.tasks.artifact.judges.capabilities.prompts import load_prompt

    override = os.environ.get('FLOW_PROMPT_FILE')
    if override:
        return Path(override).read_text(encoding='utf-8').strip()
    return load_prompt('webjudger_gen_flow.md')


TASK_TYPES = {'svg', 'web', '3d', 'game'}
CHECK_VIA = {'browser', 'source_code'}
TIERS = {'core', 'feature', 'constraint'}
# Tier describes the impact of a failed criterion; weights are owned by verification.

SOURCES = {'explicit', 'implicit'}
EVIDENCE_KINDS = {'visual', 'functional', 'both'}

_WS = re.compile(r'\s+')


def _norm(text: Any) -> str:
    """Normalize whitespace for quote matching."""
    return _WS.sub(' ', str(text)).strip()


def coerce(data: Any) -> dict[str, Any]:
    """Normalize contract structure without semantic quality checks."""
    if not isinstance(data, dict):
        raise ValueError('输出不是 JSON 对象')

    def rows(key: str) -> list[dict[str, Any]]:
        return [r for r in (data.get(key) or []) if isinstance(r, dict)]

    def pick(raw: dict[str, Any], key: str, allowed: set[str], default: str) -> str:
        value = raw.get(key)
        return value if value in allowed else default

    out: dict[str, Any] = {
        'task_type': pick(data, 'task_type', TASK_TYPES, 'web'),
        'type_basis': str(data.get('type_basis') or ''),
        'thin_request': bool(data.get('thin_request')),
        'notes': str(data.get('notes') or ''),
        'requirement_atoms': [],
        'flows': [],
        'unverifiable': [],
    }
    for index, raw in enumerate(rows('requirement_atoms'), 1):
        out['requirement_atoms'].append({
            'rid': str(raw.get('rid') or f'R{index}'),
            'quote': str(raw.get('quote') or ''),
            'demand': str(raw.get('demand') or ''),
        })
    for index, raw in enumerate(rows('flows') or rows('checkpoints'), 1):
        covers = raw.get('covers')
        covers = [covers] if isinstance(covers, str) else (covers if isinstance(covers, list) else [])
        steps = raw.get('steps')
        steps = [steps] if isinstance(steps, str) else (steps if isinstance(steps, list) else [])
        # Normalize accepted field aliases before validating contract semantics.
        via = raw.get('check_via')
        via = 'source_code' if via == 'source' else ('browser' if via == 'both' else via)
        src = 'implicit' if raw.get('source') == 'implied' else raw.get('source')
        criterion = raw.get('criterion') or raw.get('expected') or raw.get('discriminator') or ''
        out['flows'].append({
            'fid': str(raw.get('fid') or raw.get('cid') or f'F{index}'),
            'title': str(raw.get('title') or ''),
            'criterion': str(criterion),
            'check_via': via if via in CHECK_VIA else 'browser',
            'steps': [str(s) for s in steps],
            'covers': [str(c) for c in covers],
            # Use the middle tier for unrecognized values.
            'tier': pick(raw, 'tier', TIERS, 'feature'),
            'source': src if src in SOURCES else 'explicit',
            'evidence_kind': pick(raw, 'evidence_kind', EVIDENCE_KINDS, 'both'),
        })
    for raw in rows('unverifiable'):
        out['unverifiable'].append({'rid': str(raw.get('rid') or ''), 'reason': str(raw.get('reason') or '')})
    return out


def review(contract: dict[str, Any], requirement: str) -> list[str]:
    """Return non-blocking contract quality issues."""
    issues: list[str] = []
    atoms = contract['requirement_atoms']
    flows = contract['flows']
    haystack = _norm(requirement).lower()

    if not atoms and not contract['thin_request']:
        issues.append('no_atoms: 没有 requirement_atoms 且 thin_request=false')

    for kind, ids in (('rid', [a['rid'] for a in atoms]), ('fid', [f['fid'] for f in flows])):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            issues.append(f'duplicate_{kind}: {kind} 重复: {", ".join(dupes)}')

    # Validate requirement quotes against the original task text.
    for atom in atoms:
        quote = _norm(atom['quote'])
        if not quote:
            issues.append(f'empty_quote: {atom["rid"]} 的 quote 是空的')
        elif quote.lower() not in haystack:
            issues.append(f'quote_not_in_request: {atom["rid"]} 的 quote 不是需求原文的子串: {atom["quote"][:70]!r}')

    # Require flow coverage or an explicit unverifiable marker for each requirement.
    known = {a['rid'] for a in atoms}
    excused = {u['rid'] for u in contract['unverifiable']}
    covered = {rid for f in flows for rid in f['covers']}
    for rid in sorted(known - covered - excused):
        issues.append(f'uncovered_atom: {rid} 没被任何 flow 覆盖，也没列进 unverifiable')
    for rid in sorted(excused - known):
        issues.append(f'unknown_unverifiable_rid: unverifiable 引用了不存在的 {rid}')
    for rid in sorted(covered - known):
        issues.append(f'unknown_covered_rid: flow 的 covers 引用了不存在的 {rid}')
    for item in contract['unverifiable']:
        if not item['reason'].strip():
            issues.append(f'no_unverifiable_reason: {item["rid"]} 列为 unverifiable 但没写理由')

    for flow in flows:
        if not flow['criterion'].strip():
            issues.append(f'empty_criterion: {flow["fid"]} 没写 criterion，判分将无从核对')
        if flow['check_via'] == 'browser' and not flow['steps']:
            issues.append(f'no_steps: {flow["fid"]} 要在页面上验但没写 steps')

    # Report insufficient differentiation in the tier distribution.
    if flows and not any(f['tier'] == 'core' for f in flows):
        issues.append('no_core_tier: 一条 core 都没有 —— 没抓住需求主体，所有 flow 会被当同等分量')
    if all(f['tier'] == 'core' for f in flows) and len(flows) > 3:
        issues.append(f'all_core_tier: {len(flows)} 条全是 core —— 没有真的分档，等于没加权')
    return issues


def _extract_json(text: str) -> Any:
    """Extract a JSON object from optional code fences or surrounding text."""
    text = text.strip()
    fenced = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('回复里找不到 JSON 对象')
    return json.loads(text[start : end + 1])


GATEWAY_BASE = 'http://llm-api.model-eval.woa.com/v1'
# Default gateway model; FLOW_GATEWAY_MODEL may override it.
GATEWAY_MODEL = os.environ.get('FLOW_GATEWAY_MODEL') or 'api_aws_third_anthropic.claude-opus-4-7'
MAX_TOKENS = 16000
# Use direct gateway model identifiers, not Nexus/LiteLLM provider-prefixed identifiers.


def _build_client(key_alias: str = 'opus48') -> tuple[Any, str]:
    """Build the gateway client using credentials from M_<alias>_KEY."""
    from openai import AsyncOpenAI

    key = os.environ.get(f'M_{key_alias}_KEY')
    if not key:
        found = sorted(k[2:-4] for k in os.environ if k.startswith('M_') and k.endswith('_KEY'))
        raise SystemExit(
            f'环境变量 M_{key_alias}_KEY 没设。\n'
            f'先 source nexus/extensions/tasks/artifact/.env；有 key 的别名: {", ".join(found) or "(无)"}'
        )
    return AsyncOpenAI(api_key=key, base_url=GATEWAY_BASE, timeout=600.0, max_retries=2), GATEWAY_MODEL


def _failed(instance_id: str, error: str, raw: str = '') -> dict[str, Any]:
    """Build a structured contract-generation failure."""
    return {'instance_id': instance_id, 'ok': False, 'contract': None, 'issues': [], 'error': error, 'raw': raw}


async def _gen_one(
    client: Any,
    model: str,
    requirement: str,
    *,
    instance_id: str = '',
) -> dict[str, Any]:
    """Generate one contract and capture failures in the result row."""
    from nexus.extensions.tasks.artifact.judges.capabilities.prompts import render_prompt

    prompt = render_prompt(_prompt(), user_requirement=requirement)
    try:
        reply = await client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=MAX_TOKENS,
        )
        raw = (reply.choices[0].message.content or '').strip()
    except Exception as error:
        return _failed(instance_id, f'call_failed: {type(error).__name__}: {error}')

    try:
        contract = coerce(_extract_json(raw))
    except Exception as error:
        return _failed(instance_id, f'parse_failed: {error}', raw=raw[:800])
    if not contract['flows']:
        return _failed(instance_id, 'no_flows: 契约里没有任何 flow', raw=raw[:800])

    return {
        'instance_id': instance_id,
        'ok': True,
        'contract': contract,
        'issues': review(contract, requirement),
        'error': '',
        'raw': '',
    }


# ── 批量生成 ──────────────────────────────────────────────────────────────────
async def call_llm(
    rows: list[dict[str, Any]],
    *,
    key_alias: str = 'opus48',
    concurrency: int = 128,
) -> list[dict[str, Any]]:
    """Generate contracts concurrently while preserving input order."""
    client, model = _build_client(key_alias)
    limit = asyncio.Semaphore(max(1, concurrency))
    done = 0

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        instance_id = str(row.get('instance_id') or '')
        requirement = str(row.get('problem_statement') or row.get('query') or row.get('instruction') or '')
        if not requirement.strip():
            result = _failed(instance_id, 'empty_request: 需求为空')
        else:
            async with limit:
                result = await _gen_one(client, model, requirement, instance_id=instance_id)
        done += 1
        mark = 'ok  ' if result['ok'] else 'FAIL'
        if result['ok']:
            contract = result['contract']
            tail = f'{contract["task_type"]:5} {len(contract["flows"])}flows'
            if result['issues']:
                tail += f'  ({len(result["issues"])} issues)'
        else:
            tail = result['error'][:80]
        print(f'  [{done}/{len(rows)}] {mark} {instance_id[:40]:42} {tail}', flush=True)
        return result

    return list(await asyncio.gather(*(one(r) for r in rows)))


def read_data(path: Path, *, limit: int | None = None, ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Read JSONL rows with optional instance filtering."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if ids and str(row.get('instance_id')) not in ids:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def save_data(results: list[dict[str, Any]], path: Path) -> None:
    """Write one JSONL result row per instance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def summarise(results: list[dict[str, Any]]) -> None:
    """Report generation success, flow distribution, and quality issues."""
    ok = [r for r in results if r['ok']]
    print(f'\nok {len(ok)}/{len(results)}')
    if ok:
        counts = sorted(len(r['contract']['flows']) for r in ok)
        vias: Counter[str] = Counter()
        srcs: Counter[str] = Counter()
        for row in ok:
            vias.update(f['check_via'] for f in row['contract']['flows'])
            srcs.update(f['source'] for f in row['contract']['flows'])
        print(f'  task_type   : {dict(Counter(r["contract"]["task_type"] for r in ok))}')
        print(f'  flows/case  : min={counts[0]} p50={counts[len(counts) // 2]} max={counts[-1]}')
        tiers: Counter[str] = Counter()
        for row in ok:
            tiers.update(f['tier'] for f in row['contract']['flows'])
        print(f'  check_via   : {dict(vias)}')
        print(f'  tier        : {dict(tiers)}')
        print(f'  source      : {dict(srcs)}')
        print(f'  thin_request: {sum(1 for r in ok if r["contract"]["thin_request"])}')
        print(f'  unverifiable: {sum(len(r["contract"]["unverifiable"]) for r in ok)} 条')
        print(f'  clean       : {sum(1 for r in ok if not r["issues"])}/{len(ok)} 份契约零 issue')
    failed: Counter[str] = Counter(r['error'].split(':')[0] for r in results if not r['ok'])
    if failed:
        print(f'  failures    : {dict(failed.most_common())}')
    issues: Counter[str] = Counter()
    for row in ok:
        issues.update(i.split(':')[0] for i in row['issues'])
    if issues:
        print(f'  issues      : {dict(issues.most_common())}')


def show(path: Path, index: int = 0) -> None:
    """Display a contract for inspection."""
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    row = rows[index]
    print(f'instance_id : {row["instance_id"]}   ok={row["ok"]}')
    if not row['ok']:
        print(f'error       : {row["error"]}')
        if row.get('raw'):
            print(f'raw         : {row["raw"]}')
        return
    contract = row['contract']
    print(f'task_type   : {contract["task_type"]}  （{contract["type_basis"]}）')
    print(f'thin_request: {contract["thin_request"]}')
    print(f'\n需求原子 ({len(contract["requirement_atoms"])}):')
    for atom in contract['requirement_atoms']:
        print(f'  {atom["rid"]:4} {atom["demand"]}')
        print(f'       « {atom["quote"][:88]} »')
    print(f'\nflows ({len(contract["flows"])}):')
    for flow in contract['flows']:
        print(
            f'  {flow["fid"]:4} [{flow["tier"]:10}] [{flow["check_via"]:11}] '
            f'[{flow["source"]:8}] covers={",".join(flow["covers"]) or "-"}'
        )
        print(f'       {flow["criterion"]}')
        if flow['steps']:
            print(f'       步骤: {" → ".join(flow["steps"])}')
    if contract['unverifiable']:
        print(f'\nunverifiable ({len(contract["unverifiable"])}):')
        for item in contract['unverifiable']:
            print(f'  {item["rid"]:4} {item["reason"]}')
    if contract['notes']:
        print(f'\nnotes: {contract["notes"]}')
    if row['issues']:
        print(f'\nissues ({len(row["issues"])}):')
        for issue in row['issues']:
            print(f'  - {issue}')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Stage 1: 需求 → flow 契约', formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset')
    parser.add_argument('--out')
    # Select the credential alias independently of FLOW_GATEWAY_MODEL.
    parser.add_argument(
        '--key-alias',
        default=os.environ.get('FLOW_KEY_ALIAS', 'opus48'),
        help='用哪个 M_<alias>_KEY 的凭据（默认 opus48）',
    )
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--concurrency', type=int, default=int(os.environ.get('FLOW_CONCURRENCY', '128')))
    parser.add_argument('--instance-ids', default='')
    parser.add_argument('--show', metavar='FLOWS_JSONL', help='人读一条契约后退出')
    parser.add_argument('--index', type=int, default=0, help='--show 打第几条')
    args = parser.parse_args()

    if args.show:
        show(Path(args.show), args.index)
        return 0
    if not args.dataset or not args.out:
        parser.error('要么给 --dataset 和 --out，要么用 --show')

    ids = {i.strip() for i in args.instance_ids.split(',') if i.strip()} or None
    rows = read_data(Path(args.dataset), limit=args.limit, ids=ids)
    if not rows:
        raise SystemExit('没有选中任何数据')
    print(
        f'{len(rows)} 条需求  model={GATEWAY_MODEL}  key=M_{args.key_alias}_KEY  concurrency={args.concurrency}\n',
        flush=True,
    )

    results = asyncio.run(call_llm(rows, key_alias=args.key_alias, concurrency=args.concurrency))
    out = Path(args.out)
    save_data(results, out)
    summarise(results)
    print(f'\n→ {out}')
    print(f'  看一条: python {Path(__file__).name} --show {out}')
    return 0 if any(r['ok'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
