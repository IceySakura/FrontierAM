# ==============================
# File: verify_proof.py
# Desc: Semantic verification of model-generated Lean4 proofs against a
#       trusted formal statement, hardened against known reward-hacking
#       patterns.
#
# The authoritative checks all happen at the *Lean semantic layer* (elaborated
# `Expr` + kernel axiom closure), never at the source-text layer, because text
# filters have repeatedly proven bypassable:
#   - `apply?` / `hint` emit no literal `sorry` yet depend on `sorryAx`
#   - `run_cmd`/`addDecl` create axioms with no literal `axiom` keyword
#   - `local notation` hides `Real.exp` behind an innocent-looking source line
#
# Threat model and the check that stops each attack:
#   1. syntactic sorry / apply? / hint / exact?  -> `sorryAx` in axiom closure
#   2. custom `axiom` (any modifier/namespace)   -> axiom allowlist
#   3. dynamically injected axiom (`addDecl`)    -> axiom allowlist
#   4. `opaque` as an unproved assumption        -> `sorryAx` / allowlist
#   5. injected hypothesis (`variable`+`include`)-> elaborated type mismatch
#   6. shadowing `exp`/`sqrt` via `notation`     -> elaborated type mismatch
#   7. `OfNat` overload of numeric literals      -> elaborated type mismatch
#   8. redefining a `def`/`instance` that the    -> dependency-closure value
#      statement refers to                          comparison
#   9. `#exit` truncating the audit harness      -> fail-closed on missing marker
#  10. shadowing the harness itself              -> fail-closed (duplicate decl)
#  11. tab/newline injection into the payload    -> hex-encoded payload fields
#  12. proving a differently-named theorem       -> exact target constant name
# ==============================

from __future__ import annotations

import asyncio
import hashlib
import json
import re

from typing import Any


DEFAULT_SERVER_URL = 'http://lean4-server-test-v2.production.polaris:8080/check_whole'
DEFAULT_REPO_PATH = '/miniF2F-lean4'
DEFAULT_TIMEOUT = 600
DEFAULT_RETRY = 3
# Base delay for exponential backoff between transient retries.
DEFAULT_BACKOFF_SECONDS = 1.0

# Mathlib's own foundational axioms. Anything outside this set is an unproved
# assumption smuggled in by the candidate.
#
# `native_decide` is deliberately NOT allowlisted: it discharges a goal by
# running compiled code and trusting the result, bypassing kernel checking, and
# emits a per-declaration axiom named `<decl>._native.native_decide.ax_N`. Since
# that name is not in this set it is rejected as an unpermitted axiom -- which is
# the intended policy. Do not add a pattern for it without an explicit decision;
# plain `decide` is kernel-checked and needs no exemption.
DEFAULT_PERMITTED_AXIOMS = ('Classical.choice', 'propext', 'Quot.sound')

# Namespace for the audit harness. Deliberately unusual: if a candidate tries to
# pre-declare any of these names, Lean raises "has already been declared" and the
# harness markers never appear, which we treat as a failure (fail-closed).
_HARNESS_NS = 'VPAudit_7q3zx'

_MARK_BEGIN = '@@VP_BEGIN@@'
_MARK_END = '@@VP_END@@'
_MARK_TGT = '@@VP_TGT@@'
_MARK_MISS = '@@VP_MISS@@'
_MARK_DECL = '@@VP_D@@'

_THEOREM_KINDS = frozenset({'theorem', 'lemma', 'example'})
# Kinds that may legitimately carry the goal. `def`/`abbrev`/`instance` are
# supported because some datasets state the goal as a definition rather than a
# theorem.
_TARGET_KINDS = frozenset({'theorem', 'lemma', 'example', 'def', 'abbrev', 'instance'})
# Lean's ConstantInfo kind for a given source keyword. `lemma` is a macro for
# `theorem`; `abbrev`/`instance` are elaborated as `def`. The candidate's kind
# must agree with the statement's, so a `def` cannot masquerade as a `theorem`.
_SOURCE_KIND_TO_LEAN = {
    'theorem': 'theorem',
    'lemma': 'theorem',
    'example': 'theorem',
    'def': 'def',
    'abbrev': 'def',
    'instance': 'def',
}
# Non-theorem goals get a caveat in other_message: their type may be data-valued
# (any inhabitant satisfies it) rather than a proposition.
_VALUE_CHECKED_KINDS = frozenset({'def', 'abbrev', 'instance'})

# Declaration-start regex. Unlike a naive `^(theorem|lemma|...)` this tolerates
# modifiers and same-line attributes; missing them would make declarations
# invisible to the parser and silently skip their checks.
_MODIFIERS = (
    r'(?:@\[[^\]]*\]\s*|private\s+|protected\s+|noncomputable\s+|partial\s+|unsafe\s+|nonrec\s+|scoped\s+|local\s+)*'
)
_DECL_RE = re.compile(
    r'^(?P<indent>[ \t]*)'
    + _MODIFIERS
    + r'(?P<kind>theorem|lemma|def|abbrev|axiom|opaque|example|structure|class|inductive|instance)\b'
    r'(?:\s+(?P<name>[A-Za-z_α-ωΑ-Ω][\w\'.α-ωΑ-Ω]*))?',
    re.MULTILINE,
)
_NAMESPACE_RE = re.compile(r'^namespace\s+([A-Za-z_][\w\'.]*(?:\s+[A-Za-z_][\w\'.]*)*)\s*$')
_END_RE = re.compile(r'^end(?:\s+([A-Za-z_][\w\'.]*))?\s*$')
_SORRY_RE = re.compile(r'(?<![\w\'])(sorry|admit)(?![\w\'])')
# A Lean char literal is always a closed short form: 'a', '\n', 'ÿ'.
# Requiring the closing quote keeps notation like `∑'`, `⨆'`, `∏'` (whose
# apostrophe follows a non-identifier symbol) from being read as the start of a
# char literal, which would blank out the rest of the file.
_CHAR_LITERAL_RE = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^\\'\n])'")


# --------------------------------------------------------------------------- #
# source-text helpers (used only for locating the target, never for security)
# --------------------------------------------------------------------------- #
def mask_lean_noise(content: str) -> str:
    """Blank out comments / strings / char literals / «escaped idents».

    Layout is preserved so offsets stay valid. Used so that a `sorry` mentioned
    inside a comment is not mistaken for a real one (and vice versa).
    """
    if not content:
        return content
    chars = list(content)
    i = 0
    depth = 0
    hashes = 0
    state = 'code'
    escaped = False
    prev_ident = False

    def blank(idx: int) -> None:
        if chars[idx] not in '\r\n':
            chars[idx] = ' '

    while i < len(content):
        char = content[i]
        nxt = content[i + 1] if i + 1 < len(content) else ''
        if state == 'line_comment':
            if char in '\r\n':
                state = 'code'
            else:
                blank(i)
            i += 1
            continue
        if state == 'block_comment':
            if char == '/' and nxt == '-':
                blank(i)
                blank(i + 1)
                depth += 1
                i += 2
                continue
            if char == '-' and nxt == '/':
                blank(i)
                blank(i + 1)
                depth -= 1
                if depth == 0:
                    state = 'code'
                i += 2
                continue
            blank(i)
            i += 1
            continue
        if state == 'string':
            blank(i)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                state = 'code'
            i += 1
            continue
        if state == 'raw_string':
            if char == '"' and content[i + 1 : i + 1 + hashes] == '#' * hashes:
                for k in range(i, min(i + 1 + hashes, len(content))):
                    blank(k)
                i += 1 + hashes
                state = 'code'
                continue
            blank(i)
            i += 1
            continue
        if state == 'char':
            blank(i)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == "'":
                state = 'code'
            i += 1
            continue
        if state == 'escaped_identifier':
            blank(i)
            if char == '»':
                state = 'code'
            i += 1
            continue

        # state == 'code'
        raw_quote = None
        if char == 'r':
            cand = i + 1
            while cand < len(content) and content[cand] == '#':
                cand += 1
            if cand < len(content) and content[cand] == '"':
                raw_quote = cand
        if raw_quote is not None:
            for k in range(i, raw_quote + 1):
                blank(k)
            hashes = raw_quote - i - 1
            state = 'raw_string'
            i = raw_quote + 1
            prev_ident = False
            continue
        if char == '-' and nxt == '-':
            blank(i)
            blank(i + 1)
            state = 'line_comment'
            i += 2
            continue
        if char == '/' and nxt == '-':
            blank(i)
            blank(i + 1)
            depth = 1
            state = 'block_comment'
            i += 2
            continue
        if char == '"':
            blank(i)
            state = 'string'
            i += 1
            prev_ident = False
            continue
        if char == "'" and not prev_ident and _CHAR_LITERAL_RE.match(content, i):
            blank(i)
            escaped = False
            state = 'char'
            i += 1
            continue
        if char == '«':
            blank(i)
            state = 'escaped_identifier'
            i += 1
            prev_ident = False
            continue
        prev_ident = char.isalnum() or char == '_'
        i += 1
    return ''.join(chars)


def _namespace_at(masked: str, position: int) -> str:
    """Namespace prefix in effect at `position`.

    `end` only pops the namespace stack when it actually closes a namespace --
    an anonymous `end` closing a `section` must not pop it, or the target's
    fully-qualified name comes out wrong and a valid proof gets rejected.
    """
    stack: list[str] = []
    section_depth = 0
    offset = 0
    for line in masked.splitlines(keepends=True):
        start = offset
        offset += len(line)
        if start >= position:
            break
        stripped = line.strip()
        m = _NAMESPACE_RE.match(stripped)
        if m:
            stack.extend(m.group(1).split())
            continue
        if re.match(r'^section(?:\s+[A-Za-z_][\w\'.]*)?\s*$', stripped):
            section_depth += 1
            continue
        m = _END_RE.match(stripped)
        if m:
            name = m.group(1)
            if name is not None:
                # named end: close matching namespace if any
                if stack and (stack[-1] == name or '.'.join(stack).endswith(name)):
                    stack.pop()
                elif section_depth > 0:
                    section_depth -= 1
            elif section_depth > 0:
                section_depth -= 1
            elif stack:
                stack.pop()
    return '.'.join(stack)


def find_target_declaration(statement: str) -> tuple[str, str] | tuple[None, None]:
    """Locate the declaration the statement is asking to prove.

    Callers sometimes put prelude definitions in `statement` ahead of the goal,
    so we take the *last* eligible declaration rather than the first. A goal is
    usually a theorem/lemma, but a `def`/`abbrev`/`instance` is accepted too
    since some datasets state goals that way.

    Returns (fully_qualified_name, source_kind).
    """
    masked = mask_lean_noise(statement)
    best: tuple[str, str] | None = None
    for m in _DECL_RE.finditer(masked):
        kind = m.group('kind')
        name = m.group('name')
        if kind not in _TARGET_KINDS or not name:
            continue
        ns = _namespace_at(masked, m.start())
        best = (f'{ns}.{name}' if ns else name, kind)
    return best if best is not None else (None, None)


# --------------------------------------------------------------------------- #
# Lean audit harness
# --------------------------------------------------------------------------- #
def _harness(target: str) -> str:
    """Lean code appended after the content, dumping the semantic facts we trust.

    Notes on the hardening choices:
      * `Lean.collectAxioms` is called fully qualified; a candidate cannot
        shadow it (redeclaring it is a hard Lean error -> fail-closed).
      * `canonE` rewrites every binder name to `x`, so an alpha-renamed but
        otherwise identical statement is not reported as a mismatch.
      * every payload field is hex-encoded, so tabs/newlines/markers inside a
        string literal in the elaborated type cannot forge or split a record.
      * `getModuleIdxFor?` skips imported (Mathlib) constants, leaving only the
        declarations this file introduced -- exactly the untrusted surface.
    """
    body = r"""
namespace __NS__
private def hexDigit (n : Nat) : Char :=
  if n < 10 then Char.ofNat (48 + n) else Char.ofNat (87 + n)
private def hx (s : String) : String :=
  s.toUTF8.foldl (fun acc b => (acc.push (hexDigit (b.toNat / 16))).push (hexDigit (b.toNat % 16))) ""
private partial def canonE : Lean.Expr -> Lean.Expr
  | .forallE _ d b bi => .forallE `x (canonE d) (canonE b) bi
  | .lam _ d b bi => .lam `x (canonE d) (canonE b) bi
  | .letE _ t v b nd => .letE `x (canonE t) (canonE v) (canonE b) nd
  | .app f a => .app (canonE f) (canonE a)
  | .mdata _ b => canonE b
  | .proj s i b => .proj s i (canonE b)
  | e => e
private def kindOf : Lean.ConstantInfo -> String
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "def"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quot"
  | .ctorInfo _ => "ctor"
  | .recInfo _ => "rec"
  | .inductInfo _ => "inductive"
private def valOf : Lean.ConstantInfo -> String
  | .thmInfo _ => ""
  | .axiomInfo _ => ""
  | ci => match ci.value? with
          | some v => toString (canonE v)
          | none => ""

def audit (target : Lean.Name) : Lean.CoreM Unit := do
  let env <- Lean.getEnv
  IO.println "@@VP_BEGIN@@"
  match env.find? target with
  | none => IO.println ("@@VP_MISS@@" ++ hx (toString target))
  | some tci =>
    let axs <- Lean.collectAxioms target
    IO.println ("@@VP_TGT@@" ++ hx (toString target) ++ "|" ++ hx (kindOf tci)
      ++ "|" ++ hx (toString (canonE tci.type))
      ++ "|" ++ hx (String.intercalate "," (axs.toList.map toString)))
    let mut seen : Lean.NameSet := Lean.NameSet.empty
    let mut work : Array Lean.Name := #[target]
    let mut fuel : Nat := 200000
    while work.size > 0 && fuel > 0 do
      fuel := fuel - 1
      let n := work[work.size - 1]!
      work := work.pop
      if seen.contains n then
        continue
      seen := seen.insert n
      if (env.getModuleIdxFor? n).isSome then
        continue
      match env.find? n with
      | none => pure ()
      | some ci =>
        IO.println ("@@VP_D@@" ++ hx (toString n) ++ "|" ++ hx (kindOf ci)
          ++ "|" ++ hx (toString (canonE ci.type)) ++ "|" ++ hx (valOf ci))
        for m in ci.type.getUsedConstants do
          work := work.push m
        match ci.value? with
        | some v => for m in v.getUsedConstants do work := work.push m
        | none => pure ()
  IO.println "@@VP_END@@"
end __NS__
#eval __NS__.audit `__TARGET__
"""
    return body.replace('__NS__', _HARNESS_NS).replace('__TARGET__', target)


def _parse_audit(repl_res: dict[str, Any]) -> dict[str, Any]:
    """Decode harness output. Returns {'ran','missing','target','decls'}."""
    out: dict[str, Any] = {'ran': False, 'missing': None, 'target': None, 'decls': {}}
    if not isinstance(repl_res, dict):
        return out
    for msg in repl_res.get('messages') or []:
        data = msg.get('data') if isinstance(msg, dict) else None
        if not data:
            continue
        for line in str(data).splitlines():
            line = line.strip()
            if line.startswith(_MARK_BEGIN):
                out['ran'] = True
            elif line.startswith(_MARK_MISS):
                out['missing'] = _unhex(line[len(_MARK_MISS) :])
            elif line.startswith(_MARK_TGT):
                f = _split_fields(line[len(_MARK_TGT) :], 4)
                if f:
                    out['target'] = {
                        'name': f[0],
                        'kind': f[1],
                        'type': f[2],
                        'axioms': [a for a in f[3].split(',') if a],
                    }
            elif line.startswith(_MARK_DECL):
                f = _split_fields(line[len(_MARK_DECL) :], 4)
                if f:
                    out['decls'][f[0]] = {'kind': f[1], 'type': f[2], 'value': f[3]}
    return out


def _split_fields(payload: str, count: int) -> list[str] | None:
    parts = payload.split('|')
    if len(parts) != count:
        return None
    try:
        return [_unhex(p) for p in parts]
    except ValueError:
        return None


def _unhex(text: str) -> str:
    return bytes.fromhex(text.strip()).decode('utf8', errors='replace')


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #
def _normalize_url(server_url: str) -> str:
    url = server_url.strip()
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'http://' + url
    if not url.rstrip('/').endswith('/check_whole'):
        url = url.rstrip('/') + '/check_whole'
    return url


async def async_call_check_whole(
    cmd: str,
    server_url: str = DEFAULT_SERVER_URL,
    repo_path: str = DEFAULT_REPO_PATH,
    timeout: int = DEFAULT_TIMEOUT,
    retry_num: int = DEFAULT_RETRY,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    session: Any = None,
) -> dict[str, Any]:
    """POST one Lean file to the remote check_whole service.

    Only *transient* failures are retried. A 4xx means the request itself is
    malformed (bad payload, unknown route, oversized body); replaying it
    verbatim would fail identically while tripling load, so we return at once.
    Retries use exponential backoff so a struggling service is not hammered.

    Pass `session` to reuse an existing ``aiohttp.ClientSession`` across many
    calls; otherwise one is created per call.
    """
    import aiohttp

    url = _normalize_url(server_url)
    payload = {'repo_path': repo_path, 'cmd': cmd, 'timeout': timeout, 'return_repl_res': True}
    attempts = max(1, retry_num)
    last = 'not healthy'
    last_body: dict[str, Any] | None = None

    async def post(client: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return (final_result, retryable_body). final_result stops the loop."""
        nonlocal last, last_body
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with client.post(url, json=payload, proxy=None, timeout=client_timeout) as resp:
            raw = await resp.text()
            status = resp.status
        if 400 <= status < 500 and status != 429:
            # Deterministic client error: retrying cannot help.
            detail = raw[:200]
            last = f'HTTPError {status}{f" | {detail}" if detail else ""}'
            return {'is_ok': False, 'exception': last, 'http_status': status}, None
        if status >= 400:
            last = f'HTTPError {status}{f" | {raw[:200]}" if raw else ""}'
            return None, None
        try:
            body = json.loads(raw)
        except ValueError as exc:
            last = f'malformed JSON response: {exc}: {raw[:200]!r}'
            return None, None
        if isinstance(body, dict) and body:
            if body.get('exception'):
                # Service-side failure (e.g. REPL OOM). Worth retrying, but keep
                # the body so its diagnostics survive if all tries fail.
                last = str(body.get('exception'))
                last_body = body
                return None, body
            return body, None
        last = f'unexpected response shape: {type(body).__name__}'
        return None, None

    owns_session = session is None
    client = session
    try:
        if owns_session:
            client = aiohttp.ClientSession()
        for attempt in range(1, attempts + 1):
            try:
                result, _ = await post(client)
                if result is not None:
                    return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - network/protocol failures
                last = f'{type(exc).__name__}: {exc}'
            if attempt < attempts and backoff_seconds > 0:
                await asyncio.sleep(backoff_seconds * (2 ** (attempt - 1)))
    finally:
        if owns_session and client is not None:
            await client.close()

    if last_body is not None:
        # Preserve the service's own payload (repl_res, messages) on give-up.
        out = dict(last_body)
        out['is_ok'] = False
        out['exception'] = last
        return out
    return {'is_ok': False, 'exception': last}


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #
def _repl_errors(repl_res: dict[str, Any]) -> list[str]:
    out = []
    for msg in (repl_res or {}).get('messages') or []:
        if isinstance(msg, dict) and msg.get('severity') == 'error':
            out.append(str(msg.get('data') or '').strip())
    return out


def _repl_sorry_warning(repl_res: dict[str, Any]) -> bool:
    for msg in (repl_res or {}).get('messages') or []:
        if not isinstance(msg, dict):
            continue
        data = str(msg.get('data') or '').replace('`', "'")
        if "declaration uses 'sorry'" in data:
            return True
    return False


_GROUND_TRUTH_CACHE: dict[str, dict[str, Any]] = {}
# In-flight reference elaborations, keyed the same way as the cache. RL rollouts
# hit the same statement many times concurrently; without this, every one of them
# would miss the (still empty) cache and duplicate the reference call. Waiters
# share the first task's result instead.
_GROUND_TRUTH_INFLIGHT: dict[str, Any] = {}


async def _async_ground_truth(
    header: str,
    statement: str,
    target: str,
    server_url: str,
    repo_path: str,
    timeout: int,
    retry_num: int,
    use_cache: bool,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    session: Any = None,
) -> dict[str, Any]:
    """Elaborate `header+statement` to obtain the trusted reference facts.

    Only *successful* elaborations are cached. A transient failure (network
    error, service restart, timeout) must not be memoized: doing so would pin
    every later candidate for this statement to a spurious `is_ok=False` for the
    lifetime of the process.
    """
    key = hashlib.sha256(f'{header}\x00{statement}\x00{target}'.encode()).hexdigest()
    if use_cache:
        cached = _GROUND_TRUTH_CACHE.get(key)
        if cached is not None:
            return cached
        inflight = _GROUND_TRUTH_INFLIGHT.get(key)
        if inflight is not None:
            # Someone else is already elaborating this exact reference; wait for
            # it rather than issuing a duplicate call. `shield` so that a
            # cancelled waiter does not abort the shared task.
            return await asyncio.shield(inflight)

    async def elaborate() -> dict[str, Any]:
        cmd = _join(header, statement) + _harness(target)
        res = await async_call_check_whole(cmd, server_url, repo_path, timeout, retry_num, backoff_seconds, session)
        audit = _parse_audit(res.get('repl_res') or {})
        out = {'raw': res, 'audit': audit}
        # A usable reference requires the harness to have run AND reported the
        # target. Anything else is either a broken statement or a flaky call --
        # retry next time rather than caching the failure.
        usable = bool(audit['ran']) and audit['target'] is not None and not res.get('exception')
        if use_cache and usable:
            _GROUND_TRUTH_CACHE[key] = out
        return out

    if not use_cache:
        return await elaborate()

    task = asyncio.ensure_future(elaborate())
    _GROUND_TRUTH_INFLIGHT[key] = task
    try:
        return await asyncio.shield(task)
    finally:
        _GROUND_TRUTH_INFLIGHT.pop(key, None)


def _join(*parts: str) -> str:
    chunks = [p.strip('\n') for p in parts if p and p.strip()]
    return '\n\n'.join(chunks) + '\n'


def _looks_like_full_file(model_proof: str) -> bool:
    """Whether the model emitted a self-contained file (its own imports)."""
    return any(line.strip().startswith('import ') for line in mask_lean_noise(model_proof).splitlines())


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
async def async_verify_proof(
    header: str,
    statement: str,
    model_proof: str,
    server_url: str = DEFAULT_SERVER_URL,
    permitted_axioms: tuple[str, ...] | list[str] = DEFAULT_PERMITTED_AXIOMS,
    repo_path: str = DEFAULT_REPO_PATH,
    timeout: int = DEFAULT_TIMEOUT,
    retry_num: int = DEFAULT_RETRY,
    use_ground_truth_cache: bool = True,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    session: Any = None,
) -> dict[str, Any]:
    """Verify that `model_proof` really proves the theorem in `header+statement`.

    Args:
        header: preamble (imports, options, prelude definitions). Passed through
            verbatim -- no imports or options are injected on the caller's
            behalf; the model is responsible for whatever it needs.
        statement: the goal, ending in `:= by sorry`. Prelude definitions live
            either here or in `header`, never in both.
        model_proof: the model's complete output. May be a self-contained file
            (own imports) or just the body to append after `header`.
        server_url: remote `check_whole` endpoint.
        permitted_axioms: kernel axioms the proof may rely on.
        session: optional ``aiohttp.ClientSession`` to reuse across calls.

    Returns:
        {'is_ok': bool, 'repl_res': dict, 'other_message': str}
    """
    notes: list[str] = []
    permitted = set(permitted_axioms)

    # ---------- pre-checks on the trusted statement ----------
    if not statement or not statement.strip():
        return _fail({}, ['PRECHECK: statement is empty'])
    if not model_proof or not model_proof.strip():
        return _fail({}, ['PRECHECK: model_proof is empty'])

    # Resolve against header+statement: the namespace may be opened in the header
    # and left unclosed, in which case the goal's real constant name is prefixed
    # by it and looking at `statement` alone would miss the target.
    target, kind = find_target_declaration(f'{header}\n{statement}')
    if target is None:
        target, kind = find_target_declaration(statement)
    if target is None:
        return _fail(
            {},
            [
                'PRECHECK: no theorem/lemma/def declaration found in statement '
                f'(supported kinds: {", ".join(sorted(_TARGET_KINDS))})'
            ],
        )
    if kind == 'example':
        return _fail({}, ["PRECHECK: 'example' has no stable constant name and cannot be verified"])
    expected_lean_kind = _SOURCE_KIND_TO_LEAN[kind]
    if kind in _VALUE_CHECKED_KINDS:
        notes.append(
            f"NOTE: goal '{target}' is stated as {kind}, not theorem. Only its type and "
            'axiom closure are verified; if its type is data-valued (e.g. `: Int`) rather '
            'than a Prop, ANY inhabitant satisfies it -- state such goals as a theorem '
            'about the value instead.'
        )

    masked_stmt = mask_lean_noise(statement)
    if not _SORRY_RE.search(masked_stmt):
        notes.append("PRECHECK WARNING: statement does not end with ':= by sorry'")

    # ---------- resolve ground truth ----------
    gt = await _async_ground_truth(
        header,
        statement,
        target,
        server_url,
        repo_path,
        timeout,
        retry_num,
        use_ground_truth_cache,
        backoff_seconds,
        session,
    )
    gt_audit = gt['audit']
    if not gt_audit['ran'] or gt_audit['target'] is None:
        detail = gt['raw'].get('exception') or _repl_errors(gt['raw'].get('repl_res') or {})
        return _fail(
            gt['raw'].get('repl_res') or {},
            [
                'GROUND_TRUTH: could not elaborate the reference statement '
                f"'{target}'; header+statement must compile on their own",
                f'detail: {detail}',
            ],
            infra_error=True,
        )
    expected_type = gt_audit['target']['type']
    expected_decls = gt_audit['decls']

    # ---------- build and run the candidate ----------
    if _looks_like_full_file(model_proof):
        candidate_src = model_proof.strip('\n') + '\n'
        notes.append('model_proof treated as a self-contained file (it declares its own imports)')
    else:
        candidate_src = _join(header, model_proof)
        notes.append('model_proof appended after header (no imports of its own)')

    cand_raw = await async_call_check_whole(
        candidate_src + _harness(target),
        server_url,
        repo_path,
        timeout,
        retry_num,
        backoff_seconds,
        session,
    )
    repl_res = cand_raw.get('repl_res') or {}

    if cand_raw.get('exception'):
        return _fail(
            repl_res,
            notes + [f'CANDIDATE: service exception: {cand_raw["exception"]}'],
            infra_error=True,
        )

    # The service refuses some inputs before Lean ever runs (e.g. a literal
    # `axiom` declaration). Surface that verdict rather than masking it.
    server_errors = cand_raw.get('error')
    if server_errors and not repl_res.get('messages'):
        return _fail(repl_res, notes + [f'CANDIDATE: rejected by check_whole pre-check: {server_errors}'])

    violations: list[str] = []
    errors = _repl_errors(repl_res)
    if errors:
        violations.append(f'COMPILE_ERROR: {errors[:3]}')

    audit = _parse_audit(repl_res)

    # Fail-closed: `#exit`, a crash, or a shadowed harness all manifest as
    # missing markers. Never interpret "no evidence" as success.
    if not audit['ran']:
        detail = ''
        if server_errors:
            # e.g. check_whole's own text pre-filter refused a literal `axiom`
            detail = f' server_error={server_errors}'
        elif errors:
            detail = f' first_error={errors[0][:200]!r}'
        violations.append(
            'AUDIT_MISSING: harness produced no output (rejected before Lean ran, '
            'truncated by #exit, shadowed declaration, or compile failure) '
            '-- refusing to pass.' + detail
        )
    elif audit['missing'] is not None:
        violations.append(
            f"MISSING_TARGET: candidate does not define '{audit['missing']}' "
            '(wrong name, or wrapped in an unclosed namespace)'
        )
    elif audit['target'] is None:
        violations.append('AUDIT_INCOMPLETE: target record absent from harness output')
    else:
        tgt = audit['target']
        # `lemma` elaborates to `theorem`, `abbrev`/`instance` to `def`, so the
        # comparison is against the statement's own kind, not a fixed one.
        if tgt['kind'] != expected_lean_kind:
            violations.append(
                f"KIND_MISMATCH: '{target}' is a {tgt['kind']}, but the statement "
                f'declares it as {kind} (expected Lean kind {expected_lean_kind})'
            )

        # (5)(6)(7) semantic statement tampering: injected hypotheses, notation
        # shadowing and OfNat overloads all change the elaborated type.
        if tgt['type'] != expected_type:
            violations.append(f"TYPE_MISMATCH: elaborated statement of '{target}' differs from the reference")

        # NOTE: we deliberately do NOT compare the target's body against the
        # reference. The reference target is stated as `:= by sorry`, so its
        # value is always `sorryAx ...` while the candidate's is a real proof
        # term -- comparing them would reject every submission. For a `def`
        # goal, correctness is instead pinned down by the type comparison plus
        # the axiom closure; if the caller needs the body fixed as well, the
        # definition belongs in `header` (where it is compared as a dependency).

        # (1)(2)(3)(4) unproved assumptions of any origin.
        for axiom in tgt['axioms']:
            if axiom == 'sorryAx':
                violations.append(
                    f"SORRY_DEPENDENCY: '{target}' depends on sorryAx "
                    '(explicit sorry/admit, or apply?/hint/exact? partial proof)'
                )
            elif axiom not in permitted:
                violations.append(f"UNPERMITTED_AXIOM: '{target}' depends on axiom '{axiom}'")

        # (8) the statement's own prelude constants must not be redefined.
        for name, info in expected_decls.items():
            if name == target:
                continue
            got = audit['decls'].get(name)
            if got is None:
                continue
            if got['type'] != info['type']:
                violations.append(f"DEPENDENCY_TYPE_CHANGED: '{name}' was redefined with a new type")
            elif got['value'] != info['value']:
                violations.append(f"DEPENDENCY_VALUE_CHANGED: '{name}' was redefined with a different body")

        # Any locally-declared axiom/opaque reachable from the target is an
        # unproved assumption even if collectAxioms folded it away.
        for name, info in audit['decls'].items():
            if info['kind'] in {'axiom', 'opaque'} and name not in permitted:
                violations.append(f"LOCAL_ASSUMPTION: proof depends on locally declared {info['kind']} '{name}'")

    if _repl_sorry_warning(repl_res):
        violations.append('SORRY_WARNING: Lean reported "declaration uses \'sorry\'"')

    violations = _dedupe(violations)
    if violations:
        return _fail(repl_res, notes + violations)
    return {
        'is_ok': True,
        'repl_res': repl_res,
        'other_message': '\n'.join(notes + [f"verified '{target}': type matches, axioms clean"]),
        'infra_error': False,
    }


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _fail(repl_res: dict[str, Any], messages: list[str], infra_error: bool = False) -> dict[str, Any]:
    """Build a failure result.

    `infra_error=True` marks a failure that is NOT evidence against the proof
    (service exception, unusable reference). Callers doing RL scoring should
    retry these rather than treating them as a rejected proof.
    """
    return {
        'is_ok': False,
        'repl_res': repl_res,
        'other_message': '\n'.join(messages),
        'infra_error': infra_error,
    }
