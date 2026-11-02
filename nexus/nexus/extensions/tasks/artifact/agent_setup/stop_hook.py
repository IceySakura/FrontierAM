"""Stop hook — auto-replies via the in-sandbox Anthropic proxy so brainstorming-style
skills can advance past 'awaiting user confirmation' moments in non-interactive batch runs.

Strategy:
- Read ALL assistant texts from the transcript (not just the latest) and persist a running
  conversation in /tmp/_nexus_stop_hook_state.json. Each Stop event appends new assistant
  turns as user-role messages and the proxy's reply as assistant-role messages, so the
  request prefix stays stable across hook calls — Bedrock prompt cache can hit.
- The simulated-user 'system' prompt is sent with cache_control=ephemeral so the static
  context (instructions) is cached separately from the growing conversation tail.
- The simulated user is told to favor implementation over spec-only deliverables: emit
  STOP_HOOK_DONE only when actual code/HTML files are written, not just .md specs.

Persona modes (NEXUS_STOP_HOOK_PERSONA):
- "nudger" (default): push assistant to finish quickly, emit STOP_HOOK_DONE on code delivery.
- "iterative_user": after first code delivery, provide iterative design feedback (visual
  tweaks, missing interactions, new features) for N rounds before stopping. Produces
  genuine multi-turn training data.

Opt-in: set NEXUS_STOP_HOOK_ENABLE=true to enable. Default is no-op.
"""

import contextlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from pathlib import Path
from typing import Any, TypedDict


class _StopHookState(TypedDict):
    messages: list[dict[str, Any]]
    last_seen_assistant_idx: int
    count: int


MAX_AUTO_REPLIES = int(os.environ.get('NEXUS_STOP_HOOK_MAX_REPLIES', '8'))
COUNTER_FILE = Path('/tmp/_nexus_stop_hook_count')
STATE_FILE = Path('/tmp/_nexus_stop_hook_state.json')
HOOK_TIMEOUT = float(os.environ.get('NEXUS_STOP_HOOK_TIMEOUT', '60'))
HOOK_MODEL = os.environ.get('NEXUS_STOP_HOOK_MODEL', 'claude-opus-4-7')
# Max tokens for the simulated-user reply. Was hardcoded 300, which truncated verbose
# models (e.g. opus-4-8) to an empty reply (300 text tokens generated, none surfaced).
HOOK_MAX_TOKENS = int(os.environ.get('NEXUS_STOP_HOOK_MAX_REPLY_TOKENS', '64000'))

# Persona selection: "nudger" (default, existing behavior) or "iterative_user" (multi-turn feedback).
HOOK_PERSONA = os.environ.get('NEXUS_STOP_HOOK_PERSONA', 'nudger').strip().lower()
# SAFETY CEILING on iterative feedback rounds (iterative_user mode only) — NOT a target.
# The proposer decides when to stop (emits STOP_HOOK_DONE once genuinely satisfied); this
# is just a runaway guard. Round count is driven by the conversation, not by this number,
# so keep it generous. Override per-run via NEXUS_STOP_HOOK_ITERATIVE_TURNS.
ITERATIVE_TURNS = int(os.environ.get('NEXUS_STOP_HOOK_ITERATIVE_TURNS', '8'))
# JSON-encoded list of feedback direction hints from dataset (optional).
FEEDBACK_HINTS_RAW = os.environ.get('NEXUS_STOP_HOOK_FEEDBACK_HINTS', '')
ITERATIVE_TURN_FILE = Path('/tmp/_nexus_stop_hook_iterative_count')
ITERATIVE_DONE_FILE = Path('/tmp/_nexus_stop_hook_iterative_done')

# Retry the proxy a few times on transient failures (network blip, 502, timeout).
# 1 initial + N retries; total attempts = 1 + RETRIES.
HOOK_RETRIES = int(os.environ.get('NEXUS_STOP_HOOK_RETRIES', '2'))
HOOK_RETRY_BASE_DELAY = float(os.environ.get('NEXUS_STOP_HOOK_RETRY_DELAY', '1.5'))

# Limits to keep the rolling state bounded.
MAX_HISTORY_TURNS = int(os.environ.get('NEXUS_STOP_HOOK_MAX_HISTORY', '20'))
MAX_CHARS_PER_TURN = int(os.environ.get('NEXUS_STOP_HOOK_MAX_CHARS_PER_TURN', '4000'))

# Optional: let the iterative_user proposer SEE the actual code the agent wrote on disk
# (instead of only the agent's closing text), so it can ground feedback in real defects.
# Default OFF — when unset, behavior is byte-for-byte identical to before. The code
# snapshot is injected ONLY into the (untracked) proxy request, never into the persisted
# rolling state, so the landed trajectory stays clean. iterative_user persona only.
SHOW_CODE = os.environ.get('NEXUS_STOP_HOOK_SHOW_CODE', 'false').lower() == 'true'
# 代码快照上限。~160k 字符 ≈ 40k token,远低于 opus 200k context(还要留给人设+滚动对话历史)。
CODE_BUDGET = int(os.environ.get('NEXUS_STOP_HOOK_CODE_BUDGET', '160000'))  # total chars cap
CODE_MAX_FILES = int(os.environ.get('NEXUS_STOP_HOOK_CODE_MAX_FILES', '200'))
CODE_WORKING_DIR = os.environ.get('FRONTEND_WORKING_DIR', '/workspace')
# Web frontend SOURCE files only — no JSON (config/lock noise), no SVG (often binary-ish/huge),
# no .py (frontend tasks are JS/TS/HTML). If a project genuinely needs JSON/SVG visibility,
# extend via env in future.
CODE_EXTS = ('.html', '.htm', '.css', '.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx', '.vue', '.svelte')
# Skip rules:
# - any directory whose name STARTS WITH "." (covers .git, .next, .cache, .turbo, .vite, .parcel-cache,
#   .nuxt, .svelte-kit, .astro, .vercel, .pytest_cache, .mypy_cache, .venv, .env, .idea, .vscode, .DS_*)
# - exact-match dirs for non-dot build/install artifacts
CODE_SKIP_DIR_NAMES = {
    # npm/yarn/pnpm install
    'node_modules',
    # build outputs (no dot prefix)
    'dist',
    'build',
    'out',
    'public/build',
    'static/build',
    # python venv variants without dot prefix
    'venv',
    'env',
    'ENV',
    'virtualenv',
    # python install / cache
    '__pycache__',
    # vendored deps
    'vendor',
    'vendored',
    # generic temp/output
    'tmp',
    'temp',
    'output',
    'outputs',
    'logs',
    # next/nuxt/svelte server outputs that some configs rename
    'coverage',
    'storybook-static',
}

# Transcript flush 同步窗口：Stop hook 触发时，触发它的 assistant turn 不一定已经写入
# transcript JSONL。等最多 TRANSCRIPT_FLUSH_WAIT 秒、每 POLL_INTERVAL 秒重读一次，
# 直到看到比 last_seen 更新的 assistant 行；超时仍未出现 → main 走 no_block 兜底，
# 比错误用上一轮内容做决策更安全。
TRANSCRIPT_FLUSH_WAIT = float(os.environ.get('NEXUS_STOP_HOOK_FLUSH_WAIT', '3.0'))
TRANSCRIPT_POLL_INTERVAL = float(os.environ.get('NEXUS_STOP_HOOK_POLL_INTERVAL', '0.05'))
# 等长度稳定多久才认为 flush 结束（避免过早返回 N-1）。
TRANSCRIPT_STABLE_WINDOW = float(os.environ.get('NEXUS_STOP_HOOK_STABLE_WINDOW', '0.4'))

# Sentinel the simulated user can emit to actually let Claude stop.
DONE_SENTINEL = 'STOP_HOOK_DONE'

# Local "obvious done" preflight: if the latest assistant turn clearly delivered a
# code artifact, skip the proxy and let Claude stop. This avoids two failure modes
# we kept hitting in production:
#   1) proxy returns FALLBACK_REPLY → assistant gets pushed for "one more round"
#      after it already finished, leading to confused "任务已完成" replies.
#   2) LLM judge mis-fires and says "keep going" even with explicit completion text.
# Pattern: <code-file-extension> AND <past-tense / completion phrasing> within the
# same turn. If both hit, we just stop.
_CODE_FILE_EXT_RE = re.compile(
    r'\.(?:html?|tsx?|jsx?|css|scss|py|go|rs|java|kt|swift|c|cc|cpp|h|hpp|rb|php|vue|svelte|json|ya?ml|sh|sql)\b',
    re.IGNORECASE,
)
_DONE_PHRASES_RE = re.compile(
    r'(已写到|已写好|写好了|写完了|已落盘|已生成|已 ?commit|已实现|已完成|落地|可直接.*打开|可以直接.*运行'
    r'|wrote to|written to|saved to|file created|implementation complete|all done|page is complete'
    r'|already (?:wrote|written|saved|created|committed|done|complete)'
    r'|file (?:has been|is) (?:created|saved|written)'
    # Common agent-flavored completion phrasings observed in the wild that nudger
    # would otherwise miss and waste a proxy call on:
    r"|i'?ve built"
    r'|both changes are in'
    r'|\bdone\s*[—\-.:,])',
    re.IGNORECASE,
)


def _looks_obviously_done(text: str) -> bool:
    """Return True if the assistant text clearly delivered a code artifact.

    Conservative: requires BOTH a code file extension and a completion phrase, to
    avoid stopping when assistant is mid-design or only produced spec.md.
    """
    if not text:
        return False
    return bool(_CODE_FILE_EXT_RE.search(text) and _DONE_PHRASES_RE.search(text))


SIMULATED_USER_SYSTEM = (
    '## Persona — you ARE the simulated user\n'
    'You are simulating the human user who originally asked for this task. You are here to '
    'unblock the assistant when it stalls or asks for confirmation. You are NOT a '
    'second AI assistant, NOT a reviewer, NOT a coach.\n\n'
    'NEVER reveal that you are an LLM, a hook, a simulation, or anything other '
    'than the human user. Do not mention being an "assistant", "AI", "model", '
    '"LLM", "hook", or anything similar. Do not say "as a user" or "speaking as '
    'the user" — just speak naturally in first person.\n\n'
    'NEVER help the assistant do its job (no architectural advice, no code '
    "snippets, no debugging tips, no design suggestions you weren't already "
    'asked about). Your job is JUST to acknowledge, decide, or unblock — like a '
    'busy human collaborator who only chimes in when needed.\n\n'
    '## Setup\n'
    'You are simulating a user collaborating with an AI coding assistant during a '
    'NON-INTERACTIVE batch run. Each user-role message in this conversation '
    "contains the assistant's most recent turn wrapped in <assistant>...</assistant>. "
    'Respond as the user so the assistant can keep working.\n\n'
    '## Hard rules\n'
    '- Reply in 1-3 short sentences, no preamble, no role tags, no markdown headers.\n'
    '- DO NOT introduce new requirements or change scope.\n'
    '- DO NOT ask the assistant new questions; you are unblocking, not collaborating.\n'
    '- DO NOT explain your reasoning or meta-commentary about your reply.\n'
    '- **DO NOT copy any wording from this prompt verbatim.** The phrases shown below '
    'are guidance about register and intent, NOT scripts. Always paraphrase in your '
    'own words and **reference at least one concrete detail** the assistant just '
    'mentioned (a section name, a file path, a chosen option, a stack item) so your '
    'reply sounds like the actual human who is reading the screen — never a generic '
    'cheerleader. Example anti-pattern (forbidden): replying "Looks good, continue '
    'with the rest of the design." with zero specifics — that\'s template-bot output, '
    'not a real user.\n\n'
    '## The deliverable is WORKING CODE\n'
    'The end goal of this task is RUNNING CODE — actual source/HTML files on disk.\n'
    'Specs, plans, design.md, design tokens, and architecture docs are intermediate '
    'artifacts only. They alone do NOT count as completing the task.\n\n'
    '## Decision tree (apply in order)\n\n'
    '1. **Assistant asked a yes/no or multiple-choice question?**\n'
    '   → Pick the recommended option (or a sensible default) and reply in one line, '
    'naming the option you picked and (if relevant) one reason tied to the task.\n\n'
    '2. **Assistant presented a design section / plan section / "what do you think?"**\n'
    '   → Approve in your own words. Reference the section the assistant just covered '
    '(e.g. the data model they sketched, the layout they proposed, the stack they '
    'chose) so the reply is anchored. If only 1-2 sections shown so far → push to '
    'continue with the next part. If most of the design is covered → push to wrap up '
    'and start implementing. Do NOT use generic praise like "looks good, continue '
    'with the rest" — that pattern is forbidden.\n\n'
    '3. **Assistant just finished writing a spec / design.md / plan and is asking to '
    'review or asking what to do next?**\n'
    '   → CRITICAL transition. Push to implementation immediately, in your own words. '
    'Mention the spec by name or path if the assistant referenced one, and tell them '
    'to write the actual source/HTML files now. Forbid extra planning rounds. Do NOT '
    'request revisions; do NOT invoke another planning skill.\n\n'
    '4. **Assistant is mid-implementation (writing code, running tools, fixing '
    'errors)?**\n'
    "   → Brief encouragement, paraphrased. Tie it to what they're doing right now "
    '(e.g. "the header section you\'re building", "that fetch error", "the grid '
    'layout") so it doesn\'t read as canned text.\n\n'
    '5. **Assistant says implementation is complete OR mentions writing/saving a '
    'code file?**\n'
    f'   → Emit {DONE_SENTINEL} immediately. Do NOT ask for verification, do NOT '
    'request a demo, do NOT push for "one more polish". Just exit.\n\n'
    '## When to emit STOP_HOOK_DONE — favor early exit\n\n'
    f'Reply with ONLY the literal token {DONE_SENTINEL} (no other text, no '
    'punctuation) **as soon as ANY of these is true** — you do not need to wait for '
    'multiple conditions:\n'
    '  - Assistant text contains any completion phrasing such as 「文件已写到」'
    '「写完了」「写好了」「完成」「已落盘」「已生成」「已 commit」「已实现」'
    '「可以直接在浏览器打开」 OR English equivalents like "wrote to", "written to", '
    '"saved to", "file created", "implementation complete", "all done", '
    '"page is complete", "committed".\n'
    '  - Assistant references a code-file path (e.g. `.html`, `.tsx`, `.py`, `.js`, '
    '`.css`, `.go`, `.rs`) AND uses past-tense / completion language anywhere near it.\n'
    '  - Assistant is asking for review / next instruction AND a code artifact was '
    'clearly produced earlier in the conversation.\n\n'
    'Bias HARD toward early exit. Once the assistant has delivered code, continuing '
    'to push them wastes time and often makes them over-edit, refactor unnecessarily, '
    'or wander off task.\n\n'
    f'Only keep the loop alive (i.e. NOT emit {DONE_SENTINEL}) when the latest '
    'assistant turn is clearly mid-design, mid-question, or mid-spec with NO code '
    'file written yet.\n\n'
    '## Anti-patterns to avoid in your reply\n'
    '- "Let me think about that..." — you are the user, not another assistant.\n'
    '- "What about edge case X?" — you are unblocking, not adding scope.\n'
    '- "Could you show me option B?" — pick option A and move on.\n'
    '- Long compliments before the actual instruction — get to the point.'
)


SIMULATED_USER_ITERATIVE = (
    '## Persona — you ARE the user who requested this task\n'
    'You are simulating the human user who originally asked for this task. '
    'You are a designer, developer, or product person who uses AI to build things. '
    'You have taste and care about quality. You are NOT an AI assistant, '
    'NOT a reviewer — just a real human using the product and giving feedback.\n\n'
    'NEVER reveal that you are an LLM, a hook, a simulation, or anything other '
    'than the human user. Just speak naturally in first person.\n\n'
    '## Setup\n'
    'The assistant just completed a round of work. Each user-role message in this '
    "conversation contains the assistant's most recent output wrapped in "
    '<assistant>...</assistant>. You are providing iterative feedback to refine '
    'the result — like a real collaborator working over multiple rounds.\n\n'
    '## Your feedback style\n'
    '- Reply in 1-3 short sentences. Be direct, casual, like chatting on Slack.\n'
    '- Each round, give 1-2 specific, actionable requests. Never vague.\n'
    '- Reference concrete things you can see (a specific section, element, component, chart).\n'
    '- Describe problems as what you OBSERVE ("this text is cut off", "the 3D model '
    'looks stretched", "the chart legend is covering the data", "clicking this does '
    'nothing") — NOT as code-level diagnoses.\n'
    '- Do NOT explain why (no "because users expect..."), just say what you want.\n'
    '- Do NOT give code snippets or technical implementation details.\n'
    '- Do NOT repeat feedback you already gave in earlier rounds.\n\n'
    '## Anchor every round to the original task (CRITICAL)\n'
    'The <task> block in the system message lists what the user actually asked for — '
    'specific sections, components, interactions, visual style, content. Before giving '
    'feedback each round, mentally check:\n'
    '- Which requirements from the task are visibly implemented in <assistant>?\n'
    '- Which are missing, wrong, or only half-done?\n'
    '- Pick feedback that drives the assistant TOWARD what the task asked for, not '
    'wherever your imagination wanders.\n\n'
    'Do NOT invent new requirements that the task did not ask for. If the task did not '
    'mention sharing/auth/dark-mode/leaderboard, do not ask for those. Polish and '
    'refinement feedback (colors, spacing, motion) is always fine — but new FEATURES '
    'must trace back to something the task explicitly or implicitly required.\n\n'
    'Prefer "missing requirement" feedback over "polish" feedback while parts of the '
    'task are still unimplemented. Only shift to polish once the task is fully covered.\n\n'
    '## Types of feedback (adapt to the task type)\n'
    'You naturally tend to notice broken/ugly things first, then missing functionality, '
    'then polish — but this is a tendency, not a strict rule. Mix as feels natural.\n\n'
    '**Broken or obviously wrong** (you notice these first):\n'
    '- Elements overlapping, content clipped/cut off, things overflowing their container\n'
    '- Sizes clearly wrong (too big, too small, squished, stretched)\n'
    '- Things not rendering (blank areas, missing visuals, broken images/models)\n'
    '- Colors making content invisible, contrast too low to read\n'
    '- Proportions off, aspect ratios distorted, spacing collapsed\n\n'
    '**Missing functionality** (once it looks okay, you start interacting):\n'
    '- Clicking/hovering does nothing when it obviously should\n'
    '- Expected interactions not implemented (drag, rotate, zoom, filter, sort, submit)\n'
    '- Missing states: no loading indicator, no error message, no empty state\n'
    '- Incomplete content: placeholders still showing, sections missing\n'
    '- Navigation/flow gaps: nowhere to go next, no way back\n\n'
    '**Polish & enhancement** (when it works, you want it to feel better):\n'
    '- Visual refinement: colors, spacing, shadows, typography, alignment\n'
    '- Motion & transitions: smoother animations, better timing, entrance effects\n'
    '- Additional features: add controls, toggles, extra modes\n'
    '- Style/mood adjustments: "make it feel more [adjective]"\n'
    '- Alternative approaches: "try [different layout/visualization/style]"\n\n'
    '## Task-type awareness\n'
    'Adapt your feedback to whatever the task actually is. Examples:\n'
    '- Website/app → layout, navigation, responsiveness, interactions\n'
    '- 3D/WebGL → model appearance, lighting, camera angle, performance, controls\n'
    '- SVG/illustration → proportions, colors, line weight, composition, animation\n'
    '- Data visualization → readability, legend clarity, axis labels, interactivity\n'
    '- Game/interactive → controls feel, feedback, difficulty, visual clarity\n'
    'Read the original task to understand what type it is, then give feedback '
    'appropriate to that domain.\n\n'
    '## Language rule (CRITICAL)\n'
    'You MUST reply in the SAME LANGUAGE as the original task prompt (the <task> block). '
    'Detect the language of the NATURAL LANGUAGE instructions (ignore code snippets, '
    'technical terms like Three.js/GSAP/CSS, and English variable names). '
    'If the instructions are written in Chinese (even with English tech terms mixed in), '
    'you MUST reply in Chinese. If English, reply in English. NEVER mix.\n\n'
    '## When to emit STOP_HOOK_DONE\n'
    f'Reply with ONLY the literal token {DONE_SENTINEL} (nothing else) when:\n'
    '- You are told it is your final round via the <round_info> tag, OR\n'
    '- You genuinely feel the result is solid and you have no more meaningful feedback.\n'
    'It is OK to stop early if the output is already good. Do NOT '
    'invent artificial complaints just to fill rounds.\n\n'
    '## Sound like a real person, not an AI assistant (IMPORTANT)\n'
    'Real people do NOT compliment before every request. Avoid the robotic '
    '"praise + one nitpick" template at all costs. In particular:\n'
    '- Do NOT open with "Looks good / Looks solid / Nice / Great / This is clean, '
    'one thing—" or any variation. It reads as fake AI politeness. Just say the thing.\n'
    '- Do NOT use the connective "One thing—" / "Just one thing" / "A couple things—" '
    'as a recurring crutch.\n'
    '- In LATER rounds, do NOT acknowledge the previous fix before your next request. '
    'BANNED openers include: "Works well", "Works now", "That works", "Looks good now", '
    '"Feels good now", "Nice, the ... now", "Perfect, now", "The X works nicely". '
    'Just go straight to the new problem. A real user does not thank the tool every turn — '
    'they just say the next thing that bugs them.\n'
    '- VARY your tone and structure across rounds — a real user is sometimes blunt '
    '("the ball clips through the wall"), sometimes asks a question ("why does it reset '
    'when I click?"), sometimes a bit impatient ("still overflowing on mobile"), '
    'sometimes just names one concrete fix. Different opener every round.\n'
    '- It is fine — usually better — to lead straight with the problem, zero preamble.\n'
    '- Occasionally a single terse sentence is the most realistic reply.\n\n'
    'GOOD examples (varied, no acknowledgment): '
    '"The legend overlaps the top bar." / "Why does pressing space pause instead of '
    'restart?" / "Mobile still breaks — the grid spills off the right edge." / '
    '"Make the active tab actually look selected." \n'
    'BAD examples (do NOT write like these): "Looks great! One thing—..." / '
    '"Works well now. Could you also..." / "Nice, the transition feels smooth. Next..."\n\n'
    '## Anti-patterns\n'
    '- "Looks great!" with no actionable request — forbidden unless you truly have no feedback left.\n'
    '- Technical jargon ("refactor the component tree", "use flexbox", "set overflow-x") '
    '— describe what you SEE, not what to code.\n'
    '- Multiple unrelated requests in one message — keep it focused, 1-2 related items max.\n'
    '- Repeating the same feedback from a previous round.\n'
    '- Saying "as a user" or "from a UX perspective" — just state what you want directly.\n'
    '- Filler like "let me load it up first" or "let me see it running" — '
    'you have ALREADY seen the output (it is in <assistant>). Jump straight to feedback.'
)


def _parse_feedback_hints() -> list[str]:
    """Parse NEXUS_STOP_HOOK_FEEDBACK_HINTS JSON env var into a list of hint strings."""
    if not FEEDBACK_HINTS_RAW:
        return []
    try:
        hints = json.loads(FEEDBACK_HINTS_RAW)
        if isinstance(hints, list):
            return [str(h) for h in hints if h]
    except Exception:
        pass
    return []


def _get_iterative_turn_count() -> int:
    """Read the iterative turn counter (post-implementation rounds completed)."""
    try:
        return int(ITERATIVE_TURN_FILE.read_text())
    except Exception:
        return 0


def _increment_iterative_turn() -> int:
    """Increment and persist the iterative turn counter. Returns new value."""
    current = _get_iterative_turn_count()
    new_val = current + 1
    with contextlib.suppress(Exception):
        ITERATIVE_TURN_FILE.write_text(str(new_val))
    return new_val


def _emit_no_block() -> None:
    print(json.dumps({}))
    sys.exit(0)


def _emit_block(reply: str) -> None:
    print(json.dumps({'decision': 'block', 'reason': reply}))
    sys.exit(0)


def _read_original_user_prompt(transcript_path: str) -> str:
    """Return the first user-role text in the transcript — the actual task brief.

    Without this, the simulated user has no idea what the original task was and may
    drift onto whatever example the skill pulls in (we hit this bug: a brainstorming
    demo in the skill prompt got echoed back as 'change to a users table' and Claude
    happily produced an off-topic page).

    Strategy: take the first user message that's NOT a system-reminder / hook
    injection, since the skill harness wraps the actual prompt in a standalone block.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return ''
    candidates: list[str] = []
    try:
        with open(transcript_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                msg = evt.get('message') or evt
                if msg.get('role') != 'user':
                    continue
                content = msg.get('content', '')
                texts: list[str] = []
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get('type') == 'text':
                            t = b.get('text', '')
                            if t:
                                texts.append(t)
                elif isinstance(content, str):
                    texts.append(content)
                for t in texts:
                    s = t.strip()
                    if not s:
                        continue
                    if s.startswith('<system-reminder>') or 'PreToolUse:' in s[:200]:
                        continue
                    if 'Stop hook feedback' in s[:50] or s.startswith('User has answered'):
                        continue
                    candidates.append(s)
                if candidates:
                    # First "real" user message is the task brief.
                    break
    except Exception:
        return ''
    if not candidates:
        return ''
    raw = candidates[0]
    # Strip the standard skill harness wrappers if present.
    for marker in ('<command-args>', 'ARGUMENTS:'):
        idx = raw.find(marker)
        if idx >= 0:
            raw = raw[idx + len(marker) :]
    raw = raw.strip()
    # Frontend prompts can run 3-5k chars (中文设计需求 + visual spec + interaction list).
    # Truncating too aggressively drops concrete requirements that iterative_user mode
    # needs to anchor feedback. 8k is generous but still bounds outlier prompts.
    if len(raw) > 8000:
        raw = raw[:8000] + '...[truncated]'
    return raw


def _read_all_assistant_texts(transcript_path: str) -> list[str]:
    """Walk the JSONL transcript, return each assistant turn's collapsed text in order.

    Empty turns (no text blocks) are skipped to avoid noise from pure-tool-use turns.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return []
    out: list[str] = []
    try:
        with open(transcript_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                msg = evt.get('message') or evt
                if msg.get('role') != 'assistant':
                    continue
                content = msg.get('content', '')
                text = ''
                if isinstance(content, list):
                    parts = [b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text']
                    text = '\n'.join(p for p in parts if p)
                elif isinstance(content, str):
                    text = content
                if text.strip():
                    out.append(text)
    except Exception:
        return out
    return out


def _wait_for_new_assistant(
    transcript_path: str,
    last_seen_idx: int,
    timeout: float = TRANSCRIPT_FLUSH_WAIT,
    poll_interval: float = TRANSCRIPT_POLL_INTERVAL,
    stable_window: float = TRANSCRIPT_STABLE_WINDOW,
) -> list[str]:
    """轮询 transcript，等到 assistant 行数 **稳定** 一段窗口才返回。

    背景：仅等"出现 idx > last_seen + 1"会过早退出 —— 触发当前 Stop 的可能是 N，
    而 transcript 此时只 flush 到 N-1；一旦看到 N-1 就立即返回，会让 reply 错位
    一整轮（实测 4/5 错位）。改成等"长度连续 STABLE_WINDOW 秒不变"再返回，可以
    较稳地命中触发轮，同时仍受 timeout 上限保护。

    返回 best-effort 的 all_texts；超时则返回当下读到的（main 中再做 no_block 兜底）。
    """
    deadline = time.time() + timeout
    all_texts = _read_all_assistant_texts(transcript_path)
    last_len = len(all_texts)
    last_change = time.time()
    while time.time() < deadline:
        time.sleep(poll_interval)
        all_texts = _read_all_assistant_texts(transcript_path)
        if len(all_texts) != last_len:
            last_len = len(all_texts)
            last_change = time.time()
            continue
        # 已经看到比 last_seen 更新的，且长度稳定窗口内没再涨 → flush 大概率完成
        if last_len > last_seen_idx + 1 and (time.time() - last_change) >= stable_window:
            break
        # 还没出现新轮 —— 继续等
    return all_texts


def _load_state() -> _StopHookState:
    """Load the running simulated-user conversation state from disk."""
    if not STATE_FILE.exists():
        return {'messages': [], 'last_seen_assistant_idx': -1, 'count': 0}
    try:
        st = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        # Basic shape guard.
        if not isinstance(st, dict):
            raise ValueError('bad state shape')
        messages = st.get('messages')
        last_seen_assistant_idx = st.get('last_seen_assistant_idx', -1)
        count = st.get('count', 0)
        if (
            not isinstance(messages, list)
            or not all(isinstance(message, dict) for message in messages)
            or not isinstance(last_seen_assistant_idx, int)
            or not isinstance(count, int)
        ):
            raise ValueError('bad state shape')
        return {
            'messages': messages,
            'last_seen_assistant_idx': last_seen_assistant_idx,
            'count': count,
        }
    except Exception:
        return {'messages': [], 'last_seen_assistant_idx': -1, 'count': 0}


def _save_state(state: _StopHookState) -> None:
    with contextlib.suppress(Exception):
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f'{head}\n...[truncated {len(text) - max_chars} chars]...\n{tail}'


def _code_priority(rel: str, size: int) -> tuple:
    """Sort key for assembling the code snapshot. Lower tuple sorts first.

    Order: index.html → other .html → main/app/index entry scripts → everything else
    by descending size (larger files are usually the core artifact).
    """
    name = os.path.basename(rel).lower()
    ext = os.path.splitext(name)[1]
    if name == 'index.html':
        tier = 0
    elif ext == '.html':
        tier = 1
    elif os.path.splitext(name)[0] in ('main', 'app', 'index', 'script', 'sketch') and ext in (
        '.js',
        '.jsx',
        '.ts',
        '.tsx',
        '.vue',
    ):
        tier = 2
    else:
        tier = 3
    return (tier, -size)


def _read_workspace_code(working_dir: str, budget: int, max_files: int) -> str:
    """Read code files the agent wrote on disk, assembled into a snapshot for the proposer.

    Best-effort and fully defensive: any error returns '' so the caller silently falls
    back to the original (code-blind) behavior. Used ONLY when NEXUS_STOP_HOOK_SHOW_CODE
    is on; the result is injected into the (untracked) proxy request, never persisted.

    Priority: index.html > other html > entry scripts > rest by size desc.
    Total length is capped at `budget` chars; an over-budget file is truncated with a
    marker; files that don't fit at all are listed by name so the proposer knows they
    exist but weren't shown.
    """
    try:
        if not working_dir or not os.path.isdir(working_dir):
            return ''
        candidates: list[tuple[str, int]] = []
        for root, dirs, files in os.walk(working_dir):
            # Skip:
            # 1. any dir whose name starts with "." (.git, .next, .cache, .venv, .DS_*, ...)
            # 2. any dir name in the explicit skip set (node_modules, dist, build, ...)
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in CODE_SKIP_DIR_NAMES]
            for fn in files:
                if fn.startswith('.'):
                    # Skip dotfiles (.DS_Store, .gitignore, .eslintrc.*, etc.) — never source code
                    continue
                if not fn.endswith(CODE_EXTS):
                    continue
                full = os.path.join(root, fn)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                if size <= 0:
                    continue
                rel = os.path.relpath(full, working_dir)
                candidates.append((rel, size))
                # Hard cap: > max_files candidates → bail out entirely (rather than
                # arbitrarily truncating; oversized projects are usually misconfigured
                # and including a partial slice would mislead the proposer).
                if len(candidates) > max_files:
                    return ''
        if not candidates:
            return ''
        candidates.sort(key=lambda x: _code_priority(x[0], x[1]))

        parts: list[str] = []
        used = 0
        not_shown: list[str] = []
        shown = 0
        for rel, _size in candidates:
            if shown >= max_files or used >= budget:
                not_shown.append(rel)
                continue
            full = os.path.join(working_dir, rel)
            try:
                with open(full, encoding='utf-8', errors='replace') as f:
                    code = f.read()
            except OSError:
                continue
            remaining = budget - used
            header = f'### File: {rel}'
            if len(code) > remaining:
                omitted = len(code) - remaining
                code = code[:remaining] + f'\n...[truncated {omitted} more chars]'
            block = f'{header}\n```\n{code}\n```'
            parts.append(block)
            used += len(block)
            shown += 1
        if not_shown:
            parts.append('### Remaining files (not shown): ' + ', '.join(not_shown))
        return '\n\n'.join(parts)
    except Exception:
        return ''


def _trim_history(messages: list[dict]) -> list[dict]:
    """Cap the rolling history to MAX_HISTORY_TURNS user/assistant pairs (~ 2x messages)."""
    cap = MAX_HISTORY_TURNS * 2
    if len(messages) <= cap:
        return messages
    # Keep the most recent cap messages, but always start on a user-role message
    # so the proxy gets a well-formed conversation.
    trimmed = messages[-cap:]
    while trimmed and trimmed[0].get('role') != 'user':
        trimmed.pop(0)
    return trimmed


def _ask_proxy_for_user_reply(
    messages: list[dict],
    original_task: str = '',
    system_prompt: str | None = None,
    code_snapshot: str = '',
) -> str:
    """Send the rolling conversation to the in-sandbox Anthropic proxy and return its text reply.

    `code_snapshot` (when non-empty) is injected as a third system block. Putting it in
    `system` rather than `messages` keeps the rolling messages list byte-identical across
    rounds — critical for chat_tracker prefix-sharing (otherwise SHOW_CODE makes the
    trajectory fork into N copies the moment workspace transitions from empty to populated).
    """
    base_url = os.environ.get('ANTHROPIC_BASE_URL', '').rstrip('/')
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not base_url:
        return ''

    # iterative_user passes a fully-built system_prompt (already includes the
    # language directive via _build_iterative_system_prompt). The nudger
    # fallback path uses SIMULATED_USER_SYSTEM and must append the directive
    # here so its replies also match the task language.
    if system_prompt is not None:
        effective_system = system_prompt
    else:
        effective_system = SIMULATED_USER_SYSTEM + _build_language_directive(original_task)

    system_blocks: list[dict] = [
        {
            'type': 'text',
            'text': effective_system,
            'cache_control': {'type': 'ephemeral'},
        },
    ]
    if original_task:
        system_blocks.append({
            'type': 'text',
            'text': (
                '## Original task brief\n\n'
                'This is the task the assistant is REALLY working on. Anchor every '
                'reply and your STOP_HOOK_DONE judgment to this brief — ignore any '
                'unrelated examples or demos the skill harness may have shown:\n\n'
                f'<task>\n{original_task}\n</task>'
            ),
            'cache_control': {'type': 'ephemeral'},
        })
    if code_snapshot:
        # No cache_control on this block — content changes every round (agent edits files),
        # so caching it is wasted writes. The two cached blocks above (persona + task)
        # still hit per-sandbox.
        system_blocks.append({
            'type': 'text',
            'text': (
                '## Current code on disk\n\n'
                'Below is the ACTUAL files the assistant has written so far in the '
                'workspace (some may be truncated, some omitted under "Remaining files"). '
                'Use it to ground feedback in real defects — missing requirements, bugs, '
                'fake/stub implementations — but stay in the voice of a real user, not a '
                'code reviewer (no line numbers, no patches, no code snippets).\n\n'
                f'<current_code>\n{code_snapshot}\n</current_code>'
            ),
        })

    payload = {
        'model': HOOK_MODEL,
        'max_tokens': HOOK_MAX_TOKENS,
        'system': system_blocks,
        'messages': messages,
    }

    req_data = json.dumps(payload).encode('utf-8')
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }

    # Retry transient failures (5xx, timeouts, connection resets) with exponential
    # backoff. 4xx errors are NOT retried — those are payload bugs that won't fix
    # themselves. The fallback in main() handles total failure.
    last_exc: Exception | None = None
    for attempt in range(HOOK_RETRIES + 1):
        try:
            req = urllib.request.Request(
                f'{base_url}/v1/messages',
                method='POST',
                headers=headers,
                data=req_data,
            )
            with urllib.request.urlopen(req, timeout=HOOK_TIMEOUT) as resp:
                body = json.loads(resp.read())
            blocks = body.get('content', [])
            return '\n'.join(
                b.get('text', '') for b in blocks if isinstance(b, dict) and b.get('type') == 'text'
            ).strip()
        except urllib.error.HTTPError as e:
            last_exc = e
            # 4xx → no point retrying.
            if 400 <= e.code < 500:
                raise
            sys.stderr.write(f'stop_hook: HTTPError {e.code} on attempt {attempt + 1}/{HOOK_RETRIES + 1}\n')
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last_exc = e
            sys.stderr.write(f'stop_hook: {type(e).__name__} on attempt {attempt + 1}/{HOOK_RETRIES + 1}: {e}\n')
        if attempt < HOOK_RETRIES:
            time.sleep(HOOK_RETRY_BASE_DELAY * (2**attempt))
    assert last_exc is not None
    raise last_exc


def _detect_prompt_language(text: str) -> str:
    """Detect whether the prompt's natural language is Chinese or English."""
    if not text:
        return 'en'
    cjk_count = sum(1 for c in text if '一' <= c <= '鿿')
    return 'zh' if cjk_count > 20 else 'en'


def _build_language_directive(original_task: str) -> str:
    """Return a `<language_directive>` block when the task is Chinese, else ''.

    Shared by nudger (SIMULATED_USER_SYSTEM) and iterative_user
    (SIMULATED_USER_ITERATIVE) so simulated-user replies match the language
    of the original task. STOP_HOOK_DONE stays literal English on purpose —
    translating it would break main()'s done-detection.
    """
    if _detect_prompt_language(original_task) != 'zh':
        return ''
    return (
        '\n\n<language_directive>\n'
        '你必须用中文回复。即使原始任务中有英文技术术语，你的反馈也必须用中文。\n'
        f'例外：当你决定结束时，仍然只输出英文字面量 {DONE_SENTINEL}'
        '（不要翻译，不要加任何中文）。\n'
        '</language_directive>'
    )


def _build_iterative_system_prompt(original_task: str = '') -> str:
    """Build the STATIC iterative system prompt.

    Only includes content that is constant within a single sandbox (the persona +
    language directive). Per-round dynamic context (round_info, feedback hints)
    is injected per-turn into the latest user message instead — see
    _build_per_turn_prefix(). This keeps the system block byte-identical across
    rounds, so:
      - Bedrock prompt cache hits on the system prefix every round
      - chat_tracker doesn't fork into N parallel subtrees per round (its node
        equality is content-based and was creating one tree branch per varying
        round_info string, producing trajectories with duplicated prefixes)
    """
    base = SIMULATED_USER_ITERATIVE

    # Language directive depends only on the task, not on the round, so it stays
    # in the static system prompt. Shared with the nudger persona via the
    # _build_language_directive helper.
    base += _build_language_directive(original_task)

    # When SHOW_CODE is on, a separate "## Current code on disk" system block is added
    # by _ask_proxy_for_user_reply (carrying the snapshot itself). Here we just teach the
    # proposer HOW to use it. SHOW_CODE is constant within a sandbox, so adding this
    # keeps THIS system block byte-stable across rounds (cache still hits on this block).
    if SHOW_CODE:
        base += (
            '\n\n<code_visibility>\n'
            'A separate "## Current code on disk" system block carries a <current_code> '
            'snapshot of the ACTUAL files the assistant has written so far. It is a PARTIAL '
            'view — some files may be truncated, some omitted (listed under "Remaining files"). '
            'Files are labeled by their path relative to the project root.\n'
            'Use it to ground your feedback in what the code REALLY does:\n'
            '- Notice requirements the task asked for that are missing or only stubbed in the code.\n'
            '- Notice things the assistant claimed to do but the code does not actually implement, '
            'or implements incorrectly (wrong formula, broken logic, hardcoded fake values).\n'
            '- You may name a specific file (by its relative path) when it helps, e.g. "the physics in '
            '`src/sim.js` doesn\'t seem to conserve energy".\n'
            'BUT stay a real user giving feedback, NOT a code reviewer: describe what is wrong or '
            'missing and what you want, in plain language. Do NOT quote line numbers, do NOT give '
            'code snippets or patches, do NOT explain how to fix it in code.\n'
            '</code_visibility>'
        )

    return base


def _build_per_turn_prefix(iterative_turn: int, is_final_round: bool) -> str:
    """Build the per-turn dynamic prefix injected into the latest user message.

    Contains <round_info> and (optionally) <feedback_direction> — these vary
    every round and would otherwise prevent system-prompt caching. Injecting
    them here instead keeps system bytes stable.

    The returned string ends with a blank line so the original `<assistant>...`
    content remains visually separated.
    """
    parts = [f'<round_info>\nThis is feedback round {iterative_turn + 1}.']
    if is_final_round:
        # 接近安全天花板,温和收尾(不让模型把它当倒计时目标)。
        parts.append(
            'You have gone several rounds already. Unless something is clearly broken or '
            f'a core task requirement is still missing, wrap up now: emit {DONE_SENTINEL}.'
        )
    else:
        parts.append(
            'YOU decide whether to continue. If there is still a real problem or an '
            'unmet task requirement, give your next concrete piece of feedback. '
            f'If the result genuinely satisfies you, emit {DONE_SENTINEL} and stop — '
            'do NOT invent complaints just to keep going. A real user stops when happy.'
        )
    parts.append('</round_info>')

    # 柔性阶段倾向(模拟真人 workflow:先确保能用→再求好用,但不硬禁跨阶段)。
    parts.append('')
    if iterative_turn <= 1:
        parts.append(
            '<focus_this_round>\nEarly on, like a real user you are mostly checking whether '
            'it actually WORKS and whether what you asked for is even there. So you LEAN '
            'toward: things that are broken/wrong, and task requirements that are missing or '
            'half-done. (If those already look fine, it is totally natural to move on to '
            'improvements instead — do not force a bug.)\n</focus_this_round>'
        )
    else:
        parts.append(
            '<focus_this_round>\nThe basics should be working by now, so like a real user you '
            'LEAN toward making it BETTER: usability, polish, nicer interactions/motion, '
            'sensible enhancements within what the task is about. (But if you just spotted a '
            'genuine bug or a still-missing requirement, raise that first — correctness wins.)\n'
            '</focus_this_round>'
        )

    hints = _parse_feedback_hints()
    if hints:
        hint_idx = iterative_turn % len(hints)
        parts.append('')  # blank line
        parts.append('<feedback_direction>')
        parts.append("Consider this direction for your feedback (rephrase naturally, don't copy verbatim):")
        parts.append(hints[hint_idx])
        parts.append('</feedback_direction>')

    parts.append('')  # blank line before the actual <assistant>...</assistant>
    return '\n'.join(parts) + '\n'


def main() -> None:
    # Opt-in：默认 no-op。只有显式 NEXUS_STOP_HOOK_ENABLE=true 才真正运行。
    if os.environ.get('NEXUS_STOP_HOOK_ENABLE', 'false').lower() != 'true':
        _emit_no_block()

    is_iterative = HOOK_PERSONA == 'iterative_user'

    # If iterative mode already completed (STOP_HOOK_DONE was emitted), no-block immediately.
    # This prevents the trailing loop where agent keeps outputting farewells.
    if is_iterative and ITERATIVE_DONE_FILE.exists():
        sys.stderr.write('stop_hook [iterative]: already done (done file exists), exiting no-block\n')
        _emit_no_block()

    # Loop guard via a separate counter file so legacy callers and ops can `rm` it.
    try:
        count = int(COUNTER_FILE.read_text())
    except Exception:
        count = 0
    if count >= MAX_AUTO_REPLIES:
        _emit_no_block()

    # Parse hook input.
    try:
        data = json.load(sys.stdin)
    except Exception:
        _emit_no_block()

    # 调试期可设 NEXUS_STOP_HOOK_DUMP_PAYLOAD=/workspace/_nexus_stop_hook_payload.json，
    # 落一份 stdin 原文用于核实 Claude Code 升级是否新增了 message 字段。
    dump_path = os.environ.get('NEXUS_STOP_HOOK_DUMP_PAYLOAD', '')
    if dump_path:
        with contextlib.suppress(Exception):
            Path(dump_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    transcript_path = data.get('transcript_path') or data.get('transcriptPath') or ''

    # Source A（首选）：Claude Code 在 stdin 直接给 last_assistant_message —— 一定是
    # 触发本次 Stop 的那一条，不存在 transcript flush race。Source B（兜底）：旧路径
    # 兼容老版本/没这个字段的环境，仍走 transcript 轮询 + stable-window。
    last_msg = (data.get('last_assistant_message') or '').strip()
    new_turns: list[str] = []
    state: _StopHookState = _load_state()

    if last_msg:
        new_turns = [last_msg]
    else:
        # Fallback：等 transcript flush 稳定再读
        last_seen = state.get('last_seen_assistant_idx', -1)
        all_texts = _wait_for_new_assistant(transcript_path, last_seen)
        if not all_texts:
            _emit_no_block()
        if last_seen >= len(all_texts):
            state = {'messages': [], 'last_seen_assistant_idx': -1, 'count': 0}
            last_seen = -1
        new_turns = all_texts[last_seen + 1 :]
        if not new_turns:
            sys.stderr.write('stop_hook: transcript flush timed out, exiting no-block\n')
            _emit_no_block()
        state['last_seen_assistant_idx'] = len(all_texts) - 1

    # Read once per hook call; cheap (small file scan) and gives the LLM a stable anchor.
    original_task = _read_original_user_prompt(transcript_path)

    # --- Persona-specific preflight and system prompt selection ---
    if is_iterative:
        # No latch: every hook call in iterative mode advances the counter and uses
        # the iterative prompt. The previous latch design relied on _looks_obviously_done
        # to detect "code delivered" transition, but agents phrase delivery in too many
        # ways for a regex to catch reliably — misses froze the counter at 0 and the
        # iterative prompt's "feel free to STOP if it's solid" rule made the LLM bail
        # on round 1 (observed in mt.jsonl traj 5).
        # The iterative prompt itself handles the brainstorm/spec/delivery phases via
        # the "Anchor every round to the task" directive — when agent hasn't shipped
        # code yet, the LLM pushes toward implementation; when shipped, it gives design
        # feedback. ITERATIVE_TURNS becomes a true upper bound on hook rounds.
        iterative_turn = _get_iterative_turn_count()
        if iterative_turn >= ITERATIVE_TURNS:
            sys.stderr.write(f'stop_hook [iterative]: reached max ({ITERATIVE_TURNS}), exiting\n')
            with contextlib.suppress(Exception):
                ITERATIVE_DONE_FILE.write_text('1')
            _emit_no_block()

        is_final_round = iterative_turn == ITERATIVE_TURNS - 1
        system_prompt = _build_iterative_system_prompt(original_task)
        per_turn_prefix = _build_per_turn_prefix(iterative_turn, is_final_round)
    else:
        # Nudger mode: original behavior with preflight.
        if _looks_obviously_done(new_turns[-1]):
            sys.stderr.write('stop_hook: obvious-done preflight matched, exiting no-block\n')
            _emit_no_block()
        system_prompt = None  # will use default SIMULATED_USER_SYSTEM
        per_turn_prefix = ''

    # Dedup：duplicate Stop 事件可能让同一条 assistant 文本被 append 两次，
    # 比对滚动 messages 末尾的 <assistant>...</assistant> 部分，相同则跳过。
    # 注意：messages[-1] 可能在上一轮被注入了 per_turn_prefix（round_info），
    # 所以这里用 endswith 而不是 == 比较，避免被 prefix 干扰。
    def _wrap(text: str) -> str:
        return f'<assistant>\n{_truncate(text, MAX_CHARS_PER_TURN)}\n</assistant>'

    for text in new_turns:
        wrapped = _wrap(text)
        last = state['messages'][-1] if state['messages'] else None
        if last and last.get('role') == 'user':
            last_content = last.get('content', '')
            if isinstance(last_content, str) and last_content.endswith(wrapped):
                continue
        state['messages'].append({'role': 'user', 'content': wrapped})

    # Inject the per-turn prefix (round_info / feedback_direction) DIRECTLY into the
    # latest user message of state['messages']. This makes state's messages identical
    # to what the proxy sees, which matters because chat_tracker compares message
    # content for prefix-sharing — if state has clean messages and we send a prefixed
    # copy to the proxy, every round forks a new tree branch (one per round_info
    # variant), producing N trajectories with duplicated prefixes per N hook calls.
    # The prefix becomes part of the persistent rolling conversation, so all rounds
    # share the same root-to-leaf path in chat_tracker's tree.
    if per_turn_prefix and state['messages'] and state['messages'][-1].get('role') == 'user':
        state['messages'][-1] = {
            **state['messages'][-1],
            'content': per_turn_prefix + state['messages'][-1]['content'],
        }

    # Bound the running history.
    state['messages'] = _trim_history(state['messages'])

    # Ask the proxy.
    # Optionally let the iterative proposer SEE the real code on disk: read a snapshot of
    # the workspace and pass it to _ask_proxy_for_user_reply, which injects it as a
    # SYSTEM block (not into messages). Putting it in system avoids forking the
    # chat_tracker tree — messages stays byte-identical to what we persisted in
    # state['messages'], so all hook rounds share one tree path. (Earlier attempts
    # injected snapshot into messages[-1] directly, which forked the trajectory the
    # moment workspace transitioned from empty to populated — observed in the smoke
    # test as 5 sim_user trajectories instead of 1.)
    code_snapshot = ''
    if is_iterative and SHOW_CODE:
        code_snapshot = _read_workspace_code(CODE_WORKING_DIR, CODE_BUDGET, CODE_MAX_FILES)
    try:
        reply = _ask_proxy_for_user_reply(
            state['messages'],
            original_task=original_task,
            system_prompt=system_prompt,
            code_snapshot=code_snapshot,
        )
    except Exception as e:
        sys.stderr.write(f'stop_hook: proxy call failed, exiting no-block: {e}\n')
        _save_state(state)
        _emit_no_block()
    if not reply:
        sys.stderr.write('stop_hook: empty reply from proxy, exiting no-block\n')
        _save_state(state)
        _emit_no_block()

    # Record the proxy's reply as an assistant turn for next round's prefix continuity.
    state['messages'].append({'role': 'assistant', 'content': reply})
    state['messages'] = _trim_history(state['messages'])

    if DONE_SENTINEL in reply:
        # Trust the LLM. In iterative mode it sees the original task, the rolling
        # conversation, and a "Anchor every round to the task" directive — if it says
        # done, it's done. Forcing extra rounds only manufactures fake feedback that
        # contaminates training data.
        if is_iterative:
            with contextlib.suppress(Exception):
                ITERATIVE_DONE_FILE.write_text('1')
        _save_state(state)
        _emit_no_block()

    # Increment counters.
    state['count'] = count + 1
    _save_state(state)
    with contextlib.suppress(Exception):
        COUNTER_FILE.write_text(str(count + 1))

    # In iterative mode every block(reply) burns one round, no matter the phase.
    # ITERATIVE_TURNS is the hard upper bound on hook calls, not on "post-delivery rounds".
    if is_iterative:
        _increment_iterative_turn()

    _emit_block(reply)


if __name__ == '__main__':
    main()
