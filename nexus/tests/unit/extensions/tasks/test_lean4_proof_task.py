from nexus.extensions.tasks.lean4_proof import task as lean4_task


def test_extract_lean_code_pairs_last_close_with_nearest_open() -> None:
    text = '''
We should put the answer inside <final_proof>.

```lean4
theorem sample : True := by
  trivial
```

<final_proof>
theorem sample : True := by
  exact True.intro
</final_proof>
'''

    assert lean4_task._extract_lean_code(text) == 'theorem sample : True := by\n  exact True.intro'


def test_extract_lean_code_strips_fence_nested_inside_final_proof() -> None:
    text = '''
<final_proof>
```lean4
theorem sample : True := by
  trivial
```
</final_proof>
'''

    assert lean4_task._extract_lean_code(text) == 'theorem sample : True := by\n  trivial'


def test_extract_lean_code_strips_bare_and_indented_fences() -> None:
    """The fence is decoration, so any info string / indentation is stripped."""
    for opener in ['```', '```lean', '```Lean4', '```lean4 ']:
        text = f'<final_proof>\n{opener}\nimport Mathlib\n  ```\n</final_proof>'
        assert lean4_task._extract_lean_code(text) == 'import Mathlib', opener


def test_extract_lean_code_keeps_inline_backtick_span() -> None:
    """A ``code`` span is not a fence and must not be eaten as one."""
    text = '<final_proof>\n```mathlib``` is great\nimport Mathlib\n</final_proof>'

    assert lean4_task._extract_lean_code(text) == '```mathlib``` is great\nimport Mathlib'


def test_extract_lean_code_rejects_fence_without_final_proof_tags() -> None:
    """A fenced block is not an answer channel: only <final_proof> is."""
    text = '''
Here is my attempt.

```lean4
theorem sample : True := by
  trivial
```
'''

    assert lean4_task._extract_lean_code(text) == ''


def test_extract_lean_code_rejects_unclosed_final_proof_tag() -> None:
    text = '''
Mention <final_proof> without closing it.

```lean4
theorem sample : True := by
  trivial
```
'''

    assert lean4_task._extract_lean_code(text) == ''


def test_extract_lean_code_rejects_bare_prose() -> None:
    assert lean4_task._extract_lean_code('I think we should try induction on n.') == ''


def test_extract_lean_code_rejects_reply_that_trails_off_mid_tool_call() -> None:
    """Regression: a trajectory cut off after a tool call must not be submitted.

    Previously the raw-reply fallback returned this whole blob as the candidate
    proof. Lean cannot parse it, and a service-side exception on the garbage was
    then scored as SERVICE_FAILURE_SCORE (-300) instead of 0.
    """
    text = (
        'I found powerful lemmas: `Nat.card_units`, `Fintype.card_subtype_compl`. '
        'Let me now look up the reindex lemma.<tool_calls:6124c78e>\n'
        '<tool_call:6124c78e>mcp__lean4_tools_search__leansearch<tool_sep:6124c78e>\n'
        '<arg_key:6124c78e>query</arg_key:6124c78e>\n'
        '<arg_value:6124c78e>Finset sum reindex by equivalence</arg_value:6124c78e>\n'
        '</tool_call:6124c78e>\n'
        '</tool_calls:6124c78e><｜hy_eos:6124c78e｜>'
    )

    assert lean4_task._extract_lean_code(text) == ''


def test_extract_lean_code_rejects_empty_final_proof_block() -> None:
    assert lean4_task._extract_lean_code('<final_proof>\n\n</final_proof>') == ''


def test_extract_lean_code_accepts_self_contained_file() -> None:
    text = '''
<final_proof>
import Mathlib

theorem sample : True := by
  trivial
</final_proof>
'''

    assert lean4_task._extract_lean_code(text) == 'import Mathlib\n\ntheorem sample : True := by\n  trivial'


def test_looks_like_lean4_code_accepts_full_file_and_declaration_snippet() -> None:
    assert lean4_task._looks_like_lean4_code('import Mathlib\n\ntheorem s : True := by trivial')
    # Surrounding blank lines are irrelevant -- the body is stripped first.
    assert lean4_task._looks_like_lean4_code('\n\nimport Mathlib\ntheorem s : True := by trivial\n')
    assert lean4_task._looks_like_lean4_code('import Mathlib.Tactic\nimport Aesop\n')
    # verify_proof prepends the trusted header when the model omitted imports.
    assert lean4_task._looks_like_lean4_code('theorem sample : True := by\n  trivial')
    assert lean4_task._looks_like_lean4_code('lemma sample : True := by\n  trivial')
    assert lean4_task._looks_like_lean4_code('@[simp] private theorem sample : True := by\n  trivial')


def test_looks_like_lean4_code_accepts_common_top_level_commands() -> None:
    assert lean4_task._looks_like_lean4_code('set_option maxHeartbeats 0\n\ntheorem s : True := by trivial')
    assert lean4_task._looks_like_lean4_code('open scoped BigOperators\n\ntheorem s : True := by trivial')
    assert lean4_task._looks_like_lean4_code('namespace Foo\n\ntheorem s : True := by trivial\nend Foo')
    assert lean4_task._looks_like_lean4_code('noncomputable section\n\ntheorem s : True := by trivial')
    assert lean4_task._looks_like_lean4_code('#check Nat\n\ntheorem s : True := by trivial')
    assert lean4_task._looks_like_lean4_code('prelude\nimport Init\n\ntheorem s : True := by trivial')


def test_looks_like_lean4_code_allows_leading_comments() -> None:
    """Comments are legal before either imports or declaration snippets."""
    assert lean4_task._looks_like_lean4_code('-- Proof of the lemma\nimport Mathlib\n')
    assert lean4_task._looks_like_lean4_code('/- multi\n   line -/\nimport Mathlib\n')
    # Nested block comments.
    assert lean4_task._looks_like_lean4_code(
        '/- outer /- inner -/ still outer -/\ntheorem s : True := by trivial\n'
    )
    # Several comments, blank lines interleaved.
    assert lean4_task._looks_like_lean4_code('-- a\n\n/- b -/\n-- c\nlemma s : True := by trivial\n')
    # Doc comment.
    assert lean4_task._looks_like_lean4_code('/-- doc -/\ndef answer := 42\n')
    # Comment on the same line, before the command.
    assert lean4_task._looks_like_lean4_code('/- setup -/ theorem s : True := by trivial\n')


def test_looks_like_lean4_code_rejects_obvious_non_code() -> None:
    assert not lean4_task._looks_like_lean4_code('')
    assert not lean4_task._looks_like_lean4_code('-- just a comment, no code\n')
    assert not lean4_task._looks_like_lean4_code('/- nothing here -/\n')
    assert not lean4_task._looks_like_lean4_code('I think we should use induction.')
    assert not lean4_task._looks_like_lean4_code('Here is the proof:\ntheorem s : True := by trivial')
    assert not lean4_task._looks_like_lean4_code('```lean\ntheorem s : True := by trivial\n```')
    assert not lean4_task._looks_like_lean4_code('<tool_call>check_proof</tool_call>')
    assert not lean4_task._looks_like_lean4_code('<final_proof>theorem s : True := by trivial</final_proof>')
    # Tactics/terms alone are not top-level source and cannot declare the target.
    assert not lean4_task._looks_like_lean4_code('by\n  trivial')
    assert not lean4_task._looks_like_lean4_code('norm_num')


def test_looks_like_lean4_code_is_not_fooled_by_import_inside_a_comment() -> None:
    """`import` mentioned in a comment is not a real import command."""
    # A real declaration after the comment is still valid.
    assert lean4_task._looks_like_lean4_code('-- import Mathlib\ntheorem s : True := by trivial\n')
    assert not lean4_task._looks_like_lean4_code('/- import Mathlib -/\nnorm_num\n')


def test_looks_like_lean4_code_requires_a_complete_command_keyword() -> None:
    assert not lean4_task._looks_like_lean4_code('imported_lemma\n')
    assert not lean4_task._looks_like_lean4_code('importantFact : True := trivial\n')
    assert not lean4_task._looks_like_lean4_code('theoremish sample : True := by trivial\n')


def test_looks_like_lean4_code_rejects_misplaced_import() -> None:
    assert not lean4_task._looks_like_lean4_code('set_option maxHeartbeats 0\nimport Mathlib\n')
    assert not lean4_task._looks_like_lean4_code('open Nat\nimport Mathlib\n')
    assert not lean4_task._looks_like_lean4_code('theorem s : True := by trivial\nimport Mathlib\n')
