"""Patch-scale signals.

Two signals with opposite disclosure rules share this module, and the tests below
exist mainly to keep them apart. ``format_patch_scale_reference`` describes the
solution patch, so a leaked path or exact line count hands the worker the answer.
``format_current_patch_scale`` describes what the worker already wrote, so the same
detail is safe there -- what matters instead is that it never picks up the
reference's wording.

The fixture is the first patch in useful_code_bench_core200_multiturn.jsonl, kept
local so the test does not need a sibling dataset checkout.
"""

from nexus.extensions.tasks.hitl.utils.patch import (
    extract_patch_file_stats,
    extract_patch_scale_stats,
    format_current_patch_scale,
    format_patch_scale_reference,
    split_patch_by_file,
)


UCB_STYLE_PATCH = """\
diff --git a/src/djlint/formatter/expand.py b/src/djlint/formatter/expand.py
index 85470f9c2..bd2afad3c 100644
--- a/src/djlint/formatter/expand.py
+++ b/src/djlint/formatter/expand.py
@@ -23,6 +23,9 @@ def add_html_line(out_format: str, match: re.Match) -> str:
         if inside_ignored_block(config, html, match):
             return match.group(1)

+        if out_format == "\\n%s" and match.start() == 0:
+            return match.group(1)
+
         return out_format % match.group(1)

     html_tags = config.break_html_tags
@@ -45,7 +48,7 @@ def add_html_line(out_format: str, match: re.Match) -> str:
     # html tags - break after
     html = re.sub(
         re.compile(
-            rf"(</?(?:{html_tags})\\b(\\"[^\\"]*\\"|'[^']*'|{{[^}}]*}}|[^'\\">{{}}])*>)(?=[^\\n])",
+            rf"(</?(?:{html_tags})\\b(\\"[^\\"]*\\"|'[^']*'|{{[^}}]*}}|[^'\\">{{}}])*>)(?!\\s*?\\n)(?=[^\\n])",
             flags=re.IGNORECASE | re.VERBOSE,
         ),
         add_right,
@@ -54,7 +57,7 @@ def add_html_line(out_format: str, match: re.Match) -> str:

     # template tag breaks
     def should_i_move_template_tag(out_format: str, match: re.Match) -> str:
-        # ensure template tag is not inside an html tag
+        # ensure template tag is not inside an html tag and also not the first line of the file

         if inside_ignored_block(config, html, match):
             return match.group(1)
@@ -68,13 +71,14 @@ def should_i_move_template_tag(out_format: str, match: re.Match) -> str:
             html[: match.end()],
             re.MULTILINE | re.VERBOSE,
         ):
+            if out_format == "\\n%s" and match.start() == 0:
+                return match.group(1)
             return out_format % match.group(1)

         return match.group(1)

     # template tags
     # break before
-
     html = re.sub(
         re.compile(
             break_char
"""


DOCS_ONLY_PATCH = """\
diff --git a/docs/usage.md b/docs/usage.md
index 1111111..2222222 100644
--- a/docs/usage.md
+++ b/docs/usage.md
@@ -1,2 +1,4 @@
 before
+after
+more
"""


TWO_FILE_PATCH = DOCS_ONLY_PATCH + """\
diff --git a/tests/test_usage.py b/tests/test_usage.py
index 3333333..4444444 100644
--- a/tests/test_usage.py
+++ b/tests/test_usage.py
@@ -1,1 +1,2 @@
-old
+new
+extra
"""


# --- counting ---------------------------------------------------------------


def test_hunk_headers_and_file_headers_are_not_counted_as_changes():
    """``---``/``+++``/``@@`` lines start with +/- but are not content."""
    stats = extract_patch_file_stats(UCB_STYLE_PATCH)

    assert len(stats) == 1
    assert stats[0].path == 'src/djlint/formatter/expand.py'
    assert (stats[0].added, stats[0].removed) == (7, 3)
    assert stats[0].changed == 10


def test_stats_aggregate_over_files_and_dedupe_types():
    stats = extract_patch_scale_stats(TWO_FILE_PATCH)

    assert stats.file_count == 2
    assert (stats.added, stats.removed) == (4, 1)
    assert stats.file_types == ('docs', 'tests')
    # One bucket per file, in file order -- not a set, or a big file and a small
    # one sharing a bucket would collapse into a single entry.
    assert stats.file_change_buckets == ('1-10', '1-10')


def test_an_empty_patch_is_empty_rather_than_zero_shaped():
    assert extract_patch_scale_stats('').is_empty
    assert extract_patch_scale_stats('not a diff').is_empty
    assert not extract_patch_scale_stats(DOCS_ONLY_PATCH).is_empty
    assert split_patch_by_file('') == []


# --- reference signal (describes the solution) ------------------------------


def test_the_reference_signal_buckets_line_counts_but_names_files():
    """The path is disclosed; the line count is not.

    Naming the file it should touch is the point of the signal. Handing over the
    exact line count would let the worker stop as soon as the number matches.
    """
    text = format_patch_scale_reference(UCB_STYLE_PATCH)

    assert 'src/djlint/formatter/expand.py' in text
    assert '1-10' in text
    for exact in ('+7', '-3', '7 added', '3 removed', '10 lines'):
        assert exact not in text


def test_the_reference_signal_is_empty_when_there_is_nothing_to_calibrate():
    # An empty string drops the whole prompt section; a stub sentence would put a
    # meaningless expectation in front of the orchestrator.
    assert format_patch_scale_reference('') == ''


# --- current signal (describes the worker's own work) -----------------------


def test_the_current_signal_reports_scale_without_naming_files():
    signal = format_current_patch_scale(DOCS_ONLY_PATCH)

    assert '<current_patch_scale>' in signal
    assert 'Current worker patch scale: tiny' in signal
    assert 'per-file changed-line buckets: 1-10' in signal
    assert 'file types: docs' in signal
    assert 'docs/usage.md' not in signal


def test_the_current_signal_never_borrows_the_reference_wording():
    """The two signals must stay distinguishable in a prompt.

    Both end up in the orchestrator's context. If the current-patch block said
    "Expected", the orchestrator would read the worker's own output as the target.
    """
    signal = format_current_patch_scale(UCB_STYLE_PATCH)

    assert 'Expected' not in signal
    assert 'Expected' in format_patch_scale_reference(UCB_STYLE_PATCH)


def test_no_changes_is_stated_rather_than_omitted():
    """Silence and 'the worker changed nothing' mean different things here."""
    signal = format_current_patch_scale('')

    assert 'no file changes' in signal
    assert '<current_patch_scale>' in signal


def test_scale_labels_follow_both_line_and_file_count():
    """A label bounded only by lines would call a 20-file typo fix 'tiny'."""
    wide = 'diff --git a/f{0}.py b/f{0}.py\n--- a/f{0}.py\n+++ b/f{0}.py\n@@ -1 +1 @@\n-a\n+b\n'
    twenty_files = ''.join(wide.format(index) for index in range(20))

    stats = extract_patch_scale_stats(twenty_files)
    assert stats.file_count == 20
    assert stats.changed == 40
    assert 'very large' in format_current_patch_scale(twenty_files)


def test_many_files_are_summarized_instead_of_listed_one_bucket_each():
    """Past 8 files the buckets are counted, so the line cannot grow unbounded."""
    wide = 'diff --git a/f{0}.py b/f{0}.py\n--- a/f{0}.py\n+++ b/f{0}.py\n@@ -1 +1 @@\n-a\n+b\n'
    nine_files = ''.join(wide.format(index) for index in range(9))

    signal = format_current_patch_scale(nine_files)

    assert '9 file(s) at 1-10' in signal
    assert format_current_patch_scale(''.join(wide.format(i) for i in range(8))).count('1-10') == 8
