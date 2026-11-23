from reviewer.diff.parser import summarize_diff

# A real-shaped unified diff (what `Accept: application/vnd.github.diff` returns):
# 2 files, 3 added lines, 1 removed line.
TWO_FILE_DIFF = """\
diff --git a/calc.py b/calc.py
index 1111111..2222222 100644
--- a/calc.py
+++ b/calc.py
@@ -1,3 +1,4 @@
 def add(a, b):
-    return a+b
+    return a + b
+
diff --git a/stats.py b/stats.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/stats.py
@@ -0,0 +1 @@
+def mean(xs): return sum(xs) / len(xs)
"""

# Only file headers: a rename or mode change has headers but no +/- content lines.
HEADERS_ONLY_DIFF = """\
diff --git a/old.py b/new.py
--- a/old.py
+++ b/new.py
"""


class TestSummarizeDiff:
    """`summarize_diff(diff) -> (files, additions, deletions)` for the placeholder comment.

    It counts one file per `diff --git` header, and counts `+` / `-` content
    lines while skipping the `+++` / `---` file-header lines. Context lines
    (leading space) and `@@` hunk headers count as neither.
    """

    def test_counts_files_additions_deletions(self):
        assert summarize_diff(TWO_FILE_DIFF) == (2, 3, 1)

    def test_empty_diff_in_pr(self):
        assert summarize_diff("") == (0, 0, 0)

    def test_file_headers_are_not_counted_as_lines(self):
        # +++ b/new.py` / `--- a/old.py` start with +/- but are headers, not changes
        assert summarize_diff(HEADERS_ONLY_DIFF) == (1, 0, 0)
