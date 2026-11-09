"""Portfolio standard layout check: required files exist and no public grade.

No lock-hash test: the repo is not pre-registered (PROPOSAL.md is an
unlocked draft written after the results, docs/decisions/0003).
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED = ["README.md", "PROPOSAL.md", "REPORT.md", "CHANGELOG.md", "CITATION.cff", "SECURITY.md", "LICENSE",
        "requirements.txt", "level-1-learn/signal_ml.py", "level-2-build/artifacts/model/1/manifest.json",
        "level-1-learn/results_learn.json", "level-2-build/results_build.json",
        "level-3-deep/results_deep.json", "level-3-deep/results_rigor.json", "level-3-deep/JUDGMENT.md",
        "docs/decisions/0001-numpy-substitutes-for-tensorflow.md"]
GRADE = re.compile(r"of 10,000|\b\d[\d,]*/10,?000\b|Grade:|\b\d+(\.\d+)?/10\b|/1000\b|rubric score", re.I)
PUBLIC_EXT = (".md", ".jsx", ".js", ".html", ".cff")


class LayoutTests(unittest.TestCase):
    def test_required_files_exist(self):
        missing = [p for p in NEED if not os.path.exists(os.path.join(ROOT, p))]
        self.assertEqual(missing, [], "missing: " + ", ".join(missing))

    def test_readme_opens_with_the_abstract(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
            text = f.read()
        for field in ("**Question.**", "**Headline.**", "**Field.**", "**Job family.**", "**Verdict.**",
                      "## Reproduce", "## Lifecycle", "## Layout"):
            self.assertIn(field, text)

    def test_no_public_grade(self):
        hits = []
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in (".git", "node_modules", "dist", "generated")]
            for name in files:
                if name.endswith(PUBLIC_EXT):
                    path = os.path.join(d, name)
                    with open(path, encoding="utf-8") as f:
                        for n, line in enumerate(f, 1):
                            if GRADE.search(line):
                                hits.append(f"{os.path.relpath(path, ROOT)}:{n}")
        self.assertEqual(hits, [], "grades are internal only: " + ", ".join(hits))

    def test_grade_pattern_catches_a_grade(self):
        for bad in ("Grade: 7,600 of 10,000", "about 5.5/10 as research", "Total /1000"):
            self.assertTrue(GRADE.search(bad), bad)


if __name__ == "__main__":
    unittest.main()
