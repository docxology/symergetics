"""Real-filesystem controls for source-relative paper assets and captions."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check_images import ImageChecker
from validate_paper import PaperValidator


class PaperValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paper = self.root / "paper"
        self.markdown = self.paper / "markdown"
        self.markdown.mkdir(parents=True)
        self.output = self.root / "output" / "geometric" / "polyhedra"
        self.output.mkdir(parents=True)
        self.validator = PaperValidator(str(self.paper))
        self.checker = ImageChecker(str(self.paper))

    def errors(self, check):
        self.validator.results = []
        check()
        return [result for result in self.validator.results if result.severity == "error"]

    def image(self, path):
        (self.markdown / "01_abstract.md").write_text(
            f"![Figure 1: Test]({path})\n\n**Figure 1**: Detailed context.\n",
            encoding="utf-8",
        )

    def test_source_relative_image_resolves_consistently(self):
        (self.output / "figure.png").write_bytes(b"path-check fixture")
        self.image("../../output/geometric/polyhedra/figure.png")
        self.assertEqual(self.errors(self.validator._validate_image_references), [])
        self.checker._check_image_existence()
        self.checker._check_image_formatting()
        self.checker._check_path_consistency()
        self.assertEqual(self.checker.issues, [])

    def test_missing_target_is_not_masked_by_another_basename(self):
        (self.output / "same.png").write_bytes(b"path-check fixture")
        self.image("../../output/geometric/missing/same.png")
        errors = self.errors(self.validator._validate_image_references)
        self.assertEqual([result.check_name for result in errors], ["image_exists"])
        self.checker._check_image_existence()
        self.assertEqual([issue["type"] for issue in self.checker.issues], ["image_missing"])

    def test_directory_is_not_an_image_file(self):
        (self.output / "directory.png").mkdir()
        self.image("../../output/geometric/polyhedra/directory.png")
        errors = self.errors(self.validator._validate_image_references)
        self.assertEqual([result.check_name for result in errors], ["image_exists"])

    def test_escaped_and_absolute_paths_are_rejected(self):
        (self.root / "outside.png").write_bytes(b"path-check fixture")
        for path in ("../../outside.png", str(self.output / "absolute.png")):
            with self.subTest(path=path):
                self.image(path)
                errors = self.errors(self.validator._validate_image_references)
                self.assertEqual([result.check_name for result in errors], ["image_path"])

    def test_current_section_roster_and_missing_section(self):
        sections = (
            "00_title", "01_abstract", "02_introduction", "03_mathematical_foundations",
            "04_system_architecture", "05_computational_methods", "06_results",
            "07_geometric_applications", "08_pattern_discovery", "09_research_applications",
            "10_ongoing_questions_inquiries", "11_conclusion",
        )
        for section in sections:
            (self.markdown / (section + ".md")).write_text("# Section\n")
        self.assertEqual(self.errors(self.validator._validate_markdown_structure), [])
        (self.markdown / "06_results.md").unlink()
        errors = self.errors(self.validator._validate_markdown_structure)
        self.assertEqual(len(errors), 1)
        self.assertIn("06_results.md", errors[0].message)

    def test_caption_forms_after_blank_lines_match_both_gates(self):
        for caption in ("**Figure 1**: Detailed caption", "**Figure 1:** Detailed caption"):
            with self.subTest(caption=caption):
                (self.markdown / "01_abstract.md").write_text(
                    "![Figure 1: Test](../../output/geometric/polyhedra/figure.png)\n\n" + caption + "\n"
                )
                self.assertEqual(self.errors(self.validator._validate_figure_captions), [])
                self.checker.issues = []
                self.checker._check_figure_captions()
                self.assertEqual(self.checker.issues, [])

    def test_absent_or_malformed_caption_is_rejected(self):
        for caption in ("", "**Figure 1**:"):
            with self.subTest(caption=caption):
                (self.markdown / "01_abstract.md").write_text(
                    "![Figure 1: Test](../../output/geometric/polyhedra/figure.png)\n\n" + caption + "\n"
                )
                errors = self.errors(self.validator._validate_figure_captions)
                self.assertEqual([result.check_name for result in errors], ["figure_caption"])
                self.checker.issues = []
                self.checker._check_figure_captions()
                self.assertEqual(len(self.checker.issues), 1)
                self.assertEqual(self.checker.issues[0]["severity"], "error")

    def test_image_cli_fails_missing_target_then_passes_real_relative_target(self):
        self.image("../../output/geometric/polyhedra/figure.png")
        script = Path(__file__).with_name("check_images.py")
        command = [sys.executable, "-I", str(script), "--paper-dir", str(self.paper),
                   "--output", str(self.root / "image-report.md")]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
        (self.output / "figure.png").write_bytes(b"path-check fixture")
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
