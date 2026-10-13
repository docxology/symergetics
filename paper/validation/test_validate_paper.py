"""Focused real-filesystem controls for the manuscript gate.

Run: python -m unittest discover -s paper/validation -p test_validate_paper.py
"""

import tempfile
import unittest
from pathlib import Path

from validate_paper import PaperValidator


class PaperValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paper = self.root / "paper"
        self.markdown = self.paper / "markdown"
        self.markdown.mkdir(parents=True)
        self.output = self.root / "output"
        self.output.mkdir()
        self.validator = PaperValidator(str(self.paper))

    def errors(self, check):
        self.validator.results = []
        check()
        return [result for result in self.validator.results if result.severity == "error"]

    def image(self, path):
        (self.markdown / "01_abstract.md").write_text(
            "![Figure 1: Test](" + path + ")\n", encoding="utf-8"
        )

    def test_project_rooted_image_resolves_once(self):
        (self.output / "figure.png").write_bytes(b"fixture")
        self.image("output/figure.png")
        self.assertEqual(self.errors(self.validator._validate_image_references), [])

    def test_missing_image_is_error(self):
        self.image("output/missing.png")
        self.assertTrue(self.errors(self.validator._validate_image_references))

    def test_directory_is_not_an_image_file(self):
        (self.output / "directory.png").mkdir()
        self.image("output/directory.png")
        self.assertTrue(self.errors(self.validator._validate_image_references))

    def test_escaped_or_unrooted_path_is_error(self):
        (self.root / "outside.png").write_bytes(b"fixture")
        for path in ("output/../outside.png", "outside.png"):
            with self.subTest(path=path):
                self.image(path)
                self.assertTrue(self.errors(self.validator._validate_image_references))

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

    def test_style_guide_and_legacy_captions(self):
        for caption in ("**Figure 1**: Detailed caption", "**Figure 1:** Detailed caption"):
            with self.subTest(caption=caption):
                (self.markdown / "01_abstract.md").write_text(
                    "![Figure 1: Test](output/figure.png)\n\n" + caption + "\n"
                )
                self.assertEqual(self.errors(self.validator._validate_figure_captions), [])

    def test_absent_or_malformed_caption_is_error(self):
        for caption in ("", "# Heading", "**Figure 1**:"):
            with self.subTest(caption=caption):
                (self.markdown / "01_abstract.md").write_text(
                    "![Figure 1: Test](output/figure.png)\n\n" + caption + "\n"
                )
                self.assertTrue(self.errors(self.validator._validate_figure_captions))
