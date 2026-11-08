"""Real-PDF regressions for readable text, pagination, tables and source figures."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image as RasterImage
from pypdf import PdfReader
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate

from pdf_renderer import ScientificPaperRenderer


class TestScientificPaperRenderer(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paper = self.root / "paper"
        (self.paper / "markdown").mkdir(parents=True)
        self.assets = self.root / "output" / "figures"
        self.assets.mkdir(parents=True)
        self.renderer = ScientificPaperRenderer(str(self.paper), str(self.root / "pdf"))

    def raster(self, name, size, color):
        path = self.assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        RasterImage.new("RGB", size, color).save(path)
        return path

    def render_source(self, filename, content):
        (self.paper / "markdown" / filename).write_text(content, encoding="utf-8")
        return PdfReader(self.renderer.generate_pdf("preview.pdf"))

    def test_pdf_prints_distinct_h1_through_h4_headings(self):
        document = self.render_source(
            "04_system_architecture.md",
            "\n\n".join(
                "#" * level + f" Heading Level {level}\n\nSubstantive body."
                for level in range(1, 5)
            ),
        )
        sizes = {}

        def record_text(text, transform, text_matrix, font, size):
            if text.strip().startswith("Heading Level"):
                sizes[text.strip()] = size

        document.pages[0].extract_text(visitor_text=record_text)
        self.assertEqual(set(sizes), {f"Heading Level {level}" for level in range(1, 5)})
        ordered = [sizes[f"Heading Level {level}"] for level in range(1, 5)]
        self.assertTrue(all(first > second for first, second in zip(ordered, ordered[1:])))

    def test_pdf_headings_stay_with_body_not_a_spacer(self):
        body = "Opening body continues with substantive detail for the reader. " * 4
        sources = [
            ("#" * level + f" Boundary Heading {level}\n\n" + body,
             f"Boundary Heading {level}", "Opening body")
            for level in range(1, 5)
        ]
        sources.append(("```python\nvalue = 7\n```", "Python Code:", "value = 7"))
        for index, (source, heading, opening) in enumerate(sources):
            with self.subTest(heading=heading):
                path = self.root / f"boundary-{index}.pdf"
                story = [
                    Paragraph("<br/>".join(["Preface"] * 9), self.renderer.styles["Normal"]),
                    *self.renderer.process_markdown_content(source),
                ]
                SimpleDocTemplate(
                    str(path), pagesize=(300, 240),
                    leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24,
                ).build(story)
                pages = [page.extract_text() for page in PdfReader(path).pages]
                self.assertEqual(len(pages), 2)
                self.assertIn("Preface", pages[0])
                self.assertNotIn(heading, pages[0])
                self.assertIn(heading, pages[1])
                self.assertIn(opening, pages[1])

    def test_pdf_inline_code_preserves_literal_identifiers_and_link_syntax(self):
        document = self.render_source(
            "04_system_architecture.md",
            "Literal identifiers: `__init__`, `a*b*c`, `x < y` and `[name](url)`.\n",
        )
        text = document.pages[0].extract_text()
        for literal in ("__init__", "a*b*c", "x < y", "[name](url)"):
            self.assertIn(literal, text)
        self.assertEqual(len(document.pages[0].get("/Annots", [])), 0)

    def test_pdf_table_keeps_escaped_pipes_and_literal_code_in_their_column(self):
        for outer_pipes in (True, False):
            with self.subTest(outer_pipes=outer_pipes):
                lines = [
                    "| Name | Optional | Literal |",
                    "| :--- | :---: | ---: |",
                    "| entry | | `__init__` |",
                    "| other | | `a\\|b` |",
                ]
                if not outer_pipes:
                    lines = [line[1:-1] for line in lines]
                document = self.render_source(
                    "04_system_architecture.md", "\n".join(lines) + "\n",
                )
                anchors = {}

                def record_text(text, transform, text_matrix, font, size):
                    if text.strip() in ("Literal", "__init__", "a|b"):
                        anchors[text.strip()] = (
                            text_matrix[4] * transform[0]
                            + text_matrix[5] * transform[2] + transform[4]
                        )

                text = document.pages[0].extract_text(visitor_text=record_text)
                self.assertIn("__init__", text)
                self.assertIn("a|b", text)
                self.assertNotIn("---", text)
                self.assertAlmostEqual(anchors["Literal"], anchors["__init__"], places=3)
                self.assertAlmostEqual(anchors["Literal"], anchors["a|b"], places=3)

    def test_pdf_short_dash_rows_remain_literal_not_table_delimiters(self):
        document = self.render_source(
            "04_system_architecture.md",
            "| Name | Value |\n| - | - |\n| entry | 2 |\n",
        )
        self.assertIn("| - | - |", document.pages[0].extract_text())

    def test_pdf_preserves_coordinate_operators_and_volume_units(self):
        document = self.render_source(
            "03_mathematical_foundations.md",
            "# Exact coordinate and unit conventions\n\n"
            "XYZ = **a−b−c+d**; displacement a−d; scale |s|³.\n\n"
            "Volume |det|/4.\nScale |s|³.\n\n"
            "Edge 2√2; primes p ≤ n; Cartesian volume (8/3) × native.\n",
        )
        text = "\n".join(page.extract_text() for page in document.pages)
        for expression in ("a−b−c+d", "a−d", "|s|³", "2√2", "p ≤ n", "(8/3) × native",
                           "Volume |det|/4."):
            self.assertIn(expression, text)

    def test_pdf_table_keeps_empty_columns_and_stays_on_page(self):
        document = self.render_source(
            "04_system_architecture.md",
            "# Source responsibilities\n\n"
            "| Source | Optional | Responsibility |\n"
            "| --- | --- | --- |\n"
            "| [core/numbers.py](../../symergetics/core/numbers.py) | | "
            "Rational-role " + "exact rational arithmetic and root boundaries " * 8 + "|\n",
        )
        anchors = []

        def record_text(text, transform, text_matrix, font, size):
            if text.strip():
                x = (text_matrix[4] * transform[0]
                     + text_matrix[5] * transform[2] + transform[4])
                anchors.append((text.strip(), x))

        page = document.pages[0]
        text = page.extract_text(visitor_text=record_text)
        heading_x = next(x for value, x in anchors if value == "Responsibility")
        role_x = next(x for value, x in anchors if value.startswith("Rational-role"))
        self.assertAlmostEqual(role_x, heading_x, places=3)
        self.assertTrue(all(0 <= x < float(page.mediabox.width) for _, x in anchors))
        self.assertNotIn("](../../", text)
        self.assertNotIn("---", text)

    def test_exact_directory_wins_over_duplicate_basename(self):
        selected = self.raster("selected/same.png", (800, 200), "navy")
        self.raster("other/same.png", (100, 400), "orange")
        found = self.renderer._find_image_path("../../output/figures/selected/same.png")
        self.assertTrue(found.samefile(selected))
        self.assertIsNone(
            self.renderer._find_image_path("../../output/figures/missing/same.png")
        )
        with self.assertRaises(ValueError):
            self.renderer._find_image_path(str(selected))

    def test_portrait_and_landscape_keep_intrinsic_aspect(self):
        self.raster("wide.png", (800, 200), "navy")
        self.raster("tall.png", (100, 400), "orange")
        content = (
            "![Figure 1: Wide](../../output/figures/wide.png)\n\n"
            "**Figure 1**: Wide source.\n\n"
            "![Figure 2: Tall](../../output/figures/tall.png)\n"
            "**Figure 2:** Tall source.\n"
        )
        figures = [
            item for item in self.renderer.process_markdown_content(content)
            if isinstance(item, Image)
        ]
        self.assertEqual(len(figures), 2)
        self.assertAlmostEqual(figures[0].drawWidth / figures[0].drawHeight, 4.0)
        self.assertAlmostEqual(figures[1].drawWidth / figures[1].drawHeight, 0.25)

    def test_repeated_asset_keeps_both_authored_caption_occurrences(self):
        self.raster("same.png", (300, 200), "navy")
        first = self.renderer.process_markdown_content(
            "![Figure 5: First](../../output/figures/same.png)\n\n"
            "**Figure 5**: First context.\n"
        )
        second = self.renderer.process_markdown_content(
            "![Figure 8: Second](../../output/figures/same.png)\n\n"
            "**Figure 8:** Second context.\n"
        )
        combined = first + second
        self.assertEqual(sum(isinstance(item, Image) for item in combined), 2)
        captions = [
            item.getPlainText() for item in combined
            if isinstance(item, Paragraph) and item.style.name == "FigureCaption"
        ]
        self.assertEqual(captions, ["Figure 5: First context.", "Figure 8: Second context."])

    def test_missing_reference_is_not_replaced_by_another_same_named_image(self):
        self.raster("elsewhere/same.png", (300, 200), "navy")
        with self.assertRaises(FileNotFoundError):
            self.renderer.process_markdown_content(
                "![Figure 1: Missing](../../output/figures/missing/same.png)\n"
            )


if __name__ == "__main__":
    unittest.main()
