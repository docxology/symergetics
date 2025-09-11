#!/usr/bin/env python3
"""
Test Suite for Synergetics Paper PDF Renderer

This module provides comprehensive testing for the PDF generation system,
ensuring proper formatting, image integration, and content accuracy.

Author: Daniel Ari Friedman
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pdf_renderer import ScientificPaperRenderer, PaperSection


class TestScientificPaperRenderer(unittest.TestCase):
    """Test cases for the PDF renderer"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.markdown_dir = self.test_dir / "markdown"
        self.output_dir = self.test_dir / "output"
        self.markdown_dir.mkdir()
        self.output_dir.mkdir()

        # Create mock markdown files
        self._create_mock_markdown_files()

        # Initialize renderer
        self.renderer = ScientificPaperRenderer(
            paper_dir=str(self.test_dir),
            output_dir=str(self.output_dir)
        )

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir)

    def _create_mock_markdown_files(self):
        """Create mock markdown files for testing"""
        # Title section
        (self.markdown_dir / "00_title.md").write_text("""
# Test Paper Title

## Front Matter

**Authors:** Test Author
**Email:** test@example.com
**Abstract:** This is a test abstract.
""")

        # Abstract section
        (self.markdown_dir / "01_abstract.md").write_text("""
## Abstract

This is a test abstract for the paper. It contains multiple paragraphs
and demonstrates the rendering capabilities.

### Key Points

- Point 1
- Point 2
- Point 3
""")

        # Content section
        (self.markdown_dir / "02_content.md").write_text("""
## Test Section

This is regular paragraph content.

### Subsection

More content here.

```python
def test_function():
    return "test"
```

#### Sub-subsection

Even more content.
""")

    def test_initialization(self):
        """Test renderer initialization"""
        self.assertIsInstance(self.renderer, ScientificPaperRenderer)
        self.assertEqual(self.renderer.paper_dir, self.test_dir)
        self.assertEqual(self.renderer.markdown_dir, self.markdown_dir)
        self.assertEqual(self.renderer.output_dir, self.output_dir)

    def test_load_sections(self):
        """Test loading and ordering sections"""
        sections = self.renderer.load_sections()

        self.assertEqual(len(sections), 3)

        # Check ordering
        self.assertEqual(sections[0].order, 0)
        self.assertEqual(sections[1].order, 1)
        self.assertEqual(sections[2].order, 2)

        # Check titles
        self.assertIn("Test Paper Title", sections[0].title)
        self.assertEqual(sections[1].title, "Abstract")
        self.assertEqual(sections[2].title, "Test Section")

    def test_process_markdown_content(self):
        """Test markdown content processing"""
        test_content = """
## Test Header

This is a paragraph.

### Subsection

Another paragraph.

```python
code block
```

- List item 1
- List item 2
"""

        flowables = self.renderer.process_markdown_content(test_content)

        # Should have multiple flowables (headers, paragraphs, code, lists)
        self.assertGreater(len(flowables), 3)

    def test_find_image_path(self):
        """Test image path resolution"""
        # Create a mock image file
        img_file = self.output_dir / "test_image.png"
        img_file.write_text("mock image content")

        # Test finding existing image
        found_path = self.renderer._find_image_path("test_image.png")
        self.assertEqual(found_path, img_file)

        # Test finding non-existent image
        not_found = self.renderer._find_image_path("nonexistent.png")
        self.assertIsNone(not_found)

    @patch('pdf_renderer.SimpleDocTemplate')
    def test_generate_pdf(self, mock_doc):
        """Test PDF generation"""
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance

        # Generate PDF
        output_path = self.renderer.generate_pdf("test_output.pdf")

        # Verify document was created and built
        mock_doc.assert_called_once()
        mock_doc_instance.build.assert_called_once()

        # Check output path
        expected_path = self.output_dir / "test_output.pdf"
        self.assertEqual(output_path, str(expected_path))

    def test_metadata_handling(self):
        """Test metadata handling"""
        metadata = self.renderer.metadata

        self.assertIn('title', metadata)
        self.assertIn('author', metadata)
        self.assertIn('email', metadata)
        self.assertIn('orcid', metadata)
        self.assertIn('date', metadata)

    def test_styles_setup(self):
        """Test ReportLab styles setup"""
        styles = self.renderer.styles

        # Check that custom styles are defined
        required_styles = [
            'PaperTitle', 'AuthorInfo', 'Abstract',
            'SectionHeader', 'SubsectionHeader', 'CodeBlock', 'FigureCaption'
        ]

        for style_name in required_styles:
            self.assertIn(style_name, styles)


class TestPaperSection(unittest.TestCase):
    """Test cases for PaperSection dataclass"""

    def test_paper_section_creation(self):
        """Test PaperSection creation and attributes"""
        section = PaperSection(
            filename="test.md",
            title="Test Section",
            content="# Test Content",
            order=1,
            images=["fig1.png", "fig2.png"]
        )

        self.assertEqual(section.filename, "test.md")
        self.assertEqual(section.title, "Test Section")
        self.assertEqual(section.content, "# Test Content")
        self.assertEqual(section.order, 1)
        self.assertEqual(section.images, ["fig1.png", "fig2.png"])

    def test_paper_section_defaults(self):
        """Test PaperSection default values"""
        section = PaperSection(
            filename="test.md",
            title="Test Section",
            content="# Test Content",
            order=1
        )

        self.assertEqual(section.images, [])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.markdown_dir = self.test_dir / "markdown"
        self.output_dir = self.test_dir / "output"
        self.markdown_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up integration test environment"""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_full_pipeline(self):
        """Test the complete PDF generation pipeline"""
        # Create comprehensive test content
        (self.markdown_dir / "00_title.md").write_text("""
# Integration Test Paper

## Front Matter

**Authors:** Integration Test Author
**Abstract:** Full pipeline test.
""")

        (self.markdown_dir / "01_abstract.md").write_text("""
## Abstract

This tests the full PDF generation pipeline.
""")

        # Create renderer and generate PDF
        renderer = ScientificPaperRenderer(str(self.test_dir), str(self.output_dir))

        with patch('pdf_renderer.SimpleDocTemplate') as mock_doc:
            mock_doc_instance = Mock()
            mock_doc.return_value = mock_doc_instance

            output_path = renderer.generate_pdf("integration_test.pdf")

            # Verify the pipeline executed
            mock_doc.assert_called_once()
            mock_doc_instance.build.assert_called_once()

            # Check output file would be created
            expected_path = self.output_dir / "integration_test.pdf"
            self.assertEqual(output_path, str(expected_path))


def run_tests():
    """Run the test suite"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()

