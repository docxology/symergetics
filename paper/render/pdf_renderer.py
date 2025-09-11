#!/usr/bin/env python3
"""
PDF Renderer for Synergetics Research Paper

This module provides comprehensive PDF generation from markdown sections,
including proper scientific formatting, front matter, and integration
of visualizations from the output/ directory.

Author: Daniel Ari Friedman
Email: daniel@activeinference.institute
ORCID: 0000-0001-6232-9096
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Third-party imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak,
        Image, Table, TableStyle, Flowable
    )
    # Hyperlink support is built into ReportLab's HTML-like markup
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
except ImportError:
    print("Error: reportlab not installed. Install with: pip install reportlab")
    sys.exit(1)

try:
    import markdown
    from markdown.extensions import Extension
    from markdown.preprocessors import Preprocessor
except ImportError:
    print("Error: markdown not installed. Install with: pip install markdown")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: pyyaml not installed. Install with: pip install pyyaml")
    sys.exit(1)


@dataclass
class PaperSection:
    """Represents a paper section with metadata"""
    filename: str
    title: str
    content: str
    order: int
    images: List[str] = None

    def __post_init__(self):
        if self.images is None:
            self.images = []


class ScientificPaperRenderer:
    """Main renderer for scientific papers with Synergetics formatting"""

    def __init__(self, paper_dir: str, output_dir: str = None):
        self.paper_dir = Path(paper_dir)
        self.markdown_dir = self.paper_dir / "markdown"
        self.output_dir = Path(output_dir) if output_dir else self.paper_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

        # Initialize ReportLab components
        self.styles = self._setup_styles()

        # Track document-level figure numbering
        self.document_figure_counter = 1
        self.processed_figures = set()  # Track processed images to prevent duplicates

        # Paper metadata
        self.metadata = {
            'title': 'Symergetics: Symbolic Synergetics for Rational Arithmetic, Geometric Pattern Discovery, All-Integer Accounting',
            'author': 'Daniel Ari Friedman',
            'email': 'daniel@activeinference.institute',
            'orcid': '0000-0001-6232-9096',
            'date': datetime.now().strftime('%B %d, %Y'),
            'institution': 'Independent Researcher',
            'affiliation': 'Active Inference Institute'
        }

    def _setup_styles(self) -> Dict:
        """Setup ReportLab styles for scientific formatting"""
        styles = getSampleStyleSheet()

        # Title style
        styles.add(ParagraphStyle(
            name='PaperTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.black
        ))

        # Author style
        styles.add(ParagraphStyle(
            name='AuthorInfo',
            parent=styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=20
        ))

        # Abstract style
        styles.add(ParagraphStyle(
            name='Abstract',
            parent=styles['Normal'],
            fontSize=11,
            leftIndent=20,
            rightIndent=20,
            alignment=TA_JUSTIFY,
            spaceAfter=15
        ))

        # Section headers
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.black
        ))

        # Subsection headers
        styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.red
        ))

        # Code blocks
        styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            leftIndent=20,
            rightIndent=20,
            background=colors.lightgrey,
            borderColor=colors.black,
            borderWidth=1,
            borderPadding=8,
            spaceAfter=12,
            spaceBefore=6
        ))

        # Python code blocks (special styling)
        styles.add(ParagraphStyle(
            name='PythonCodeBlock',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            leftIndent=25,
            rightIndent=25,
            background=colors.whitesmoke,
            borderColor=colors.red,
            borderWidth=2,
            borderPadding=10,
            spaceAfter=15,
            spaceBefore=8
        ))

        # Mermaid diagram blocks
        styles.add(ParagraphStyle(
            name='MermaidBlock',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=8,
            leftIndent=15,
            rightIndent=15,
            background=colors.lightgrey,
            borderColor=colors.red,
            borderWidth=1,
            borderPadding=6,
            spaceAfter=12,
            spaceBefore=6,
            textColor=colors.darkred
        ))

        # Figure captions
        styles.add(ParagraphStyle(
            name='FigureCaption',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=15,
            fontStyle='italic'
        ))

        # Link style
        styles.add(ParagraphStyle(
            name='Link',
            parent=styles['Normal'],
            textColor=colors.red,
            underline=True
        ))

        # Blockquote style
        styles.add(ParagraphStyle(
            name='Blockquote',
            parent=styles['Normal'],
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=10,
            background=colors.whitesmoke
        ))

        # Inline code style
        styles.add(ParagraphStyle(
            name='InlineCode',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            background=colors.lightgrey,
            borderPadding=2
        ))

        # Quote style
        styles.add(ParagraphStyle(
            name='Quote',
            parent=styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            fontStyle='italic'
        ))

        # Quote author style
        styles.add(ParagraphStyle(
            name='QuoteAuthor',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontStyle='italic'
        ))

        return styles

    def load_sections(self) -> List[PaperSection]:
        """Load the exact 11 specified paper sections in correct order"""
        sections = []

        # Define the exact files to include in the correct order
        paper_files = [
            ("00_title.md", "Title"),
            ("01_abstract.md", "Abstract"),
            ("02_introduction.md", "Introduction"),
            ("03_mathematical_foundations.md", "Mathematical Foundations"),
            ("04_system_architecture.md", "System Architecture"),
            ("05_computational_methods.md", "Computational Methods"),
            ("06_geometric_applications.md", "Geometric Applications"),
            ("07_pattern_discovery.md", "Pattern Discovery"),
            ("08_research_applications.md", "Research Applications"),
            ("09_conclusion.md", "Conclusion"),
            ("10_ongoing_questions_inquiries.md", "Ongoing Questions and Inquiries")
        ]

        for order, (filename, title) in enumerate(paper_files, 0):
            file_path = self.markdown_dir / filename

            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')

                # Find image references in content
                images = re.findall(r'!\[.*?\]\((.*?)\)', content)

                section = PaperSection(
                    filename=filename,
                    title=title,
                    content=content,
                    order=order,
                    images=images
                )
                sections.append(section)
                print(f"Loaded section {order}: {title} ({filename})")
            else:
                print(f"Warning: Section file not found: {filename}")

        return sections


    def process_markdown_content(self, content: str) -> List[Flowable]:
        """Convert markdown content to ReportLab flowables with comprehensive formatting"""
        flowables = []

        # Split content into lines for processing
        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Headers
            if line.startswith('# '):
                # Main header (skip title as it's handled separately)
                i += 1
                continue
            elif line.startswith('## '):
                title = self._process_inline_formatting(line[3:].strip())
                flowables.append(Paragraph(title, self.styles['SectionHeader']))
            elif line.startswith('### '):
                title = self._process_inline_formatting(line[4:].strip())
                flowables.append(Paragraph(title, self.styles['SubsectionHeader']))
            elif line.startswith('#### '):
                title = self._process_inline_formatting(line[5:].strip())
                flowables.append(Paragraph(title, self.styles['SubsectionHeader']))
            elif line.startswith('##### ') or line.startswith('###### '):
                title = self._process_inline_formatting(line.lstrip('#').strip())
                flowables.append(Paragraph(title, self.styles['Normal']))

            # Code blocks
            elif line.startswith('```'):
                code_lines = []
                language = line[3:].strip()  # Get language if specified
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                code_content = '\n'.join(code_lines)

                # Choose appropriate style based on language
                if language.lower() == 'python':
                    # Special handling for Python code - truncate long blocks
                    truncated_code = self._truncate_code_block(code_content, max_lines=15)
                    flowables.append(Paragraph(f"🐍 Python Code:", self.styles['SectionHeader']))
                    flowables.append(Spacer(1, 4))
                    flowables.append(Paragraph(truncated_code, self.styles['PythonCodeBlock']))

                    # Add truncation note if code was truncated
                    original_lines = len(code_content.split('\n'))
                    if original_lines > 15:
                        flowables.append(Paragraph(f"[Code truncated - showing 15/{original_lines} lines]", self.styles['Normal']))
                elif language.lower() == 'mermaid':
                    # Special handling for Mermaid diagrams
                    flowables.append(Paragraph(f"📊 Mermaid Diagram", self.styles['SectionHeader']))
                    flowables.append(Spacer(1, 4))

                    # Try to render diagram as image
                    diagram_id = f"mermaid_{len(flowables)}"  # Unique ID for this diagram
                    image_path = self._render_mermaid_diagram(code_content, diagram_id)

                    if image_path:
                        # Check if the image exists at the expected location
                        abs_image_path = (self.paper_dir / image_path).absolute()
                        if abs_image_path.exists():
                            # Successfully rendered as image - include it in PDF
                            try:
                                # Calculate proper dimensions based on image aspect ratio
                                width_px, height_px = self._get_image_dimensions(abs_image_path)
                                aspect_ratio = width_px / height_px

                                # Set maximum width and calculate height to maintain aspect ratio
                                max_width = 5.5 * inch  # Leave some margin
                                max_height = 6 * inch   # Reasonable maximum height

                                if aspect_ratio > 1:  # Landscape/wide image
                                    img_width = min(max_width, max_height * aspect_ratio)
                                    img_height = img_width / aspect_ratio
                                else:  # Portrait/tall image
                                    img_height = min(max_height, max_width / aspect_ratio)
                                    img_width = img_height * aspect_ratio

                                img = Image(str(abs_image_path), width=img_width, height=img_height)
                                img.hAlign = 'CENTER'
                                flowables.append(img)
                                flowables.append(Spacer(1, 6))

                                # Generate comprehensive caption
                                caption = self._generate_diagram_caption(code_content, diagram_id)
                                flowables.append(Paragraph(caption, self.styles['FigureCaption']))

                            except Exception as e:
                                print(f"Warning: Failed to add Mermaid image to PDF: {e}")
                                # Fall back to text representation
                                formatted_mermaid = self._format_mermaid_content(code_content)
                                flowables.append(Paragraph(formatted_mermaid, self.styles['MermaidBlock']))
                        else:
                            # Image file doesn't exist - use text representation
                            print(f"Info: Mermaid image not found at {abs_image_path}, using text representation for diagram {diagram_id}")
                            formatted_mermaid = self._format_mermaid_content(code_content)
                            flowables.append(Paragraph(formatted_mermaid, self.styles['MermaidBlock']))
                            flowables.append(Spacer(1, 6))
                            flowables.append(Paragraph("💡 Note: This Mermaid diagram represents a visual graph structure. The above shows the textual representation of nodes, connections, and relationships that would be displayed visually in diagram rendering software.", self.styles['Normal']))
                    else:
                        # Failed to render image - use text representation
                        print(f"Info: Using text representation for Mermaid diagram {diagram_id}")
                        formatted_mermaid = self._format_mermaid_content(code_content)
                        flowables.append(Paragraph(formatted_mermaid, self.styles['MermaidBlock']))
                        flowables.append(Spacer(1, 6))
                        flowables.append(Paragraph("💡 Note: This Mermaid diagram represents a visual graph structure. The above shows the textual representation of nodes, connections, and relationships that would be displayed visually in diagram rendering software.", self.styles['Normal']))
                else:
                    # Add language label if specified
                    if language:
                        code_content = f"#{language}\n{code_content}"
                    flowables.append(Paragraph(code_content, self.styles['CodeBlock']))

            # Numbered lists
            elif re.match(r'^\d+\.', line):
                list_items = []
                while i < len(lines) and re.match(r'^\d+\.', lines[i].strip()):
                    item_text = re.sub(r'^\d+\.\s*', '', lines[i].strip())
                    item_text = self._process_inline_formatting(item_text)
                    list_items.append(f"• {item_text}")
                    i += 1
                list_content = '<br/>'.join(list_items)
                flowables.append(Paragraph(list_content, self.styles['Normal']))
                continue

            # Bullet lists
            elif line.startswith('- ') or line.startswith('* ') or line.startswith('+ '):
                list_items = []
                while i < len(lines) and (lines[i].strip().startswith(('- ', '* ', '+ '))):
                    item_text = lines[i].strip()[2:]
                    item_text = self._process_inline_formatting(item_text)
                    list_items.append(f"• {item_text}")
                    i += 1
                list_content = '<br/>'.join(list_items)
                flowables.append(Paragraph(list_content, self.styles['Normal']))
                continue

            # Tables (basic support)
            elif '|' in line and i + 1 < len(lines) and '|' in lines[i + 1]:
                table_lines = [line]
                i += 1
                while i < len(lines) and '|' in lines[i].strip():
                    table_lines.append(lines[i])
                    i += 1

                table_data = self._parse_markdown_table(table_lines)
                if table_data:
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 14),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    flowables.append(table)
                    flowables.append(Spacer(1, 12))
                continue

            # Images (process image and its detailed caption as a unit)
            elif line.startswith('!['):
                img_match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
                if img_match:
                    alt_text = img_match.group(1)
                    img_path = img_match.group(2)

                    # Check if this image has already been processed to prevent duplicates
                    if img_path in self.processed_figures:
                        print(f"Skipping duplicate figure: {img_path}")
                        i += 1
                        continue

                    # Mark this image as processed
                    self.processed_figures.add(img_path)

                    # Look for detailed caption immediately following the image
                    detailed_caption = None
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        # Look for pattern like "**Figure X**: Detailed description..."
                        caption_match = re.match(r'^\*\*Figure\s+(\d+):\*\*\s*(.+)$', next_line)
                        if caption_match:
                            detailed_caption = caption_match.group(2).strip()

                    # Try to find image in output directory
                    full_img_path = self._find_image_path(img_path)
                    if full_img_path:
                        try:
                            # Scale image appropriately based on type
                            if full_img_path.suffix.lower() in ['.svg']:
                                # SVG images - use smaller size
                                img = Image(full_img_path, width=4*inch, height=3*inch)
                            elif full_img_path.suffix.lower() in ['.pdf']:
                                # PDF images - use medium size
                                img = Image(full_img_path, width=5*inch, height=3.5*inch)
                            else:
                                # PNG/JPG images - use standard size
                                img = Image(full_img_path, width=5*inch, height=3.5*inch)

                            img.hAlign = 'CENTER'
                            flowables.append(img)

                            # Create figure caption using detailed caption if available
                            if detailed_caption:
                                figure_caption = f"Figure {self.document_figure_counter}: {detailed_caption}"
                            elif alt_text and not alt_text.lower().startswith('figure'):
                                # Use alt text if it's descriptive and doesn't already start with "Figure"
                                figure_caption = f"Figure {self.document_figure_counter}: {alt_text}"
                            else:
                                figure_caption = f"Figure {self.document_figure_counter}"

                            flowables.append(Paragraph(figure_caption, self.styles['FigureCaption']))

                            # Increment document figure counter for next figure
                            self.document_figure_counter += 1

                            # Skip the detailed caption line if it was processed
                            if detailed_caption:
                                i += 1  # Skip the next line (the detailed caption)

                        except Exception as e:
                            print(f"Warning: Could not load image {img_path}: {e}")
                            flowables.append(Paragraph(f"[Image: {alt_text}]", self.styles['Normal']))
                    else:
                        print(f"Warning: Image not found: {img_path}")
                        flowables.append(Paragraph(f"[Image: {alt_text}]", self.styles['Normal']))

            # Horizontal rules
            elif line.startswith('---') or line.startswith('***') or line.startswith('___'):
                # Add a horizontal line
                flowables.append(Spacer(1, 6))
                # We could add a line here if needed
                flowables.append(Spacer(1, 6))

            # Blockquotes
            elif line.startswith('>'):
                quote_lines = []
                while i < len(lines) and lines[i].strip().startswith('>'):
                    quote_line = lines[i].strip()[1:].strip()
                    quote_lines.append(quote_line)
                    i += 1
                quote_content = self._process_inline_formatting(' '.join(quote_lines))
                flowables.append(Paragraph(quote_content, self.styles['Normal']))
                continue

            # Regular paragraphs and other content
            elif line:
                # Accumulate paragraph content
                para_lines = [line]
                i += 1
                while i < len(lines) and lines[i].strip() and not (
                    lines[i].strip().startswith('#') or
                    lines[i].strip().startswith('```') or
                    lines[i].strip().startswith('![') or
                    re.match(r'^\d+\.', lines[i].strip()) or
                    lines[i].strip().startswith(('- ', '* ', '+ ')) or
                    (lines[i].strip().startswith('>')) or
                    (lines[i].strip().startswith('---') or
                     lines[i].strip().startswith('***') or
                     lines[i].strip().startswith('___'))
                ):
                    para_lines.append(lines[i])
                    i += 1

                para_content = ' '.join(para_lines)
                para_content = self._process_inline_formatting(para_content)
                if para_content.strip():
                    flowables.append(Paragraph(para_content, self.styles['Normal']))
                continue

            i += 1

        return flowables

    def _process_inline_formatting(self, text: str) -> str:
        """Process inline Markdown formatting with safe HTML generation"""
        if not text:
            return text

        # Escape HTML characters first
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')

        # Process inline code `code` - replace with simple text markers
        text = re.sub(r'`([^`\n]+)`', r'[\1]', text)

        # Process bold **text** and __text__
        text = re.sub(r'\*\*([^*\n]+)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__([^_\n]+)__', r'<b>\1</b>', text)

        # Process italic *text* and _text_ - very conservative approach
        text = re.sub(r'(?<![\w\*])\*([^*\n,.;:!?]+)\*(?!\w)', r'<i>\1</i>', text)
        text = re.sub(r'(?<![\w_])_([^_\n,.;:!?]+)_(?!\w)', r'<i>\1</i>', text)

        # Process links [text](url) - convert to ReportLab hyperlink format
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" color="red">\1</a>', text)
        
        # Process plain URLs as hyperlinks (only if not already in a link tag)
        text = re.sub(r'(?<!<a href=")(https?://[^\s]+)(?!">)', r'<a href="\1" color="red">\1</a>', text)

        # Process strikethrough ~~text~~
        text = re.sub(r'~~([^~\n]+)~~', r'<strike>\1</strike>', text)
        
        # Remove emoji characters that cause PDF rendering issues
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        return text

    def _parse_markdown_table(self, table_lines: List[str]) -> List[List[str]]:
        """Parse Markdown table into data structure"""
        if len(table_lines) < 2:
            return None

        # Split table lines into cells
        table_data = []
        for line in table_lines:
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    table_data.append(cells)

        return table_data if len(table_data) > 1 else None

    def _truncate_code_block(self, code_content: str, max_lines: int = 15) -> str:
        """Truncate code blocks that are too long for PDF display."""
        lines = code_content.split('\n')

        if len(lines) <= max_lines:
            return code_content

        # Keep first part and add truncation indicator
        truncated_lines = lines[:max_lines - 1]
        truncated_lines.append("    # ... [additional code lines truncated for PDF] ...")

        return '\n'.join(truncated_lines)

    def _get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """Get the dimensions of an image file."""
        try:
            from PIL import Image as PILImage
            with PILImage.open(image_path) as img:
                return img.size  # Returns (width, height)
        except ImportError:
            # Fallback if PIL is not available - use file command
            try:
                import subprocess
                result = subprocess.run(['file', str(image_path)],
                                      capture_output=True, text=True)
                # Parse output like: "PNG image data, 267 x 590, 8-bit/color RGBA, non-interlaced"
                if 'x' in result.stdout:
                    dims = result.stdout.split(',')[1].strip().split('x')
                    if len(dims) == 2:
                        return int(dims[0].strip()), int(dims[1].strip())
            except Exception:
                pass

        # Ultimate fallback - assume square
        return 400, 400

    def _generate_diagram_caption(self, mermaid_content: str, diagram_id: str) -> str:
        """Generate a comprehensive caption for a Mermaid diagram."""
        lines = mermaid_content.strip().split('\n')
        diagram_type = "flowchart"
        key_elements = []

        for line in lines:
            line = line.strip()
            if line.startswith('graph '):
                if 'TD' in line:
                    diagram_type = "top-down flowchart"
                elif 'LR' in line:
                    diagram_type = "left-to-right flowchart"
                elif 'BT' in line:
                    diagram_type = "bottom-up flowchart"
                elif 'RL' in line:
                    diagram_type = "right-to-left flowchart"
            elif line.startswith('sequenceDiagram'):
                diagram_type = "sequence diagram"
            elif line.startswith('flowchart '):
                diagram_type = "modern flowchart"
            elif '[' in line and ']' in line and '-->' in line:
                # Extract node connections
                parts = line.split('-->')
                if len(parts) >= 2:
                    source = parts[0].strip()
                    target = parts[1].strip()
                    if '[' in source and ']' in source:
                        source_node = source[source.find('[')+1:source.find(']')]
                        key_elements.append(f"{source_node}")
            elif 'subgraph' in line.lower():
                diagram_type = "hierarchical diagram with subgraphs"

        # Generate comprehensive caption
        if diagram_type == "sequence diagram":
            caption = f"Figure: Sequence diagram showing the interaction flow and message passing between system components in the Synergetics framework."
        elif "subgraph" in diagram_type:
            caption = f"Figure: Hierarchical diagram illustrating the modular architecture and component relationships within the Synergetics system, organized into logical subgraphs."
        elif key_elements:
            elements_str = ", ".join(key_elements[:3])  # Show first 3 key elements
            if len(key_elements) > 3:
                elements_str += "..."
            caption = f"Figure: {diagram_type.capitalize()} depicting the relationships and data flow between key components including {elements_str}."
        else:
            caption = f"Figure: {diagram_type.capitalize()} illustrating key concepts and relationships in the Synergetics mathematical framework."

        return caption

    def _format_mermaid_content(self, mermaid_content: str) -> str:
        """Format Mermaid diagram content for better readability in text form."""
        # Clean and format the mermaid content for display
        lines = mermaid_content.strip().split('\n')
        formatted_lines = []

        # Add header
        formatted_lines.append("Mermaid Diagram Structure:")
        formatted_lines.append("=" * 30)

        for i, line in enumerate(lines, 1):
            # Clean up the line and add line numbers
            cleaned_line = line.strip()
            if cleaned_line:  # Only add non-empty lines
                formatted_line = f"{i:2d}: {cleaned_line}"
                formatted_lines.append(formatted_line)

        formatted_lines.append("")
        formatted_lines.append("Key Components:")
        formatted_lines.append("-" * 15)

        # Extract and explain key components
        components = []
        for line in lines:
            line = line.strip()
            if line.startswith('graph '):
                components.append(f"• Graph type: {line.split()[1]}")
            elif '-->' in line or '---' in line:
                components.append("• Connection/relationship arrow")
            elif '[' in line and ']' in line:
                # Extract node labels
                start = line.find('[')
                end = line.find(']')
                if start != -1 and end != -1:
                    label = line[start+1:end]
                    components.append(f"• Node: {label}")

        if components:
            formatted_lines.extend(components)
        else:
            formatted_lines.append("• Diagram structure analysis available")

        return '\n'.join(formatted_lines)

    def _render_mermaid_diagram(self, mermaid_content: str, diagram_id: str) -> Optional[str]:
        """
        Render Mermaid diagram as PNG image using Mermaid CLI.

        Args:
            mermaid_content: The Mermaid diagram source code
            diagram_id: Unique identifier for the diagram

        Returns:
            Path to the rendered PNG image, or None if rendering failed
        """
        try:
            import subprocess
            import tempfile
            import os

            # Create temporary file for Mermaid source
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as temp_file:
                temp_file.write(mermaid_content)
                temp_file_path = temp_file.name

            # Define output path for PNG
            output_dir = self.paper_dir / "mermaid_images"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{diagram_id}.png"

            # Render diagram using Mermaid CLI with higher resolution
            cmd = [
                'mmdc',
                '-i', temp_file_path,
                '-o', str(output_path),
                '-t', 'default',
                '-b', 'transparent',
                '-s', '2.0',  # Scale factor for higher resolution
                '-w', '1200',  # Width in pixels
                '-H', '800'   # Height in pixels (will be adjusted by aspect ratio)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Clean up temporary file
            os.unlink(temp_file_path)

            if result.returncode == 0 and output_path.exists():
                # Return the path relative to paper directory for proper inclusion
                return str(output_path.relative_to(self.paper_dir))
            else:
                print(f"Warning: Mermaid CLI failed for diagram {diagram_id}")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}...")  # Truncate long error messages
                return None

        except Exception as e:
            print(f"Warning: Failed to render Mermaid diagram {diagram_id}: {e}")
            return None

    def _find_image_path(self, img_path: str) -> Optional[Path]:
        """Find image file in output directory and project directories"""
        # Remove any leading path components
        img_filename = Path(img_path).name

        # Search locations in order of priority
        search_paths = [
            # Main output directory
            self.paper_dir.parent / 'output',
            # Paper output directory
            self.paper_dir / 'output',
            # Mermaid images directory (for generated diagrams)
            self.paper_dir / 'mermaid_images',
            # Docs directory (for documentation images)
            self.paper_dir.parent / 'docs',
            # Project root
            self.paper_dir.parent
        ]

        for search_path in search_paths:
            if search_path.exists():
                # Search recursively in each path
                for pattern in ['**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.svg', '**/*.pdf']:
                    try:
                        matches = list(search_path.glob(pattern))
                        for match in matches:
                            if match.name == img_filename:
                                return match
                    except Exception:
                        continue

        return None

    def create_front_matter(self) -> List[Flowable]:
        """Create scientific front matter"""
        flowables = []

        # Title
        flowables.append(Paragraph(self.metadata['title'], self.styles['PaperTitle']))

        # Author information
        author_info = f"""
        <b>{self.metadata['author']}</b><br/>
        {self.metadata['institution']}<br/>
        {self.metadata['affiliation']}<br/>
        Email: {self.metadata['email']}<br/>
        ORCID: {self.metadata['orcid']}<br/>
        {self.metadata['date']}
        """
        flowables.append(Paragraph(author_info, self.styles['AuthorInfo']))

        # Add some space
        flowables.append(Spacer(1, 30))

        return flowables

    def create_title_page(self, content: str) -> List[Flowable]:
        """Create a proper title page from the title section content"""
        flowables = []
        
        # Parse the content to extract title page elements
        lines = content.split('\n')
        title = ""
        author_info = []
        keywords = []
        repository = ""
        date = ""
        version = ""
        quote = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                title = line[2:].strip()
            elif line == "## Title Page":
                current_section = "title_page"
            elif line == "## Front Matter":
                current_section = "front_matter"
            elif line.startswith('**Author:**') and current_section == "title_page":
                continue  # Skip the label
            elif line.startswith('**Affiliation:**') and current_section == "title_page":
                continue  # Skip the label
            elif line.startswith('**Keywords:**') and current_section == "title_page":
                continue  # Skip the label
            elif line.startswith('**Repository:**') and current_section == "title_page":
                continue  # Skip the label
            elif line.startswith('**Date:**') and current_section == "title_page":
                date = line[8:].strip()
            elif line.startswith('**Version:**') and current_section == "title_page":
                version = line[10:].strip()
            elif line.startswith('**') and line.endswith('**') and current_section == "title_page":
                # Skip bold labels
                continue
            elif line.startswith('---') and current_section == "title_page":
                continue  # Skip separators
            elif line.startswith('"') and line.endswith('"') and current_section == "title_page":
                quote = line.strip('"')
            elif line.startswith('—') and current_section == "title_page":
                # Skip attribution line
                continue
            elif line and current_section == "title_page":
                if not author_info:
                    # First non-empty line after author label
                    author_info.append(line)
                elif len(author_info) == 1:
                    # Second line (email)
                    author_info.append(line)
                elif len(author_info) == 2:
                    # Third line (ORCID)
                    author_info.append(line)
                elif len(author_info) == 3:
                    # Fourth line (affiliation)
                    author_info.append(line)
                elif not keywords:
                    # Keywords line
                    keywords = [kw.strip() for kw in line.split(',')]
                elif not repository:
                    # Repository line
                    repository = line
        
        # Create title page content
        if title:
            flowables.append(Paragraph(title, self.styles['PaperTitle']))
            flowables.append(Spacer(1, 40))
        
        # Author information
        if author_info:
            author_text = "<br/>".join(author_info)
            flowables.append(Paragraph(author_text, self.styles['AuthorInfo']))
            flowables.append(Spacer(1, 30))
        
        # Keywords
        if keywords:
            keywords_text = f"<b>Keywords:</b> {', '.join(keywords)}"
            flowables.append(Paragraph(keywords_text, self.styles['BodyText']))
            flowables.append(Spacer(1, 20))
        
        # Repository
        if repository:
            flowables.append(Paragraph(f"<b>Repository:</b> {repository}", self.styles['BodyText']))
            flowables.append(Spacer(1, 20))
        
        # Date and version
        if date or version:
            meta_text = []
            if date:
                meta_text.append(f"<b>Date:</b> {date}")
            if version:
                meta_text.append(f"<b>Version:</b> {version}")
            flowables.append(Paragraph(" | ".join(meta_text), self.styles['BodyText']))
            flowables.append(Spacer(1, 40))
        
        # Quote
        if quote:
            flowables.append(Paragraph(f'"{quote}"', self.styles['Quote']))
            flowables.append(Spacer(1, 20))
            flowables.append(Paragraph("— Buckminster Fuller", self.styles['QuoteAuthor']))
        
        return flowables

    def generate_pdf(self, output_filename: str = None) -> str:
        """Generate complete PDF from all sections"""
        if not output_filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"synergetics_paper_{timestamp}.pdf"

        output_path = self.output_dir / output_filename

        # Create PDF document with hyperlink support
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
            title=self.metadata['title'],
            author=self.metadata['author']
        )

        # Build document content
        story = []

        # Load and process sections (includes title/front matter from markdown)
        sections = self.load_sections()

        for section in sections:
            print(f"Processing section: {section.title}")

            # Add page break before new sections (except first)
            if story:
                story.append(PageBreak())

            # Special handling for title section
            if section.filename == "00_title.md":
                section_flowables = self.create_title_page(section.content)
            else:
                # Process section content normally
                section_flowables = self.process_markdown_content(section.content)
            
            story.extend(section_flowables)

        # Generate PDF
        print(f"Generating PDF: {output_path}")
        doc.build(story)

        return str(output_path)


def main():
    """Main entry point for PDF generation"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate PDF from Synergetics paper markdown')
    parser.add_argument('--paper-dir', default='paper',
                       help='Paper directory containing markdown sections')
    parser.add_argument('--output', help='Output PDF filename')
    parser.add_argument('--output-dir', help='Output directory for PDF')

    args = parser.parse_args()

    # Initialize renderer
    renderer = ScientificPaperRenderer(args.paper_dir, args.output_dir)

    # Generate PDF
    output_path = renderer.generate_pdf(args.output)

    print(f"PDF generated successfully: {output_path}")


if __name__ == '__main__':
    main()
