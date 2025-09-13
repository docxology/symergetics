# Synergetics Paper PDF Renderer

A comprehensive PDF generation system for the Synergetics research paper, providing scientific formatting with integrated visualizations from the `output/` directory.

## Overview

This rendering system converts modular markdown sections into a professionally formatted PDF with:

- **Scientific front matter** with author information and ORCID
- **Proper academic formatting** following research paper conventions
- **Integrated visualizations** from the `output/` directory
- **Code syntax highlighting** and proper mathematical notation
- **Table of contents** and cross-references
- **High-quality image handling** with automatic scaling

## Features

### Document Structure
- Modular markdown sections with automatic ordering
- Scientific front matter with complete author metadata
- Proper section numbering and cross-references
- Automatic table of contents generation

### Visual Integration
- Automatic discovery and inclusion of images from `output/`
- Support for PNG, SVG, PDF, and JPEG formats
- Intelligent image scaling and positioning
- Figure captions and numbering

### Code and Mathematics
- Syntax highlighting for code blocks
- Proper formatting for mathematical expressions
- Support for LaTeX math (when integrated)
- Code block styling with background and borders

### Output Quality
- High-resolution PDF generation (PDF 1.5+)
- Embedded fonts for consistent rendering
- Optimized for print and digital viewing
- Compression options for smaller file sizes

## Installation

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Optional: Install LaTeX for advanced math rendering
# sudo apt-get install texlive-latex-extra  # Ubuntu/Debian
# brew install mactex                         # macOS
```

### Setup

1. Ensure the paper structure exists:
   ```
   paper/
   ├── markdown/     # Markdown sections
   ├── render/       # This rendering system
   └── output/       # Generated PDFs
   ```

2. Verify markdown sections are present in `paper/markdown/`

3. Check that visualizations exist in the main `output/` directory

## Usage

### Basic PDF Generation

```bash
# From the paper/render directory
python pdf_renderer.py

# Or specify custom paths
python pdf_renderer.py --paper-dir ../paper --output my_paper.pdf
```

### Command Line Options

```bash
python pdf_renderer.py [options]

Options:
  --paper-dir DIR      Paper directory containing markdown sections (default: paper)
  --output FILE        Output PDF filename (default: auto-generated)
  --output-dir DIR     Output directory for PDF (default: paper/output)
  --help               Show help message
```

### Configuration

Edit `config.yaml` to customize:

- Author information and metadata
- PDF formatting and styling
- Image handling preferences
- Section ordering and TOC settings
- Quality assurance options

## File Structure

```
paper/render/
├── __init__.py           # Package initialization
├── pdf_renderer.py       # Main PDF generation engine
├── config.yaml          # Configuration settings
├── test_renderer.py     # Test suite
├── requirements.txt     # Python dependencies
└── README.md           # This documentation
```

## Markdown Format

### Supported Elements

#### Headers
```markdown
# Main Title (handled by front matter)
## Section Header
### Subsection Header
#### Sub-subsection Header
```

#### Text Formatting
```markdown
*Italic text*
**Bold text**
`Inline code`
```

#### Code Blocks
```markdown
```python
def example_function():
    return "Hello, Synergetics!"
```
```

#### Lists
```markdown
- Item 1
- Item 2
  - Nested item
```

#### Images
```markdown
![Figure 1: Tetrahedron visualization](output/geometric/polyhedra/tetrahedron_3d.png)
```

#### Links and References
```markdown
[Link text](url)
[@citation_key]
```

### Image Integration

Images are automatically discovered from the `output/` directory. Use relative paths in markdown:

```markdown
![Figure 1: IVM Lattice](output/geometric/lattice/ivm_lattice_size_3.png)
![Mathematical Pattern](output/mathematical/pattern_discovery/pattern_analysis.png)
```

## Configuration Options

### Metadata Configuration

```yaml
metadata:
  title: "Paper Title"
  author: "Daniel Ari Friedman"
  email: "daniel@activeinference.institute"
  orcid: "0000-0001-6232-9096"
  affiliation: "Active Inference Institute"
```

### PDF Formatting

```yaml
pdf:
  pagesize: "A4"
  margins:
    left: 72
    right: 72
    top: 72
    bottom: 72
```

### Image Settings

```yaml
images:
  max_width: 6.0    # inches
  max_height: 4.0   # inches
  dpi: 300
  formats: ["png", "jpg", "svg", "pdf"]
```

## Testing

### Run Test Suite

```bash
# Run all tests
python -m pytest test_renderer.py -v

# Run with coverage
python -m pytest test_renderer.py --cov=pdf_renderer --cov-report=html
```

### Test Coverage

The test suite covers:
- PDF generation pipeline
- Markdown processing
- Image integration
- Style configuration
- Error handling
- Integration testing

## Troubleshooting

### Common Issues

#### Missing Images
- Ensure `output/` directory exists with visualizations
- Check image file formats are supported
- Verify relative paths in markdown are correct

#### Import Errors
```bash
pip install -r requirements.txt
```

#### PDF Generation Errors
- Check write permissions in output directory
- Ensure sufficient disk space
- Verify markdown files are properly formatted

#### Font Issues
- Install additional fonts if needed
- Check font embedding settings in config
- Ensure system fonts are available

### Debug Mode

Enable debug logging:

```python
# In pdf_renderer.py or via config
development:
  debug_mode: true
  log_level: "DEBUG"
```

## Integration with Build System

### Automated PDF Generation

Integrate with the main project build:

```bash
# Add to run.py or CI/CD pipeline
python -m paper.render.pdf_renderer --output symergetics_paper.pdf
```

### Continuous Integration

Example GitHub Actions workflow:

```yaml
- name: Generate Paper PDF
  run: |
    cd paper/render
    python pdf_renderer.py --output ${{ github.sha }}_paper.pdf
```

## Contributing

### Code Style

```bash
# Format code
black *.py

# Lint code
ruff *.py

# Type checking
mypy *.py
```

### Adding Features

1. Add functionality to `pdf_renderer.py`
2. Update configuration in `config.yaml`
3. Add tests to `test_renderer.py`
4. Update documentation in `README.md`

## License

This rendering system is part of the Synergetics project and follows the same MIT license.

## Support

For issues and questions:
- Check the test suite for examples
- Review configuration options
- Examine the main Synergetics documentation

---

**Author:** Daniel Ari Friedman
**Email:** daniel@activeinference.institute
**ORCID:** 0000-0001-6232-9096

