#!/usr/bin/env python3
"""
Quick script to run the Synergetics paper PDF renderer

This script provides a simple interface to generate the PDF with default settings.

Usage:
    python run_render.py                    # Generate with default settings
    python run_render.py --output custom.pdf  # Custom output filename
    python run_render.py --help             # Show help
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pdf_renderer import ScientificPaperRenderer


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate PDF from Synergetics paper markdown sections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_render.py
  python run_render.py --output my_paper.pdf
  python run_render.py --paper-dir ../custom_paper --output-dir ../output
        """
    )

    parser.add_argument(
        '--paper-dir',
        default='../paper',
        help='Directory containing paper markdown and render folders (default: ../paper)'
    )

    parser.add_argument(
        '--output',
        help='Output PDF filename (default: auto-generated with timestamp)'
    )

    parser.add_argument(
        '--output-dir',
        help='Directory to save PDF (default: paper/output)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output during generation'
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    if args.paper_dir == '../paper':
        paper_dir = script_dir.parent
    else:
        paper_dir = Path(args.paper_dir)

    # Verify paper directory exists
    if not paper_dir.exists():
        print(f"Error: Paper directory not found: {paper_dir}")
        sys.exit(1)

    markdown_dir = paper_dir / "markdown"
    if not markdown_dir.exists():
        print(f"Error: Markdown directory not found: {markdown_dir}")
        sys.exit(1)

    # Check for markdown files
    md_files = list(markdown_dir.glob("*.md"))
    if not md_files:
        print(f"Error: No markdown files found in: {markdown_dir}")
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(md_files)} markdown files:")
        for md_file in sorted(md_files):
            print(f"  - {md_file.name}")

    # Initialize renderer
    try:
        renderer = ScientificPaperRenderer(
            paper_dir=str(paper_dir),
            output_dir=args.output_dir
        )

        if args.verbose:
            print("Renderer initialized successfully")
            print(f"Output directory: {renderer.output_dir}")

    except Exception as e:
        print(f"Error initializing renderer: {e}")
        sys.exit(1)

    # Generate PDF
    try:
        if args.verbose:
            print("Starting PDF generation...")

        output_path = renderer.generate_pdf(args.output)

        print("✓ PDF generated successfully!")
        print(f"  Output: {output_path}")

        # Check file size
        pdf_file = Path(output_path)
        if pdf_file.exists():
            size_mb = pdf_file.stat().st_size / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")
        if args.verbose:
            print("\nNext steps:")
            print("  - Review the generated PDF")
            print("  - Check image quality and positioning")
            print("  - Verify all sections are included")
            print("  - Test printing if needed")

    except Exception as e:
        print(f"Error generating PDF: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
