#!/usr/bin/env python3
"""
Figure Scanner and Registration Script

This script scans the output directory for visualization files and automatically
registers them with the figure manager to create comprehensive captions and
references for the paper.

Author: Daniel Ari Friedman
Email: daniel@activeinference.institute
ORCID: 0000-0001-6232-9096
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import re

# Add parent directory to path to import figure_manager
sys.path.insert(0, str(Path(__file__).parent))

from figure_manager import figure_manager, register_figure_auto


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from filename"""
    metadata = {}

    # Extract backend information
    if '_matplotlib' in filename or '3d.png' in filename:
        metadata['backend'] = 'matplotlib'
        metadata['generated_by'] = 'matplotlib'
    elif '_plotly' in filename or '3d.html' in filename:
        metadata['backend'] = 'plotly'
        metadata['generated_by'] = 'plotly'
    elif '_ascii' in filename or '.txt' in filename:
        metadata['backend'] = 'ascii'
        metadata['generated_by'] = 'ascii_art'
    elif '_seaborn' in filename:
        metadata['backend'] = 'seaborn'
        metadata['generated_by'] = 'seaborn'
    else:
        metadata['backend'] = 'matplotlib'  # default
        metadata['generated_by'] = 'matplotlib'

    # Extract resolution if available
    if 'high_res' in filename or '300dpi' in filename:
        metadata['resolution'] = '300 DPI'
    elif 'low_res' in filename:
        metadata['resolution'] = '150 DPI'
    else:
        metadata['resolution'] = '300 DPI'  # default

    # Extract data size information
    if 'large' in filename:
        metadata['data_scale'] = 'large_dataset'
    elif 'small' in filename:
        metadata['data_scale'] = 'small_dataset'
    else:
        metadata['data_scale'] = 'standard'

    return metadata


def scan_output_directory(output_dir: Path) -> List[Dict[str, Any]]:
    """Scan output directory for visualization files"""
    visualization_files = []

    # File extensions to include
    extensions = ['*.png', '*.svg', '*.html', '*.txt']

    for ext in extensions:
        for file_path in output_dir.rglob(ext):
            if file_path.is_file():
                # Skip certain directories/files
                if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'node_modules']):
                    continue

                # Get relative path from output directory
                rel_path = file_path.relative_to(output_dir)

                # Extract metadata
                metadata = extract_metadata_from_filename(str(file_path))

                file_info = {
                    'full_path': file_path,
                    'relative_path': rel_path,
                    'filename': file_path.name,
                    'extension': file_path.suffix,
                    'size_bytes': file_path.stat().st_size,
                    'metadata': metadata
                }

                visualization_files.append(file_info)

    return sorted(visualization_files, key=lambda x: x['relative_path'])


def register_visualizations(visualization_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Register all visualizations with the figure manager"""
    registered_figures = []

    for file_info in visualization_files:
        try:
            # Register the figure
            figure_metadata = register_figure_auto(
                filename=str(file_info['relative_path']),
                metadata=file_info['metadata']
            )

            registered_info = {
                'figure_number': figure_metadata.number,
                'filename': file_info['filename'],
                'category': figure_metadata.category,
                'subcategory': figure_metadata.subcategory,
                'title': figure_metadata.title,
                'description': figure_metadata.description,
                'file_size': file_info['size_bytes'],
                'metadata': file_info['metadata']
            }

            registered_figures.append(registered_info)
            print(f"✅ Registered Figure {figure_metadata.number}: {figure_metadata.title}")

        except Exception as e:
            print(f"❌ Failed to register {file_info['filename']}: {e}")

    return registered_figures


def generate_figure_references(registered_figures: List[Dict[str, Any]]) -> str:
    """Generate markdown content with figure references"""
    content = "# Figures and Visualizations\n\n"
    content += "This section contains all figures referenced in the paper.\n\n"

    # Group by category
    categories = {}
    for fig in registered_figures:
        cat = fig['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(fig)

    for category, figures in categories.items():
        content += f"## {category.title()} Visualizations\n\n"

        for fig in sorted(figures, key=lambda x: x['figure_number']):
            content += f"### Figure {fig['figure_number']}: {fig['title']}\n\n"
            content += f"**File**: `{fig['filename']}`\n\n"
            content += f"**Description**: {fig['description']}\n\n"
            content += f"**Category**: {fig['category']}/{fig['subcategory']}\n\n"
            content += f"**File Size**: {fig['file_size']:,} bytes\n\n"

            # Add technical details
            if fig['metadata']:
                content += "**Technical Details**:\n"
                for key, value in fig['metadata'].items():
                    content += f"- {key}: {value}\n"
                content += "\n"

            # Add figure reference
            content += "**Figure Reference**:\n"
            content += "```markdown\n"
            content += f"![Figure {fig['figure_number']}: {fig['title']}](output/{fig['filename']})\n\n"
            content += f"**Figure {fig['figure_number']}**: {fig['description']}\n"
            content += "```\n\n"

            content += "---\n\n"

    return content


def main():
    """Main function to scan and register figures"""
    print("🔍 Scanning output directory for visualizations...")

    # Determine project root and output directory
    script_dir = Path(__file__).parent
    if script_dir.name == 'render':
        project_root = script_dir.parent.parent
    else:
        project_root = script_dir.parent

    output_dir = project_root / "output"

    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return False

    # Scan for visualization files
    visualization_files = scan_output_directory(output_dir)
    print(f"📁 Found {len(visualization_files)} visualization files")

    if not visualization_files:
        print("⚠️ No visualization files found")
        return False

    # Register all visualizations
    print("\n📝 Registering figures...")
    registered_figures = register_visualizations(visualization_files)

    if not registered_figures:
        print("❌ No figures were successfully registered")
        return False

    print(f"\n✅ Successfully registered {len(registered_figures)} figures")

    # Save figure registry
    registry_path = figure_manager.save_figure_registry()
    print(f"💾 Figure registry saved to: {registry_path}")

    # Generate figure references markdown
    figure_content = generate_figure_references(registered_figures)

    # Save to markdown file
    figures_md_path = project_root / "paper" / "markdown" / "figures.md"
    with open(figures_md_path, 'w') as f:
        f.write(figure_content)

    print(f"📄 Figure references saved to: {figures_md_path}")

    # Print summary
    print("\n" + "="*60)
    print("FIGURE REGISTRATION SUMMARY")
    print("="*60)

    print(f"Total visualization files found: {len(visualization_files)}")
    print(f"Successfully registered figures: {len(registered_figures)}")

    # Category breakdown
    categories = {}
    for fig in registered_figures:
        cat = fig['category']
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1

    print("\nBreakdown by category:")
    for category, count in categories.items():
        print(f"  {category.title()}: {count} figures")

    print(f"\nFigure numbers range: 1 - {len(registered_figures)}")
    print("="*60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
