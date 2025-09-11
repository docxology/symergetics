#!/usr/bin/env python3
"""
Mermaid Diagram Generation Script for Symergetics Paper

This script generates Mermaid diagrams for the paper by processing markdown files
and converting any Mermaid blocks to PNG images.
"""

import os
import sys
import subprocess
from pathlib import Path
import re
import tempfile

def extract_mermaid_blocks(markdown_file):
    """Extract Mermaid blocks from a markdown file."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all Mermaid blocks
    mermaid_pattern = r'```mermaid\s*\n(.*?)\n```'
    blocks = re.findall(mermaid_pattern, content, re.DOTALL)

    return blocks

def generate_mermaid_png(mermaid_code, output_path, title="Diagram"):
    """Generate PNG from Mermaid code using mermaid-cli if available."""
    try:
        # Try using mermaid-cli if available
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            f.write(mermaid_code)
            temp_file = f.name

        # Try mermaid-cli
        result = subprocess.run([
            'mmdc', '-i', temp_file, '-o', str(output_path), '-t', 'dark', '-b', 'transparent'
        ], capture_output=True, text=True)

        # Clean up temp file
        os.unlink(temp_file)

        if result.returncode == 0:
            print(f"✅ Generated Mermaid diagram: {output_path}")
            return True
        else:
            print(f"⚠️ mermaid-cli not available or failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("⚠️ mermaid-cli (mmdc) not found, skipping Mermaid generation")
        return False
    except Exception as e:
        print(f"⚠️ Error generating Mermaid diagram: {e}")
        return False

def process_markdown_files():
    """Process all markdown files and generate Mermaid diagrams."""
    paper_dir = Path(__file__).parent.parent
    markdown_dir = paper_dir / "markdown"
    mermaid_dir = paper_dir / "mermaid_images"

    # Create mermaid_images directory
    mermaid_dir.mkdir(exist_ok=True)

    if not markdown_dir.exists():
        print("❌ Markdown directory not found")
        return False

    total_diagrams = 0
    successful_diagrams = 0

    # Process each markdown file
    for md_file in markdown_dir.glob("*.md"):
        print(f"Processing {md_file.name}...")

        blocks = extract_mermaid_blocks(md_file)

        if not blocks:
            print(f"  No Mermaid blocks found in {md_file.name}")
            continue

        print(f"  Found {len(blocks)} Mermaid blocks in {md_file.name}")

        # Generate PNG for each block
        for i, block in enumerate(blocks):
            output_name = f"{md_file.stem}_diagram_{i+1}.png"
            output_path = mermaid_dir / output_name

            if generate_mermaid_png(block, output_path, f"{md_file.stem} Diagram {i+1}"):
                successful_diagrams += 1

            total_diagrams += 1

    if total_diagrams > 0:
        print(f"\n📊 Mermaid Generation Summary:")
        print(f"   Total diagrams found: {total_diagrams}")
        print(f"   Successfully generated: {successful_diagrams}")
        print(f"   Output directory: {mermaid_dir}")

        if successful_diagrams == total_diagrams:
            print("✅ All Mermaid diagrams generated successfully")
            return True
        else:
            print("⚠️ Some Mermaid diagrams could not be generated")
            return True  # Don't fail the whole process
    else:
        print("ℹ️ No Mermaid blocks found in any markdown files")
        print("✅ Mermaid diagram check completed (no diagrams to generate)")
        return True

def main():
    """Main function for Mermaid diagram generation."""
    print("🧜 Generating Mermaid diagrams for Symergetics paper...")
    print("="*60)

    success = process_markdown_files()

    if success:
        print("\n✅ Mermaid diagram generation completed")
    else:
        print("\n❌ Mermaid diagram generation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
