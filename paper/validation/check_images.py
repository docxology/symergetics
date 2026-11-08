#!/usr/bin/env python3
"""
Image Integration Checker

This script specifically validates image integration and references
in the Symergetics paper markdown sections.

Author: Daniel Ari Friedman
Email: daniel@activeinference.institute
ORCID: 0000-0001-6232-9096
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import argparse


class ImageChecker:
    """Validates image integration and references"""
    
    def __init__(self, paper_dir: str = "paper"):
        self.paper_dir = Path(paper_dir)
        self.markdown_dir = self.paper_dir / "markdown"
        self.output_dir = self.paper_dir.parent / "output"
        self.issues: List[Dict] = []
        
    def check_all(self) -> List[Dict]:
        """Run all image-related checks"""
        print("🖼️  Checking image integration...")
        print("=" * 40)
        
        self.issues = []
        
        # Run all checks
        self._check_image_references()
        self._check_image_existence()
        self._check_figure_captions()
        self._check_image_formatting()
        self._check_duplicate_references()
        self._check_path_consistency()
        
        return self.issues
    
    def _check_image_references(self):
        """Check image reference format"""
        print("📋 Checking image reference format...")
        
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                matches = re.findall(image_pattern, line)
                for alt_text, img_path in matches:
                    # Check alt text format
                    if not alt_text.startswith('Figure '):
                        self.issues.append({
                            'type': 'alt_text_format',
                            'severity': 'warning',
                            'file': str(md_file),
                            'line': i,
                            'message': f"Alt text should start with 'Figure X:': '{alt_text}'",
                            'suggestion': f"Change to: 'Figure X: {alt_text}'"
                        })
                    
                    # Check if alt text has descriptive title
                    if alt_text.startswith('Figure ') and ':' not in alt_text:
                        self.issues.append({
                            'type': 'alt_text_descriptive',
                            'severity': 'warning',
                            'file': str(md_file),
                            'line': i,
                            'message': f"Alt text should include descriptive title: '{alt_text}'",
                            'suggestion': f"Change to: '{alt_text}: Descriptive Title'"
                        })
    
    def _check_image_existence(self):
        """Check if referenced images exist"""
        print("🔍 Checking image existence...")
        
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        missing_images = []
        
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                matches = re.findall(image_pattern, line)
                for alt_text, img_path in matches:
                    full_path = (md_file.parent / img_path).resolve()
                    if not full_path.is_file():
                        missing_images.append({
                            'file': str(md_file),
                            'line': i,
                            'path': img_path,
                            'full_path': str(full_path)
                        })
        
        for missing in missing_images:
            self.issues.append({
                'type': 'image_missing',
                'severity': 'error',
                'file': missing['file'],
                'line': missing['line'],
                'message': f"Image file not found: {missing['path']}",
                'suggestion': f"Check if file exists at: {missing['full_path']}"
            })
    
    def _check_figure_captions(self):
        """Check a nonempty caption after optional separating blank lines."""
        print("📝 Checking figure captions...")
        for md_file in self.markdown_dir.glob("*.md"):
            lines = md_file.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if not line.startswith("!["):
                    continue
                caption_index = index + 1
                while caption_index < len(lines) and not lines[caption_index].strip():
                    caption_index += 1
                caption = lines[caption_index].strip() if caption_index < len(lines) else ""
                if not re.match(r"\*\*Figure \d+(?:\*\*:|:\*\*) .+", caption):
                    self.issues.append({
                        "type": "caption_format" if caption.startswith("**Figure") else "caption_missing",
                        "severity": "error", "file": str(md_file), "line": index + 1,
                        "message": "Missing or malformed detailed figure caption",
                        "suggestion": "Use '**Figure N**: Description' after optional blank lines",
                    })
    
    def _check_image_formatting(self):
        """Check image formatting and styling"""
        print("🎨 Checking image formatting...")
        
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                matches = re.findall(image_pattern, line)
                for alt_text, img_path in matches:
                    try:
                        if Path(img_path).is_absolute():
                            raise ValueError("absolute image path")
                        (md_file.parent / img_path).resolve().relative_to(self.output_dir.resolve())
                    except ValueError:
                        self.issues.append({
                            "type": "path_format", "severity": "error",
                            "file": str(md_file), "line": i,
                            "message": f"Image path must resolve from Markdown into project output/: {img_path}",
                            "suggestion": "Use a source-relative path such as ../../output/category/subcategory/figure.png",
                        })
                    
                    # Check file extension
                    if not img_path.lower().endswith(('.png', '.svg', '.pdf', '.jpg', '.jpeg')):
                        self.issues.append({
                            'type': 'file_extension',
                            'severity': 'warning',
                            'file': str(md_file),
                            'line': i,
                            'message': f"Unsupported file extension: '{img_path}'",
                            'suggestion': "Use .png, .svg, .pdf, .jpg, or .jpeg"
                        })
    
    def _check_duplicate_references(self):
        """Check for duplicate image references"""
        print("🔄 Checking for duplicate references...")
        
        image_refs = {}
        
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                if line.startswith('!['):
                    if line in image_refs:
                        self.issues.append({
                            'type': 'duplicate_reference',
                            'severity': 'warning',
                            'file': str(md_file),
                            'line': i,
                            'message': f"Duplicate image reference: '{line}'",
                            'suggestion': "Consider if this duplication is intentional"
                        })
                    else:
                        image_refs[line] = (str(md_file), i)
    
    def _check_path_consistency(self):
        """Check category/subcategory organization after source-path resolution."""
        print("📁 Checking path consistency...")
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content):
                try:
                    relative = (md_file.parent / path).resolve().relative_to(self.output_dir.resolve())
                except ValueError:
                    continue  # Already reported as an invalid source/output path.
                if len(relative.parts) < 3:
                    self.issues.append({
                        "type": "path_structure", "severity": "warning",
                        "file": str(md_file), "line": 0,
                        "message": f"Image is not organized by category/subcategory: {path}",
                        "suggestion": "Keep generated images under output/category/subcategory/",
                    })
    
    def generate_report(self) -> str:
        """Generate image check report"""
        report = []
        report.append("# Image Integration Check Report")
        report.append("")
        report.append(f"**Paper Directory:** {self.paper_dir}")
        report.append(f"**Output Directory:** {self.output_dir}")
        report.append(f"**Total Issues:** {len(self.issues)}")
        report.append("")
        
        # Group by severity
        errors = [i for i in self.issues if i['severity'] == 'error']
        warnings = [i for i in self.issues if i['severity'] == 'warning']
        
        report.append(f"**Errors:** {len(errors)}")
        report.append(f"**Warnings:** {len(warnings)}")
        report.append("")
        
        # Summary
        if not errors and not warnings:
            report.append("✅ **All image checks passed!**")
        elif not errors:
            report.append("⚠️ **Warnings found, but no errors.**")
        else:
            report.append("❌ **Errors found that need to be fixed.**")
        
        report.append("")
        
        # Detailed issues
        if errors:
            report.append("## Errors")
            report.append("")
            for issue in errors:
                report.append(f"### {issue['type'].replace('_', ' ').title()}")
                report.append(f"- **File:** {issue['file']}")
                if issue['line'] > 0:
                    report.append(f"- **Line:** {issue['line']}")
                report.append(f"- **Message:** {issue['message']}")
                if 'suggestion' in issue:
                    report.append(f"- **Suggestion:** {issue['suggestion']}")
                report.append("")
        
        if warnings:
            report.append("## Warnings")
            report.append("")
            for issue in warnings:
                report.append(f"### {issue['type'].replace('_', ' ').title()}")
                report.append(f"- **File:** {issue['file']}")
                if issue['line'] > 0:
                    report.append(f"- **Line:** {issue['line']}")
                report.append(f"- **Message:** {issue['message']}")
                if 'suggestion' in issue:
                    report.append(f"- **Suggestion:** {issue['suggestion']}")
                report.append("")
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "image_check_report.md"):
        """Save image check report to file"""
        report = self.generate_report()
        report_path = self.paper_dir / filename
        report_path.write_text(report, encoding='utf-8')
        print(f"📄 Image check report saved to: {report_path}")
        return report_path


def main():
    """Main image check function"""
    parser = argparse.ArgumentParser(description='Check image integration in Symergetics paper')
    parser.add_argument('--paper-dir', default='paper',
                       help='Paper directory path')
    parser.add_argument('--output', default='image_check_report.md',
                       help='Output report filename')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = ImageChecker(args.paper_dir)
    
    # Run checks
    issues = checker.check_all()
    
    # Generate and save report
    report_path = checker.save_report(args.output)
    
    # Print summary
    errors = len([i for i in issues if i['severity'] == 'error'])
    warnings = len([i for i in issues if i['severity'] == 'warning'])
    
    print("\n" + "=" * 40)
    print("IMAGE CHECK SUMMARY")
    print("=" * 40)
    print(f"⚠️  Warnings: {warnings}")
    print(f"❌ Errors: {errors}")
    
    if errors > 0:
        print("\n❌ Image check failed with errors!")
        sys.exit(1)
    elif warnings > 0:
        print("\n⚠️  Image check passed with warnings.")
        sys.exit(0)
    else:
        print("\n✅ All image checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
