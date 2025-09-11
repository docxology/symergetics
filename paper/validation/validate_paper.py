#!/usr/bin/env python3
"""
Paper Validation Script

This script validates the Symergetics paper for proper formatting, image integration,
and adherence to style guidelines.

Author: Daniel Ari Friedman
Email: daniel@activeinference.institute
ORCID: 0000-0001-6232-9096
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import json
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    passed: bool
    message: str
    severity: str  # 'error', 'warning', 'info'
    file_path: Optional[str] = None
    line_number: Optional[int] = None


class PaperValidator:
    """Validates paper content against style guidelines"""
    
    def __init__(self, paper_dir: str = "paper"):
        self.paper_dir = Path(paper_dir)
        self.markdown_dir = self.paper_dir / "markdown"
        self.output_dir = self.paper_dir.parent / "output"
        self.results: List[ValidationResult] = []
        
    def validate_all(self) -> List[ValidationResult]:
        """Run all validation checks"""
        print("🔍 Validating Symergetics Paper...")
        print("=" * 50)
        
        # Clear previous results
        self.results = []
        
        # Run all validation checks
        self._validate_markdown_structure()
        self._validate_image_references()
        self._validate_figure_captions()
        self._validate_links()
        self._validate_mathematical_notation()
        self._validate_code_blocks()
        self._validate_section_consistency()
        self._validate_output_integration()
        
        return self.results
    
    def _validate_markdown_structure(self):
        """Validate markdown file structure and headers"""
        print("📝 Validating markdown structure...")
        
        required_sections = [
            "00_title.md", "01_abstract.md", "02_introduction.md",
            "03_mathematical_foundations.md", "04_system_architecture.md",
            "05_computational_methods.md", "06_geometric_applications.md",
            "07_pattern_discovery.md", "08_research_applications.md",
            "09_conclusion.md"
        ]
        
        for section in required_sections:
            file_path = self.markdown_dir / section
            if not file_path.exists():
                self.results.append(ValidationResult(
                    "markdown_structure",
                    False,
                    f"Required section missing: {section}",
                    "error",
                    str(file_path)
                ))
            else:
                self.results.append(ValidationResult(
                    "markdown_structure",
                    True,
                    f"Section found: {section}",
                    "info",
                    str(file_path)
                ))
        
        # Check header hierarchy
        for md_file in self.markdown_dir.glob("*.md"):
            if md_file.name == "00_title.md":
                continue  # Skip title file
                
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                if line.startswith('#'):
                    # Check for proper header hierarchy
                    if line.startswith('####') and not any(l.startswith('###') for l in lines[:i]):
                        self.results.append(ValidationResult(
                            "header_hierarchy",
                            False,
                            f"H4 header without H3 parent: {line.strip()}",
                            "warning",
                            str(md_file),
                            i
                        ))
    
    def _validate_image_references(self):
        """Validate image references and paths"""
        print("🖼️  Validating image references...")
        
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                matches = re.findall(image_pattern, line)
                for alt_text, img_path in matches:
                    # Check if image path starts with output/
                    if not img_path.startswith('output/'):
                        self.results.append(ValidationResult(
                            "image_path",
                            False,
                            f"Image path should start with 'output/': {img_path}",
                            "error",
                            str(md_file),
                            i
                        ))
                    
                    # Check if image file exists
                    full_path = self.output_dir / img_path
                    if not full_path.exists():
                        self.results.append(ValidationResult(
                            "image_exists",
                            False,
                            f"Image file not found: {img_path}",
                            "error",
                            str(md_file),
                            i
                        ))
                    
                    # Check alt text format
                    if not alt_text.startswith('Figure '):
                        self.results.append(ValidationResult(
                            "image_alt_text",
                            False,
                            f"Alt text should start with 'Figure X:': {alt_text}",
                            "warning",
                            str(md_file),
                            i
                        ))
    
    def _validate_figure_captions(self):
        """Validate figure captions"""
        print("📋 Validating figure captions...")
        
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                if line.startswith('!['):
                    # Check if next line has detailed caption
                    if i < len(lines) and lines[i].strip().startswith('**Figure'):
                        caption_line = lines[i].strip()
                        # Validate caption format
                        if not re.match(r'\*\*Figure \d+:\*\* .+', caption_line):
                            self.results.append(ValidationResult(
                                "figure_caption",
                                False,
                                f"Invalid caption format: {caption_line}",
                                "error",
                                str(md_file),
                                i + 1
                            ))
                    else:
                        self.results.append(ValidationResult(
                            "figure_caption",
                            False,
                            f"Missing detailed caption after image reference",
                            "error",
                            str(md_file),
                            i
                        ))
    
    def _validate_links(self):
        """Validate links and references"""
        print("🔗 Validating links...")
        
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                matches = re.findall(link_pattern, line)
                for link_text, url in matches:
                    # Check GitHub links have 🔗 emoji
                    if 'github.com' in url and '🔗' not in link_text:
                        self.results.append(ValidationResult(
                            "link_format",
                            False,
                            f"GitHub links should include 🔗 emoji: {link_text}",
                            "warning",
                            str(md_file),
                            i
                        ))
                    
                    # Check for "click here" or similar
                    if link_text.lower() in ['click here', 'this link', 'here']:
                        self.results.append(ValidationResult(
                            "link_text",
                            False,
                            f"Use descriptive link text instead of '{link_text}'",
                            "warning",
                            str(md_file),
                            i
                        ))
    
    def _validate_mathematical_notation(self):
        """Validate mathematical notation"""
        print("🧮 Validating mathematical notation...")
        
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                # Check for inline code formatting
                if '`' in line:
                    # Check if mathematical terms are properly formatted
                    if re.search(r'\b(IVM|quadray|tetrahedron|octahedron)\b', line):
                        if not re.search(r'`[^`]*\b(IVM|quadray|tetrahedron|octahedron)\b[^`]*`', line):
                            self.results.append(ValidationResult(
                                "math_notation",
                                False,
                                f"Mathematical terms should be in backticks: {line.strip()}",
                                "warning",
                                str(md_file),
                                i
                            ))
    
    def _validate_code_blocks(self):
        """Validate code blocks"""
        print("💻 Validating code blocks...")
        
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            in_code_block = False
            code_language = None
            
            for i, line in enumerate(lines, 1):
                if line.strip().startswith('```'):
                    if not in_code_block:
                        # Starting code block
                        in_code_block = True
                        code_language = line.strip()[3:].strip()
                    else:
                        # Ending code block
                        in_code_block = False
                        code_language = None
                elif in_code_block and code_language == 'python':
                    # Check Python code formatting
                    if line.strip() and not line.startswith('    ') and not line.strip().startswith('#'):
                        self.results.append(ValidationResult(
                            "code_indentation",
                            False,
                            f"Python code should be indented: {line.strip()}",
                            "warning",
                            str(md_file),
                            i
                        ))
    
    def _validate_section_consistency(self):
        """Validate consistency across sections"""
        print("📊 Validating section consistency...")
        
        # Check for consistent terminology
        terminology = {
            'Symergetics': 0,
            'Synergetics': 0,
            'IVM': 0,
            'quadray': 0
        }
        
        for md_file in self.markdown_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8').lower()
            for term in terminology:
                terminology[term] += content.count(term.lower())
        
        # Check for consistent usage
        if terminology['symergetics'] > 0 and terminology['synergetics'] > 0:
            self.results.append(ValidationResult(
                "terminology_consistency",
                False,
                f"Mixed usage of 'Symergetics' and 'Synergetics': {terminology}",
                "warning"
            ))
    
    def _validate_output_integration(self):
        """Validate output directory integration"""
        print("📁 Validating output integration...")
        
        if not self.output_dir.exists():
            self.results.append(ValidationResult(
                "output_directory",
                False,
                f"Output directory not found: {self.output_dir}",
                "error"
            ))
            return
        
        # Count files by category
        categories = {}
        for ext in ['*.png', '*.svg', '*.pdf', '*.jpg', '*.jpeg']:
            for file_path in self.output_dir.rglob(ext):
                category = file_path.parent.name
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1
        
        if not categories:
            self.results.append(ValidationResult(
                "output_files",
                False,
                "No visualization files found in output directory",
                "warning"
            ))
        else:
            self.results.append(ValidationResult(
                "output_files",
                True,
                f"Found visualization files: {categories}",
                "info"
            ))
    
    def generate_report(self) -> str:
        """Generate validation report"""
        report = []
        report.append("# Symergetics Paper Validation Report")
        report.append("")
        report.append(f"**Generated:** {Path.cwd()}")
        report.append(f"**Total Checks:** {len(self.results)}")
        report.append("")
        
        # Group results by severity
        errors = [r for r in self.results if r.severity == 'error']
        warnings = [r for r in self.results if r.severity == 'warning']
        info = [r for r in self.results if r.severity == 'info']
        
        report.append(f"**Errors:** {len(errors)}")
        report.append(f"**Warnings:** {len(warnings)}")
        report.append(f"**Info:** {len(info)}")
        report.append("")
        
        # Summary
        if not errors and not warnings:
            report.append("✅ **All validations passed!**")
        elif not errors:
            report.append("⚠️ **Warnings found, but no errors.**")
        else:
            report.append("❌ **Errors found that need to be fixed.**")
        
        report.append("")
        
        # Detailed results
        if errors:
            report.append("## Errors")
            report.append("")
            for result in errors:
                report.append(f"- **{result.check_name}**: {result.message}")
                if result.file_path:
                    report.append(f"  - File: {result.file_path}")
                if result.line_number:
                    report.append(f"  - Line: {result.line_number}")
                report.append("")
        
        if warnings:
            report.append("## Warnings")
            report.append("")
            for result in warnings:
                report.append(f"- **{result.check_name}**: {result.message}")
                if result.file_path:
                    report.append(f"  - File: {result.file_path}")
                if result.line_number:
                    report.append(f"  - Line: {result.line_number}")
                report.append("")
        
        if info:
            report.append("## Information")
            report.append("")
            for result in info:
                report.append(f"- **{result.check_name}**: {result.message}")
                if result.file_path:
                    report.append(f"  - File: {result.file_path}")
                report.append("")
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "validation_report.md"):
        """Save validation report to file"""
        report = self.generate_report()
        report_path = self.paper_dir / filename
        report_path.write_text(report, encoding='utf-8')
        print(f"📄 Validation report saved to: {report_path}")
        return report_path


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='Validate Symergetics paper')
    parser.add_argument('--paper-dir', default='paper',
                       help='Paper directory path')
    parser.add_argument('--output', default='validation_report.md',
                       help='Output report filename')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = PaperValidator(args.paper_dir)
    
    # Run validation
    results = validator.validate_all()
    
    # Generate and save report
    report_path = validator.save_report(args.output)
    
    # Print summary
    errors = len([r for r in results if r.severity == 'error'])
    warnings = len([r for r in results if r.severity == 'warning'])
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {len(results) - errors - warnings}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"❌ Errors: {errors}")
    
    if errors > 0:
        print("\n❌ Validation failed with errors!")
        sys.exit(1)
    elif warnings > 0:
        print("\n⚠️  Validation passed with warnings.")
        sys.exit(0)
    else:
        print("\n✅ All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
