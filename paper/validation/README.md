# Paper Validation System

A comprehensive validation system for the Symergetics research paper, ensuring proper formatting, image integration, and adherence to style guidelines.

## Overview

This validation system provides automated checks for:
- Markdown structure and formatting
- Image reference validation
- Figure caption compliance
- Link formatting standards
- Mathematical notation consistency
- Code block formatting
- Output integration verification

## Quick Start

### Basic Validation

```bash
# Run all validations
python paper/validation/validate_paper.py

# Run with custom paper directory
python paper/validation/validate_paper.py --paper-dir /path/to/paper

# Generate detailed report
python paper/validation/validate_paper.py --output detailed_report.md
```

### Image-Specific Validation

```bash
# Check only image integration
python paper/validation/check_images.py

# Verbose output
python paper/validation/check_images.py --verbose
```

## Validation Checks

### 1. Markdown Structure

**Checks:**
- Required sections present (00_title.md through 09_conclusion.md)
- Proper header hierarchy (H1 → H2 → H3 → H4)
- Consistent formatting across sections

**Example Issues:**
```
❌ Required section missing: 05_computational_methods.md
⚠️ H4 header without H3 parent: #### Sub-subsection
```

### 2. Image References

**Checks:**
- Image paths start with `output/`
- Referenced images exist in output directory
- Alt text follows `Figure X: Title` format
- Proper file extensions (.png, .svg, .pdf, .jpg, .jpeg)

**Example Issues:**
```
❌ Image path should start with 'output/': images/figure.png
❌ Image file not found: output/geometric/polyhedra/tetrahedron.png
⚠️ Alt text should start with 'Figure X:': image
```

### 3. Figure Captions

**Checks:**
- Every image has a detailed caption
- Caption format: `**Figure X**: Description`
- Captions provide meaningful context
- No missing captions after image references

**Example Issues:**
```
❌ Missing detailed caption after image reference
❌ Invalid caption format: **Figure 1** Description
```

### 4. Link Formatting

**Checks:**
- GitHub links include 🔗 emoji
- Descriptive link text (not "click here")
- Proper URL formatting
- Internal vs external link distinction

**Example Issues:**
```
⚠️ GitHub links should include 🔗 emoji: core module
⚠️ Use descriptive link text instead of 'click here'
```

### 5. Mathematical Notation

**Checks:**
- Technical terms in backticks
- Consistent terminology usage
- Proper mathematical formatting
- Variable and function naming

**Example Issues:**
```
⚠️ Mathematical terms should be in backticks: IVM coordinate system
```

### 6. Code Blocks

**Checks:**
- Proper indentation for Python code
- Language specification in code blocks
- Consistent formatting
- Appropriate code length

**Example Issues:**
```
⚠️ Python code should be indented: def function():
```

### 7. Output Integration

**Checks:**
- Output directory exists
- Visualization files present
- Proper file organization
- Category structure compliance

**Example Issues:**
```
❌ Output directory not found: /path/to/output
⚠️ No visualization files found in output directory
```

## Report Generation

### Validation Report

The main validation script generates a comprehensive report:

```markdown
# Symergetics Paper Validation Report

**Generated:** /path/to/paper
**Total Checks:** 25

**Errors:** 0
**Warnings:** 3
**Info:** 22

✅ **All validations passed!**

## Warnings
- **link_format**: GitHub links should include 🔗 emoji
- **math_notation**: Mathematical terms should be in backticks
- **code_indentation**: Python code should be indented
```

### Image Check Report

Specialized report for image integration:

```markdown
# Image Integration Check Report

**Paper Directory:** paper
**Output Directory:** output
**Total Issues:** 5

**Errors:** 1
**Warnings:** 4

❌ **Errors found that need to be fixed.**

## Errors
### Image Missing
- **File:** paper/markdown/03_mathematical_foundations.md
- **Line:** 23
- **Message:** Image file not found: output/geometric/coordinates/quadray_coordinate_0_0_0_0.png
- **Suggestion:** Check if file exists at: /path/to/output/geometric/coordinates/quadray_coordinate_0_0_0_0.png
```

## Integration with Build Process

### Pre-commit Validation

Add validation to your workflow:

```bash
#!/bin/bash
# pre-commit hook

echo "🔍 Validating paper before commit..."

# Run validation
python paper/validation/validate_paper.py
if [ $? -ne 0 ]; then
    echo "❌ Validation failed. Please fix issues before committing."
    exit 1
fi

echo "✅ Validation passed. Proceeding with commit."
```

### CI/CD Integration

```yaml
# .github/workflows/validate-paper.yml
name: Validate Paper
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: pip install -r paper/render/requirements.txt
    - name: Validate paper
      run: python paper/validation/validate_paper.py
    - name: Check images
      run: python paper/validation/check_images.py
```

## Customization

### Adding New Checks

Extend the validation system by adding new methods to `PaperValidator`:

```python
def _validate_custom_check(self):
    """Custom validation check"""
    # Your validation logic here
    pass
```

### Custom Report Formats

Modify the report generation methods to output different formats:

```python
def generate_json_report(self) -> str:
    """Generate JSON format report"""
    import json
    return json.dumps(self.results, indent=2)
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install required dependencies
pip install -r paper/render/requirements.txt
```

**2. Path Issues**
```bash
# Run from project root
cd /path/to/symergetics
python paper/validation/validate_paper.py
```

**3. Permission Errors**
```bash
# Ensure write permissions
chmod +x paper/validation/*.py
```

### Debug Mode

Enable verbose output for debugging:

```bash
python paper/validation/validate_paper.py --verbose
python paper/validation/check_images.py --verbose
```

## Best Practices

### Regular Validation

1. **Before editing**: Run validation to check current state
2. **During editing**: Use validation to catch issues early
3. **Before committing**: Ensure all validations pass
4. **Before publishing**: Run comprehensive validation

### Fixing Issues

1. **Start with errors**: Fix all errors before addressing warnings
2. **Follow suggestions**: Use the provided suggestions as guidance
3. **Test changes**: Re-run validation after making changes
4. **Document fixes**: Update style guide if needed

### Maintaining Quality

1. **Consistent formatting**: Follow established patterns
2. **Regular updates**: Keep validation rules current
3. **Team coordination**: Share validation results
4. **Continuous improvement**: Refine validation rules based on experience

---

**Last Updated:** January 2025  
**Version:** 1.0.0  
**Maintainer:** Daniel Ari Friedman (daniel@activeinference.institute)
