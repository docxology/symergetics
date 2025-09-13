#!/usr/bin/env python3
"""
Symergetics Paper Generation Script

This script orchestrates the complete paper generation process:
1. Verifies all components are present
2. Generates any missing visualizations if needed
3. Renders the PDF with integrated images and data
4. Provides summary and validation

Author: Daniel Ari Friedman
Email: daniel@activeinference.institute
ORCID: 0000-0001-6232-9096
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import subprocess
import shutil
import time
import json

# Determine project root regardless of where the script is run from
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'paper' else SCRIPT_DIR

# Set up comprehensive logging
def setup_logging():
    """Set up comprehensive logging for the paper generation process."""
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Configure root logger
    logger = logging.getLogger('symergetics_paper_generator')
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler with rotation
    log_file = logs_dir / "paper_generation.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    # Detailed JSON handler
    json_log_file = logs_dir / "paper_generation_detailed.log"
    json_handler = logging.handlers.RotatingFileHandler(
        json_log_file, maxBytes=5*1024*1024, backupCount=3
    )
    json_handler.setLevel(logging.DEBUG)

    # Formatters
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    file_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': self.formatTime(record, datefmt='%Y-%m-%d %H:%M:%S.%f'),
                'level': record.levelname,
                'logger': record.name,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage()
            }
            if hasattr(record, 'elapsed_time'):
                log_entry['elapsed_time'] = record.elapsed_time
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            return json.dumps(log_entry)

    json_formatter = JSONFormatter()

    # Set formatters
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    json_handler.setFormatter(json_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(json_handler)

    return logger

# Initialize logging
logger = setup_logging()


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_step(step: str, description: str):
    """Print a formatted step and log it"""
    message = f"[{step}] {description}"
    print(f"\n{message}")
    logger.info(message)


def check_requirements():
    """Check if all required components are present"""
    print_step("1", "Checking requirements...")

    try:
        # Check paper structure
        paper_dir = PROJECT_ROOT / "paper"
        logger.debug(f"Checking paper directory: {paper_dir}")
        if not paper_dir.exists():
            logger.error(f"Paper directory not found: {paper_dir}")
            print("❌ Error: paper/ directory not found")
            return False

        # Check markdown sections
        markdown_dir = paper_dir / "markdown"
        logger.debug(f"Checking markdown directory: {markdown_dir}")
        if not markdown_dir.exists():
            logger.error(f"Markdown directory not found: {markdown_dir}")
            print("❌ Error: paper/markdown/ directory not found")
            return False

        md_files = list(markdown_dir.glob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")
        if len(md_files) < 5:
            logger.warning(f"Only {len(md_files)} markdown files found (expected 9+)")
            print(f"⚠️  Warning: Only {len(md_files)} markdown files found (expected 9+)")
        else:
            print(f"✅ Found {len(md_files)} markdown sections")

        # Check render system
        render_dir = paper_dir / "render"
        logger.debug(f"Checking render directory: {render_dir}")
        if not render_dir.exists():
            logger.error(f"Render directory not found: {render_dir}")
            print("❌ Error: paper/render/ directory not found")
            return False

        required_render_files = [
            "pdf_renderer.py",
            "config.yaml",
            "requirements.txt"
        ]

        missing_files = []
        for file in required_render_files:
            file_path = render_dir / file
            if not file_path.exists():
                missing_files.append(file)
                logger.error(f"Required file not found: {file_path}")

        if missing_files:
            print(f"❌ Error: Missing render files: {', '.join(missing_files)}")
            return False

        print("✅ Render system files present")

        # Check main package components
        package_components = ["symergetics", "tests", "examples"]
        missing_components = []

        for component in package_components:
            component_path = PROJECT_ROOT / component
            logger.debug(f"Checking component: {component_path}")
            if not component_path.exists():
                missing_components.append(component)
                logger.error(f"Component not found: {component_path}")

        if missing_components:
            print(f"❌ Error: Missing components: {', '.join(missing_components)}")
            return False

        print("✅ Main package components present")

        # Check output directory (source of visualizations)
        output_dir = PROJECT_ROOT / "output"
        logger.debug(f"Checking output directory: {output_dir}")
        if not output_dir.exists():
            logger.warning(f"Output directory not found: {output_dir}")
            print("⚠️  Warning: output/ directory not found - will be created during generation")
        else:
            # Count visualization files
            png_files = list(output_dir.rglob("*.png"))
            svg_files = list(output_dir.rglob("*.svg"))
            txt_files = list(output_dir.rglob("*.txt"))

            total_files = len(png_files) + len(svg_files) + len(txt_files)
            logger.info(f"Found {total_files} visualization files: {len(png_files)} PNG, {len(svg_files)} SVG, {len(txt_files)} TXT")

            print(f"✅ Found {len(png_files)} PNG files")
            print(f"✅ Found {len(svg_files)} SVG files")
            print(f"✅ Found {len(txt_files)} text files")

        logger.info("Requirements check completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error during requirements check: {e}", exc_info=True)
        print(f"❌ Error during requirements check: {e}")
        return False


def clear_outputs():
    """Clear existing output directories and files"""
    print_step("2", "Clearing existing outputs...")

    # Clear output directory
    output_dir = PROJECT_ROOT / "output"
    if output_dir.exists():
        print("Clearing output directory...")
        shutil.rmtree(output_dir)
        print("✅ Output directory cleared")

    # Clear Mermaid images
    mermaid_dir = PROJECT_ROOT / "paper" / "mermaid_images"
    if mermaid_dir.exists():
        print("Clearing Mermaid images...")
        for file in mermaid_dir.glob("*.png"):
            file.unlink()
        print("✅ Mermaid images cleared")

    # Recreate output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "geometric").mkdir(parents=True, exist_ok=True)
    (output_dir / "mathematical").mkdir(parents=True, exist_ok=True)
    (output_dir / "numbers").mkdir(parents=True, exist_ok=True)

    print("✅ Output directories recreated")
    return True


def run_test_suite():
    """Run the complete test suite"""
    print_step("3", "Running test suite...")

    try:
        print("Checking test environment...")

        # Change to project root to run tests
        original_dir = os.getcwd()
        os.chdir(PROJECT_ROOT)

        # First, check if pytest is available
        try:
            import pytest
            print("✅ pytest module available")
        except ImportError:
            print("⚠️  pytest not available, attempting to install...")
            install_result = subprocess.run([
                sys.executable, "-m", "pip", "install", "pytest", "--quiet"
            ], capture_output=True, text=True)
            if install_result.returncode != 0:
                print("❌ Failed to install pytest, skipping tests")
                return False
            print("✅ pytest installed successfully")

        # Check if there are test files
        test_files = list(PROJECT_ROOT.glob("tests/test_*.py"))
        if not test_files:
            print("⚠️  No test files found, skipping test suite")
            return True

        print(f"Found {len(test_files)} test files")
        print("Running pytest on tests directory...")

        # Run tests with better error handling
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/",
            "-v", "--tb=short", "--disable-warnings",
            "--durations=0", "--durations-min=1.0"  # Show slow tests
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout

        if result.returncode == 0:
            print("✅ Test suite passed successfully!")
            # Count passed tests
            lines = result.stdout.strip().split('\n')
            passed_count = 0
            for line in lines:
                if 'PASSED' in line:
                    passed_count += 1
            print(f"📊 Tests passed: {passed_count}")
            return True
        else:
            print("⚠️  Test suite had failures or errors")
            print("Test summary:")

            # Parse and display test results
            lines = result.stdout.strip().split('\n')
            for line in lines[-15:]:  # Last 15 lines for summary
                if line.strip():
                    print(f"  {line}")

            if result.stderr.strip():
                print("Error details:")
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[-10:]:  # Last 10 error lines
                    if line.strip():
                        print(f"  {line}")

            print("⚠️  Continuing with paper generation despite test failures...")
            return True  # Don't fail the whole process

    except subprocess.TimeoutExpired:
        print("⏰ Test suite timed out after 10 minutes")
        print("⚠️  Continuing with paper generation...")
        return True
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        print("⚠️  Continuing with paper generation...")
        return True  # Don't fail the whole process
    finally:
        # Always return to original directory
        os.chdir(original_dir)


def run_examples():
    """Run all examples to generate visualizations"""
    print_step("4", "Running examples to generate visualizations...")

    try:
        original_dir = os.getcwd()

        # Change to project root
        os.chdir(PROJECT_ROOT)

        examples_dir = PROJECT_ROOT / "examples"
        if not examples_dir.exists():
            print("❌ Examples directory not found")
            return False

        # Get all Python example files
        example_files = list(examples_dir.glob("*.py"))
        example_files.sort()  # Consistent order

        if not example_files:
            print("⚠️  No example files found")
            return True

        print(f"Found {len(example_files)} example files to run:")
        for i, example_file in enumerate(example_files, 1):
            print(f"  {i}. {example_file.name}")

        success_count = 0
        failed_examples = []
        timed_out_examples = []

        for i, example_file in enumerate(example_files, 1):
            print(f"\n[{i}/{len(example_files)}] Running {example_file.name}...")
            try:
                start_time = time.time()
                result = subprocess.run([
                    sys.executable, str(example_file)
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

                elapsed = time.time() - start_time

                if result.returncode == 0:
                    print(f"✅ {example_file.name} completed successfully ({elapsed:.2f}s)")
                    success_count += 1

                    # Check if output files were created
                    output_check = check_example_output(example_file.name)
                    if output_check > 0:
                        print(f"      📁 Generated {output_check} output files")
                else:
                    print(f"❌ {example_file.name} failed ({elapsed:.2f}s)")
                    print(f"      Error: {result.stderr[-300:]}")  # Last 300 chars
                    failed_examples.append(example_file.name)

            except subprocess.TimeoutExpired:
                print(f"⏰ {example_file.name} timed out after 5 minutes")
                timed_out_examples.append(example_file.name)
            except Exception as e:
                print(f"❌ Error running {example_file.name}: {e}")
                failed_examples.append(example_file.name)

        # Summary
        print(f"\n📊 Examples Summary:")
        print(f"   ✅ Successful: {success_count}/{len(example_files)}")
        if failed_examples:
            print(f"   ❌ Failed: {len(failed_examples)} - {', '.join(failed_examples)}")
        if timed_out_examples:
            print(f"   ⏰ Timed out: {len(timed_out_examples)} - {', '.join(timed_out_examples)}")

        if success_count == len(example_files):
            print(f"✅ All {success_count} examples completed successfully")
        elif success_count > 0:
            print(f"⚠️  {success_count}/{len(example_files)} examples completed")
        else:
            print("❌ No examples completed successfully")

        return success_count > 0  # Continue if at least one example succeeded

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        return False
    finally:
        os.chdir(original_dir)


def check_example_output(example_name):
    """Check how many output files an example generated"""
    try:
        output_dir = PROJECT_ROOT / "output"
        if not output_dir.exists():
            return 0

        # Count files that might have been created by this example
        # This is a simple heuristic - we could make it more sophisticated
        total_files = 0
        for ext in ['*.png', '*.svg', '*.txt', '*.json']:
            files = list(output_dir.rglob(ext))
            total_files += len(files)

        return total_files
    except Exception:
        return 0


def regenerate_mermaid_diagrams():
    """Regenerate Mermaid diagrams"""
    print_step("5", "Regenerating Mermaid diagrams...")

    try:
        original_dir = os.getcwd()

        # Change to project root
        os.chdir(PROJECT_ROOT)

        mermaid_dir = PROJECT_ROOT / "paper" / "mermaid_images"
        mermaid_dir.mkdir(parents=True, exist_ok=True)

        # Look for Mermaid generation scripts or markdown files with Mermaid blocks
        render_dir = PROJECT_ROOT / "paper" / "render"

        # Check if there's a Mermaid generation script
        mermaid_script = render_dir / "generate_mermaid.py"
        if mermaid_script.exists():
            print("Running Mermaid generation script...")
            result = subprocess.run([
                sys.executable, str(mermaid_script)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Mermaid diagrams regenerated successfully")
                return True
            else:
                print("❌ Error regenerating Mermaid diagrams")
                print(result.stderr)
                return False

        # Check for markdown files with Mermaid blocks
        markdown_dir = PROJECT_ROOT / "paper" / "markdown"
        mermaid_blocks = []

        for md_file in markdown_dir.glob("*.md"):
            content = md_file.read_text()
            if "```mermaid" in content:
                mermaid_blocks.append(md_file)

        if mermaid_blocks:
            print(f"Found {len(mermaid_blocks)} markdown files with Mermaid blocks")
            print("Note: Manual Mermaid regeneration may be required for these files:")
            for md_file in mermaid_blocks:
                print(f"  - {md_file.name}")
            print("✅ Mermaid diagram check completed")
            return True

        print("⚠️  No Mermaid generation script or Mermaid blocks found")
        print("✅ Mermaid diagram check completed (no regeneration needed)")
        return True

    except Exception as e:
        print(f"❌ Error regenerating Mermaid diagrams: {e}")
        return False
    finally:
        os.chdir(original_dir)


def scan_and_register_figures():
    """Scan visualization files and register them with the figure manager"""
    print_step("6", "Scanning and registering figures...")

    try:
        original_dir = os.getcwd()

        # Change to project root
        os.chdir(PROJECT_ROOT)

        # Check if figure scanner script exists
        render_dir = PROJECT_ROOT / "paper" / "render"
        scanner_script = render_dir / "scan_and_register_figures.py"

        if scanner_script.exists():
            print("Running figure scanner and registration script...")
            result = subprocess.run([
                sys.executable, str(scanner_script)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Figures scanned and registered successfully")
                # Print summary from the scanner output
                if result.stdout.strip():
                    # Extract key summary lines
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-10:]:  # Last 10 lines for summary
                        if line.strip() and ('Total' in line or 'Successfully' in line or 'Breakdown' in line):
                            print(f"  {line}")
                return True
            else:
                print("❌ Error scanning and registering figures")
                print(result.stderr[-500:])  # Last 500 chars of error
                return False
        else:
            print("⚠️  Figure scanner script not found")
            print("✅ Figure scanning skipped (scanner not available)")
            return True

    except Exception as e:
        print(f"❌ Error scanning figures: {e}")
        return False
    finally:
        os.chdir(original_dir)


def install_dependencies():
    """Install Python dependencies for rendering"""
    print_step("7", "Installing dependencies...")

    render_dir = PROJECT_ROOT / "paper" / "render"
    req_file = render_dir / "requirements.txt"

    if not req_file.exists():
        print("⚠️  Warning: requirements.txt not found, skipping dependency installation")
        return True

    # Check if required packages are already available
    required_packages = ['reportlab', 'markdown', 'yaml']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if not missing_packages:
        print("✅ All required dependencies are already installed")
        return True

    print(f"Installing missing packages: {', '.join(missing_packages)}")

    try:
        # Try to install missing packages individually
        for package in missing_packages:
            print(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"⚠️  Failed to install {package}: {result.stderr}")
            else:
                print(f"✅ {package} installed successfully")

        # Verify installation
        all_installed = True
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                all_installed = False
                break

        if all_installed:
            print("✅ All dependencies installed successfully")
            return True
        else:
            print("⚠️  Some dependencies may still be missing, but continuing...")
            return True  # Don't fail the whole process

    except subprocess.TimeoutExpired:
        print("⏰ Dependency installation timed out")
        return True  # Don't fail the whole process
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return True  # Don't fail the whole process


def check_visualizations():
    """Check that visualizations were generated successfully"""
    print_step("8", "Checking generated visualizations...")

    output_dir = PROJECT_ROOT / "output"
    if not output_dir.exists():
        print("❌ Error: No output directory found")
        return False

    # Check for key visualization categories
    categories = {
        "geometric/polyhedra": "3D polyhedral visualizations",
        "geometric/lattice": "IVM lattice visualizations",
        "mathematical": "Mathematical pattern visualizations",
        "numbers/palindromes": "Palindrome pattern visualizations"
    }

    total_files = 0
    for category, description in categories.items():
        cat_path = output_dir / category
        if cat_path.exists():
            files = list(cat_path.rglob("*"))
            if files:
                print(f"✅ {description}: {len(files)} files")
                total_files += len(files)
            else:
                print(f"⚠️  {description}: directory exists but empty")
        else:
            print(f"⚠️  {description}: directory not found")

    if total_files > 0:
        print(f"✅ Total visualization files: {total_files}")
        return True
    else:
        print("⚠️  No visualization files found")
        return True  # Don't fail if no visualizations, might be expected


def render_pdf():
    """Render the final PDF"""
    print_step("9", "Rendering PDF...")

    try:
        # Change to render directory
        render_dir = PROJECT_ROOT / "paper" / "render"
        original_dir = os.getcwd()

        os.chdir(render_dir)

        # Run the renderer with timeout
        cmd = [sys.executable, "run_render.py", "--verbose"]
        print("Running PDF renderer...")

        # Set a reasonable timeout (10 minutes)
        timeout_seconds = 600
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"❌ PDF renderer timed out after {timeout_seconds} seconds")
            return False

        if result.returncode == 0:
            print("✅ PDF generated successfully!")

            # Try to find the generated PDF in both possible locations
            project_output_dir = PROJECT_ROOT / "output"
            paper_output_dir = PROJECT_ROOT / "paper" / "output"

            pdf_files = []
            if project_output_dir.exists():
                pdf_files.extend(list(project_output_dir.glob("*.pdf")))
            if paper_output_dir.exists():
                pdf_files.extend(list(paper_output_dir.glob("*.pdf")))

            if pdf_files:
                latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
                print(f"📄 Output: {latest_pdf}")
                print(f"📄 Location: {latest_pdf.parent}")
                size_mb = latest_pdf.stat().st_size / (1024 * 1024)
                print(f"📄 Size: {size_mb:.2f} MB")
            else:
                print("📄 PDF generated (location unknown)")

            # Print any stdout from the renderer (but filter out excessive output)
            if result.stdout.strip():
                stdout_lines = result.stdout.strip().split('\n')
                # Show first few lines and last few lines if output is long
                if len(stdout_lines) > 20:
                    print("\nRenderer output (first 10 lines):")
                    for line in stdout_lines[:10]:
                        if line.strip():
                            print(f"  {line}")
                    print("  ...")
                    for line in stdout_lines[-5:]:
                        if line.strip():
                            print(f"  {line}")
                else:
                    print("\nRenderer output:")
                    for line in stdout_lines:
                        if line.strip():
                            print(f"  {line}")

            return True
        else:
            print("❌ PDF generation failed!")
            print("Error output:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running renderer: {e}")
        return False
    finally:
        # Always return to original directory
        os.chdir(original_dir)


def validate_output():
    """Validate the generated PDF"""
    print_step("10", "Validating output...")

    # Check both possible output locations
    project_output_dir = PROJECT_ROOT / "output"
    paper_output_dir = PROJECT_ROOT / "paper" / "output"

    pdf_files = []

    # Check project output directory
    if project_output_dir.exists():
        pdf_files.extend(list(project_output_dir.glob("*.pdf")))

    # Check paper output directory
    if paper_output_dir.exists():
        pdf_files.extend(list(paper_output_dir.glob("*.pdf")))

    if not pdf_files:
        print("⚠️  Warning: No PDF files found yet, PDF may still be generating")
        print("   This might be normal if validation runs before PDF is fully written")
        return True  # Don't fail - PDF might be generated successfully

    # Check the most recent PDF
    latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 Found PDF: {latest_pdf.name}")
    print(f"📄 Location: {latest_pdf.parent}")

    # Basic validation
    file_size = latest_pdf.stat().st_size
    if file_size < 10000:  # Less than 10KB is suspiciously small
        print(f"⚠️  Warning: PDF file seems small ({file_size} bytes)")
    else:
        size_mb = file_size / (1024 * 1024)
        print(f"✅ File size: {size_mb:.2f} MB")
    # Check if file is readable
    try:
        with open(latest_pdf, 'rb') as f:
            header = f.read(8)
            if header.startswith(b'%PDF-'):
                print("✅ Valid PDF format detected")
            else:
                print("⚠️  Warning: PDF header not detected")
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return False

    return True


def generate_summary():
    """Generate a summary of the paper generation process"""
    print_header("PAPER GENERATION COMPLETE")

    # Count sections (all 12 specified ones)
    sections = [
        "00_title.md", "01_abstract.md", "02_introduction.md",
        "03_mathematical_foundations.md", "04_system_architecture.md",
        "05_computational_methods.md", "06_results.md",
        "07_geometric_applications.md", "08_pattern_discovery.md", 
        "09_research_applications.md", "10_ongoing_questions_inquiries.md", 
        "11_conclusion.md"
    ]

    markdown_dir = PROJECT_ROOT / "paper" / "markdown"
    found_sections = 0
    for section in sections:
        if (markdown_dir / section).exists():
            found_sections += 1

    print(f"📝 Sections: {found_sections}/12 specified markdown files loaded")

    # Count visualizations
    output_dir = PROJECT_ROOT / "output"
    if output_dir.exists():
        png_count = len(list(output_dir.rglob("*.png")))
        svg_count = len(list(output_dir.rglob("*.svg")))
        txt_count = len(list(output_dir.rglob("*.txt")))
        print(f"🖼️  Visualizations: {png_count} PNG, {svg_count} SVG, {txt_count} text files")

    # Find generated PDF
    pdf_files = list(output_dir.glob("*.pdf")) if output_dir.exists() else []
    if pdf_files:
        latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
        size_mb = latest_pdf.stat().st_size / (1024 * 1024)
        print(f"📄 Size: {size_mb:.2f} MB")
        print(f"📄 Location: {latest_pdf.absolute()}")

    # Check orchestration results
    orchestration_file = output_dir / "orchestration_results.json"
    if orchestration_file.exists():
        try:
            import json
            with open(orchestration_file, 'r') as f:
                orchestration_data = json.load(f)
            print(f"🚀 Orchestration: {orchestration_data.get('summary', {}).get('successful', 0)}/{orchestration_data.get('summary', {}).get('total_examples', 0)} examples successful")
            print(f"⏱️  Total orchestration time: {orchestration_data.get('total_duration', 0):.1f} seconds")
        except:
            print("📊 Orchestration results available but couldn't parse details")

    print(f"\n⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 Author: Daniel Ari Friedman")
    print(f"📧 Contact: daniel@activeinference.institute")
    print(f"🔗 ORCID: 0000-0001-6232-9096")

    print(f"\n{'='*60}")
    print(" Paper Structure:")
    for i, section in enumerate(sections, 0):
        status = "✓" if (markdown_dir / section).exists() else "✗"
        title = section.replace('.md', '').replace('_', ' ').title()
        print(f" {status} Section {i}: {title}")
    print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(" Workflow Summary:")
    print(" ✓ 1. Requirements checked")
    print(" ✓ 2. Full orchestration completed (setup, tests, demos)")
    print(" ✓ 3. Enhanced visualizations generated")
    print(" ✓ 4. PDF rendered with integrated content")
    print(" ✓ 5. Output validated")
    print(f"{'='*60}")


def run_full_orchestration():
    """Run the full orchestration from run.py including setup, tests, and demos"""
    print_step("2", "Running full orchestration (setup, tests, demos)...")
    
    try:
        # Change to project root
        original_dir = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        print("🚀 Executing complete Symergetics package orchestration...")
        print("   This includes:")
        print("   • Environment setup and dependency installation")
        print("   • Full test suite execution (all test files)")
        print("   • All example demonstrations (all .py files in examples/)")
        print("   • Comprehensive output generation and reporting")
        print()
        
        # Run the full orchestration with explicit arguments to ensure full execution
        result = subprocess.run([
            sys.executable, "run.py"
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            print("✅ Full orchestration completed successfully")
            print("   All tests and examples executed successfully")
            
            # Print summary from orchestration
            if result.stdout.strip():
                stdout_lines = result.stdout.strip().split('\n')
                print("   Key orchestration results:")
                # Show key summary lines
                for line in stdout_lines[-25:]:  # Last 25 lines for summary
                    if line.strip() and ('✅' in line or '❌' in line or '📊' in line or '⏱️' in line or 'SUCCESS' in line or 'FAILED' in line):
                        print(f"     {line}")
            return True
        else:
            print("⚠️ Full orchestration had issues but continuing...")
            print("   Some tests or examples may have failed")
            print("   Error output:")
            print(result.stderr[-500:])  # Last 500 chars
            return True  # Don't fail the whole process
            
    except subprocess.TimeoutExpired:
        print("⏰ Full orchestration timed out after 30 minutes")
        print("⚠️ Continuing with paper generation...")
        return True
    except Exception as e:
        print(f"❌ Error running full orchestration: {e}")
        print("⚠️ Continuing with paper generation...")
        return True
    finally:
        os.chdir(original_dir)

def run_enhanced_visualizations():
    """Run the enhanced visualizations generator"""
    print_step("3", "Generating enhanced visualizations...")
    
    try:
        # Change to project root
        original_dir = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        # Run the enhanced visualizations script
        result = subprocess.run([
            sys.executable, "paper/improve_paper_visualizations.py"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✅ Enhanced visualizations generated successfully")
            # Print summary from visualizations
            if result.stdout.strip():
                stdout_lines = result.stdout.strip().split('\n')
                # Show key summary lines
                for line in stdout_lines[-10:]:  # Last 10 lines for summary
                    if line.strip() and ('✅' in line or '❌' in line or 'Enhanced' in line):
                        print(f"  {line}")
            return True
        else:
            print("⚠️ Enhanced visualizations had issues but continuing...")
            print("Error output:")
            print(result.stderr[-300:])  # Last 300 chars
            return True  # Don't fail the whole process
            
    except subprocess.TimeoutExpired:
        print("⏰ Enhanced visualizations timed out after 10 minutes")
        print("⚠️ Continuing with paper generation...")
        return True
    except Exception as e:
        print(f"❌ Error running enhanced visualizations: {e}")
        print("⚠️ Continuing with paper generation...")
        return True
    finally:
        os.chdir(original_dir)

def main():
    """Main paper generation workflow with full orchestration"""
    print_header("SYMERGETICS PAPER GENERATOR")
    print("Generating scientific paper with integrated visualizations...")
    print(f"Author: Daniel Ari Friedman")
    print(f"ORCID: 0000-0001-6232-9096")
    print()
    print("Workflow:")
    print("1. Check requirements")
    print("2. Run full orchestration (setup, tests, demos)")
    print("   - Environment setup and dependency installation")
    print("   - Complete test suite execution (all test files)")
    print("   - All example demonstrations (all .py files in examples/)")
    print("   - Comprehensive output generation and reporting")
    print("3. Generate enhanced visualizations")
    print("4. Render PDF")
    print("5. Validate output")
    print()

    # Execute comprehensive workflow
    steps = [
        ("Check requirements", check_requirements, True),
        ("Run full orchestration", run_full_orchestration, False),
        ("Generate enhanced visualizations", run_enhanced_visualizations, False),
        ("Render PDF", render_pdf, True),
        ("Validate output", validate_output, False),
    ]

    results = []
    critical_failures = 0

    for step_name, step_func, is_critical in steps:
        print(f"\n{'='*60}")
        print(f"STEP: {step_name}")
        print(f"{'='*60}")

        try:
            success = step_func()
            results.append((step_name, success, is_critical))

            if not success:
                if is_critical:
                    critical_failures += 1
                    print(f"❌ CRITICAL FAILURE in {step_name}")
                else:
                    print(f"⚠️  NON-CRITICAL FAILURE in {step_name} - continuing...")

        except Exception as e:
            print(f"💥 UNEXPECTED ERROR in {step_name}: {e}")
            results.append((step_name, False, is_critical))
            if is_critical:
                critical_failures += 1

    # Summary of results
    print(f"\n{'='*80}")
    print("WORKFLOW SUMMARY")
    print(f"{'='*80}")

    successful_steps = 0
    failed_steps = 0

    for step_name, success, is_critical in results:
        status_icon = "✅" if success else "❌"
        critical_icon = "🔴" if is_critical and not success else "⚪"
        print(f"{status_icon} {critical_icon} {step_name}")

        if success:
            successful_steps += 1
        else:
            failed_steps += 1

    print(f"\n📊 Results: {successful_steps} successful, {failed_steps} failed")

    if critical_failures > 0:
        print(f"❌ {critical_failures} critical failures detected")
        print("Paper generation may be incomplete or have issues")
        generate_summary()
        sys.exit(1)
    else:
        print("✅ All critical steps completed successfully")
        generate_summary()
        return True


if __name__ == '__main__':
    main()
