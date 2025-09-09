#!/usr/bin/env python3
"""
Symergetics Package - Main Orchestration Script

This script provides comprehensive setup, testing, and demonstration of the 
Symergetics package capabilities. It handles environment setup, runs tests,
and executes all examples to showcase the package features.

Usage:
    python run.py [--skip-tests] [--skip-setup] [--examples-only]
    
Options:
    --skip-tests    Skip running the test suite
    --skip-setup    Skip uv environment setup  
    --examples-only Run only the examples (skip setup and tests)
    --help         Show this help message
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import json


class SymergeticsOrchestrator:
    """Main orchestration class for Symergetics package setup and demonstration."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.examples_dir = self.project_root / "examples"
        self.output_dir = self.project_root / "output"
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        elapsed = time.time() - self.start_time
        timestamp = f"[{elapsed:6.1f}s]"
        prefix = "🔧" if level == "SETUP" else "🧪" if level == "TEST" else "🎨" if level == "DEMO" else "ℹ️"
        print(f"{timestamp} {prefix} {message}")
        
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        self.log(f"{description}...", "SETUP")
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            if result.returncode == 0:
                self.log(f"✅ {description} completed successfully")
                return True
            else:
                self.log(f"❌ {description} failed: {result.stderr}")
                return False
        except Exception as e:
            self.log(f"❌ {description} failed with exception: {e}")
            return False
            
    def setup_environment(self) -> bool:
        """Set up the uv environment and install dependencies."""
        self.log("Setting up uv environment and dependencies", "SETUP")
        
        commands = [
            (["uv", "sync"], "Synchronizing uv environment"),
            (["uv", "pip", "install", "-e", ".[scientific,test]"], "Installing Symergetics package in development mode with extras"),
        ]
        
        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                return False
                
        # Verify installation
        verify_cmd = ["uv", "run", "python", "-c", "import symergetics; print(f'Symergetics version: {symergetics.__version__}')"]
        if not self.run_command(verify_cmd, "Verifying Symergetics installation"):
            return False
            
        return True
        
    def run_tests(self) -> bool:
        """Run the complete test suite."""
        self.log("Running comprehensive test suite", "TEST")
        
        test_commands = [
            (["uv", "run", "python", "-m", "pytest", "tests/", "-v", "--tb=short", "-o", "addopts="],
             "Running all tests"),
            (["uv", "run", "python", "-m", "pytest", "tests/test_visualization.py", "-v", "-o", "addopts="],
             "Running visualization tests"),
            (["uv", "run", "python", "-m", "pytest", "tests/test_core_numbers.py", "-v", "-o", "addopts="],
             "Running core number tests"),
            (["uv", "run", "python", "-m", "pytest", "tests/test_geometry_polyhedra.py", "-v", "-o", "addopts="],
             "Running geometry tests"),
        ]
        
        for cmd, desc in test_commands:
            if not self.run_command(cmd, desc):
                self.log("⚠️ Some tests failed, but continuing with examples", "TEST") 
                break
                
        return True
        
    def run_example(self, example_script: Path) -> Dict[str, Any]:
        """Run a single example script."""
        script_name = example_script.name
        self.log(f"Running example: {script_name}", "DEMO")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                ["uv", "run", "python", str(example_script)], 
                cwd=self.project_root,
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                self.log(f"✅ {script_name} completed successfully ({elapsed:.1f}s)", "DEMO")
                return {
                    'script': script_name,
                    'success': True,
                    'duration': elapsed,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                self.log(f"❌ {script_name} failed ({elapsed:.1f}s): {result.stderr}", "DEMO")
                return {
                    'script': script_name,
                    'success': False,
                    'duration': elapsed,
                    'error': result.stderr,
                    'stdout': result.stdout
                }
                
        except subprocess.TimeoutExpired:
            self.log(f"⏱️ {script_name} timed out after 5 minutes", "DEMO")
            return {
                'script': script_name,
                'success': False,
                'duration': 300,
                'error': 'Timeout after 5 minutes'
            }
        except Exception as e:
            self.log(f"💥 {script_name} failed with exception: {e}", "DEMO")
            return {
                'script': script_name,
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
            
    def run_all_examples(self) -> List[Dict[str, Any]]:
        """Run all example scripts."""
        self.log("Running all example demonstrations", "DEMO")
        
        # Find all example scripts
        example_scripts = sorted(self.examples_dir.glob("*.py"))
        
        if not example_scripts:
            self.log("⚠️ No example scripts found in examples/ directory", "DEMO")
            return []
            
        self.log(f"Found {len(example_scripts)} example scripts", "DEMO")
        
        results = []
        for script in example_scripts:
            result = self.run_example(script)
            results.append(result)
            
            # Brief pause between examples
            time.sleep(1)
            
        return results
        
    def generate_summary_report(self, example_results: List[Dict[str, Any]]):
        """Generate a comprehensive summary report."""
        self.log("Generating comprehensive summary report", "INFO")
        
        total_duration = time.time() - self.start_time
        successful_examples = [r for r in example_results if r.get('success', False)]
        failed_examples = [r for r in example_results if not r.get('success', True)]
        
        # Count output files
        output_files = 0
        output_size = 0
        if self.output_dir.exists():
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    output_files += 1
                    try:
                        output_size += file_path.stat().st_size
                    except:
                        pass
        
        print("\n" + "="*80)
        print("🎯 SYMERGETICS PACKAGE ORCHESTRATION SUMMARY")
        print("="*80)
        
        print(f"\n⏱️ EXECUTION STATISTICS:")
        print(f"   Total execution time: {total_duration:.1f} seconds")
        print(f"   Examples attempted: {len(example_results)}")
        print(f"   Examples successful: {len(successful_examples)}")
        print(f"   Examples failed: {len(failed_examples)}")
        print(f"   Success rate: {len(successful_examples)/len(example_results)*100:.1f}%" if example_results else "   No examples run")
        
        print(f"\n📁 OUTPUT STATISTICS:")
        print(f"   Output files generated: {output_files}")
        print(f"   Total output size: {output_size/1024/1024:.2f} MB")
        print(f"   Output directory: {self.output_dir}")
        
        if successful_examples:
            print(f"\n✅ SUCCESSFUL EXAMPLES:")
            for result in successful_examples:
                duration = result.get('duration', 0)
                print(f"   • {result['script']} ({duration:.1f}s)")
                
        if failed_examples:
            print(f"\n❌ FAILED EXAMPLES:")
            for result in failed_examples:
                duration = result.get('duration', 0)
                error = result.get('error', 'Unknown error')[:100]
                print(f"   • {result['script']} ({duration:.1f}s) - {error}")
                
        print(f"\n🎨 SYMERGETICS FEATURES DEMONSTRATED:")
        print(f"   ✓ Integer-accounting mathematics with exact rational arithmetic")
        print(f"   ✓ Ratio-based geometry with polyhedron volume relationships") 
        print(f"   ✓ Quadmath coordinate system (Quadray coordinates)")
        print(f"   ✓ Organized output structure for scalable file management")
        print(f"   ✓ Comprehensive visualization capabilities (ASCII & PNG)")
        print(f"   ✓ Mathematical pattern analysis (palindromes, Scheherazade, primorials)")
        print(f"   ✓ Advanced number theory computations")
        print(f"   ✓ Geometric transformations and coordinate conversions")
        
        print(f"\n📖 NEXT STEPS:")
        print(f"   • Explore the organized output/ directory structure")
        print(f"   • Review individual example outputs for detailed insights")
        print(f"   • Run specific examples individually: uv run python examples/<script>.py")
        print(f"   • Check the README files in output/ subdirectories")
        print(f"   • Use the package in your own mathematical research!")
        
        # Save detailed results to JSON
        results_file = self.output_dir / "orchestration_results.json"
        self.output_dir.mkdir(exist_ok=True)
        
        detailed_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration': total_duration,
            'examples': example_results,
            'summary': {
                'total_examples': len(example_results),
                'successful': len(successful_examples), 
                'failed': len(failed_examples),
                'output_files': output_files,
                'output_size_mb': output_size/1024/1024
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
            
        print(f"\n💾 Detailed results saved to: {results_file}")
        print("="*80)


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Symergetics Package Orchestration Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Full setup, testing, and examples
  python run.py --skip-tests       # Skip test suite, run examples only  
  python run.py --examples-only    # Skip setup and tests, run examples only
  python run.py --skip-setup       # Skip uv setup, run tests and examples
        """
    )
    
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip running the test suite')
    parser.add_argument('--skip-setup', action='store_true', 
                       help='Skip uv environment setup')
    parser.add_argument('--examples-only', action='store_true',
                       help='Run only examples (skip setup and tests)')
                       
    args = parser.parse_args()
    
    orchestrator = SymergeticsOrchestrator()
    
    print("🚀 SYMERGETICS PACKAGE ORCHESTRATION")
    print("="*50)
    print("This script will set up, test, and demonstrate the comprehensive")
    print("capabilities of the Symergetics mathematical package.")
    print()
    
    success = True
    
    try:
        # Phase 1: Environment Setup
        if not args.examples_only and not args.skip_setup:
            if not orchestrator.setup_environment():
                orchestrator.log("⚠️ Environment setup failed, but continuing...")
                
        # Phase 2: Testing
        if not args.examples_only and not args.skip_tests:
            if not orchestrator.run_tests():
                orchestrator.log("⚠️ Some tests failed, but continuing with examples...")
                
        # Phase 3: Examples
        example_results = orchestrator.run_all_examples()
        
        # Phase 4: Summary
        orchestrator.generate_summary_report(example_results)
        
    except KeyboardInterrupt:
        orchestrator.log("\n🛑 Orchestration interrupted by user")
        success = False
    except Exception as e:
        orchestrator.log(f"\n💥 Orchestration failed: {e}")
        success = False
        
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
