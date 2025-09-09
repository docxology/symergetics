#!/usr/bin/env python3
"""
Organized Output Structure Demo

This demo specifically showcases the new organized output directory structure
introduced to make visualization outputs much more manageable and well-organized.

Run this script to see how outputs are now automatically organized into
logical categories and subcategories.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from symergetics.visualization import (
    set_config, get_config, create_output_structure_readme, 
    list_output_structure, plot_polyhedron, plot_palindromic_pattern,
    plot_continued_fraction
)
from symergetics.core.numbers import SymergeticsNumber
import numpy as np


def demonstrate_organized_structure():
    """Demonstrate the organized output structure."""
    print("SYMERGETICS ORGANIZED OUTPUT STRUCTURE DEMO")
    print("=" * 50)
    
    # Configure for organized output
    config = {
        'output_dir': 'output/demos',
        'backend': 'ascii',  # Use ASCII for compatibility
        'organize_by_type': True,  # Enable organized structure
        'include_timestamps': False  # Clean filenames for demo
    }
    
    set_config(config)
    print(f"✓ Configuration set:")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Organized structure: {config['organize_by_type']}")
    
    # Initialize structure with README files
    create_output_structure_readme()
    print(f"✓ Created organized directory structure with README files")
    
    print(f"\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS BY CATEGORY")
    print("=" * 50)
    
    # Generate some visualizations to demonstrate organization
    visualizations = []
    
    print(f"\n1. GEOMETRIC VISUALIZATIONS:")
    print(f"-" * 30)
    
    # Polyhedron (goes to geometric/polyhedra/)
    result = plot_polyhedron('tetrahedron', backend='ascii')
    visualizations.append(result)
    print(f"✓ Tetrahedron: {result['files'][0]}")
    
    # Another polyhedron
    result = plot_polyhedron('cube', backend='ascii') 
    visualizations.append(result)
    print(f"✓ Cube: {result['files'][0]}")
    
    print(f"\n2. NUMBER PATTERN VISUALIZATIONS:")
    print(f"-" * 30)
    
    # Palindromes (goes to numbers/palindromes/)
    result = plot_palindromic_pattern(12321, backend='ascii')
    visualizations.append(result)
    print(f"✓ Palindrome 12321: {result['files'][0]}")
    
    # Scheherazade number (goes to numbers/scheherazade/)
    sn = SymergeticsNumber(1001)
    result = plot_palindromic_pattern(sn, backend='ascii')
    visualizations.append(result)
    print(f"✓ Scheherazade 1001: {result['files'][0]}")
    
    print(f"\n3. MATHEMATICAL VISUALIZATIONS:")
    print(f"-" * 30)
    
    # Continued fraction (goes to mathematical/continued_fractions/)
    result = plot_continued_fraction(np.pi, max_terms=5, backend='ascii')
    visualizations.append(result)
    print(f"✓ π continued fraction: {result['files'][0]}")
    
    # Show the organized structure
    print(f"\n" + "=" * 50)
    print("ORGANIZED STRUCTURE ANALYSIS")
    print("=" * 50)
    
    structure_info = list_output_structure()
    print(f"Output directory: {structure_info['base_path']}")
    print(f"Organized structure enabled: {structure_info['organized']}")
    print(f"\nDirectory Structure Created:")
    
    for category, info in structure_info['categories'].items():
        if info['exists']:
            total_files = sum(sc.get('file_count', 0) for sc in info['subcategories'].values() if sc.get('exists', False))
            print(f"\n📁 {category}/ ({total_files} files)")
            
            for subcat, subcat_info in info['subcategories'].items():
                if subcat_info.get('exists', False):
                    file_count = subcat_info.get('file_count', 0)
                    print(f"   └── {subcat}/ ({file_count} files)")
                    
                    # Show sample files
                    sample_files = subcat_info.get('files', [])[:3]
                    for file_name in sample_files:
                        print(f"       • {file_name}")
    
    print(f"\n" + "=" * 50)
    print("DIRECTORY TREE VISUALIZATION")
    print("=" * 50)
    
    base_path = Path(structure_info['base_path'])
    if base_path.exists():
        print_directory_tree(base_path, max_depth=3)
    
    print(f"\n" + "=" * 50)
    print("BENEFITS OF ORGANIZED STRUCTURE")  
    print("=" * 50)
    
    print("✓ Logical categorization by visualization type")
    print("✓ Easy to find specific types of outputs")
    print("✓ Scalable to hundreds or thousands of files")
    print("✓ Clear README files explain each category")
    print("✓ Backwards compatible (can be disabled)")
    print("✓ Works with all visualization backends")
    print("✓ Integrates seamlessly with batch processing")
    
    print("\n📖 Check output/demos/README.md for detailed structure explanation")
    print("🔍 Explore the subdirectories to see the organized outputs!")


def print_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """Print a directory tree structure."""
    if current_depth >= max_depth:
        return
        
    if current_depth == 0:
        print(f"{path.name}/")
        prefix = ""
    
    try:
        items = sorted(path.iterdir())
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        # Print directories first
        for i, dir_item in enumerate(dirs):
            is_last_dir = i == len(dirs) - 1 and len(files) == 0
            connector = "└── " if is_last_dir else "├── "
            print(f"{prefix}{connector}{dir_item.name}/")
            
            # Recursively print subdirectories
            if current_depth < max_depth - 1:
                extension = "    " if is_last_dir else "│   "
                print_directory_tree(dir_item, prefix + extension, max_depth, current_depth + 1)
        
        # Print files
        for i, file_item in enumerate(files):
            is_last = i == len(files) - 1
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{file_item.name}")
            
    except PermissionError:
        print(f"{prefix}[Permission Denied]")


if __name__ == "__main__":
    demonstrate_organized_structure()
