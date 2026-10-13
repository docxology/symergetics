"""
Visualization module for Symergetics package.

Provides comprehensive visualization capabilities for geometric objects,
mathematical patterns, and symbolic computations in the Synergetics framework.

Features:
- Modular visualization system with multiple backends
- Configurable output formats (PNG, SVG, HTML, matplotlib figures)
- Geometric visualizations (polyhedra, coordinate systems, IVM lattice)
- Mathematical visualizations (palindromes, Scheherazade patterns, primorials)
- Symbolic computation visualizations (continued fractions, base conversions)
- Exact arithmetic pattern analysis and SSRCD visualizations

Author: Symergetics Team
"""

import json
import math
import os
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default configuration with organized output structure
DEFAULT_CONFIG = {
    "output_dir": "output",
    "backend": "matplotlib",  # matplotlib, plotly, ascii
    "organize_by_type": True,  # Organize outputs into type-based subdirectories
    "include_timestamps": False,  # Include timestamps in filenames for uniqueness
    "figure_size": (10, 8),
    "dpi": 300,  # High DPI for quality PNG output
    "colors": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "accent": "#2ca02c",
        "background": "#f8f9fa",
        "grid": "#e9ecef"
    },
    "fonts": {
        "title": {"size": 14, "weight": "bold"},
        "label": {"size": 12},
        "annotation": {"size": 10}
    },
    "formats": ["png", "svg", "html"],
    "png_options": {
        "transparent": False,
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.1
    },
    "animation": {
        "fps": 30,
        "duration": 5.0
    },
    # Organized output structure
    "output_structure": {
        "geometric": {
            "polyhedra": "3D polyhedron visualizations",
            "coordinates": "Quadray coordinate system plots",
            "lattice": "IVM lattice structures", 
            "transformations": "Geometric transformations"
        },
        "mathematical": {
            "continued_fractions": "Continued fraction analysis",
            "base_conversions": "Number base conversions",
            "pattern_analysis": "Mathematical pattern analysis",
            "ssrcd": "SSRCD (Scheherazade) analysis"
        },
        "numbers": {
            "palindromes": "Palindromic number patterns",
            "scheherazade": "Scheherazade number analysis", 
            "primorials": "Primorial distributions",
            "mnemonics": "Memory aid visualizations"
        },
        "batch": {
            "sessions": "Batch processing sessions",
            "comparisons": "Comparative analyses"
        },
        "reports": {
            "summaries": "Summary reports",
            "exports": "Data exports"
        }
    }
}

# Global configuration
_config = DEFAULT_CONFIG.copy()


def set_config(config: Dict[str, Any]) -> None:
    """
    Update the global visualization configuration.

    Args:
        config: Configuration dictionary to merge with defaults

    Raises:
        ValueError: Dimensions/DPI are invalid or exceed the nominal pixel budget.
    """
    global _config
    candidate = _config.copy()
    for key, value in config.items():
        # Merge into a candidate so invalid rendering settings never partially apply.
        if isinstance(value, dict) and isinstance(candidate.get(key), dict):
            merged = dict(candidate[key])
            merged.update(value)
            candidate[key] = merged
        else:
            candidate[key] = value
    _validate_canvas(candidate["figure_size"], candidate["dpi"])
    _config = candidate


# 100M nominal pixels require 400MB for one RGBA buffer before encoder overhead.
# Existing 32x24-inch manuscript figures at 300 DPI remain within this boundary.
MAX_CANVAS_PIXELS = 100_000_000


def _validate_canvas(figure_size: Any, dpi: Any) -> None:
    """Reject invalid dimensions and oversized nominal render canvases."""
    try:
        width, height = figure_size
        values = (width, height, dpi)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError("figure dimensions and DPI must be numeric")
        dimensions = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "figure_size must contain two finite positive dimensions; "
            "DPI must be finite and positive"
        ) from error
    if any(not math.isfinite(value) or value <= 0 for value in dimensions):
        raise ValueError("figure dimensions and DPI must be finite and positive")
    pixel_width = dimensions[0] * dimensions[2]
    pixel_height = dimensions[1] * dimensions[2]
    pixels = pixel_width * pixel_height
    if not math.isfinite(pixels) or pixels > MAX_CANVAS_PIXELS:
        raise ValueError(
            "render canvas exceeds 100,000,000 nominal pixels; "
            "reduce figure_size or DPI"
        )


def get_config() -> Dict[str, Any]:
    """Get the current visualization configuration."""
    return _config.copy()


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = DEFAULT_CONFIG.copy()


def ensure_output_dir(subdirs: Optional[List[str]] = None) -> Path:
    """
    Ensure the output directory exists, optionally with subdirectories.
    
    Args:
        subdirs: List of subdirectory names to create (e.g., ['geometric', 'polyhedra'])
        
    Returns:
        Path: The final output directory path
    """
    output_dir = Path(_config["output_dir"])
    
    # Create the subdirectory path if specified
    if subdirs:
        for subdir in subdirs:
            output_dir = output_dir / subdir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_organized_output_path(category: str, subcategory: str, filename: str) -> Path:
    """
    Get organized output path for a file based on visualization type.
    
    Args:
        category: Main category (e.g., 'geometric', 'mathematical', 'numbers')
        subcategory: Subcategory (e.g., 'polyhedra', 'palindromes', 'continued_fractions')
        filename: Base filename
        
    Returns:
        Path: Full path where the file should be saved
        
    Examples:
        >>> get_organized_output_path('geometric', 'polyhedra', 'tetrahedron_3d.png')
        Path('output/geometric/polyhedra/tetrahedron_3d.png')
    """
    if _config["organize_by_type"]:
        # Create organized structure
        output_dir = ensure_output_dir([category, subcategory])
    else:
        # Use flat structure
        output_dir = ensure_output_dir()
    
    # Add timestamp if requested
    if _config["include_timestamps"]:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
    return output_dir / filename
    


def create_output_structure_readme():
    """Create README files explaining the output directory structure."""
    if not _config["organize_by_type"]:
        return
    
    base_dir = Path(_config["output_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Main README
    main_readme = base_dir / "README.md"
    with open(main_readme, 'w') as f:
        f.write("# Symergetics Visualization Outputs\n\n")
        f.write("This directory contains organized visualizations from the Symergetics package.\n\n")
        f.write("## Directory Structure\n\n")
        
        for category, subcategories in _config["output_structure"].items():
            f.write(f"### {category.title()}/\n")
            for subcat, description in subcategories.items():
                f.write(f"- **{subcat}/**: {description}\n")
            f.write("\n")
        
        f.write("## File Naming Conventions\n\n")
        f.write("- Files are named descriptively based on their content\n")
        f.write("- Timestamps are included when `include_timestamps` is enabled\n")
        f.write("- Multiple formats may be generated based on configuration\n\n")
        f.write("Generated by Symergetics visualization system.\n")
    
    # Category READMEs
    for category, subcategories in _config["output_structure"].items():
        cat_dir = base_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        cat_readme = cat_dir / "README.md"
        with open(cat_readme, 'w') as f:
            f.write(f"# {category.title()} Visualizations\n\n")
            f.write("## Subcategories\n\n")
            for subcat, description in subcategories.items():
                f.write(f"### {subcat}/\n{description}\n\n")


def list_output_structure() -> Dict[str, Any]:
    """
    Get information about the current output structure.
    
    Returns:
        Dict containing structure information and statistics
    """
    base_dir = Path(_config["output_dir"])
    if not base_dir.exists():
        return {"exists": False, "structure": _config["output_structure"]}
    
    structure_info = {
        "exists": True,
        "base_path": str(base_dir),
        "organized": _config["organize_by_type"],
        "categories": {}
    }
    
    for category, subcategories in _config["output_structure"].items():
        cat_path = base_dir / category
        cat_info = {
            "exists": cat_path.exists(),
            "subcategories": {}
        }
        
        if cat_path.exists():
            for subcat in subcategories:
                subcat_path = cat_path / subcat
                if subcat_path.exists():
                    files = list(subcat_path.glob("*"))
                    cat_info["subcategories"][subcat] = {
                        "exists": True,
                        "file_count": len([f for f in files if f.is_file()]),
                        "files": [f.name for f in files if f.is_file()][:10]  # First 10 files
                    }
                else:
                    cat_info["subcategories"][subcat] = {"exists": False, "file_count": 0}
        
        structure_info["categories"][category] = cat_info
    
    return structure_info


def save_config(filename: str = "visualization_config.json") -> None:
    """Save current configuration to file."""
    output_dir = ensure_output_dir()
    config_path = output_dir / filename

    with open(config_path, 'w') as f:
        json.dump(_config, f, indent=2)


def load_config(filename: str = "visualization_config.json") -> None:
    """Load configuration from file."""
    output_dir = ensure_output_dir()
    config_path = output_dir / filename

    if config_path.exists():
        with open(config_path) as f:
            loaded_config = json.load(f)
            set_config(loaded_config)


# Import visualization modules
from .geometry import *
from .mathematical import *
from .numbers import *


# Utility functions
def batch_visualize(visualization_tasks: List[Dict[str, Any]],
                   backend: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Execute multiple visualization tasks in batch.

    Args:
        visualization_tasks: List of visualization task dictionaries
        backend: Visualization backend to use for all tasks

    Returns:
        List[Dict]: Results of all visualization tasks

    Examples:
        >>> tasks = [
        ...     {'function': 'plot_polyhedron', 'args': ['tetrahedron']},
        ...     {'function': 'plot_scheherazade_pattern', 'args': [6]}
        ... ]
        >>> results = batch_visualize(tasks)
    """
    results = []

    for task in visualization_tasks:
        func_name = task.get('function')
        args = task.get('args', [])
        kwargs = task.get('kwargs', {})

        if backend:
            kwargs['backend'] = backend

        # Map function names to actual functions
        func_map = {
            # Original functions
            'plot_polyhedron': plot_polyhedron,
            'plot_quadray_coordinate': plot_quadray_coordinate,
            'plot_ivm_lattice': plot_ivm_lattice,
            'plot_palindromic_pattern': plot_palindromic_pattern,
            'plot_scheherazade_pattern': plot_scheherazade_pattern,
            'plot_primorial_distribution': plot_primorial_distribution,
            'plot_continued_fraction': plot_continued_fraction,
            'plot_base_conversion': plot_base_conversion,
            'plot_pattern_analysis': plot_pattern_analysis,
            'plot_ssrcd_analysis': plot_ssrcd_analysis,

            # New enhanced geometry visualizations
            'plot_polyhedron_3d': plot_polyhedron_3d,
            'plot_polyhedron_graphical_abstract': plot_polyhedron_graphical_abstract,
            'plot_polyhedron_wireframe': plot_polyhedron_wireframe,

            # New number visualizations
            'plot_palindromic_heatmap': plot_palindromic_heatmap,
            'plot_scheherazade_network': plot_scheherazade_network,
            'plot_primorial_spectrum': plot_primorial_spectrum,

            # New mathematical visualizations
            'plot_continued_fraction_convergence': plot_continued_fraction_convergence,
            'plot_pattern_analysis_radar': plot_pattern_analysis_radar,

            # Animation
            'create_animation': create_animation,
        }

        if func_name in func_map:
            try:
                result = func_map[func_name](*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'task': task
                })
        else:
            results.append({
                'error': f'Unknown visualization function: {func_name}',
                'task': task
            })

    return results


def create_animation(frames: List[Dict[str, Any]],
                    output_filename: str = "animation",
                    fps: Optional[int] = None) -> Dict[str, Any]:
    """
    Create an animated GIF from a sequence of frame specifications.

    Each frame renders a rotating synergetic polyhedron. A frame dict may
    override:
        'polyhedron': name understood by plot_polyhedron ('tetrahedron',
            'octahedron', 'cube', 'cuboctahedron'; default per frame index)
        'azimuth': view azimuth in degrees (default: evenly spread over 360)
        'elevation': view elevation in degrees (default 20)

    Args:
        frames: List of frame specifications (dicts); length gives the frame count
        output_filename: Output filename (without extension)
        fps: Frames per second (defaults to the configured animation fps)

    Returns:
        Dict: Animation metadata and file paths
    """
    if not frames:
        raise ValueError("create_animation requires at least one frame")

    fps = fps or _config["animation"]["fps"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
    except ImportError as e:
        raise ImportError(f"Animation requires matplotlib and pillow: {e}")

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from ..geometry.polyhedra import (
        create_cube,
        create_cuboctahedron,
        create_octahedron,
        create_unit_tetrahedron,
    )

    shape_factories = {
        'tetrahedron': create_unit_tetrahedron,
        'octahedron': create_octahedron,
        'cube': create_cube,
        'cuboctahedron': create_cuboctahedron,
    }
    default_shapes = ['tetrahedron', 'octahedron', 'cube', 'cuboctahedron']
    colors_cfg = get_config()["colors"]
    frame_images = []
    for index, frame in enumerate(frames):
        shape = frame.get('polyhedron', default_shapes[index % len(default_shapes)])
        if shape not in shape_factories:
            raise ValueError(
                f"Unknown polyhedron '{shape}'. Available: {sorted(shape_factories)}")
        azimuth = frame.get('azimuth', 360.0 * index / len(frames))
        elevation = frame.get('elevation', 20.0)

        polyhedron = shape_factories[shape]()
        vertices = np.array(polyhedron.to_xyz_vertices())
        vertex_index = {v: i for i, v in enumerate(polyhedron.vertices)}

        _validate_canvas(get_config()["figure_size"],
                         max(get_config()["dpi"], float(plt.rcParams["figure.dpi"])))
        figure = plt.figure(figsize=get_config()["figure_size"])
        axis = figure.add_subplot(111, projection='3d')
        face_tris = [
            [vertices[vertex_index[v]] for v in face[m:m + 3]]
            for face in polyhedron.faces()
            for m in range(1, len(face) - 1)
        ]
        axis.add_collection3d(Poly3DCollection(
            face_tris, facecolors=colors_cfg["primary"], alpha=0.35,
            edgecolors=colors_cfg["secondary"]))
        axis.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                     color=colors_cfg["accent"], s=30)
        axis.set_box_aspect((1, 1, 1))
        axis.set_axis_off()
        axis.view_init(elev=elevation, azim=azimuth)
        figure.canvas.draw()
        buffer = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        frame_images.append(Image.fromarray(buffer))
        plt.close(figure)

    output_path = get_organized_output_path('geometric', 'animations',
                                            f"{output_filename}.gif")
    frame_duration_ms = int(1000 / fps)
    frame_images[0].save(
        str(output_path), save_all=True, append_images=frame_images[1:],
        duration=frame_duration_ms, loop=0)

    return {
        'files': [str(output_path)],
        'metadata': {
            'type': 'animation',
            'frames': len(frames),
            'fps': fps,
            'duration': len(frames) / fps,
            'output_filename': output_filename,
            'backend': 'matplotlib'
        }
    }


def export_visualization(data: Dict[str, Any],
                        format: str = "json",
                        filename: Optional[str] = None) -> str:
    """
    Export visualization metadata to a file.

    Args:
        data: Visualization data to export
        format: Export format ('json', 'yaml', 'txt')
        filename: Output filename

    Returns:
        str: Path to exported file
    """
    if filename is None:
        import time
        timestamp = int(time.time())
        filename = f"visualization_export_{timestamp}"

    output_dir = ensure_output_dir()

    if format == "json":
        filepath = output_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format == "txt":
        filepath = output_dir / f"{filename}.txt"
        with open(filepath, 'w') as f:
            f.write(f"Visualization Export\n{'='*50}\n\n")
            f.write(json.dumps(data, indent=2, default=str))
    else:
        raise ValueError(f"Unsupported export format: {format}")

    return str(filepath)


def batch_process(data_list: List[Any],
                 visualization_func,
                 **kwargs) -> List[Dict[str, Any]]:
    """
    Apply a visualization function to a list of data items.

    Args:
        data_list: List of data items to visualize
        visualization_func: Visualization function to apply
        **kwargs: Additional arguments for the visualization function

    Returns:
        List[Dict]: Results of all visualizations
    """
    results = []

    for i, data in enumerate(data_list):
        try:
            result = visualization_func(data, **kwargs)
            results.append(result)
        except Exception as e:
            results.append({
                'error': str(e),
                'index': i,
                'data': str(data)[:100]  # Truncate for readability
            })

    return results


def create_visualization_report(results: List[Dict[str, Any]],
                               title: str = "Visualization Report") -> str:
    """
    Create a summary report of visualization results.

    Args:
        results: List of visualization results
        title: Report title

    Returns:
        str: Path to report file
    """
    output_dir = ensure_output_dir()

    # Generate timestamp for unique filename
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"visualization_report_{timestamp}.txt"
    filepath = output_dir / filename

    successful = 0
    failed = 0
    total_files = 0

    with open(filepath, 'w') as f:
        f.write(f"{title}\n{'='*60}\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total visualizations: {len(results)}\n\n")

        for i, result in enumerate(results):
            if 'error' in result:
                failed += 1
                f.write(f"❌ Visualization {i+1}: FAILED\n")
                f.write(f"   Error: {result['error']}\n")
            else:
                successful += 1
                files = result.get('files', [])
                total_files += len(files)
                f.write(f"✅ Visualization {i+1}: SUCCESS\n")
                f.write(f"   Type: {result.get('metadata', {}).get('type', 'unknown')}\n")
                f.write(f"   Files generated: {len(files)}\n")
                for file_path in files:
                    f.write(f"   - {file_path}\n")

            f.write("\n")

        f.write("Summary:\n")
        f.write(f"- Successful: {successful}\n")
        f.write(f"- Failed: {failed}\n")
        f.write(f"- Total files generated: {total_files}\n")

    return str(filepath)


# Import visualization modules to make functions available
from .advanced import (
    create_comparative_analysis_visualization,
    create_geometric_mnemonics_visualization,
    create_mega_graphical_abstract,
    create_pattern_discovery_visualization,
    create_statistical_analysis_dashboard,
)
from .geometry import (
    plot_ivm_lattice,
    plot_polyhedron,
    plot_polyhedron_3d,
    plot_polyhedron_graphical_abstract,
    plot_polyhedron_wireframe,
    plot_quadray_coordinate,
)
from .mathematical import (
    plot_base_conversion,
    plot_base_conversion_matrix,
    plot_continued_fraction,
    plot_continued_fraction_convergence,
    plot_pattern_analysis,
    plot_pattern_analysis_radar,
)
from .numbers import (
    plot_palindromic_heatmap,
    plot_palindromic_pattern,
    plot_primorial_distribution,
    plot_primorial_spectrum,
    plot_scheherazade_network,
    plot_scheherazade_pattern,
)

__all__ = [
    # Configuration functions
    "set_config",
    "get_config", 
    "reset_config",
    "ensure_output_dir",
    "get_organized_output_path",
    "create_output_structure_readme",
    "list_output_structure",
    
    # Core visualization functions
    "plot_polyhedron",
    "plot_quadray_coordinate", 
    "plot_ivm_lattice",
    "plot_palindromic_pattern",
    "plot_scheherazade_pattern",
    "plot_primorial_distribution",
    "plot_continued_fraction",
    "plot_base_conversion",
    "plot_pattern_analysis",

    # Enhanced geometry visualizations
    "plot_coordinate_transformation",
    "plot_polyhedron_3d",
    "plot_polyhedron_graphical_abstract",
    "plot_polyhedron_wireframe",

    # Advanced visualizations
    "create_comparative_analysis_visualization",
    "create_pattern_discovery_visualization",
    "create_statistical_analysis_dashboard",
    "create_geometric_mnemonics_visualization",
    "create_mega_graphical_abstract",

    # Extended number and mathematical visualizations
    "plot_mnemonic_visualization",
    "plot_palindromic_heatmap",
    "plot_scheherazade_network",
    "plot_primorial_spectrum",
    "plot_ssrcd_analysis",
    "plot_continued_fraction_convergence",
    "plot_base_conversion_matrix",
    "plot_pattern_analysis_radar",
    
    # Utility functions
    "batch_visualize",
    "export_visualization",
    "create_visualization_report",
    "batch_process",
    "create_animation",
    
    # Configuration management
    "save_config",
    "load_config",
]
