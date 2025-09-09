"""
Geometric Visualizations for Synergetics

This module provides visualization capabilities for geometric objects in the
Synergetics framework, including polyhedra, coordinate systems, and transformations.

Features:
- 3D polyhedron visualization (tetrahedron, octahedron, cube, cuboctahedron)
- Quadray coordinate system visualization
- IVM lattice visualization
- Coordinate transformation animations
- Interactive 3D plots with multiple backends

Author: Symergetics Team
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from pathlib import Path
import warnings

from ..core.coordinates import QuadrayCoordinate, urner_embedding
from ..geometry.polyhedra import (
    Tetrahedron, Octahedron, Cube, Cuboctahedron,
    SymergeticsPolyhedron
)
from . import ensure_output_dir, get_organized_output_path
from . import _config


def plot_polyhedron(polyhedron: Union[str, SymergeticsPolyhedron],
                   backend: Optional[str] = None,
                   **kwargs) -> Dict[str, Any]:
    """
    Visualize a polyhedron in 3D space.

    Args:
        polyhedron: Polyhedron to visualize ('tetrahedron', 'octahedron', 'cube', 'cuboctahedron')
                   or SymergeticsPolyhedron instance
        backend: Visualization backend ('matplotlib', 'plotly', 'ascii')
        **kwargs: Additional visualization parameters

    Returns:
        Dict: Visualization metadata and file paths

    Examples:
        >>> plot_polyhedron('tetrahedron')
        {'files': ['output/tetrahedron_3d.png'], 'metadata': {...}}

        >>> plot_polyhedron('cuboctahedron', backend='plotly')
        {'files': ['output/cuboctahedron_3d.html'], 'metadata': {...}}
    """
    if isinstance(polyhedron, str):
        poly_classes = {
            'tetrahedron': Tetrahedron,
            'octahedron': Octahedron,
            'cube': Cube,
            'cuboctahedron': Cuboctahedron
        }

        if polyhedron not in poly_classes:
            raise ValueError(f"Unknown polyhedron: {polyhedron}")

        poly_obj = poly_classes[polyhedron]()
    else:
        poly_obj = polyhedron

    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_polyhedron_matplotlib(poly_obj, **kwargs)
    elif backend == "plotly":
        return _plot_polyhedron_plotly(poly_obj, **kwargs)
    elif backend == "ascii":
        return _plot_polyhedron_ascii(poly_obj, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_polyhedron_matplotlib(polyhedron: SymergeticsPolyhedron,
                               show_edges: bool = True,
                               show_faces: bool = True,
                               show_vertices: bool = True,
                               alpha: float = 0.7,
                               **kwargs) -> Dict[str, Any]:
    """Plot polyhedron using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Create figure
    fig = plt.figure(figsize=_config["figure_size"])
    ax = fig.add_subplot(111, projection='3d')

    # Get vertex coordinates
    vertices_xyz = polyhedron.to_xyz_vertices()
    if not vertices_xyz:
        raise ValueError("No vertices to plot")

    # Convert to numpy array for easier manipulation
    vertices = np.array(vertices_xyz)

    # Plot vertices
    if show_vertices:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                  color=_config["colors"]["primary"], s=50, alpha=1.0)

    # Plot edges
    if show_edges:
        # Get edges from faces (simplified approach)
        try:
            faces = polyhedron.faces()
            if faces:
                for face in faces:
                    face_vertices = np.array([v.to_xyz() for v in face])
                    # Plot face edges
                    for i in range(len(face_vertices)):
                        start = face_vertices[i]
                        end = face_vertices[(i + 1) % len(face_vertices)]
                        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                               color=_config["colors"]["secondary"], linewidth=2)
        except:
            # Fallback: connect all vertices (may not be optimal)
            pass

    # Plot faces
    if show_faces:
        try:
            faces = polyhedron.faces()
            if faces:
                face_colors = plt.cm.viridis(np.linspace(0, 1, len(faces)))
                for i, face in enumerate(faces):
                    face_vertices = np.array([v.to_xyz() for v in face])
                    if len(face_vertices) >= 3:
                        face_collection = Poly3DCollection([face_vertices],
                                                         alpha=alpha,
                                                         facecolors=[face_colors[i]],
                                                         edgecolors=_config["colors"]["secondary"])
                        ax.add_collection3d(face_collection)
        except:
            pass

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{type(polyhedron).__name__} - Volume: {polyhedron.volume()} IVM units')

    # Set equal aspect ratio
    max_range = np.ptp(vertices, axis=0).max() / 2.0
    mid_x = vertices[:, 0].mean()
    mid_y = vertices[:, 1].mean()
    mid_z = vertices[:, 2].mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Save plot using organized structure
    base_filename = f"{type(polyhedron).__name__.lower()}_3d"
    files = []

    for fmt in _config["formats"]:
        if fmt == "html":
            continue  # matplotlib doesn't support HTML directly

        filename = f"{base_filename}.{fmt}"
        filepath = get_organized_output_path('geometric', 'polyhedra', filename)
        fig.savefig(filepath, 
                    dpi=_config["dpi"], 
                    facecolor=_config["png_options"]["facecolor"],
                    bbox_inches=_config["png_options"]["bbox_inches"],
                    pad_inches=_config["png_options"]["pad_inches"],
                    transparent=_config["png_options"]["transparent"])
        files.append(str(filepath))

    plt.close(fig)

    return {
        'files': files,
        'metadata': {
            'type': 'polyhedron_3d',
            'polyhedron': type(polyhedron).__name__,
            'vertices': len(polyhedron.vertices),
            'volume': polyhedron.volume(),
            'backend': 'matplotlib'
        }
    }


def _plot_polyhedron_plotly(polyhedron: SymergeticsPolyhedron,
                           **kwargs) -> Dict[str, Any]:
    """Plot polyhedron using plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for plotly backend")

    # Get vertex coordinates
    vertices_xyz = polyhedron.to_xyz_vertices()
    if not vertices_xyz:
        raise ValueError("No vertices to plot")

    vertices = np.array(vertices_xyz)

    # Create mesh data
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Create figure
    fig = go.Figure()

    # Add vertices
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=_config["colors"]["primary"],
            opacity=1.0
        ),
        name='Vertices'
    ))

    # Add edges (simplified - connect all vertices)
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            fig.add_trace(go.Scatter3d(
                x=[vertices[i, 0], vertices[j, 0]],
                y=[vertices[i, 1], vertices[j, 1]],
                z=[vertices[i, 2], vertices[j, 2]],
                mode='lines',
                line=dict(color=_config["colors"]["secondary"], width=3),
                showlegend=False
            ))

    # Update layout
    fig.update_layout(
        title=f'{type(polyhedron).__name__} - Volume: {polyhedron.volume()} IVM units',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=_config["figure_size"][0] * 100,
        height=_config["figure_size"][1] * 100
    )

    # Save plot using organized structure
    filename = f"{type(polyhedron).__name__.lower()}_3d.html"
    filepath = get_organized_output_path('geometric', 'polyhedra', filename)
    fig.write_html(str(filepath))
    files = [str(filepath)]

    return {
        'files': files,
        'metadata': {
            'type': 'polyhedron_3d',
            'polyhedron': type(polyhedron).__name__,
            'vertices': len(polyhedron.vertices),
            'volume': polyhedron.volume(),
            'backend': 'plotly'
        }
    }


def _plot_polyhedron_ascii(polyhedron: SymergeticsPolyhedron,
                          **kwargs) -> Dict[str, Any]:
    """Create ASCII art representation of polyhedron."""
    poly_name = type(polyhedron).__name__

    # Simple ASCII representations
    ascii_art = {
        'Tetrahedron': """
           /\\
          /  \\
         /____\\
        /\\    /\\
       /__\\__/__\\
      /\\  /\\  /\\
     /__\\/__\\/__\\
""",
        'Octahedron': """
          /\\
         /  \\
        /____\\
        \\    /
         \\  /
          \\/
""",
        'Cube': """
       +-----+
      /     /|
     +-----+ |
     |     | +
     |     |/
     +-----+
""",
        'Cuboctahedron': """
         /\\
        /  \\
       +----+
      /    /|
     +----+ |
     |    | +
     |    |/
     +----+
"""
    }

    art = ascii_art.get(poly_name, f"ASCII representation of {poly_name}")

    # Save to file using organized structure
    filename = f"{poly_name.lower()}_ascii.txt"
    filepath = get_organized_output_path('geometric', 'polyhedra', filename)

    with open(filepath, 'w') as f:
        f.write(f"{poly_name}\n")
        f.write(f"Volume: {polyhedron.volume()} IVM units\n")
        f.write(f"Vertices: {len(polyhedron.vertices)}\n\n")
        f.write("ASCII Art:\n")
        f.write(art)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'polyhedron_ascii',
            'polyhedron': poly_name,
            'vertices': len(polyhedron.vertices),
            'volume': polyhedron.volume(),
            'backend': 'ascii'
        }
    }


def plot_quadray_coordinate(coord: QuadrayCoordinate,
                           show_lattice: bool = True,
                           lattice_size: int = 3,
                           backend: Optional[str] = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Visualize a Quadray coordinate in the IVM lattice.

    Args:
        coord: Quadray coordinate to visualize
        show_lattice: Whether to show the surrounding lattice
        lattice_size: Size of the lattice to display
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_quadray_matplotlib(coord, show_lattice, lattice_size, **kwargs)
    elif backend == "ascii":
        return _plot_quadray_ascii(coord, show_lattice, lattice_size, **kwargs)
    else:
        raise ValueError(f"Unsupported backend for Quadray visualization: {backend}")


def _plot_quadray_matplotlib(coord: QuadrayCoordinate,
                            show_lattice: bool = True,
                            lattice_size: int = 3,
                            **kwargs) -> Dict[str, Any]:
    """Plot Quadray coordinate using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    fig = plt.figure(figsize=_config["figure_size"])
    ax = fig.add_subplot(111, projection='3d')

    # Convert coordinate to XYZ
    x, y, z = coord.to_xyz()

    # Plot the main coordinate
    ax.scatter([x], [y], [z], color=_config["colors"]["primary"],
              s=200, alpha=1.0, label=f'Coordinate ({coord.a},{coord.b},{coord.c},{coord.d})')

    # Show lattice points if requested
    if show_lattice:
        lattice_points = []

        # Generate nearby lattice points
        for a in range(-lattice_size, lattice_size + 1):
            for b in range(-lattice_size, lattice_size + 1):
                for c in range(-lattice_size, lattice_size + 1):
                    for d in range(-lattice_size, lattice_size + 1):
                        if abs(a) + abs(b) + abs(c) + abs(d) <= lattice_size:
                            lattice_coord = QuadrayCoordinate(a, b, c, d)
                            lattice_points.append(lattice_coord.to_xyz())

        if lattice_points:
            lattice_array = np.array(lattice_points)
            ax.scatter(lattice_array[:, 0], lattice_array[:, 1], lattice_array[:, 2],
                      color=_config["colors"]["grid"], s=20, alpha=0.3, label='IVM Lattice')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Quadray Coordinate ({coord.a},{coord.b},{coord.c},{coord.d})')

    # Set equal aspect ratio
    all_points = [(x, y, z)]
    if show_lattice:
        all_points.extend(lattice_points)

    if all_points:
        points_array = np.array(all_points)
        max_range = np.ptp(points_array, axis=0).max() / 2.0
        if max_range > 0:
            mid_x = points_array[:, 0].mean()
            mid_y = points_array[:, 1].mean()
            mid_z = points_array[:, 2].mean()
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()

    # Save plot using organized structure
    filename = f"quadray_coordinate_{coord.a}_{coord.b}_{coord.c}_{coord.d}.png"
    filepath = get_organized_output_path('geometric', 'coordinates', filename)
    fig.savefig(filepath, dpi=_config["dpi"], bbox_inches='tight')
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'quadray_coordinate',
            'coordinate': (coord.a, coord.b, coord.c, coord.d),
            'xyz_position': (x, y, z),
            'show_lattice': show_lattice,
            'lattice_size': lattice_size,
            'backend': 'matplotlib'
        }
    }


def _plot_quadray_ascii(coord: QuadrayCoordinate,
                       show_lattice: bool = True,
                       lattice_size: int = 3,
                       **kwargs) -> Dict[str, Any]:
    """Create ASCII representation of Quadray coordinate."""
    lines = []
    lines.append(f"Quadray Coordinate: ({coord.a}, {coord.b}, {coord.c}, {coord.d})")
    lines.append(f"XYZ Position: {coord.to_xyz()}")
    lines.append("")

    if show_lattice:
        lines.append(f"IVM Lattice (size ±{lattice_size}):")
        lines.append("Nearby coordinates:")

        nearby_coords = []
        for a in range(coord.a - lattice_size, coord.a + lattice_size + 1):
            for b in range(coord.b - lattice_size, coord.b + lattice_size + 1):
                for c in range(coord.c - lattice_size, coord.c + lattice_size + 1):
                    for d in range(coord.d - lattice_size, coord.d + lattice_size + 1):
                        if abs(a) + abs(b) + abs(c) + abs(d) <= lattice_size * 2:
                            nearby_coords.append((a, b, c, d))

        for quad in nearby_coords[:20]:  # Limit output
            marker = " <-- CURRENT" if quad == (coord.a, coord.b, coord.c, coord.d) else ""
            lines.append(f"  {quad}{marker}")

        if len(nearby_coords) > 20:
            lines.append(f"  ... and {len(nearby_coords) - 20} more coordinates")

    # Save to file using organized structure
    filename = f"quadray_coordinate_{coord.a}_{coord.b}_{coord.c}_{coord.d}_ascii.txt"
    filepath = get_organized_output_path('geometric', 'coordinates', filename)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'quadray_coordinate_ascii',
            'coordinate': (coord.a, coord.b, coord.c, coord.d),
            'show_lattice': show_lattice,
            'lattice_size': lattice_size,
            'backend': 'ascii'
        }
    }


def plot_ivm_lattice(size: int = 5,
                    highlight_coordinates: Optional[List[QuadrayCoordinate]] = None,
                    backend: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
    """
    Visualize the Isotropic Vector Matrix (IVM) lattice.

    Args:
        size: Size of the lattice to generate (affects computation time)
        highlight_coordinates: Specific coordinates to highlight
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_ivm_lattice_matplotlib(size, highlight_coordinates, **kwargs)
    elif backend == "ascii":
        return _plot_ivm_lattice_ascii(size, highlight_coordinates, **kwargs)
    else:
        raise ValueError(f"Unsupported backend for IVM lattice: {backend}")


def _plot_ivm_lattice_matplotlib(size: int = 5,
                                highlight_coordinates: Optional[List[QuadrayCoordinate]] = None,
                                **kwargs) -> Dict[str, Any]:
    """Plot IVM lattice using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    fig = plt.figure(figsize=_config["figure_size"])
    ax = fig.add_subplot(111, projection='3d')

    # Generate lattice points
    lattice_points = []
    for a in range(-size, size + 1):
        for b in range(-size, size + 1):
            for c in range(-size, size + 1):
                for d in range(-size, size + 1):
                    if abs(a) + abs(b) + abs(c) + abs(d) <= size:
                        coord = QuadrayCoordinate(a, b, c, d, normalize=False)
                        lattice_points.append(coord.to_xyz())

    if lattice_points:
        lattice_array = np.array(lattice_points)

        # Plot all lattice points
        ax.scatter(lattice_array[:, 0], lattice_array[:, 1], lattice_array[:, 2],
                  color=_config["colors"]["grid"], s=20, alpha=0.4, label='IVM Lattice')

    # Highlight specific coordinates if provided
    if highlight_coordinates:
        for i, coord in enumerate(highlight_coordinates):
            x, y, z = coord.to_xyz()
            ax.scatter([x], [y], [z],
                      color=_config["colors"]["accent"],
                      s=100, alpha=1.0,
                      label=f'Highlight {i+1}: ({coord.a},{coord.b},{coord.c},{coord.d})')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Isotropic Vector Matrix (IVM) Lattice - Size ±{size}')

    # Set equal aspect ratio
    if lattice_points:
        max_range = np.ptp(lattice_array, axis=0).max() / 2.0
        if max_range > 0:
            mid_x = lattice_array[:, 0].mean()
            mid_y = lattice_array[:, 1].mean()
            mid_z = lattice_array[:, 2].mean()
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()

    # Save plot using organized structure
    filename = f"ivm_lattice_size_{size}.png"
    filepath = get_organized_output_path('geometric', 'lattice', filename)
    fig.savefig(filepath, dpi=_config["dpi"], bbox_inches='tight')
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'ivm_lattice',
            'lattice_size': size,
            'total_points': len(lattice_points),
            'highlighted_coordinates': len(highlight_coordinates) if highlight_coordinates else 0,
            'backend': 'matplotlib'
        }
    }


def _plot_ivm_lattice_ascii(size: int = 5,
                           highlight_coordinates: Optional[List[QuadrayCoordinate]] = None,
                           **kwargs) -> Dict[str, Any]:
    """Create ASCII representation of IVM lattice."""
    lines = []
    lines.append(f"Isotropic Vector Matrix (IVM) Lattice - Size ±{size}")
    lines.append("=" * 50)
    lines.append("")

    # Count lattice points
    point_count = 0
    for a in range(-size, size + 1):
        for b in range(-size, size + 1):
            for c in range(-size, size + 1):
                for d in range(-size, size + 1):
                    if abs(a) + abs(b) + abs(c) + abs(d) <= size:
                        point_count += 1

    lines.append(f"Total lattice points: {point_count}")
    lines.append("")

    if highlight_coordinates:
        lines.append("Highlighted coordinates:")
        for i, coord in enumerate(highlight_coordinates):
            lines.append(f"  {i+1}: ({coord.a}, {coord.b}, {coord.c}, {coord.d}) -> {coord.to_xyz()}")
        lines.append("")

    lines.append("IVM Properties:")
    lines.append("- Each point has 12 nearest neighbors")
    lines.append("- Forms cuboctahedral coordination")
    lines.append("- Used in sphere packing and crystal structures")
    lines.append("- Related to Fuller's vector equilibrium")

    # Save to file
    output_dir = ensure_output_dir()
    filename = f"ivm_lattice_size_{size}_ascii.txt"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'ivm_lattice_ascii',
            'lattice_size': size,
            'total_points': point_count,
            'highlighted_coordinates': len(highlight_coordinates) if highlight_coordinates else 0,
            'backend': 'ascii'
        }
    }


def plot_coordinate_transformation(coord: QuadrayCoordinate,
                                 transform_func,
                                 steps: int = 20,
                                 backend: Optional[str] = None,
                                 **kwargs) -> Dict[str, Any]:
    """
    Visualize a coordinate transformation as an animation or sequence.

    Args:
        coord: Initial coordinate to transform
        transform_func: Function that takes a coordinate and returns transformed coordinate
        steps: Number of animation steps
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_transformation_matplotlib(coord, transform_func, steps, **kwargs)
    else:
        # For other backends, just show start and end positions
        end_coord = transform_func(coord)
        return {
            'files': [],
            'metadata': {
                'type': 'coordinate_transformation',
                'start': (coord.a, coord.b, coord.c, coord.d),
                'end': (end_coord.a, end_coord.b, end_coord.c, end_coord.d),
                'backend': backend
            }
        }


def plot_polyhedron_3d(polyhedron: Union[str, SymergeticsPolyhedron],
                      show_wireframe: bool = True,
                      show_surface: bool = True,
                      elevation: float = 20,
                      azimuth: float = 45,
                      backend: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
    """
    Create an enhanced 3D visualization of polyhedra with customizable viewing angles.

    Args:
        polyhedron: Polyhedron to visualize ('tetrahedron', 'octahedron', 'cube', 'cuboctahedron')
                   or SymergeticsPolyhedron instance
        show_wireframe: Whether to show wireframe edges
        show_surface: Whether to show filled surfaces
        elevation: Camera elevation angle in degrees
        azimuth: Camera azimuth angle in degrees
        backend: Visualization backend ('matplotlib', 'plotly')
        **kwargs: Additional visualization parameters

    Returns:
        Dict: Visualization metadata and file paths

    Examples:
        >>> plot_polyhedron_3d('cuboctahedron', elevation=30, azimuth=60)
        {'files': ['output/geometric/polyhedra/cuboctahedron_3d_enhanced.png'], 'metadata': {...}}
    """
    if isinstance(polyhedron, str):
        poly_classes = {
            'tetrahedron': Tetrahedron,
            'octahedron': Octahedron,
            'cube': Cube,
            'cuboctahedron': Cuboctahedron
        }

        if polyhedron not in poly_classes:
            raise ValueError(f"Unknown polyhedron: {polyhedron}")

        poly_obj = poly_classes[polyhedron]()
    else:
        poly_obj = polyhedron

    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_polyhedron_3d_matplotlib(poly_obj, show_wireframe, show_surface,
                                            elevation, azimuth, **kwargs)
    elif backend == "plotly":
        return _plot_polyhedron_3d_plotly(poly_obj, show_wireframe, show_surface,
                                        elevation, azimuth, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_polyhedron_3d_matplotlib(polyhedron: SymergeticsPolyhedron,
                                 show_wireframe: bool = True,
                                 show_surface: bool = True,
                                 elevation: float = 20,
                                 azimuth: float = 45,
                                 **kwargs) -> Dict[str, Any]:
    """Create enhanced 3D matplotlib visualization."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    fig = plt.figure(figsize=(_config["figure_size"][0] * 1.2, _config["figure_size"][1] * 1.2))
    ax = fig.add_subplot(111, projection='3d')

    vertices_xyz = polyhedron.to_xyz_vertices()
    if not vertices_xyz:
        raise ValueError("No vertices to plot")

    vertices = np.array(vertices_xyz)

    # Enhanced wireframe rendering
    if show_wireframe:
        # Get edges from faces
        try:
            faces = polyhedron.faces()
            if faces:
                for face in faces:
                    face_vertices = np.array([v.to_xyz() for v in face])
                    for i in range(len(face_vertices)):
                        start = face_vertices[i]
                        end = face_vertices[(i + 1) % len(face_vertices)]
                        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                               color=_config["colors"]["secondary"], linewidth=2, alpha=0.8)
        except:
            pass

    # Enhanced surface rendering
    if show_surface:
        try:
            faces = polyhedron.faces()
            if faces:
                face_colors = plt.cm.viridis(np.linspace(0, 1, len(faces)))
                for i, face in enumerate(faces):
                    face_vertices = np.array([v.to_xyz() for v in face])
                    if len(face_vertices) >= 3:
                        # Create surface with lighting effects
                        face_collection = Poly3DCollection([face_vertices],
                                                         alpha=0.7,
                                                         facecolors=[face_colors[i]],
                                                         edgecolors=_config["colors"]["secondary"],
                                                         linewidths=1)
                        ax.add_collection3d(face_collection)
        except:
            pass

    # Add vertex labels
    for i, vertex in enumerate(vertices):
        ax.text(vertex[0], vertex[1], vertex[2], f'V{i}',
               color=_config["colors"]["primary"], fontsize=8, ha='center')

    # Set camera position
    ax.view_init(elev=elevation, azim=azimuth)

    # Enhanced labels and styling
    ax.set_xlabel('X', fontsize=_config["fonts"]["label"]["size"])
    ax.set_ylabel('Y', fontsize=_config["fonts"]["label"]["size"])
    ax.set_zlabel('Z', fontsize=_config["fonts"]["label"]["size"])

    title = f'{type(polyhedron).__name__} - 3D Enhanced View\nVolume: {polyhedron.volume()} IVM units'
    ax.set_title(title, fontsize=_config["fonts"]["title"]["size"],
                fontweight=_config["fonts"]["title"]["weight"])

    # Enhanced axis limits
    max_range = np.ptp(vertices, axis=0).max() / 2.0
    mid_x = vertices[:, 0].mean()
    mid_y = vertices[:, 1].mean()
    mid_z = vertices[:, 2].mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add grid and enhance appearance
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(_config["colors"]["background"])

    # Save with enhanced settings
    base_filename = f"{type(polyhedron).__name__.lower()}_3d_enhanced"
    files = []

    for fmt in _config["formats"]:
        if fmt == "html":
            continue

        filename = f"{base_filename}.{fmt}"
        filepath = get_organized_output_path('geometric', 'polyhedra', filename)
        fig.savefig(filepath,
                    dpi=_config["dpi"],
                    facecolor=_config["png_options"]["facecolor"],
                    bbox_inches=_config["png_options"]["bbox_inches"],
                    pad_inches=_config["png_options"]["pad_inches"],
                    transparent=_config["png_options"]["transparent"])
        files.append(str(filepath))

    plt.close(fig)

    return {
        'files': files,
        'metadata': {
            'type': 'polyhedron_3d_enhanced',
            'polyhedron': type(polyhedron).__name__,
            'vertices': len(polyhedron.vertices),
            'volume': polyhedron.volume(),
            'view_elevation': elevation,
            'view_azimuth': azimuth,
            'show_wireframe': show_wireframe,
            'show_surface': show_surface,
            'backend': 'matplotlib'
        }
    }


def plot_polyhedron_graphical_abstract(polyhedron: Union[str, SymergeticsPolyhedron],
                                     show_volume_ratios: bool = True,
                                     show_coordinates: bool = True,
                                     backend: Optional[str] = None,
                                     **kwargs) -> Dict[str, Any]:
    """
    Create a graphical abstract visualization combining multiple views and data.

    Args:
        polyhedron: Polyhedron to visualize
        show_volume_ratios: Whether to display volume relationships
        show_coordinates: Whether to show coordinate values
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    if isinstance(polyhedron, str):
        poly_classes = {
            'tetrahedron': Tetrahedron,
            'octahedron': Octahedron,
            'cube': Cube,
            'cuboctahedron': Cuboctahedron
        }

        if polyhedron not in poly_classes:
            raise ValueError(f"Unknown polyhedron: {polyhedron}")

        poly_obj = poly_classes[polyhedron]()
    else:
        poly_obj = polyhedron

    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_polyhedron_graphical_abstract_matplotlib(poly_obj,
                                                           show_volume_ratios,
                                                           show_coordinates,
                                                           **kwargs)
    else:
        raise ValueError(f"Graphical abstract requires matplotlib backend, got: {backend}")


def _plot_polyhedron_graphical_abstract_matplotlib(polyhedron: SymergeticsPolyhedron,
                                                show_volume_ratios: bool = True,
                                                show_coordinates: bool = True,
                                                **kwargs) -> Dict[str, Any]:
    """Create graphical abstract using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Create a 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    vertices_xyz = polyhedron.to_xyz_vertices()
    if not vertices_xyz:
        raise ValueError("No vertices to plot")

    vertices = np.array(vertices_xyz)

    # Subplot 1: 3D View
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               color=_config["colors"]["primary"], s=50, alpha=0.8)
    ax1.set_title('3D Structure', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Subplot 2: 2D Projections
    ax2.scatter(vertices[:, 0], vertices[:, 1], color=_config["colors"]["secondary"], s=50, alpha=0.8)
    ax2.set_title('XY Projection', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Subplot 3: Volume Information
    if show_volume_ratios:
        polyhedra_info = {
            'Tetrahedron': {'volume': 1, 'color': '#1f77b4'},
            'Octahedron': {'volume': 4, 'color': '#ff7f0e'},
            'Cube': {'volume': 3, 'color': '#2ca02c'},
            'Cuboctahedron': {'volume': 20, 'color': '#d62728'}
        }

        names = list(polyhedra_info.keys())
        volumes = [polyhedra_info[name]['volume'] for name in names]
        colors = [polyhedra_info[name]['color'] for name in names]

        bars = ax3.bar(names, volumes, color=colors, alpha=0.7)
        ax3.set_title('Volume Ratios', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Volume (IVM units)')

        # Add value labels on bars
        for bar, volume in zip(bars, volumes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{volume}', ha='center', va='bottom', fontweight='bold')

    # Subplot 4: Coordinate Information
    if show_coordinates:
        coord_text = f"Polyhedron: {type(polyhedron).__name__}\n"
        coord_text += f"Volume: {polyhedron.volume()} IVM units\n"
        coord_text += f"Vertices: {len(polyhedron.vertices)}\n\n"

        if show_coordinates and len(vertices) <= 8:  # Only show if reasonable number
            coord_text += "Vertex Coordinates:\n"
            for i, vertex in enumerate(vertices):
                coord_text += ".2f"

        ax4.text(0.1, 0.5, coord_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
        ax4.set_title('Geometric Properties', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

    # Overall title
    fig.suptitle(f'{type(polyhedron).__name__} - Graphical Abstract',
                fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()

    # Save the graphical abstract
    base_filename = f"{type(polyhedron).__name__.lower()}_graphical_abstract"
    files = []

    for fmt in _config["formats"]:
        if fmt == "html":
            continue

        filename = f"{base_filename}.{fmt}"
        filepath = get_organized_output_path('geometric', 'polyhedra', filename)
        fig.savefig(filepath,
                    dpi=_config["dpi"],
                    facecolor=_config["png_options"]["facecolor"],
                    bbox_inches=_config["png_options"]["bbox_inches"],
                    pad_inches=_config["png_options"]["pad_inches"])
        files.append(str(filepath))

    plt.close(fig)

    return {
        'files': files,
        'metadata': {
            'type': 'polyhedron_graphical_abstract',
            'polyhedron': type(polyhedron).__name__,
            'vertices': len(polyhedron.vertices),
            'volume': polyhedron.volume(),
            'show_volume_ratios': show_volume_ratios,
            'show_coordinates': show_coordinates,
            'backend': 'matplotlib'
        }
    }


def plot_polyhedron_wireframe(polyhedron: Union[str, SymergeticsPolyhedron],
                             elevation: float = 20,
                             azimuth: float = 45,
                             backend: Optional[str] = None,
                             **kwargs) -> Dict[str, Any]:
    """
    Create a clean wireframe visualization focusing on structural edges.

    Args:
        polyhedron: Polyhedron to visualize
        elevation: Camera elevation angle
        azimuth: Camera azimuth angle
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    if isinstance(polyhedron, str):
        poly_classes = {
            'tetrahedron': Tetrahedron,
            'octahedron': Octahedron,
            'cube': Cube,
            'cuboctahedron': Cuboctahedron
        }

        if polyhedron not in poly_classes:
            raise ValueError(f"Unknown polyhedron: {polyhedron}")

        poly_obj = poly_classes[polyhedron]()
    else:
        poly_obj = polyhedron

    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_polyhedron_wireframe_matplotlib(poly_obj, elevation, azimuth, **kwargs)
    else:
        raise ValueError(f"Wireframe visualization requires matplotlib backend, got: {backend}")


def _plot_polyhedron_wireframe_matplotlib(polyhedron: SymergeticsPolyhedron,
                                       elevation: float = 20,
                                       azimuth: float = 45,
                                       **kwargs) -> Dict[str, Any]:
    """Create wireframe visualization using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    fig = plt.figure(figsize=_config["figure_size"])
    ax = fig.add_subplot(111, projection='3d')

    vertices_xyz = polyhedron.to_xyz_vertices()
    if not vertices_xyz:
        raise ValueError("No vertices to plot")

    vertices = np.array(vertices_xyz)

    # Plot vertices as small dots
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
              color=_config["colors"]["primary"], s=30, alpha=0.6)

    # Draw edges
    try:
        faces = polyhedron.faces()
        if faces:
            for face in faces:
                face_vertices = np.array([v.to_xyz() for v in face])
                # Close the face by connecting last vertex to first
                face_vertices = np.vstack([face_vertices, face_vertices[0]])

                # Plot edges
                ax.plot(face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2],
                       color=_config["colors"]["secondary"], linewidth=2, alpha=0.8)
    except:
        # Fallback: connect all vertices (may create extra edges)
        pass

    # Set camera position
    ax.view_init(elev=elevation, azim=azimuth)

    # Clean styling - minimal labels
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(f'{type(polyhedron).__name__} - Wireframe Structure',
                fontsize=12, fontweight='bold')

    # Set equal aspect ratio
    max_range = np.ptp(vertices, axis=0).max() / 2.0
    mid_x = vertices[:, 0].mean()
    mid_y = vertices[:, 1].mean()
    mid_z = vertices[:, 2].mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Remove grid for cleaner look
    ax.grid(False)

    # Save wireframe visualization
    base_filename = f"{type(polyhedron).__name__.lower()}_wireframe"
    files = []

    for fmt in _config["formats"]:
        if fmt == "html":
            continue

        filename = f"{base_filename}.{fmt}"
        filepath = get_organized_output_path('geometric', 'polyhedra', filename)
        fig.savefig(filepath,
                    dpi=_config["dpi"],
                    facecolor='white',
                    bbox_inches='tight',
                    pad_inches=0.1)
        files.append(str(filepath))

    plt.close(fig)

    return {
        'files': files,
        'metadata': {
            'type': 'polyhedron_wireframe',
            'polyhedron': type(polyhedron).__name__,
            'vertices': len(polyhedron.vertices),
            'volume': polyhedron.volume(),
            'view_elevation': elevation,
            'view_azimuth': azimuth,
            'backend': 'matplotlib'
        }
    }


def _plot_transformation_matplotlib(coord: QuadrayCoordinate,
                                   transform_func,
                                   steps: int = 20,
                                   **kwargs) -> Dict[str, Any]:
    """Plot coordinate transformation using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation
    except ImportError:
        raise ImportError("matplotlib is required for transformation animation")

    fig = plt.figure(figsize=_config["figure_size"])
    ax = fig.add_subplot(111, projection='3d')

    # Generate transformation path
    path_coords = [coord]
    current = coord

    for _ in range(steps):
        try:
            current = transform_func(current)
            path_coords.append(current)
        except:
            break  # Stop if transformation fails

    # Convert to XYZ coordinates
    path_xyz = [c.to_xyz() for c in path_coords]
    path_array = np.array(path_xyz)

    # Plot transformation path
    if len(path_array) > 1:
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
               color=_config["colors"]["primary"], linewidth=3,
               label='Transformation Path')

        # Plot start and end points
        ax.scatter([path_array[0, 0]], [path_array[0, 1]], [path_array[0, 2]],
                  color=_config["colors"]["secondary"], s=150, alpha=1.0,
                  label='Start')

        ax.scatter([path_array[-1, 0]], [path_array[-1, 1]], [path_array[-1, 2]],
                  color=_config["colors"]["accent"], s=150, alpha=1.0,
                  label='End')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Coordinate Transformation ({len(path_coords)} steps)')

    ax.legend()

    # Save plot using organized structure
    filename = f"coordinate_transformation_{len(path_coords)}_steps.png"
    filepath = get_organized_output_path('geometric', 'transformations', filename)
    fig.savefig(filepath, dpi=_config["dpi"], bbox_inches='tight')
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'coordinate_transformation',
            'steps': len(path_coords),
            'start_coordinate': (coord.a, coord.b, coord.c, coord.d),
            'end_coordinate': (path_coords[-1].a, path_coords[-1].b, path_coords[-1].c, path_coords[-1].d),
            'backend': 'matplotlib'
        }
    }
