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

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..core.coordinates import QuadrayCoordinate, enumerate_ivm_sites, urner_embedding
from ..geometry.polyhedra import (
    Cube,
    Cuboctahedron,
    Octahedron,
    SymergeticsPolyhedron,
    Tetrahedron,
)
from . import _validate_canvas, ensure_output_dir, get_config, get_organized_output_path


def _edge_indices(polyhedron):
    """Declared topology, independent of vertex equality or mutable hashes."""
    return sorted({tuple(sorted((face[i], face[(i + 1) % len(face)])))
                   for face in polyhedron.face_indices() for i in range(len(face))})


def _polyhedron_frame(polyhedron):
    """Subtract an exact anchor before embedding; never subtract rounded XYZ."""
    if not polyhedron.vertices:
        raise ValueError("No vertices to plot")
    anchor = polyhedron.vertices[0]
    return np.asarray(polyhedron.to_xyz_vertices(relative_to=anchor), dtype=float), anchor


def _display_bounds(points):
    points = np.asarray(points, dtype=float)
    low, high = points.min(axis=0), points.max(axis=0)
    center = low + (high - low) / 2
    half = float(np.max(high - low)) * 0.55
    if half == 0:
        half = 1.0
    return [(float(c - half), float(c + half)) for c in center]


def _fit_3d(ax, points, *, relative=False):
    """Fit every displayed point using extrema, with a unit metric box."""
    bounds = _display_bounds(points)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    ax.set_box_aspect((1, 1, 1))
    for axis, label in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel), 'XYZ'):
        axis(f'Δ{label} (anchor-relative)' if relative else label)
    return bounds


def _plotly_scene(points, *, relative=False, elevation=20, azimuth=45):
    elevation, azimuth = np.deg2rad([elevation, azimuth])
    eye = dict(x=2.5 * np.cos(elevation) * np.cos(azimuth),
               y=2.5 * np.cos(elevation) * np.sin(azimuth),
               z=2.5 * np.sin(elevation))
    scene = dict(aspectmode='cube', camera=dict(eye=eye, up=dict(x=0, y=0, z=1)))
    for axis, bounds in zip('xyz', _display_bounds(points)):
        label = f'Δ{axis.upper()} (anchor-relative)' if relative else axis.upper()
        scene[f'{axis}axis'] = dict(title=label, range=bounds)
    return scene


def _geometry_metadata(polyhedron, anchor):
    return dict(coordinate_frame='anchor-relative XYZ (scale 1)',
                origin_quadray_exact=anchor.as_tuple(),
                geometry_analysis=polyhedron.analyze_geometry())


def _draw_polyhedron(ax, polyhedron, vertices, *, show_edges=True,
                     show_faces=True, show_vertices=True, alpha=0.7):
    """Render the same declared triangle fan used by model measurements."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    colors = get_config()['colors']
    if show_faces:
        triangles = [vertices[list(triangle)] for triangle in polyhedron.triangulated_faces()]
        if triangles:
            ax.add_collection3d(Poly3DCollection(
                triangles, facecolors=colors['primary'], alpha=alpha,
                edgecolors='none'))
    if show_edges:
        for a, b in _edge_indices(polyhedron):
            edge = vertices[[a, b]]
            ax.plot(*edge.T, color=colors['secondary'], linewidth=2)
    if show_vertices:
        ax.scatter(*vertices.T, color=colors['primary'], s=40)


def _render_polyhedron_matplotlib(polyhedron, *, suffix, kind, show_edges=True,
                                  show_faces=True, show_vertices=True, alpha=0.7,
                                  elevation=20, azimuth=45):
    import matplotlib.pyplot as plt
    vertices, anchor = _polyhedron_frame(polyhedron)
    config = get_config()
    _validate_canvas(config['figure_size'], max(config['dpi'], float(plt.rcParams['figure.dpi'])))
    fig = plt.figure(figsize=config['figure_size'])
    try:
        ax = fig.add_subplot(111, projection='3d')
        _draw_polyhedron(ax, polyhedron, vertices, show_edges=show_edges,
                         show_faces=show_faces, show_vertices=show_vertices, alpha=alpha)
        _fit_3d(ax, vertices, relative=True)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_title(f'{type(polyhedron).__name__} — {polyhedron.volume()} IVM units\n'
                     f'XYZ relative to quadray anchor {anchor.as_tuple()}')
        files = []
        for fmt in config['formats']:
            if fmt == 'html':
                continue
            path = get_organized_output_path('geometric', 'polyhedra',
                                             f'{type(polyhedron).__name__.lower()}_{suffix}.{fmt}')
            fig.savefig(path, dpi=config['dpi'], **config['png_options'])
            files.append(str(path))
    finally:
        plt.close(fig)
    return dict(files=files, metadata=dict(
        type=kind, polyhedron=type(polyhedron).__name__, vertices=len(vertices),
        volume=polyhedron.volume(), backend='matplotlib',
        view_elevation=elevation, view_azimuth=azimuth,
        show_wireframe=show_edges, show_surface=show_faces,
        **_geometry_metadata(polyhedron, anchor)))


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

    Polyhedron views display scale-1 XYZ displacements from the exact first
    vertex anchor, recorded as ``metadata.origin_quadray_exact``. Surfaces use
    the declared triangle fan, including warped-face interpretation, not a
    convex hull. ``geometry_analysis`` reports live area, solid/vertex centroids
    and validity diagnostics; it is not a self-intersection test.

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

    backend = backend or get_config()["backend"]

    if backend == "matplotlib":
        return _plot_polyhedron_matplotlib(poly_obj, **kwargs)
    elif backend == "plotly":
        return _plot_polyhedron_plotly(poly_obj, **kwargs)
    elif backend == "ascii":
        return _plot_polyhedron_ascii(poly_obj, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_polyhedron_matplotlib(polyhedron: SymergeticsPolyhedron,
                               show_edges: bool = True, show_faces: bool = True,
                               show_vertices: bool = True, alpha: float = 0.7,
                               **kwargs) -> Dict[str, Any]:
    """Plot indexed live geometry in an explicitly anchor-relative frame."""
    return _render_polyhedron_matplotlib(
        polyhedron, suffix='3d', kind='polyhedron_3d', show_edges=show_edges,
        show_faces=show_faces, show_vertices=show_vertices, alpha=alpha,
        elevation=kwargs.get('elevation', 20), azimuth=kwargs.get('azimuth', 45))


def _render_polyhedron_plotly(polyhedron, *, enhanced=False, show_edges=True,
                              show_faces=True, show_vertices=True, alpha=0.3,
                              elevation=20, azimuth=45):
    import plotly.graph_objects as go
    vertices, anchor = _polyhedron_frame(polyhedron)
    x, y, z = vertices.T
    fig = go.Figure()
    colors = get_config()['colors']
    if show_faces:
        triangles = polyhedron.triangulated_faces()
        if triangles:
            i, j, k = zip(*triangles)
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                                   opacity=alpha, color=colors['primary'],
                                   flatshading=True, name='Faces'))
    if show_edges:
        for a, b in _edge_indices(polyhedron):
            edge = vertices[[a, b]]
            fig.add_trace(go.Scatter3d(x=edge[:, 0], y=edge[:, 1], z=edge[:, 2],
                                      mode='lines', line=dict(color=colors['secondary'], width=3),
                                      name='Edges', showlegend=False))
    if show_vertices:
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                  marker=dict(size=8, color=colors['primary']), name='Vertices'))
    fig.update_layout(
        title=f'{type(polyhedron).__name__} — {polyhedron.volume()} IVM units<br>'
              f'XYZ relative to quadray anchor {anchor.as_tuple()}',
        scene=_plotly_scene(vertices, relative=True, elevation=elevation, azimuth=azimuth),
        width=get_config()['figure_size'][0] * 100,
        height=get_config()['figure_size'][1] * 100)
    suffix = '3d_enhanced' if enhanced else '3d'
    path = get_organized_output_path('geometric', 'polyhedra',
                                     f'{type(polyhedron).__name__.lower()}_{suffix}.html')
    fig.write_html(str(path))
    return dict(files=[str(path)], metadata=dict(
        type='polyhedron_3d_enhanced' if enhanced else 'polyhedron_3d',
        polyhedron=type(polyhedron).__name__, vertices=len(vertices), volume=polyhedron.volume(),
        backend='plotly', show_wireframe=show_edges, show_surface=show_faces,
        show_vertices=show_vertices, elevation=elevation, azimuth=azimuth,
        **_geometry_metadata(polyhedron, anchor)))


def _plot_polyhedron_plotly(polyhedron: SymergeticsPolyhedron,
                           show_edges=True, show_faces=True, show_vertices=True,
                           alpha=0.3, **kwargs) -> Dict[str, Any]:
    """Honor the same visibility and viewing contract as Matplotlib."""
    return _render_polyhedron_plotly(
        polyhedron, show_edges=show_edges, show_faces=show_faces,
        show_vertices=show_vertices, alpha=alpha,
        elevation=kwargs.get('elevation', 20), azimuth=kwargs.get('azimuth', 45))


def _plot_polyhedron_3d_plotly(polyhedron: SymergeticsPolyhedron,
                              show_wireframe=True, show_surface=True,
                              elevation=20, azimuth=45, **kwargs) -> Dict[str, Any]:
    """Create enhanced Plotly geometry with the requested physical camera."""
    return _render_polyhedron_plotly(
        polyhedron, enhanced=True, show_edges=show_wireframe,
        show_faces=show_surface, show_vertices=kwargs.get('show_vertices', show_surface),
        elevation=elevation, azimuth=azimuth)


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
        lattice_size: Nonnegative IVM graph-shell radius about the origin
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or get_config()["backend"]

    if backend == "matplotlib":
        return _plot_quadray_matplotlib(coord, show_lattice, lattice_size, **kwargs)
    elif backend == "ascii":
        return _plot_quadray_ascii(coord, show_lattice, lattice_size, **kwargs)
    elif backend == "plotly":
        return _plot_quadray_plotly(coord, show_lattice, lattice_size, **kwargs)
    else:
        raise ValueError(f"Unsupported backend for Quadray visualization: {backend}")


def _plot_quadray_matplotlib(coord: QuadrayCoordinate,
                            show_lattice: bool = True,
                            lattice_size: int = 3,
                            **kwargs) -> Dict[str, Any]:
    """Plot Quadray coordinate using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    lattice_points = (
        [site.to_xyz() for site in enumerate_ivm_sites(lattice_size)] if show_lattice else []
    )
    _validate_canvas(get_config()["figure_size"], max(get_config()["dpi"], float(plt.rcParams["figure.dpi"])))
    fig = plt.figure(figsize=get_config()["figure_size"])
    ax = fig.add_subplot(111, projection='3d')

    # Convert coordinate to XYZ
    x, y, z = coord.to_xyz()

    # Plot the main coordinate
    ax.scatter([x], [y], [z], color=get_config()["colors"]["primary"],
              s=200, alpha=1.0, label=f'Coordinate ({coord.a},{coord.b},{coord.c},{coord.d})')

    # Add direction vectors for quadray axes
    # Define the four fundamental quadray directions
    quadray_directions = [
        QuadrayCoordinate(1, 0, 0, 0),  # A-direction (red)
        QuadrayCoordinate(0, 1, 0, 0),  # B-direction (green)
        QuadrayCoordinate(0, 0, 1, 0),  # C-direction (blue)
        QuadrayCoordinate(0, 0, 0, 1),  # D-direction (yellow)
    ]

    direction_colors = ['red', 'green', 'blue', 'orange']
    direction_labels = ['A-axis', 'B-axis', 'C-axis', 'D-axis']

    # Scale factor for vectors (adjust based on visualization scale)
    vector_scale = 0.8

    for i, (direction, color, label) in enumerate(zip(quadray_directions, direction_colors, direction_labels)):
        # Calculate direction vector in XYZ space
        dir_xyz = direction.to_xyz()

        # Scale the vector
        scaled_dir = np.array(dir_xyz) * vector_scale

        # Plot vector from origin
        ax.quiver(0, 0, 0, scaled_dir[0], scaled_dir[1], scaled_dir[2],
                 color=color, alpha=0.7, linewidth=2, arrow_length_ratio=0.1,
                 label=label if i == 0 else "")  # Only label first vector to avoid clutter

    # Show lattice points if requested
    if show_lattice:
        if lattice_points:
            lattice_array = np.array(lattice_points)
            ax.scatter(lattice_array[:, 0], lattice_array[:, 1], lattice_array[:, 2],
                      color=get_config()["colors"]["grid"], s=20, alpha=0.3, label='IVM Lattice')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Quadray Coordinate ({coord.a},{coord.b},{coord.c},{coord.d})')

    # Fit the point, global lattice context, origin and all basis endpoints.
    all_points = [(x, y, z), (0, 0, 0)]
    all_points.extend(np.array(direction.to_xyz()) * vector_scale for direction in quadray_directions)
    all_points.extend(lattice_points)
    _fit_3d(ax, all_points)

    # Create a custom legend with direction vectors
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=get_config()["colors"]["primary"],
                  markersize=10, label=f'Coordinate ({coord.a},{coord.b},{coord.c},{coord.d})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=get_config()["colors"]["grid"],
                  markersize=6, alpha=0.3, label='IVM Lattice') if show_lattice else None,
        plt.Line2D([0], [0], color='red', linewidth=2, label='A-axis'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='B-axis'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='C-axis'),
        plt.Line2D([0], [0], color='orange', linewidth=2, label='D-axis'),
    ]
    # Filter out None elements
    legend_elements = [elem for elem in legend_elements if elem is not None]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

    # Save plot using organized structure
    filename = f"quadray_coordinate_{coord.a}_{coord.b}_{coord.c}_{coord.d}.png"
    filepath = get_organized_output_path('geometric', 'coordinates', filename)
    fig.savefig(filepath, dpi=get_config()["dpi"], bbox_inches='tight')
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
        lines.append(f"IVM Lattice (graph-shell radius {lattice_size}):")
        lines.append("Lattice coordinates about the origin:")

        nearby_coords = [site.as_tuple() for site in enumerate_ivm_sites(lattice_size)]

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
        size: Nonnegative graph-shell radius about the origin (12-neighbor moves)
        highlight_coordinates: Specific coordinates to highlight
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or get_config()["backend"]

    if backend == "matplotlib":
        return _plot_ivm_lattice_matplotlib(size, highlight_coordinates, **kwargs)
    elif backend == "ascii":
        return _plot_ivm_lattice_ascii(size, highlight_coordinates, **kwargs)
    elif backend == "plotly":
        return _plot_ivm_lattice_plotly(size, highlight_coordinates, **kwargs)
    else:
        raise ValueError(f"Unsupported backend for IVM lattice: {backend}")


def _plot_ivm_lattice_matplotlib(size: int = 5,
                                highlight_coordinates: Optional[List[QuadrayCoordinate]] = None,
                                **kwargs) -> Dict[str, Any]:
    """Plot IVM lattice using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    lattice_points = [site.to_xyz() for site in enumerate_ivm_sites(size)]

    _validate_canvas(get_config()["figure_size"], max(get_config()["dpi"], float(plt.rcParams["figure.dpi"])))
    fig = plt.figure(figsize=get_config()["figure_size"])
    ax = fig.add_subplot(111, projection='3d')

    if lattice_points:
        lattice_array = np.array(lattice_points)

        # Plot all lattice points
        ax.scatter(lattice_array[:, 0], lattice_array[:, 1], lattice_array[:, 2],
                  color=get_config()["colors"]["grid"], s=20, alpha=0.4, label='IVM Lattice')

    # Highlight specific coordinates if provided
    if highlight_coordinates:
        for i, coord in enumerate(highlight_coordinates):
            x, y, z = coord.to_xyz()
            ax.scatter([x], [y], [z],
                      color=get_config()["colors"]["accent"],
                      s=100, alpha=1.0,
                      label=f'Highlight {i+1}: ({coord.a},{coord.b},{coord.c},{coord.d})')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Isotropic Vector Matrix (IVM) Lattice - Graph-shell radius {size}')

    displayed = lattice_points + [coord.to_xyz() for coord in (highlight_coordinates or [])]
    _fit_3d(ax, displayed)

    ax.legend()

    # Save plot using organized structure
    filename = f"ivm_lattice_size_{size}.png"
    filepath = get_organized_output_path('geometric', 'lattice', filename)
    fig.savefig(filepath, dpi=get_config()["dpi"], bbox_inches='tight')
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
    lines.append(f"Isotropic Vector Matrix (IVM) Lattice - Graph-shell radius {size}")
    lines.append("=" * 50)
    lines.append("")

    point_count = len(enumerate_ivm_sites(size))

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
    backend = get_config()["backend"] if backend is None else backend

    if backend == "matplotlib":
        return _plot_transformation_matplotlib(coord, transform_func, steps, **kwargs)
    elif backend in ("ascii", "plotly"):
        # These backends report start and end positions without a saved artifact.
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
    else:
        raise ValueError(f"Unsupported backend for coordinate transformation: {backend}")


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

    backend = backend or get_config()["backend"]

    if backend == "matplotlib":
        return _plot_polyhedron_3d_matplotlib(poly_obj, show_wireframe, show_surface,
                                            elevation, azimuth, **kwargs)
    elif backend == "plotly":
        return _plot_polyhedron_3d_plotly(poly_obj, show_wireframe, show_surface,
                                        elevation, azimuth, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_polyhedron_3d_matplotlib(polyhedron: SymergeticsPolyhedron,
                                 show_wireframe=True, show_surface=True,
                                 elevation=20, azimuth=45, **kwargs) -> Dict[str, Any]:
    """Create an enhanced view of the live indexed surface."""
    return _render_polyhedron_matplotlib(
        polyhedron, suffix='3d_enhanced', kind='polyhedron_3d_enhanced',
        show_edges=show_wireframe, show_faces=show_surface,
        show_vertices=kwargs.get('show_vertices', show_surface),
        elevation=elevation, azimuth=azimuth)


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

    backend = backend or get_config()["backend"]

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
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Create a 2x2 subplot layout
    _validate_canvas((16, 12), max(get_config()["dpi"], float(plt.rcParams["figure.dpi"])))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    vertices, anchor = _polyhedron_frame(polyhedron)
    analysis = polyhedron.analyze_geometry()

    # Subplot 1: 3D View
    ax1.remove()
    ax1 = fig.add_subplot(221, projection='3d')
    _draw_polyhedron(ax1, polyhedron, vertices)
    _fit_3d(ax1, vertices, relative=True)
    ax1.set_title('Anchor-relative XYZ structure', fontsize=12, fontweight='bold')

    # Subplot 2: 2D Projections
    ax2.scatter(vertices[:, 0], vertices[:, 1], color=get_config()["colors"]["secondary"], s=50, alpha=0.8)
    ax2.set_title('XY Projection', fontsize=12, fontweight='bold')
    ax2.set_xlabel('ΔX (anchor-relative)')
    ax2.set_ylabel('ΔY (anchor-relative)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    if show_coordinates:
        # Exact centroid displacement first, including independent input gauges.
        for centroid, label, marker, color in (
            (analysis['vertex_centroid_exact'], 'Vertex centroid', 'x', 'black'),
            (analysis['volume_centroid_exact'], 'Solid centroid', '*', 'crimson'),
        ):
            if centroid is None:
                continue
            a, b, c, d = (value - origin for value, origin in
                          zip(centroid, anchor.as_tuple()))
            point = np.array([float(a - b - c + d), float(a + b - c - d),
                              float(a - b + c - d)])
            ax1.scatter(*point, marker=marker, color=color, s=90, label=label)
            ax2.scatter(*point[:2], marker=marker, color=color, s=90, label=label)
        ax1.legend(fontsize=7)
        ax2.legend(fontsize=7)

    # Subplot 3: Volume Information
    if show_volume_ratios:
        models = [Tetrahedron(), Octahedron(), Cube(), Cuboctahedron()]
        names = [type(model).__name__ for model in models] + ['Current shape']
        volumes = [float(model.volume()) for model in models] + [float(polyhedron.volume())]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        bars = ax3.bar(names, volumes, color=colors, alpha=0.7)
        ax3.set_title('Canonical reference and live volume', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', labelrotation=20)
        ax3.set_ylabel('Volume (IVM units)')

        # Add value labels on bars
        for bar, volume in zip(bars, volumes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{volume}', ha='center', va='bottom', fontweight='bold')

    # Subplot 4: Coordinate Information
    if show_coordinates:
        vertex_centroid = '(' + ', '.join(map(str, analysis['vertex_centroid_exact'])) + ')'
        solid_centroid = (None if analysis['volume_centroid_exact'] is None else
                          '(' + ', '.join(map(str, analysis['volume_centroid_exact'])) + ')')
        coord_text = (
            f"Polyhedron: {type(polyhedron).__name__}\n"
            f"Volume: {analysis['volume_ivm']} IVM; {float(analysis['volume_xyz']):.6g} XYZ³\n"
            f"Surface area: {analysis['surface_area']:.6g} XYZ² (triangle fan)\n"
            f"Vertices: {len(vertices)} declared; {analysis['physical_vertex_count']} physical\n"
            f"Affine rank: {analysis['affine_rank']}; closed topology: {analysis['closed']}\n"
            f"Convex: {analysis['is_convex']}; nonplanar faces: {analysis['nonplanar_faces']}\n"
            f"Exact vertex centroid (quadray):\n{vertex_centroid}\n"
            f"Exact solid centroid (quadray):\n{solid_centroid}\n"
            f"Display origin (exact quadray): {anchor.as_tuple()}\n"
            "Closed, non-self-intersecting triangulated-surface interpretation."
        )

        ax4.text(0.1, 0.5, coord_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
        ax4.set_title('Geometric Properties', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

    # Overall title
    fig.suptitle(f'{type(polyhedron).__name__} - Graphical Abstract',
                fontsize=16, fontweight='bold', y=0.98)

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    # Save the graphical abstract
    base_filename = f"{type(polyhedron).__name__.lower()}_graphical_abstract"
    files = []

    for fmt in get_config()["formats"]:
        if fmt == "html":
            continue

        filename = f"{base_filename}.{fmt}"
        filepath = get_organized_output_path('geometric', 'polyhedra', filename)
        fig.savefig(filepath,
                    dpi=get_config()["dpi"],
                    facecolor=get_config()["png_options"]["facecolor"],
                    bbox_inches=get_config()["png_options"]["bbox_inches"],
                    pad_inches=get_config()["png_options"]["pad_inches"])
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
            'backend': 'matplotlib',
            **_geometry_metadata(polyhedron, anchor)
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

    backend = backend or get_config()["backend"]

    if backend == "matplotlib":
        return _plot_polyhedron_wireframe_matplotlib(poly_obj, elevation, azimuth, **kwargs)
    else:
        raise ValueError(f"Wireframe visualization requires matplotlib backend, got: {backend}")


def _plot_polyhedron_wireframe_matplotlib(polyhedron: SymergeticsPolyhedron,
                                       elevation=20, azimuth=45, **kwargs) -> Dict[str, Any]:
    """Create a wireframe using declared indexed edges."""
    return _render_polyhedron_matplotlib(
        polyhedron, suffix='wireframe', kind='polyhedron_wireframe',
        show_faces=False, elevation=elevation, azimuth=azimuth)


def _plot_transformation_matplotlib(coord: QuadrayCoordinate,
                                   transform_func,
                                   steps: int = 20,
                                   **kwargs) -> Dict[str, Any]:
    """Plot a complete transformation path, propagating callback failures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for transformation animation")

    # Complete the real transformation before creating a figure or output path.
    path_coords = [coord]
    current = coord
    for _ in range(steps):
        current = transform_func(current)
        path_coords.append(current)
    path_array = np.array([c.to_xyz() for c in path_coords])

    _validate_canvas(get_config()["figure_size"], max(get_config()["dpi"], float(plt.rcParams["figure.dpi"])))
    fig = plt.figure(figsize=get_config()["figure_size"])
    try:
        ax = fig.add_subplot(111, projection='3d')
        if len(path_array) > 1:
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                    color=get_config()["colors"]["primary"], linewidth=3,
                    label='Transformation Path')
            ax.scatter([path_array[0, 0]], [path_array[0, 1]], [path_array[0, 2]],
                       color=get_config()["colors"]["secondary"], s=150, alpha=1.0,
                       label='Start')
            ax.scatter([path_array[-1, 0]], [path_array[-1, 1]], [path_array[-1, 2]],
                       color=get_config()["colors"]["accent"], s=150, alpha=1.0,
                       label='End')
            ax.legend()

        _fit_3d(ax, path_array)
        ax.set_title(f'Coordinate Transformation ({len(path_coords)} steps)')

        filename = f"coordinate_transformation_{len(path_coords)}_steps.png"
        filepath = get_organized_output_path('geometric', 'transformations', filename)
        fig.savefig(filepath, dpi=get_config()["dpi"], bbox_inches='tight')
    finally:
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


# ---------------------------------------------------------------------------
# Plotly backends for coordinate and lattice plots
# ---------------------------------------------------------------------------

def _plotly_grid(width_scale: float = 1.0, height_scale: float = 1.0):
    """Return (plotly modules, width, height) for the configured figure size."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for plotly backend")
    size = get_config()["figure_size"]
    return go, int(size[0] * 100 * width_scale), int(size[1] * 100 * height_scale)


def _plot_quadray_plotly(coord: QuadrayCoordinate, show_lattice: bool = True,
                         lattice_size: int = 3, **kwargs) -> Dict[str, Any]:
    """Plot a Quadray coordinate (and optionally its IVM shell) using plotly."""
    go, width, height = _plotly_grid()
    colors_cfg = get_config()["colors"]

    fig = go.Figure()
    if show_lattice:
        lattice_points = [site.to_xyz() for site in enumerate_ivm_sites(lattice_size)]
        if lattice_points:
            arr = np.array(lattice_points)
            fig.add_trace(go.Scatter3d(
                x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode='markers',
                marker=dict(size=3, color=colors_cfg["grid"], opacity=0.5),
                name='IVM Lattice'))

    xyz = np.array(coord.to_xyz())
    fig.add_trace(go.Scatter3d(
        x=[xyz[0]], y=[xyz[1]], z=[xyz[2]], mode='markers',
        marker=dict(size=10, color=colors_cfg["accent"]),
        name=f'Quadray {coord.as_tuple()}'))
    basis = np.array([[1 if j == i else 0 for j in range(4)] for i in range(4)],
                     dtype=float)
    basis_xyz = basis @ urner_embedding().T
    for i, name in enumerate('ABCD'):
        fig.add_trace(go.Scatter3d(
            x=[0, basis_xyz[i, 0]], y=[0, basis_xyz[i, 1]], z=[0, basis_xyz[i, 2]],
            mode='lines', line=dict(color=colors_cfg["secondary"], width=4),
            name=f'Basis {name}', showlegend=i == 0))

    displayed = [coord.to_xyz(), (0, 0, 0), *basis_xyz]
    if show_lattice:
        displayed.extend(lattice_points)
    fig.update_layout(title=f'Quadray Coordinate {coord.as_tuple()}',
                      scene=_plotly_scene(displayed), width=width, height=height)

    filepath = get_organized_output_path('geometric', 'coordinates',
                                         f"quadray_coordinate_plotly_{abs(hash(coord.as_tuple())) % 10000}.html")
    fig.write_html(str(filepath), include_plotlyjs='cdn')

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'quadray_coordinate',
            'coordinate': coord.as_tuple(),
            'show_lattice': show_lattice,
            'lattice_size': lattice_size,
            'backend': 'plotly'
        }
    }


def _plot_ivm_lattice_plotly(size: int = 5,
                             highlight_coordinates: Optional[List[QuadrayCoordinate]] = None,
                             **kwargs) -> Dict[str, Any]:
    """Plot the IVM lattice using plotly."""
    go, width, height = _plotly_grid()
    colors_cfg = get_config()["colors"]

    lattice_points = [site.to_xyz() for site in enumerate_ivm_sites(size)]

    fig = go.Figure()
    if lattice_points:
        arr = np.array(lattice_points)
        fig.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode='markers',
            marker=dict(size=3, color=colors_cfg["grid"], opacity=0.4),
            name='IVM Lattice'))

    if highlight_coordinates:
        highlighted = np.array([c.to_xyz() for c in highlight_coordinates])
        fig.add_trace(go.Scatter3d(
            x=highlighted[:, 0], y=highlighted[:, 1], z=highlighted[:, 2],
            mode='markers', marker=dict(size=8, color=colors_cfg["accent"]),
            name='Highlighted'))

    displayed = lattice_points + [c.to_xyz() for c in (highlight_coordinates or [])]
    fig.update_layout(title=f'IVM Lattice (graph-shell radius {size})',
                      scene=_plotly_scene(displayed), width=width, height=height)

    filepath = get_organized_output_path('geometric', 'lattice',
                                         f"ivm_lattice_plotly_size_{size}.html")
    fig.write_html(str(filepath), include_plotlyjs='cdn')

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'ivm_lattice',
            'size': size,
            'lattice_size': size,
            'total_points': len(lattice_points),
            'lattice_points': len(lattice_points),
            'highlighted': len(highlight_coordinates or []),
            'backend': 'plotly'
        }
    }
