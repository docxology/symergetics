"""
Geometry module for Synergetics package.

Contains classes and functions for geometric objects and transformations
in the synergetic coordinate system.
"""

from .polyhedra import (
    Cube,
    Cuboctahedron,
    Octahedron,
    SymergeticsPolyhedron,
    Tetrahedron,
)
from .transformations import (
    apply_transform_function,
    coordinate_transform,
    reflect,
    rotate_around_axis,
    scale,
    translate,
)

__all__ = [
    # Polyhedra
    "SymergeticsPolyhedron",
    "Tetrahedron",
    "Octahedron",
    "Cube",
    "Cuboctahedron",

    # Transformations
    "translate",
    "scale",
    "reflect",
    "coordinate_transform",
    "rotate_around_axis",
    "apply_transform_function"
]
