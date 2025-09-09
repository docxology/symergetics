"""
Geometry module for Synergetics package.

Contains classes and functions for geometric objects and transformations
in the synergetic coordinate system.
"""

from .polyhedra import (
    SymergeticsPolyhedron,
    Tetrahedron,
    Octahedron,
    Cube,
    Cuboctahedron
)
from .transformations import (
    translate,
    scale,
    reflect,
    coordinate_transform,
    rotate_around_axis,
    apply_transform_function
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
