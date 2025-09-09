"""
Symergetics: A Comprehensive Research Analysis and Design Framework

Based on Buckminster Fuller's Synergetics system, this package implements
exact rational arithmetic for geometric and cosmological calculations.

Core Features:
- Exact rational arithmetic using fractions.Fraction
- Quadray coordinate system for tetrahedral geometry
- Synergetic polyhedron library with precise volume ratios
- Scheherazade number generation and analysis
- Cosmic hierarchy scaling relationships

Author: Symergetics Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Symergetics Team"
__license__ = "MIT"

# Import core classes for easy access
from .core.numbers import SymergeticsNumber
from .core.coordinates import QuadrayCoordinate
from .core.constants import SymergeticsConstants

# Import geometry classes
from .geometry.polyhedra import (
    Tetrahedron,
    Octahedron,
    Cube,
    Cuboctahedron,
    SymergeticsPolyhedron
)

# Import computation functions
from .computation.primorials import primorial, scheherazade_power
from .computation.palindromes import is_palindromic, extract_palindromic_patterns

# Import utilities
from .utils.conversion import (
    rational_to_float,
    xyz_to_quadray,
    quadray_to_xyz,
    float_to_exact_rational
)
from .utils.mnemonics import mnemonic_encode, mnemonic_decode

__all__ = [
    # Core
    "SymergeticsNumber",
    "QuadrayCoordinate",
    "SymergeticsConstants",

    # Geometry
    "Tetrahedron",
    "Octahedron",
    "Cube",
    "Cuboctahedron",
    "SymergeticsPolyhedron",

    # Computation
    "primorial",
    "scheherazade_power",
    "is_palindromic",
    "extract_palindromic_patterns",

    # Utils
    "rational_to_float",
    "xyz_to_quadray",
    "quadray_to_xyz",
    "float_to_exact_rational",
    "mnemonic_encode",
    "mnemonic_decode"
]
