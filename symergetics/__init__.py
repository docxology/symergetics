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
from .computation.palindromes import extract_palindromic_patterns, is_palindromic

# Import computation functions
from .computation.primorials import primorial, scheherazade_power
from .core.constants import SymergeticsConstants
from .core.coordinates import QuadrayCoordinate
from .core.numbers import SymergeticsNumber

# Import geometry classes
from .geometry.polyhedra import (
    Cube,
    Cuboctahedron,
    Octahedron,
    SymergeticsPolyhedron,
    Tetrahedron,
)

# Import utilities
from .utils.conversion import (
    float_to_exact_rational,
    quadray_to_xyz,
    rational_to_float,
    xyz_to_quadray,
)
from .utils.mnemonics import mnemonic_decode, mnemonic_encode

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
