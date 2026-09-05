"""
Quadray Coordinate System for Synergetic Geometry

This module implements Fuller's Quadray coordinate system, a four-dimensional
tetrahedral coordinate system based on the comprehensive QuadMath reference
implementation (https://github.com/docxology/QuadMath).

Key Features:
- Integer-based Quadray coordinates with proper normalization
- Correct Urner embedding matrix for XYZ conversion
- Exact integer volume calculations using determinants
- IVM (Isotropic Vector Matrix) lattice support
- Integration with SymergeticsNumber for rational arithmetic

Author: Symergetics Team
Based on QuadMath implementation by docxology
"""

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .numbers import SymergeticsNumber


def _snap_to_int(value: float) -> int:
    """Round to the nearest integer with exact halves rounding away from zero.

    Python's round() uses banker's rounding, which maps 0.5 -> 0 and would
    snap genuine lattice points (whose pseudoinverse solution has exact .5
    components) to the wrong coordinates.
    """
    return int(math.copysign(math.floor(abs(value) + 0.5), value))



class QuadrayCoordinate:
    """
    Four-coordinate tetrahedral coordinate system based on Fuller's Synergetics.

    Implements the Quadray system following the QuadMath reference implementation,
    using integer coordinates with proper normalization and embedding matrices.

    In the Quadray system, coordinates (a, b, c, d) are non-negative integers
    with at least one coordinate being zero after normalization. This represents
    positions in the Isotropic Vector Matrix (IVM) lattice.

    Attributes:
        a, b, c, d: The four coordinate values (integers)
    """

    def __init__(self, a: Union[int, float, SymergeticsNumber],
                 b: Union[int, float, SymergeticsNumber],
                 c: Union[int, float, SymergeticsNumber],
                 d: Union[int, float, SymergeticsNumber],
                 normalize: bool = True):
        """
        Initialize a Quadray coordinate with four values.

        Args:
            a, b, c, d: The four coordinate values
            normalize: Whether to normalize using the IVM convention

        Note:
            Following QuadMath convention, coordinates are converted to integers
            and normalized so that at least one component is zero.
        """
        self.a = _snap_to_int(float(a.value) if hasattr(a, 'value') else float(a))
        self.b = _snap_to_int(float(b.value) if hasattr(b, 'value') else float(b))
        self.c = _snap_to_int(float(c.value) if hasattr(c, 'value') else float(c))
        self.d = _snap_to_int(float(d.value) if hasattr(d, 'value') else float(d))

        if normalize:
            self._normalize_ivm()

    def _normalize_ivm(self):
        """
        Normalize using IVM convention: subtract min(a,b,c,d) from all components.

        This ensures at least one coordinate is zero and maintains the
        non-negative integer property of the IVM lattice.
        """
        k = min(self.a, self.b, self.c, self.d)
        self.a -= k
        self.b -= k
        self.c -= k
        self.d -= k

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return coordinates as (a, b, c, d) tuple."""
        return (self.a, self.b, self.c, self.d)

    def as_array(self) -> np.ndarray:
        """Return coordinates as numpy array for matrix operations."""
        return np.array([[self.a], [self.b], [self.c], [self.d]], dtype=float)

    def __repr__(self) -> str:
        """String representation of the Quadray coordinate."""
        return f"QuadrayCoordinate(a={self.a}, b={self.b}, c={self.c}, d={self.d})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"({self.a}, {self.b}, {self.c}, {self.d})"

    def __eq__(self, other: 'QuadrayCoordinate') -> bool:
        """Test equality with another QuadrayCoordinate."""
        if not isinstance(other, QuadrayCoordinate):
            return False
        return (self.a == other.a and self.b == other.b and
                self.c == other.c and self.d == other.d)

    def __hash__(self) -> int:
        """Hash function for use in sets and dictionaries."""
        return hash((self.a, self.b, self.c, self.d))

    def add(self, other: 'QuadrayCoordinate') -> 'QuadrayCoordinate':
        """Component-wise addition."""
        return QuadrayCoordinate(
            self.a + other.a,
            self.b + other.b,
            self.c + other.c,
            self.d + other.d,
            normalize=False
        )

    def sub(self, other: 'QuadrayCoordinate') -> 'QuadrayCoordinate':
        """Component-wise subtraction."""
        return QuadrayCoordinate(
            self.a - other.a,
            self.b - other.b,
            self.c - other.c,
            self.d - other.d,
            normalize=False
        )

    def to_xyz(self, embedding: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        Convert Quadray coordinates to Cartesian XYZ coordinates using proper Urner embedding.

        Uses the standard Urner embedding matrix that maps the four quadray axes 
        (A,B,C,D) to the vertices of a regular tetrahedron in R^3, following
        established Quadray coordinate mathematics from Darrel Jarmusch (1981).

        The transformation preserves tetrahedral symmetry and typically maps
        integer Quadray coordinates to non-integer XYZ coordinates, which is
        the expected mathematical behavior.

        Args:
            embedding: Optional 3x4 embedding matrix (uses Urner embedding if None)

        Returns:
            Tuple[float, float, float]: (x, y, z) Cartesian coordinates
        """
        if embedding is None:
            embedding = urner_embedding()
        
        # Apply the Urner embedding matrix transformation
        quadray_vector = np.array([self.a, self.b, self.c, self.d], dtype=float)
        xyz_vector = embedding @ quadray_vector
        
        return (float(xyz_vector[0]), float(xyz_vector[1]), float(xyz_vector[2]))

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float,
                 embedding: Optional[np.ndarray] = None) -> 'QuadrayCoordinate':
        """
        Convert Cartesian XYZ coordinates to Quadray coordinates.

        Uses the Moore-Penrose pseudoinverse of the Urner embedding matrix
        to find the least-squares solution, then normalizes to IVM convention.

        Args:
            x, y, z: Cartesian coordinates
            embedding: Optional 3x4 embedding matrix

        Returns:
            QuadrayCoordinate: Equivalent Quadray coordinates
        """
        if embedding is None:
            embedding = urner_embedding()

        # Use pseudoinverse for the overdetermined system (3 equations, 4 unknowns)
        xyz_vector = np.array([x, y, z], dtype=float)
        
        # Moore-Penrose pseudoinverse
        embedding_pinv = np.linalg.pinv(embedding)
        quadray_solution = embedding_pinv @ xyz_vector
        
        # The (1,1,1,1) kernel splits the lattice into two phase classes:
        # integer quadray solutions and half-integer ones. Try both the
        # unshifted and the half-shifted rounding and keep the candidate
        # whose embedding is closest to the requested point.
        best = None
        best_err = None
        for shift in (0.0, 0.5):
            candidate = tuple(_snap_to_int(coord + shift) for coord in quadray_solution)
            err_vec = embedding @ np.array(candidate, dtype=float) - xyz_vector
            err = float(np.dot(err_vec, err_vec))
            if best_err is None or err < best_err:
                best_err = err
                best = candidate
        a, b, c, d = best
        
        return cls(a, b, c, d, normalize=True)

    def magnitude(self, embedding: Optional[np.ndarray] = None) -> float:
        """
        Calculate Euclidean magnitude using XYZ embedding.

        Args:
            embedding: Optional 3x4 embedding matrix

        Returns:
            float: Euclidean norm of the embedded vector
        """
        x, y, z = self.to_xyz(embedding)
        return float((x * x + y * y + z * z) ** 0.5)

    def dot(self, other: 'QuadrayCoordinate',
            embedding: Optional[np.ndarray] = None) -> float:
        """
        Calculate Euclidean dot product using XYZ embedding.

        Args:
            other: Another Quadray coordinate
            embedding: Optional 3x4 embedding matrix

        Returns:
            float: Dot product in embedded Euclidean space
        """
        x1, y1, z1 = self.to_xyz(embedding)
        x2, y2, z2 = other.to_xyz(embedding)
        return float(x1 * x2 + y1 * y2 + z1 * z2)

    def distance_to(self, other: 'QuadrayCoordinate',
                   embedding: Optional[np.ndarray] = None) -> float:
        """
        Calculate Euclidean distance to another Quadray coordinate.

        Args:
            other: Another Quadray coordinate
            embedding: Optional 3x4 embedding matrix

        Returns:
            float: Euclidean distance between coordinates
        """
        diff = self.sub(other)
        return diff.magnitude(embedding)

    def to_dict(self) -> Dict[str, int]:
        """Convert coordinate to dictionary representation."""
        return {
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'd': self.d
        }

    def copy(self) -> 'QuadrayCoordinate':
        """Create a copy of this coordinate."""
        return QuadrayCoordinate(self.a, self.b, self.c, self.d, normalize=False)


# Helper functions for coordinate transformations
def urner_embedding(scale: float = 1.0) -> np.ndarray:
    """
    Return a 3x4 Urner-style symmetric embedding matrix (Fuller.4D -> Coxeter.4D slice).

    The rows map the four quadray axes (A,B,C,D) to the vertices of a regular
    tetrahedron in R^3. Scaling the matrix scales all resulting XYZ coordinates.

    Parameters:
        scale: Uniform scalar applied to the embedding (default 1.0)

    Returns:
        np.ndarray: A 3x4 matrix suitable for quadray_to_xyz conversion
    """
    M = np.array([
        [1.0, -1.0, -1.0, 1.0],  # X coordinate
        [1.0, 1.0, -1.0, -1.0],  # Y coordinate
        [1.0, -1.0, 1.0, -1.0],  # Z coordinate
    ], dtype=float)
    return scale * M


# Standard Quadray coordinate constants
ORIGIN = QuadrayCoordinate(0, 0, 0, 0)

# IVM neighbors (12-around-one for cuboctahedron/vector equilibrium)
# These are all permutations of {2,1,1,0}
IVM_NEIGHBORS = [
    QuadrayCoordinate(2, 1, 1, 0),
    QuadrayCoordinate(2, 1, 0, 1),
    QuadrayCoordinate(2, 0, 1, 1),
    QuadrayCoordinate(1, 2, 1, 0),
    QuadrayCoordinate(1, 2, 0, 1),
    QuadrayCoordinate(1, 1, 2, 0),
    QuadrayCoordinate(1, 1, 0, 2),
    QuadrayCoordinate(1, 0, 2, 1),
    QuadrayCoordinate(1, 0, 1, 2),
    QuadrayCoordinate(0, 2, 1, 1),
    QuadrayCoordinate(0, 1, 2, 1),
    QuadrayCoordinate(0, 1, 1, 2),
]

# Tetrahedron vertices (unit volume = 1 in IVM units)
TETRAHEDRON_VERTICES = [
    QuadrayCoordinate(0, 0, 0, 0),    # Origin
    QuadrayCoordinate(2, 1, 0, 1),    # Vertex 1
    QuadrayCoordinate(2, 1, 1, 0),    # Vertex 2
    QuadrayCoordinate(2, 0, 1, 1),    # Vertex 3
]

# Cube vertices (volume = 3 in tetrahedra): face diagonal equals the tetra edge
# (cube edge = tetra edge / sqrt(2)). In XYZ these are the 8 points
# (+/-1, +/-1, +/-1), i.e. the 4 Urner basis directions and their complements.
CUBE_VERTICES = [
    QuadrayCoordinate(1, 0, 0, 0),    # +col1
    QuadrayCoordinate(0, 1, 0, 0),    # +col2
    QuadrayCoordinate(0, 0, 1, 0),    # +col3
    QuadrayCoordinate(0, 0, 0, 1),    # +col4
    QuadrayCoordinate(0, 1, 1, 1),    # -col1
    QuadrayCoordinate(1, 0, 1, 1),    # -col2
    QuadrayCoordinate(1, 1, 0, 1),    # -col3
    QuadrayCoordinate(1, 1, 1, 0),    # -col4
]

# Octahedron vertices (volume = 4 in tetrahedra): the 6 IVM points at one edge
# length from the origin along the +axis pair sums, ordered [+X,-X,+Y,-Y,+Z,-Z].
OCTAHEDRON_VERTICES = [
    QuadrayCoordinate(1, 0, 0, 1),    # +X
    QuadrayCoordinate(0, 1, 1, 0),    # -X
    QuadrayCoordinate(1, 1, 0, 0),    # +Y
    QuadrayCoordinate(0, 0, 1, 1),    # -Y
    QuadrayCoordinate(1, 0, 1, 0),    # +Z
    QuadrayCoordinate(0, 1, 0, 1),    # -Z
]
