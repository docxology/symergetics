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
from fractions import Fraction
from itertools import product
from operator import index
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .numbers import SymergeticsNumber


def _as_fraction(value) -> Fraction:
    """Keep exact numeric inputs exact, including SymergeticsNumber values."""
    if isinstance(value, SymergeticsNumber):
        value = value.value
    return Fraction(value)


def _snap_to_int(value) -> int:
    """Round exact numeric values to integers, with halves away from zero."""
    value = _as_fraction(value)
    whole, remainder = divmod(abs(value.numerator), value.denominator)
    rounded = whole + (2 * remainder >= value.denominator)
    return -rounded if value < 0 else rounded


def _validated_embedding(embedding: Optional[np.ndarray]) -> np.ndarray:
    """Validate a finite rank-three embedding with the all-ones gauge kernel.

    Row sums may differ from zero only by floating-point summation roundoff.
    The first three columns define the embedding in the d=0 gauge; the last
    column is consequently their negative sum.
    """
    if embedding is None:
        return urner_embedding()
    matrix = np.asarray(embedding, dtype=float)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        raise ValueError("embedding must be a finite 3x4 matrix")
    tolerance = 32 * np.finfo(float).eps
    for row in matrix:
        magnitude = max(abs(v) for v in row)
        if magnitude and abs(math.fsum(v / magnitude for v in row)) > tolerance:
            raise ValueError("embedding must annihilate the all-ones gauge")
    a, b, c = [[Fraction(v) for v in row[:3]] for row in matrix]
    determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                   - a[1] * (b[0] * c[2] - b[2] * c[0])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]))
    if not determinant:
        raise ValueError("embedding must have rank three")
    return matrix


def _canonical(values):
    minimum = min(values)
    return tuple(value - minimum for value in values)


def _nearest_quadray(matrix, target):
    """Solve the dimension-three closest-vector problem with exact bounds.

    Binary floating-point matrix entries are interpreted as exact rationals.
    A regular tetrahedral embedding has a body-centered cubic image; otherwise
    exact lattice reduction and closest-first sphere enumeration visit every
    possible minimizer, with coefficients mapped back to the original axes.
    """
    basis = [tuple(Fraction(matrix[i, j]) for i in range(3)) for j in range(3)]
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    fourth = tuple(-sum(column[i] for column in basis) for i in range(3))
    columns = basis + [fourth]
    scale_squared = dot(columns[0], columns[0]) / 3
    if all(dot(a, b) == scale_squared * (3 if i == j else -1)
           for i, a in enumerate(columns) for j, b in enumerate(columns)):
        # E = R U with R.T R = scale_squared I. In U's Cartesian frame,
        # lattice sites have either three even or three odd integer coordinates.
        urner = ((1, -1, -1, 1), (1, 1, -1, -1), (1, -1, 1, -1))
        projections = [dot(column, target) for column in columns]
        xyz = [sum(u * p for u, p in zip(row, projections)) / (4 * scale_squared)
               for row in urner]
        candidates = []
        for parity in (0, 1):
            choices = []
            for value in xyz:
                lower = 2 * ((value - parity) // 2) + parity
                upper = lower + 2
                errors = (abs(value - lower), abs(value - upper))
                choices.append([n for n, error in zip((lower, upper), errors)
                                if error == min(errors)])
            for x, y, z in product(*choices):
                coordinate = _canonical(((y + z) // 2, (y - x) // 2,
                                         (z - x) // 2, 0))
                error = sum((a - b) ** 2 for a, b in zip((x, y, z), xyz))
                candidates.append((error, coordinate))
        return min(candidates)[1]

    # Put short directions first: sphere enumeration fixes the last direction
    # first, so a tiny spacing must not create an enormous outer search range.
    # LLL also removes integer shears; a permutation alone cannot handle them.
    order = sorted(range(3), key=lambda i: (dot(basis[i], basis[i]), basis[i]))
    basis = [basis[i] for i in order]
    coefficients = [tuple(int(i == j) for j in range(3)) for i in order]

    def gram_schmidt():
        orthogonal = []
        norms = []
        mu = [[Fraction(0) for _ in range(3)] for _ in range(3)]
        for j, column in enumerate(basis):
            vector = list(column)
            for i in range(j):
                mu[i][j] = dot(column, orthogonal[i]) / norms[i]
                vector = [v - mu[i][j] * u for v, u in zip(vector, orthogonal[i])]
            orthogonal.append(vector)
            norms.append(dot(vector, vector))
        return orthogonal, norms, mu

    orthogonal, norms, mu = gram_schmidt()
    k = 1
    while k < 3:
        for j in range(k - 1, -1, -1):
            multiple = _snap_to_int(mu[j][k])
            if multiple:
                basis[k] = tuple(v - multiple * u for v, u in zip(basis[k], basis[j]))
                coefficients[k] = tuple(v - multiple * u
                                        for v, u in zip(coefficients[k], coefficients[j]))
                orthogonal, norms, mu = gram_schmidt()
        if norms[k] >= (Fraction(3, 4) - mu[k - 1][k] ** 2) * norms[k - 1]:
            k += 1
        else:
            basis[k], basis[k - 1] = basis[k - 1], basis[k]
            coefficients[k], coefficients[k - 1] = coefficients[k - 1], coefficients[k]
            orthogonal, norms, mu = gram_schmidt()
            k = max(1, k - 1)

    def canonical_candidate():
        original = (sum(chosen[j] * coefficients[j][i] for j in range(3))
                    for i in range(3))
        return _canonical((*original, 0))

    centers = [dot(target, vector) / norm for vector, norm in zip(orthogonal, norms)]
    chosen = [0, 0, 0]
    for i in range(2, -1, -1):
        chosen[i] = _snap_to_int(centers[i] - sum(mu[i][j] * chosen[j]
                                                for j in range(i + 1, 3)))
    best = canonical_candidate()
    best_error = sum(norms[i] * (chosen[i] + sum(mu[i][j] * chosen[j]
                                                for j in range(i + 1, 3))
                                 - centers[i]) ** 2 for i in range(3))

    def search(i, partial_error):
        nonlocal best, best_error
        if i < 0:
            candidate = canonical_candidate()
            if (partial_error, candidate) < (best_error, best):
                best_error, best = partial_error, candidate
            return
        center = centers[i] - sum(mu[i][j] * chosen[j] for j in range(i + 1, 3))
        lower = center.numerator // center.denominator
        upper = lower + 1
        # Merge the integers on either side of the center in exact distance
        # order. Recheck the current radius after each descent, including ties;
        # a better candidate can eliminate the remaining interval immediately.
        while True:
            if center - lower <= upper - center:
                value = lower
                lower -= 1
            else:
                value = upper
                upper += 1
            error = partial_error + norms[i] * (value - center) ** 2
            if error > best_error:
                break
            chosen[i] = value
            search(i - 1, error)

    search(2, Fraction(0))
    return best



class QuadrayCoordinate:
    """
    Four-coordinate tetrahedral coordinate system based on Fuller's Synergetics.

    Implements the Quadray system following the QuadMath reference implementation,
    using integer coordinates with proper normalization and embedding matrices.

    Integer tuples modulo a common additive gauge represent a three-dimensional
    lattice. Normalization subtracts the minimum component. Its integer image is
    distinct from the 12-neighbor IVM/FCC sublattice.

    Attributes:
        a, b, c, d: The four coordinate values (integers)
    """

    def __init__(self, a: Union[int, float, Fraction, SymergeticsNumber],
                 b: Union[int, float, Fraction, SymergeticsNumber],
                 c: Union[int, float, Fraction, SymergeticsNumber],
                 d: Union[int, float, Fraction, SymergeticsNumber],
                 normalize: bool = True):
        """
        Initialize a Quadray coordinate with four values.

        Args:
            a, b, c, d: The four coordinate values
            normalize: Whether to normalize using the IVM convention

        Note:
            Inputs are rounded to integers with exact halves away from zero.
            Integer, Fraction and SymergeticsNumber inputs are not converted to
            float. Normalization then subtracts the minimum integer component.
        """
        self.a = _snap_to_int(a)
        self.b = _snap_to_int(b)
        self.c = _snap_to_int(c)
        self.d = _snap_to_int(d)

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
        Convert to floating-point Cartesian coordinates.

        The optional finite rank-three 3x4 embedding must annihilate the
        all-ones gauge (within summation roundoff). Its first three columns
        define the d=0 gauge. Gauge removal uses exact integer subtraction
        before crossing the floating-point XYZ boundary, so a large common
        offset cannot erase small spatial displacements.

        Args:
            embedding: Gauge-compatible 3x4 matrix; defaults to Urner embedding.

        Returns:
            Tuple[float, float, float]: (x, y, z) Cartesian coordinates
        """
        embedding = _validated_embedding(embedding)
        differences = (self.a - self.d, self.b - self.d, self.c - self.d)
        xyz_vector = embedding[:, :3] @ np.array(differences, dtype=float)
        return tuple(float(value) for value in xyz_vector)

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float,
                 embedding: Optional[np.ndarray] = None) -> 'QuadrayCoordinate':
        """
        Project XYZ onto the nearest integer-quadray lattice site.

        Distances are Euclidean under the supplied finite, gauge-compatible,
        rank-three 3x4 embedding. This projects onto the full integer-quadray
        image, not just its IVM/FCC sublattice. Exact distance ties select the
        lexicographically smallest normalized (a,b,c,d) tuple.

        Integer and rational XYZ inputs stay exact; finite float inputs and
        embedding entries are interpreted as their exact binary values.
        Regular tetrahedral embeddings use a body-centered-cubic fast path;
        skewed embeddings use exhaustive bounded closest-vector enumeration
        in three dimensions, rather than component-wise rounding.
        """
        matrix = _validated_embedding(embedding)
        target = tuple(_as_fraction(value) for value in (x, y, z))
        return cls(*_nearest_quadray(matrix, target))

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


def enumerate_ivm_sites(size: int) -> Tuple[QuadrayCoordinate, ...]:
    """Return unique canonical FCC/IVM sites within ``size`` graph shells.

    ``size`` is a nonnegative integer graph radius about the origin, using
    the twelve ``IVM_NEIGHBORS`` moves (not a Cartesian or coefficient bound).
    The origin comes first, followed by each shell in lexicographic canonical
    coordinate order. Radii 0, 1, 2 and 3 contain 1, 13, 55 and 147 sites.
    Every result has minimum component zero and component sum divisible by four.
    """
    size = index(size)
    if size < 0:
        raise ValueError("IVM graph-shell radius must be nonnegative")
    origin = (0, 0, 0, 0)
    seen = {origin}
    frontier = [origin]
    sites = [QuadrayCoordinate(*origin)]
    moves = [neighbor.as_tuple() for neighbor in IVM_NEIGHBORS]
    for _ in range(size):
        shell = {
            _canonical(tuple(a + b for a, b in zip(site, move)))
            for site in frontier for move in moves
        } - seen
        frontier = sorted(shell)
        seen.update(shell)
        sites.extend(QuadrayCoordinate(*site) for site in frontier)
    return tuple(sites)

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
