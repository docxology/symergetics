"""
Polyhedra and Geometric Objects for Synergetic Geometry

This module implements classes for regular polyhedra and geometric calculations
using the Quadray coordinate system and exact integer arithmetic.

Key Features:
- Tetrahedron, Octahedron, Cube, Cuboctahedron classes
- Exact volume calculations using integer determinants
- Vertex, edge, and face calculations
- Integration with Quadray coordinate system

Author: Symergetics Team
Based on QuadMath implementation by docxology
"""

from fractions import Fraction
from typing import List, Optional, Tuple

from ..core.coordinates import (
    CUBE_VERTICES,
    IVM_NEIGHBORS,
    OCTAHEDRON_VERTICES,
    QuadrayCoordinate,
    urner_embedding,
)


def integer_tetra_volume(p0: QuadrayCoordinate, p1: QuadrayCoordinate,
                        p2: QuadrayCoordinate, p3: QuadrayCoordinate) -> Fraction:
    """
    Compute tetra-volume using determinant in IVM units (Fuller.4D).

    Calculates the volume of a tetrahedron defined by four Quadray coordinates
    using exact rational arithmetic. This follows the QuadMath approach.

    Args:
        p0, p1, p2, p3: The four vertices of the tetrahedron

    Returns:
        Fraction: The volume in IVM units (regular tetrahedron = 1). Lattice
        tetrahedra can have fractional volumes (e.g. the corner tetrahedron
        spanned by the origin and three basis directions has volume 1/4).
    """
    # Calculate vectors from p0 to the other points
    v1 = p1.sub(p0)
    v2 = p2.sub(p0)
    v3 = p3.sub(p0)

    # Project to 3D for determinant calculation
    # This uses the projection: (a-d, b-d, c-d) for each coordinate
    def project(q: QuadrayCoordinate) -> Tuple[int, int, int]:
        return (q.a - q.d, q.b - q.d, q.c - q.d)

    M = [
        list(project(v1)),
        list(project(v2)),
        list(project(v3))
    ]

    # Calculate 3x3 determinant
    det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
           M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))

    # Volume in IVM units: the unit tetrahedron has |det| = 4, so divide by 4
    # exactly. (The previous conditional normalization returned the raw
    # determinant for non-multiples of 4, mixing scales.)
    return Fraction(abs(det), 4)


def ace_tetra_volume_5x5(p0: QuadrayCoordinate, p1: QuadrayCoordinate,
                        p2: QuadrayCoordinate, p3: QuadrayCoordinate) -> Fraction:
    """
    Tom Ace 5x5 determinant method for tetra-volume in IVM units.

    Uses a 5x5 matrix with the four vertices plus a normalization row
    to compute exact integer volume.

    Args:
        p0, p1, p2, p3: The four vertices of the tetrahedron

    Returns:
        Fraction: The volume in IVM units (regular tetrahedron = 1); exact
        rational, matching integer_tetra_volume for every lattice input.
    """
    # Create 5x5 matrix for determinant calculation
    A = [
        [p0.a, p0.b, p0.c, p0.d, 1],
        [p1.a, p1.b, p1.c, p1.d, 1],
        [p2.a, p2.b, p2.c, p2.d, 1],
        [p3.a, p3.b, p3.c, p3.d, 1],
        [1, 1, 1, 1, 0],  # Normalization row
    ]

    # Calculate 5x5 determinant using Bareiss algorithm
    det = _bareiss_determinant_5x5(A)
    volume = Fraction(abs(det), 4)
    return volume


def _bareiss_determinant_5x5(matrix: List[List[int]]) -> int:
    """
    Calculate 5x5 determinant using Bareiss algorithm for exact integer arithmetic.

    Args:
        matrix: 5x5 integer matrix

    Returns:
        int: The determinant
    """
    # Bareiss algorithm implementation for 5x5 matrix
    A = [row[:] for row in matrix]  # Copy matrix
    n = 5

    for k in range(n-1):
        # Find pivot
        if A[k][k] == 0:
            # Look for non-zero pivot below
            for i in range(k+1, n):
                if A[i][k] != 0:
                    # Swap rows
                    A[k], A[i] = A[i], A[k]
                    break

        if A[k][k] == 0:
            return 0  # Singular matrix

        # Eliminate below
        for i in range(k+1, n):
            for j in range(k+1, n):
                A[i][j] = A[i][j] * A[k][k] - A[i][k] * A[k][j]
                if k > 0:
                    A[i][j] //= A[k-1][k-1] if A[k-1][k-1] != 0 else 1

    return A[n-1][n-1]


class SymergeticsPolyhedron:
    """
    Base class for polyhedra in the synergetic geometry system.

    Provides common functionality for all polyhedral forms, including
    volume calculations, vertex coordinates, and geometric properties.
    """

    def __init__(self, vertices: List[QuadrayCoordinate]):
        """
        Initialize a polyhedron with its vertices.

        Args:
            vertices: List of Quadray coordinates defining the polyhedron vertices
        """
        self.vertices = vertices
        self._volume_cache: Optional[Fraction] = None

    def volume(self) -> Fraction:
        """
        Calculate the volume of the polyhedron in IVM units.

        Cached; delegates the actual computation to _calculate_volume.

        Returns:
            Fraction: Volume in tetrahedral (IVM) units
        """
        if self._volume_cache is None:
            self._volume_cache = self._calculate_volume()
        return self._volume_cache

    def faces(self) -> List[List[QuadrayCoordinate]]:
        """
        Get the faces of the polyhedron as vertex lists.

        Must be provided by subclasses; used by the default volume method.
        """
        raise NotImplementedError

    def edges(self) -> List[Tuple[QuadrayCoordinate, QuadrayCoordinate]]:
        """
        Get the edges of the polyhedron, derived from its faces.

        Each pair of vertices consecutive on a face forms an edge; pairs are
        deduplicated. Subclasses may override with explicit edge lists.

        Returns:
            List[Tuple]: Unique vertex pairs connected by edges
        """
        seen = set()
        edges = []
        for face in self.faces():
            n = len(face)
            for i in range(n):
                a, b = face[i], face[(i + 1) % n]
                key = tuple(sorted((a.as_tuple(), b.as_tuple())))
                if key not in seen:
                    seen.add(key)
                    edges.append((a, b))
        return edges

    def _calculate_volume(self) -> Fraction:
        """
        Calculate volume by fan decomposition from vertices[0].

        For a convex polyhedron, the cones from any vertex over the faces
        that do not contain that vertex tile the polyhedron exactly, so the
        exact rational volume is the sum of their tetra volumes.

        Returns:
            Fraction: Total volume in IVM units
        """
        faces = self.faces()  # subclasses must provide the face list
        apex = self.vertices[0]
        total = Fraction(0)
        for face in faces:
            if apex in face:
                continue
            for i in range(1, len(face) - 1):
                total += integer_tetra_volume(apex, face[0], face[i], face[i + 1])
        return total

    def centroid(self) -> QuadrayCoordinate:
        """
        Calculate the centroid (average) of all vertices.

        Note:
            The centroid is snapped to the integer Quadray lattice with floor
            division. For solids whose true centroid falls between lattice
            points (e.g. a tetrahedron) the result is the floor of the exact
            rational centroid, not the centroid itself.

        Returns:
            QuadrayCoordinate: Floor-snapped centroid lattice point
        """
        if not self.vertices:
            return QuadrayCoordinate(0, 0, 0, 0)

        # Sum all coordinates
        total_a = sum(v.a for v in self.vertices)
        total_b = sum(v.b for v in self.vertices)
        total_c = sum(v.c for v in self.vertices)
        total_d = sum(v.d for v in self.vertices)

        n = len(self.vertices)
        return QuadrayCoordinate(
            total_a // n, total_b // n, total_c // n, total_d // n
        )

    def to_xyz_vertices(self, embedding=None) -> List[Tuple[float, float, float]]:
        """
        Convert all vertices to XYZ coordinates.

        Args:
            embedding: Optional 3x4 embedding matrix

        Returns:
            List[Tuple[float, float, float]]: XYZ coordinates of all vertices
        """
        if embedding is None:
            embedding = urner_embedding()
        return [v.to_xyz(embedding) for v in self.vertices]

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the polyhedron."""
        return len(self.vertices)

    @property
    def num_edges(self) -> int:
        """Number of edges (to be implemented by subclasses)."""
        raise NotImplementedError

    @property
    def num_faces(self) -> int:
        """Number of faces (to be implemented by subclasses)."""
        raise NotImplementedError


class Tetrahedron(SymergeticsPolyhedron):
    """
    Regular tetrahedron in synergetic geometry.

    The fundamental building block with volume = 1 in IVM units.
    """

    def __init__(self, vertices: Optional[List[QuadrayCoordinate]] = None):
        """
        Initialize a tetrahedron.

        Args:
            vertices: Optional list of 4 vertices (uses standard IVM tetrahedron if None)
        """
        if vertices is None:
            # Standard IVM tetrahedron vertices
            vertices = [
                QuadrayCoordinate(0, 0, 0, 0),    # Origin
                QuadrayCoordinate(2, 1, 0, 1),    # Vertex 1
                QuadrayCoordinate(2, 1, 1, 0),    # Vertex 2
                QuadrayCoordinate(2, 0, 1, 1),    # Vertex 3
            ]
        elif len(vertices) != 4:
            raise ValueError("Tetrahedron must have exactly 4 vertices")

        super().__init__(vertices)

    def _calculate_volume(self) -> Fraction:
        """Calculate volume of tetrahedron (exact rational IVM units)."""
        return integer_tetra_volume(*self.vertices)

    @property
    def num_edges(self) -> int:
        """Number of edges in a tetrahedron."""
        return 6

    @property
    def num_faces(self) -> int:
        """Number of faces in a tetrahedron."""
        return 4

    def faces(self) -> List[List[QuadrayCoordinate]]:
        """
        Get all faces of the tetrahedron.

        Returns:
            List[List]: Triangular faces defined by vertex triples
        """
        return [
            [self.vertices[0], self.vertices[1], self.vertices[2]],  # Base
            [self.vertices[0], self.vertices[1], self.vertices[3]],  # Side 1
            [self.vertices[0], self.vertices[2], self.vertices[3]],  # Side 2
            [self.vertices[1], self.vertices[2], self.vertices[3]],  # Side 3
        ]


class Octahedron(SymergeticsPolyhedron):
    """
    Regular octahedron in synergetic geometry.

    Composed of 4 tetrahedra with volume = 4 in IVM units.
    """

    def __init__(self, vertices: Optional[List[QuadrayCoordinate]] = None):
        """
        Initialize an octahedron.

        Args:
            vertices: Optional list of 6 vertices (uses standard IVM octahedron if None)
        """
        if vertices is None:
            # Standard IVM octahedron vertices, ordered [+X, -X, +Y, -Y, +Z, -Z]
            vertices = [v.copy() for v in OCTAHEDRON_VERTICES]
        elif len(vertices) != 6:
            raise ValueError("Octahedron must have exactly 6 vertices")

        super().__init__(vertices)


    @property
    def num_edges(self) -> int:
        """Number of edges in an octahedron."""
        return 12

    @property
    def num_faces(self) -> int:
        """Number of faces in an octahedron."""
        return 8

    def faces(self) -> List[List[QuadrayCoordinate]]:
        """
        Get all triangular faces of the octahedron.

        Returns:
            List[List]: Triangular faces defined by vertex triples
        """
        # Octahedron vertices: [0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z]
        # Faces are triangles connecting vertices
        return [
            [self.vertices[0], self.vertices[2], self.vertices[4]],  # +X, +Y, +Z
            [self.vertices[0], self.vertices[2], self.vertices[5]],  # +X, +Y, -Z
            [self.vertices[0], self.vertices[3], self.vertices[4]],  # +X, -Y, +Z
            [self.vertices[0], self.vertices[3], self.vertices[5]],  # +X, -Y, -Z
            [self.vertices[1], self.vertices[2], self.vertices[4]],  # -X, +Y, +Z
            [self.vertices[1], self.vertices[2], self.vertices[5]],  # -X, +Y, -Z
            [self.vertices[1], self.vertices[3], self.vertices[4]],  # -X, -Y, +Z
            [self.vertices[1], self.vertices[3], self.vertices[5]],  # -X, -Y, -Z
        ]


class Cube(SymergeticsPolyhedron):
    """
    Cube (hexahedron) in synergetic geometry.

    Composed of 3 tetrahedra with volume = 3 in IVM units.
    """

    def __init__(self, vertices: Optional[List[QuadrayCoordinate]] = None):
        """
        Initialize a cube.

        Args:
            vertices: Optional list of 8 vertices (uses standard IVM cube if None)
        """
        if vertices is None:
            # Standard IVM cube vertices: face diagonal equals the tetra edge
            vertices = [v.copy() for v in CUBE_VERTICES]
        elif len(vertices) != 8:
            raise ValueError("Cube must have exactly 8 vertices")

        super().__init__(vertices)

    def faces(self) -> List[List[QuadrayCoordinate]]:
        """
        Get the 6 square faces of the cube (vertex order from CUBE_VERTICES).

        Returns:
            List[List]: Square faces defined by 4-vertex cycles
        """
        v = self.vertices
        return [
            [v[0], v[5], v[3], v[6]],  # +col1 plane
            [v[4], v[1], v[7], v[2]],  # -col1 plane
            [v[0], v[6], v[1], v[7]],  # +col2 plane
            [v[5], v[3], v[4], v[2]],  # -col2 plane
            [v[0], v[7], v[2], v[5]],  # +col3 plane
            [v[6], v[3], v[4], v[1]],  # -col3 plane
        ]


    @property
    def num_edges(self) -> int:
        """Number of edges in a cube."""
        return 12

    @property
    def num_faces(self) -> int:
        """Number of faces in a cube."""
        return 6


class Cuboctahedron(SymergeticsPolyhedron):
    """
    Cuboctahedron (vector equilibrium) in synergetic geometry.

    The 12-around-one IVM neighbors form a cuboctahedron with volume = 20.
    """

    def __init__(self, vertices: Optional[List[QuadrayCoordinate]] = None):
        """
        Initialize a cuboctahedron.

        Args:
            vertices: Optional list of 12 vertices (uses IVM neighbors if None)
        """
        if vertices is None:
            # The 12 IVM neighbors (all permutations of {2,1,1,0}), in the
            # deterministic order defined by IVM_NEIGHBORS.
            vertices = [n.copy() for n in IVM_NEIGHBORS]
        elif len(vertices) != 12:
            raise ValueError("Cuboctahedron must have exactly 12 vertices")

        super().__init__(vertices)

    def faces(self) -> List[List[QuadrayCoordinate]]:
        """
        Get the 14 faces: 8 triangles + 6 squares.

        Indices refer to the IVM_NEIGHBORS order of the default vertices.
        """
        v = self.vertices
        triangles = [
            (1, 2, 0), (1, 6, 4), (8, 2, 7), (8, 6, 11),
            (3, 5, 0), (3, 9, 4), (10, 5, 7), (10, 9, 11),
        ]
        squares = [
            (1, 2, 8, 6), (3, 5, 10, 9), (0, 1, 4, 3),
            (7, 8, 11, 10), (0, 2, 7, 5), (4, 6, 11, 9),
        ]
        return [[v[i] for i in face] for face in triangles + squares]

    @property
    def num_edges(self) -> int:
        """Number of edges in a cuboctahedron."""
        return 24

    @property
    def num_faces(self) -> int:
        """Number of faces in a cuboctahedron (8 triangles + 6 squares)."""
        return 14


# Convenience functions for creating standard polyhedra
def create_unit_tetrahedron() -> Tetrahedron:
    """Create a unit tetrahedron (volume = 1)."""
    return Tetrahedron()


def create_octahedron() -> Octahedron:
    """Create a regular octahedron (volume = 4)."""
    return Octahedron()


def create_cube() -> Cube:
    """Create a cube (volume = 3)."""
    return Cube()


def create_cuboctahedron() -> Cuboctahedron:
    """Create a cuboctahedron/vector equilibrium (volume = 20)."""
    return Cuboctahedron()


# Volume ratio constants (matches SymergeticsConstants)
VOLUME_RATIOS = {
    'tetrahedron': 1,
    'octahedron': 4,
    'cube': 3,
    'cuboctahedron': 20,
    'rhombic_dodecahedron': 6,
    'rhombic_triacontahedron': 120,
}
