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

from copy import copy
from fractions import Fraction
import math
from typing import List, Optional, Tuple

from ..core.coordinates import (
    CUBE_VERTICES,
    IVM_NEIGHBORS,
    OCTAHEDRON_VERTICES,
    QuadrayCoordinate,
    _validated_embedding,
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

def _project(vertex):
    """Exact coordinates on the rank-three quadray quotient."""
    return (vertex.a - vertex.d, vertex.b - vertex.d, vertex.c - vertex.d)


def _difference(left, right):
    return tuple(a - b for a, b in zip(left, right))


def _cross(left, right):
    a, b, c = left
    x, y, z = right
    return (b * z - c * y, c * x - a * z, a * y - b * x)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _shell_contains(point, triangles, points):
    """Exact even-odd ray test; the point must not lie on this shell.

    Directions (1, n, n**2) avoid any finite collection of bad planes after
    finitely many retries. Retry coplanar rays and edge/vertex hits rather than
    counting an ambiguous crossing. Degenerate triangles contribute no area.
    Nonintersection of the supplied shells is a precondition, not checked here.
    """
    direction_index = 0
    while True:
        direction = (1, direction_index, direction_index**2)
        crossings = 0
        for i, j, k in triangles:
            a = points[i]
            u = _difference(points[j], a)
            v = _difference(points[k], a)
            normal = _cross(u, v)
            if not any(normal):
                continue
            offset = _difference(point, a)
            perpendicular = _cross(direction, v)
            determinant = _dot(u, perpendicular)
            if determinant == 0:
                if _dot(normal, offset) == 0:
                    break
                continue
            first = _dot(offset, perpendicular)
            other = _cross(offset, u)
            second = _dot(direction, other)
            distance = _dot(v, other)
            if determinant < 0:
                determinant, first, second, distance = (
                    -determinant, -first, -second, -distance)
            if distance <= 0 or first < 0 or second < 0 or first + second > determinant:
                continue
            if first == 0 or second == 0 or first + second == determinant:
                break
            crossings += 1
        else:
            return crossings % 2 == 1
        direction_index += 1


def _float_area(value):
    try:
        return float(value)
    except OverflowError:
        return math.copysign(math.inf, 1 if value > 0 else -1)



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

    def with_vertices(self, vertices: List[QuadrayCoordinate]) -> 'SymergeticsPolyhedron':
        """Return a same-type shallow copy with an independently owned vertex list.

        Constructors are not rerun: indexed topology and other non-vertex state
        are retained by reference. The supplied coordinate objects are retained,
        not cloned; transforms supply newly created coordinates. Subclasses with
        geometry-dependent caches or different state ownership must override this
        protocol to update that state.
        """
        result = copy(self)
        result.vertices = list(vertices)
        return result

    def volume(self) -> Fraction:
        """
        Calculate the volume of the polyhedron in IVM units.

        Recomputed from the current publicly mutable vertices on every call.
        Uses the declared fan-triangulated closed, orientable surface. Shells
        must not intersect themselves or each other. Disjoint solids add;
        nested shells alternate solid/cavity by even-odd enclosure, independent
        of their input winding. Concavity and nonplanar declared fans are
        supported; this is not a convex-hull calculation.

        Returns:
            Fraction: Volume in tetrahedral (IVM) units
        """
        return self._calculate_volume()

    def face_indices(self) -> Tuple[Tuple[int, ...], ...]:
        """Declared face cycles, indexed by position, independent of geometry."""
        raise NotImplementedError

    def faces(self) -> List[List[QuadrayCoordinate]]:
        """Map declared cycles to the current, publicly mutable vertices."""
        return [[self.vertices[i] for i in face] for face in self.face_indices()]

    def triangulated_faces(self) -> Tuple[Tuple[int, int, int], ...]:
        """Return the declared face fans as vertex-index triples.

        Warped or concave polygon faces mean precisely this triangulated
        surface, not a fitted plane or a convex hull. Solid measurements
        require closed orientable shells with no self- or mutual intersections;
        nested shells use even-odd solid/cavity semantics.
        """
        return tuple((face[0], face[i], face[i + 1])
                     for face in self.face_indices()
                     for i in range(1, len(face) - 1))

    def edges(self) -> List[Tuple[QuadrayCoordinate, QuadrayCoordinate]]:
        """
        Get the edges of the polyhedron, derived from its faces.

        Consecutive vertex indices form edges, deduplicated by index rather
        than coordinate value, so collapsed geometry retains its topology.

        Returns:
            List[Tuple]: Unique vertex pairs connected by edges
        """
        seen = set()
        edges = []
        for face in self.face_indices():
            for i, a in enumerate(face):
                b = face[(i + 1) % len(face)]
                key = tuple(sorted((a, b)))
                if key not in seen:
                    seen.add(key)
                    edges.append((self.vertices[a], self.vertices[b]))
        return edges

    def _oriented_shells(self):
        """Orient each connected fan shell by adjacency, retaining diagonals."""
        triangles = self.triangulated_faces()
        incidence = {}
        for index, triangle in enumerate(triangles):
            for a, b in zip(triangle, triangle[1:] + triangle[:1]):
                incidence.setdefault(tuple(sorted((a, b))), []).append(
                    (index, 1 if a < b else -1))
        closed = bool(triangles) and all(len(edge) == 2 for edge in incidence.values())
        if not closed:
            return (triangles,), False
        adjacent = [[] for _ in triangles]
        for (left, ld), (right, rd) in incidence.values():
            relation = -ld * rd
            adjacent[left].append((right, relation))
            adjacent[right].append((left, relation))
        signs = {}
        components = []
        for start in range(len(triangles)):
            if start in signs:
                continue
            signs[start] = 1
            pending = [start]
            component = []
            while pending:
                current = pending.pop()
                component.append(current)
                for neighbor, relation in adjacent[current]:
                    expected = signs[current] * relation
                    if neighbor in signs:
                        if signs[neighbor] != expected:
                            return (triangles,), False
                    else:
                        signs[neighbor] = expected
                        pending.append(neighbor)
            components.append(component)
        return tuple(tuple(triangles[i] if signs[i] == 1 else
                           (triangles[i][0], triangles[i][2], triangles[i][1])
                           for i in component)
                     for component in components), True

    def _solid_moments(self):
        """Positive shell moments combined with exact even-odd enclosure signs."""
        shells, closed = self._oriented_shells()
        if not closed:
            raise ValueError("Solid measurements require a closed orientable mesh")
        points = [_project(vertex) for vertex in self.vertices]
        anchor = points[0]
        relative = [_difference(point, anchor) for point in points]
        shell_moments = []
        for triangles in shells:
            determinant = 0
            moments = [0, 0, 0]
            for i, j, k in triangles:
                a, b, c = relative[i], relative[j], relative[k]
                weight = _dot(a, _cross(b, c))
                determinant += weight
                for axis in range(3):
                    moments[axis] += weight * (a[axis] + b[axis] + c[axis])
            if determinant < 0:
                determinant = -determinant
                moments = [-moment for moment in moments]
            shell_moments.append((determinant, moments))
        if len(shells) == 1:
            determinant, moments = shell_moments[0]
            return determinant, moments, anchor
        determinant = 0
        moments = [0, 0, 0]
        for index, (weight, shell_moment) in enumerate(shell_moments):
            if weight == 0:
                continue
            representative = points[shells[index][0][0]]
            depth = sum(_shell_contains(representative, other, points)
                        for other_index, other in enumerate(shells)
                        if other_index != index and shell_moments[other_index][0])
            sign = -1 if depth % 2 else 1
            determinant += sign * weight
            for axis in range(3):
                moments[axis] += sign * shell_moment[axis]
        return determinant, moments, anchor

    def _calculate_volume(self) -> Fraction:
        """Live enclosed volume in IVM units, from signed oriented cones.

        Requires closed orientable shells without self- or mutual intersections.
        Disjoint solids add; nesting alternates cavities and solids regardless
        of winding. Concavity and warped faces' declared fans are supported.
        """
        determinant, _, _ = self._solid_moments()
        return Fraction(determinant, 4)

    def volume_centroid_exact(self) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
        """Uniform-density solid centroid, not the generally different vertex mean.

        Uses the same even-odd enclosed mass as volume(), not input shell
        winding. Return the canonical minimum-zero rational quadray tuple.
        A zero-volume solid has no defined mass centroid and raises ValueError.
        """
        determinant, moments, anchor = self._solid_moments()
        if determinant == 0:
            raise ValueError("Volume centroid is undefined for zero volume")
        values = tuple(Fraction(origin) + Fraction(moment, 4 * determinant)
                       for origin, moment in zip(anchor, moments)) + (Fraction(0),)
        minimum = min(values)
        return tuple(value - minimum for value in values)

    def surface_area(self, embedding=None) -> float:
        """Live fan-triangle area in embedding-squared units (default scale 1).

        Exact displacements and cross products precede floating conversion.
        Cross products share a scale before summing triangle norms; only the
        total is rescaled, retaining representable subnormal aggregate areas.
        Area is generally irrational; no planarity or convexity is assumed.
        """
        matrix = _validated_embedding(embedding)
        rows = [tuple(Fraction(value) for value in row[:3]) for row in matrix]
        points = [_project(vertex) for vertex in self.vertices]
        crosses = []
        scale = Fraction(0)
        for i, j, k in self.triangulated_faces():
            u = _difference(points[j], points[i])
            v = _difference(points[k], points[i])
            cross = _cross(tuple(_dot(row, u) for row in rows),
                           tuple(_dot(row, v) for row in rows))
            crosses.append(cross)
            scale = max(scale, *(abs(value) for value in cross))
        if scale == 0:
            return 0.0
        total = math.fsum(math.hypot(*(float(value / scale) for value in cross))
                          for cross in crosses)
        return _float_area(Fraction(total) * scale / 2)

    def analyze_geometry(self):
        """Live exact topology/shape diagnostics and default scale-1 metrics.

        ``closed`` describes indexed topology, even when physical vertices
        coincide. ``nonplanar_faces`` contains declared face indices.
        Convexity requires rank three and supporting fan triangles, independently
        of declared-face planarity. This is not an intersection detector:
        solid metrics assume shells have no self- or mutual intersections and
        use even-odd enclosure. Cartesian volume is 8/3 times IVM at scale 1.
        """
        points = [_project(vertex) for vertex in self.vertices]
        vectors = [_difference(point, points[0]) for point in points[1:]] if points else []
        first = next((v for v in vectors if any(v)), None)
        normal = next((_cross(first, v) for v in vectors if any(_cross(first, v))),
                      None) if first else None
        rank = (3 if any(_dot(normal, v) for v in vectors) else 2) if normal else int(first is not None)
        nonplanar = []
        for index, face in enumerate(self.face_indices()):
            relative = [_difference(points[i], points[face[0]]) for i in face[1:]]
            direction = next((v for v in relative if any(v)), None)
            perpendicular = next((_cross(direction, v) for v in relative
                                  if any(_cross(direction, v))), None) if direction else None
            if perpendicular and any(_dot(perpendicular, v) for v in relative):
                nonplanar.append(index)
        shells, closed = self._oriented_shells()
        convex = closed and rank == 3 and len(shells) == 1
        if convex:
            for i, j, k in shells[0]:
                normal = _cross(_difference(points[j], points[i]),
                                _difference(points[k], points[i]))
                sides = [_dot(normal, _difference(point, points[i])) for point in points]
                if any(side > 0 for side in sides) and any(side < 0 for side in sides):
                    convex = False
                    break
        volume = self.volume() if closed else None
        return {
            "affine_rank": rank,
            "physical_vertex_count": len(set(points)),
            "closed": closed,
            "nonplanar_faces": tuple(nonplanar),
            "is_convex": convex,
            "surface_area": self.surface_area(),
            "volume_ivm": volume,
            "volume_xyz": volume * Fraction(8, 3) if volume is not None else None,
            "vertex_centroid_exact": self.centroid_exact() if points else None,
            "volume_centroid_exact": self.volume_centroid_exact() if volume else None,
        }

    def centroid(self) -> QuadrayCoordinate:
        """
        Calculate the centroid (average) of all vertices.

        Note:
            Each vertex is independently put in the minimum-zero gauge, then
            component means are floor-snapped and normalized. This intentional
            lattice approximation is neither an exact vertex mean nor a solid
            mass centroid; use the explicitly named exact methods for those.

        Returns:
            QuadrayCoordinate: Floor-snapped centroid lattice point
        """
        if not self.vertices:
            return QuadrayCoordinate(0, 0, 0, 0)

        # Remove each input's independent gauge before intentional floor snapping.
        canonical = [tuple(value - min(vertex.as_tuple()) for value in vertex.as_tuple())
                     for vertex in self.vertices]
        total_a, total_b, total_c, total_d = map(sum, zip(*canonical))

        n = len(self.vertices)
        return QuadrayCoordinate(
            total_a // n, total_b // n, total_c // n, total_d // n
        )

    def centroid_exact(self) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
        """Return the exact vertex mean in the minimum-zero gauge.

        Components are converted to rational numbers before arithmetic. The
        result is a tuple, not an integer-snapping QuadrayCoordinate; unlike
        centroid(), this method never floors the mean. Independent input gauge
        shifts leave the result unchanged. Empty vertex lists raise ValueError.
        """
        if not self.vertices:
            raise ValueError("At least one vertex is required")
        totals = [Fraction(0) for _ in range(4)]
        for vertex in self.vertices:
            for i, value in enumerate(vertex.as_tuple()):
                totals[i] += Fraction(value)
        minimum = min(totals)
        count = len(self.vertices)
        a, b, c, d = ((total - minimum) / count for total in totals)
        return a, b, c, d

    def to_xyz_vertices(self, embedding=None, *, relative_to=None) -> List[Tuple[float, float, float]]:
        """
        Convert all vertices to XYZ coordinates.

        Args:
            embedding: Optional 3x4 embedding matrix
            relative_to: Optional QuadrayCoordinate anchor. Subtraction occurs
                exactly before embedding, preserving small translated shapes.

        Returns:
            List[Tuple[float, float, float]]: XYZ coordinates of all vertices
        """
        vertices = self.vertices if relative_to is None else [
            vertex.sub(relative_to) for vertex in self.vertices]
        return [vertex.to_xyz(embedding) for vertex in vertices]

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


    @property
    def num_edges(self) -> int:
        """Number of edges in a tetrahedron."""
        return 6

    @property
    def num_faces(self) -> int:
        """Number of faces in a tetrahedron."""
        return 4

    def face_indices(self) -> Tuple[Tuple[int, ...], ...]:
        return ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


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

    def face_indices(self) -> Tuple[Tuple[int, ...], ...]:
        # Vertices ordered [+X, -X, +Y, -Y, +Z, -Z].
        return ((0, 2, 4), (0, 2, 5), (0, 3, 4), (0, 3, 5),
                (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5))


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

    def face_indices(self) -> Tuple[Tuple[int, ...], ...]:
        return ((0, 5, 3, 6), (4, 1, 7, 2), (0, 6, 1, 7),
                (5, 3, 4, 2), (0, 7, 2, 5), (6, 3, 4, 1))


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

    def face_indices(self) -> Tuple[Tuple[int, ...], ...]:
        # Indices refer to the deterministic IVM_NEIGHBORS order.
        return ((1, 2, 0), (1, 6, 4), (8, 2, 7), (8, 6, 11),
                (3, 5, 0), (3, 9, 4), (10, 5, 7), (10, 9, 11),
                (1, 2, 8, 6), (3, 5, 10, 9), (0, 1, 4, 3),
                (7, 8, 11, 10), (0, 2, 7, 5), (4, 6, 11, 9))

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
