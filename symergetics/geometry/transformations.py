"""
Geometric Transformations for Synergetic Coordinate System

This module provides functions for transforming geometric objects in the
Quadray coordinate system, including translations, rotations, scaling,
and coordinate system conversions.

Key Features:
- Coordinate system conversions between Quadray and Cartesian
- Geometric transformations (translate, rotate, scale)
- Projection and embedding operations
- Integration with numpy for matrix operations

Author: Symergetics Team
"""

import math
from fractions import Fraction
from numbers import Integral

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from ..core.coordinates import (
    QuadrayCoordinate, _as_fraction, _float_metric, _validated_embedding,
    _xyz_exact, urner_embedding,
)
from .polyhedra import SymergeticsPolyhedron


def translate(coord: QuadrayCoordinate, offset: QuadrayCoordinate) -> QuadrayCoordinate:
    """
    Translate a Quadray coordinate by an offset vector.

    Args:
        coord: The coordinate to translate
        offset: The translation vector

    Returns:
        QuadrayCoordinate: The translated coordinate
    """
    return coord.add(offset)


def translate_polyhedron(polyhedron: SymergeticsPolyhedron,
                        offset: QuadrayCoordinate) -> SymergeticsPolyhedron:
    """
    Translate all vertices of a polyhedron by an offset vector.

    Args:
        polyhedron: The polyhedron to translate
        offset: The translation vector

    Returns:
        SymergeticsPolyhedron: Same-type copy via ``with_vertices``; non-vertex
            state is shallow-shared unless the subclass overrides that protocol.
    """
    translated_vertices = [translate(vertex, offset) for vertex in polyhedron.vertices]
    return polyhedron.with_vertices(translated_vertices)


def scale(coord: QuadrayCoordinate, factor: Union[int, float]) -> QuadrayCoordinate:
    """Scale a spatial point, independently of its all-ones gauge.

    Integral factors (including integral floats) preserve integer coordinates
    exactly. Other finite factors scale in XYZ space and snap once to the
    nearest integer quadray using :meth:`QuadrayCoordinate.from_xyz`, including
    its deterministic tie rule. Thus fractional scaling need not be invertible.
    """
    if isinstance(factor, Integral):
        integer_factor = int(factor)
    else:
        factor = float(factor)
        if not math.isfinite(factor):
            raise ValueError("Scaling factor must be finite")
        integer_factor = int(factor) if factor.is_integer() else None

    values = coord.as_tuple()
    gauge = min(values)
    canonical = tuple(value - gauge for value in values)
    if integer_factor is not None:
        return QuadrayCoordinate(*(value * integer_factor for value in canonical))

    xyz = QuadrayCoordinate(*canonical).to_xyz()
    return QuadrayCoordinate.from_xyz(*(value * factor for value in xyz))


def scale_polyhedron(polyhedron: SymergeticsPolyhedron,
                    factor: Union[int, float]) -> SymergeticsPolyhedron:
    """
    Scale all vertices of a polyhedron by a scalar factor.

    Args:
        polyhedron: The polyhedron to scale
        factor: The scaling factor

    Returns:
        SymergeticsPolyhedron: A new polyhedron with scaled vertices
    """
    scaled_vertices = [scale(vertex, factor) for vertex in polyhedron.vertices]
    return polyhedron.with_vertices(scaled_vertices)


def reflect(coord: QuadrayCoordinate, axis: str = 'origin') -> QuadrayCoordinate:
    """Reflect a spatial point through the origin or a named plane.

    ``origin`` is exact integer inversion. Each of ``a``, ``b``, ``c``, ``d``
    names the plane through the origin perpendicular to that Urner basis
    vector, not a component sign change. Plane reflections happen in Euclidean
    XYZ space with exact rational plane arithmetic, then snap once to the nearest
    integer quadray with ``from_xyz``'s deterministic ties. A reflected point off
    the lattice is quantized; distances and involution are exact only without snapping.
    """
    if axis == 'origin':
        return scale(coord, -1)
    if axis not in ('a', 'b', 'c', 'd'):
        raise ValueError(f"Unknown reflection axis: {axis}")

    normal = urner_embedding()[:, ('a', 'b', 'c', 'd').index(axis)]
    return _plane_transform(coord, normal, (0, 0, 0), 2)


def reflect_polyhedron(polyhedron: SymergeticsPolyhedron,
                      axis: str = 'origin') -> SymergeticsPolyhedron:
    """
    Reflect all vertices through the origin or a named basis-normal plane.

    Args:
        polyhedron: The polyhedron to reflect
        axis: The axis of reflection

    Returns:
        SymergeticsPolyhedron: A new polyhedron with reflected vertices
    """
    reflected_vertices = [reflect(vertex, axis) for vertex in polyhedron.vertices]
    return polyhedron.with_vertices(reflected_vertices)


def coordinate_transform(coord: QuadrayCoordinate,
                        matrix: np.ndarray) -> QuadrayCoordinate:
    """Apply a linear map on quadrays modulo the all-ones gauge.

    ``matrix @ ones(4)`` must be proportional to ``ones(4)`` so equivalent
    quadrays remain equivalent. Integral matrices are applied with exact
    integer arithmetic. Other matrices act continuously, followed by one
    Euclidean nearest-lattice projection via ``from_xyz``; component truncation
    or independent rounding is not a geometric projection.
    """
    matrix = np.asarray(matrix)
    if matrix.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4")

    # Preserve integer maps without passing either coefficients or coordinates
    # through float, including matrices supplied as integral-valued floats.
    entries = list(matrix.flat)
    integral = all(
        isinstance(value, Integral)
        or isinstance(value, (float, np.floating))
        and math.isfinite(value) and value.is_integer()
        for value in entries
    )
    values = coord.as_tuple()
    gauge = min(values)
    canonical = tuple(value - gauge for value in values)
    if integral:
        rows = [tuple(int(value) for value in row) for row in matrix]
        row_sums = [sum(row) for row in rows]
        if any(total != row_sums[0] for total in row_sums[1:]):
            raise ValueError("Transformation matrix must preserve the all-ones gauge")
        return QuadrayCoordinate(*(
            sum(coefficient * value for coefficient, value in zip(row, canonical))
            for row in rows
        ))

    matrix = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Transformation matrix must be finite")
    row_sums = matrix @ np.ones(4)
    tolerance = 16 * np.finfo(float).eps * np.max(np.sum(np.abs(matrix), axis=1))
    if np.max(np.abs(row_sums - row_sums[0])) > tolerance:
        raise ValueError("Transformation matrix must preserve the all-ones gauge")

    transformed = matrix @ np.asarray(canonical, dtype=float)
    transformed -= transformed.min()
    xyz = urner_embedding() @ transformed
    return QuadrayCoordinate.from_xyz(*xyz)


def rotate_around_axis(coord: QuadrayCoordinate, axis: str, angle_degrees: float) -> QuadrayCoordinate:
    """
    Rotate a coordinate around a specified axis by a given angle.

    This function rotates in XYZ space, then snaps to the nearest integer
    quadray using ``from_xyz``'s deterministic ties. Off-lattice rotations
    are quantized and need not preserve lengths or be invertible.

    Args:
        coord: The coordinate to rotate
        axis: Rotation axis ('x', 'y', 'z')
        angle_degrees: Rotation angle in degrees

    Returns:
        QuadrayCoordinate: The rotated coordinate
    """
    # Convert to XYZ
    x, y, z = coord.to_xyz()

    # Convert angle to radians
    angle_rad = np.radians(angle_degrees)

    # Perform rotation in XYZ space
    if axis == 'x':
        # Rotate around X axis
        new_y = y * np.cos(angle_rad) - z * np.sin(angle_rad)
        new_z = y * np.sin(angle_rad) + z * np.cos(angle_rad)
        x, y, z = x, new_y, new_z
    elif axis == 'y':
        # Rotate around Y axis
        new_x = x * np.cos(angle_rad) + z * np.sin(angle_rad)
        new_z = -x * np.sin(angle_rad) + z * np.cos(angle_rad)
        x, y, z = new_x, y, new_z
    elif axis == 'z':
        # Rotate around Z axis
        new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        x, y, z = new_x, new_y, z
    else:
        raise ValueError(f"Unknown rotation axis: {axis}")

    # Convert back to Quadray
    return QuadrayCoordinate.from_xyz(x, y, z)


def _plane_fraction(value):
    """Preserve each NumPy floating scalar's represented binary value."""
    if isinstance(value, np.floating):
        return Fraction(*value.as_integer_ratio())
    return _as_fraction(value)


def _plane_transform(coord, normal, plane_point, factor):
    """Apply an exact Euclidean plane map before one nearest-lattice snap."""
    try:
        normal = tuple(_plane_fraction(value) for value in normal)
        if len(normal) != 3 or not any(normal):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Plane normal must be a finite nonzero three-vector") from exc
    try:
        plane_point = tuple(_plane_fraction(value) for value in plane_point)
        if len(plane_point) != 3:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Plane point must be a finite three-vector") from exc
    point = _xyz_exact(urner_embedding(), coord)
    distance = sum((value - origin) * weight
                   for value, origin, weight in zip(point, plane_point, normal))
    amount = factor * distance / sum(weight * weight for weight in normal)
    return QuadrayCoordinate.from_xyz(*(
        value - amount * weight for value, weight in zip(point, normal)))


def project_to_plane(coord: QuadrayCoordinate, plane_normal: Tuple[float, float, float],
                    plane_point: Optional[Tuple[float, float, float]] = None) -> QuadrayCoordinate:
    """
    Project a coordinate onto a plane defined by a normal vector.

    Args:
        coord: The coordinate to project
        plane_normal: Normal vector of the plane (in XYZ space)
        plane_point: A point on the plane (defaults to origin)

    Returns:
        QuadrayCoordinate: Nearest integer quadray to the exact rational Euclidean
            projection, with ``from_xyz``'s deterministic ties. Quantization may
            move the result off the specified plane. Finite Python and NumPy
            floating plane inputs denote their exact represented binary values.
    """
    if plane_point is None:
        plane_point = (0, 0, 0)
    return _plane_transform(coord, plane_normal, plane_point, 1)


def apply_transform_function(coords: List[QuadrayCoordinate],
                           transform_func: Callable[[QuadrayCoordinate], QuadrayCoordinate]) -> List[QuadrayCoordinate]:
    """
    Apply a transformation function to a list of coordinates.

    Args:
        coords: List of coordinates to transform
        transform_func: Function that takes a QuadrayCoordinate and returns one

    Returns:
        List[QuadrayCoordinate]: The transformed coordinates
    """
    return [transform_func(coord) for coord in coords]


def create_rotation_matrix(axis: str, angle_degrees: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for XYZ space rotations.

    Args:
        axis: Rotation axis ('x', 'y', 'z')
        angle_degrees: Rotation angle in degrees

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    elif axis == 'y':
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    elif axis == 'z':
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unknown rotation axis: {axis}")


def xyz_to_quadray_matrix(embedding: Optional[np.ndarray] = None) -> np.ndarray:
    """Return the continuous zero-sum 4x3 inverse of the effective embedding.

    The supplied embedding must be a finite rank-three 3x4 matrix annihilating
    the all-ones gauge within the accepted summation-roundoff tolerance. Its
    first three columns define the d=0 map; the fourth is their negative sum.
    The result is that effective map's minimum-norm Moore-Penrose inverse.
    Rational inversion preserves anisotropic rank and units, rounding only the
    final entries to float. It neither normalizes nor snaps to integer quadrays.
    """
    embedding = _validated_embedding(embedding)
    # Invert the effective d=0 map exactly, without globally scaling away a
    # small independent axis. Cofactor columns give its 3x3 inverse.
    a, b, c = [tuple(_as_fraction(value) for value in row[:3]) for row in embedding]
    def cross(left, right):
        x, y, z = left
        u, v, w = right
        return (y * w - z * v, z * u - x * w, x * v - y * u)
    columns = (cross(b, c), cross(c, a), cross(a, b))
    determinant = sum(value * cofactor for value, cofactor in zip(a, columns[0]))
    inverse = [tuple(value / determinant for value in row) for row in zip(*columns)]
    gauge = tuple(-sum(row[j] for row in inverse) / 4 for j in range(3))
    return np.array([
        [_float_metric(value + offset) for value, offset in zip(row, gauge)]
        for row in inverse
    ] + [[_float_metric(value) for value in gauge]])


def compose_transforms(*transforms: Callable[[QuadrayCoordinate], QuadrayCoordinate]) -> Callable[[QuadrayCoordinate], QuadrayCoordinate]:
    """
    Compose multiple transformation functions into a single function.

    Args:
        *transforms: Transformation functions to compose

    Returns:
        Callable: Composed transformation function
    """
    def composed_transform(coord: QuadrayCoordinate) -> QuadrayCoordinate:
        result = coord
        for transform in transforms:
            result = transform(result)
        return result

    return composed_transform


# Convenience functions for common transformations
def translate_by_vector(dx: int, dy: int, dz: int) -> Callable[[QuadrayCoordinate], QuadrayCoordinate]:
    """
    Create a translation function for a given vector.

    Args:
        dx, dy, dz: Translation components in XYZ space

    Returns:
        Callable: Translation function
    """
    offset = QuadrayCoordinate.from_xyz(dx, dy, dz)

    def translate_func(coord: QuadrayCoordinate) -> QuadrayCoordinate:
        return translate(coord, offset)

    return translate_func


def scale_by_factor(factor: float) -> Callable[[QuadrayCoordinate], QuadrayCoordinate]:
    """
    Create a scaling function for a given factor.

    Args:
        factor: Scaling factor

    Returns:
        Callable: Scaling function
    """
    def scale_func(coord: QuadrayCoordinate) -> QuadrayCoordinate:
        return scale(coord, factor)

    return scale_func


def rotate_around_xyz_axis(axis: str, angle_degrees: float) -> Callable[[QuadrayCoordinate], QuadrayCoordinate]:
    """
    Create a rotation function around a specified XYZ axis.

    Args:
        axis: Rotation axis ('x', 'y', 'z')
        angle_degrees: Rotation angle in degrees

    Returns:
        Callable: Rotation function
    """
    def rotate_func(coord: QuadrayCoordinate) -> QuadrayCoordinate:
        return rotate_around_axis(coord, axis, angle_degrees)

    return rotate_func
