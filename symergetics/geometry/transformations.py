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

from typing import List, Tuple, Union, Optional, Callable
import numpy as np
from ..core.coordinates import QuadrayCoordinate, urner_embedding
from ..core.numbers import SymergeticsNumber
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
        SymergeticsPolyhedron: A new polyhedron with translated vertices
    """
    translated_vertices = [translate(vertex, offset) for vertex in polyhedron.vertices]
    # Create a new instance of the same type
    return type(polyhedron)(translated_vertices)


def scale(coord: QuadrayCoordinate, factor: Union[int, float]) -> QuadrayCoordinate:
    """
    Scale a Quadray coordinate by a scalar factor.

    Note: Scaling can break the integer properties of the IVM lattice.
    The result is converted back to integers.

    Args:
        coord: The coordinate to scale
        factor: The scaling factor

    Returns:
        QuadrayCoordinate: The scaled coordinate
    """
    scaled_a = int(coord.a * factor)
    scaled_b = int(coord.b * factor)
    scaled_c = int(coord.c * factor)
    scaled_d = int(coord.d * factor)

    return QuadrayCoordinate(scaled_a, scaled_b, scaled_c, scaled_d)


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
    return type(polyhedron)(scaled_vertices)


def reflect(coord: QuadrayCoordinate, axis: str = 'origin') -> QuadrayCoordinate:
    """
    Reflect a coordinate across a specified axis or plane.

    Args:
        coord: The coordinate to reflect
        axis: The axis of reflection ('origin', 'a', 'b', 'c', 'd')

    Returns:
        QuadrayCoordinate: The reflected coordinate
    """
    if axis == 'origin':
        # Reflect through origin: negate all coordinates
        return QuadrayCoordinate(-coord.a, -coord.b, -coord.c, -coord.d)
    elif axis == 'a':
        # Reflect across a=constant plane: negate b, c, d
        return QuadrayCoordinate(coord.a, -coord.b, -coord.c, -coord.d)
    elif axis == 'b':
        # Reflect across b=constant plane: negate a, c, d
        return QuadrayCoordinate(-coord.a, coord.b, -coord.c, -coord.d)
    elif axis == 'c':
        # Reflect across c=constant plane: negate a, b, d
        return QuadrayCoordinate(-coord.a, -coord.b, coord.c, -coord.d)
    elif axis == 'd':
        # Reflect across d=constant plane: negate a, b, c
        return QuadrayCoordinate(-coord.a, -coord.b, -coord.c, coord.d)
    else:
        raise ValueError(f"Unknown reflection axis: {axis}")


def reflect_polyhedron(polyhedron: SymergeticsPolyhedron,
                      axis: str = 'origin') -> SymergeticsPolyhedron:
    """
    Reflect all vertices of a polyhedron across a specified axis.

    Args:
        polyhedron: The polyhedron to reflect
        axis: The axis of reflection

    Returns:
        SymergeticsPolyhedron: A new polyhedron with reflected vertices
    """
    reflected_vertices = [reflect(vertex, axis) for vertex in polyhedron.vertices]
    return type(polyhedron)(reflected_vertices)


def coordinate_transform(coord: QuadrayCoordinate,
                        matrix: np.ndarray) -> QuadrayCoordinate:
    """
    Apply a 4x4 transformation matrix to a Quadray coordinate.

    Args:
        coord: The coordinate to transform
        matrix: 4x4 transformation matrix

    Returns:
        QuadrayCoordinate: The transformed coordinate
    """
    if matrix.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4")

    # Convert coordinate to vector
    vec = np.array([[coord.a], [coord.b], [coord.c], [coord.d]], dtype=float)

    # Apply transformation
    transformed = matrix @ vec

    # Convert back to Quadray coordinate
    return QuadrayCoordinate(
        int(transformed[0, 0]),
        int(transformed[1, 0]),
        int(transformed[2, 0]),
        int(transformed[3, 0])
    )


def rotate_around_axis(coord: QuadrayCoordinate, axis: str, angle_degrees: float) -> QuadrayCoordinate:
    """
    Rotate a coordinate around a specified axis by a given angle.

    This function converts to XYZ space, performs the rotation, then converts back.

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


def project_to_plane(coord: QuadrayCoordinate, plane_normal: Tuple[float, float, float],
                    plane_point: Optional[Tuple[float, float, float]] = None) -> QuadrayCoordinate:
    """
    Project a coordinate onto a plane defined by a normal vector.

    Args:
        coord: The coordinate to project
        plane_normal: Normal vector of the plane (in XYZ space)
        plane_point: A point on the plane (defaults to origin)

    Returns:
        QuadrayCoordinate: The projected coordinate
    """
    if plane_point is None:
        plane_point = (0.0, 0.0, 0.0)

    # Convert to XYZ space
    point = coord.to_xyz()
    plane_normal = np.array(plane_normal)
    plane_point = np.array(plane_point)
    point = np.array(point)

    # Vector from plane point to the coordinate point
    vec = point - plane_point

    # Project vector onto plane normal
    projection = (np.dot(vec, plane_normal) / np.dot(plane_normal, plane_normal)) * plane_normal

    # Projected point
    projected_point = point - projection

    # Convert back to Quadray
    return QuadrayCoordinate.from_xyz(
        projected_point[0], projected_point[1], projected_point[2]
    )


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
    """
    Get the inverse transformation matrix from XYZ to Quadray coordinates.

    Args:
        embedding: Optional 3x4 embedding matrix

    Returns:
        np.ndarray: 3x3 matrix for XYZ to Quadray transformation
    """
    if embedding is None:
        embedding = urner_embedding()

    # For the Urner embedding, we can analytically find the inverse
    # This is a simplified approach - full matrix inversion could be used
    # for more general embeddings

    # Urner embedding inverse (approximate):
    # a = (x + y + z) / 2
    # b = (-x + y - z) / 2
    # c = (-x - y + z) / 2
    # d = (x - y - z) / 2

    inverse_matrix = np.array([
        [0.5, 0.5, 0.5],   # a coefficients
        [-0.5, 0.5, -0.5],  # b coefficients
        [-0.5, -0.5, 0.5],  # c coefficients
        [0.5, -0.5, -0.5],  # d coefficients
    ])

    return inverse_matrix


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
    offset = QuadrayCoordinate.from_xyz(float(dx), float(dy), float(dz))

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
