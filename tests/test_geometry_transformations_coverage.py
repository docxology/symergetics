#!/usr/bin/env python3
"""
Comprehensive Geometry Transformations Coverage Tests

This module contains tests to improve code coverage for geometry transformations,
focusing on the missing lines identified in the coverage report.
"""

import pytest
import numpy as np
from typing import List, Callable
from unittest.mock import patch, MagicMock

# Import the modules to test
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.geometry.polyhedra import Tetrahedron
from symergetics.geometry.transformations import (
    translate, scale, reflect, rotate_around_axis,
    translate_polyhedron, scale_polyhedron, reflect_polyhedron,
    coordinate_transform, project_to_plane, apply_transform_function,
    create_rotation_matrix, xyz_to_quadray_matrix, compose_transforms,
    translate_by_vector, scale_by_factor, rotate_around_xyz_axis
)


class TestTranslatePolyhedronCoverage:
    """Test coverage for translate_polyhedron function."""

    def test_translate_polyhedron_basic(self):
        """Test basic polyhedron translation."""
        tetra = Tetrahedron()
        offset = QuadrayCoordinate(1, 0, 0, 0)

        result = translate_polyhedron(tetra, offset)

        # Verify it's a new instance of the same type
        assert isinstance(result, Tetrahedron)
        assert result is not tetra

        # Verify vertices are translated
        for i, vertex in enumerate(result.vertices):
            expected = translate(tetra.vertices[i], offset)
            assert vertex.a == expected.a
            assert vertex.b == expected.b
            assert vertex.c == expected.c
            assert vertex.d == expected.d


class TestScalePolyhedronCoverage:
    """Test coverage for scale_polyhedron function."""

    def test_scale_polyhedron_basic(self):
        """Test basic polyhedron scaling."""
        tetra = Tetrahedron()
        factor = 2.0

        result = scale_polyhedron(tetra, factor)

        # Verify it's a new instance of the same type
        assert isinstance(result, Tetrahedron)
        assert result is not tetra

        # Verify vertices are scaled
        for i, vertex in enumerate(result.vertices):
            expected = scale(tetra.vertices[i], factor)
            assert vertex.a == expected.a
            assert vertex.b == expected.b
            assert vertex.c == expected.c
            assert vertex.d == expected.d

    def test_scale_polyhedron_zero_factor(self):
        """Test polyhedron scaling with zero factor."""
        tetra = Tetrahedron()
        result = scale_polyhedron(tetra, 0)

        # All vertices should be at origin
        for vertex in result.vertices:
            assert vertex.a == 0
            assert vertex.b == 0
            assert vertex.c == 0
            assert vertex.d == 0


class TestReflectPolyhedronCoverage:
    """Test coverage for reflect_polyhedron function."""

    def test_reflect_polyhedron_origin(self):
        """Test polyhedron reflection through origin."""
        tetra = Tetrahedron()

        result = reflect_polyhedron(tetra, 'origin')

        # Verify it's a new instance of the same type
        assert isinstance(result, Tetrahedron)
        assert result is not tetra

        # Verify vertices are reflected
        for i, vertex in enumerate(result.vertices):
            expected = reflect(tetra.vertices[i], 'origin')
            assert vertex.a == expected.a
            assert vertex.b == expected.b
            assert vertex.c == expected.c
            assert vertex.d == expected.d

    def test_reflect_polyhedron_axis(self):
        """Test polyhedron reflection across different axes."""
        tetra = Tetrahedron()

        for axis in ['a', 'b', 'c', 'd']:
            result = reflect_polyhedron(tetra, axis)
            assert isinstance(result, Tetrahedron)
            assert result is not tetra


class TestReflectAxisCoverage:
    """Test coverage for reflect function with different axes."""

    def test_reflect_axis_a(self):
        """Test reflection across a=constant plane."""
        coord = QuadrayCoordinate(1, 2, 3, 4, normalize=False)
        result = reflect(coord, 'a')

        # Should negate b, c, d, then normalize
        # Original: (1, 2, 3, 4) -> reflected: (1, -2, -3, -4) -> normalized: (6, 3, 2, 1)
        expected = QuadrayCoordinate(1, -2, -3, -4, normalize=True)
        assert result.a == expected.a
        assert result.b == expected.b
        assert result.c == expected.c
        assert result.d == expected.d

    def test_reflect_axis_b(self):
        """Test reflection across b=constant plane."""
        coord = QuadrayCoordinate(1, 2, 3, 4, normalize=False)
        result = reflect(coord, 'b')

        # Should negate a, c, d, then normalize
        # Original: (1, 2, 3, 4) -> reflected: (-1, 2, -3, -4) -> normalized: (3, 6, 1, 0)
        expected = QuadrayCoordinate(-1, 2, -3, -4, normalize=True)
        assert result.a == expected.a
        assert result.b == expected.b
        assert result.c == expected.c
        assert result.d == expected.d

    def test_reflect_axis_c(self):
        """Test reflection across c=constant plane."""
        coord = QuadrayCoordinate(1, 2, 3, 4, normalize=False)
        result = reflect(coord, 'c')

        # Should negate a, b, d, then normalize
        # Original: (1, 2, 3, 4) -> reflected: (-1, -2, 3, -4) -> normalized: (2, 1, 6, -1) -> (3, 2, 7, 0)
        expected = QuadrayCoordinate(-1, -2, 3, -4, normalize=True)
        assert result.a == expected.a
        assert result.b == expected.b
        assert result.c == expected.c
        assert result.d == expected.d

    def test_reflect_axis_d(self):
        """Test reflection across d=constant plane."""
        coord = QuadrayCoordinate(1, 2, 3, 4, normalize=False)
        result = reflect(coord, 'd')

        # Should negate a, b, c, then normalize
        # Original: (1, 2, 3, 4) -> reflected: (-1, -2, -3, 4) -> normalized: (1, 0, -1, 6) -> (2, 1, 0, 7)
        expected = QuadrayCoordinate(-1, -2, -3, 4, normalize=True)
        assert result.a == expected.a
        assert result.b == expected.b
        assert result.c == expected.c
        assert result.d == expected.d


class TestRotateAroundAxisCoverage:
    """Test coverage for rotate_around_axis function."""

    def test_rotate_around_x_axis(self):
        """Test rotation around X axis."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        result = rotate_around_axis(coord, 'x', 90)

        # Should be a valid QuadrayCoordinate
        assert isinstance(result, QuadrayCoordinate)

    def test_rotate_around_y_axis(self):
        """Test rotation around Y axis."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        result = rotate_around_axis(coord, 'y', 90)

        # Should be a valid QuadrayCoordinate
        assert isinstance(result, QuadrayCoordinate)

    def test_rotate_around_z_axis(self):
        """Test rotation around Z axis."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        result = rotate_around_axis(coord, 'z', 90)

        # Should be a valid QuadrayCoordinate
        assert isinstance(result, QuadrayCoordinate)

    def test_rotate_around_axis_zero_angle(self):
        """Test rotation with zero angle."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        result = rotate_around_axis(coord, 'x', 0)

        # Should be a valid QuadrayCoordinate (conversion may change values slightly)
        assert isinstance(result, QuadrayCoordinate)

    def test_rotate_around_axis_360_degrees(self):
        """Test rotation with 360 degrees."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        result = rotate_around_axis(coord, 'x', 360)

        # Should be a valid QuadrayCoordinate
        assert isinstance(result, QuadrayCoordinate)


class TestProjectToPlaneCoverage:
    """Test coverage for project_to_plane function."""

    def test_project_to_plane_basic(self):
        """Test basic plane projection."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        plane_normal = (0, 0, 1)  # XY plane

        result = project_to_plane(coord, plane_normal)

        # Should be a valid QuadrayCoordinate
        assert isinstance(result, QuadrayCoordinate)

    def test_project_to_plane_with_plane_point(self):
        """Test plane projection with custom plane point."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        plane_normal = (1, 0, 0)  # YZ plane
        plane_point = (5, 0, 0)

        result = project_to_plane(coord, plane_normal, plane_point)

        # Should be a valid QuadrayCoordinate
        assert isinstance(result, QuadrayCoordinate)

    def test_project_to_plane_origin_point(self):
        """Test plane projection with None plane point (defaults to origin)."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        plane_normal = (0, 1, 0)  # XZ plane

        result = project_to_plane(coord, plane_normal, None)

        # Should be a valid QuadrayCoordinate
        assert isinstance(result, QuadrayCoordinate)


class TestApplyTransformFunctionCoverage:
    """Test coverage for apply_transform_function."""

    def test_apply_transform_function_basic(self):
        """Test applying transform function to coordinate list."""
        coords = [
            QuadrayCoordinate(1, 2, 3, 4),
            QuadrayCoordinate(5, 6, 7, 8)
        ]

        # Simple transform function
        def scale_transform(coord):
            return scale(coord, 2)

        result = apply_transform_function(coords, scale_transform)

        # Should return list of same length
        assert len(result) == len(coords)

        # Each coordinate should be transformed
        for i, coord in enumerate(result):
            expected = scale(coords[i], 2)
            assert coord.a == expected.a
            assert coord.b == expected.b
            assert coord.c == expected.c
            assert coord.d == expected.d

    def test_apply_transform_function_empty_list(self):
        """Test applying transform function to empty list."""
        coords = []

        def identity_transform(coord):
            return coord

        result = apply_transform_function(coords, identity_transform)

        # Should return empty list
        assert result == []


class TestCreateRotationMatrixCoverage:
    """Test coverage for create_rotation_matrix function."""

    def test_create_rotation_matrix_x_axis(self):
        """Test 3D rotation matrix for X axis."""
        matrix = create_rotation_matrix('x', 90)

        # Should be 3x3 matrix
        assert matrix.shape == (3, 3)

        # Check some known values for 90-degree rotation around X
        assert abs(matrix[0, 0] - 1) < 1e-10  # cos(90) = 0, but X axis rotation preserves X
        assert abs(matrix[1, 1] - 0) < 1e-10  # cos(90) = 0
        assert abs(matrix[1, 2] - (-1)) < 1e-10  # -sin(90) = -1
        assert abs(matrix[2, 1] - 1) < 1e-10   # sin(90) = 1
        assert abs(matrix[2, 2] - 0) < 1e-10   # cos(90) = 0

    def test_create_rotation_matrix_y_axis(self):
        """Test 3D rotation matrix for Y axis."""
        matrix = create_rotation_matrix('y', 90)

        # Should be 3x3 matrix
        assert matrix.shape == (3, 3)

        # Check some known values for 90-degree rotation around Y
        assert abs(matrix[0, 0] - 0) < 1e-10   # cos(90) = 0
        assert abs(matrix[0, 2] - 1) < 1e-10   # sin(90) = 1
        assert abs(matrix[1, 1] - 1) < 1e-10   # Y axis rotation preserves Y
        assert abs(matrix[2, 0] - (-1)) < 1e-10 # -sin(90) = -1
        assert abs(matrix[2, 2] - 0) < 1e-10   # cos(90) = 0

    def test_create_rotation_matrix_z_axis(self):
        """Test 3D rotation matrix for Z axis."""
        matrix = create_rotation_matrix('z', 90)

        # Should be 3x3 matrix
        assert matrix.shape == (3, 3)

        # Check some known values for 90-degree rotation around Z
        assert abs(matrix[0, 0] - 0) < 1e-10   # cos(90) = 0
        assert abs(matrix[0, 1] - (-1)) < 1e-10 # -sin(90) = -1
        assert abs(matrix[1, 0] - 1) < 1e-10   # sin(90) = 1
        assert abs(matrix[1, 1] - 0) < 1e-10   # cos(90) = 0
        assert abs(matrix[2, 2] - 1) < 1e-10   # Z axis rotation preserves Z

    def test_create_rotation_matrix_zero_angle(self):
        """Test 3D rotation matrix with zero angle (should be identity)."""
        for axis in ['x', 'y', 'z']:
            matrix = create_rotation_matrix(axis, 0)

            # Should be identity matrix
            expected = np.eye(3)
            np.testing.assert_array_almost_equal(matrix, expected)


class TestXyzToQuadrayMatrixCoverage:
    """Test coverage for xyz_to_quadray_matrix function."""

    def test_xyz_to_quadray_matrix_basic(self):
        """Test basic XYZ to Quadray matrix."""
        matrix = xyz_to_quadray_matrix()

        # Should be 4x3 matrix
        assert matrix.shape == (4, 3)

    def test_xyz_to_quadray_matrix_with_custom_embedding(self):
        """Test XYZ to Quadray matrix with custom embedding matrix."""
        custom_embedding = np.eye(4)  # Identity matrix
        matrix = xyz_to_quadray_matrix(custom_embedding)

        # Should still work
        assert matrix.shape == (4, 3)


class TestComposeTransformsCoverage:
    """Test coverage for compose_transforms function."""

    def test_compose_transforms_basic(self):
        """Test basic transform composition."""
        def transform1(coord):
            return translate(coord, QuadrayCoordinate(1, 0, 0, 0))

        def transform2(coord):
            return scale(coord, 2)

        composed = compose_transforms(transform1, transform2)

        coord = QuadrayCoordinate(1, 2, 3, 4)
        result = composed(coord)

        # Should apply both transforms: first translate, then scale
        intermediate = transform1(coord)
        expected = transform2(intermediate)

        assert result.a == expected.a
        assert result.b == expected.b
        assert result.c == expected.c
        assert result.d == expected.d

    def test_compose_transforms_single_transform(self):
        """Test composing a single transform."""
        def transform(coord):
            return scale(coord, 3)

        composed = compose_transforms(transform)

        coord = QuadrayCoordinate(1, 2, 3, 4)
        result = composed(coord)
        expected = transform(coord)

        assert result.a == expected.a
        assert result.b == expected.b
        assert result.c == expected.c
        assert result.d == expected.d

    def test_compose_transforms_no_transforms(self):
        """Test composing no transforms (should be identity)."""
        composed = compose_transforms()

        coord = QuadrayCoordinate(1, 2, 3, 4)
        result = composed(coord)

        # Should be unchanged
        assert result.a == coord.a
        assert result.b == coord.b
        assert result.c == coord.c
        assert result.d == coord.d


class TestCoordinateTransformCoverage:
    """Test coverage for coordinate_transform function."""

    def test_coordinate_transform_basic(self):
        """Test basic coordinate transformation."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        transform_matrix = np.eye(4)  # Identity matrix

        result = coordinate_transform(coord, transform_matrix)

        # Should be unchanged (identity transform)
        assert result.a == coord.a
        assert result.b == coord.b
        assert result.c == coord.c
        assert result.d == coord.d

    def test_coordinate_transform_with_scaling(self):
        """Test coordinate transformation with scaling."""
        coord = QuadrayCoordinate(1, 2, 3, 4)

        # Scaling matrix (multiply each coordinate by 2)
        transform_matrix = np.array([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2]
        ])

        result = coordinate_transform(coord, transform_matrix)

        # Should be scaled by 2 in each coordinate
        assert result.a == coord.a * 2
        assert result.b == coord.b * 2
        assert result.c == coord.c * 2
        assert result.d == coord.d * 2

    def test_coordinate_transform_invalid_matrix_shape(self):
        """Test coordinate transformation with invalid matrix shape."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        invalid_matrix = np.eye(3)  # 3x3 matrix

        with pytest.raises(ValueError, match="Transformation matrix must be 4x4"):
            coordinate_transform(coord, invalid_matrix)
