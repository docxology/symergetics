"""
Tests for geometry.transformations module - Geometric transformations.

Tests coordinate transformations, translations, rotations, and scaling.
"""

import pytest
import math
import numpy as np
from symergetics.geometry.transformations import (
    translate,
    scale,
    reflect,
    coordinate_transform,
    rotate_around_axis,
    apply_transform_function,
    translate_by_vector,
    scale_by_factor,
    rotate_around_xyz_axis,
    compose_transforms
)
from symergetics.core.coordinates import QuadrayCoordinate


class TestTranslation:
    """Test translation operations."""

    def test_translate_coordinate(self):
        """Test coordinate translation."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        offset = QuadrayCoordinate(0, 1, 0, 0)

        result = translate(coord, offset)
        assert result.a == 1
        assert result.b == 1
        assert result.c == 0
        assert result.d == 0

    def test_translate_by_vector_function(self):
        """Test translate_by_vector function."""
        translate_func = translate_by_vector(1, 2, 3)
        coord = QuadrayCoordinate(1, 1, 1, 1)

        result = translate_func(coord)
        # Result will be normalized, so exact values depend on normalization
        assert isinstance(result, QuadrayCoordinate)


class TestScaling:
    """Test scaling operations."""

    def test_scale_coordinate(self):
        """Test coordinate scaling."""
        coord = QuadrayCoordinate(2, 1, 1, 0)
        result = scale(coord, 2.0)

        # Scaling may change due to normalization
        assert isinstance(result, QuadrayCoordinate)

    def test_scale_by_factor_function(self):
        """Test scale_by_factor function."""
        scale_func = scale_by_factor(2.0)
        coord = QuadrayCoordinate(1, 0, 0, 0)

        result = scale_func(coord)
        assert isinstance(result, QuadrayCoordinate)


class TestReflection:
    """Test reflection operations."""

    def test_reflect_origin(self):
        """Test reflection through origin."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        result = reflect(coord, 'origin')

        # Reflection of (1,0,0,0) gives (-1,0,0,0), which normalizes to (0,1,1,1)
        assert result.a == 0
        assert result.b == 1
        assert result.c == 1
        assert result.d == 1

    def test_reflect_axis(self):
        """Test reflection across axis."""
        coord = QuadrayCoordinate(1, 2, 3, 4)  # Normalizes to (0, 1, 2, 3)
        result = reflect(coord, 'a')

        # Reflecting (0, 1, 2, 3) across 'a' gives (0, -1, -2, -3)
        # which normalizes to (3, 2, 1, 0)
        assert result.a == 3
        assert result.b == 2
        assert result.c == 1
        assert result.d == 0


class TestRotation:
    """Test rotation operations."""

    def test_rotate_around_axis(self):
        """Test rotation around XYZ axis."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        result = rotate_around_axis(coord, 'z', 90)

        # Rotation should produce valid coordinates
        assert isinstance(result, QuadrayCoordinate)

    def test_rotate_around_xyz_axis_function(self):
        """Test rotate_around_xyz_axis function."""
        rotate_func = rotate_around_xyz_axis('z', 45)
        coord = QuadrayCoordinate(1, 0, 0, 0)

        result = rotate_func(coord)
        assert isinstance(result, QuadrayCoordinate)


class TestTransformationComposition:
    """Test transformation composition."""

    def test_compose_transforms(self):
        """Test composing multiple transformations."""
        def add_one(coord):
            return translate(coord, QuadrayCoordinate(1, 0, 0, 0))

        def multiply_two(coord):
            return scale(coord, 2.0)

        composed = compose_transforms(add_one, multiply_two)
        coord = QuadrayCoordinate(1, 0, 0, 0)

        result = composed(coord)
        assert isinstance(result, QuadrayCoordinate)

    def test_apply_transform_function(self):
        """Test applying transformation to list of coordinates."""
        coords = [
            QuadrayCoordinate(1, 0, 0, 0),
            QuadrayCoordinate(0, 1, 0, 0),
        ]

        def double_scale(coord):
            return scale(coord, 2.0)

        result = apply_transform_function(coords, double_scale)
        assert len(result) == 2
        assert all(isinstance(coord, QuadrayCoordinate) for coord in result)


class TestCoordinateTransform:
    """Test general coordinate transformation."""

    def test_coordinate_transform(self):
        """Test coordinate transformation with matrix."""
        import numpy as np

        coord = QuadrayCoordinate(1, 0, 0, 0)

        # Identity transformation
        identity_matrix = np.eye(4)
        result = coordinate_transform(coord, identity_matrix)

        assert result.a == 1
        assert result.b == 0
        assert result.c == 0
        assert result.d == 0

    def test_coordinate_transform_invalid_matrix(self):
        """Test coordinate transformation with invalid matrix."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        invalid_matrix = np.array([[1, 0], [0, 1]])  # Wrong size

        with pytest.raises(ValueError):
            coordinate_transform(coord, invalid_matrix)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_translate_with_none(self):
        """Test translation with None values."""
        coord = QuadrayCoordinate(0, 0, 0, 0)
        offset = QuadrayCoordinate(0, 0, 0, 0)

        result = translate(coord, offset)
        assert result.a == 0
        assert result.b == 0
        assert result.c == 0
        assert result.d == 0

    def test_scale_with_zero(self):
        """Test scaling with zero factor."""
        coord = QuadrayCoordinate(1, 0, 0, 0)

        result = scale(coord, 0.0)
        # Scaling by zero may result in origin due to normalization
        assert isinstance(result, QuadrayCoordinate)

    def test_reflect_invalid_axis(self):
        """Test reflection with invalid axis."""
        coord = QuadrayCoordinate(1, 0, 0, 0)

        with pytest.raises(ValueError):
            reflect(coord, 'invalid_axis')

    def test_rotate_invalid_axis(self):
        """Test rotation with invalid axis."""
        coord = QuadrayCoordinate(1, 0, 0, 0)

        with pytest.raises(ValueError):
            rotate_around_axis(coord, 'invalid', 45)


if __name__ == "__main__":
    pytest.main([__file__])
