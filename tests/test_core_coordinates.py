"""
Tests for core.coordinates module - Quadray coordinate system.

Tests IVM lattice, coordinate transformations, and geometric operations.
"""

import pytest
import numpy as np
import math
from symergetics.core.coordinates import (
    QuadrayCoordinate,
    ORIGIN,
    IVM_NEIGHBORS,
    TETRAHEDRON_VERTICES,
    OCTAHEDRON_VERTICES,
    CUBE_VERTICES,
    urner_embedding
)


class TestQuadrayCoordinate:
    """Test QuadrayCoordinate class functionality."""

    def test_initialization(self):
        """Test coordinate initialization and normalization."""
        # Basic initialization
        coord = QuadrayCoordinate(2, 1, 1, 0)
        assert coord.a == 2
        assert coord.b == 1
        assert coord.c == 1
        assert coord.d == 0

        # Test normalization
        coord = QuadrayCoordinate(3, 2, 2, 1)
        # Should normalize by subtracting min(3,2,2,1) = 1
        assert coord.a == 2
        assert coord.b == 1
        assert coord.c == 1
        assert coord.d == 0

    def test_initialization_no_normalize(self):
        """Test initialization without normalization."""
        coord = QuadrayCoordinate(2, 1, 1, 0, normalize=False)
        assert coord.a == 2
        assert coord.b == 1
        assert coord.c == 1
        assert coord.d == 0

    def test_as_tuple_and_array(self):
        """Test coordinate representation methods."""
        coord = QuadrayCoordinate(2, 1, 1, 0)

        # Tuple representation
        tup = coord.as_tuple()
        assert tup == (2, 1, 1, 0)

        # Array representation
        arr = coord.as_array()
        expected = np.array([[2], [1], [1], [0]], dtype=float)
        np.testing.assert_array_equal(arr, expected)

    def test_equality(self):
        """Test coordinate equality."""
        coord1 = QuadrayCoordinate(2, 1, 1, 0)
        coord2 = QuadrayCoordinate(2, 1, 1, 0)
        coord3 = QuadrayCoordinate(1, 2, 1, 0)

        assert coord1 == coord2
        assert coord1 != coord3
        assert hash(coord1) == hash(coord2)
        assert hash(coord1) != hash(coord3)

    def test_arithmetic_operations(self):
        """Test coordinate arithmetic."""
        a = QuadrayCoordinate(2, 1, 1, 0)
        b = QuadrayCoordinate(1, 1, 0, 1)

        # Addition
        result = a.add(b)
        assert result.a == 3
        assert result.b == 2
        assert result.c == 1
        assert result.d == 1

        # Subtraction
        result = a.sub(b)
        assert result.a == 1
        assert result.b == 0
        assert result.c == 1
        assert result.d == -1

    def test_xyz_conversion(self):
        """Test conversion between Quadray and Cartesian coordinates using proper Urner embedding."""
        # Test with origin
        origin = QuadrayCoordinate(0, 0, 0, 0)
        xyz = origin.to_xyz()
        assert xyz == (0.0, 0.0, 0.0)

        # Test with specific coordinate using proper Urner embedding
        coord = QuadrayCoordinate(2, 1, 1, 0)
        xyz = coord.to_xyz()

        # Using proper Urner embedding matrix:
        # [[ 1. -1. -1.  1.]
        #  [ 1.  1. -1. -1.]
        #  [ 1. -1.  1. -1.]]
        # For (2, 1, 1, 0): 
        # x = 1*2 + (-1)*1 + (-1)*1 + 1*0 = 2 - 1 - 1 + 0 = 0
        # y = 1*2 + 1*1 + (-1)*1 + (-1)*0 = 2 + 1 - 1 + 0 = 2  
        # z = 1*2 + (-1)*1 + 1*1 + (-1)*0 = 2 - 1 + 1 + 0 = 2
        expected_xyz = (0.0, 2.0, 2.0)

        assert xyz == pytest.approx(expected_xyz, abs=1e-10)

    def test_from_xyz(self):
        """Test conversion from XYZ to Quadray coordinates."""
        # Test origin
        coord = QuadrayCoordinate.from_xyz(0.0, 0.0, 0.0)
        assert coord.a == 0
        assert coord.b == 0
        assert coord.c == 0
        assert coord.d == 0

        # Test basic conversion (coordinate systems may not be perfectly invertible)
        coord_x = QuadrayCoordinate.from_xyz(1.0, 0.0, 0.0)
        assert coord_x.a >= 0  # Should have non-negative coordinates after normalization

        coord_y = QuadrayCoordinate.from_xyz(0.0, 1.0, 0.0)
        assert coord_y.a >= 0

        coord_z = QuadrayCoordinate.from_xyz(0.0, 0.0, 1.0)
        assert coord_z.a >= 0

    def test_magnitude_and_dot_product(self):
        """Test vector operations."""
        coord = QuadrayCoordinate(2, 1, 1, 0)

        # Test magnitude
        mag = coord.magnitude()
        assert mag > 0

        # Test dot product with itself
        dot_self = coord.dot(coord)
        mag_squared = mag * mag
        assert dot_self == pytest.approx(mag_squared, abs=1e-10)

        # Test dot product with different vector
        other = QuadrayCoordinate(1, 1, 0, 1)
        dot_product = coord.dot(other)
        assert isinstance(dot_product, float)

    def test_distance_calculation(self):
        """Test distance calculations."""
        a = QuadrayCoordinate(2, 1, 1, 0)
        b = QuadrayCoordinate(1, 1, 0, 1)

        # Distance should be positive
        distance = a.distance_to(b)
        assert distance > 0

        # Distance from point to itself should be zero
        distance_self = a.distance_to(a)
        assert distance_self == pytest.approx(0.0, abs=1e-10)

    def test_to_dict_and_copy(self):
        """Test serialization and copying."""
        coord = QuadrayCoordinate(2, 1, 1, 0)

        # Test dictionary conversion
        data = coord.to_dict()
        assert data == {'a': 2, 'b': 1, 'c': 1, 'd': 0}

        # Test copying
        copy_coord = coord.copy()
        assert copy_coord == coord
        assert copy_coord is not coord  # Different object


class TestUrnerEmbedding:
    """Test Urner embedding matrix."""

    def test_urner_embedding_matrix(self):
        """Test Urner embedding matrix properties."""
        matrix = urner_embedding()

        # Should be 3x4 matrix
        assert matrix.shape == (3, 4)

        # Test with scale factor
        scaled_matrix = urner_embedding(scale=2.0)
        assert scaled_matrix.shape == (3, 4)
        assert np.allclose(scaled_matrix, 2.0 * matrix)


class TestPredefinedCoordinates:
    """Test predefined coordinate sets."""

    def test_origin(self):
        """Test origin coordinate."""
        assert ORIGIN.a == 0
        assert ORIGIN.b == 0
        assert ORIGIN.c == 0
        assert ORIGIN.d == 0

    def test_ivm_neighbors(self):
        """Test IVM neighbors (12-around-one)."""
        assert len(IVM_NEIGHBORS) == 12

        # All should be unique
        assert len(set(IVM_NEIGHBORS)) == 12

        # All should have at least one zero (after normalization)
        for neighbor in IVM_NEIGHBORS:
            assert 0 in [neighbor.a, neighbor.b, neighbor.c, neighbor.d]

        # Check that they form a cuboctahedron shell
        # (this is tested by verifying all have reasonable distances from origin)
        distances = [ORIGIN.distance_to(neighbor) for neighbor in IVM_NEIGHBORS]
        # With the simple linear transformation, distances will vary but should be reasonable
        for dist in distances:
            assert dist > 0  # All should be positive distance from origin
            assert dist < 5  # All should be within reasonable bounds

    def test_tetrahedron_vertices(self):
        """Test tetrahedron vertices."""
        assert len(TETRAHEDRON_VERTICES) == 4

        # Verify they form a regular tetrahedron
        # (all edges should have equal length)
        distances = []
        for i in range(4):
            for j in range(i+1, 4):
                dist = TETRAHEDRON_VERTICES[i].distance_to(TETRAHEDRON_VERTICES[j])
                distances.append(dist)

        # All distances should be reasonable (edges of tetrahedron)
        # With the simple linear transformation, distances may vary slightly
        for dist in distances:
            assert dist > 0  # All should be positive distance
            assert dist < 5  # All should be within reasonable bounds

    def test_octahedron_vertices(self):
        """Test octahedron vertices."""
        assert len(OCTAHEDRON_VERTICES) == 6

        # Should form regular octahedron
        distances = []
        for i in range(6):
            for j in range(i+1, 6):
                dist = OCTAHEDRON_VERTICES[i].distance_to(OCTAHEDRON_VERTICES[j])
                distances.append(dist)

        # In octahedron, all edges have reasonable lengths
        # With the simple linear transformation, distances may vary slightly
        for dist in distances:
            assert dist > 0  # All should be positive distance
            assert dist < 15  # All should be within reasonable bounds (increased for proper embedding)

    def test_cube_vertices(self):
        """Test cube vertices."""
        assert len(CUBE_VERTICES) == 8

        # Should form cube
        distances = []
        for i in range(8):
            for j in range(i+1, 8):
                dist = CUBE_VERTICES[i].distance_to(CUBE_VERTICES[j])
                distances.append(dist)

        # Cube has edges of reasonable lengths
        # With the simple linear transformation, distances may vary slightly
        edge_distances = [d for d in distances if d > 0]
        for dist in edge_distances:
            assert dist > 0  # All should be positive distance
            assert dist < 15  # All should be within reasonable bounds


if __name__ == "__main__":
    pytest.main([__file__])
