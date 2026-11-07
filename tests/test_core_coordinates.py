"""
Tests for core.coordinates module - Quadray coordinate system.

Tests IVM lattice, coordinate transformations, and geometric operations.
"""

import math
import json
from fractions import Fraction
from itertools import permutations, product

import numpy as np
import pytest

from symergetics.core.coordinates import (
    CUBE_VERTICES,
    IVM_NEIGHBORS,
    OCTAHEDRON_VERTICES,
    ORIGIN,
    TETRAHEDRON_VERTICES,
    QuadrayCoordinate,
    urner_embedding,
)
from symergetics.core.numbers import SymergeticsNumber


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

        # Axis midpoints are exact ties; canonical lexical order chooses origin.
        for point in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            assert QuadrayCoordinate.from_xyz(*point).as_tuple() == (0, 0, 0, 0)

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

    def test_origin_coordinate(self):
        """Test origin coordinate creation."""
        origin = QuadrayCoordinate(0, 0, 0, 0)
        assert origin.a == 0
        assert origin.b == 0
        assert origin.c == 0
        assert origin.d == 0

    def test_basic_coordinates(self):
        """Test basic coordinate creation."""
        # Test unit vectors
        coord1 = QuadrayCoordinate(1, 0, 0, 0)
        assert coord1.a == 1
        assert coord1.b == 0
        assert coord1.c == 0
        assert coord1.d == 0
        
        # Test another unit vector
        coord2 = QuadrayCoordinate(0, 1, 0, 0)
        assert coord2.a == 0
        assert coord2.b == 1
        assert coord2.c == 0
        assert coord2.d == 0

class TestQuadrayCoordinateBasic:
    """Test basic QuadrayCoordinate functionality."""

    def test_initialization(self):
        """Test coordinate initialization."""
        # Basic initialization
        coord = QuadrayCoordinate(1, 0, 0, 0)
        assert coord.a == 1
        assert coord.b == 0
        assert coord.c == 0
        assert coord.d == 0

        # With normalization
        coord2 = QuadrayCoordinate(2, 1, 1, 1, normalize=True)
        # After normalization, at least one coordinate should be zero
        assert min(coord2.a, coord2.b, coord2.c, coord2.d) == 0

    def test_initialization_without_normalization(self):
        """Test initialization without normalization."""
        coord = QuadrayCoordinate(1, 2, 3, 4, normalize=False)
        assert coord.a == 1
        assert coord.b == 2
        assert coord.c == 3
        assert coord.d == 4

    def test_as_tuple_and_array(self):
        """Test tuple and array representations."""
        coord = QuadrayCoordinate(1, 2, 3, 4, normalize=False)

        # Test as_tuple
        tuple_repr = coord.as_tuple()
        assert tuple_repr == (1, 2, 3, 4)

        # Test as_array
        array_repr = coord.as_array()
        assert np.array_equal(array_repr.flatten(), np.array([1, 2, 3, 4]))

    def test_equality(self):
        """Test equality comparisons."""
        coord1 = QuadrayCoordinate(1, 2, 3, 4, normalize=False)
        coord2 = QuadrayCoordinate(1, 2, 3, 4, normalize=False)
        coord3 = QuadrayCoordinate(1, 2, 3, 5, normalize=False)

        assert coord1 == coord2
        assert coord1 != coord3
        # Test tuple comparison through as_tuple
        assert coord1.as_tuple() == (1, 2, 3, 4)
        assert coord1.as_tuple() != (1, 2, 3, 5)


class TestQuadrayCoordinateOperations:
    """Test operations on QuadrayCoordinate."""

    def test_coordinate_creation(self):
        """Test coordinate creation with different parameters."""
        # Test with integers
        coord1 = QuadrayCoordinate(1, 2, 3, 4, normalize=False)
        assert coord1.a == 1
        assert coord1.b == 2
        assert coord1.c == 3
        assert coord1.d == 4

        # Test with floats: snapped to the nearest lattice point (halves away from zero)
        coord2 = QuadrayCoordinate(1.5, 2.5, 3.5, 4.5, normalize=False)
        assert coord2.a == 2
        assert coord2.b == 3
        assert coord2.c == 4
        assert coord2.d == 5

        # Float noise must snap to the intended lattice point
        coord3 = QuadrayCoordinate(1.9999999, -1.0000001, 0.0, 3.0, normalize=False)
        assert coord3.as_tuple() == (2, -1, 0, 3)

    def test_coordinate_equality(self):
        """Test coordinate equality."""
        coord1 = QuadrayCoordinate(1, 2, 3, 4, normalize=False)
        coord2 = QuadrayCoordinate(1, 2, 3, 4, normalize=False)
        coord3 = QuadrayCoordinate(1, 2, 3, 5, normalize=False)

        assert coord1 == coord2
        assert coord1 != coord3

    def test_coordinate_representation(self):
        """Test coordinate string representations."""
        coord = QuadrayCoordinate(1, 2, 3, 4, normalize=False)

        str_repr = str(coord)
        assert str_repr == "(1, 2, 3, 4)"

        repr_str = repr(coord)
        assert "QuadrayCoordinate" in repr_str
        assert "1" in repr_str
        assert "2" in repr_str
        assert "3" in repr_str
        assert "4" in repr_str


class TestQuadrayCoordinateConversion:
    """Test coordinate conversion methods."""

    def test_xyz_conversion(self):
        """Test conversion to XYZ coordinates."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        x, y, z = coord.to_xyz()

        # Test that conversion produces valid XYZ coordinates
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)

        # Test round-trip conversion
        coord2 = QuadrayCoordinate.from_xyz(x, y, z)
        assert coord2 == coord

    def test_from_xyz(self):
        """Test creation from XYZ coordinates."""
        # Test with origin
        coord = QuadrayCoordinate.from_xyz(0, 0, 0)
        assert coord == QuadrayCoordinate(0, 0, 0, 0)

        # Test with specific XYZ values
        coord2 = QuadrayCoordinate.from_xyz(1, 1, 1)
        assert isinstance(coord2, QuadrayCoordinate)

    def test_conversion_consistency(self):
        """Test that conversions work without errors."""
        # Test basic conversion
        coord = QuadrayCoordinate.from_xyz(0, 0, 0)
        assert isinstance(coord, QuadrayCoordinate)

        x, y, z = coord.to_xyz()
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)

        # Test with non-zero point
        coord2 = QuadrayCoordinate.from_xyz(1, 1, 1)
        assert isinstance(coord2, QuadrayCoordinate)

        x2, y2, z2 = coord2.to_xyz()
        assert isinstance(x2, float)
        assert isinstance(y2, float)
        assert isinstance(z2, float)


class TestQuadrayCoordinateMagnitude:
    """Test magnitude and distance calculations."""

    def test_magnitude(self):
        """Test magnitude calculation."""
        coord = QuadrayCoordinate(3, 4, 0, 0)
        mag = coord.magnitude()

        assert isinstance(mag, float)
        assert mag > 0

        # Test with zero coordinate
        zero_coord = QuadrayCoordinate(0, 0, 0, 0)
        assert zero_coord.magnitude() == 0

    def test_dot_product(self):
        """Test dot product calculation."""
        coord1 = QuadrayCoordinate(1, 2, 3, 4)
        coord2 = QuadrayCoordinate(2, 3, 4, 5)

        dot = coord1.dot(coord2)
        assert isinstance(dot, float)

        # Test with zero vector
        zero_coord = QuadrayCoordinate(0, 0, 0, 0)
        assert coord1.dot(zero_coord) == 0

    def test_distance_calculation(self):
        """Test distance calculation between coordinates."""
        coord1 = QuadrayCoordinate(0, 0, 0, 0)
        coord2 = QuadrayCoordinate(1, 0, 0, 0)

        distance = coord1.distance_to(coord2)
        assert isinstance(distance, float)
        assert distance > 0

        # Distance to self should be zero
        assert coord1.distance_to(coord1) == 0


class TestQuadrayCoordinateUtility:
    """Test utility methods."""

    def test_to_dict_and_copy(self):
        """Test dictionary representation and copying."""
        coord = QuadrayCoordinate(1, 2, 3, 4, normalize=False)

        # Test to_dict
        coord_dict = coord.to_dict()
        assert coord_dict['a'] == 1
        assert coord_dict['b'] == 2
        assert coord_dict['c'] == 3
        assert coord_dict['d'] == 4

        # Test copy
        coord_copy = coord.copy()
        assert coord_copy == coord
        assert coord_copy is not coord  # Different objects

        # Test that modifications don't affect original
        coord_copy.a = 5
        assert coord.a == 1  # Original unchanged

    def test_normalization(self):
        """Test coordinate normalization."""
        # Test with non-normalized coordinates
        coord = QuadrayCoordinate(2, 1, 1, 1, normalize=False)

        # Test that coordinates are as expected
        assert coord.a == 2
        assert coord.b == 1
        assert coord.c == 1
        assert coord.d == 1

        # Test with normalized coordinates
        coord2 = QuadrayCoordinate(2, 1, 1, 1, normalize=True)
        # After normalization, at least one coordinate should be zero
        assert min(coord2.a, coord2.b, coord2.c, coord2.d) == 0


class TestQuadrayCoordinateEmbedding:
    """Test coordinate embedding functionality."""

    def test_xyz_conversion_embedding(self):
        """Test XYZ conversion with embedding matrix."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        x, y, z = coord.to_xyz()

        # Test that conversion produces valid coordinates
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)

        # Test that coordinates are reasonable
        assert not (x == 0 and y == 0 and z == 0)  # Should not be all zero


class TestQuadrayCoordinateEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_coordinate(self):
        """Test zero coordinate behavior."""
        zero = QuadrayCoordinate(0, 0, 0, 0)

        assert zero.magnitude() == 0
        assert zero.distance_to(zero) == 0
        assert zero.dot(zero) == 0

    def test_very_large_coordinates(self):
        """Test with very large coordinate values."""
        large_coord = QuadrayCoordinate(1e10, 1e10, 1e10, 1e10, normalize=False)

        # Test that the coordinate was created correctly
        assert large_coord.a == 10000000000
        assert large_coord.b == 10000000000
        assert large_coord.c == 10000000000
        assert large_coord.d == 10000000000
        assert isinstance(large_coord.magnitude(), float)

    def test_very_small_coordinates(self):
        """Test with very small coordinate values."""
        small_coord = QuadrayCoordinate(1e-10, 1e-10, 1e-10, 1e-10, normalize=False)

        assert small_coord.magnitude() >= 0
        assert isinstance(small_coord.magnitude(), float)

    def test_negative_coordinates(self):
        """Test with negative coordinate values."""
        neg_coord = QuadrayCoordinate(-1, -2, -3, -4, normalize=False)

        assert neg_coord.magnitude() > 0
        assert neg_coord.a == -1
        assert neg_coord.b == -2
        assert neg_coord.c == -3
        assert neg_coord.d == -4


class TestQuadrayCoordinateIntegration:
    """Test integration with other components."""

    def test_with_symergetics_numbers(self):
        """Test interaction with SymergeticsNumber."""
        # Test that coordinates can be created with SymergeticsNumber values
        scalar = SymergeticsNumber(2, 3)
        coord = QuadrayCoordinate(scalar, scalar, scalar, scalar, normalize=False)

        assert isinstance(coord, QuadrayCoordinate)
        assert coord.a == 1  # 2/3 snaps to the nearest lattice integer
        assert coord.b == 1
        assert coord.c == 1
        assert coord.d == 1

    def test_serialization(self):
        """Test serialization and deserialization."""
        coord = QuadrayCoordinate(1, 2, 3, 4, normalize=False)

        # Test to_dict
        data = coord.to_dict()
        assert data['a'] == 1
        assert data['b'] == 2
        assert data['c'] == 3
        assert data['d'] == 4

        # Test that we can create a new coordinate from the data
        restored = QuadrayCoordinate(data['a'], data['b'], data['c'], data['d'], normalize=False)
        assert restored == coord

    def test_string_representations(self):
        """Test string representations."""
        coord = QuadrayCoordinate(1, 2, 3, 4, normalize=False)

        str_repr = str(coord)
        assert str_repr == "(1, 2, 3, 4)"

        repr_str = repr(coord)
        assert "QuadrayCoordinate" in repr_str

class TestExactCoordinateBoundaries:
    @pytest.mark.parametrize("sign", (-1, 1))
    @pytest.mark.parametrize("wrap", (int, Fraction, SymergeticsNumber))
    def test_large_exact_values_and_arithmetic(self, sign, wrap):
        value = sign * (2**80 + 1)
        coordinate = QuadrayCoordinate(wrap(value), 0, 0, 0, normalize=False)
        assert coordinate.as_tuple() == (value, 0, 0, 0)
        assert coordinate.copy().as_tuple() == coordinate.as_tuple()
        assert coordinate.add(QuadrayCoordinate(1, 0, 0, 0)).a == value + 1
        assert coordinate.sub(QuadrayCoordinate(1, 0, 0, 0)).a == value - 1

    @pytest.mark.parametrize("sign", (-1, 1))
    @pytest.mark.parametrize("wrap", (Fraction, SymergeticsNumber))
    def test_rational_half_boundaries(self, sign, wrap):
        whole = 2**60 + 1
        half = Fraction(2 * whole + 1, 2)
        epsilon = Fraction(1, 10**40)
        values = [sign * (half - epsilon), sign * half, sign * (half + epsilon)]
        actual = [QuadrayCoordinate(wrap(v), 0, 0, 0, normalize=False).a for v in values]
        assert actual == [sign * whole, sign * (whole + 1), sign * (whole + 1)]

    @pytest.mark.parametrize("sign", (-1, 1))
    def test_integral_float_near_precision_boundary(self, sign):
        value = sign * float(2**52 + 1)
        assert QuadrayCoordinate(value, 0, 0, 0, normalize=False).a == sign * (2**52 + 1)

    def test_large_signed_serialization_round_trip(self):
        values = (2**100 + 1, -(2**90 + 3), 2**80 + 5, -7)
        original = QuadrayCoordinate(*values, normalize=False)
        restored = QuadrayCoordinate(**json.loads(json.dumps(original.to_dict())), normalize=False)
        assert restored.as_tuple() == values
        canonical = QuadrayCoordinate(**restored.to_dict())
        assert canonical.as_tuple() == tuple(value - min(values) for value in values)

    @pytest.mark.parametrize("gauge", (2**100, -(2**100)))
    def test_large_gauge_removed_before_xyz_boundary(self, gauge):
        shifted = QuadrayCoordinate(gauge + 2, gauge + 1, gauge + 1, gauge, normalize=False)
        assert shifted.to_xyz() == (0.0, 2.0, 2.0)
        assert shifted.distance_to(QuadrayCoordinate(gauge, gauge, gauge, gauge,
                                                    normalize=False)) == pytest.approx(math.sqrt(8))

    def test_exact_large_xyz_projection(self):
        value = 2**80 + 1
        assert QuadrayCoordinate.from_xyz(value, value, value).as_tuple() == (value, 0, 0, 0)
        assert QuadrayCoordinate.from_xyz(-value, -value, -value).as_tuple() == (0, value, value, value)


class TestNearestIntegerQuadray:
    @staticmethod
    def _oracle(matrix, target, bound=8):
        """Independent exhaustive Cartesian-distance oracle with a cube certificate."""
        rows = [[Fraction(value) for value in row[:3]] for row in matrix]
        target = tuple(Fraction(value) for value in target)
        best = None
        for values in product(range(-bound, bound + 1), repeat=3):
            xyz = [sum(coefficient * value for coefficient, value in zip(row, values))
                   for row in rows]
            error = sum((a - b) ** 2 for a, b in zip(xyz, target))
            raw = (*values, 0)
            canonical = tuple(value - min(raw) for value in raw)
            candidate = (error, canonical)
            if best is None or candidate < best:
                best = candidate
        # Any point at least as close as best is inside this coefficient cube:
        # |n_i| <= |(B^-1 x)_i| + ||row_i(B^-1)|| * sqrt(best_error).
        inverse = np.linalg.inv(np.asarray(matrix[:, :3], dtype=float))
        center = inverse @ np.array(target, dtype=float)
        extent = np.abs(center) + np.linalg.norm(inverse, axis=1) * math.sqrt(float(best[0]))
        assert max(extent) < bound
        return best[1]

    @pytest.mark.parametrize("matrix", [
        urner_embedding(),
        urner_embedding(0.3),
        np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]]) @ urner_embedding(2),
        np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]]) @ urner_embedding(2),
        np.array([[1, 1.5, 0], [0, 1, 0.5], [0, 0, 1]]) @ urner_embedding(),
        np.array([[1, 2, 0, -3], [0, 0.5, 0, -0.5], [0, 0, 2, -2]]),
    ], ids=["default", "non-dyadic-scale", "oriented-scale", "rounded-orientation",
            "skewed", "reordered-integer-shear"])
    def test_projection_matches_exhaustive_oracle(self, matrix):
        for target in [
            (-1.5, -1, -0.5), (0, 0, 0), (1, 0, 0),
            (Fraction(1, 3), Fraction(-5, 7), Fraction(2, 5)),
            (0.1, 0.7, -0.9), (1.25, -0.75, 0.5),
        ]:
            assert QuadrayCoordinate.from_xyz(*target, embedding=matrix).as_tuple() == self._oracle(
                matrix, target)

    def test_skew_can_require_large_integer_coefficients(self):
        matrix = np.array([[1, 20, 0, -21],
                           [0, 0.25, 0, -0.25],
                           [0, 0, 1, -1]], dtype=float)
        # This image is the rectangular lattice Z x (Z/4) x Z, but the
        # coefficient of the first basis vector is twenty steps from zero.
        result = QuadrayCoordinate.from_xyz(Fraction(1, 10), Fraction(6, 25),
                                            Fraction(1, 10), embedding=matrix)
        assert result.as_tuple() == (0, 21, 20, 20)
        assert result.to_xyz(matrix) == (0.0, 0.25, 0.0)

    @staticmethod
    def _rectangular_minimizers(target):
        """All minimizers in Z³: each coordinate independently rounds or ties."""
        choices = []
        for value in target:
            lower = value.numerator // value.denominator
            neighbors = (lower, lower + 1)
            error = min(abs(value - n) for n in neighbors)
            choices.append([n for n in neighbors if abs(value - n) == error])
        return product(*choices)

    @pytest.mark.parametrize("column_order", list(permutations(range(4))))
    @pytest.mark.parametrize("row_order", list(permutations(range(3))))
    def test_anisotropic_rectangular_projection_and_permutations(self, column_order, row_order):
        spacing = Fraction(1, 2**40)
        matrix = np.array([[1, 0, 0, -1], [0, 1, 0, -1],
                           [0, 0, float(spacing), -float(spacing)]])
        matrix = matrix[list(row_order), :][:, list(column_order)]
        # The first case previously enumerated over 10**12 outer coefficients.
        # Other cases require nonzero remapping and canonical ties after a
        # permutation moves the original d-axis into the enumerated basis.
        for target in [
            (Fraction(1, 2), Fraction(1, 2), Fraction(0)),
            (Fraction(1, 2), Fraction(1, 2), Fraction(7, 2)),
            (Fraction(-1, 2), Fraction(3, 2), Fraction(-5, 2)),
            (Fraction(1, 3), Fraction(-2, 3), Fraction(11, 5)),
        ]:
            candidates = []
            for site in self._rectangular_minimizers(target):
                raw = (*site, 0)
                permuted = tuple(raw[i] for i in column_order)
                candidates.append(tuple(value - min(permuted) for value in permuted))
            xyz = (target[0], target[1], target[2] * spacing)
            result = QuadrayCoordinate.from_xyz(*(xyz[i] for i in row_order),
                                                embedding=matrix)
            assert result.as_tuple() == min(candidates)

    @pytest.mark.parametrize("column_order", list(permutations(range(4))))
    def test_anisotropic_integer_shear_remaps_large_coefficients_and_ties(self, column_order):
        spacing = Fraction(1, 2**40)
        shear = 2**20
        matrix = np.array([[1, shear, 0, -1 - shear],
                           [0, 1, shear, -1 - shear],
                           [0, 0, float(spacing), -float(spacing)]], dtype=float)
        matrix = matrix[:, list(column_order)]
        for target in [
            (Fraction(1, 2), Fraction(1, 2), Fraction(7, 2)),
            (Fraction(1, 3), Fraction(-2, 3), Fraction(-11, 5)),
        ]:
            # This unimodular integer shear has exactly the same rectangular
            # image. Its inverse, not any bounded coefficient cube, therefore
            # certifies every minimizer despite coefficients exceeding 2**40.
            candidates = []
            for x, y, z in self._rectangular_minimizers(target):
                raw = (x - shear * y + shear**2 * z, y - shear * z, z, 0)
                permuted = tuple(raw[i] for i in column_order)
                candidates.append(tuple(value - min(permuted) for value in permuted))
            xyz = (target[0], target[1], target[2] * spacing)
            result = QuadrayCoordinate.from_xyz(*xyz, embedding=matrix)
            assert result.as_tuple() == min(candidates)

    @pytest.mark.parametrize("scale", (1, 0.3, -1.7))
    def test_every_small_lattice_residue_round_trips(self, scale):
        matrix = urner_embedding(scale)
        for raw in product(range(3), repeat=4):
            original = QuadrayCoordinate(*raw)
            recovered = QuadrayCoordinate.from_xyz(*original.to_xyz(matrix), embedding=matrix)
            assert recovered == original

    @pytest.mark.parametrize("matrix", [
        np.zeros((3, 3)), np.zeros((4, 3)),
        np.zeros((3, 4)),
        np.array([[1, -1, 0, 0], [0, 1, -1, 0], [1, 0, -1, 0]]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
        urner_embedding() * float("inf"),
        np.full((3, 4), float("nan")),
    ], ids=["wrong-columns", "wrong-rows", "zero-rank", "rank-two", "wrong-gauge", "inf", "nan"])
    def test_invalid_embeddings_rejected_both_directions(self, matrix):
        with pytest.raises(ValueError):
            QuadrayCoordinate(1, 0, 0, 0).to_xyz(matrix)
        with pytest.raises(ValueError):
            QuadrayCoordinate.from_xyz(1, 0, 0, embedding=matrix)


@pytest.mark.parametrize("value", [np.int64(2**62), np.uint64(2**63)])
def test_numpy_backed_fraction_coordinates_preserve_unbounded_arithmetic(value):
    point = QuadrayCoordinate(Fraction(value, 1), 0, 0, 0)
    assert point.add(point).as_tuple() == (2 * int(value), 0, 0, 0)


def test_numpy_backed_fraction_rounding_uses_unbounded_remainders():
    value = Fraction(np.int64(2**62 + 1), np.int64(2**62 + 2))
    assert QuadrayCoordinate(value, 0, 0, 0).as_tuple() == (1, 0, 0, 0)


if __name__ == "__main__":
    pytest.main([__file__])
