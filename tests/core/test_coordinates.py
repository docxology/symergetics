"""
Comprehensive tests for QuadrayCoordinate and coordinate system functionality.
"""

import pytest
import numpy as np
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.core.numbers import SymergeticsNumber


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
        
        # Test with floats
        coord2 = QuadrayCoordinate(1.5, 2.5, 3.5, 4.5, normalize=False)
        assert coord2.a == 1  # Converted to int
        assert coord2.b == 2
        assert coord2.c == 3
        assert coord2.d == 4
    
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


class TestPredefinedCoordinates:
    """Test predefined coordinate values."""
    
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
        assert coord.a == 0  # 2/3 converted to int
        assert coord.b == 0
        assert coord.c == 0
        assert coord.d == 0
    
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


if __name__ == "__main__":
    pytest.main([__file__])
