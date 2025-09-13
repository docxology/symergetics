"""
Comprehensive real tests for geometry modules.

Tests actual geometry functions with real Symergetics methods.
All tests use real functions and avoid mocks.
"""

import pytest
import math
from symergetics.core.numbers import SymergeticsNumber
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.geometry.polyhedra import (
    integer_tetra_volume,
    ace_tetra_volume_5x5,
    create_unit_tetrahedron,
    create_octahedron,
    create_cube,
    create_cuboctahedron
)
from symergetics.geometry.transformations import (
    translate,
    translate_polyhedron,
    scale,
    scale_polyhedron,
    reflect,
    reflect_polyhedron,
    coordinate_transform,
    rotate_around_axis,
    project_to_plane,
    apply_transform_function
)


class TestPolyhedraFunctions:
    """Test polyhedra creation and volume functions."""
    
    def test_create_unit_tetrahedron(self):
        """Test unit tetrahedron creation."""
        tetra = create_unit_tetrahedron()
        assert tetra is not None
        assert hasattr(tetra, 'vertices')
        assert hasattr(tetra, 'faces')
        assert hasattr(tetra, 'num_edges')
    
    def test_create_octahedron(self):
        """Test octahedron creation."""
        octa = create_octahedron()
        assert octa is not None
        assert hasattr(octa, 'vertices')
        assert hasattr(octa, 'num_edges')
        assert hasattr(octa, 'num_faces')
    
    def test_create_cube(self):
        """Test cube creation."""
        cube = create_cube()
        assert cube is not None
        assert hasattr(cube, 'vertices')
        assert hasattr(cube, 'num_edges')
        assert hasattr(cube, 'num_faces')
    
    def test_create_cuboctahedron(self):
        """Test cuboctahedron creation."""
        cuboct = create_cuboctahedron()
        assert cuboct is not None
        assert hasattr(cuboct, 'vertices')
        assert hasattr(cuboct, 'num_edges')
        assert hasattr(cuboct, 'num_faces')
    
    def test_integer_tetra_volume(self):
        """Test integer tetrahedron volume calculation."""
        # Create four points forming a tetrahedron
        p0 = QuadrayCoordinate(0, 0, 0, 0)
        p1 = QuadrayCoordinate(1, 0, 0, 0)
        p2 = QuadrayCoordinate(0, 1, 0, 0)
        p3 = QuadrayCoordinate(0, 0, 1, 0)
        
        volume = integer_tetra_volume(p0, p1, p2, p3)
        assert isinstance(volume, int)
        assert volume > 0
    
    def test_ace_tetra_volume_5x5(self):
        """Test ACE tetrahedron volume calculation."""
        # Create four points for 5x5 matrix calculation
        p0 = QuadrayCoordinate(0, 0, 0, 0)
        p1 = QuadrayCoordinate(1, 0, 0, 0)
        p2 = QuadrayCoordinate(0, 1, 0, 0)
        p3 = QuadrayCoordinate(0, 0, 1, 0)
        
        volume = ace_tetra_volume_5x5(p0, p1, p2, p3)
        assert isinstance(volume, int)
        assert volume >= 0


class TestTransformations:
    """Test geometric transformation functions."""
    
    def test_translate_coordinate(self):
        """Test coordinate translation."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        offset = QuadrayCoordinate(1, 1, 1, 1)
        
        result = translate(coord, offset)
        assert isinstance(result, QuadrayCoordinate)
        assert result.a == coord.a + offset.a
        assert result.b == coord.b + offset.b
        assert result.c == coord.c + offset.c
        assert result.d == coord.d + offset.d
    
    def test_scale_coordinate(self):
        """Test coordinate scaling."""
        coord = QuadrayCoordinate(2, 4, 6, 8)
        factor = 2
        
        result = scale(coord, factor)
        assert isinstance(result, QuadrayCoordinate)
        assert result.a == coord.a * factor
        assert result.b == coord.b * factor
        assert result.c == coord.c * factor
        assert result.d == coord.d * factor
    
    def test_reflect_coordinate(self):
        """Test coordinate reflection."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        
        result = reflect(coord, 'origin')
        assert isinstance(result, QuadrayCoordinate)
        # Reflection should change the coordinate values
    
    def test_coordinate_transform(self):
        """Test coordinate transformation."""
        import numpy as np
        coord = QuadrayCoordinate(1, 2, 3, 4)
        
        # Create identity matrix
        identity_matrix = np.eye(4)
        result = coordinate_transform(coord, identity_matrix)
        assert isinstance(result, QuadrayCoordinate)
    
    def test_rotate_around_axis(self):
        """Test rotation around axis."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        
        result = rotate_around_axis(coord, 'x', 90.0)
        assert isinstance(result, QuadrayCoordinate)
    
    def test_project_to_plane(self):
        """Test projection to plane."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        plane_normal = (0, 0, 1)  # XY plane
        
        result = project_to_plane(coord, plane_normal)
        assert isinstance(result, QuadrayCoordinate)
    
    def test_apply_transform_function(self):
        """Test applying transform function to multiple coordinates."""
        coords = [
            QuadrayCoordinate(1, 0, 0, 0),
            QuadrayCoordinate(0, 1, 0, 0),
            QuadrayCoordinate(0, 0, 1, 0)
        ]
        
        def double_transform(coord):
            return scale(coord, 2)
        
        result = apply_transform_function(coords, double_transform)
        assert isinstance(result, list)
        assert len(result) == len(coords)
        assert all(isinstance(c, QuadrayCoordinate) for c in result)


class TestPolyhedronTransformations:
    """Test polyhedron transformation functions."""
    
    def test_translate_polyhedron(self):
        """Test polyhedron translation."""
        tetra = create_unit_tetrahedron()
        offset = QuadrayCoordinate(1, 1, 1, 1)
        
        result = translate_polyhedron(tetra, offset)
        assert result is not None
        assert hasattr(result, 'vertices')
    
    def test_scale_polyhedron(self):
        """Test polyhedron scaling."""
        tetra = create_unit_tetrahedron()
        factor = 2
        
        result = scale_polyhedron(tetra, factor)
        assert result is not None
        assert hasattr(result, 'vertices')
    
    def test_reflect_polyhedron(self):
        """Test polyhedron reflection."""
        tetra = create_unit_tetrahedron()
        
        result = reflect_polyhedron(tetra, 'origin')
        assert result is not None
        assert hasattr(result, 'vertices')


class TestGeometryIntegration:
    """Test integration between geometry modules."""
    
    def test_volume_calculation_consistency(self):
        """Test volume calculation consistency."""
        # Create a simple tetrahedron
        p0 = QuadrayCoordinate(0, 0, 0, 0)
        p1 = QuadrayCoordinate(1, 0, 0, 0)
        p2 = QuadrayCoordinate(0, 1, 0, 0)
        p3 = QuadrayCoordinate(0, 0, 1, 0)
        
        # Both methods should give consistent results
        vol1 = integer_tetra_volume(p0, p1, p2, p3)
        vol2 = ace_tetra_volume_5x5(p0, p1, p2, p3)
        
        # Results should be reasonable (both positive or both zero)
        assert (vol1 >= 0 and vol2 >= 0) or (vol1 < 0 and vol2 < 0)
    
    def test_transform_chain(self):
        """Test chaining multiple transformations."""
        coord = QuadrayCoordinate(1, 1, 1, 1)
        
        # Apply translation then scaling
        translated = translate(coord, QuadrayCoordinate(1, 0, 0, 0))
        scaled = scale(translated, 2)
        
        assert isinstance(scaled, QuadrayCoordinate)
        # The add method doesn't normalize, so (1,1,1,1) + (1,0,0,0) = (2,1,1,1)
        # Then scale by 2 gives (4,2,2,2), but normalization subtracts min(4,2,2,2)=2
        # So final result is (2,0,0,0)
        assert scaled.a == 2
        assert scaled.b == 0
        assert scaled.c == 0
        assert scaled.d == 0
    
    def test_polyhedron_properties(self):
        """Test polyhedron property consistency."""
        tetra = create_unit_tetrahedron()
        
        # Check that vertices are QuadrayCoordinates
        assert all(isinstance(v, QuadrayCoordinate) for v in tetra.vertices)
        
        # Check that faces are lists of indices (only tetrahedron has faces method)
        if hasattr(tetra, 'faces') and callable(tetra.faces):
            faces = tetra.faces()
            assert all(isinstance(face, (list, tuple)) for face in faces)
        
        # Check that num_edges is an integer
        assert isinstance(tetra.num_edges, int)


class TestGeometryEdgeCases:
    """Test edge cases for geometry functions."""
    
    def test_zero_volume_tetrahedron(self):
        """Test tetrahedron with zero volume (coplanar points)."""
        # Four coplanar points should have zero volume
        p0 = QuadrayCoordinate(0, 0, 0, 0)
        p1 = QuadrayCoordinate(1, 0, 0, 0)
        p2 = QuadrayCoordinate(0, 1, 0, 0)
        p3 = QuadrayCoordinate(1, 1, 0, 0)  # Coplanar with others
        
        volume = integer_tetra_volume(p0, p1, p2, p3)
        assert volume == 0
    
    def test_negative_scaling(self):
        """Test negative scaling factor."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        
        result = scale(coord, -1)
        assert isinstance(result, QuadrayCoordinate)
        # After scaling by -1: (-1, -2, -3, -4)
        # After normalization (subtract min=-4): (3, 2, 1, 0)
        assert result.a == 3
        assert result.b == 2
        assert result.c == 1
        assert result.d == 0
    
    def test_large_coordinates(self):
        """Test with large coordinate values."""
        large_coord = QuadrayCoordinate(10**6, 10**6, 10**6, 10**6)
        
        # Test translation
        offset = QuadrayCoordinate(1, 1, 1, 1)
        result = translate(large_coord, offset)
        assert isinstance(result, QuadrayCoordinate)
        
        # Test scaling
        result = scale(large_coord, 2)
        assert isinstance(result, QuadrayCoordinate)
    
    def test_rotation_edge_cases(self):
        """Test rotation with edge case angles."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        
        # Test 0 degree rotation
        result0 = rotate_around_axis(coord, 'x', 0.0)
        assert isinstance(result0, QuadrayCoordinate)
        
        # Test 360 degree rotation
        result360 = rotate_around_axis(coord, 'x', 360.0)
        assert isinstance(result360, QuadrayCoordinate)
        
        # Test negative angle
        result_neg = rotate_around_axis(coord, 'x', -90.0)
        assert isinstance(result_neg, QuadrayCoordinate)
