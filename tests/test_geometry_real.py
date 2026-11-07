"""
Comprehensive real tests for geometry modules.

Tests actual geometry functions with real Symergetics methods.
All tests use real functions and avoid mocks.
"""

import math

import pytest

from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.core.numbers import SymergeticsNumber
from symergetics.geometry.polyhedra import (
    ace_tetra_volume_5x5,
    create_cube,
    create_cuboctahedron,
    create_octahedron,
    create_unit_tetrahedron,
    integer_tetra_volume,
)
from symergetics.geometry.transformations import (
    apply_transform_function,
    coordinate_transform,
    project_to_plane,
    reflect,
    reflect_polyhedron,
    rotate_around_axis,
    scale,
    scale_polyhedron,
    translate,
    translate_polyhedron,
)

def spatial(coord):
    """Independent integer Cartesian oracle for transformation behavior."""
    a, b, c, d = coord.as_tuple()
    return (a - b - c + d, a + b - c - d, a - b + c - d)



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
        from fractions import Fraction

        # Corner tetrahedron spanned by the origin and three basis directions:
        # exactly one quarter of the unit IVM tetrahedron.
        p0 = QuadrayCoordinate(0, 0, 0, 0)
        p1 = QuadrayCoordinate(1, 0, 0, 0)
        p2 = QuadrayCoordinate(0, 1, 0, 0)
        p3 = QuadrayCoordinate(0, 0, 1, 0)

        volume = integer_tetra_volume(p0, p1, p2, p3)
        assert volume == Fraction(1, 4)

        # Unit IVM tetrahedron has volume exactly 1
        unit = integer_tetra_volume(
            QuadrayCoordinate(0, 0, 0, 0),
            QuadrayCoordinate(2, 1, 0, 1),
            QuadrayCoordinate(2, 1, 1, 0),
            QuadrayCoordinate(2, 0, 1, 1),
        )
        assert unit == 1
    
    def test_ace_tetra_volume_5x5(self):
        """Test ACE tetrahedron volume calculation."""
        from fractions import Fraction

        # Create four points for 5x5 matrix calculation
        p0 = QuadrayCoordinate(0, 0, 0, 0)
        p1 = QuadrayCoordinate(1, 0, 0, 0)
        p2 = QuadrayCoordinate(0, 1, 0, 0)
        p3 = QuadrayCoordinate(0, 0, 1, 0)

        volume = ace_tetra_volume_5x5(p0, p1, p2, p3)
        assert volume == Fraction(1, 4)

        # ACE method agrees with the projection determinant method
        assert volume == integer_tetra_volume(p0, p1, p2, p3)


class TestTransformations:
    """Exercise transformations against spatial results, not implementation details."""

    def test_translate_coordinate(self):
        coord = QuadrayCoordinate(1, 2, 3, 4)
        offset = QuadrayCoordinate(1, 1, 1, 1, normalize=False)
        assert spatial(translate(coord, offset)) == spatial(coord)

    def test_scale_coordinate(self):
        coord = QuadrayCoordinate(2, 4, 6, 8)
        assert spatial(scale(coord, 2)) == tuple(2 * value for value in spatial(coord))

    def test_reflect_coordinate(self):
        coord = QuadrayCoordinate(1, 2, 3, 4)
        assert spatial(reflect(coord, 'origin')) == tuple(-value for value in spatial(coord))

    def test_coordinate_transform(self):
        import numpy as np
        coord = QuadrayCoordinate(1, 2, 3, 4)
        assert coordinate_transform(coord, np.eye(4)) == coord

    def test_rotate_around_axis(self):
        coord = QuadrayCoordinate(1, 0, 0, 0)
        assert spatial(rotate_around_axis(coord, 'x', 90.0)) == (1, -1, 1)

    def test_project_to_plane(self):
        coord = QuadrayCoordinate(1, 2, 3, 4)
        assert spatial(project_to_plane(coord, (0, 0, 1))) == (0, -4, 0)

    def test_apply_transform_function(self):
        coords = [
            QuadrayCoordinate(1, 0, 0, 0),
            QuadrayCoordinate(0, 1, 0, 0),
            QuadrayCoordinate(0, 0, 1, 0),
        ]
        result = apply_transform_function(coords, lambda point: scale(point, 2))
        assert [spatial(point) for point in result] == [(2, 2, 2), (-2, 2, -2), (-2, -2, 2)]




class TestPolyhedronTransformations:
    """Exercise transform consumers with exact tetrahedral volumes."""

    def test_translate_polyhedron(self):
        tetra = create_unit_tetrahedron()
        offset = QuadrayCoordinate(1, 0, 0, 0)
        result = translate_polyhedron(tetra, offset)
        assert result.volume() == tetra.volume() == 1
        assert [spatial(point) for point in result.vertices] == [
            tuple(value + 1 for value in spatial(point)) for point in tetra.vertices
        ]

    def test_scale_polyhedron(self):
        tetra = create_unit_tetrahedron()
        result = scale_polyhedron(tetra, 2)
        assert result.volume() == 8
        assert tetra.volume() == 1

    def test_reflect_polyhedron(self):
        tetra = create_unit_tetrahedron()
        result = reflect_polyhedron(tetra, 'origin')
        assert result.volume() == tetra.volume() == 1
        assert [spatial(point) for point in result.vertices] == [
            tuple(-value for value in spatial(point)) for point in tetra.vertices
        ]




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
        """Gauge-shifted origin followed by translation and exact scaling."""
        coord = QuadrayCoordinate(*(2**60,) * 4, normalize=False)
        translated = translate(coord, QuadrayCoordinate(1, 0, 0, 0))
        assert scale(translated, 2).as_tuple() == (2, 0, 0, 0)

    
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
        assert result.a == 3
        assert result.b == 2
        assert result.c == 1
        assert result.d == 0
    
    def test_large_coordinates(self):
        """A large gauge must not hide a small physical displacement."""
        gauge = 2**60
        coord = QuadrayCoordinate(gauge + 1, gauge, gauge, gauge, normalize=False)
        result = scale(coord, 0.75)
        assert result.as_tuple() == (1, 0, 0, 0)

    
    def test_rotation_edge_cases(self):
        """Whole turns preserve points; a negative quarter-turn has known XYZ."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        assert rotate_around_axis(coord, 'x', 0.0) == coord
        assert rotate_around_axis(coord, 'x', 360.0) == coord
        assert spatial(rotate_around_axis(coord, 'x', -90.0)) == (1, 1, -1)
