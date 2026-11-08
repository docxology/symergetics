"""
Tests for geometry.polyhedra module - Polyhedral geometry classes.

Tests volume calculations, vertex operations, and geometric properties.
"""

import math
from fractions import Fraction

import pytest
from symergetics.core.coordinates import QuadrayCoordinate, urner_embedding
from symergetics.geometry.transformations import (
    reflect_polyhedron,
    scale_polyhedron,
    translate_polyhedron,
)

from symergetics.geometry.polyhedra import (
    Cube,
    Cuboctahedron,
    Octahedron,
    SymergeticsPolyhedron,
    Tetrahedron,
    ace_tetra_volume_5x5,
    integer_tetra_volume,
)


def test_exact_centroid_primitive_tetrahedron_and_existing_floor_snap():
    from symergetics.core.coordinates import QuadrayCoordinate

    tetra = Tetrahedron([QuadrayCoordinate(*row) for row in (
        (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))])
    result = tetra.centroid_exact()
    assert result == (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(0))
    assert all(isinstance(value, Fraction) for value in result)
    assert tetra.centroid() == QuadrayCoordinate(0, 0, 0, 0)


def test_primitive_corner_tetrahedron_preserves_fractional_ivm_units():
    tetra = Tetrahedron([QuadrayCoordinate(*row) for row in (
        (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))])
    analysis = tetra.analyze_geometry()
    assert analysis["volume_ivm"] == Fraction(1, 4)
    assert analysis["volume_xyz"] == Fraction(2, 3)
    assert analysis["surface_area"] == pytest.approx(3 * math.sqrt(2) + 2 * math.sqrt(3))
    assert tetra.volume_centroid_exact() == tetra.centroid_exact()


@pytest.mark.parametrize("scale", (1, 10**400 + 1))
def test_exact_centroid_huge_inputs_and_independent_gauges(scale):
    from symergetics.core.coordinates import QuadrayCoordinate

    rows = [(scale, 0, 0, 0), (0, scale, scale, scale)]
    gauges = [-(10**450), 10**500]
    for normalize in (False, True):
        poly = SymergeticsPolyhedron([
            QuadrayCoordinate(*(value + gauge for value in row), normalize=normalize)
            for row, gauge in zip(rows, gauges)])
        assert poly.centroid_exact() == (Fraction(0),) * 4
    poly = SymergeticsPolyhedron([
        QuadrayCoordinate(scale, 0, 0, 0), QuadrayCoordinate(0, 0, 0, 0)])
    assert poly.centroid_exact() == (
        Fraction(scale, 2), Fraction(0), Fraction(0), Fraction(0))


def test_exact_centroid_empty_is_undefined_but_snapped_origin_is_preserved():
    from symergetics.core.coordinates import QuadrayCoordinate

    poly = SymergeticsPolyhedron([])
    with pytest.raises(ValueError, match="At least one"):
        poly.centroid_exact()
    assert poly.centroid() == QuadrayCoordinate(0, 0, 0, 0)


def test_tetrahedron_volume_tracks_replaced_and_mutated_vertices():
    from symergetics.core.coordinates import QuadrayCoordinate

    tetra = Tetrahedron([QuadrayCoordinate(*row) for row in (
        (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))])
    assert tetra.volume() == Fraction(1, 4)
    tetra.vertices[1] = QuadrayCoordinate(2, 0, 0, 0)
    assert tetra.volume() == Fraction(1, 2)
    tetra.vertices[1].a = 3
    assert tetra.volume() == Fraction(3, 4)


class TestVolumeCalculations:
    """Test volume calculation functions."""

    def test_integer_tetra_volume_unit(self):
        """Test unit tetrahedron volume."""
        # Standard IVM tetrahedron
        p0 = (0, 0, 0, 0)
        p1 = (2, 1, 0, 1)
        p2 = (2, 1, 1, 0)
        p3 = (2, 0, 1, 1)

        # Convert to QuadrayCoordinates
        from symergetics.core.coordinates import QuadrayCoordinate
        points = [QuadrayCoordinate(*p) for p in [p0, p1, p2, p3]]

        volume = integer_tetra_volume(*points)
        assert volume == 1  # Unit tetrahedron

    def test_integer_tetra_volume_zero(self):
        """Test degenerate tetrahedron (zero volume)."""
        from symergetics.core.coordinates import QuadrayCoordinate

        # All points at same location
        p0 = QuadrayCoordinate(0, 0, 0, 0)
        p1 = QuadrayCoordinate(0, 0, 0, 0)
        p2 = QuadrayCoordinate(0, 0, 0, 0)
        p3 = QuadrayCoordinate(0, 0, 0, 0)

        volume = integer_tetra_volume(p0, p1, p2, p3)
        assert volume == 0

    def test_ace_tetra_volume_5x5(self):
        """Test Ace 5x5 determinant volume calculation."""
        from symergetics.core.coordinates import QuadrayCoordinate

        # Unit tetrahedron
        p0 = QuadrayCoordinate(0, 0, 0, 0)
        p1 = QuadrayCoordinate(2, 1, 0, 1)
        p2 = QuadrayCoordinate(2, 1, 1, 0)
        p3 = QuadrayCoordinate(2, 0, 1, 1)

        volume = ace_tetra_volume_5x5(p0, p1, p2, p3)
        assert volume == 1


@pytest.mark.parametrize("normalize", (False, True))
def test_large_translated_gauge_shifted_tetrahedron_volume_agreement(normalize):
    """Constructor and subtraction retain the exact det/4 volume contract."""
    from symergetics.core.coordinates import QuadrayCoordinate

    p, q, r = 2**60 + 1, -(2**57 + 3), 5
    translation = (2**100 + 17, -(2**95) + 1, 2**80 + 11, -(2**90) + 9)
    edges = ((0, 0, 0, 0), (p, 0, 0, 0), (0, q, 0, 0), (0, 0, r, 0))
    gauges = (2**110, -(2**109), 2**108 + 1, -(2**107) + 3)
    vertices = [
        QuadrayCoordinate(*(a + b + gauge for a, b in zip(translation, edge)),
                          normalize=normalize)
        for edge, gauge in zip(edges, gauges)
    ]
    expected = Fraction(abs(p * q * r), 4)
    assert integer_tetra_volume(*vertices) == expected
    assert ace_tetra_volume_5x5(*vertices) == expected


class TestSymergeticsPolyhedron:
    """Test base SymergeticsPolyhedron class."""

    def test_base_class_abstract(self):
        """Test that base class requires implementation."""
        from symergetics.core.coordinates import QuadrayCoordinate

        # Should not be able to instantiate base class directly
        with pytest.raises(NotImplementedError):
            poly = SymergeticsPolyhedron([])
            poly.volume()

    def test_centroid_calculation(self):
        """Test centroid calculation."""
        from symergetics.core.coordinates import QuadrayCoordinate

        vertices = [
            QuadrayCoordinate(0, 0, 0, 0),
            QuadrayCoordinate(2, 0, 0, 0),
            QuadrayCoordinate(0, 2, 0, 0),
            QuadrayCoordinate(0, 0, 2, 0),
        ]

        poly = Tetrahedron(vertices)
        centroid = poly.centroid()

        # Centroid should be at (0.5, 0.5, 0.5, 0) after normalization
        assert centroid.a == 0  # Normalized
        assert centroid.b == 0
        assert centroid.c == 0
        assert centroid.d == 0

    def test_xyz_conversion(self):
        """Test conversion to XYZ coordinates."""
        from symergetics.core.coordinates import QuadrayCoordinate

        vertices = [
            QuadrayCoordinate(0, 0, 0, 0),
            QuadrayCoordinate(2, 1, 0, 1),
        ]

        poly = SymergeticsPolyhedron(vertices)
        xyz_coords = poly.to_xyz_vertices()

        assert len(xyz_coords) == 2
        assert all(len(coord) == 3 for coord in xyz_coords)
        assert all(isinstance(coord[i], (int, float)) for coord in xyz_coords for i in range(3))


class TestTetrahedron:
    """Test Tetrahedron class."""

    def test_tetrahedron_creation(self):
        """Test tetrahedron creation and properties."""
        tetra = Tetrahedron()

        assert tetra.num_vertices == 4
        assert tetra.num_edges == 6
        assert tetra.num_faces == 4

    def test_tetrahedron_volume(self):
        """Test tetrahedron volume calculation."""
        tetra = Tetrahedron()
        volume = tetra.volume()

        assert volume == 1  # Unit tetrahedron in IVM units

    def test_tetrahedron_edges(self):
        """Test tetrahedron edge calculation."""
        tetra = Tetrahedron()
        edges = tetra.edges()

        assert len(edges) == 6  # Tetrahedron has 6 edges

        # Each edge should be a pair of vertices
        for edge in edges:
            assert len(edge) == 2
            assert all(isinstance(v, type(tetra.vertices[0])) for v in edge)

    def test_tetrahedron_faces(self):
        """Test tetrahedron face calculation."""
        tetra = Tetrahedron()
        faces = tetra.faces()

        assert len(faces) == 4  # Tetrahedron has 4 triangular faces

        # Each face should be a triangle
        for face in faces:
            assert len(face) == 3
            assert all(isinstance(v, type(tetra.vertices[0])) for v in face)

    def test_tetrahedron_custom_vertices(self):
        """Test tetrahedron with custom vertices."""
        from symergetics.core.coordinates import QuadrayCoordinate

        custom_vertices = [
            QuadrayCoordinate(0, 0, 0, 0),
            QuadrayCoordinate(1, 0, 0, 0),
            QuadrayCoordinate(0, 1, 0, 0),
            QuadrayCoordinate(0, 0, 1, 0),
        ]

        tetra = Tetrahedron(custom_vertices)
        assert tetra.num_vertices == 4

        # Volume might be different for non-standard tetrahedron
        volume = tetra.volume()
        assert volume >= 0


class TestOctahedron:
    """Test Octahedron class."""

    def test_octahedron_creation(self):
        """Test octahedron creation and properties."""
        octa = Octahedron()

        assert octa.num_vertices == 6
        assert octa.num_edges == 12
        assert octa.num_faces == 8

    def test_octahedron_volume(self):
        """Test octahedron volume calculation."""
        octa = Octahedron()
        volume = octa.volume()

        assert volume == 4


class TestCube:
    """Test Cube class."""

    def test_cube_creation(self):
        """Test cube creation and properties."""
        cube = Cube()

        assert cube.num_vertices == 8
        assert cube.num_edges == 12
        assert cube.num_faces == 6

    def test_cube_volume(self):
        """Test cube volume calculation."""
        cube = Cube()
        volume = cube.volume()

        # Cube should have volume 3 (3 tetrahedra)
        assert volume == 3


class TestCuboctahedron:
    """Test Cuboctahedron class."""

    def test_cuboctahedron_creation(self):
        """Test cuboctahedron creation and properties."""
        cubocta = Cuboctahedron()

        assert cubocta.num_vertices == 12
        assert cubocta.num_edges == 24
        assert cubocta.num_faces == 14  # 8 triangles + 6 squares

    def test_cuboctahedron_volume(self):
        """Test cuboctahedron volume calculation."""
        cubocta = Cuboctahedron()
        volume = cubocta.volume()

        # Cuboctahedron should have volume 20
        assert volume == 20


class TestVolumeRatios:
    """Test volume ratios between polyhedra."""

    def test_relative_volumes(self):
        """Test that volumes are in correct ratios."""
        tetra = Tetrahedron()
        octa = Octahedron()
        cube = Cube()
        cubocta = Cuboctahedron()

        tetra_vol = tetra.volume()
        octa_vol = octa.volume()
        cube_vol = cube.volume()
        cubocta_vol = cubocta.volume()

        # Check ratios
        assert octa_vol == pytest.approx(4 * tetra_vol, abs=0.1)
        assert cube_vol == pytest.approx(3 * tetra_vol, abs=0.1)
        assert cubocta_vol == pytest.approx(20 * tetra_vol, abs=0.1)


@pytest.mark.parametrize("kind,area,volume,triangle_count", [
    (Tetrahedron, 8 * math.sqrt(3), 1, 4),
    (Cube, 24, 3, 12),
    (Octahedron, 16 * math.sqrt(3), 4, 8),
    (Cuboctahedron, 48 + 16 * math.sqrt(3), 20, 20),
])
def test_live_mesh_canonical_metric_oracles(kind, area, volume, triangle_count):
    poly = kind()
    analysis = poly.analyze_geometry()
    assert len(poly.triangulated_faces()) == triangle_count
    assert poly.surface_area() == pytest.approx(area)
    assert analysis["surface_area"] == pytest.approx(area)
    assert analysis["volume_ivm"] == volume
    assert analysis["volume_xyz"] == Fraction(8 * volume, 3)
    assert analysis["affine_rank"] == 3
    assert analysis["physical_vertex_count"] == poly.num_vertices
    assert analysis["closed"] is True
    assert analysis["nonplanar_faces"] == ()
    assert analysis["is_convex"] is True
    assert analysis["volume_centroid_exact"] == poly.centroid_exact()
    for factor in (3, -3):
        transformed = scale_polyhedron(poly, factor)
        assert transformed.surface_area() == pytest.approx(area * 9)
        assert transformed.volume() == volume * 27
        assert poly.surface_area(urner_embedding(factor)) == pytest.approx(area * 9)


def test_concave_octahedron_signed_volume_and_solid_centroid():
    # A height-four square pyramid minus a height-two pyramid with the same
    # base: Cartesian volume 64/3, centroid (0,-3/2,0).
    poly = Octahedron([QuadrayCoordinate(*row) for row in (
        (2, 0, 0, 2), (0, 2, 2, 0), (0, 0, 1, 1),
        (0, 0, 2, 2), (2, 0, 2, 0), (0, 2, 0, 2))])
    analysis = poly.analyze_geometry()
    assert poly.volume() == 8
    assert analysis["volume_xyz"] == Fraction(64, 3)
    assert analysis["is_convex"] is False
    assert analysis["closed"] is True
    assert analysis["nonplanar_faces"] == ()
    assert poly.volume_centroid_exact() == (
        Fraction(0), Fraction(0), Fraction(3, 4), Fraction(3, 4))
    assert analysis["vertex_centroid_exact"] != analysis["volume_centroid_exact"]
    inverted = scale_polyhedron(poly, -1)
    assert inverted.volume() == 8
    assert inverted.volume_centroid_exact() == (
        Fraction(3, 4), Fraction(3, 4), Fraction(0), Fraction(0))


def test_asymmetric_bipyramid_distinguishes_vertex_mean_and_mass_centroid():
    poly = Octahedron([QuadrayCoordinate.from_xyz(*xyz) for xyz in (
        (4, 0, 0), (-2, 0, 0), (0, 2, 0),
        (0, -2, 0), (0, 0, 2), (0, 0, -2))])
    assert poly.volume() == 6
    assert poly.centroid_exact() == (
        Fraction(1, 6), Fraction(0), Fraction(0), Fraction(1, 6))
    assert poly.volume_centroid_exact() == (
        Fraction(1, 4), Fraction(0), Fraction(0), Fraction(1, 4))


@pytest.mark.parametrize("kind", [Tetrahedron, Cube, Octahedron, Cuboctahedron])
def test_measurements_ignore_independent_gauges_and_huge_translations(kind):
    original = kind()
    translation = QuadrayCoordinate(2**200 + 3, -2**180, 7, 0, normalize=False)
    translated = translate_polyhedron(original, translation)
    shifted = kind([
        QuadrayCoordinate(*(value + (-1)**i * 10**300 for value in vertex.as_tuple()),
                          normalize=False)
        for i, vertex in enumerate(translated.vertices)
    ])
    assert shifted.volume() == original.volume()
    assert shifted.surface_area() == original.surface_area()
    assert shifted.to_xyz_vertices(relative_to=shifted.vertices[0]) == (
        original.to_xyz_vertices(relative_to=original.vertices[0]))
    assert shifted.volume_centroid_exact() == translated.volume_centroid_exact()
    # Test exact translation covariance in the quotient, without embedding.
    before = original.volume_centroid_exact()
    after = shifted.volume_centroid_exact()
    for i in range(3):
        assert after[i] - after[3] == (
            before[i] - before[3] + translation.as_tuple()[i] - translation.d)
    assert shifted.analyze_geometry()["is_convex"] is True


def test_floor_centroid_removes_each_independent_gauge_before_snapping():
    rows = ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))
    poly = Tetrahedron([QuadrayCoordinate(*row) for row in rows])
    expected = poly.centroid()
    poly.vertices = [
        QuadrayCoordinate(*(value + gauge for value in row), normalize=False)
        for row, gauge in zip(rows, (3, -10**400, 10**500, -9))
    ]
    assert poly.centroid() == expected == QuadrayCoordinate(0, 0, 0, 0)
    assert poly.centroid_exact() == (Fraction(1, 4),) * 3 + (Fraction(0),)


def test_indexed_geometry_remains_live_when_vertices_coincide_or_mutate():
    poly = Cube()
    faces = poly.face_indices()
    triangles = poly.triangulated_faces()
    poly.vertices[5] = poly.vertices[0]
    assert poly.face_indices() == faces
    assert poly.triangulated_faces() == triangles
    assert poly.faces()[0][1] is poly.vertices[5]
    assert len(poly.edges()) == 12
    assert poly.analyze_geometry()["physical_vertex_count"] == 7

    tetra = Tetrahedron([QuadrayCoordinate.from_xyz(*xyz) for xyz in (
        (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2))])
    assert tetra.surface_area() == pytest.approx(6 + 2 * math.sqrt(3))
    assert tetra.volume() == Fraction(1, 2)
    tetra.vertices[1] = QuadrayCoordinate.from_xyz(4, 0, 0)
    assert tetra.surface_area() == 16
    assert tetra.volume() == 1
    assert tetra.volume_centroid_exact() == tetra.centroid_exact()
    tetra.vertices[1].a += 1
    assert tetra.surface_area() != 16
    assert tetra.volume() != 1


@pytest.mark.parametrize("xyz,rank,count", [
    (((0, 0, 0),) * 4, 0, 1),
    (((0, 0, 0), (2, 0, 0), (4, 0, 0), (6, 0, 0)), 1, 4),
    (((0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0)), 2, 4),
])
def test_exact_rank_zero_volume_and_declared_closed_topology(xyz, rank, count):
    poly = Tetrahedron([QuadrayCoordinate.from_xyz(*point) for point in xyz])
    data = poly.analyze_geometry()
    assert data["affine_rank"] == rank
    assert data["physical_vertex_count"] == count
    assert data["closed"] is True
    assert data["is_convex"] is False
    assert data["volume_ivm"] == data["volume_xyz"] == 0
    assert data["volume_centroid_exact"] is None
    assert len(poly.edges()) == 6
    with pytest.raises(ValueError, match="zero volume"):
        poly.volume_centroid_exact()


def test_convex_warped_cube_uses_supporting_fan_triangles_not_face_planarity():
    poly = Cube()
    poly.vertices[0] = QuadrayCoordinate(2, 0, 0, 0)
    data = poly.analyze_geometry()
    assert data["nonplanar_faces"] == (0, 2, 4)
    assert data["is_convex"] is True
    assert data["closed"] is True
    assert data["volume_ivm"] == Fraction(9, 2)
    # The supporting-triangle diagnostic still distinguishes a genuine dent.
    poly.vertices[0] = QuadrayCoordinate(0, 0, 0, 0)
    assert poly.analyze_geometry()["is_convex"] is False


class _ShellMesh(SymergeticsPolyhedron):
    """Compose real indexed shells, optionally reversing individual windings."""

    def __init__(self, shells, reverse=()):
        vertices = []
        faces = []
        for index, shell in enumerate(shells):
            offset = len(vertices)
            for face in shell.face_indices():
                cycle = tuple(offset + vertex for vertex in face)
                faces.append(cycle[::-1] if index in reverse else cycle)
            vertices.extend(vertex.copy() for vertex in shell.vertices)
        super().__init__(vertices)
        self._faces = tuple(faces)

    def face_indices(self):
        return self._faces


@pytest.mark.parametrize("reverse", ((), (0,), (1,), (0, 1)))
def test_disjoint_reflected_tetrahedra_add_positive_exact_mass(reverse):
    original = Tetrahedron()
    reflected = translate_polyhedron(
        scale_polyhedron(original, -1), QuadrayCoordinate(10, 0, 0, 10))
    poly = _ShellMesh((original, reflected), reverse)
    data = poly.analyze_geometry()
    assert data["closed"] is True
    assert data["is_convex"] is False
    assert data["volume_ivm"] == 2
    assert data["volume_xyz"] == Fraction(16, 3)
    assert data["volume_centroid_exact"] == (
        Fraction(5), Fraction(0), Fraction(0), Fraction(5))
    assert data["surface_area"] == pytest.approx(16 * math.sqrt(3))
    # Preserve exact placement through independently gauged huge translations.
    shift = (2**200 + 3, -2**190, 7, 0)
    before = poly.volume_centroid_exact()
    poly.vertices = [
        QuadrayCoordinate(*(value + delta + (-1)**i * 10**300
                            for value, delta in zip(vertex.as_tuple(), shift)),
                          normalize=False)
        for i, vertex in enumerate(poly.vertices)
    ]
    assert poly.volume() == 2
    after = poly.volume_centroid_exact()
    for axis in range(3):
        assert after[axis] - after[3] == before[axis] - before[3] + shift[axis]


@pytest.mark.parametrize("reverse", ((), (0,), (1,), (0, 1)))
def test_nested_cavity_subtracts_mass_and_centroid_independently_of_winding(reverse):
    outer = scale_polyhedron(Cube(), 5)
    cavity = translate_polyhedron(
        scale_polyhedron(Cube(), 2), QuadrayCoordinate(1, 0, 0, 1))
    # Start with the cavity so enclosure is not inferred from component order.
    poly = _ShellMesh((cavity, outer), reverse)
    assert poly.volume() == 351  # 3 * (5**3 - 2**3)
    assert poly.volume_centroid_exact() == (
        Fraction(0), Fraction(8, 117), Fraction(8, 117), Fraction(0))
    data = poly.analyze_geometry()
    assert data["is_convex"] is False
    assert data["surface_area"] == 24 * (25 + 4)
    # Public in-place vertex mutations must change shell moments immediately.
    for vertex in poly.vertices[:8]:
        vertex.a -= 1
        vertex.d -= 1
    assert poly.volume() == 351
    assert poly.volume_centroid_exact() == (Fraction(0),) * 4
    # Floating containment would lose these concentric shells' separation.
    for vertex in poly.vertices:
        vertex.a += 2**200
        vertex.b -= 2**190
    assert poly.volume() == 351
    centroid = poly.volume_centroid_exact()
    assert tuple(centroid[i] - centroid[3] for i in range(3)) == (
        Fraction(2**200), Fraction(-2**190), Fraction(0))


def test_three_nested_shells_restore_island_mass_after_cavity():
    outer = scale_polyhedron(Cube(), 5)
    translation = QuadrayCoordinate(1, 0, 0, 1)
    cavity = translate_polyhedron(scale_polyhedron(Cube(), 2), translation)
    island = translate_polyhedron(Cube(), translation)
    poly = _ShellMesh((island, outer, cavity), reverse=(0, 2))
    assert poly.volume() == 354  # 375 - 24 + 3
    assert poly.volume_centroid_exact() == (
        Fraction(0), Fraction(7, 118), Fraction(7, 118), Fraction(0))
    assert poly.surface_area() == 24 * (25 + 4 + 1)


def test_warped_face_diagnostics_preserve_declared_fan_interpretation():
    poly = reflect_polyhedron(Cube(), "a")
    data = poly.analyze_geometry()
    assert data["nonplanar_faces"] == tuple(range(6))
    assert data["is_convex"] is False
    assert data["closed"] is True
    assert data["surface_area"] == pytest.approx(12 * math.sqrt(6))
    assert data["volume_ivm"] == 3
    assert len(poly.triangulated_faces()) == 12


def test_surface_area_custom_metric_and_range_safe_boundary():
    import numpy as np

    poly = Cube()
    metric = np.diag((2, 3, 4)) @ urner_embedding()
    # Original side length two -> side lengths four, six, eight.
    assert poly.surface_area(metric) == 208
    huge = scale_polyhedron(poly, 10**100)
    assert huge.surface_area() == pytest.approx(24e200)
    # A tiny embedding balances arbitrarily large integer edges before float.
    enormous = scale_polyhedron(poly, 10**400)
    matrix = urner_embedding(1e-200)
    assert enormous.surface_area(matrix) == math.inf
    balanced = scale_polyhedron(poly, 10**200)
    assert balanced.surface_area(matrix) == pytest.approx(24)
    # Each individual triangle area rounds to zero, but the sum is subnormal.
    tiny_scale = 1e-162
    expected = float(24 * Fraction(tiny_scale)**2)
    assert expected > 0
    assert poly.surface_area(urner_embedding(tiny_scale)) == expected
    with pytest.raises(ValueError, match="rank three"):
        poly.surface_area(np.zeros((3, 4)))


if __name__ == "__main__":
    pytest.main([__file__])
