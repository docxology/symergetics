"""Behavioral matrix, projection and polyhedron transformation regressions."""

from itertools import combinations

import numpy as np
import pytest

from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.geometry.polyhedra import Tetrahedron
from symergetics.geometry.transformations import (
    compose_transforms,
    create_rotation_matrix,
    project_to_plane,
    reflect_polyhedron,
    rotate_around_axis,
    rotate_around_xyz_axis,
    scale_polyhedron,
    translate_polyhedron,
    xyz_to_quadray_matrix,
)


# Independent Cartesian definition, not the production embedding factory.
EMBEDDING = np.array([[1, -1, -1, 1], [1, 1, -1, -1], [1, -1, 1, -1]], dtype=float)


def spatial(coord):
    a, b, c, d = coord.as_tuple()
    return np.array((a - b - c + d, a + b - c - d, a - b + c - d))


def squared_edges(polyhedron):
    points = [spatial(vertex) for vertex in polyhedron.vertices]
    return [np.dot(a - b, a - b) for a, b in combinations(points, 2)]


class TestInverseEmbedding:
    @pytest.mark.parametrize("scale", [0.125, 1, 2, -3, 1e-20, 1e20])
    def test_inverse_restores_cartesian_coordinates_and_zero_sum_gauge(self, scale):
        embedding = scale * EMBEDDING
        inverse = xyz_to_quadray_matrix(embedding)
        assert inverse.shape == (4, 3)
        np.testing.assert_allclose(embedding @ inverse, np.eye(3), atol=2e-14)
        xyz = np.array((1.25, -3.5, 2.75)) * scale
        continuous = inverse @ xyz
        np.testing.assert_allclose(embedding @ continuous, xyz, rtol=2e-14, atol=abs(scale) * 2e-14)
        assert sum(continuous) == pytest.approx(0, abs=2e-14)

    def test_default_embedding_inverse_has_correct_units(self):
        inverse = xyz_to_quadray_matrix()
        np.testing.assert_allclose(EMBEDDING @ inverse, np.eye(3), atol=2e-14)
        np.testing.assert_allclose(inverse @ np.array([1, 1, 1]), [0.75, -0.25, -0.25, -0.25])

    def test_custom_nonsymmetric_embedding_uses_supplied_geometry(self):
        # A sheared anisotropic embedding is not a scaled regular tetrahedron.
        deformation = np.array([[2, 0.5, -0.25], [0, 0.75, 0.5], [0, 0, -1.5]])
        embedding = deformation @ EMBEDDING
        inverse = xyz_to_quadray_matrix(embedding)
        np.testing.assert_allclose(embedding @ inverse, np.eye(3), atol=2e-14)
        continuous = inverse @ np.array([3, -2, 7])
        np.testing.assert_allclose(embedding @ continuous, [3, -2, 7], atol=2e-14)
        assert sum(continuous) == pytest.approx(0, abs=2e-14)
        # Minimum norm representative: any nonzero gauge increases its norm.
        for gauge in (-3.0, 0.5, 2.0):
            assert np.dot(continuous + gauge, continuous + gauge) > np.dot(continuous, continuous)

    @pytest.mark.parametrize("embedding", [
        np.eye(4), np.ones((3, 4)), np.zeros((3, 4)),
        np.array([[1, -1, 0, 0], [2, -2, 0, 0], [0, 0, 1, -1]]),
        np.array([[1, -1, -1, 2], [1, 1, -1, -1], [1, -1, 1, -1]]),
        EMBEDDING * np.nan, EMBEDDING * np.inf,
    ])
    def test_invalid_embeddings_rejected(self, embedding):
        with pytest.raises(ValueError):
            xyz_to_quadray_matrix(embedding)


class TestPolyhedronTransformations:
    def test_translation_preserves_volume_and_all_vertex_displacements(self):
        tetra = Tetrahedron()
        before = [vertex.as_tuple() for vertex in tetra.vertices]
        offset = QuadrayCoordinate(2, 0, 1, 0)
        result = translate_polyhedron(tetra, offset)
        assert result is not tetra
        for original, translated in zip(tetra.vertices, result.vertices):
            np.testing.assert_array_equal(spatial(translated) - spatial(original), (1, 1, 3))
        assert squared_edges(result) == squared_edges(tetra)
        assert result.volume() == tetra.volume() == 1
        assert [vertex.as_tuple() for vertex in tetra.vertices] == before

    @pytest.mark.parametrize("factor", [-2, 0, 3])
    def test_integral_scaling_changes_lengths_and_volume_in_spatial_units(self, factor):
        tetra = Tetrahedron()
        result = scale_polyhedron(tetra, factor)
        assert result is not tetra
        for original, transformed in zip(tetra.vertices, result.vertices):
            np.testing.assert_array_equal(spatial(transformed), factor * spatial(original))
        assert squared_edges(result) == [factor**2 * edge for edge in squared_edges(tetra)]
        assert result.volume() == abs(factor)**3
        assert tetra.volume() == 1

    @pytest.mark.parametrize("axis,normal", [
        ("a", (1, 1, 1)), ("b", (-1, 1, -1)),
        ("c", (-1, -1, 1)), ("d", (1, -1, -1)),
    ])
    def test_plane_reflection_preserves_lattice_polyhedron_geometry(self, axis, normal):
        # Multiples of three remain on the lattice for these plane normals.
        tetra = Tetrahedron([
            QuadrayCoordinate(0, 0, 0, 0), QuadrayCoordinate(6, 3, 0, 3),
            QuadrayCoordinate(6, 3, 3, 0), QuadrayCoordinate(6, 0, 3, 3),
        ])
        normal = np.array(normal)
        result = reflect_polyhedron(tetra, axis)
        for original, transformed in zip(tetra.vertices, result.vertices):
            xyz = spatial(original)
            np.testing.assert_array_equal(spatial(transformed), xyz - 2 * np.dot(xyz, normal) * normal / 3)
        assert squared_edges(result) == squared_edges(tetra)
        assert result.volume() == tetra.volume() == 27

    def test_origin_reflection_preserves_volume_and_reverses_every_point(self):
        tetra = Tetrahedron()
        result = reflect_polyhedron(tetra)
        for original, transformed in zip(tetra.vertices, result.vertices):
            np.testing.assert_array_equal(spatial(transformed), -spatial(original))
        assert squared_edges(result) == squared_edges(tetra)
        assert result.volume() == tetra.volume() == 1


class TestRotationGeometry:
    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    @pytest.mark.parametrize("angle", [-37, 0, 90, 360])
    def test_rotation_matrices_are_proper_isometries(self, axis, angle):
        matrix = create_rotation_matrix(axis, angle)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=2e-14)
        assert np.linalg.det(matrix) == pytest.approx(1)
        np.testing.assert_allclose(create_rotation_matrix(axis, -angle) @ matrix, np.eye(3), atol=2e-14)

    @pytest.mark.parametrize("axis,expected", [
        ("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)),
    ])
    def test_rotation_keeps_its_axis_fixed(self, axis, expected):
        np.testing.assert_allclose(create_rotation_matrix(axis, 37) @ expected, expected, atol=2e-14)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    @pytest.mark.parametrize("angle", [0, 360])
    def test_full_turns_recover_lattice_point(self, axis, angle):
        point = QuadrayCoordinate(0, 1, 2, 3)
        assert rotate_around_axis(point, axis, angle) == point

    def test_rotation_factory_composes_quarter_turns(self):
        rotate = compose_transforms(rotate_around_xyz_axis("z", 90), rotate_around_xyz_axis("z", 90))
        point = QuadrayCoordinate(1, 0, 0, 0)
        np.testing.assert_array_equal(spatial(rotate(point)), (-1, -1, 1))

    def test_invalid_axis_rejected(self):
        with pytest.raises(ValueError):
            create_rotation_matrix("invalid", 30)


class TestPlaneProjection:
    @pytest.mark.parametrize("normal,point,expected", [
        ((0, 0, 1), None, (2, -4, 0)),
        ((1, 0, 0), (4, 0, 0), (4, -4, -2)),
        ((0, 1, 0), None, (2, 0, -2)),
    ])
    def test_axis_plane_projection_at_exact_lattice_sites(self, normal, point, expected):
        # Original XYZ=(2,-4,-2); these projections require no quantization.
        coord = QuadrayCoordinate(0, 0, 1, 3)
        result = project_to_plane(coord, normal, point)
        np.testing.assert_array_equal(spatial(result), expected)


@pytest.mark.parametrize("scale", [2.0**1023, 2.0**-1020])
def test_continuous_inverse_preserves_extreme_finite_embedding_units(scale):
    embedding = scale * EMBEDDING
    inverse = xyz_to_quadray_matrix(embedding)
    np.testing.assert_allclose(embedding @ inverse, np.eye(3), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(inverse * scale, EMBEDDING.T / 4, rtol=2e-14, atol=2e-14)


def test_continuous_inverse_retains_independent_extremely_anisotropic_axes():
    scales = np.array([1e308, 1e-308, 1.0])
    embedding = scales[:, None] * EMBEDDING
    inverse = xyz_to_quadray_matrix(embedding)
    # Analytic Urner oracle; no SVD or production inversion helper.
    expected = EMBEDDING.T * (1 / scales)[None, :] / 4
    assert np.isfinite(inverse).all()
    np.testing.assert_allclose(inverse, expected, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(inverse.sum(axis=0), np.zeros(3))
    # Normalize Cartesian units before multiplication: a raw E @ inverse
    # would overflow cancelling off-diagonal products even for the oracle.
    np.testing.assert_allclose(
        EMBEDDING @ (inverse * scales[None, :]), np.eye(3), rtol=2e-15, atol=2e-15)
    continuous = np.array([0.75, -0.25, -0.25, -0.25])
    xyz = embedding @ continuous
    np.testing.assert_allclose(inverse @ xyz, continuous, rtol=2e-15, atol=2e-15)


def test_continuous_inverse_uses_effective_first_columns_with_tolerated_fourth_roundoff():
    embedding = EMBEDDING.copy()
    embedding[0, 3] = np.nextafter(embedding[0, 3], np.inf)
    inverse = xyz_to_quadray_matrix(embedding)
    np.testing.assert_array_equal(inverse, EMBEDDING.T / 4)
