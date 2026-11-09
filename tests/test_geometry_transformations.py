"""Spatial and exact-arithmetic contracts for quadray transformations."""

import math
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.geometry.transformations import (
    apply_transform_function,
    compose_transforms,
    coordinate_transform,
    reflect,
    project_to_plane,
    rotate_around_axis,
    scale,
    scale_by_factor,
    translate,
    translate_by_vector,
)


def spatial(coord):
    """Independent integer Urner oracle, without a float conversion."""
    a, b, c, d = coord.as_tuple()
    return np.array((a - b - c + d, a + b - c - d, a - b + c - d))


def nearest_squared_error(target):
    """Enumerate nearby XYZ sites: the integer image has common parity."""
    ranges = [range(math.floor(value) - 2, math.floor(value) + 4) for value in target]
    return min(
        sum((value - expected) ** 2 for value, expected in zip(site, target))
        for site in product(*ranges)
        if site[0] % 2 == site[1] % 2 == site[2] % 2
    )


def shifted(coord, gauge=2**60):
    return QuadrayCoordinate(*(value + gauge for value in coord.as_tuple()), normalize=False)


class TestTranslation:
    def test_translate_coordinate(self):
        point = QuadrayCoordinate(2, 1, 0, 3)
        offset = QuadrayCoordinate(0, 1, 2, 0)
        np.testing.assert_array_equal(spatial(translate(point, offset)), spatial(point) + spatial(offset))

    def test_translate_by_vector_function(self):
        # This vector is itself a lattice point, so no snapping is needed.
        result = translate_by_vector(1, 3, -1)(QuadrayCoordinate(1, 0, 0, 0))
        np.testing.assert_array_equal(spatial(result), (2, 4, 0))

    @pytest.mark.parametrize("xyz", [
        (2**60 + 2, 0, 0),
        (2**100 + 1, -(2**90) + 1, 3),
    ])
    def test_integer_xyz_translation_preserves_exact_lattice_displacement(self, xyz):
        point = QuadrayCoordinate(7, 2, 1, 0)
        result = translate_by_vector(*xyz)(point)
        # Python integers, not to_xyz or a binary64 expected value.
        a, b, c, d = result.sub(point).as_tuple()
        assert (a - b - c + d, a + b - c - d, a - b + c - d) == xyz


@pytest.mark.parametrize("magnitude", [1, -1, 1e200, -1e200, 1e-200, -1e-200])
def test_plane_projection_is_invariant_under_normal_rescaling(magnitude):
    point = QuadrayCoordinate(2, 0, 0, 0)
    expected = QuadrayCoordinate(2, 1, 1, 0)
    assert project_to_plane(point, (magnitude, 0, 0)) == expected
    # Non-origin plane, with a projection which is itself exactly on-lattice.
    expected = QuadrayCoordinate.from_xyz(4, 2, 2)
    assert project_to_plane(point, (magnitude, 0, 0), (4, 0, 0)) == expected


def test_numpy_float32_plane_inputs_preserve_exact_projection_and_nearest_site():
    point = QuadrayCoordinate(2, 0, 0, 0)
    normal = tuple(np.float32(value) for value in (0.5, 0.25, 0.0))
    # Exact Cartesian target (-2/5, 4/5, 2); nearest common-parity site.
    np.testing.assert_array_equal(spatial(project_to_plane(point, normal)), (0, 0, 2))
    plane_point = tuple(np.float32(value) for value in (0.5, 0.25, 0.0))
    # Offset plane gives (1/10, 21/20, 2), nearest to (0, 2, 2).
    np.testing.assert_array_equal(
        spatial(project_to_plane(point, (0.5, 0.25, 0.0), plane_point)), (0, 2, 2)
    )


@pytest.mark.parametrize("normal", [
    (0, 0, 0), (math.inf, 0, 0), (math.nan, 0, 0), (1, 0), (1, 0, 0, 0),
])
def test_plane_projection_rejects_undefined_normals(normal):
    with pytest.raises(ValueError):
        project_to_plane(QuadrayCoordinate(2, 0, 0, 0), normal)


@pytest.mark.parametrize("size", [6, 10**308])
def test_plane_operations_preserve_finite_geometry_despite_dot_product_overflow(size):
    point = QuadrayCoordinate(size, 0, 0, 0)
    assert tuple(spatial(reflect(point, "a"))) == (-size,) * 3
    assert tuple(spatial(project_to_plane(point, (1, 1, 1)))) == (0, 0, 0)
    assert tuple(spatial(project_to_plane(point, (1, 1, 1), (2, 2, 2)))) == (2, 2, 2)
    assert tuple(spatial(project_to_plane(point, (1, 1, 0), (4, 0, 0)))) == (2, 2, size)


@pytest.mark.parametrize("size", [7, 10**308 + 1])
def test_oblique_projection_snaps_once_to_nearest_site_with_exact_ties(size):
    point = QuadrayCoordinate(size, 0, 0, 0)
    normal = (1, 2, 3)
    # Dot((size,size,size), normal) / dot(normal,normal) = 3*size/7.
    target = tuple(Fraction(size) - Fraction(3 * size * weight, 7) for weight in normal)
    result = project_to_plane(point, normal)
    error = sum((int(value) - expected)**2 for value, expected in zip(spatial(result), target))
    assert error == nearest_squared_error(target)
    assert project_to_plane(shifted(point, 10**400), normal) == result
    assert project_to_plane(point, tuple(-weight for weight in normal)) == result


class TestScaling:
    @pytest.mark.parametrize("factor", [-3, -1, 0, 1, 2, 2.0])
    def test_integer_scale_is_exact_above_float_precision(self, factor):
        n = 2**60 + 1
        point = QuadrayCoordinate(n, 2, 0, 7)
        result = scale(point, factor)
        values = tuple(int(factor) * value for value in point.as_tuple())
        expected = tuple(value - min(values) for value in values)
        assert result.as_tuple() == expected
        assert scale(shifted(point), factor).as_tuple() == expected

    @pytest.mark.parametrize("factor", [0.5, -0.5, 0.3, -1.25])
    def test_fractional_scale_is_gauge_invariant_and_spatially_nearest(self, factor):
        point = QuadrayCoordinate(3, 1, 2, 0)
        target = spatial(point) * factor
        result = scale(point, factor)
        assert scale(shifted(point), factor) == result
        error = np.sum((spatial(result) - target) ** 2)
        assert error == pytest.approx(nearest_squared_error(target))

    def test_fractional_scale_does_not_truncate_components(self):
        # XYZ (1,1,1) * .75 is uniquely nearer (1,1,1) than the origin.
        point = QuadrayCoordinate(1, 0, 0, 0)
        assert scale(point, 0.75) == point
        assert scale_by_factor(0.75)(shifted(point)) == point

    @pytest.mark.parametrize("factor", [math.inf, -math.inf, math.nan])
    def test_nonfinite_factor_rejected(self, factor):
        with pytest.raises(ValueError):
            scale(QuadrayCoordinate(1, 0, 0, 0), factor)


class TestReflection:
    @pytest.mark.parametrize("axis", ["origin", "a", "b", "c", "d"])
    def test_shifted_origin_stays_at_origin(self, axis):
        result = reflect(QuadrayCoordinate(*(2**60,) * 4, normalize=False), axis)
        assert result.as_tuple() == (0, 0, 0, 0)

    def test_origin_inversion_remains_exact(self):
        point = QuadrayCoordinate(2**60 + 1, 0, 5, 9)
        result = reflect(shifted(point), "origin")
        maximum = max(point.as_tuple())
        assert result.as_tuple() == tuple(maximum - value for value in point.as_tuple())
        assert reflect(result, "origin") == point

    @pytest.mark.parametrize("index,axis", tuple(enumerate("abcd")))
    def test_named_plane_fixes_tangents_and_reverses_normal(self, index, axis):
        # Each basis vector has norm squared 3 and pairwise dot product -1.
        # One normal plus three of another basis is tangent to this plane.
        values = [0] * 4
        values[index] = 1
        values[(index + 1) % 4] = 3
        tangent = QuadrayCoordinate(*values)
        assert reflect(tangent, axis) == tangent
        normal_values = [0] * 4
        normal_values[index] = 2
        normal = QuadrayCoordinate(*normal_values)
        np.testing.assert_array_equal(spatial(reflect(normal, axis)), -spatial(normal))

    @pytest.mark.parametrize("axis,normal", [
        ("a", (1, 1, 1)), ("b", (-1, 1, -1)),
        ("c", (-1, -1, 1)), ("d", (1, -1, -1)),
    ])
    def test_named_plane_euclidean_reflection_on_lattice(self, axis, normal):
        point = QuadrayCoordinate(3, 6, 9, 0)
        xyz = spatial(point)
        normal = np.array(normal)
        expected = xyz - 2 * np.dot(xyz, normal) * normal / 3
        result = reflect(point, axis)
        np.testing.assert_array_equal(spatial(result), expected)
        assert reflect(shifted(point), axis) == result
        assert reflect(result, axis) == point
        assert np.dot(spatial(result), spatial(result)) == np.dot(xyz, xyz)

    def test_off_lattice_reflection_uses_nearest_spatial_site(self):
        point = QuadrayCoordinate(0, 1, 2, 3)
        xyz = spatial(point)
        target = xyz - 2 * sum(xyz) / 3
        result = reflect(point, "a")
        error = np.sum((spatial(result) - target) ** 2)
        assert error == pytest.approx(nearest_squared_error(target))
        assert reflect(shifted(point), "a") == result

    def test_invalid_plane_rejected(self):
        with pytest.raises(ValueError):
            reflect(QuadrayCoordinate(1, 0, 0, 0), "invalid")


class TestCoordinateTransform:
    @pytest.mark.parametrize("matrix", [np.eye(4, dtype=int), -np.eye(4), 3 * np.eye(4)])
    def test_integer_maps_are_exact_and_gauge_invariant(self, matrix):
        point = QuadrayCoordinate(2**60 + 1, 2, 0, 7)
        factor = int(matrix[0, 0])
        values = tuple(value * factor for value in point.as_tuple())
        expected = tuple(value - min(values) for value in values)
        assert coordinate_transform(point, matrix).as_tuple() == expected
        assert coordinate_transform(shifted(point), matrix).as_tuple() == expected

    def test_basis_permutation_is_a_spatial_rotation(self):
        # Swapping b/c exchanges Y/Z while leaving X unchanged.
        matrix = np.eye(4, dtype=int)[[0, 2, 1, 3]]
        point = QuadrayCoordinate(5, 1, 2, 0)
        result = coordinate_transform(shifted(point), matrix)
        x, y, z = spatial(point)
        np.testing.assert_array_equal(spatial(result), (x, z, y))

    @pytest.mark.parametrize("factor", [0.5, 0.75, -0.3])
    def test_fractional_matrix_quantizes_in_xyz(self, factor):
        point = QuadrayCoordinate(3, 1, 2, 0)
        target = spatial(point) * factor
        result = coordinate_transform(point, factor * np.eye(4))
        assert result == coordinate_transform(shifted(point), factor * np.eye(4))
        assert np.sum((spatial(result) - target) ** 2) == pytest.approx(nearest_squared_error(target))

    def test_general_spatial_shear_uses_nearest_lattice_projection(self):
        # Lift a Cartesian shear to the zero-sum quadray representatives.
        embedding = np.array([[1, -1, -1, 1], [1, 1, -1, -1], [1, -1, 1, -1]])
        shear = np.array([[1, 0.3, 0], [0, 1, -0.2], [0, 0, 1]])
        matrix = embedding.T @ shear @ embedding / 4
        point = QuadrayCoordinate(4, 0, 1, 2)
        target = shear @ spatial(point)
        result = coordinate_transform(point, matrix)
        assert coordinate_transform(shifted(point), matrix) == result
        assert np.sum((spatial(result) - target) ** 2) == pytest.approx(nearest_squared_error(target))

    def test_added_output_gauge_does_not_change_map(self):
        point = QuadrayCoordinate(3, 1, 0, 2)
        matrix = 0.75 * np.eye(4)
        output_gauge = np.ones((4, 1)) @ np.array([[0.25, -0.5, 1.5, 0.75]])
        assert coordinate_transform(point, matrix + output_gauge) == coordinate_transform(point, matrix)

    @pytest.mark.parametrize("matrix", [np.diag([1, 2, 3, 4]), np.diag([0.5, 1, 1, 1])])
    def test_gauge_incompatible_map_rejected(self, matrix):
        with pytest.raises(ValueError):
            coordinate_transform(QuadrayCoordinate(1, 0, 0, 0), matrix)

    @pytest.mark.parametrize("matrix", [np.eye(3), np.zeros((4, 3))])
    def test_invalid_shape_rejected(self, matrix):
        with pytest.raises(ValueError):
            coordinate_transform(QuadrayCoordinate(1, 0, 0, 0), matrix)

    @pytest.mark.parametrize("value", [math.inf, math.nan])
    def test_nonfinite_matrix_rejected(self, value):
        matrix = np.eye(4)
        matrix[0, 0] = value
        with pytest.raises(ValueError):
            coordinate_transform(QuadrayCoordinate(1, 0, 0, 0), matrix)


class TestTransformationComposition:
    def test_composition_preserves_order(self):
        offset = QuadrayCoordinate(0, 1, 0, 0)
        composed = compose_transforms(lambda point: translate(point, offset), scale_by_factor(2))
        point = QuadrayCoordinate(1, 0, 0, 0)
        np.testing.assert_array_equal(spatial(composed(point)), (0, 4, 0))

    def test_list_transform_preserves_order_and_values(self):
        points = [QuadrayCoordinate(1, 0, 0, 0), QuadrayCoordinate(0, 1, 0, 0)]
        result = apply_transform_function(points, scale_by_factor(2))
        assert [point.as_tuple() for point in result] == [(2, 0, 0, 0), (0, 2, 0, 0)]
        assert apply_transform_function([], scale_by_factor(2)) == []


class TestRotation:
    @pytest.mark.parametrize("axis,expected", [
        ("x", (1, -1, 1)), ("y", (1, 1, -1)), ("z", (-1, 1, 1)),
    ])
    def test_quarter_turn(self, axis, expected):
        result = rotate_around_axis(QuadrayCoordinate(1, 0, 0, 0), axis, 90)
        np.testing.assert_array_equal(spatial(result), expected)

    def test_invalid_rotation_axis_rejected(self):
        with pytest.raises(ValueError):
            rotate_around_axis(QuadrayCoordinate(1, 0, 0, 0), "invalid", 45)
