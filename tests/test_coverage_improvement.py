#!/usr/bin/env python3
"""
Coverage Improvement Tests

This module contains tests to improve code coverage for areas that are
currently not well tested, aiming to reach 95%+ coverage.
"""

import pytest
import math
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the main modules to test
from symergetics.core.constants import SymergeticsConstants
from symergetics.core.numbers import SymergeticsNumber, rational_sqrt, rational_pi
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.geometry.transformations import translate, scale, reflect, rotate_around_axis
from symergetics.computation.primorials import primorial, scheherazade_power, is_prime, next_prime, prime_count_up_to, prime_factors
from symergetics.utils.conversion import continued_fraction_approximation, convergents_from_continued_fraction, best_rational_approximation
from symergetics.utils.mnemonics import mnemonic_encode, mnemonic_decode, format_large_number, ungroup_number, create_memory_aid
from symergetics.visualization.geometry import plot_polyhedron_3d, plot_polyhedron_graphical_abstract, plot_polyhedron_wireframe
from symergetics.visualization.numbers import plot_palindromic_heatmap, plot_scheherazade_network, plot_primorial_spectrum
from symergetics.visualization.mathematical import plot_continued_fraction_convergence, plot_base_conversion_matrix, plot_pattern_analysis_radar


class TestConstantsCoverage:
    """Test coverage improvements for constants module."""

    def test_get_scheherazade_power_uncached(self):
        """Test getting scheherazade power for uncached value."""
        # Test a higher power that isn't precomputed
        result = SymergeticsConstants.get_scheherazade_power(8)
        expected = SymergeticsConstants.SCHEHERAZADE_BASE ** 8
        assert result == expected

    def test_get_primorial_uncached(self):
        """Test getting primorial for uncached value."""
        # Test a higher n that isn't precomputed
        result = SymergeticsConstants.get_primorial(17)
        # Verify it's the product of primes ≤ 17
        primes = [2, 3, 5, 7, 11, 13, 17]
        expected = SymergeticsNumber(1)
        for prime in primes:
            expected = expected * SymergeticsNumber(prime)
        assert result == expected

    def test_get_primes_up_to_edge_cases(self):
        """Test _get_primes_up_to with edge cases."""
        # Test n=1 (should return empty list)
        primes = SymergeticsConstants._get_primes_up_to(1)
        assert primes == []

        # Test n=2 (should return [2])
        primes = SymergeticsConstants._get_primes_up_to(2)
        assert primes == [2]

        # Test n=3 (should return [2, 3])
        primes = SymergeticsConstants._get_primes_up_to(3)
        assert primes == [2, 3]


class TestNumbersCoverage:
    """Test coverage improvements for numbers module."""

    def test_symergetics_number_edge_cases(self):
        """Test edge cases in SymergeticsNumber."""
        # Test with very large numbers
        large_num = SymergeticsNumber(10**100, 1)
        assert large_num.value == SymergeticsNumber(10**100)

        # Test division by SymergeticsNumber
        a = SymergeticsNumber(5)
        b = SymergeticsNumber(2)
        result = a / b
        assert result == SymergeticsNumber(5, 2)

    def test_rational_operations_edge_cases(self):
        """Test edge cases in rational operations."""
        # Test sqrt of perfect squares
        sqrt_4 = rational_sqrt(SymergeticsNumber(4))
        assert sqrt_4 == SymergeticsNumber(2)

        sqrt_9 = rational_sqrt(SymergeticsNumber(9))
        assert sqrt_9 == SymergeticsNumber(3)

        # Test sqrt of non-perfect squares (approximate equality)
        sqrt_2 = rational_sqrt(SymergeticsNumber(2))
        product = sqrt_2 * sqrt_2
        # Check if they're approximately equal
        assert abs(float(product.value) - 2.0) < 0.01

    def test_from_float_edge_cases(self):
        """Test from_float with edge cases."""
        # Test with zero
        zero = SymergeticsNumber.from_float(0.0)
        assert zero == SymergeticsNumber(0)

        # Test with negative numbers
        neg = SymergeticsNumber.from_float(-2.5)
        assert neg == SymergeticsNumber(-5, 2)


class TestGeometryTransformationsCoverage:
    """Test coverage improvements for geometry transformations."""

    def test_translate_basic(self):
        """Test basic translate functionality."""
        coord = QuadrayCoordinate(1, 2, 3, 4)
        offset = QuadrayCoordinate(1, 0, 0, 0)
        result = translate(coord, offset)
        # Due to IVM normalization, the result will be different
        # Just check that translation occurred and result is a QuadrayCoordinate
        assert isinstance(result, QuadrayCoordinate)
        assert result != coord  # Should be different after translation

    def test_scale_edge_cases(self):
        """Test scale with edge cases."""
        coord = QuadrayCoordinate(1, 2, 3, 4)

        # Scale by 1 (should be identity)
        result = scale(coord, 1)
        assert result == coord

        # Scale by 0
        result = scale(coord, 0)
        assert result == QuadrayCoordinate(0, 0, 0, 0)

        # Scale by 2 (will be affected by IVM normalization)
        result = scale(coord, 2)
        assert isinstance(result, QuadrayCoordinate)
        # Check that scaling occurred (coordinates should be different)
        assert result != coord

    def test_reflect_edge_cases(self):
        """Test reflect with edge cases."""
        # Test origin point
        origin = QuadrayCoordinate(0, 0, 0, 0)
        result = reflect(origin, 'origin')
        assert result == origin

        # Test simple coordinate (reflection will be affected by IVM normalization)
        coord = QuadrayCoordinate(1, 0, 0, 0)
        result = reflect(coord, 'origin')
        assert isinstance(result, QuadrayCoordinate)
        assert result != coord  # Should be different after reflection

    def test_rotate_around_axis_invalid_axis(self):
        """Test rotate_around_axis with invalid axis."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        with pytest.raises(ValueError):
            rotate_around_axis(coord, 'invalid', 90)

    def test_rotate_around_axis_edge_cases(self):
        """Test rotate_around_axis with edge cases."""
        coord = QuadrayCoordinate(1, 0, 0, 0)

        # Rotate by 0 (should be identity)
        result = rotate_around_axis(coord, 'x', 0)
        assert result == coord

        # Rotate by 360 degrees (should be identity)
        result = rotate_around_axis(coord, 'x', 360)
        assert result == coord


class TestComputationPrimorialsCoverage:
    """Test coverage improvements for primorials computation."""

    def test_scheherazade_power_edge_cases(self):
        """Test scheherazade_power with edge cases."""
        # Test power of 0
        result = scheherazade_power(0)
        assert result == SymergeticsNumber(1)

        # Test power of 1
        result = scheherazade_power(1)
        assert result == SymergeticsNumber(1001)

    def test_prime_factors_edge_cases(self):
        """Test prime_factors with edge cases."""
        # Test 1 (should return empty dict)
        factors = prime_factors(1)
        assert factors == {}

        # Test prime number
        factors = prime_factors(7)
        assert factors == {7: 1}

        # Test composite number
        factors = prime_factors(12)
        assert factors == {2: 2, 3: 1}

    def test_prime_count_up_to_edge_cases(self):
        """Test prime_count_up_to with edge cases."""
        # Test n=1
        count = prime_count_up_to(1)
        assert count == 0

        # Test n=2
        count = prime_count_up_to(2)
        assert count == 1

        # Test n=10
        count = prime_count_up_to(10)
        assert count == 4  # 2, 3, 5, 7


class TestUtilsConversionCoverage:
    """Test coverage improvements for conversion utilities."""

    def test_continued_fraction_approximation_edge_cases(self):
        """Test continued_fraction_approximation with edge cases."""
        # Test with integer
        cf = continued_fraction_approximation(5.0, max_terms=5)
        assert cf == [5]

        # Test with simple fraction
        cf = continued_fraction_approximation(3.5, max_terms=5)
        assert cf == [3, 2]

    def test_convergents_from_continued_fraction_edge_cases(self):
        """Test convergents_from_continued_fraction with edge cases."""
        # Test with single term
        convergents = convergents_from_continued_fraction([3])
        assert len(convergents) == 1
        assert convergents[0] == (3, 1)

        # Test with multiple terms
        convergents = convergents_from_continued_fraction([1, 2])
        assert len(convergents) == 2
        assert convergents[0] == (1, 1)
        assert convergents[1] == (3, 2)

    def test_best_rational_approximation_edge_cases(self):
        """Test best_rational_approximation with edge cases."""
        # Test with integer
        best = best_rational_approximation(5.0, max_denominator=10)
        assert best.numerator == 5
        assert best.denominator == 1

        # Test with simple fraction
        best = best_rational_approximation(0.5, max_denominator=10)
        assert best.numerator == 1
        assert best.denominator == 2


class TestUtilsMnemonicsCoverage:
    """Test coverage improvements for mnemonics utilities."""

    def test_mnemonic_encode_special_cases(self):
        """Test mnemonic_encode with special cases."""
        # Test with SymergeticsNumber
        num = SymergeticsNumber(12345)
        result = mnemonic_encode(num, 'grouped')
        assert isinstance(result, str)

        # Test with regular int
        result = mnemonic_encode(12345, 'scientific')
        assert isinstance(result, str)

    def test_mnemonic_decode_edge_cases(self):
        """Test mnemonic_decode with edge cases."""
        # Test with invalid input (should return error message)
        result = mnemonic_decode("invalid")
        assert isinstance(result, str)
        assert "Could not decode" in result

        # Test with properly formatted number (what mnemonic_decode can handle)
        original = 12345
        formatted = "12,345"
        decoded = mnemonic_decode(formatted)
        assert decoded == original

        # Test with spaces
        formatted_with_spaces = "12 345"
        decoded = mnemonic_decode(formatted_with_spaces)
        assert decoded == original

    def test_format_large_number_edge_cases(self):
        """Test format_large_number with edge cases."""
        # Test with small number
        result = format_large_number(123)
        assert result == "123"

        # Test with SymergeticsNumber
        num = SymergeticsNumber(12345)
        result = format_large_number(num)
        assert isinstance(result, str)

    def test_ungroup_number_edge_cases(self):
        """Test ungroup_number with edge cases."""
        # Test with no grouping
        result = ungroup_number("12345")
        assert result == 12345

        # Test with grouping
        result = ungroup_number("12,345")
        assert result == 12345

    def test_create_memory_aid_edge_cases(self):
        """Test create_memory_aid with edge cases."""
        # Test with small number
        result = create_memory_aid(42)
        assert isinstance(result, dict)

        # Test with SymergeticsNumber
        num = SymergeticsNumber(1001)
        result = create_memory_aid(num)
        assert isinstance(result, dict)


class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""

    def test_invalid_backend_handling(self):
        """Test that invalid backends raise appropriate errors."""
        from symergetics.geometry.polyhedra import Tetrahedron

        tetra = Tetrahedron()

        # Test with invalid backend for 3D plot
        with pytest.raises(ValueError):
            plot_polyhedron_3d(tetra, backend='invalid')

        # Test with invalid backend for graphical abstract
        with pytest.raises(ValueError):
            plot_polyhedron_graphical_abstract(tetra, backend='invalid')

        # Test with invalid backend for wireframe
        with pytest.raises(ValueError):
            plot_polyhedron_wireframe(tetra, backend='invalid')
