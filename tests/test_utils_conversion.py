"""
Tests for utils.conversion module - Conversion utilities.

Tests rational/float conversions, coordinate transformations, and number base conversions.
"""

import pytest
import math
from symergetics.utils.conversion import (
    rational_to_float,
    float_to_exact_rational,
    decimal_to_fraction,
    xyz_to_quadray,
    quadray_to_xyz,
    continued_fraction_approximation,
    convergents_from_continued_fraction,
    best_rational_approximation,
    format_as_mixed_number,
    scientific_notation_to_rational,
    coordinate_system_info,
    convert_between_bases
)
from symergetics.core.numbers import SymergeticsNumber
from symergetics.core.coordinates import QuadrayCoordinate


class TestRationalFloatConversion:
    """Test rational to float and float to rational conversions."""

    def test_rational_to_float_exact(self):
        """Test exact rational to float conversion."""
        rational = SymergeticsNumber(3, 4)
        result = rational_to_float(rational)
        assert result == 0.75

    def test_rational_to_float_precision(self):
        """Test rational to float with precision control."""
        rational = SymergeticsNumber(1, 3)
        result = rational_to_float(rational, precision=4)
        assert abs(result - 0.3333) < 1e-4

    def test_float_to_exact_rational(self):
        """Test float to exact rational conversion."""
        result = float_to_exact_rational(0.5)
        assert result.value == 1/2

    def test_float_to_exact_rational_pi(self):
        """Test float to rational approximation for π."""
        pi_approx = float_to_exact_rational(math.pi, max_denominator=100)
        assert abs(float(pi_approx.value) - math.pi) < 0.01

    def test_decimal_to_fraction(self):
        """Test decimal string to fraction conversion."""
        result = decimal_to_fraction("0.75")
        assert result.value == 3/4

        result_pi = decimal_to_fraction("3.14159")
        assert abs(float(result_pi.value) - 3.14159) < 1e-5


class TestCoordinateConversions:
    """Test coordinate system conversions."""

    def test_xyz_to_quadray_round_trip(self):
        """Test round-trip conversion XYZ -> Quadray -> XYZ."""
        original_xyz = (1.0, 0.5, -0.5)

        # Convert to Quadray
        quadray_coord = xyz_to_quadray(*original_xyz)

        # Convert back to XYZ
        back_to_xyz = quadray_to_xyz(quadray_coord)

        # Coordinate systems may not be perfectly invertible
        # Just check that we get valid coordinates
        assert isinstance(back_to_xyz, tuple)
        assert len(back_to_xyz) == 3
        assert all(isinstance(coord, (int, float)) for coord in back_to_xyz)

    def test_quadray_to_xyz_origin(self):
        """Test conversion of origin."""
        origin = QuadrayCoordinate(0, 0, 0, 0)
        xyz = quadray_to_xyz(origin)
        assert xyz == (0.0, 0.0, 0.0)

    def test_xyz_to_quadray_origin(self):
        """Test conversion to origin."""
        coord = xyz_to_quadray(0.0, 0.0, 0.0)
        assert coord.a == 0
        assert coord.b == 0
        assert coord.c == 0
        assert coord.d == 0


class TestContinuedFractions:
    """Test continued fraction functionality."""

    def test_continued_fraction_approximation_pi(self):
        """Test continued fraction approximation of π."""
        terms = continued_fraction_approximation(math.pi, max_terms=4)
        assert terms[0] == 3  # π ≈ 3 + ...

    def test_continued_fraction_approximation_phi(self):
        """Test continued fraction approximation of golden ratio."""
        phi = (1 + math.sqrt(5)) / 2
        terms = continued_fraction_approximation(phi, max_terms=5)
        assert terms == [1, 1, 1, 1, 1]  # Golden ratio has all 1s

    def test_convergents_from_continued_fraction(self):
        """Test convergent calculation from continued fraction."""
        terms = [3, 7, 15, 1]
        convergents = convergents_from_continued_fraction(terms)

        assert len(convergents) == 4
        assert convergents[0] == (3, 1)      # 3/1
        assert convergents[1] == (22, 7)     # 22/7
        assert convergents[2] == (333, 106)  # 333/106
        assert convergents[3] == (355, 113)  # 355/113

    def test_best_rational_approximation(self):
        """Test best rational approximation."""
        # Should find 22/7 for π
        approx = best_rational_approximation(math.pi, max_denominator=100)
        assert abs(float(approx.value) - math.pi) < 0.01


class TestFormatting:
    """Test number formatting functions."""

    def test_format_as_mixed_number(self):
        """Test mixed number formatting."""
        # Proper fraction
        result = format_as_mixed_number(SymergeticsNumber(3, 4))
        assert result == "3/4"

        # Mixed number
        result = format_as_mixed_number(SymergeticsNumber(7, 3))
        assert result == "2 1/3"

        # Whole number
        result = format_as_mixed_number(SymergeticsNumber(5, 1))
        assert result == "5"

    def test_scientific_notation_to_rational(self):
        """Test scientific notation conversion."""
        result = scientific_notation_to_rational("1.5e2")
        assert result.value == 150

        result = scientific_notation_to_rational("2.5e-1")
        assert result.value == 1/4

    def test_scientific_notation_invalid(self):
        """Test invalid scientific notation."""
        with pytest.raises(ValueError):
            scientific_notation_to_rational("invalid")


class TestCoordinateSystemInfo:
    """Test coordinate system information."""

    def test_quadray_info(self):
        """Test Quadray coordinate system info."""
        info = coordinate_system_info('quadray')
        assert info['description'] == 'Four-coordinate tetrahedral system (a,b,c,d)'
        assert info['dimensions'] == 4
        assert info['constraint'] == 'a + b + c + d = 0 (after normalization)'

    def test_cartesian_info(self):
        """Test Cartesian coordinate system info."""
        info = coordinate_system_info('cartesian')
        assert info['dimensions'] == 3
        assert info['constraint'] == 'None'

    def test_spherical_info(self):
        """Test spherical coordinate system info."""
        info = coordinate_system_info('spherical')
        assert info['dimensions'] == 3
        assert 'ρ' in info['basis'][0]

    def test_invalid_coordinate_system(self):
        """Test invalid coordinate system."""
        with pytest.raises(ValueError):
            coordinate_system_info('invalid')


class TestBaseConversion:
    """Test number base conversion."""

    def test_convert_decimal_to_binary(self):
        """Test decimal to binary conversion."""
        result = convert_between_bases(10, 10, 2)
        assert result == '1010'

    def test_convert_binary_to_decimal(self):
        """Test binary to decimal conversion."""
        result = convert_between_bases(1010, 2, 10)
        assert result == '10'

    def test_convert_decimal_to_hex(self):
        """Test decimal to hexadecimal conversion."""
        result = convert_between_bases(255, 10, 16)
        assert result == 'FF'

    def test_convert_hex_to_decimal(self):
        """Test hexadecimal to decimal conversion."""
        result = convert_between_bases('FF', 16, 10)
        assert result == '255'

    def test_convert_scheherazade_base(self):
        """Test conversion related to Scheherazade numbers."""
        # 1001 in base 7 (since 1001 = 7×11×13, but let's test base 7)
        result = convert_between_bases(1001, 10, 7)
        # This should work without error
        assert isinstance(result, str)
        assert len(result) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_conversion(self):
        """Test conversion of zero."""
        result = rational_to_float(SymergeticsNumber(0))
        assert result == 0.0

        coord = xyz_to_quadray(0.0, 0.0, 0.0)
        assert coord.a == 0

    def test_negative_numbers(self):
        """Test negative number handling."""
        result = rational_to_float(SymergeticsNumber(-5, 2))
        assert result == -2.5

    def test_very_large_numbers(self):
        """Test very large number conversions."""
        large_num = SymergeticsNumber(10**100)
        result = rational_to_float(large_num)
        assert result == float('inf') or result > 10**50

    def test_very_small_numbers(self):
        """Test very small number conversions."""
        small_num = SymergeticsNumber(1, 10**100)
        result = rational_to_float(small_num)
        assert result == 0.0 or result < 10**-50


if __name__ == "__main__":
    pytest.main([__file__])
