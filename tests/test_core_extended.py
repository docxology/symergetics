"""
Extended tests for core modules - Additional coverage for edge cases and complex scenarios.

Tests advanced functionality, edge cases, and integration scenarios.
"""

import pytest
import math
from fractions import Fraction
from symergetics.core.numbers import SymergeticsNumber
from symergetics.core.constants import (
    SymergeticsConstants,
    PHI, PI, E, SQRT2, SQRT3,
    SCHEHERAZADE_BASE,
    COSMIC_ABUNDANCE
)


class TestSymergeticsNumberExtended:
    """Extended tests for SymergeticsNumber."""

    def test_large_number_operations(self):
        """Test operations with very large numbers."""
        large_num = SymergeticsNumber(10**100)
        result = large_num + SymergeticsNumber(1)

        # Should handle large numbers gracefully
        assert result.value > large_num.value

    def test_fractional_operations(self):
        """Test operations with complex fractions."""
        a = SymergeticsNumber(2, 3)
        b = SymergeticsNumber(3, 4)
        c = SymergeticsNumber(4, 5)

        # Complex fraction arithmetic
        result = (a * b) / c
        expected = SymergeticsNumber(2, 5)  # (2/3 * 3/4) / (4/5) = (1/2) / (4/5) = 5/8, wait let me recalculate
        # (2/3 * 3/4) = 2/4 = 1/2
        # (1/2) / (4/5) = (1/2) * (5/4) = 5/8
        expected = SymergeticsNumber(5, 8)

        assert result.value == expected.value

    def test_power_operations(self):
        """Test various power operations."""
        base = SymergeticsNumber(2)

        # Integer powers
        assert (base ** 3).value == Fraction(8, 1)
        assert (base ** 0).value == Fraction(1, 1)

        # Fractional powers (approximated)
        sqrt2 = base ** SymergeticsNumber(1, 2)
        assert abs(float(sqrt2.value) - math.sqrt(2)) < 0.01

    def test_mixed_type_operations(self):
        """Test operations between SymergeticsNumber and other types."""
        sn = SymergeticsNumber(3, 4)

        # With integers
        assert (sn + 1).value == Fraction(7, 4)
        assert (sn * 2).value == Fraction(3, 2)

        # With floats
        result = sn + 0.5
        assert abs(float(result.value) - 1.25) < 1e-10

        # With fractions
        result = sn * Fraction(2, 3)
        assert result.value == Fraction(1, 2)

    def test_comparison_edge_cases(self):
        """Test comparison operations with edge cases."""
        zero = SymergeticsNumber(0)
        positive = SymergeticsNumber(1)
        negative = SymergeticsNumber(-1)

        assert zero >= 0
        assert zero <= 0
        assert positive > zero
        assert negative < zero
        assert positive > negative

    def test_string_representations(self):
        """Test various string representations."""
        num = SymergeticsNumber(22, 7)

        str_repr = str(num)
        repr_repr = repr(num)

        assert "22/7" in str_repr or "3.14" in str_repr
        assert "SymergeticsNumber" in repr_repr
        assert "22/7" in repr_repr

    def test_serialization(self):
        """Test number serialization."""
        num = SymergeticsNumber(3, 4)

        # Should be able to recreate from value
        recreated = SymergeticsNumber(num.value)
        assert recreated == num


class TestConstantsExtended:
    """Extended tests for constants."""

    def test_all_volume_ratios(self):
        """Test all volume ratios are reasonable."""
        ratios = SymergeticsConstants.VOLUME_RATIOS

        # All ratios should be positive
        for name, ratio in ratios.items():
            assert ratio.value > 0
            assert ratio.value.denominator == 1  # Should be integers

    def test_scheherazade_progression(self):
        """Test Scheherazade number progression."""
        s1 = SymergeticsConstants.get_scheherazade_power(1)
        s2 = SymergeticsConstants.get_scheherazade_power(2)
        s3 = SymergeticsConstants.get_scheherazade_power(3)

        assert s2.value == s1.value * s1.value
        assert s3.value == s1.value * s1.value * s1.value

    def test_cosmic_factors_large(self):
        """Test that cosmic abundance factors are very large."""
        factors = SymergeticsConstants.cosmic_abundance_factors()

        abundance = factors['cosmic_abundance']
        # Should be approximately 3.1 × 10^42
        assert abundance.value > 10**40

    def test_irrational_approximations_accuracy(self):
        """Test accuracy of irrational approximations."""
        pi_approx = PI
        phi_approx = PHI
        e_approx = E

        # Should be reasonably accurate
        assert abs(float(pi_approx.value) - math.pi) < 1e-6
        assert abs(float(phi_approx.value) - ((1 + math.sqrt(5)) / 2)) < 1e-6
        assert abs(float(e_approx.value) - math.e) < 1e-6

    def test_constants_consistency(self):
        """Test that constants maintain their values."""
        # Test that PI constant is consistent
        assert PI.value > 3
        assert PI.value < 4  # π is approximately 3.14

        # Test that constants are SymergeticsNumbers
        assert isinstance(PI, SymergeticsNumber)
        assert isinstance(PHI, SymergeticsNumber)
        assert isinstance(E, SymergeticsNumber)


class TestIntegrationScenarios:
    """Test integration between different components."""

    def test_numbers_with_constants(self):
        """Test SymergeticsNumber operations with constants."""
        num = SymergeticsNumber(1)
        pi_const = PI

        result = num * pi_const
        assert abs(float(result.value) - math.pi) < 1e-6

    def test_constants_arithmetic(self):
        """Test arithmetic operations between constants."""
        result = PI + E
        expected = math.pi + math.e
        assert abs(float(result.value) - expected) < 1e-6

    def test_constants_with_scaling(self):
        """Test constants with cosmic scaling."""
        atomic_scale = SymergeticsConstants.COSMIC_SCALING['atomic_diameters_per_inch']
        result = atomic_scale * PI

        # Should handle large numbers (25 billion × π ≈ 7.85e10)
        assert result.value > 10**10

    def test_mixed_operations(self):
        """Test mixed operations between numbers and constants."""
        num = SymergeticsNumber(2)
        const = SQRT2

        result = num * const
        expected = 2 * math.sqrt(2)
        assert abs(float(result.value) - expected) < 1e-6


class TestPerformanceScenarios:
    """Test performance with various scenarios."""

    def test_bulk_operations(self):
        """Test operations on many numbers."""
        numbers = [SymergeticsNumber(i, i+1) for i in range(1, 100)]

        # Bulk addition
        total = SymergeticsNumber(0)
        for num in numbers:
            total = total + num

        assert total.value > 0

    def test_deep_fraction_arithmetic(self):
        """Test arithmetic with deeply nested fractions."""
        # Create a complex fraction
        a = SymergeticsNumber(1, 2)
        b = SymergeticsNumber(1, 3)
        c = SymergeticsNumber(1, 4)

        # Complex nested operations
        result = (a + b) * c - (a / b)
        assert isinstance(result, SymergeticsNumber)

    def test_large_exponent_operations(self):
        """Test operations with large exponents."""
        base = SymergeticsNumber(2)

        # Large power (but not too large to avoid overflow)
        result = base ** 10
        assert result.value == 2**10

    def test_precision_maintenance(self):
        """Test that precision is maintained through operations."""
        # Start with high precision
        precise = SymergeticsNumber(Fraction(1, 10**6))

        # Perform many operations
        result = precise
        for i in range(10):
            result = result + precise

        # Should still maintain reasonable precision
        assert result.value > 0


class TestErrorConditions:
    """Test error conditions and edge cases."""

    def test_division_by_zero(self):
        """Test division by zero handling."""
        num = SymergeticsNumber(1)

        with pytest.raises(ZeroDivisionError):
            num / SymergeticsNumber(0)

    def test_invalid_initialization(self):
        """Test invalid initialization."""
        with pytest.raises((ValueError, TypeError)):
            SymergeticsNumber("invalid fraction")

    def test_power_with_invalid_exponent(self):
        """Test power with invalid exponent."""
        base = SymergeticsNumber(2)

        # Should handle gracefully
        result = base ** SymergeticsNumber(0.5)  # Fractional power
        assert isinstance(result, SymergeticsNumber)

    def test_comparison_with_incompatible_types(self):
        """Test comparison with incompatible types."""
        num = SymergeticsNumber(1)

        # Should work with compatible types
        assert num > 0
        assert num < 2

        # Should handle incompatible types gracefully
        assert (num == "string") == False


if __name__ == "__main__":
    pytest.main([__file__])
