"""
Extended tests for utils modules - Additional coverage for utility functions.

Tests advanced conversion scenarios, mnemonic operations, and edge cases.
"""

import pytest
import math
from symergetics.utils.conversion import (
    rational_to_float,
    float_to_exact_rational,
    decimal_to_fraction,
    scientific_notation_to_rational,
    continued_fraction_approximation,
    convergents_from_continued_fraction,
    best_rational_approximation,
    format_as_mixed_number,
    coordinate_system_info,
    convert_between_bases
)
from symergetics.utils.mnemonics import (
    mnemonic_encode,
    mnemonic_decode,
    format_large_number,
    create_memory_aid,
    visualize_number_pattern,
    compare_number_patterns,
    generate_synergetics_mnemonics
)
from symergetics.core.numbers import SymergeticsNumber
from fractions import Fraction


class TestConversionExtended:
    """Extended tests for conversion utilities."""

    def test_rational_to_float_precision(self):
        """Test rational to float with extreme precision."""
        # Very small rational
        small_rational = SymergeticsNumber(1, 10**6)
        result = rational_to_float(small_rational, precision=10)
        assert abs(result - 1e-6) < 1e-10

    def test_float_to_rational_extreme_values(self):
        """Test float to rational with extreme values."""
        # Very small float
        small_float = 1e-10
        result = float_to_exact_rational(small_float)
        # Small floats have limited precision in rational approximation
        assert abs(float(result.value) - small_float) < 1e-9

        # Very large float
        large_float = 1e10
        result = float_to_exact_rational(large_float)
        assert abs(float(result.value) - large_float) < 1e-6

    def test_decimal_to_fraction_complex(self):
        """Test decimal to fraction with complex decimals."""
        # Repeating decimal
        result = decimal_to_fraction("0.3333333333")
        expected = SymergeticsNumber(1, 3)
        assert abs(float(result.value) - float(expected.value)) < 1e-6

    def test_scientific_notation_edge_cases(self):
        """Test scientific notation with edge cases."""
        # Very small
        result = scientific_notation_to_rational("1e-100")
        assert float(result.value) < 1e-99

        # Very large
        result = scientific_notation_to_rational("1e100")
        assert float(result.value) > 1e99

        # Negative exponent
        result = scientific_notation_to_rational("-2.5e-3")
        assert abs(float(result.value) - (-0.0025)) < 1e-6

    def test_continued_fraction_pi(self):
        """Test continued fraction for π."""
        terms = continued_fraction_approximation(math.pi, 10)
        assert len(terms) == 10
        assert terms[0] == 3  # π ≈ 3 + ...

    def test_convergents_pi(self):
        """Test convergents for π."""
        terms = [3, 7, 15, 1]
        convergents = convergents_from_continued_fraction(terms)

        assert len(convergents) == 4
        # Test that convergents get closer to π
        pi_approx = float(convergents[-1][0]) / convergents[-1][1]
        assert abs(pi_approx - math.pi) < 0.01

    def test_best_rational_phi(self):
        """Test best rational approximation for golden ratio."""
        phi = (1 + math.sqrt(5)) / 2
        approx = best_rational_approximation(phi, max_denominator=100)

        assert abs(float(approx.value) - phi) < 0.01

    def test_format_mixed_number_edge_cases(self):
        """Test mixed number formatting edge cases."""
        # Whole number
        result = format_as_mixed_number(SymergeticsNumber(5))
        assert result == "5"

        # Negative fraction
        result = format_as_mixed_number(SymergeticsNumber(-7, 3))
        assert "2 1/3" in result or "-2 1/3" in result

        # Very small fraction
        result = format_as_mixed_number(SymergeticsNumber(1, 100))
        assert "1/100" in result

    def test_coordinate_system_info_all(self):
        """Test all coordinate system information."""
        systems = ['quadray', 'cartesian', 'spherical']

        for system in systems:
            info = coordinate_system_info(system)
            assert 'description' in info
            assert 'dimensions' in info
            assert 'basis' in info

    def test_base_conversion_comprehensive(self):
        """Test base conversion comprehensively."""
        # Binary to decimal
        result = convert_between_bases(111, 2, 10)
        assert result == '7'

        # Decimal to hexadecimal
        result = convert_between_bases(255, 10, 16)
        assert result.upper() == 'FF'

        # Hexadecimal to decimal
        result = convert_between_bases('A', 16, 10)
        assert result == '10'


class TestMnemonicsExtended:
    """Extended tests for mnemonic utilities."""

    def test_mnemonic_encode_special_cases(self):
        """Test mnemonic encoding for special cases."""
        # Zero
        result = mnemonic_encode(0)
        assert '0' in result

        # Negative numbers
        result = mnemonic_encode(-42)
        assert 'negative' in result.lower()

        # Very large numbers
        large_num = 10**20
        result = mnemonic_encode(large_num)
        assert isinstance(result, str)

    def test_mnemonic_decode_edge_cases(self):
        """Test mnemonic decoding edge cases."""
        # Invalid input
        result = mnemonic_decode("not a number")
        assert isinstance(result, str)

        # Empty string
        result = mnemonic_decode("")
        assert isinstance(result, str)

    def test_format_large_number_edge_cases(self):
        """Test large number formatting edge cases."""
        # Very large number
        large = 10**100
        result = format_large_number(large)
        assert len(result) > 50  # Should handle large numbers

        # Custom grouping
        result = format_large_number(123456789, grouping=2)
        assert ',' in result

    def test_create_memory_aid_comprehensive(self):
        """Test comprehensive memory aid creation."""
        num = 12345
        aids = create_memory_aid(num)

        expected_styles = ['grouped', 'scientific', 'words', 'patterns', 'synergetics_context']
        for style in expected_styles:
            assert style in aids
            assert isinstance(aids[style], str)
            assert len(aids[style]) > 0

    def test_visualize_pattern_special_cases(self):
        """Test pattern visualization for special cases."""
        # Palindrome
        result = visualize_number_pattern(12321)
        assert isinstance(result, str)

        # All same digits
        result = visualize_number_pattern(11111)
        assert isinstance(result, str)

        # SymergeticsNumber
        frac = SymergeticsNumber(3, 4)
        result = visualize_number_pattern(frac)
        assert isinstance(result, str)

    def test_compare_patterns_edge_cases(self):
        """Test pattern comparison edge cases."""
        # Same numbers
        comparison = compare_number_patterns(123, 123)
        assert comparison['same_length'] == True
        assert comparison['same_digit_sum'] == True

        # Different lengths
        comparison = compare_number_patterns(123, 12345)
        assert comparison['same_length'] == False

    def test_synergetics_mnemonics_all_keys(self):
        """Test that all Synergetics keys are present."""
        mnemonics = generate_synergetics_mnemonics()

        key_numbers = [
            '1001', '30030', '1002001', '1006015020015006001',
            '25000000000', '1296000', '360', '4096', '20', '4', '3'
        ]

        for key in key_numbers:
            assert key in mnemonics
            assert len(mnemonics[key]) > 0


class TestIntegrationExtended:
    """Test integration between conversion and mnemonic utilities."""

    def test_conversion_to_mnemonic(self):
        """Test converting numbers and then creating mnemonics."""
        # Convert scientific notation to rational
        rational = scientific_notation_to_rational("1.23e-4")

        # Create mnemonic
        mnemonic = mnemonic_encode(rational.value.numerator)

        assert isinstance(mnemonic, str)
        assert len(mnemonic) > 0

    def test_mnemonic_with_converted_number(self):
        """Test mnemonic creation with converted numbers."""
        # Convert decimal to fraction
        frac = decimal_to_fraction("0.142857")  # 1/7

        # Create memory aid
        aid = create_memory_aid(frac.value.numerator)

        assert 'words' in aid
        assert isinstance(aid['words'], str)

    def test_pattern_analysis_with_conversion(self):
        """Test pattern analysis with converted numbers."""
        # Create a number through conversion
        rational = best_rational_approximation(math.pi, max_denominator=100)
        num = rational.value.numerator

        # Analyze patterns
        from symergetics.utils.mnemonics import visualize_number_pattern
        pattern = visualize_number_pattern(num)

        assert isinstance(pattern, str)

    def test_base_conversion_with_mnemonics(self):
        """Test base conversion integrated with mnemonics."""
        # Convert hex to decimal
        decimal_str = convert_between_bases('FF', 16, 10)
        decimal_num = int(decimal_str)

        # Create mnemonic
        mnemonic = mnemonic_encode(decimal_num)

        assert '255' in mnemonic or 'FF' in mnemonic.upper()


class TestPerformanceExtended:
    """Test performance aspects of utility functions."""

    def test_bulk_conversions(self):
        """Test bulk conversion operations."""
        numbers = [f"0.{i}" for i in range(10, 100, 10)]

        fractions = []
        for num_str in numbers:
            frac = decimal_to_fraction(num_str)
            fractions.append(frac)

        assert len(fractions) == len(numbers)
        assert all(isinstance(f, SymergeticsNumber) for f in fractions)

    def test_bulk_mnemonic_operations(self):
        """Test bulk mnemonic operations."""
        numbers = list(range(100, 200, 10))

        mnemonics = []
        for num in numbers:
            mnemonic = mnemonic_encode(num)
            mnemonics.append(mnemonic)

        assert len(mnemonics) == len(numbers)
        assert all(isinstance(m, str) for m in mnemonics)

    def test_memory_aid_bulk(self):
        """Test bulk memory aid creation."""
        numbers = [10**i for i in range(1, 6)]

        for num in numbers:
            aids = create_memory_aid(num)
            assert len(aids) > 0


class TestErrorHandlingExtended:
    """Test error handling in utility functions."""

    def test_conversion_error_handling(self):
        """Test error handling in conversion functions."""
        # Invalid decimal
        with pytest.raises(ValueError):
            decimal_to_fraction("invalid")

        # Invalid scientific notation
        with pytest.raises(ValueError):
            scientific_notation_to_rational("invalid")

        # Invalid coordinate system
        with pytest.raises(ValueError):
            coordinate_system_info("invalid")

    def test_mnemonic_error_handling(self):
        """Test error handling in mnemonic functions."""
        # Should handle gracefully
        result = mnemonic_decode("not a number")
        assert isinstance(result, str)

    def test_pattern_analysis_error_handling(self):
        """Test error handling in pattern analysis."""
        # Should handle edge cases gracefully
        result = visualize_number_pattern(0)
        assert isinstance(result, str)

        result = compare_number_patterns(0, 1)
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__])
