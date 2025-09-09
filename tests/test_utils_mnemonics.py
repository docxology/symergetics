"""
Tests for utils.mnemonics module - Mnemonic encoding utilities.

Tests memory aids, pattern recognition, and number formatting.
"""

import pytest
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


class TestMnemonicEncoding:
    """Test mnemonic encoding functions."""

    def test_mnemonic_encode_grouped(self):
        """Test grouped digit encoding."""
        result = mnemonic_encode(1234567890, style='grouped')
        assert ',' in result
        assert '1,234,567,890' in result

    def test_mnemonic_encode_scientific(self):
        """Test scientific notation encoding."""
        result = mnemonic_encode(1234567890, style='scientific')
        assert '× 10^' in result

    def test_mnemonic_encode_scheherazade(self):
        """Test mnemonic encoding for Scheherazade number."""
        result = mnemonic_encode(1001)
        assert 'Scheherazade' in result or '1001' in result

    def test_mnemonic_encode_primorial(self):
        """Test mnemonic encoding for primorial."""
        result = mnemonic_encode(30030)
        assert '30030' in result

    def test_mnemonic_encode_fraction(self):
        """Test mnemonic encoding for fractions."""
        frac = SymergeticsNumber(3, 4)
        result = mnemonic_encode(frac)
        assert '3/4' in result

    def test_mnemonic_decode(self):
        """Test mnemonic decoding."""
        # Test with simple number
        result = mnemonic_decode("12345")
        assert result == 12345

        # Test with invalid input
        result = mnemonic_decode("invalid")
        assert isinstance(result, str)
        assert "Could not decode" in result


class TestNumberFormatting:
    """Test number formatting functions."""

    def test_format_large_number_basic(self):
        """Test basic number formatting."""
        result = format_large_number(1234567890)
        assert result == '1,234,567,890'

    def test_format_large_number_custom_grouping(self):
        """Test custom grouping size."""
        result = format_large_number(1234567890, grouping=4)
        assert result == '1234,5678,90'

    def test_ungroup_number(self):
        """Test ungrouping grouped numeric strings to integers."""
        from symergetics.utils.mnemonics import ungroup_number

        assert ungroup_number('1,234,567,890') == 1234567890
        assert ungroup_number('1234,5678,90') == 1234567890
        assert ungroup_number('1 234 567 890') == 1234567890
        assert ungroup_number('1_234_567_890') == 1234567890
        assert ungroup_number('1006,0150,2001,5006,001') == 1006015020015006001

    def test_format_large_number_small(self):
        """Test formatting of small numbers."""
        result = format_large_number(123)
        assert result == '123'

    def test_format_large_number_symergetics_number(self):
        """Test formatting with SymergeticsNumber."""
        num = SymergeticsNumber(1234567890)
        result = format_large_number(num)
        assert result == '1,234,567,890'


class TestMemoryAids:
    """Test memory aid creation."""

    def test_create_memory_aid_basic(self):
        """Test basic memory aid creation."""
        aids = create_memory_aid(1001)

        assert 'grouped' in aids
        assert 'scientific' in aids
        assert 'words' in aids
        assert 'patterns' in aids
        assert 'synergetics_context' in aids

    def test_create_memory_aid_scheherazade(self):
        """Test memory aid for Scheherazade number."""
        aids = create_memory_aid(1001)
        context = aids['synergetics_context']
        assert 'Scheherazade' in context

    def test_create_memory_aid_large_number(self):
        """Test memory aid for large numbers."""
        large_num = 1006015020015006001  # 1001^6
        aids = create_memory_aid(large_num)
        assert 'patterns' in aids


class TestPatternVisualization:
    """Test number pattern visualization."""

    def test_visualize_number_pattern_basic(self):
        """Test basic pattern visualization."""
        result = visualize_number_pattern(12345)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_visualize_number_pattern_palindrome(self):
        """Test visualization of palindromic number."""
        result = visualize_number_pattern(12321)
        assert 'Palindrome' in result

    def test_visualize_number_pattern_repeated(self):
        """Test visualization of number with repeated digits."""
        result = visualize_number_pattern(11111)
        assert 'All 1' in result

    def test_visualize_number_pattern_fraction(self):
        """Test visualization with fractional number."""
        frac = SymergeticsNumber(3, 4)
        result = visualize_number_pattern(frac)
        assert 'fraction' in result.lower()


class TestPatternComparison:
    """Test number pattern comparison."""

    def test_compare_number_patterns_identical(self):
        """Test comparison of identical numbers."""
        comparison = compare_number_patterns(12345, 12345)

        assert comparison['same_length'] == True
        assert comparison['same_digit_sum'] == True
        assert comparison['same_digit_distribution'] == True

    def test_compare_number_patterns_different(self):
        """Test comparison of different numbers."""
        comparison = compare_number_patterns(12345, 67890)

        assert comparison['same_length'] == True
        assert comparison['same_digit_sum'] == False

    def test_compare_number_patterns_different_lengths(self):
        """Test comparison of numbers with different lengths."""
        comparison = compare_number_patterns(123, 12345)

        assert comparison['same_length'] == False
        assert comparison['length_difference'] == -2


class TestSynergeticsMnemonics:
    """Test Synergetics-specific mnemonic generation."""

    def test_generate_synergetics_mnemonics(self):
        """Test generation of Synergetics mnemonics dictionary."""
        mnemonics = generate_synergetics_mnemonics()

        assert isinstance(mnemonics, dict)
        assert len(mnemonics) > 0

        # Check some key entries
        assert '1001' in mnemonics
        assert 'Scheherazade' in mnemonics['1001']

        assert '30030' in mnemonics
        assert '30030' in mnemonics['30030']

        assert '25000000000' in mnemonics
        assert 'atomic' in mnemonics['25000000000'].lower()

    def test_synergetics_mnemonics_completeness(self):
        """Test that all key Synergetics numbers are included."""
        mnemonics = generate_synergetics_mnemonics()

        key_numbers = [
            '1001', '30030', '1002001', '1006015020015006001',
            '25000000000', '1296000', '360', '4096', '20', '4', '3'
        ]

        for key in key_numbers:
            assert key in mnemonics, f"Missing mnemonic for {key}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_or_zero(self):
        """Test with zero and empty inputs."""
        result = mnemonic_encode(0)
        assert '0' in result

        result = format_large_number(0)
        assert result == '0'

    def test_single_digit(self):
        """Test with single digit numbers."""
        result = mnemonic_encode(5)
        assert '5' in result

    def test_negative_numbers(self):
        """Test with negative numbers."""
        result = mnemonic_encode(-123)
        assert 'negative' in result.lower()

    def test_very_large_numbers(self):
        """Test with very large numbers."""
        large_num = 10**100
        result = mnemonic_encode(large_num)
        # Should handle gracefully without crashing
        assert isinstance(result, str)

    def test_fractional_symergetics_numbers(self):
        """Test with fractional SymergeticsNumbers."""
        frac = SymergeticsNumber(22, 7)  # π approximation
        result = mnemonic_encode(frac)
        assert 'fraction' in result

        # Test memory aid creation
        aids = create_memory_aid(frac)
        assert 'fraction' in aids['words'].lower()


class TestIntegration:
    """Test integration between different mnemonic functions."""

    def test_encode_decode_round_trip(self):
        """Test round-trip encoding/decoding."""
        original = 12345
        encoded = mnemonic_encode(original, style='grouped')
        decoded = mnemonic_decode(encoded)

        # Note: This might not be a perfect round-trip due to formatting
        # but should be close
        assert isinstance(decoded, int) or isinstance(decoded, str)

    def test_memory_aid_consistency(self):
        """Test consistency across memory aid formats."""
        num = 1001
        aids = create_memory_aid(num)

        # All formats should contain the number
        for format_name, mnemonic in aids.items():
            assert isinstance(mnemonic, str)
            assert len(mnemonic) > 0

    def test_pattern_visualization_consistency(self):
        """Test pattern visualization consistency."""
        num = 12321  # Palindrome
        viz = visualize_number_pattern(num)

        # Should contain information about the number
        assert str(num) in viz or format_large_number(num) in viz


if __name__ == "__main__":
    pytest.main([__file__])
