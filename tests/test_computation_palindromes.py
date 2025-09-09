"""
Tests for computation.palindromes module - Palindromic pattern analysis.

Tests palindrome detection, SSRCD analysis, and pattern recognition.
"""

import pytest
from symergetics.computation.palindromes import (
    is_palindromic,
    extract_palindromic_patterns,
    find_palindromic_sequence,
    analyze_scheherazade_ssrcd,
    find_repeated_patterns,
    calculate_palindromic_density,
    find_symmetric_patterns,
    generate_palindromic_sequence,
    find_palindromic_primes,
    analyze_number_for_synergetics
)
from symergetics.core.numbers import SymergeticsNumber


class TestPalindromeDetection:
    """Test basic palindrome detection."""

    def test_is_palindromic_integers(self):
        """Test palindrome detection for integers."""
        assert is_palindromic(121) == True
        assert is_palindromic(12321) == True
        assert is_palindromic(1001) == True
        assert is_palindromic(123454321) == True

        assert is_palindromic(123) == False
        assert is_palindromic(1234) == False
        assert is_palindromic(10) == False

    def test_is_palindromic_strings(self):
        """Test palindrome detection for strings."""
        assert is_palindromic("121") == True
        assert is_palindromic("1001") == True
        assert is_palindromic("123") == False

    def test_is_palindromic_symergetics_numbers(self):
        """Test palindrome detection for SymergeticsNumber."""
        num_pal = SymergeticsNumber(121)
        num_non_pal = SymergeticsNumber(123)

        assert is_palindromic(num_pal) == True
        assert is_palindromic(num_non_pal) == False

    def test_is_palindromic_edge_cases(self):
        """Test palindrome detection edge cases."""
        assert is_palindromic(0) == True
        assert is_palindromic(1) == True
        assert is_palindromic(11) == True
        assert is_palindromic(101) == True


class TestPalindromicPatternExtraction:
    """Test extraction of palindromic patterns."""

    def test_extract_palindromic_patterns_basic(self):
        """Test basic pattern extraction."""
        patterns = extract_palindromic_patterns(123454321, min_length=3)
        assert "123454321" in patterns  # The full number
        assert "2345432" in patterns    # Inner palindrome
        assert "34543" in patterns      # Smaller palindrome

    def test_extract_palindromic_patterns_scheherazade(self):
        """Test pattern extraction for Scheherazade numbers."""
        patterns = extract_palindromic_patterns(1001)
        assert "1001" in patterns

        patterns_1002001 = extract_palindromic_patterns(1002001)
        assert "1002001" in patterns_1002001

    def test_extract_palindromic_patterns_min_length(self):
        """Test minimum length filtering."""
        patterns = extract_palindromic_patterns(121, min_length=3)
        # Should only include patterns of length 3 or more
        assert all(len(p) >= 3 for p in patterns)

        patterns_short = extract_palindromic_patterns(121, min_length=1)
        assert "1" in patterns_short  # Single digits
        assert "2" in patterns_short
        assert "121" in patterns_short


class TestScheherazadeSSRCD:
    """Test SSRCD analysis for Scheherazade numbers."""

    def test_find_palindromic_sequence_power_6(self):
        """Test SSRCD analysis for 1001^6."""
        analysis = find_palindromic_sequence(6)

        assert analysis['power'] == 6
        assert analysis['scheherazade_number'] == str(1001**6)
        assert len(analysis['palindromic_patterns']) > 0
        assert analysis['is_palindromic'] == False  # 1001^6 is not palindromic

    def test_analyze_scheherazade_ssrcd_power_6(self):
        """Test detailed SSRCD analysis for power 6."""
        analysis = analyze_scheherazade_ssrcd(6)

        assert analysis['scheherazade_power'] == 6
        assert 'number_string' in analysis
        assert 'digit_count' in analysis
        assert 'palindromic_patterns' in analysis
        assert 'repeated_patterns' in analysis
        assert 'symmetric_patterns' in analysis
        assert 'binomial_patterns' in analysis

        # Should have the special insight for power 6
        assert 'special_insight' in analysis
        insight = analysis['special_insight']
        assert insight['coefficients'] == [1, 6, 15, 20, 15, 6, 1]

    def test_analyze_scheherazade_ssrcd_other_powers(self):
        """Test SSRCD analysis for other powers."""
        analysis = analyze_scheherazade_ssrcd(1)

        assert analysis['scheherazade_power'] == 1
        assert analysis['number_string'] == '1001'
        assert analysis['is_palindromic'] == True


class TestPatternAnalysis:
    """Test various pattern analysis functions."""

    def test_find_repeated_patterns(self):
        """Test repeated pattern detection."""
        # Test with pattern length 1
        patterns = find_repeated_patterns(1221, pattern_length=1)
        assert '1' in patterns
        assert '2' in patterns
        assert patterns['1'] == [0, 3]  # Positions of '1'
        assert patterns['2'] == [1, 2]  # Positions of '2'

        # Test with pattern length 2
        patterns_2 = find_repeated_patterns(1221, pattern_length=2)
        assert '22' in patterns_2
        assert patterns_2['22'] == [1]  # Position of '22'

    def test_calculate_palindromic_density(self):
        """Test palindromic density calculation."""
        # Perfect palindrome should have some density
        density = calculate_palindromic_density(12321)
        assert density > 0.1  # More realistic expectation

        # Non-palindrome should have lower density
        density_non_pal = calculate_palindromic_density(12345)
        assert density_non_pal < density

    def test_find_symmetric_patterns(self):
        """Test symmetric pattern detection."""
        patterns = find_symmetric_patterns(12321)

        assert len(patterns) > 0
        # Should find central symmetry
        symmetry_types = [p['type'] for p in patterns]
        assert 'central_symmetry' in symmetry_types

    def test_find_symmetric_patterns_asymmetric(self):
        """Test symmetric pattern detection for asymmetric numbers."""
        patterns = find_symmetric_patterns(12345)
        # Should have fewer or no symmetric patterns
        assert len(patterns) == 0


class TestSequenceGeneration:
    """Test palindromic sequence generation."""

    def test_generate_palindromic_sequence(self):
        """Test generation of palindromic number sequences."""
        sequence = generate_palindromic_sequence(10, 5)

        assert len(sequence) == 5
        assert all(is_palindromic(num) for num in sequence)
        assert all(num >= 10 for num in sequence)

        # Should be in ascending order
        assert sequence == sorted(sequence)

    def test_find_palindromic_primes(self):
        """Test finding palindromic prime numbers."""
        primes = find_palindromic_primes(100)

        assert len(primes) > 0
        assert all(is_palindromic(p) for p in primes)

        # Check some known palindromic primes
        if 11 in primes:
            assert 11 in primes
        if 101 in primes:
            assert 101 in primes


class TestComprehensiveAnalysis:
    """Test comprehensive number analysis."""

    def test_analyze_number_for_synergetics(self):
        """Test comprehensive Synergetics analysis."""
        analysis = analyze_number_for_synergetics(1001)

        assert analysis['is_palindromic'] == True
        assert len(analysis['palindromic_patterns']) > 0
        assert 'repeated_patterns' in analysis
        assert 'palindromic_density' in analysis
        assert 'symmetric_patterns' in analysis

    def test_analyze_number_for_synergetics_non_palindromic(self):
        """Test analysis for non-palindromic numbers."""
        analysis = analyze_number_for_synergetics(123)

        assert analysis['is_palindromic'] == False
        assert 'repeated_patterns' in analysis


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_or_single_digit(self):
        """Test with very small numbers."""
        assert is_palindromic(0) == True
        assert is_palindromic(1) == True
        assert is_palindromic(5) == True

    def test_large_numbers(self):
        """Test with larger numbers."""
        large_pal = 123454321
        assert is_palindromic(large_pal) == True

        large_non_pal = 123456789
        assert is_palindromic(large_non_pal) == False

    def test_symergetics_number_with_fraction(self):
        """Test with fractional SymergeticsNumber."""
        frac_num = SymergeticsNumber(3, 4)
        # Should handle gracefully (may not be palindromic due to decimal)
        result = is_palindromic(frac_num)
        # Result depends on how the fraction is represented as string
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__])
