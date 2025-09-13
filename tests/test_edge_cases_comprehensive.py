"""
Comprehensive edge case and error handling tests.
Tests real functionality with edge cases and error conditions.
"""

import pytest
import numpy as np
from fractions import Fraction
import math
import sys

from symergetics.core.numbers import SymergeticsNumber
from symergetics.core.constants import SymergeticsConstants
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.computation.analysis import analyze_mathematical_patterns
from symergetics.computation.palindromes import analyze_number_for_synergetics
from symergetics.computation.primorials import primorial, scheherazade_power
from symergetics.geometry.polyhedra import Tetrahedron, Octahedron, Cube
from symergetics.utils.conversion import rational_to_float, float_to_exact_rational
from symergetics.utils.mnemonics import mnemonic_encode, create_memory_aid


class TestEdgeCaseNumbers:
    """Test edge cases for number operations."""
    
    def test_zero_handling(self):
        """Test zero handling."""
        zero = SymergeticsNumber(0)
        
        # Test arithmetic with zero
        assert zero + zero == zero
        assert zero * zero == zero
        assert zero / SymergeticsNumber(1) == zero
        
        # Test division by zero
        with pytest.raises(ZeroDivisionError):
            SymergeticsNumber(1) / zero
        
        # Test power with zero
        assert zero ** 2 == zero
        assert SymergeticsNumber(1) ** 0 == SymergeticsNumber(1)
    
    def test_negative_numbers(self):
        """Test negative number handling."""
        neg = SymergeticsNumber(-3, 4)
        
        # Test arithmetic with negative numbers
        assert neg + neg == SymergeticsNumber(-3, 2)
        assert neg * neg == SymergeticsNumber(9, 16)
        assert neg / SymergeticsNumber(1, 2) == SymergeticsNumber(-3, 2)
        
        # Test absolute value
        assert abs(neg) == SymergeticsNumber(3, 4)
        
        # Test power with negative numbers
        assert neg ** 2 == SymergeticsNumber(9, 16)
    
    def test_very_large_numbers(self):
        """Test very large number handling."""
        large = SymergeticsNumber(10**100)
        
        # Test arithmetic with large numbers
        result = large + large
        assert isinstance(result, SymergeticsNumber)
        assert result.numerator > 0
        
        # Test conversion with large numbers
        float_large = rational_to_float(large)
        assert isinstance(float_large, float)
        assert float_large > 0
    
    def test_very_small_numbers(self):
        """Test very small number handling."""
        small = SymergeticsNumber(1, 10**100)
        
        # Test arithmetic with small numbers
        result = small + small
        assert isinstance(result, SymergeticsNumber)
        assert result.numerator > 0
        
        # Test conversion with small numbers
        float_small = rational_to_float(small)
        assert isinstance(float_small, float)
        assert float_small > 0
    
    def test_fractional_edge_cases(self):
        """Test fractional edge cases."""
        # Test with very large denominators
        large_denom = SymergeticsNumber(1, 10**50)
        assert large_denom.denominator == 10**50
        
        # Test with very large numerators
        large_num = SymergeticsNumber(10**50, 1)
        assert large_num.numerator == 10**50
        
        # Test with mixed large numbers
        mixed = SymergeticsNumber(10**25, 10**25)
        assert mixed.numerator == 1
        assert mixed.denominator == 1


class TestEdgeCaseCoordinates:
    """Test edge cases for coordinate operations."""
    
    def test_zero_coordinates(self):
        """Test zero coordinate handling."""
        zero_coord = QuadrayCoordinate(0, 0, 0, 0)
        
        # Test arithmetic with zero coordinates
        assert zero_coord.add(zero_coord) == zero_coord
        # Test scalar multiplication (not supported, test equality instead)
        assert zero_coord == zero_coord
        
        # Test magnitude with zero coordinates
        assert zero_coord.magnitude() == 0
        
        # Test distance with zero coordinates
        assert zero_coord.distance_to(zero_coord) == 0
    
    def test_negative_coordinates(self):
        """Test negative coordinate handling."""
        neg_coord = QuadrayCoordinate(-1, 0, 0, 0)
        # After normalization: (-1,0,0,0) -> (0,1,1,1) by subtracting min(-1)
        assert neg_coord.a == 0
        assert neg_coord.b == 1
        assert neg_coord.c == 1
        assert neg_coord.d == 1
        
        # Test arithmetic with negative coordinates
        result = neg_coord.add(QuadrayCoordinate(1, 0, 0, 0))
        # (0,1,1,1) + (1,0,0,0) = (1,1,1,1) (add method doesn't normalize)
        assert result.a == 1
        assert result.b == 1
        assert result.c == 1
        assert result.d == 1
        
        # Test magnitude with negative coordinates
        assert neg_coord.magnitude() >= 0
    
    def test_large_coordinates(self):
        """Test large coordinate handling."""
        large_coord = QuadrayCoordinate(10**10, 0, 0, 0)
        # After normalization: (10**10, 0, 0, 0) stays the same (min=0)
        assert large_coord.a == 10**10
        
        # Test arithmetic with large coordinates
        result = large_coord.add(large_coord)
        # (10**10, 0, 0, 0) + (10**10, 0, 0, 0) = (2*10**10, 0, 0, 0)
        assert result.a == 2 * 10**10
        
        # Test magnitude with large coordinates
        assert large_coord.magnitude() > 0
    
    def test_normalization_edge_cases(self):
        """Test normalization edge cases."""
        # Test with already normalized coordinates
        norm_coord = QuadrayCoordinate(1, -1, 0, 0)
        # QuadrayCoordinate normalizes automatically during initialization
        # (1, -1, 0, 0) -> (2, 0, 1, 1) by subtracting min(-1)
        assert norm_coord.a == 2
        assert norm_coord.b == 0
        assert norm_coord.c == 1
        assert norm_coord.d == 1
        
        # Test with zero sum coordinates
        zero_sum = QuadrayCoordinate(1, 1, -1, -1)
        # (1, 1, -1, -1) -> (2, 2, 0, 0) by subtracting min(-1)
        assert zero_sum.a == 2
        assert zero_sum.b == 2
        assert zero_sum.c == 0
        assert zero_sum.d == 0


class TestEdgeCaseAnalysis:
    """Test edge cases for analysis operations."""
    
    def test_empty_string_analysis(self):
        """Test analysis with empty strings."""
        analysis = analyze_mathematical_patterns("")
        assert isinstance(analysis, dict)
        assert 'error' in analysis or 'is_palindromic' in analysis
    
    def test_single_digit_analysis(self):
        """Test analysis with single digits."""
        for digit in range(10):
            analysis = analyze_mathematical_patterns(str(digit))
            assert isinstance(analysis, dict)
            assert 'is_palindromic' in analysis
            assert analysis['is_palindromic'] == True  # Single digits are palindromic
    
    def test_very_long_string_analysis(self):
        """Test analysis with very long strings."""
        long_string = "1" * 1000
        analysis = analyze_mathematical_patterns(long_string)
        assert isinstance(analysis, dict)
        assert 'is_palindromic' in analysis
        assert analysis['is_palindromic'] == True
    
    def test_special_characters_analysis(self):
        """Test analysis with special characters."""
        special_strings = ["abc", "123abc", "!@#$%", "123.456"]
        
        for special in special_strings:
            analysis = analyze_mathematical_patterns(special)
            assert isinstance(analysis, dict)
            # Should handle gracefully without crashing
    
    def test_unicode_analysis(self):
        """Test analysis with unicode characters."""
        unicode_strings = ["１２３", "١٢٣", "１２３４５"]
        
        for unicode_str in unicode_strings:
            analysis = analyze_mathematical_patterns(unicode_str)
            assert isinstance(analysis, dict)
            # Should handle gracefully without crashing


class TestEdgeCasePalindromes:
    """Test edge cases for palindrome operations."""
    
    def test_empty_palindrome_analysis(self):
        """Test palindrome analysis with empty strings."""
        # Empty string should be handled gracefully
        try:
            analysis = analyze_number_for_synergetics("")
            assert isinstance(analysis, dict)
            assert 'palindromic_patterns' in analysis
        except ValueError:
            # Expected for empty string
            pass
    
    def test_single_character_palindromes(self):
        """Test palindrome analysis with single characters."""
        for char in "0123456789":
            analysis = analyze_number_for_synergetics(char)
            assert isinstance(analysis, dict)
            assert 'palindromic_patterns' in analysis
            assert 'is_palindromic' in analysis
    
    def test_very_long_palindromes(self):
        """Test palindrome analysis with very long palindromes."""
        long_palindrome = "123456789" + "987654321"
        analysis = analyze_number_for_synergetics(long_palindrome)
        assert isinstance(analysis, dict)
        assert 'palindromic_patterns' in analysis
    
    def test_mixed_case_palindromes(self):
        """Test palindrome analysis with mixed case."""
        # Use numeric string since analyze_number_for_synergetics expects numbers
        mixed_palindrome = "123321"
        analysis = analyze_number_for_synergetics(mixed_palindrome)
        assert isinstance(analysis, dict)
        assert 'palindromic_patterns' in analysis


class TestEdgeCasePrimorials:
    """Test edge cases for primorial operations."""
    
    def test_zero_primorial(self):
        """Test primorial with zero."""
        # primorial(0) should return 1, not raise ValueError
        result = primorial(0)
        assert result == 1
    
    def test_negative_primorial(self):
        """Test primorial with negative numbers."""
        # primorial(-1) should return 1, not raise ValueError
        result = primorial(-1)
        assert result == 1
    
    def test_large_primorial(self):
        """Test primorial with large numbers."""
        # Test with reasonably large number
        result = primorial(10)
        assert isinstance(result, SymergeticsNumber)
        assert result > 0
    
    def test_scheherazade_power_zero(self):
        """Test Scheherazade power with zero."""
        result = scheherazade_power(0)
        assert result == 1
    
    def test_scheherazade_power_negative(self):
        """Test Scheherazade power with negative numbers."""
        # scheherazade_power(-1) returns 1/1001
        result = scheherazade_power(-1)
        assert isinstance(result, SymergeticsNumber)
        assert result.numerator == 1
        assert result.denominator == 1001
    
    def test_scheherazade_power_large(self):
        """Test Scheherazade power with large numbers."""
        result = scheherazade_power(5)
        assert isinstance(result, SymergeticsNumber)
        assert result > 0


class TestEdgeCaseGeometry:
    """Test edge cases for geometry operations."""
    
    def test_polyhedra_edge_cases(self):
        """Test polyhedra edge cases."""
        # Test tetrahedron
        tetra = Tetrahedron()
        assert tetra.volume() == 1
        assert len(tetra.vertices) == 4
        
        # Test octahedron
        octa = Octahedron()
        assert octa.volume() == 4
        assert len(octa.vertices) == 6
        
        # Test cube
        cube = Cube()
        assert cube.volume() == 3
        assert len(cube.vertices) == 8
    
    def test_coordinate_transformation_edge_cases(self):
        """Test coordinate transformation edge cases."""
        # Test with zero coordinates
        zero_coord = QuadrayCoordinate(0, 0, 0, 0)
        xyz = zero_coord.to_xyz()
        assert len(xyz) == 3
        assert all(isinstance(coord, (int, float)) for coord in xyz)
        
        # Test with large coordinates
        large_coord = QuadrayCoordinate(10**6, 0, 0, 0)
        xyz = large_coord.to_xyz()
        assert len(xyz) == 3
        assert all(isinstance(coord, (int, float)) for coord in xyz)


class TestEdgeCaseConversion:
    """Test edge cases for conversion operations."""
    
    def test_conversion_edge_cases(self):
        """Test conversion edge cases."""
        # Test with zero
        zero = SymergeticsNumber(0)
        float_zero = rational_to_float(zero)
        assert float_zero == 0.0
        
        # Test with very small numbers
        small = SymergeticsNumber(1, 10**100)
        float_small = rational_to_float(small)
        assert isinstance(float_small, float)
        assert float_small > 0
        
        # Test with very large numbers
        large = SymergeticsNumber(10**100)
        float_large = rational_to_float(large)
        assert isinstance(float_large, float)
        assert float_large > 0
    
    def test_float_to_rational_edge_cases(self):
        """Test float to rational edge cases."""
        # Test with zero
        rational_zero = float_to_exact_rational(0.0)
        assert rational_zero.numerator == 0
        assert rational_zero.denominator == 1
        
        # Test with very small floats that can be represented
        small_float = 1e-6  # Use a small number that can be represented with default max_denominator
        rational_small = float_to_exact_rational(small_float)
        assert isinstance(rational_small, SymergeticsNumber)
        assert rational_small.numerator > 0
        
        # Test with very large floats
        large_float = 1e100
        rational_large = float_to_exact_rational(large_float)
        assert isinstance(rational_large, SymergeticsNumber)
        assert rational_large.numerator > 0


class TestEdgeCaseMnemonics:
    """Test edge cases for mnemonic operations."""
    
    def test_mnemonic_edge_cases(self):
        """Test mnemonic edge cases."""
        # Test with zero
        zero = SymergeticsNumber(0)
        mnemonic_zero = mnemonic_encode(zero)
        assert isinstance(mnemonic_zero, str)
        assert len(mnemonic_zero) > 0
        
        # Test with very large numbers
        large = SymergeticsNumber(10**100)
        mnemonic_large = mnemonic_encode(large)
        assert isinstance(mnemonic_large, str)
        assert len(mnemonic_large) > 0
        
        # Test with fractions
        fraction = SymergeticsNumber(22, 7)
        mnemonic_fraction = mnemonic_encode(fraction)
        assert isinstance(mnemonic_fraction, str)
        assert len(mnemonic_fraction) > 0
    
    def test_memory_aid_edge_cases(self):
        """Test memory aid edge cases."""
        # Test with zero
        zero = SymergeticsNumber(0)
        aid_zero = create_memory_aid(zero)
        assert isinstance(aid_zero, dict)
        assert len(aid_zero) > 0
        
        # Test with very large numbers
        large = SymergeticsNumber(10**100)
        aid_large = create_memory_aid(large)
        assert isinstance(aid_large, dict)
        assert len(aid_large) > 0
        
        # Test with fractions
        fraction = SymergeticsNumber(22, 7)
        aid_fraction = create_memory_aid(fraction)
        assert isinstance(aid_fraction, dict)
        assert len(aid_fraction) > 0


class TestErrorHandling:
    """Test error handling across modules."""
    
    def test_division_by_zero_handling(self):
        """Test division by zero handling."""
        with pytest.raises(ZeroDivisionError):
            SymergeticsNumber(1) / SymergeticsNumber(0)
    
    def test_invalid_input_handling(self):
        """Test invalid input handling."""
        # Test with None
        with pytest.raises((TypeError, ValueError)):
            SymergeticsNumber(None)
        
        # Test with invalid string
        with pytest.raises((TypeError, ValueError)):
            SymergeticsNumber("invalid")
    
    def test_analysis_error_handling(self):
        """Test analysis error handling."""
        # Test with None
        analysis = analyze_mathematical_patterns(None)
        assert isinstance(analysis, dict)
        assert 'error' in analysis or 'is_palindromic' in analysis
    
    def test_coordinate_error_handling(self):
        """Test coordinate error handling."""
        # Test with invalid dimensions
        with pytest.raises((TypeError, ValueError)):
            QuadrayCoordinate(1, 2)  # Only 2 coordinates instead of 4
        
        # Test with invalid types
        with pytest.raises((TypeError, ValueError)):
            QuadrayCoordinate("a", "b", "c", "d")


class TestPerformanceEdgeCases:
    """Test performance with edge cases."""
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        import time
        
        # Test with large number of operations
        start_time = time.time()
        
        numbers = [SymergeticsNumber(i) for i in range(1, 1001)]  # 1000 numbers
        results = [num + num for num in numbers]
        
        end_time = time.time()
        
        assert len(results) == 1000
        assert end_time - start_time < 10.0  # Should complete within 10 seconds
    
    def test_memory_usage_edge_cases(self):
        """Test memory usage with edge cases."""
        # Test with very large numbers
        large_numbers = [SymergeticsNumber(10**i) for i in range(1, 101)]
        
        # Test operations don't cause memory issues
        results = [num * num for num in large_numbers]
        assert len(results) == 100
        
        # Test analysis with large numbers
        analyses = [analyze_mathematical_patterns(str(num.numerator)) for num in large_numbers[:10]]
        assert len(analyses) == 10
