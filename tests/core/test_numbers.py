"""
Comprehensive tests for SymergeticsNumber class and core number functionality.
"""

import pytest
import math
from fractions import Fraction
from symergetics.core.numbers import SymergeticsNumber
from symergetics.core.constants import SymergeticsConstants


class TestSymergeticsNumberBasic:
    """Test basic SymergeticsNumber functionality."""
    
    def test_initialization_with_integers(self):
        """Test initialization with various integer types."""
        # Positive integers
        num1 = SymergeticsNumber(42)
        assert num1.numerator == 42
        assert num1.denominator == 1
        assert num1 == 42
        
        # Negative integers
        num2 = SymergeticsNumber(-17)
        assert num2.numerator == -17
        assert num2.denominator == 1
        assert num2 == -17
        
        # Zero
        num3 = SymergeticsNumber(0)
        assert num3.numerator == 0
        assert num3.denominator == 1
        assert num3 == 0
    
    def test_initialization_with_fractions(self):
        """Test initialization with fractions."""
        # Simple fractions
        num1 = SymergeticsNumber(Fraction(3, 4))
        assert num1.numerator == 3
        assert num1.denominator == 4
        assert num1.to_float() == 0.75
        
        # Negative fractions
        num2 = SymergeticsNumber(Fraction(-5, 8))
        assert num2.numerator == -5
        assert num2.denominator == 8
        assert num2.to_float() == -0.625
    
    def test_initialization_with_floats(self):
        """Test initialization with floating point numbers."""
        # Simple float
        num1 = SymergeticsNumber(3.14159)
        assert abs(num1.to_float() - 3.14159) < 1e-10
        
        # Negative float
        num2 = SymergeticsNumber(-2.71828)
        assert abs(num2.to_float() - (-2.71828)) < 1e-10
    
    def test_initialization_with_strings(self):
        """Test initialization with string representations."""
        # Integer string
        num1 = SymergeticsNumber("42")
        assert num1 == 42
        
        # Fraction string
        num2 = SymergeticsNumber("3/4")
        assert num2 == Fraction(3, 4)
        
        # Decimal string
        num3 = SymergeticsNumber("3.14159")
        assert abs(num3.to_float() - 3.14159) < 1e-10


class TestSymergeticsNumberArithmetic:
    """Test arithmetic operations on SymergeticsNumber."""
    
    def test_addition(self):
        """Test addition operations."""
        num1 = SymergeticsNumber(3, 4)
        num2 = SymergeticsNumber(1, 2)
        
        # Addition with SymergeticsNumber
        result1 = num1 + num2
        assert result1 == SymergeticsNumber(5, 4)
        
        # Addition with integers
        result2 = num1 + 2
        assert result2 == SymergeticsNumber(11, 4)
        
        # Addition with floats
        result3 = num1 + 0.5
        assert abs(result3.to_float() - 1.25) < 1e-10
    
    def test_subtraction(self):
        """Test subtraction operations."""
        num1 = SymergeticsNumber(5, 4)
        num2 = SymergeticsNumber(1, 2)
        
        # Subtraction with SymergeticsNumber
        result1 = num1 - num2
        assert result1 == SymergeticsNumber(3, 4)
        
        # Subtraction with integers
        result2 = num1 - 1
        assert result2 == SymergeticsNumber(1, 4)
    
    def test_multiplication(self):
        """Test multiplication operations."""
        num1 = SymergeticsNumber(3, 4)
        num2 = SymergeticsNumber(2, 3)
        
        # Multiplication with SymergeticsNumber
        result1 = num1 * num2
        assert result1 == SymergeticsNumber(1, 2)
        
        # Multiplication with integers
        result2 = num1 * 4
        assert result2 == 3
        
        # Multiplication with floats
        result3 = num1 * 2.0
        assert abs(result3.to_float() - 1.5) < 1e-10
    
    def test_division(self):
        """Test division operations."""
        num1 = SymergeticsNumber(3, 4)
        num2 = SymergeticsNumber(1, 2)
        
        # Division with SymergeticsNumber
        result1 = num1 / num2
        assert result1 == SymergeticsNumber(3, 2)
        
        # Division with integers
        result2 = num1 / 2
        assert result2 == SymergeticsNumber(3, 8)
    
    def test_power_operations(self):
        """Test power operations."""
        num = SymergeticsNumber(2, 3)
        
        # Positive integer power
        result1 = num ** 2
        assert result1 == SymergeticsNumber(4, 9)
        
        # Negative integer power
        result2 = num ** -2
        assert result2 == SymergeticsNumber(9, 4)
        
        # Zero power
        result3 = num ** 0
        assert result3 == 1


class TestSymergeticsNumberComparison:
    """Test comparison operations on SymergeticsNumber."""
    
    def test_equality(self):
        """Test equality comparisons."""
        num1 = SymergeticsNumber(3, 4)
        num2 = SymergeticsNumber(6, 8)
        num3 = SymergeticsNumber(1, 2)
        
        assert num1 == num2  # Same value, different representation
        assert num1 != num3  # Different values
        assert num1 == 0.75  # Comparison with float
        assert num1 == Fraction(3, 4)  # Comparison with Fraction
    
    def test_ordering(self):
        """Test ordering comparisons."""
        num1 = SymergeticsNumber(1, 2)
        num2 = SymergeticsNumber(3, 4)
        num3 = SymergeticsNumber(1, 2)
        
        assert num1 < num2
        assert num2 > num1
        assert num1 <= num2
        assert num2 >= num1
        assert num1 <= num3
        assert num1 >= num3


class TestSymergeticsNumberProperties:
    """Test properties and methods of SymergeticsNumber."""
    
    def test_properties(self):
        """Test basic properties."""
        num = SymergeticsNumber(15, 6)
        
        assert num.numerator == 5  # Should be simplified
        assert num.denominator == 2
        assert num.denominator != 1  # Not an integer
        assert num > 0  # Positive
        assert num != 0  # Not zero
    
    def test_string_representations(self):
        """Test string representations."""
        num = SymergeticsNumber(5, 2)
        
        assert str(num) == "5/2 (2.50000000)"
        assert repr(num) == "SymergeticsNumber(5/2) ≈ 2.500000"
        assert f"{num}" == "5/2 (2.50000000)"
        # Format strings not supported by SymergeticsNumber
    
    def test_conversion_methods(self):
        """Test conversion methods."""
        num = SymergeticsNumber(7, 3)
        
        assert num.to_float() == pytest.approx(7/3)
        # Test that we can access the underlying fraction
        assert num.value == Fraction(7, 3)
        assert num.numerator == 7
        assert num.denominator == 3


class TestSymergeticsNumberSpecialMethods:
    """Test special mathematical methods."""
    
    def test_sqrt(self):
        """Test square root calculation."""
        num = SymergeticsNumber(16, 9)
        sqrt_result = SymergeticsNumber.sqrt(num)
        
        assert sqrt_result == SymergeticsNumber(4, 3)
        assert sqrt_result ** 2 == num
    
    def test_pi_approximation(self):
        """Test pi approximation."""
        pi_approx = SymergeticsNumber.pi()
        
        assert abs(pi_approx.to_float() - math.pi) < 1e-10
        assert isinstance(pi_approx, SymergeticsNumber)
    
    def test_scheherazade_properties(self):
        """Test Scheherazade number properties."""
        # Test with 1001 (Scheherazade base)
        scheherazade = SymergeticsNumber(1001)
        
        # Test to_scheherazade_base method
        power, coeffs = scheherazade.to_scheherazade_base()
        assert power == 1
        assert coeffs == [1, 0]
    
    def test_palindrome_properties(self):
        """Test palindrome properties."""
        # Test palindromic number
        palindrome = SymergeticsNumber(12321)
        assert palindrome.is_palindromic() == True
        
        # Test non-palindromic number
        non_palindrome = SymergeticsNumber(12345)
        assert non_palindrome.is_palindromic() == False


class TestSymergeticsNumberEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_division(self):
        """Test division by zero."""
        num = SymergeticsNumber(1, 2)
        
        with pytest.raises(ZeroDivisionError):
            num / 0
        
        with pytest.raises(ZeroDivisionError):
            num / SymergeticsNumber(0)
    
    def test_very_large_numbers(self):
        """Test with very large numbers."""
        large_num = SymergeticsNumber(10**100)
        assert large_num == 10**100
        assert large_num.denominator == 1  # Is an integer
    
    def test_very_small_numbers(self):
        """Test with very small numbers."""
        small_num = SymergeticsNumber(1, 10**100)
        assert small_num > 0
        assert small_num.denominator != 1  # Not an integer
    
    def test_negative_zero(self):
        """Test handling of negative zero."""
        neg_zero = SymergeticsNumber(-0)
        assert neg_zero == 0
        assert neg_zero == 0  # Is zero


class TestSymergeticsNumberIntegration:
    """Test integration with other Symergetics components."""
    
    def test_with_constants(self):
        """Test interaction with SymergeticsConstants."""
        constants = SymergeticsConstants()
        
        # Test with volume ratios
        tetra_vol = constants.get_volume_ratio('tetrahedron')
        assert isinstance(tetra_vol, SymergeticsNumber)
        assert tetra_vol > 0
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        num = SymergeticsNumber(7, 3)
        
        # Test basic properties that can be serialized
        assert num.numerator == 7
        assert num.denominator == 3
        assert num.value == Fraction(7, 3)
        
        # Test that we can recreate from components
        restored = SymergeticsNumber(num.numerator, num.denominator)
        assert restored == num
    
    def test_copy_operations(self):
        """Test copy operations."""
        num = SymergeticsNumber(5, 7)
        
        # Test that we can create a new instance with same value
        copied = SymergeticsNumber(num.numerator, num.denominator)
        assert copied == num
        assert copied is not num  # Different objects
        
        # Test that operations don't affect original
        copied += 1
        assert copied != num
        assert num == SymergeticsNumber(5, 7)  # Original unchanged


if __name__ == "__main__":
    pytest.main([__file__])
