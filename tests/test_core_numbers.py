"""
Tests for core.numbers module - SymergeticsNumber class.

Tests exact rational arithmetic, palindrome detection, and mnemonic encoding.
"""

import pytest
from fractions import Fraction
import math
from symergetics.core.numbers import SymergeticsNumber, rational_sqrt, rational_pi


class TestSymergeticsNumber:
    """Test SymergeticsNumber class functionality."""

    def test_initialization(self):
        """Test various initialization methods."""
        # Integer initialization
        n1 = SymergeticsNumber(5)
        assert n1.value == Fraction(5, 1)

        # Fraction initialization
        n2 = SymergeticsNumber(3, 4)
        assert n2.value == Fraction(3, 4)

        # String initialization
        n3 = SymergeticsNumber("7/3")
        assert n3.value == Fraction(7, 3)

        # Fraction object initialization
        n4 = SymergeticsNumber(Fraction(5, 7))
        assert n4.value == Fraction(5, 7)

    def test_arithmetic_operations(self):
        """Test basic arithmetic operations maintain exact precision."""
        a = SymergeticsNumber(3, 4)  # 3/4
        b = SymergeticsNumber(1, 2)  # 1/2

        # Addition
        result = a + b
        assert result.value == Fraction(5, 4)

        # Subtraction
        result = a - b
        assert result.value == Fraction(1, 4)

        # Multiplication
        result = a * b
        assert result.value == Fraction(3, 8)

        # Division
        result = a / b
        assert result.value == Fraction(3, 2)

        # Power
        result = a ** 2
        assert result.value == Fraction(9, 16)

    def test_comparison_operations(self):
        """Test comparison operations."""
        a = SymergeticsNumber(3, 4)  # 0.75
        b = SymergeticsNumber(1, 2)  # 0.5
        c = SymergeticsNumber(3, 4)  # 0.75

        assert a > b
        assert b < a
        assert a >= c
        assert a <= c
        assert a == c
        assert a != b

    def test_unary_operations(self):
        """Test unary operations."""
        a = SymergeticsNumber(3, 4)

        # Negation
        neg_a = -a
        assert neg_a.value == Fraction(-3, 4)

        # Absolute value
        abs_a = abs(a)
        assert abs_a.value == Fraction(3, 4)

        abs_neg_a = abs(-a)
        assert abs_neg_a.value == Fraction(3, 4)

    def test_properties(self):
        """Test number properties."""
        a = SymergeticsNumber(6, 8)  # Should simplify to 3/4

        assert a.numerator == 3
        assert a.denominator == 4

    def test_float_conversion(self):
        """Test conversion to floating point."""
        a = SymergeticsNumber(1, 3)

        # Default precision
        float_val = a.to_float()
        assert abs(float_val - 1/3) < 1e-10

        # Specific precision
        float_val = a.to_float(precision=4)
        assert abs(float_val - 0.3333) < 1e-4

    def test_scheherazade_base(self):
        """Test Scheherazade base representation."""
        # Test with 1001 itself
        num = SymergeticsNumber(1001)
        power, coefficients = num.to_scheherazade_base()
        assert power == 1
        assert coefficients == [1, 0]  # 1*1001^1 + 0*1001^0

        # Test with 1001^2
        num = SymergeticsNumber(1002001)
        power, coefficients = num.to_scheherazade_base()
        assert power == 2
        assert coefficients == [1, 0, 0]  # 1*1001^2 + 0*1001^1 + 0*1001^0

    def test_mnemonic_encoding(self):
        """Test mnemonic encoding for large numbers."""
        # Small number
        small = SymergeticsNumber(42)
        mnemonic = small.to_mnemonic()
        assert "42" in mnemonic

        # Large number
        large = SymergeticsNumber(1234567890)
        mnemonic = large.to_mnemonic()
        assert "billion" in mnemonic or "," in mnemonic

    def test_palindrome_detection(self):
        """Test palindrome detection."""
        # Palindromic numbers
        pal1 = SymergeticsNumber(121)
        pal2 = SymergeticsNumber(1001)
        pal3 = SymergeticsNumber(12321)

        assert pal1.is_palindromic()
        assert pal2.is_palindromic()
        assert pal3.is_palindromic()

        # Non-palindromic numbers
        non_pal1 = SymergeticsNumber(123)
        non_pal2 = SymergeticsNumber(1234)

        assert not non_pal1.is_palindromic()
        assert not non_pal2.is_palindromic()

    def test_from_float(self):
        """Test creation from floating point with rational approximation."""
        # Exact rational
        num = SymergeticsNumber.from_float(0.5)
        assert num.value == Fraction(1, 2)

        # Rational approximation
        num = SymergeticsNumber.from_float(math.pi, max_denominator=100)
        assert abs(float(num.value) - math.pi) < 0.01

    def test_sqrt_functionality(self):
        """Test square root functionality."""
        # Perfect square
        num = SymergeticsNumber(9)
        sqrt_num = num.sqrt(num)
        assert sqrt_num.value == Fraction(3, 1)

        # Rational approximation
        num = SymergeticsNumber(2)
        sqrt_num = num.sqrt(num, max_denominator=100)
        assert abs(float(sqrt_num.value) - math.sqrt(2)) < 0.01

    def test_pi_approximation(self):
        """Test π approximation."""
        pi_approx = SymergeticsNumber.pi(digits=50)
        assert abs(float(pi_approx.value) - math.pi) < 1e-10


class TestUtilityFunctions:
    """Test utility functions in numbers module."""

    def test_rational_sqrt(self):
        """Test rational square root function."""
        result = rational_sqrt(SymergeticsNumber(4))
        assert result.value == Fraction(2, 1)

    def test_rational_pi(self):
        """Test rational π function."""
        pi_val = rational_pi(digits=10)
        assert abs(float(pi_val.value) - math.pi) < 1e-8


    def test_gcd_import_compatibility(self):
        """Test that gcd import works correctly for different Python versions."""
        # This tests the import logic at the top of the file
        from symergetics.core.numbers import SymergeticsNumber
        import math

        # Test that basic operations work (which use gcd internally)
        a = SymergeticsNumber(12, 18)  # Should be reduced to 2/3
        assert a.value == Fraction(2, 3)

    def test_string_representation(self):
        """Test string representation methods."""
        n = SymergeticsNumber(3, 4)

        # Test __str__ (includes fraction and decimal)
        assert str(n) == "3/4 (0.75000000)"

        # Test __repr__
        assert repr(n).startswith("SymergeticsNumber")

    def test_hash_functionality(self):
        """Test hash functionality for use in sets and dictionaries."""
        n1 = SymergeticsNumber(2, 3)
        n2 = SymergeticsNumber(4, 6)  # Should be equal to n1
        n3 = SymergeticsNumber(1, 2)

        # Test that equal numbers have equal hashes
        assert hash(n1) == hash(n2)
        assert hash(n1) != hash(n3)

        # Test use in sets
        num_set = {n1, n2, n3}
        assert len(num_set) == 2  # n1 and n2 should be considered the same

    def test_bool_conversion(self):
        """Test boolean conversion."""
        zero = SymergeticsNumber(0)
        non_zero = SymergeticsNumber(5)

        assert bool(zero) == False
        assert bool(non_zero) == True

    def test_power_operations_edge_cases(self):
        """Test power operations with edge cases."""
        base = SymergeticsNumber(2)

        # Zero power
        result = base ** 0
        assert result.value == Fraction(1, 1)

        # First power
        result = base ** 1
        assert result.value == Fraction(2, 1)

        # Negative integer power
        result = base ** (-1)
        assert result.value == Fraction(1, 2)

    def test_large_number_operations(self):
        """Test operations with very large numbers."""
        large_num = SymergeticsNumber(10**50, 3)
        small_num = SymergeticsNumber(1, 10**20)

        result = large_num + small_num
        assert result.value.numerator > 10**49  # Should handle large numbers

    def test_fractional_power_edge_cases(self):
        """Test fractional powers with edge cases."""
        # Square root of perfect square
        n = SymergeticsNumber(9)
        result = n ** Fraction(1, 2)
        assert result.value == Fraction(3, 1)

        # Cube root of perfect cube
        n = SymergeticsNumber(8)
        result = n ** Fraction(1, 3)
        assert result.value == Fraction(2, 1)

    def test_comparison_with_regular_numbers(self):
        """Test comparison with regular Python numbers."""
        n = SymergeticsNumber(3, 2)

        assert n > 1
        assert n < 2
        assert n == 1.5
        assert n >= 1.5
        assert n <= 1.5

    def test_division_by_zero_protection(self):
        """Test that division by zero raises appropriate errors."""
        n = SymergeticsNumber(1)

        with pytest.raises(ZeroDivisionError):
            result = n / 0

        with pytest.raises(ZeroDivisionError):
            result = n / SymergeticsNumber(0)


if __name__ == "__main__":
    pytest.main([__file__])
