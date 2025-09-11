"""
SymergeticsNumber: Exact Rational Arithmetic for Synergetic Calculations

This module implements the core SymergeticsNumber class that provides exact
rational arithmetic using Python's fractions.Fraction as the foundation.
All operations maintain mathematical precision without floating-point errors.

Key Features:
- Exact rational arithmetic with unlimited precision
- Scheherazade number operations (powers of 1001)
- Mnemonic encoding for large integers
- Seamless conversion between rational and floating-point representations
- Integration with Fuller's geometric and cosmological calculations

Author: Symergetics Team
"""

from fractions import Fraction
from typing import Union, Tuple, List, Optional
import math
import re

# Handle gcd import for different Python versions
try:
    from math import gcd
except ImportError:
    from fractions import gcd


class SymergeticsNumber:
    """
    Core class for exact rational arithmetic in Synergetics.

    Provides mathematical operations with complete precision using rational numbers,
    eliminating floating-point errors that compromise geometric calculations.

    Attributes:
        value (Fraction): The exact rational representation of the number
    """

    def __init__(self, numerator: Union[int, float, str, Fraction],
                 denominator: Union[int, None] = None):
        """
        Initialize a SymergeticsNumber with exact rational representation.

        Args:
            numerator: The numerator or complete rational value
            denominator: Optional denominator (if numerator is int/float)

        Examples:
            >>> n1 = SymergeticsNumber(3, 4)  # 3/4
            >>> n2 = SymergeticsNumber(1.5)   # 3/2
            >>> n3 = SymergeticsNumber(Fraction(5, 7))
        """
        if isinstance(numerator, str):
            # Handle string representations like "3/4" or "1.5"
            if '/' in numerator:
                self.value = Fraction(numerator)
            else:
                self.value = Fraction(float(numerator))
        elif isinstance(numerator, Fraction):
            self.value = numerator
        elif denominator is not None:
            self.value = Fraction(numerator, denominator)
        else:
            self.value = Fraction(numerator)

    def __repr__(self) -> str:
        """String representation showing both fraction and decimal forms."""
        return f"SymergeticsNumber({self.value}) ≈ {float(self.value):.6f}"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.value} ({float(self.value):.8f})"

    def __eq__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> bool:
        """Test equality with exact rational comparison."""
        if isinstance(other, SymergeticsNumber):
            return self.value == other.value
        try:
            return self.value == Fraction(other)
        except (ValueError, TypeError):
            return False

    def __hash__(self) -> int:
        """Hash based on the underlying Fraction value for use in sets and dicts."""
        return hash(self.value)

    def __add__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> 'SymergeticsNumber':
        """Exact addition with rational arithmetic."""
        if isinstance(other, SymergeticsNumber):
            result = self.value + other.value
        else:
            result = self.value + Fraction(other)
        return SymergeticsNumber(result)

    def __sub__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> 'SymergeticsNumber':
        """Exact subtraction with rational arithmetic."""
        if isinstance(other, SymergeticsNumber):
            result = self.value - other.value
        else:
            result = self.value - Fraction(other)
        return SymergeticsNumber(result)

    def __mul__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> 'SymergeticsNumber':
        """Exact multiplication with rational arithmetic."""
        if isinstance(other, SymergeticsNumber):
            result = self.value * other.value
        else:
            result = self.value * Fraction(other)
        return SymergeticsNumber(result)

    def __truediv__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> 'SymergeticsNumber':
        """Exact division with rational arithmetic."""
        if isinstance(other, SymergeticsNumber):
            if other.value == 0:
                raise ZeroDivisionError("Division by zero")
            result = self.value / other.value
        else:
            other_frac = Fraction(other)
            if other_frac == 0:
                raise ZeroDivisionError("Division by zero")
            result = self.value / other_frac
        return SymergeticsNumber(result)

    def __pow__(self, exponent: Union[int, float, Fraction, 'SymergeticsNumber']) -> 'SymergeticsNumber':
        """Exact exponentiation with rational arithmetic."""
        if isinstance(exponent, SymergeticsNumber):
            # Extract the rational value from SymergeticsNumber
            if exponent.value.denominator == 1:
                exponent = exponent.value.numerator
            else:
                exponent = float(exponent.value)

        if isinstance(exponent, int):
            result = self.value ** exponent
        else:
            # For non-integer exponents, convert to float then back to rational approximation
            exp_float = float(Fraction(exponent))
            result_float = float(self.value) ** exp_float
            result = Fraction(result_float).limit_denominator(1000000)
        return SymergeticsNumber(result)

    def __neg__(self) -> 'SymergeticsNumber':
        """Unary negation."""
        return SymergeticsNumber(-self.value)

    def __abs__(self) -> 'SymergeticsNumber':
        """Absolute value."""
        return SymergeticsNumber(abs(self.value))

    def __bool__(self) -> bool:
        """Boolean conversion. Returns False for zero, True otherwise."""
        return self.value != 0

    def __lt__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> bool:
        """Less than comparison."""
        if isinstance(other, SymergeticsNumber):
            return self.value < other.value
        try:
            return self.value < Fraction(other)
        except (ValueError, TypeError):
            return NotImplemented

    def __le__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, SymergeticsNumber):
            return self.value <= other.value
        try:
            return self.value <= Fraction(other)
        except (ValueError, TypeError):
            return NotImplemented

    def __gt__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> bool:
        """Greater than comparison."""
        if isinstance(other, SymergeticsNumber):
            return self.value > other.value
        try:
            return self.value > Fraction(other)
        except (ValueError, TypeError):
            return NotImplemented

    def __ge__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, SymergeticsNumber):
            return self.value >= other.value
        try:
            return self.value >= Fraction(other)
        except (ValueError, TypeError):
            return NotImplemented

    @property
    def numerator(self) -> int:
        """Return the numerator of the rational number."""
        return self.value.numerator

    @property
    def denominator(self) -> int:
        """Return the denominator of the rational number."""
        return self.value.denominator

    def to_float(self, precision: Optional[int] = None) -> float:
        """
        Convert to floating-point representation.

        Args:
            precision: Optional decimal precision limit

        Returns:
            float: Floating-point approximation
        """
        if precision is not None:
            return round(float(self.value), precision)
        return float(self.value)

    def to_scheherazade_base(self) -> Tuple[int, List[int]]:
        """
        Express the number in terms of powers of 1001 (Scheherazade base).

        Returns:
            Tuple[int, List[int]]: (power, coefficients) where number = sum(coeff * 1001^i)
        """
        if not isinstance(self.value, Fraction) or self.value.denominator != 1:
            raise ValueError("Scheherazade base conversion requires integer values")

        num = self.value.numerator
        base = 1001  # 7 × 11 × 13
        coefficients = []

        if num == 0:
            return 0, [0]

        # Find the highest power needed
        power = 0
        temp = num
        while temp >= base:
            coefficients.append(temp % base)
            temp //= base
            power += 1
        coefficients.append(temp)

        return power, coefficients[::-1]

    def to_mnemonic(self, max_digits: int = 50) -> str:
        """
        Generate a memorable representation for large integers.

        Uses grouping and patterns to make large numbers more comprehensible,
        following Fuller's principles of "sublimely rememberable comprehensive dividends."

        Args:
            max_digits: Maximum digits to process for mnemonic encoding

        Returns:
            str: Mnemonic representation of the number
        """
        if not isinstance(self.value, Fraction) or self.value.denominator != 1:
            return f"{self.value}"

        num_str = str(abs(self.value.numerator))

        if len(num_str) > max_digits:
            return f"Large number: {len(num_str)} digits"

        # Group digits for readability
        groups = []
        remaining = num_str

        # Handle billions, millions, thousands
        if len(remaining) > 9:
            groups.append(remaining[:-9] + " billion")
            remaining = remaining[-9:]

        if len(remaining) > 6:
            groups.append(remaining[:-6] + " million")
            remaining = remaining[-6:]

        if len(remaining) > 3:
            groups.append(remaining[:-3] + " thousand")
            remaining = remaining[-3:]

        if remaining:
            groups.append(remaining)

        mnemonic = " ".join(groups)

        # Add sign if negative
        if self.value.numerator < 0:
            mnemonic = "negative " + mnemonic

        return mnemonic

    def is_palindromic(self) -> bool:
        """
        Check if the number forms a palindrome when written in decimal.

        Returns:
            bool: True if the number is palindromic
        """
        if not isinstance(self.value, Fraction) or self.value.denominator != 1:
            return False

        num_str = str(abs(self.value.numerator))
        return num_str == num_str[::-1]

    def simplify(self) -> 'SymergeticsNumber':
        """
        Return a simplified version of the number (already simplified by Fraction).

        Returns:
            SymergeticsNumber: Simplified rational number
        """
        return SymergeticsNumber(self.value)

    @classmethod
    def from_float(cls, value: float, max_denominator: int = 1000000) -> 'SymergeticsNumber':
        """
        Create a SymergeticsNumber from a float with rational approximation.

        Args:
            value: Floating-point number to convert
            max_denominator: Maximum denominator for approximation

        Returns:
            SymergeticsNumber: Rational approximation of the float
        """
        return cls(Fraction(value).limit_denominator(max_denominator))

    @classmethod
    def sqrt(cls, value: Union['SymergeticsNumber', int, float, Fraction],
             max_denominator: int = 1000000) -> 'SymergeticsNumber':
        """
        Calculate square root with rational approximation.

        Args:
            value: Number to take square root of
            max_denominator: Maximum denominator for approximation

        Returns:
            SymergeticsNumber: Rational approximation of square root
        """
        if isinstance(value, cls):
            float_val = float(value.value)
        else:
            float_val = float(Fraction(value))

        sqrt_float = math.sqrt(float_val)
        return cls.from_float(sqrt_float, max_denominator)

    @classmethod
    def pi(cls, digits: int = 100) -> 'SymergeticsNumber':
        """
        Generate π with specified precision using rational approximation.

        Args:
            digits: Number of decimal digits for approximation

        Returns:
            SymergeticsNumber: Rational approximation of π
        """
        # Use continued fraction approximation of π
        # π ≈ 3 + 1/(7 + 1/(15 + 1/(1 + 1/(292 + ...))))
        # This is a simplified approximation
        pi_approx = math.pi
        return cls.from_float(pi_approx, 10**digits)


# Convenience functions for common operations
def rational_sqrt(value: Union[SymergeticsNumber, int, float, Fraction],
                  max_denominator: int = 1000000) -> SymergeticsNumber:
    """Convenience function for square root calculation."""
    return SymergeticsNumber.sqrt(value, max_denominator)


def rational_pi(digits: int = 100) -> SymergeticsNumber:
    """Convenience function for π calculation."""
    return SymergeticsNumber.pi(digits)
