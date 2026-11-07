"""
SymergeticsNumber: Exact Rational Arithmetic for Synergetic Calculations

This module implements the core SymergeticsNumber class that provides exact
rational arithmetic using Python's fractions.Fraction as the foundation.
Integer powers and perfect rational roots are exact; irrational powers and roots
are returned as bounded-denominator rational approximations.

Key Features:
- Exact rational arithmetic with unlimited precision
- Scheherazade number operations (powers of 1001)
- Mnemonic encoding for large integers
- Seamless conversion between rational and floating-point representations
- Integration with Fuller's geometric and cosmological calculations

Author: Symergetics Team
"""

import math
import operator
from decimal import Decimal, localcontext
from fractions import Fraction
from numbers import Integral, Rational
from typing import List, Optional, Tuple, Union

# math.gcd exists on every supported Python (>=3.8); fractions.gcd was removed
# in Python 3.9. Re-exported for backwards compatibility (see
# tests/test_core_comprehensive.py).
from math import gcd  # noqa: F401


def _native_fraction(value: Union[int, float, str, Rational],
                     denominator: Optional[Rational] = None) -> Fraction:
    """Promote rational fields before Fraction can perform fixed-width arithmetic."""
    if isinstance(value, Integral):
        value = int(value)
    elif isinstance(value, Rational) and (
        type(value.numerator) is not int or type(value.denominator) is not int
    ):
        value = Fraction(int(value.numerator), int(value.denominator))
    if denominator is None:
        return value if isinstance(value, Fraction) else Fraction(value)
    if isinstance(denominator, Integral):
        denominator = int(denominator)
    elif isinstance(denominator, Rational):
        denominator = _native_fraction(denominator)
    return Fraction(value, denominator)


def _integer_nth_root(value: int, degree: int) -> int:
    """Return the floor of a nonnegative integer's root without float conversion."""
    if value < 2 or degree == 1:
        return value
    if degree >= value.bit_length():
        return 1
    if degree == 2:
        return math.isqrt(value)

    root = 1 << ((value.bit_length() + degree - 1) // degree)
    while True:
        candidate = ((degree - 1) * root + value // root ** (degree - 1)) // degree
        if candidate >= root:
            return root
        root = candidate


def _rational_power(value: Fraction, exponent: Fraction, max_denominator: int) -> Fraction:
    """Keep exact results rational; approximate only genuinely irrational results."""
    power, degree = exponent.numerator, exponent.denominator
    if degree == 1:
        return value ** power
    if value < 0 and degree % 2 == 0:
        raise ValueError("An even root of a negative rational is not real")
    if not value and power < 0:
        raise ZeroDivisionError("Zero cannot be raised to a negative power")

    magnitude = abs(value)
    numerator = _integer_nth_root(magnitude.numerator, degree)
    denominator = _integer_nth_root(magnitude.denominator, degree)
    if numerator ** degree == magnitude.numerator and denominator ** degree == magnitude.denominator:
        root = Fraction(-numerator if value < 0 else numerator, denominator)
        return root ** power

    # Decimal avoids binary overflow/underflow. Include input and exponent digits
    # to retain ratios close to one, plus output integer digits and guard digits
    # beyond the resolution needed by a bounded-denominator approximation.
    input_bits = max(magnitude.numerator.bit_length(), magnitude.denominator.bit_length(),
                     abs(power).bit_length(), degree.bit_length())
    output_bits = ((abs(magnitude.numerator.bit_length() - magnitude.denominator.bit_length())
                    + 1) * abs(power) // degree + 1)
    # For near-one inputs, log2(1 + x) <= 2*x gives a much tighter
    # magnitude bound when a large exponent compensates for a tiny difference.
    near_one_bits = (2 * abs(magnitude.numerator - magnitude.denominator) * abs(power)
                     // (min(magnitude.numerator, magnitude.denominator) * degree) + 1)
    output_bits = min(output_bits, near_one_bits)
    decimal_digits = (max(input_bits, output_bits) * 30103 // 100000) + 1
    cap_digits = max_denominator.bit_length() * 30103 // 100000 + 1
    precision = decimal_digits + 2 * cap_digits + 20
    with localcontext() as context:
        context.prec = precision
        context.Emax = max(context.Emax, precision * 2)
        context.Emin = min(context.Emin, -precision * 2)
        base = Decimal(magnitude.numerator) / Decimal(magnitude.denominator)
        decimal_power = Decimal(power) / Decimal(degree)
        approximation = Fraction(context.power(base, decimal_power)).limit_denominator(max_denominator)
    return -approximation if value < 0 and power % 2 else approximation


class SymergeticsNumber:
    """
    Core class for exact rational arithmetic in Synergetics.

    Integer powers and perfect rational roots retain complete precision.
    Irrational roots and powers use bounded-denominator rational approximations.

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
        if isinstance(numerator, (str, Fraction)):
            self.value = _native_fraction(numerator)
        else:
            self.value = _native_fraction(numerator, denominator)

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
            return self.value == _native_fraction(other)
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
            result = self.value + _native_fraction(other)
        return SymergeticsNumber(result)

    def __sub__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> 'SymergeticsNumber':
        """Exact subtraction with rational arithmetic."""
        if isinstance(other, SymergeticsNumber):
            result = self.value - other.value
        else:
            result = self.value - _native_fraction(other)
        return SymergeticsNumber(result)

    def __mul__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> 'SymergeticsNumber':
        """Exact multiplication with rational arithmetic."""
        if isinstance(other, SymergeticsNumber):
            result = self.value * other.value
        else:
            result = self.value * _native_fraction(other)
        return SymergeticsNumber(result)

    def __truediv__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> 'SymergeticsNumber':
        """Exact division with rational arithmetic."""
        if isinstance(other, SymergeticsNumber):
            if other.value == 0:
                raise ZeroDivisionError("Division by zero")
            result = self.value / other.value
        else:
            other_frac = _native_fraction(other)
            if other_frac == 0:
                raise ZeroDivisionError("Division by zero")
            result = self.value / other_frac
        return SymergeticsNumber(result)

    def __pow__(self, exponent: Union[int, float, Fraction, 'SymergeticsNumber']) -> 'SymergeticsNumber':
        """Raise to an exact rational exponent, approximating only irrational results.

        Float exponents retain their exact binary rational value: use
        ``Fraction(1, 3)``, not ``1 / 3``, to request a real cube root of a
        negative number. Irrational results have denominator at most 1,000,000.
        """
        rational_exponent = exponent.value if isinstance(exponent, SymergeticsNumber) else _native_fraction(exponent)
        return SymergeticsNumber(_rational_power(self.value, rational_exponent, 1000000))

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
            return self.value < _native_fraction(other)
        except (ValueError, TypeError):
            return NotImplemented

    def __le__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, SymergeticsNumber):
            return self.value <= other.value
        try:
            return self.value <= _native_fraction(other)
        except (ValueError, TypeError):
            return NotImplemented

    def __gt__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> bool:
        """Greater than comparison."""
        if isinstance(other, SymergeticsNumber):
            return self.value > other.value
        try:
            return self.value > _native_fraction(other)
        except (ValueError, TypeError):
            return NotImplemented

    def __ge__(self, other: Union['SymergeticsNumber', int, float, Fraction]) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, SymergeticsNumber):
            return self.value >= other.value
        try:
            return self.value >= _native_fraction(other)
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
        return cls(_native_fraction(value).limit_denominator(max_denominator))

    @classmethod
    def nth_root(cls, value: Union['SymergeticsNumber', int, float, Fraction],
                 degree: int, max_denominator: int = 1000000) -> 'SymergeticsNumber':
        """Return a real root, exactly when the result is rational.

        ``degree`` and ``max_denominator`` must be positive integers. Perfect
        rational roots bypass the denominator cap, even for huge or tiny inputs.
        Negative inputs require an odd degree; non-real roots raise ValueError.
        Irrational results approximate a high-precision Decimal evaluation with
        denominator at most ``max_denominator``; they are not symbolic values or
        certified correctly rounded roots. A small cap can round tiny roots to
        zero. Float inputs mean their exact binary rational values.
        """
        degree = operator.index(degree)
        max_denominator = operator.index(max_denominator)
        if degree < 1:
            raise ValueError("degree must be a positive integer")
        if max_denominator < 1:
            raise ValueError("max_denominator must be a positive integer")
        rational_value = value.value if isinstance(value, SymergeticsNumber) else _native_fraction(value)
        return cls(_rational_power(rational_value, Fraction(1, degree), max_denominator))

    @classmethod
    def sqrt(cls, value: Union['SymergeticsNumber', int, float, Fraction],
             max_denominator: int = 1000000) -> 'SymergeticsNumber':
        """Return an exact rational square root or a bounded approximation.

        See :meth:`nth_root` for precision, domain, and denominator-cap behavior.
        """
        return cls.nth_root(value, 2, max_denominator)

    @classmethod
    def pi(cls, digits: int = 100) -> 'SymergeticsNumber':
        """
        Generate a rational approximation of pi with the requested precision.

        Uses Machin's formula
            pi = 16*arctan(1/5) - 4*arctan(1/239)
        evaluated with exact integer arithmetic, so the result is accurate
        to at least `digits` decimal places (unlike a float-derived value,
        which is limited to ~15-16 significant digits).

        Args:
            digits: Number of decimal digits for approximation

        Returns:
            SymergeticsNumber: Rational approximation of pi
        """
        if digits < 1:
            raise ValueError("digits must be a positive integer")
        guard = digits + 10  # guard digits absorb per-term rounding error
        scale = 10 ** guard

        def arctan_inv(x: int) -> int:
            """Return arctan(1/x) * scale as an integer, via the Taylor series."""
            total = 0
            k = 0
            power = x  # x^(2k+1)
            while True:
                term = scale // (power * (2 * k + 1))
                if term == 0:
                    break
                total += term if k % 2 == 0 else -term
                power *= x * x
                k += 1
            return total

        # pi = 4 * (4*arctan(1/5) - arctan(1/239)), computed exactly
        pi_scaled = 4 * (4 * arctan_inv(5) - arctan_inv(239))
        return cls(Fraction(pi_scaled, scale))


# Convenience functions for common operations
def rational_sqrt(value: Union[SymergeticsNumber, int, float, Fraction],
                  max_denominator: int = 1000000) -> SymergeticsNumber:
    """Convenience function for square root calculation."""
    return SymergeticsNumber.sqrt(value, max_denominator)


def rational_pi(digits: int = 100) -> SymergeticsNumber:
    """Convenience function for π calculation."""
    return SymergeticsNumber.pi(digits)
