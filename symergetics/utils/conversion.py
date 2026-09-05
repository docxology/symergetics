"""
Conversion Utilities for Symergetics

This module provides functions for converting between different number
representations and coordinate systems used in Synergetics.

Key Features:
- Rational to floating-point conversions with precision control
- Coordinate system conversions (Quadray ↔ XYZ)
- Decimal to exact fraction conversions
- High-precision arithmetic utilities

Author: Symergetics Team
"""

import math
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.coordinates import QuadrayCoordinate
from ..core.numbers import SymergeticsNumber


def rational_to_float(rational: Union[SymergeticsNumber, Fraction],
                     precision: Optional[int] = None) -> float:
    """
    Convert a rational number to floating-point with optional precision control.

    Args:
        rational: Rational number to convert
        precision: Optional decimal places to round to

    Returns:
        float: Floating-point representation

    Examples:
        >>> rational_to_float(SymergeticsNumber(3, 4))  # 3/4 = 0.75
        0.75
        >>> rational_to_float(SymergeticsNumber(1, 3), precision=4)  # 1/3 ≈ 0.3333
        0.3333
    """
    if isinstance(rational, SymergeticsNumber):
        value = float(rational.value)
    else:
        value = float(rational)

    if precision is not None:
        return round(value, precision)

    return value


def float_to_exact_rational(value: float,
                          max_denominator: int = 1000000) -> SymergeticsNumber:
    """
    Convert a floating-point number to an exact rational approximation.

    Uses continued fraction approximation to find the best rational
    within the specified maximum denominator.

    Args:
        value: Floating-point number to convert
        max_denominator: Maximum denominator for approximation

    Returns:
        SymergeticsNumber: Rational approximation

    Examples:
        >>> float_to_exact_rational(0.75)  # Exactly 3/4
        SymergeticsNumber(3/4)
        >>> float_to_exact_rational(math.pi, max_denominator=100)
        SymergeticsNumber(311/99)  # closest rational with denominator <= 100
    """
    frac = Fraction(value).limit_denominator(max_denominator)
    return SymergeticsNumber(frac)


def decimal_to_fraction(decimal_str: str) -> SymergeticsNumber:
    """
    Convert a decimal string to an exact rational fraction.

    Args:
        decimal_str: String representation of decimal number

    Returns:
        SymergeticsNumber: Exact rational representation

    Examples:
        >>> decimal_to_fraction("0.75")
        SymergeticsNumber(3/4)
        >>> decimal_to_fraction("3.14159")
        SymergeticsNumber(314159/100000)
    """
    return SymergeticsNumber(Fraction(decimal_str))


def xyz_to_quadray(x: Union[float, SymergeticsNumber],
                  y: Union[float, SymergeticsNumber],
                  z: Union[float, SymergeticsNumber],
                  embedding: Optional[np.ndarray] = None) -> QuadrayCoordinate:
    """
    Convert Cartesian XYZ coordinates to Quadray coordinates.

    Uses the Moore-Penrose pseudoinverse of the Urner embedding matrix to
    find the nearest IVM lattice point (same algorithm as
    QuadrayCoordinate.from_xyz).

    Args:
        x, y, z: Cartesian coordinates
        embedding: Optional 3x4 embedding matrix

    Returns:
        QuadrayCoordinate: Nearest Quadray lattice coordinates

    Examples:
        >>> coord = xyz_to_quadray(2.0, 0.0, 0.0)
        >>> coord.to_xyz()  # Round-trips an IVM lattice point
        (2.0, 0.0, 0.0)
    """
    return QuadrayCoordinate.from_xyz(float(x), float(y), float(z), embedding)


def quadray_to_xyz(coord: QuadrayCoordinate,
                  embedding: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    Convert Quadray coordinates to Cartesian XYZ coordinates.

    Uses the Urner embedding matrix for the conversion.

    Args:
        coord: Quadray coordinate to convert
        embedding: Optional 3x4 embedding matrix

    Returns:
        Tuple[float, float, float]: (x, y, z) Cartesian coordinates
    """
    return coord.to_xyz(embedding)


def continued_fraction_approximation(value: float, max_terms: int = 10) -> List[int]:
    """
    Generate continued fraction approximation of a real number.

    Args:
        value: Real number to approximate
        max_terms: Maximum number of continued fraction terms

    Returns:
        List[int]: Continued fraction coefficients

    Examples:
        >>> continued_fraction_approximation(math.pi, 4)  # π ≈ [3, 7, 15, 1]
        [3, 7, 15, 1]
        >>> continued_fraction_approximation((1 + math.sqrt(5))/2, 5)  # Golden ratio
        [1, 1, 1, 1, 1, ...]
    """
    terms = []
    try:
        current = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"continued_fraction_approximation requires a real number, got {value!r}"
        )

    for _ in range(max_terms):
        integer_part = math.floor(current)
        terms.append(integer_part)

        fractional_part = current - integer_part
        if abs(fractional_part) < 1e-10:
            break

        current = 1.0 / fractional_part

    return terms


def convergents_from_continued_fraction(terms: List[int]) -> List[Tuple[int, int]]:
    """
    Generate convergents from continued fraction terms.

    Args:
        terms: Continued fraction coefficients

    Returns:
        List[Tuple[int, int]]: List of (numerator, denominator) pairs

    Examples:
        >>> convergents_from_continued_fraction([3, 7, 15, 1])
        [(3, 1), (22, 7), (333, 106), (355, 113)]
    """
    convergents = []

    if not terms:
        return convergents

    # First convergent
    h_prev = 1
    k_prev = 0
    h_curr = terms[0]
    k_curr = 1
    convergents.append((h_curr, k_curr))

    for term in terms[1:]:
        h_next = term * h_curr + h_prev
        k_next = term * k_curr + k_prev

        convergents.append((h_next, k_next))

        h_prev, k_prev = h_curr, k_curr
        h_curr, k_curr = h_next, k_next

    return convergents


def best_rational_approximation(value: float,
                              max_denominator: int = 1000000) -> SymergeticsNumber:
    """
    Find the best rational approximation to a real number.

    Uses continued fraction convergents to find the optimal approximation
    within the specified maximum denominator.

    Args:
        value: Real number to approximate
        max_denominator: Maximum denominator allowed

    Returns:
        SymergeticsNumber: Best rational approximation
    """
    terms = continued_fraction_approximation(value, 20)
    convergents = convergents_from_continued_fraction(terms)

    best_approx = SymergeticsNumber(0)
    best_error = float('inf')

    for num, den in convergents:
        if den > max_denominator:
            continue

        approx = SymergeticsNumber(num, den)
        error = abs(float(approx.value) - value)

        if error < best_error:
            best_error = error
            best_approx = approx

    return best_approx


def format_as_mixed_number(rational: Union[SymergeticsNumber, Fraction]) -> str:
    """
    Format a rational number as a mixed number.

    Args:
        rational: Rational number to format

    Returns:
        str: Mixed number representation

    Examples:
        >>> format_as_mixed_number(SymergeticsNumber(7, 3))  # 7/3 = 2 + 1/3
        "2 1/3"
        >>> format_as_mixed_number(SymergeticsNumber(5, 2))  # 5/2 = 2 + 1/2
        "2 1/2"
    """
    if isinstance(rational, SymergeticsNumber):
        frac = rational.value
    else:
        frac = Fraction(rational)

    if frac.denominator == 1:
        return str(frac.numerator)

    # Floor division gives the standard mixed-number form for both signs:
    # the whole part is floored and the fractional part stays non-negative
    # (e.g. -7/3 -> -3 2/3, which reads as -3 + 2/3 = -7/3).
    whole_part = frac.numerator // frac.denominator
    fractional_part = frac - Fraction(whole_part)

    if whole_part == 0:
        return f"{frac.numerator}/{frac.denominator}"

    if fractional_part.numerator == 0:
        return str(whole_part)
    else:
        return f"{whole_part} {fractional_part.numerator}/{fractional_part.denominator}"


def scientific_notation_to_rational(scientific_str: str) -> SymergeticsNumber:
    """
    Convert scientific notation string to exact rational.

    Args:
        scientific_str: Number in scientific notation (e.g., "1.23e-4")

    Returns:
        SymergeticsNumber: Exact rational representation

    Examples:
        >>> scientific_notation_to_rational("1.5e2")  # 1.5 × 10² = 150
        SymergeticsNumber(150)
        >>> scientific_notation_to_rational("2.5e-1")  # 2.5 × 10⁻¹ = 1/4
        SymergeticsNumber(1/4)
    """
    try:
        value = float(scientific_str)
        return float_to_exact_rational(value)
    except ValueError:
        raise ValueError(f"Invalid scientific notation: {scientific_str}")


def coordinate_system_info(coord_system: str) -> Dict[str, Union[str, int, List[str]]]:
    """
    Get information about different coordinate systems used in Synergetics.

    Args:
        coord_system: Name of coordinate system ('quadray', 'cartesian', 'spherical')

    Returns:
        Dict: Information about the coordinate system
    """
    systems = {
        'quadray': {
            'description': 'Four-coordinate tetrahedral system (a,b,c,d)',
            'dimensions': 4,
            'constraint': 'non-negative integers with min(a, b, c, d) = 0',
            'basis': ['A', 'B', 'C', 'D'],
            'use_case': 'Synergetic geometry and IVM lattice'
        },
        'cartesian': {
            'description': 'Three-coordinate rectangular system (x,y,z)',
            'dimensions': 3,
            'constraint': 'None',
            'basis': ['X', 'Y', 'Z'],
            'use_case': 'Standard Euclidean geometry'
        },
        'spherical': {
            'description': 'Spherical coordinate system (ρ,θ,φ)',
            'dimensions': 3,
            'constraint': 'ρ ≥ 0, 0 ≤ θ ≤ π, 0 ≤ φ < 2π',
            'basis': ['ρ (radius)', 'θ (polar angle)', 'φ (azimuthal angle)'],
            'use_case': 'Spherical symmetry and radial phenomena'
        }
    }

    if coord_system.lower() not in systems:
        available = list(systems.keys())
        raise ValueError(f"Unknown coordinate system '{coord_system}'. Available: {available}")

    return systems[coord_system.lower()]


def convert_between_bases(number: int, from_base: int, to_base: int) -> str:
    """
    Convert a number between different number bases.

    Args:
        number: Number to convert
        from_base: Source base
        to_base: Target base

    Returns:
        str: Number in target base

    Examples:
        >>> convert_between_bases(1001, 10, 7)  # 1001 in base 7
        '2630'
    """
    if not (2 <= from_base <= 36) or not (2 <= to_base <= 36):
        raise ValueError("Bases must be between 2 and 36")

    def to_digits(n: int, base: int) -> str:
        """Render a non-negative integer in the given base."""
        if n == 0:
            return '0'
        digits = []
        while n > 0:
            remainder = n % base
            if remainder < 10:
                digits.append(str(remainder))
            else:
                digits.append(chr(ord('A') + remainder - 10))
            n //= base
        return ''.join(reversed(digits))

    if from_base == 10:
        if not isinstance(number, int):
            raise ValueError("number must be an int when from_base is 10")
        sign = '-' if number < 0 else ''
        return sign + to_digits(abs(number), to_base)

    # Parse the source representation (string form; may carry a leading '-')
    text = str(number).strip()
    sign = ''
    if text.startswith('-'):
        sign = '-'
        text = text[1:]
    elif text.startswith('+'):
        text = text[1:]
    if not text:
        raise ValueError(f"Invalid number for base {from_base}: {number}")

    decimal = 0
    for digit in text:
        if digit.isdigit():
            digit_value = int(digit)
        elif 'A' <= digit.upper() <= 'Z':
            digit_value = 10 + ord(digit.upper()) - ord('A')
        else:
            raise ValueError(f"Invalid digit '{digit}' for base {from_base}")
        if digit_value >= from_base:
            raise ValueError(f"Digit '{digit}' is out of range for base {from_base}")
        decimal = decimal * from_base + digit_value

    if to_base == 10:
        return sign + str(decimal)
    return ('-' if sign and decimal else '') + to_digits(decimal, to_base)
