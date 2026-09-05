"""
Palindromic Number Analysis for Synergetics

This module provides functions for analyzing palindromic patterns in numbers,
particularly in Scheherazade numbers and other large integers from Fuller's
Synergetics system.

Key Features:
- Palindrome detection for integers and decimals
- Pattern extraction from large numbers
- SSRCD (Sublimely Rememberable Comprehensive Dividends) analysis
- Binomial coefficient pattern recognition

Author: Symergetics Team
"""

import math
from typing import Dict, List, Union

from ..core.numbers import SymergeticsNumber


def is_palindromic(number: Union[int, str, SymergeticsNumber]) -> bool:
    """
    Test if a number forms a palindrome when written in decimal.

    A palindromic number reads the same forwards and backwards.

    Args:
        number: Number to test (int, str, or SymergeticsNumber)

    Returns:
        bool: True if the number is palindromic

    Examples:
        >>> is_palindromic(121)  # 121 reads the same forwards/backwards
        True
        >>> is_palindromic(123)
        False
        >>> is_palindromic("1001")  # Scheherazade base
        True
    """
    if isinstance(number, SymergeticsNumber):
        # Convert to string representation of numerator
        num_str = str(abs(number.value.numerator))
    else:
        num_str = str(abs(int(number)))

    return num_str == num_str[::-1]


def extract_palindromic_patterns(number: Union[int, SymergeticsNumber],
                               min_length: int = 3) -> List[str]:
    """
    Extract all palindromic substrings from a number's decimal representation.

    Args:
        number: Number to analyze
        min_length: Minimum length of palindromic patterns to extract

    Returns:
        List[str]: List of palindromic substrings found
    """
    if isinstance(number, SymergeticsNumber):
        num_str = str(abs(number.value.numerator))
    else:
        num_str = str(abs(int(number)))

    patterns = set()

    # Check all substrings for palindromic property
    n = len(num_str)
    for i in range(n):
        for j in range(i + min_length, n + 1):
            substring = num_str[i:j]
            if is_palindromic(substring):
                patterns.add(substring)

    return sorted(patterns, key=len, reverse=True)


def find_palindromic_sequence(scheherazade_power: int) -> Dict[str, Union[str, List[int]]]:
    """
    Analyze a Scheherazade number power for palindromic patterns and SSRCD.

    Fuller identified that powers of 1001 contain "sublimely rememberable
    comprehensive dividends" (SSRCD) that correspond to Pascal's triangle
    coefficients and other meaningful patterns.

    Args:
        scheherazade_power: Power of 1001 to analyze

    Returns:
        Dict: Analysis results including patterns and coefficients
    """
    from .primorials import scheherazade_power as calc_power

    number = calc_power(scheherazade_power)
    num_str = str(number.value.numerator)

    analysis = {
        'power': scheherazade_power,
        'scheherazade_number': str(number.value),
        'digit_length': len(num_str),
        'is_palindromic': is_palindromic(number),
        'palindromic_patterns': extract_palindromic_patterns(number),
        'pascal_coefficients': [],
        'ssrcd_patterns': []
    }

    # Pascal's triangle row for (1 + 10^3)^n: the decimal expansion of 1001^n
    # is the 3-digit zero-padded concatenation of C(n, k) as long as every
    # coefficient fits in three digits (verified against the actual digits).
    coeffs = [math.comb(scheherazade_power, k) for k in range(scheherazade_power + 1)]
    analysis['pascal_coefficients'] = coeffs
    if all(c < 1000 for c in coeffs):
        blocks = [f"{c:03d}" for c in coeffs]
        if int(''.join(blocks)) == number.value.numerator:
            analysis['ssrcd_patterns'] = blocks
        else:
            analysis['ssrcd_patterns'] = ['coefficient blocks disrupted by carries']

    return analysis


def analyze_binomial_patterns(number_str: str) -> List[Dict[str, Union[int, str]]]:
    """
    Analyze a number string for patterns that correspond to binomial coefficients.

    Args:
        number_str: String representation of the number to analyze

    Returns:
        List[Dict]: List of pattern matches with their positions and values
    """
    patterns = []

    # Common binomial coefficient sequences to look for
    binomial_sequences = [
        ([1, 6, 15, 20, 15, 6, 1], "Pascal's Triangle Row 6"),
        ([1, 7, 21, 35, 35, 21, 7, 1], "Pascal's Triangle Row 7"),
        ([1, 8, 28, 56, 70, 56, 28, 8, 1], "Pascal's Triangle Row 8"),
    ]

    for coeffs, description in binomial_sequences:
        # The decimal expansion of (10^3 + 1)^n is the zero-padded
        # concatenation of 3-digit coefficient blocks; search for that
        # concatenated form (with and without leading zero padding).
        padded = ''.join(f"{c:03d}" for c in coeffs)
        probes = {padded, padded.lstrip('0') or '0'}
        for probe in probes:
            start = 0
            while True:
                idx = number_str.find(probe, start)
                if idx == -1:
                    break
                patterns.append({
                    'position': idx,
                    'coefficients': coeffs,
                    'description': description,
                    'substring': probe
                })
                start = idx + 1

    return patterns


def find_repeated_patterns(number: Union[int, SymergeticsNumber],
                          pattern_length: int = 2) -> Dict[str, List[int]]:
    """
    Find repeated digit patterns in a number.

    Args:
        number: Number to analyze
        pattern_length: Length of patterns to look for

    Returns:
        Dict: Dictionary mapping patterns to their positions
    """
    if isinstance(number, SymergeticsNumber):
        num_str = str(abs(number.value.numerator))
    else:
        num_str = str(abs(int(number)))

    patterns = {}

    for i in range(len(num_str) - pattern_length + 1):
        pattern = num_str[i:i + pattern_length]

        if pattern not in patterns:
            patterns[pattern] = []

        patterns[pattern].append(i)

    # Return all patterns (not just repeated ones) for comprehensive analysis
    return patterns


def calculate_palindromic_density(number: Union[int, SymergeticsNumber]) -> float:
    """
    Calculate the density of palindromic patterns in a number.

    Args:
        number: Number to analyze

    Returns:
        float: Ratio of center-symmetric digit-pair matches (odd-length
        palindromic centers) to total digits
    """
    if isinstance(number, SymergeticsNumber):
        num_str = str(abs(number.value.numerator))
    else:
        num_str = str(abs(int(number)))

    if len(num_str) == 0:
        return 0.0

    palindromic_count = 0

    # Count digits that are part of palindromic substrings
    for i in range(len(num_str)):
        # Check for palindromes centered at position i
        for radius in range(1, min(i, len(num_str) - 1 - i) + 1):
            if num_str[i - radius] == num_str[i + radius]:
                palindromic_count += 1
            else:
                break

    return palindromic_count / len(num_str)


def find_symmetric_patterns(number: Union[int, SymergeticsNumber]) -> List[Dict[str, Union[int, str]]]:
    """
    Find symmetric patterns around the center of a number.

    Handles both centerings: odd-length palindromes (mirror through a center
    digit) and even-length palindromes (mirror between the two middle digits).
    The reported score is the fraction of the maximum possible symmetric pairs
    for the best centering (1.0 for a perfect palindrome).

    Args:
        number: Number to analyze

    Returns:
        List[Dict]: List of symmetric pattern descriptions
    """
    if isinstance(number, SymergeticsNumber):
        num_str = str(abs(number.value.numerator))
    else:
        num_str = str(abs(int(number)))

    n = len(num_str)
    if n < 3:
        return []

    def center_pairs(center: int, even_axis: bool):
        """Contiguous symmetric pairs from the given centering."""
        pairs = []
        if even_axis:
            radius = 0
            while center - radius >= 0 and center + 1 + radius < n:
                if num_str[center - radius] == num_str[center + 1 + radius]:
                    pairs.append((center - radius, center + 1 + radius,
                                  num_str[center - radius]))
                    radius += 1
                else:
                    break
            max_pairs = min(center + 1, n - center - 1)
        else:
            radius = 1
            while center - radius >= 0 and center + radius < n:
                if num_str[center - radius] == num_str[center + radius]:
                    pairs.append((center - radius, center + radius,
                                  num_str[center - radius]))
                    radius += 1
                else:
                    break
            max_pairs = min(center, n - 1 - center)
        return pairs, max_pairs

    best = None
    candidate_centers = {(n - 1) // 2, n // 2}
    for center in candidate_centers:
        for even_axis in (False, True):
            pairs, max_pairs = center_pairs(center, even_axis)
            if pairs:
                score = len(pairs) / max_pairs if max_pairs else 0.0
                candidate = {
                    'type': 'central_symmetry',
                    'center': center,
                    'even_axis': even_axis,
                    'pairs': pairs,
                    'symmetry_score': score,
                }
                if best is None or score > best['symmetry_score']:
                    best = candidate

    return [best] if best else []


def analyze_scheherazade_ssrcd(power: int) -> Dict[str, Union[int, str, List, Dict]]:
    """
    Comprehensive analysis of SSRCD patterns in Scheherazade numbers.

    Fuller identified that powers of 1001 contain "sublimely rememberable
    comprehensive dividends" that reveal fundamental mathematical patterns.

    Args:
        power: Power of 1001 to analyze

    Returns:
        Dict: Comprehensive analysis of patterns and relationships
    """
    from .primorials import scheherazade_power as calc_power

    number = calc_power(power)
    num_str = str(number.value.numerator)

    analysis = {
        'scheherazade_power': power,
        'number_string': num_str,
        'digit_count': len(num_str),
        'is_palindromic': is_palindromic(number),
        'palindromic_patterns': extract_palindromic_patterns(number),
        'repeated_patterns': find_repeated_patterns(number),
        'palindromic_density': calculate_palindromic_density(number),
        'symmetric_patterns': find_symmetric_patterns(number),
        'binomial_patterns': analyze_binomial_patterns(num_str),
    }

    # Pascal's row insight: valid while every coefficient fits a 3-digit block
    coeffs = [math.comb(power, k) for k in range(power + 1)]
    if all(c < 1000 for c in coeffs) and int(''.join(f"{c:03d}" for c in coeffs)) == number.value.numerator:
        analysis['special_insight'] = {
            'description': f"Contains complete Pascal's triangle row {power}",
            'coefficients': coeffs,
            'mathematical_significance': f"Binomial coefficients C({power},k) for k=0 to {power}",
            'fuller_interpretation': "Sublimely rememberable comprehensive dividend"
        }

    return analysis


def generate_palindromic_sequence(start: int, count: int) -> List[int]:
    """
    Generate a sequence of palindromic numbers starting from a given number.

    Args:
        start: Starting number for the sequence
        count: Number of palindromic numbers to generate

    Returns:
        List[int]: List of palindromic numbers
    """
    palindromes = []
    current = start

    while len(palindromes) < count:
        if is_palindromic(current):
            palindromes.append(current)
        current += 1

    return palindromes


def find_palindromic_primes(limit: int) -> List[int]:
    """
    Find all palindromic prime numbers up to a given limit.

    Args:
        limit: Upper limit for the search

    Returns:
        List[int]: List of palindromic prime numbers
    """
    from .primorials import is_prime

    palindromic_primes = []

    for num in range(2, limit + 1):
        if is_palindromic(num) and is_prime(num):
            palindromic_primes.append(num)

    return palindromic_primes



def analyze_number_for_synergetics(number: Union[int, SymergeticsNumber]) -> Dict[str, Union[bool, float, List, Dict]]:
    """
    Perform comprehensive Synergetics analysis on a number.

    Args:
        number: Number to analyze

    Returns:
        Dict: Complete analysis including all palindromic and pattern properties
    """
    analysis = {
        'is_palindromic': is_palindromic(number),
        'palindromic_patterns': extract_palindromic_patterns(number),
        'repeated_patterns': find_repeated_patterns(number),
        'palindromic_density': calculate_palindromic_density(number),
        'symmetric_patterns': find_symmetric_patterns(number),
    }

    # Add Synergetics-specific insights
    if isinstance(number, SymergeticsNumber):
        analysis['scheherazade_analysis'] = analyze_scheherazade_ssrcd(1)  # Placeholder
        analysis['prime_factors'] = {}  # Could integrate with primorials module

    return analysis
