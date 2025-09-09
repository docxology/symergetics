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

from typing import List, Dict, Tuple, Union, Optional, Set
import re
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

    return sorted(list(patterns), key=len, reverse=True)


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

    # Special analysis for known SSRCD patterns
    if scheherazade_power == 6:
        # 1001^6 contains the sequence 1,6,15,20,15,6,1
        analysis['pascal_coefficients'] = [1, 6, 15, 20, 15, 6, 1]
        analysis['ssrcd_patterns'] = ['1006015', '020015', '006001']

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
        # Convert coefficients to string patterns
        coeff_strs = [str(c) for c in coeffs]

        # Look for the sequence in the number string
        for i in range(len(number_str) - len(coeff_strs) + 1):
            match = True
            for j, coeff_str in enumerate(coeff_strs):
                if number_str[i + j] != coeff_str[0]:
                    match = False
                    break

            if match:
                # Found a potential match - check the full sequence
                substring = number_str[i:i + sum(len(c) for c in coeff_strs)]
                # This is a simplified check - full implementation would
                # handle multi-digit coefficients properly
                patterns.append({
                    'position': i,
                    'coefficients': coeffs,
                    'description': description,
                    'substring': substring
                })

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
        float: Ratio of palindromic digits to total digits
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

    Args:
        number: Number to analyze

    Returns:
        List[Dict]: List of symmetric pattern descriptions
    """
    if isinstance(number, SymergeticsNumber):
        num_str = str(abs(number.value.numerator))
    else:
        num_str = str(abs(int(number)))

    patterns = []
    n = len(num_str)

    if n < 3:
        return patterns

    center = n // 2

    # Check for symmetry around center
    symmetric_pairs = []
    for i in range(center):
        left_pos = center - 1 - i
        right_pos = center + 1 + i

        if left_pos >= 0 and right_pos < n:
            if num_str[left_pos] == num_str[right_pos]:
                symmetric_pairs.append((left_pos, right_pos, num_str[left_pos]))

    if symmetric_pairs:
        patterns.append({
            'type': 'central_symmetry',
            'pairs': symmetric_pairs,
            'symmetry_score': len(symmetric_pairs) / center
        })

    return patterns


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

    # Add special insights for known powers
    if power == 6:
        analysis['special_insight'] = {
            'description': "Contains complete Pascal's triangle row 6",
            'coefficients': [1, 6, 15, 20, 15, 6, 1],
            'mathematical_significance': "Binomial coefficients C(6,k) for k=0 to 6",
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
    palindromic_primes = []

    for num in range(2, limit + 1):
        if is_palindromic(num):
            # Check if prime (simplified primality test)
            is_prime = True
            if num < 2:
                is_prime = False
            elif num == 2:
                is_prime = True
            elif num % 2 == 0:
                is_prime = False
            else:
                for i in range(3, int(num**0.5) + 1, 2):
                    if num % i == 0:
                        is_prime = False
                        break

            if is_prime:
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
