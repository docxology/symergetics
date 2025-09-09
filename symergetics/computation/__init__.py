"""
Computation module for Symergetics package.

Contains functions for primorial calculations, Scheherazade numbers,
palindromic sequences, and other computational mathematics from
Fuller's Synergetics system.
"""

from .primorials import primorial, scheherazade_power, factorial_decline
from .palindromes import is_palindromic, extract_palindromic_patterns, find_palindromic_sequence

__all__ = [
    # Primorials
    "primorial",
    "scheherazade_power",
    "factorial_decline",

    # Palindromes
    "is_palindromic",
    "extract_palindromic_patterns",
    "find_palindromic_sequence"
]
