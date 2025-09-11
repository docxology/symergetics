"""
Computation module for Symergetics package.

Contains functions for primorial calculations, Scheherazade numbers,
palindromic sequences, and other computational mathematics from
Fuller's Synergetics system.
"""

from .primorials import primorial, scheherazade_power, factorial_decline
from .palindromes import is_palindromic, extract_palindromic_patterns, find_palindromic_sequence
from .analysis import (
    analyze_mathematical_patterns,
    compare_mathematical_domains,
    generate_comprehensive_report,
    PatternMetrics,
    ComparativeAnalysis
)
from .geometric_mnemonics import (
    analyze_geometric_mnemonics,
    create_integer_ratio_visualization,
    generate_geometric_mnemonic_report,
    GeometricMnemonic
)

__all__ = [
    # Primorials
    "primorial",
    "scheherazade_power",
    "factorial_decline",

    # Palindromes
    "is_palindromic",
    "extract_palindromic_patterns",
    "find_palindromic_sequence",

    # Advanced Analysis
    "analyze_mathematical_patterns",
    "compare_mathematical_domains",
    "generate_comprehensive_report",
    "PatternMetrics",
    "ComparativeAnalysis",

    # Geometric Mnemonics
    "analyze_geometric_mnemonics",
    "create_integer_ratio_visualization",
    "generate_geometric_mnemonic_report",
    "GeometricMnemonic"
]
