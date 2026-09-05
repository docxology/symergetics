"""
Computation module for Symergetics package.

Contains functions for primorial calculations, Scheherazade numbers,
palindromic sequences, and other computational mathematics from
Fuller's Synergetics system.
"""

from .analysis import (
    ComparativeAnalysis,
    PatternMetrics,
    analyze_mathematical_patterns,
    compare_mathematical_domains,
    generate_comprehensive_report,
)
from .geometric_mnemonics import (
    GeometricMnemonic,
    analyze_geometric_mnemonics,
    create_integer_ratio_visualization,
    generate_geometric_mnemonic_report,
)
from .palindromes import (
    extract_palindromic_patterns,
    find_palindromic_sequence,
    is_palindromic,
)
from .primorials import factorial_decline, primorial, scheherazade_power

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
