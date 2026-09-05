"""
Utilities module for Symergetics package.

Contains utility functions for conversions, mnemonic encoding,
formatting, and other helpful operations.
"""

from .conversion import (
    best_rational_approximation,
    continued_fraction_approximation,
    convergents_from_continued_fraction,
    convert_between_bases,
    coordinate_system_info,
    decimal_to_fraction,
    float_to_exact_rational,
    format_as_mixed_number,
    quadray_to_xyz,
    rational_to_float,
    scientific_notation_to_rational,
    xyz_to_quadray,
)
from .mnemonics import (
    compare_number_patterns,
    create_memory_aid,
    format_large_number,
    generate_synergetics_mnemonics,
    mnemonic_decode,
    mnemonic_encode,
    ungroup_number,
    visualize_number_pattern,
)
from .reporting import (
    AnalysisSummary,
    ReportMetrics,
    export_report_to_csv,
    export_report_to_json,
    export_report_to_markdown,
    generate_comparative_report,
    generate_performance_report,
    generate_statistical_summary,
)

__all__ = [
    # Conversions
    "rational_to_float",
    "float_to_exact_rational",
    "decimal_to_fraction",
    "xyz_to_quadray",
    "quadray_to_xyz",
    "continued_fraction_approximation",
    "convergents_from_continued_fraction",
    "best_rational_approximation",
    "format_as_mixed_number",
    "scientific_notation_to_rational",
    "coordinate_system_info",
    "convert_between_bases",

    # Mnemonics
    "mnemonic_encode",
    "mnemonic_decode",
    "format_large_number",
    "ungroup_number",
    "create_memory_aid",
    "visualize_number_pattern",
    "compare_number_patterns",
    "generate_synergetics_mnemonics",

    # Reporting
    "ReportMetrics",
    "AnalysisSummary",
    "generate_statistical_summary",
    "generate_comparative_report",
    "export_report_to_json",
    "export_report_to_csv",
    "export_report_to_markdown",
    "generate_performance_report",
]
