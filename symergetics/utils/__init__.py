"""
Utilities module for Symergetics package.

Contains utility functions for conversions, mnemonic encoding,
formatting, and other helpful operations.
"""

from .conversion import (
    rational_to_float,
    float_to_exact_rational,
    xyz_to_quadray,
    quadray_to_xyz,
    decimal_to_fraction
)
from .mnemonics import (
    mnemonic_encode,
    mnemonic_decode,
    format_large_number,
    create_memory_aid
)
from .reporting import (
    generate_statistical_summary,
    generate_comparative_report,
    export_report_to_json,
    export_report_to_csv,
    export_report_to_markdown,
    generate_performance_report,
    ReportMetrics,
    AnalysisSummary
)

__all__ = [
    # Conversions
    "rational_to_float",
    "float_to_exact_rational",
    "xyz_to_quadray",
    "quadray_to_xyz",
    "decimal_to_fraction",

    # Mnemonics
    "mnemonic_encode",
    "mnemonic_decode",
    "format_large_number",
    "create_memory_aid",

    # Reporting
    "generate_statistical_summary",
    "generate_comparative_report",
    "export_report_to_json",
    "export_report_to_csv",
    "export_report_to_markdown",
    "generate_performance_report",
    "ReportMetrics",
    "AnalysisSummary"
]
