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
    "create_memory_aid"
]
