"""
Core module for Symergetics package.

Contains fundamental classes for exact rational arithmetic and coordinate systems.
"""

from .constants import SymergeticsConstants
from .coordinates import QuadrayCoordinate
from .numbers import SymergeticsNumber

__all__ = [
    "SymergeticsNumber",
    "QuadrayCoordinate",
    "SymergeticsConstants"
]
