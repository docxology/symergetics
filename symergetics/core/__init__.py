"""
Core module for Symergetics package.

Contains fundamental classes for exact rational arithmetic and coordinate systems.
"""

from .numbers import SymergeticsNumber
from .coordinates import QuadrayCoordinate
from .constants import SymergeticsConstants

__all__ = [
    "SymergeticsNumber",
    "QuadrayCoordinate",
    "SymergeticsConstants"
]
