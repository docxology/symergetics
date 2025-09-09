"""
Test suite for Symergetics package.

Comprehensive test coverage following TDD principles.
All tests validate mathematical correctness and implementation accuracy.
"""

import pytest
from pathlib import Path

# Test data and fixtures
TEST_DATA_DIR = Path(__file__).parent / "test_data"

__all__ = [
    "TEST_DATA_DIR"
]
