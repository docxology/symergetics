"""
Tests for computation.primorials module - Primorial and Scheherazade calculations.

Tests primorial calculations, Scheherazade numbers, and related mathematical functions.
"""

import pytest
from symergetics.computation.primorials import (
    primorial,
    scheherazade_power,
    factorial_decline,
    cosmic_abundance_factors,
    scheherazade_pascal_coefficients,
    prime_factors,
    is_prime,
    next_prime,
    prime_count_up_to
)
from symergetics.core.numbers import SymergeticsNumber


class TestPrimorial:
    """Test primorial calculations."""

    def test_primorial_basic(self):
        """Test basic primorial calculations."""
        # 2# = 2
        assert primorial(2).value == 2

        # 3# = 2 × 3 = 6
        assert primorial(3).value == 6

        # 5# = 2 × 3 × 5 = 30
        assert primorial(5).value == 30

        # 7# = 2 × 3 × 5 × 7 = 210
        assert primorial(7).value == 210

    def test_primorial_13(self):
        """Test the important 13# primorial."""
        p13 = primorial(13)
        expected = 2 * 3 * 5 * 7 * 11 * 13
        assert p13.value == expected
        assert p13.value == 30030

    def test_primorial_large(self):
        """Test larger primorial calculations."""
        p17 = primorial(17)
        expected = 2 * 3 * 5 * 7 * 11 * 13 * 17
        assert p17.value == expected
        assert p17.value == 510510

    def test_primorial_edge_cases(self):
        """Test primorial edge cases."""
        # n < 2 should return 1
        assert primorial(1).value == 1

        # Test with SymergeticsNumber input
        result = primorial(SymergeticsNumber(5))
        assert result.value == 30


class TestScheherazadePower:
    """Test Scheherazade number calculations."""

    def test_scheherazade_base(self):
        """Test Scheherazade base (1001^1)."""
        s1 = scheherazade_power(1)
        assert s1.value == 1001

    def test_scheherazade_square(self):
        """Test Scheherazade square (1001^2)."""
        s2 = scheherazade_power(2)
        assert s2.value == 1001 * 1001
        assert s2.value == 1002001

    def test_scheherazade_cube(self):
        """Test Scheherazade cube (1001^3)."""
        s3 = scheherazade_power(3)
        assert s3.value == 1001 ** 3
        assert s3.value == 1003003001

    def test_scheherazade_power_6(self):
        """Test the famous 1001^6 with Pascal's triangle."""
        s6 = scheherazade_power(6)
        expected = 1001 ** 6

        # Check it's the right magnitude
        assert s6.value == expected

        # Check it contains Pascal's triangle coefficients
        num_str = str(s6.value)
        # Should contain patterns like 1, 6, 15, 20, 15, 6, 1
        assert '1006015' in num_str  # Part of the pattern

    def test_scheherazade_pascal_coefficients(self):
        """Test extraction of Pascal's coefficients."""
        coeffs = scheherazade_pascal_coefficients(6)
        expected = [1, 6, 15, 20, 15, 6, 1]
        assert coeffs == expected

    def test_scheherazade_power_zero(self):
        """Test 1001^0 = 1."""
        s0 = scheherazade_power(0)
        assert s0.value == 1

    def test_scheherazade_power_with_number(self):
        """Test with SymergeticsNumber input."""
        result = scheherazade_power(SymergeticsNumber(2))
        assert result.value == 1002001


class TestFactorialDecline:
    """Test factorial decline sequences."""

    def test_factorial_decline_basic(self):
        """Test basic factorial decline."""
        sequence = factorial_decline(3)
        # Should be [6, 3, 1] for n=3
        # 3! = 6, then 6/3 = 3, then 3/2 = 1.5, but this might be integer only
        assert len(sequence) == 4  # n+1 elements
        assert sequence[0].value == 6  # 3!

    def test_factorial_decline_4(self):
        """Test factorial decline for n=4."""
        sequence = factorial_decline(4)
        assert len(sequence) == 5
        assert sequence[0].value == 24  # 4!


class TestCosmicAbundance:
    """Test cosmic abundance number calculations."""

    def test_cosmic_abundance_factors(self):
        """Test the 14-illion cosmic abundance number."""
        factors = cosmic_abundance_factors()

        assert factors['factor_count'] == 14
        assert len(factors['primes']) == 14
        assert len(factors['exponents']) == 14

        # Check some key primes and exponents
        assert 2 in factors['primes']
        assert 3 in factors['primes']
        assert 13 in factors['primes']

        # Verify the final number
        abundance = factors['cosmic_abundance']
        # The 14-illion number is actually around 3.1 × 10^42, not 10^100
        assert abundance.value > 10**40  # Should be very large

    def test_cosmic_abundance_known_factors(self):
        """Test specific factors in cosmic abundance."""
        factors = cosmic_abundance_factors()

        # 2^12
        index_2 = factors['primes'].index(2)
        assert factors['exponents'][index_2] == 12

        # 3^8
        index_3 = factors['primes'].index(3)
        assert factors['exponents'][index_3] == 8

        # 13^6
        index_13 = factors['primes'].index(13)
        assert factors['exponents'][index_13] == 6


class TestPrimeUtilities:
    """Test prime-related utility functions."""

    def test_is_prime(self):
        """Test prime detection."""
        assert is_prime(2) == True
        assert is_prime(3) == True
        assert is_prime(5) == True
        assert is_prime(7) == True
        assert is_prime(11) == True
        assert is_prime(13) == True

        assert is_prime(1) == False
        assert is_prime(4) == False
        assert is_prime(6) == False
        assert is_prime(8) == False
        assert is_prime(9) == False

    def test_next_prime(self):
        """Test next prime calculation."""
        assert next_prime(2) == 3
        assert next_prime(3) == 5
        assert next_prime(5) == 7
        assert next_prime(7) == 11
        assert next_prime(11) == 13

    def test_prime_count_up_to(self):
        """Test prime counting."""
        assert prime_count_up_to(10) == 4  # 2, 3, 5, 7
        assert prime_count_up_to(20) == 8  # + 11, 13, 17, 19

    def test_prime_factors(self):
        """Test prime factorization."""
        # Prime number
        assert prime_factors(13) == {13: 1}

        # Composite numbers
        assert prime_factors(12) == {2: 2, 3: 1}
        assert prime_factors(30) == {2: 1, 3: 1, 5: 1}
        assert prime_factors(100) == {2: 2, 5: 2}


class TestPrecomputedValues:
    """Test precomputed values in COMMON_* dictionaries."""

    def test_common_primorials(self):
        """Test common primorial values."""
        from symergetics.computation.primorials import COMMON_PRIMORIALS

        assert COMMON_PRIMORIALS[2].value == 2
        assert COMMON_PRIMORIALS[3].value == 6
        assert COMMON_PRIMORIALS[5].value == 30
        assert COMMON_PRIMORIALS[13].value == 30030

    def test_common_scheherazade(self):
        """Test common Scheherazade values."""
        from symergetics.computation.primorials import COMMON_SCHEHERAZADE

        assert COMMON_SCHEHERAZADE[1].value == 1001
        assert COMMON_SCHEHERAZADE[2].value == 1002001
        assert COMMON_SCHEHERAZADE[3].value == 1003003001


if __name__ == "__main__":
    pytest.main([__file__])
