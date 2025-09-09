#!/usr/bin/env python3
"""
Computation Primorials Coverage Tests

This module contains tests to improve code coverage for computation.primorials module,
focusing on the missing lines identified in the coverage report.
"""

import pytest
import math
from symergetics.core.numbers import SymergeticsNumber
from symergetics.computation.primorials import (
    primorial, scheherazade_power, factorial_decline,
    primorial_sequence, _get_primes_up_to, is_prime, next_prime,
    prime_count_up_to, prime_factors
)


class TestPrimorialErrorCases:
    """Test error cases in primorial function."""

    def test_primorial_fractional_symergetics_number(self):
        """Test primorial with fractional SymergeticsNumber."""
        fractional = SymergeticsNumber(3, 2)  # 3/2

        with pytest.raises(ValueError, match="Primorial requires integer input"):
            primorial(fractional)

    def test_primorial_negative_integer(self):
        """Test primorial with negative integer."""
        # This should work as primorial(1) = 1
        result = primorial(-1)
        assert result == SymergeticsNumber(1)

    def test_primorial_zero(self):
        """Test primorial with zero."""
        result = primorial(0)
        assert result == SymergeticsNumber(1)

    def test_primorial_one(self):
        """Test primorial with one."""
        result = primorial(1)
        assert result == SymergeticsNumber(1)


class TestScheherazadePowerErrorCases:
    """Test error cases in scheherazade_power function."""

    def test_scheherazade_power_fractional_symergetics_number(self):
        """Test scheherazade_power with fractional SymergeticsNumber."""
        fractional = SymergeticsNumber(5, 2)  # 5/2

        with pytest.raises(ValueError, match="Scheherazade power requires integer input"):
            scheherazade_power(fractional)

    def test_scheherazade_power_zero(self):
        """Test scheherazade_power with zero."""
        result = scheherazade_power(0)
        assert result == SymergeticsNumber(1)  # 1001^0 = 1

    def test_scheherazade_power_one(self):
        """Test scheherazade_power with one."""
        result = scheherazade_power(1)
        assert result == SymergeticsNumber(1001)  # 1001^1 = 1001

    def test_scheherazade_power_negative(self):
        """Test scheherazade_power with negative exponent."""
        result = scheherazade_power(-1)
        assert result == SymergeticsNumber(1) / SymergeticsNumber(1001)  # 1001^(-1) = 1/1001


class TestFactorialDeclineErrorCases:
    """Test error cases in factorial_decline function."""

    def test_factorial_decline_negative(self):
        """Test factorial_decline with negative input."""
        with pytest.raises(ValueError, match="n must be non-negative"):
            factorial_decline(-1)

    def test_factorial_decline_zero(self):
        """Test factorial_decline with zero."""
        result = factorial_decline(0)
        assert len(result) == 1
        assert result[0] == SymergeticsNumber(1)  # 0! = 1

    def test_factorial_decline_one(self):
        """Test factorial_decline with one."""
        result = factorial_decline(1)
        assert len(result) == 2
        assert result[0] == SymergeticsNumber(1)  # 0! = 1
        assert result[1] == SymergeticsNumber(1)  # 1! = 1

    def test_factorial_decline_small_n(self):
        """Test factorial_decline with small n."""
        result = factorial_decline(3)
        assert len(result) == 4
        assert result[0] == SymergeticsNumber(6)     # 3!
        assert result[1] == SymergeticsNumber(2)     # 3!/3 = 2
        assert result[2] == SymergeticsNumber(1)     # 2/2 = 1
        assert result[3] == SymergeticsNumber(1)     # 1/1 = 1


class TestPrimorialSequenceFunction:
    """Test primorial_sequence function."""

    def test_primorial_sequence_small_max_n(self):
        """Test primorial_sequence with small max_n."""
        result = primorial_sequence(max_n=5)

        # Should have entries for n=2,3,4,5
        assert 2 in result
        assert 3 in result
        assert 4 in result
        assert 5 in result

        # Check specific values
        assert result[2] == SymergeticsNumber(2)    # First prime
        assert result[3] == SymergeticsNumber(6)    # 2*3
        assert result[4] == SymergeticsNumber(6)    # Still 2*3 (4 is not prime)
        assert result[5] == SymergeticsNumber(30)   # 2*3*5

    def test_primorial_sequence_larger_max_n(self):
        """Test primorial_sequence with larger max_n."""
        result = primorial_sequence(max_n=10)

        # Should have more entries
        assert len(result) >= 9  # n from 2 to 10

        # Check that primorials are non-decreasing
        prev_value = SymergeticsNumber(1)
        for n in range(2, 11):
            assert n in result
            assert result[n] >= prev_value
            prev_value = result[n]

    def test_primorial_sequence_max_n_less_than_2(self):
        """Test primorial_sequence with max_n < 2."""
        result = primorial_sequence(max_n=1)

        # Should return empty dict
        assert result == {}


class TestPrimesUpToFunction:
    """Test _get_primes_up_to function."""

    def test_get_primes_up_to_small_n(self):
        """Test _get_primes_up_to with small n."""
        result = _get_primes_up_to(10)

        expected = [2, 3, 5, 7]
        assert result == expected

    def test_get_primes_up_to_twenty(self):
        """Test _get_primes_up_to with n=20."""
        result = _get_primes_up_to(20)

        expected = [2, 3, 5, 7, 11, 13, 17, 19]
        assert result == expected

    def test_get_primes_up_to_one(self):
        """Test _get_primes_up_to with n=1."""
        result = _get_primes_up_to(1)

        # Should return empty list
        assert result == []

    def test_get_primes_up_to_zero(self):
        """Test _get_primes_up_to with n=0."""
        result = _get_primes_up_to(0)

        # Should return empty list
        assert result == []


class TestPrimeUtilityFunctions:
    """Test prime utility functions."""

    def test_is_prime_edge_cases(self):
        """Test is_prime with edge cases."""
        assert is_prime(2) == True
        assert is_prime(3) == True
        assert is_prime(4) == False
        assert is_prime(1) == False
        assert is_prime(0) == False
        assert is_prime(-1) == False

    def test_next_prime_basic(self):
        """Test next_prime with basic cases."""
        assert next_prime(2) == 3
        assert next_prime(3) == 5
        assert next_prime(4) == 5
        assert next_prime(5) == 7
        assert next_prime(6) == 7

    def test_next_prime_large_gap(self):
        """Test next_prime with larger gaps."""
        assert next_prime(13) == 17  # Gap after 13
        assert next_prime(17) == 19  # Gap after 17

    def test_prime_count_up_to_various_n(self):
        """Test prime_count_up_to with various n."""
        assert prime_count_up_to(1) == 0
        assert prime_count_up_to(2) == 1
        assert prime_count_up_to(10) == 4  # 2, 3, 5, 7
        assert prime_count_up_to(20) == 8  # 2, 3, 5, 7, 11, 13, 17, 19

    def test_prime_factors_basic(self):
        """Test prime_factors with basic numbers."""
        assert prime_factors(1) == {}
        assert prime_factors(2) == {2: 1}
        assert prime_factors(3) == {3: 1}
        assert prime_factors(4) == {2: 2}
        assert prime_factors(6) == {2: 1, 3: 1}
        assert prime_factors(8) == {2: 3}
        assert prime_factors(9) == {3: 2}
        assert prime_factors(12) == {2: 2, 3: 1}

    def test_prime_factors_prime_number(self):
        """Test prime_factors with prime number."""
        assert prime_factors(13) == {13: 1}
        assert prime_factors(17) == {17: 1}

    def test_prime_factors_large_number(self):
        """Test prime_factors with larger number."""
        result = prime_factors(30)
        assert result == {2: 1, 3: 1, 5: 1}


class TestPrimorialLargeValues:
    """Test primorial with larger values."""

    def test_primorial_medium_values(self):
        """Test primorial with medium n values."""
        result_5 = primorial(5)
        assert result_5 == SymergeticsNumber(2 * 3 * 5)  # 30

        result_7 = primorial(7)
        assert result_7 == SymergeticsNumber(2 * 3 * 5 * 7)  # 210

    def test_scheherazade_power_medium_values(self):
        """Test scheherazade_power with medium exponents."""
        result_2 = scheherazade_power(2)
        expected_2 = SymergeticsNumber(1001) ** 2
        assert result_2 == expected_2

        result_3 = scheherazade_power(3)
        expected_3 = SymergeticsNumber(1001) ** 3
        assert result_3 == expected_3


class TestIntegrationWithSymergeticsNumber:
    """Test integration between primorial functions and SymergeticsNumber."""

    def test_primorial_with_symergetics_number_integer(self):
        """Test primorial with integer SymergeticsNumber."""
        num = SymergeticsNumber(4)
        result = primorial(num)
        expected = primorial(4)
        assert result == expected

    def test_scheherazade_power_with_symergetics_number_integer(self):
        """Test scheherazade_power with integer SymergeticsNumber."""
        num = SymergeticsNumber(3)
        result = scheherazade_power(num)
        expected = scheherazade_power(3)
        assert result == expected

    def test_factorial_decline_with_various_inputs(self):
        """Test factorial_decline with various valid inputs."""
        # Test n=2
        result_2 = factorial_decline(2)
        assert len(result_2) == 3
        assert result_2[2] == SymergeticsNumber(1)  # 2!/2/1 = 1

        # Test n=4
        result_4 = factorial_decline(4)
        assert len(result_4) == 5
        assert result_4[4] == SymergeticsNumber(1)  # 4!/4/3/2/1 = 1
