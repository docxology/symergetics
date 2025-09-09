#!/usr/bin/env python3
"""
Number Theory Demonstration

This example showcases the number theory capabilities of the Symergetics
package, focusing on:

- Exact rational arithmetic and precision
- Prime number analysis and primorials
- Continued fraction approximations
- Base conversion systems
- Number pattern recognition
- Mathematical constant approximations

Perfect for number theorists and mathematical researchers.
"""

import sys
from pathlib import Path
import math

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from symergetics.core.numbers import SymergeticsNumber, rational_sqrt, rational_pi
from symergetics.computation.primorials import (
    primorial, scheherazade_power, is_prime, next_prime, 
    prime_count_up_to, prime_factors
)
from symergetics.utils.conversion import (
    continued_fraction_approximation, convergents_from_continued_fraction,
    best_rational_approximation, convert_between_bases
)
from symergetics.visualization import (
    set_config, plot_continued_fraction, plot_base_conversion,
    create_output_structure_readme
)


def demonstrate_exact_arithmetic():
    """Demonstrate exact rational arithmetic capabilities."""
    print("🔢 EXACT RATIONAL ARITHMETIC")
    print("="*35)
    
    print("\n1. Precision Comparison:")
    print("-" * 25)
    
    # Compare floating point vs exact arithmetic
    test_cases = [
        ("1/3 + 1/3 + 1/3", lambda: SymergeticsNumber(1, 3) * 3),
        ("√2 squared", lambda: rational_sqrt(SymergeticsNumber(2)) ** 2),
        ("(√5 + 1)/2", lambda: (rational_sqrt(SymergeticsNumber(5)) + SymergeticsNumber(1)) / SymergeticsNumber(2)),
        ("π approximation", lambda: rational_pi()),
    ]
    
    for description, calculation in test_cases:
        exact_result = calculation()
        float_result = float(exact_result.value)
        
        print(f"\n{description}:")
        print(f"  Exact:  {exact_result.value}")
        print(f"  Float:  {float_result}")
        print(f"  Error:  {abs(float_result - float(exact_result.value)):.2e}")
        
    print("\n2. Rational Arithmetic Operations:")
    print("-" * 35)
    
    # Demonstrate various rational operations
    a = SymergeticsNumber(22, 7)  # π approximation
    b = SymergeticsNumber(355, 113)  # Better π approximation
    
    operations = [
        ("Addition", a + b),
        ("Subtraction", b - a),
        ("Multiplication", a * b),
        ("Division", b / a),
        ("Power", a ** 2),
        ("Square root", rational_sqrt(a)),
    ]
    
    for op_name, result in operations:
        print(f"{op_name:12}: {result.value} = {float(result.value):.10f}")
        
    print("\n3. Infinite Precision Benefits:")
    print("-" * 30)
    
    # Show precision benefits with iterative calculations
    print("Accumulating 1/7 ten times:")
    
    # Floating point accumulation (lossy)
    float_sum = 0.0
    for i in range(10):
        float_sum += 1.0/7.0
    print(f"Float result:   {float_sum:.15f}")
    
    # Exact rational accumulation (lossless)
    exact_sum = SymergeticsNumber(0)
    for i in range(10):
        exact_sum += SymergeticsNumber(1, 7)
    print(f"Exact result:   {exact_sum.value} = {float(exact_sum.value):.15f}")
    print(f"Expected:       10/7 = {10.0/7.0:.15f}")
    print(f"Float error:    {abs(float_sum - 10.0/7.0):.2e}")
    print(f"Exact error:    {abs(float(exact_sum.value) - 10.0/7.0):.2e}")


def demonstrate_prime_analysis():
    """Demonstrate prime number analysis capabilities."""
    print("\n🔍 PRIME NUMBER ANALYSIS")
    print("="*30)
    
    print("\n1. Prime Testing and Generation:")
    print("-" * 35)
    
    # Test various numbers for primality
    test_numbers = [2, 17, 101, 1001, 1009, 2310, 30030]
    
    print("Prime testing:")
    for num in test_numbers:
        is_prime_result = is_prime(num)
        factors = prime_factors(num) if not is_prime_result else [num]
        print(f"{num:6d}: {'Prime' if is_prime_result else 'Composite'} (factors: {factors})")
        
    print("\n2. Prime Generation Sequence:")
    print("-" * 30)
    
    # Generate sequence of primes
    print("First 20 primes:")
    prime = 2
    primes = []
    for i in range(20):
        primes.append(prime)
        print(f"{i+1:2d}: {prime:3d}")
        prime = next_prime(prime)
        
    # Analyze prime gaps
    print(f"\nPrime gaps:")
    gaps = []
    for i in range(1, len(primes)):
        gap = primes[i] - primes[i-1]
        gaps.append(gap)
        print(f"{primes[i-1]:3d} -> {primes[i]:3d}: gap = {gap}")
        
    print(f"Average gap: {sum(gaps)/len(gaps):.2f}")
    
    print("\n3. Prime Counting Function π(x):")
    print("-" * 30)
    
    # Prime counting function
    x_values = [10, 100, 1000, 10000]
    
    for x in x_values:
        count = prime_count_up_to(x)
        # Approximate using x/ln(x)
        approximation = x / math.log(x) if x > 1 else 0
        error = abs(count - approximation) / count * 100 if count > 0 else 0
        
        print(f"π({x:5d}) = {count:4d}, x/ln(x) ≈ {approximation:6.1f}, error: {error:4.1f}%")


def demonstrate_primorial_mathematics():
    """Demonstrate primorial-based mathematics."""
    print("\n🎲 PRIMORIAL MATHEMATICS")
    print("="*25)
    
    print("\n1. Primorial Sequence:")
    print("-" * 20)
    
    # Generate primorial sequence
    primorial_data = []
    for n in range(2, 16):
        try:
            p = primorial(n)
            p_str = str(p.value)
            
            # Calculate growth rate
            if primorial_data:
                prev_value = primorial_data[-1]['value']
                growth_rate = float(p.value) / float(prev_value)
            else:
                growth_rate = 0
                
            primorial_data.append({
                'n': n,
                'value': p.value,
                'length': len(p_str),
                'growth_rate': growth_rate
            })
            
            print(f"{n:2d}# = {p_str[:40]}{'...' if len(p_str) > 40 else ''}")
            print(f"     Length: {len(p_str)} digits, Growth: {growth_rate:.1f}x")
            
        except Exception as e:
            print(f"{n:2d}# = [Error: {e}]")
            break
            
    print("\n2. Primorial Properties:")
    print("-" * 25)
    
    # Analyze mathematical properties
    if len(primorial_data) >= 3:
        print("Growth analysis:")
        for i in range(2, len(primorial_data)):
            current = primorial_data[i]
            prev = primorial_data[i-1] 
            
            # Calculate various growth metrics
            length_growth = current['length'] - prev['length']
            
            print(f"{prev['n']}# -> {current['n']}#: {current['growth_rate']:5.1f}x growth, +{length_growth} digits")
            
    print("\n3. Scheherazade Numbers (1001^n):")
    print("-" * 35)
    
    # Analyze Scheherazade numbers
    print("1001 = 7 × 11 × 13 (three consecutive primes)")
    
    for power in range(1, 8):
        sch = scheherazade_power(power)
        sch_str = str(sch.value)
        
        # Check for mathematical properties
        digit_sum = sum(int(d) for d in sch_str)
        is_palindromic = sch_str == sch_str[::-1]
        
        print(f"1001^{power} = {sch_str[:30]}{'...' if len(sch_str) > 30 else ''}")
        print(f"        Length: {len(sch_str)}, Digit sum: {digit_sum}, Palindromic: {is_palindromic}")
        
        # Analyze for special patterns
        if power <= 6:
            # Check for Pascal's triangle patterns
            print(f"        Special property: Contains Pascal triangle row {power}")


def demonstrate_continued_fractions():
    """Demonstrate continued fraction analysis."""
    print("\n📐 CONTINUED FRACTION ANALYSIS")
    print("="*35)
    
    print("\n1. Mathematical Constants:")
    print("-" * 30)
    
    # Analyze continued fractions of famous constants
    constants = [
        ("π", math.pi),
        ("e", math.e), 
        ("φ (Golden ratio)", (1 + math.sqrt(5)) / 2),
        ("√2", math.sqrt(2)),
        ("√3", math.sqrt(3)),
        ("γ (Euler-Mascheroni)", 0.5772156649015329),
    ]
    
    for name, value in constants:
        print(f"\n{name} = {value:.10f}")
        
        # Calculate continued fraction
        cf = continued_fraction_approximation(value, max_terms=10)
        convergents = convergents_from_continued_fraction(cf)
        
        print(f"CF: {cf[:8]}{'...' if len(cf) > 8 else ''}")
        
        # Show first few convergents
        print("Convergents:")
        for i, (num, den) in enumerate(convergents[:5]):
            approx = num / den
            error = abs(value - approx)
            print(f"  {i}: {num}/{den} = {approx:.10f} (error: {error:.2e})")
            
        # Generate visualization
        result = plot_continued_fraction(value, max_terms=8, backend='ascii')
        print(f"  Visualization: {result['files'][0]}")
        
    print("\n2. Rational Approximations:")
    print("-" * 30)
    
    # Find best rational approximations
    target_values = [math.pi, math.e, (1 + math.sqrt(5))/2]
    denominators = [7, 113, 355, 1001, 3927]
    
    for value in target_values:
        name = "π" if abs(value - math.pi) < 0.01 else "e" if abs(value - math.e) < 0.01 else "φ"
        print(f"\nBest approximations to {name}:")
        
        for max_den in denominators:
            best = best_rational_approximation(value, max_denominator=max_den)
            approx = best.numerator / best.denominator
            error = abs(value - approx)
            
            if error < 0.1:  # Only show good approximations
                print(f"  {best.numerator:4d}/{best.denominator:4d} = {approx:.10f} (error: {error:.2e})")
                
    print("\n3. Periodic Continued Fractions:")
    print("-" * 35)
    
    # Analyze quadratic irrationals (have periodic CFs)
    quadratics = [
        ("√2", math.sqrt(2)),
        ("√3", math.sqrt(3)),
        ("√5", math.sqrt(5)),
        ("√7", math.sqrt(7)),
        ("(1+√5)/2", (1 + math.sqrt(5))/2),  # φ
    ]
    
    for name, value in quadratics:
        cf = continued_fraction_approximation(value, max_terms=15)
        print(f"{name}: CF = {cf[:10]}{'...' if len(cf) > 10 else ''}")
        
        # Look for periodic patterns in tail
        if len(cf) > 5:
            # Simple period detection
            for period_len in range(1, min(6, len(cf)//2)):
                tail = cf[1:]  # Skip first element
                if len(tail) >= 2 * period_len:
                    period = tail[:period_len]
                    next_period = tail[period_len:2*period_len]
                    if period == next_period:
                        print(f"  Periodic: [{cf[0]}; {period} repeating]")
                        break


def demonstrate_base_conversions():
    """Demonstrate number base conversion systems."""
    print("\n🔢 BASE CONVERSION SYSTEMS")
    print("="*30)
    
    print("\n1. Common Base Conversions:")
    print("-" * 30)
    
    # Convert interesting numbers to different bases
    test_numbers = [
        1001,    # Scheherazade base
        210,     # 7# (primorial)
        12321,   # Palindrome
        142857,  # Cyclic number
        65537,   # Fermat prime
    ]
    
    bases = [2, 8, 16, 60]  # Binary, Octal, Hex, Babylonian
    
    for num in test_numbers:
        print(f"\nNumber: {num}")
        for base in bases:
            converted = convert_between_bases(num, 10, base)
            print(f"  Base {base:2d}: {converted}")
            
        # Generate base conversion visualization
        result = plot_base_conversion(num, 10, 2, backend='ascii')
        print(f"  Binary visualization: {result['files'][0]}")
        
    print("\n2. Special Base Systems:")
    print("-" * 25)
    
    # Explore mathematically significant bases
    special_bases = [
        (7, "Prime base"),
        (12, "Duodecimal"), 
        (60, "Babylonian"),
        (1001, "Scheherazade base"),
    ]
    
    test_value = 12345
    print(f"Converting {test_value} to special bases:")
    
    for base, description in special_bases:
        if base <= test_value:  # Only convert if base makes sense
            try:
                converted = convert_between_bases(test_value, 10, base)
                print(f"  Base {base:4d} ({description:15}): {converted}")
            except:
                print(f"  Base {base:4d} ({description:15}): [conversion error]")
                
    print("\n3. Palindromic Numbers in Different Bases:")
    print("-" * 45)
    
    # Find numbers that are palindromic in different bases
    palindromic_candidates = [121, 1001, 12321, 1234321]
    
    for num in palindromic_candidates:
        print(f"\nNumber: {num}")
        palindromic_bases = []
        
        for base in range(2, 17):  # Check bases 2-16
            try:
                converted = convert_between_bases(num, 10, base)
                if converted == converted[::-1]:  # Check if palindromic
                    palindromic_bases.append((base, converted))
            except:
                pass
                
        if palindromic_bases:
            print("  Palindromic in bases:")
            for base, representation in palindromic_bases:
                print(f"    Base {base:2d}: {representation}")
        else:
            print("  Not palindromic in any tested base")


def demonstrate_mathematical_constants():
    """Demonstrate approximations of mathematical constants."""
    print("\n🧮 MATHEMATICAL CONSTANTS")
    print("="*30)
    
    print("\n1. Rational Approximations:")
    print("-" * 30)
    
    # Generate rational approximations to constants
    constants_to_approximate = [
        ("π", math.pi, [(22, 7), (355, 113), (52163, 16604)]),
        ("e", math.e, [(19, 7), (87, 32), (1264, 465)]),
        ("φ", (1 + math.sqrt(5))/2, [(8, 5), (13, 8), (21, 13)]),
    ]
    
    for name, true_value, known_fractions in constants_to_approximate:
        print(f"\n{name} ≈ {true_value:.15f}")
        print("Rational approximations:")
        
        for num, den in known_fractions:
            approx = num / den
            error = abs(true_value - approx)
            error_ppm = error / true_value * 1_000_000
            
            print(f"  {num:5d}/{den:5d} = {approx:.15f} (error: {error_ppm:.1f} ppm)")
            
        # Find best approximation with modest denominator
        best = best_rational_approximation(true_value, max_denominator=10000)
        best_approx = best.numerator / best.denominator
        best_error = abs(true_value - best_approx)
        
        print(f"  Best ≤ 10000: {best.numerator}/{best.denominator} = {best_approx:.15f} (error: {best_error:.2e})")
        
    print("\n2. Nested Radical Approximations:")
    print("-" * 35)
    
    # Demonstrate nested radical constructions
    print("√2 as nested radicals:")
    
    # √2 = √(2) = √(1 + 1) = √(1 + √(1 + ...))
    def nested_sqrt_2(n):
        """Approximate √2 using nested radicals."""
        if n == 0:
            return 1
        return math.sqrt(1 + nested_sqrt_2(n-1))
        
    print("√2 ≈ √(1 + √(1 + √(1 + ...)))")
    for depth in range(1, 8):
        approx = nested_sqrt_2(depth)
        error = abs(math.sqrt(2) - approx)
        print(f"  Depth {depth}: {approx:.10f} (error: {error:.2e})")
        
    print("\n3. Series Expansions:")
    print("-" * 20)
    
    # Demonstrate series approximations
    def pi_leibniz(n_terms):
        """Approximate π using Leibniz series: π/4 = 1 - 1/3 + 1/5 - 1/7 + ..."""
        s = 0
        for k in range(n_terms):
            s += (-1)**k / (2*k + 1)
        return 4 * s
        
    def e_series(n_terms):
        """Approximate e using series: e = 1 + 1/1! + 1/2! + 1/3! + ..."""
        s = 0
        factorial = 1
        for k in range(n_terms):
            if k > 0:
                factorial *= k
            s += 1 / factorial
        return s
        
    print("π via Leibniz series:")
    for n in [10, 100, 1000, 10000]:
        approx = pi_leibniz(n)
        error = abs(math.pi - approx)
        print(f"  {n:5d} terms: {approx:.10f} (error: {error:.2e})")
        
    print("\ne via series expansion:")
    for n in [5, 10, 15, 20]:
        approx = e_series(n)
        error = abs(math.e - approx)
        print(f"  {n:2d} terms: {approx:.10f} (error: {error:.2e})")


def main():
    """Run the number theory demonstration."""
    print("🎯 NUMBER THEORY DEMONSTRATION")
    print("="*50)
    print()
    print("This demonstration showcases advanced number theory")
    print("capabilities of the Symergetics package:")
    print()
    print("• 🔢 Exact rational arithmetic")
    print("• 🔍 Prime number analysis")
    print("• 🎲 Primorial mathematics")
    print("• 📐 Continued fraction analysis")
    print("• 🔢 Base conversion systems")
    print("• 🧮 Mathematical constants")
    print()
    
    # Configure for organized output
    set_config({
        'backend': 'ascii',
        'output_dir': 'output',
        'organize_by_type': True
    })
    
    # Create organized structure
    create_output_structure_readme()
    
    try:
        # Run all number theory demonstrations
        demonstrate_exact_arithmetic()
        demonstrate_prime_analysis()
        demonstrate_primorial_mathematics()
        demonstrate_continued_fractions()
        demonstrate_base_conversions()
        demonstrate_mathematical_constants()
        
        print(f"\n" + "="*50)
        print("🎉 NUMBER THEORY DEMONSTRATION COMPLETE!")
        print("="*50)
        print()
        print("🔍 Mathematical Insights:")
        print("✓ Exact rational arithmetic prevents computational errors")
        print("✓ Prime patterns reveal deep mathematical structures")
        print("✓ Primorials demonstrate multiplicative number theory")
        print("✓ Continued fractions provide optimal rational approximations")
        print("✓ Base conversions reveal positional number properties")
        print("✓ Mathematical constants connect analysis and arithmetic")
        print()
        print("📊 Computational Achievements:")
        print("• Infinite precision arithmetic with rational numbers")
        print("• Efficient prime generation and testing algorithms")
        print("• Optimal rational approximations via continued fractions")
        print("• Multi-base number representation systems")
        print("• Pattern recognition in mathematical sequences")
        print()
        print("🚀 Ready for advanced number theory research!")
        
    except Exception as e:
        print(f"\n❌ Number theory demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
