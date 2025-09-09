#!/usr/bin/env python3
"""
Mathematical Patterns Demonstration

This example showcases the advanced mathematical pattern analysis capabilities
of the Symergetics package, including:

- Palindromic number analysis
- Scheherazade numbers (powers of 1001) 
- Primorial distributions
- Pattern recognition and entropy analysis
- Number theory connections

Perfect for mathematicians and researchers interested in number patterns.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from symergetics.core.numbers import SymergeticsNumber
from symergetics.computation.palindromes import (
    is_palindromic, extract_palindromic_patterns, analyze_scheherazade_ssrcd,
    calculate_palindromic_density, find_symmetric_patterns
)
from symergetics.computation.primorials import primorial, scheherazade_power, is_prime, prime_factors
from symergetics.visualization import (
    set_config, plot_palindromic_pattern, plot_scheherazade_pattern,
    plot_primorial_distribution, create_output_structure_readme
)
import math


def analyze_palindromic_patterns():
    """Analyze various palindromic number patterns."""
    print("🔄 PALINDROMIC PATTERN ANALYSIS")
    print("="*50)
    
    print("\n1. Perfect Palindromes:")
    print("-" * 30)
    
    perfect_palindromes = [
        121, 12321, 1234321, 123454321, 12345654321,
        1001, 1002001, 100030001
    ]
    
    for num in perfect_palindromes:
        is_pal = is_palindromic(num)
        patterns = extract_palindromic_patterns(num)
        density = calculate_palindromic_density(num)
        
        print(f"{num:>12}: palindromic={is_pal}, patterns={len(patterns)}, density={density:.1%}")
        
        # Generate visualization
        result = plot_palindromic_pattern(num, backend='ascii')
        print(f"          -> {result['files'][0]}")
        
    print("\n2. Scheherazade Palindromic Properties:")
    print("-" * 40)
    
    # Analyze Scheherazade numbers for palindromic properties
    for power in range(1, 7):
        sch_num = scheherazade_power(power)
        is_pal = is_palindromic(sch_num)
        
        analysis = analyze_scheherazade_ssrcd(power)
        
        print(f"1001^{power} = {str(sch_num.value)[:20]}{'...' if len(str(sch_num.value)) > 20 else ''}")
        print(f"   Length: {len(str(sch_num.value))} digits")
        print(f"   Palindromic: {is_pal}")
        print(f"   Density: {analysis['palindromic_density']:.1%}")
        
        if 'special_insight' in analysis:
            insight = analysis['special_insight']
            print(f"   Special: {insight['description']}")
            
        # Generate visualization
        result = plot_scheherazade_pattern(power, backend='ascii')
        print(f"   -> {result['files'][0]}")
        print()
        
    print("\n3. Symmetric Pattern Analysis:")
    print("-" * 30)
    
    # Analyze symmetric patterns
    test_numbers = [12321, 1234321, 11223311, 142857142857]
    
    for num in test_numbers:
        symmetries = find_symmetric_patterns(num)
        print(f"{num}:")
        for sym in symmetries:
            print(f"   Type: {sym['type']}")
            if 'pairs' in sym:
                print(f"   Pairs: {len(sym['pairs'])} symmetric pairs")
        print()


def analyze_primorial_patterns():
    """Analyze primorial number patterns and distributions."""
    print("\n🔢 PRIMORIAL PATTERN ANALYSIS")
    print("="*50)
    
    print("\n1. Primorial Sequence and Properties:")
    print("-" * 40)
    
    primorial_data = []
    
    for n in range(2, 20):
        try:
            p = primorial(n)
            p_str = str(p.value)
            
            # Analyze digit patterns
            digits = [int(d) for d in p_str]
            digit_counts = {}
            for d in digits:
                digit_counts[d] = digit_counts.get(d, 0) + 1
                
            # Calculate entropy
            total_digits = len(digits)
            entropy = 0.0
            for count in digit_counts.values():
                if count > 0:
                    prob = count / total_digits
                    entropy -= prob * math.log2(prob)
                    
            # Check for interesting properties
            has_pattern = is_palindromic(p.value)
            
            primorial_data.append({
                'n': n,
                'value': p.value,
                'length': len(p_str),
                'entropy': entropy,
                'palindromic': has_pattern,
                'digit_counts': digit_counts
            })
            
            print(f"{n:2d}# = {p_str[:30]}{'...' if len(p_str) > 30 else ''}")
            print(f"     Length: {len(p_str)}, Entropy: {entropy:.3f}, Palindromic: {has_pattern}")
            
        except Exception as e:
            print(f"{n:2d}# = [calculation error: {e}]")
            break
            
    print("\n2. Primorial Distribution Analysis:")
    print("-" * 30)
    
    # Generate comprehensive distribution analysis
    result = plot_primorial_distribution(max_n=15, backend='ascii')
    print(f"Distribution analysis: {result['files'][0]}")
    
    # Analyze growth patterns
    if len(primorial_data) > 1:
        print(f"\nGrowth Pattern Analysis:")
        for i in range(1, min(len(primorial_data), 8)):
            current = primorial_data[i]
            previous = primorial_data[i-1]
            
            ratio = current['value'] / previous['value']
            length_growth = current['length'] - previous['length']
            entropy_change = current['entropy'] - previous['entropy']
            
            print(f"{previous['n']}# -> {current['n']}#: ratio={ratio:.1f}, +{length_growth} digits, Δentropy={entropy_change:+.3f}")
            
    print("\n3. Prime Factor Distribution:")
    print("-" * 30)
    
    # Show how prime factors accumulate
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    running_product = 1
    
    print("Prime accumulation in primorials:")
    for prime in primes:
        running_product *= prime
        print(f"× {prime:2d} = {running_product:>12} ({'prime' if is_prime(prime) else 'composite'})")
        if running_product > 10**10:
            break


def analyze_special_number_patterns():
    """Analyze special mathematical number patterns."""
    print("\n✨ SPECIAL NUMBER PATTERN ANALYSIS") 
    print("="*50)
    
    print("\n1. Cyclic Numbers:")
    print("-" * 20)
    
    # Analyze 142857 (1/7 cyclic number)
    cyclic = 142857
    print(f"Cyclic number: {cyclic}")
    print(f"Properties:")
    
    # Show cyclic properties
    for mult in range(1, 7):
        product = cyclic * mult
        print(f"  {cyclic} × {mult} = {product}")
        
    # Check if palindromic patterns exist in multiples
    pal_result = plot_palindromic_pattern(cyclic, backend='ascii')
    print(f"Pattern analysis: {pal_result['files'][0]}")
    
    print("\n2. Repunits (11111...):")
    print("-" * 25)
    
    # Analyze repunit patterns
    repunits = [11, 111, 1111, 11111, 111111]
    
    for rep in repunits:
        factors_dict = prime_factors(rep)
        factors = []
        for prime, count in factors_dict.items():
            factors.extend([prime] * count)
        is_pal = is_palindromic(rep)
        print(f"{rep}: palindromic={is_pal}, factors={factors}")
        
    print("\n3. Powers of Special Numbers:")
    print("-" * 30)
    
    # Analyze powers of interesting numbers
    special_bases = [11, 101, 111, 121]
    
    for base in special_bases:
        print(f"\nPowers of {base}:")
        for power in range(1, 5):
            result = base ** power
            is_pal = is_palindromic(result)
            patterns = extract_palindromic_patterns(result)
            print(f"  {base}^{power} = {result} (palindromic={is_pal}, patterns={len(patterns)})")
            
    print("\n4. Fibonacci-Like Sequences:")
    print("-" * 30)
    
    # Generate and analyze Fibonacci-like patterns
    def analyze_sequence(name, generator, length=10):
        print(f"\n{name} sequence analysis:")
        sequence = []
        for i in range(length):
            num = generator(i)
            sequence.append(num)
            is_pal = is_palindromic(num)
            print(f"  F({i}) = {num} (palindromic={is_pal})")
            
        return sequence
        
    # Traditional Fibonacci
    def fibonacci(n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b
        
    fib_seq = analyze_sequence("Fibonacci", fibonacci, 12)
    
    # Lucas sequence
    def lucas(n):
        if n == 0:
            return 2
        if n == 1:
            return 1
        a, b = 2, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b
        
    lucas_seq = analyze_sequence("Lucas", lucas, 10)


def demonstrate_entropy_analysis():
    """Demonstrate entropy analysis of number patterns."""
    print("\n📊 ENTROPY AND RANDOMNESS ANALYSIS")
    print("="*50)
    
    print("\n1. Digit Distribution Entropy:")
    print("-" * 30)
    
    # Analyze different types of numbers for entropy
    test_numbers = [
        ("Random-like", 1234567890),
        ("Highly structured", 1111111111),
        ("Palindrome", 123454321),
        ("Scheherazade", 1003003001),
        ("Primorial 7#", 210),
        ("π digits", 31415926535897932384626433832795),
    ]
    
    for name, number in test_numbers:
        digits = [int(d) for d in str(number)]
        digit_counts = {}
        for d in digits:
            digit_counts[d] = digit_counts.get(d, 0) + 1
            
        total_digits = len(digits)
        entropy = 0.0
        for count in digit_counts.values():
            if count > 0:
                prob = count / total_digits
                entropy -= prob * math.log2(prob)
                
        max_entropy = math.log2(10)  # Maximum possible for 10 digits
        entropy_ratio = entropy / max_entropy
        
        print(f"{name:15}: {number}")
        print(f"                Entropy: {entropy:.3f} bits ({entropy_ratio:.1%} of maximum)")
        print(f"                Assessment: {'Random-like' if entropy_ratio > 0.9 else 'Highly structured' if entropy_ratio < 0.5 else 'Moderately structured'}")
        print()
        
    print("2. Pattern Complexity Analysis:")
    print("-" * 30)
    
    # Compare complexity of different number types
    complexity_tests = [
        1001**2,   # Scheherazade
        12345**2,  # Sequential
        int(str(primorial(7).value)),  # Primorial
    ]
    
    for num in complexity_tests:
        patterns = extract_palindromic_patterns(num)
        density = calculate_palindromic_density(num)
        
        # Simple complexity measure
        num_str = str(num)
        unique_digits = len(set(num_str))
        digit_variety = unique_digits / 10
        
        print(f"Number: {num}")
        print(f"  Palindromic patterns: {len(patterns)}")
        print(f"  Palindromic density: {density:.1%}")
        print(f"  Digit variety: {unique_digits}/10 ({digit_variety:.1%})")
        print(f"  Complexity index: {(len(patterns) * density * digit_variety):.3f}")
        print()


# Utility functions
# Note: Using symergetics.computation.primorials.is_prime and prime_factors instead of local implementations


def main():
    """Run the mathematical patterns demonstration."""
    print("🎯 MATHEMATICAL PATTERNS ANALYSIS DEMO")
    print("="*60)
    print()
    print("This demonstration showcases advanced mathematical pattern")
    print("analysis capabilities of the Symergetics package:")
    print()
    print("• 🔄 Palindromic number analysis")
    print("• 🔢 Scheherazade number patterns") 
    print("• 🎲 Primorial distributions")
    print("• ✨ Special number properties")
    print("• 📊 Entropy and complexity analysis")
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
        # Run all pattern analyses
        analyze_palindromic_patterns()
        analyze_primorial_patterns()
        analyze_special_number_patterns()
        demonstrate_entropy_analysis()
        
        print(f"\n" + "="*60)
        print("🎉 MATHEMATICAL PATTERNS ANALYSIS COMPLETE!")
        print("="*60)
        print()
        print("🔍 Key Discoveries:")
        print("✓ Palindromic patterns reveal deep mathematical structures")
        print("✓ Scheherazade numbers contain Pascal triangle coefficients")
        print("✓ Primorials show structured digit distributions")
        print("✓ Entropy analysis distinguishes random vs. structured numbers")
        print("✓ Pattern recognition enables mathematical classification")
        print()
        print("📊 Analysis Results:")
        print("• All visualizations saved to organized output/ structure")
        print("• Entropy calculations reveal pattern complexity")
        print("• Palindromic density quantifies structural symmetry")
        print("• Prime factorization patterns uncovered")
        print()
        print("🚀 Ready for advanced number theory research!")
        
    except Exception as e:
        print(f"\n❌ Pattern analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
