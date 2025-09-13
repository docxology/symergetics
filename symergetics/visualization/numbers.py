"""
Number Visualizations for Synergetics

This module provides visualization capabilities for numerical patterns and
relationships in the Synergetics framework, including palindromes, Scheherazade
numbers, primorials, and mnemonic encodings.

Features:
- Palindromic pattern visualization
- Scheherazade number pattern analysis
- Primorial distribution plots
- Mnemonic encoding visualizations
- Pattern recognition heatmaps
- Number sequence animations

Author: Symergetics Team
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from pathlib import Path
import math

from ..core.numbers import SymergeticsNumber
from ..computation.palindromes import (
    is_palindromic, extract_palindromic_patterns,
    find_repeated_patterns, calculate_palindromic_density,
    find_symmetric_patterns, analyze_scheherazade_ssrcd
)
from ..computation.primorials import primorial, scheherazade_power, COMMON_PRIMORIALS
from ..utils.mnemonics import create_memory_aid, format_large_number
from . import _config, ensure_output_dir, get_organized_output_path


def plot_palindromic_pattern(number: Union[int, SymergeticsNumber],
                           backend: Optional[str] = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Visualize palindromic patterns in a number.

    Args:
        number: Number to analyze for palindromic patterns
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_palindrome_matplotlib(number, **kwargs)
    elif backend == "ascii":
        return _plot_palindrome_ascii(number, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_palindrome_matplotlib(number: Union[int, SymergeticsNumber],
                               **kwargs) -> Dict[str, Any]:
    """Plot palindromic patterns using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Get number as string
    if isinstance(number, SymergeticsNumber):
        if number.value.denominator != 1:
            num_str = str(number.value)
        else:
            num_str = str(number.value.numerator)
    else:
        try:
            num_str = str(abs(int(number)))
        except (ValueError, TypeError):
            # Handle invalid input gracefully
            return {
                'files': [],
                'metadata': {
                    'error': f'Invalid input: {number}',
                    'backend': 'matplotlib',
                    'function': 'plot_palindromic_pattern'
                }
            }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(_config["figure_size"][0], _config["figure_size"][1] * 1.5))

    # Top plot: Digit pattern
    digits = [int(d) for d in num_str]
    positions = list(range(len(digits)))

    # Create color map based on palindromic properties
    colors = []
    for i, digit in enumerate(digits):
        # Check if this digit is part of a palindromic pattern
        is_palindrome_digit = False
        for radius in range(1, min(i, len(digits) - 1 - i) + 1):
            if digits[i - radius] == digits[i + radius]:
                is_palindrome_digit = True
                break
        colors.append(_config["colors"]["accent"] if is_palindrome_digit else _config["colors"]["primary"])

    bars = ax1.bar(positions, digits, color=colors, alpha=0.7)
    ax1.set_title(f'Palindromic Pattern Analysis: {format_large_number(int(num_str) if num_str.isdigit() else 0)}')
    ax1.set_xlabel('Digit Position')
    ax1.set_ylabel('Digit Value')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([str(i) for i in range(len(num_str))])

    # Add digit labels on bars
    for i, (pos, digit) in enumerate(zip(positions, digits)):
        ax1.text(pos, digit + 0.1, str(digit), ha='center', va='bottom')

    # Bottom plot: Symmetry heatmap
    symmetry_matrix = np.zeros((len(digits), len(digits)))

    for i in range(len(digits)):
        for j in range(len(digits)):
            if i != j and digits[i] == digits[j]:
                symmetry_matrix[i, j] = 1

    im = ax2.imshow(symmetry_matrix, cmap='Blues', aspect='equal')
    ax2.set_title('Digit Symmetry Heatmap')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Position')

    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Symmetry (1=Same digit)')

    plt.tight_layout()

    # Save plot using organized structure
    filename = f"palindrome_pattern_{num_str[:20]}.png"
    filepath = get_organized_output_path('numbers', 'palindromes', filename)
    fig.savefig(filepath, 
                dpi=_config["dpi"], 
                facecolor=_config["png_options"]["facecolor"],
                bbox_inches=_config["png_options"]["bbox_inches"],
                pad_inches=_config["png_options"]["pad_inches"],
                transparent=_config["png_options"]["transparent"])
    plt.close(fig)

    # Get pattern analysis
    patterns = extract_palindromic_patterns(number)
    density = calculate_palindromic_density(number)
    symmetry = find_symmetric_patterns(number)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'palindrome_pattern',
            'number': num_str,
            'is_palindromic': is_palindromic(number),
            'pattern_count': len(patterns),
            'palindromic_density': density,
            'symmetry_patterns': len(symmetry),
            'backend': 'matplotlib'
        }
    }


def _plot_palindrome_ascii(number: Union[int, SymergeticsNumber],
                          **kwargs) -> Dict[str, Any]:
    """Create ASCII art representation of palindromic patterns."""
    # Get number as string
    if isinstance(number, SymergeticsNumber):
        if number.value.denominator != 1:
            num_str = str(number.value)
        else:
            num_str = str(number.value.numerator)
    else:
        try:
            num_str = str(abs(int(number)))
        except (ValueError, TypeError):
            # Handle invalid input gracefully
            return {
                'files': [],
                'metadata': {
                    'error': f'Invalid input: {number}',
                    'backend': 'ascii',
                    'function': 'plot_palindromic_pattern'
                }
            }

    lines = []
    lines.append(f"Palindromic Pattern Analysis: {format_large_number(int(num_str) if num_str.isdigit() else 0)}")
    lines.append("=" * 60)
    lines.append("")

    # Basic properties
    lines.append(f"Number: {num_str}")
    lines.append(f"Is Palindromic: {is_palindromic(number)}")
    lines.append(f"Length: {len(num_str)} digits")
    lines.append("")

    # Pattern analysis
    patterns = extract_palindromic_patterns(number)
    lines.append(f"Palindromic Patterns Found: {len(patterns)}")
    if patterns:
        lines.append("Patterns (longest first):")
        for pattern in patterns[:10]:  # Show first 10
            lines.append(f"  {pattern}")
        if len(patterns) > 10:
            lines.append(f"  ... and {len(patterns) - 10} more")
    lines.append("")

    # Symmetry analysis
    symmetry = find_symmetric_patterns(number)
    lines.append(f"Symmetric Patterns: {len(symmetry)}")
    for pattern in symmetry:
        lines.append(f"  Type: {pattern['type']}")
        if 'pairs' in pattern:
            lines.append(f"  Pairs: {len(pattern['pairs'])}")
    lines.append("")

    # Visual representation
    lines.append("Digit Pattern:")
    digits = list(num_str)
    lines.append(" ".join(digits))

    # Mark palindromic positions
    palindrome_marks = []
    for i, digit in enumerate(digits):
        is_palindrome_pos = False
        for radius in range(1, min(i, len(digits) - 1 - i) + 1):
            if digits[i - radius] == digits[i + radius]:
                is_palindrome_pos = True
                break
        palindrome_marks.append("*" if is_palindrome_pos else " ")

    lines.append(" ".join(palindrome_marks))
    lines.append("(* = part of palindromic pattern)")
    lines.append("")

    # Density information
    density = calculate_palindromic_density(number)
    lines.append(f"Palindromic Density: {density:.3%}")
    lines.append("")
    
    # Detailed digit frequency analysis
    digits_int = [int(d) for d in num_str]
    digit_counts = {}
    for d in digits_int:
        digit_counts[d] = digit_counts.get(d, 0) + 1
    
    lines.append("Digit Frequency Analysis:")
    for digit in range(10):
        count = digit_counts.get(digit, 0)
        if count > 0:
            percentage = count / len(digits_int) * 100
            bars = "█" * int(percentage / 2)
            lines.append(f"  {digit}: {count:4d} ({percentage:5.1f}%) {bars}")
        else:
            lines.append(f"  {digit}: {count:4d} ({0:5.1f}%) ")
    
    # Calculate entropy for randomness assessment
    total_digits = len(digits_int)
    entropy = 0.0
    for count in digit_counts.values():
        if count > 0:
            prob = count / total_digits
            entropy -= prob * math.log2(prob)
    
    lines.append("")
    lines.append(f"Statistical Analysis:")
    lines.append(f"  Entropy: {entropy:.3f} bits (max: 3.322 for uniform distribution)")
    if entropy > 3.0:
        lines.append(f"  Assessment: High entropy - relatively uniform digit distribution")
    elif entropy > 2.0:
        lines.append(f"  Assessment: Medium entropy - some pattern structure present")
    else:
        lines.append(f"  Assessment: Low entropy - highly structured digit patterns")
    
    # Most and least common digits (among digits that actually appear)
    appearing_digits = [(k, v) for k, v in digit_counts.items() if v > 0]
    if appearing_digits:
        most_common = max(appearing_digits, key=lambda x: x[1])
        least_common = min(appearing_digits, key=lambda x: x[1])
        lines.append(f"  Most common digit: {most_common[0]} ({most_common[1]} times)")
        lines.append(f"  Least common digit: {least_common[0]} ({least_common[1]} times)")
        
        # Count of unique digits that appear
        unique_digits = len(appearing_digits)
        lines.append(f"  Unique digits used: {unique_digits}/10 possible digits")

    # Save to file using organized structure
    filename = f"palindrome_pattern_{num_str[:20]}_ascii.txt"
    filepath = get_organized_output_path('numbers', 'palindromes', filename)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'palindrome_pattern_ascii',
            'number': num_str,
            'is_palindromic': is_palindromic(number),
            'pattern_count': len(patterns),
            'palindromic_density': density,
            'backend': 'ascii'
        }
    }


def plot_scheherazade_pattern(power: int,
                             backend: Optional[str] = None,
                             **kwargs) -> Dict[str, Any]:
    """
    Visualize patterns in Scheherazade numbers (powers of 1001).

    Args:
        power: Power of 1001 to analyze
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_scheherazade_matplotlib(power, **kwargs)
    elif backend == "ascii":
        return _plot_scheherazade_ascii(power, **kwargs)
    elif backend == "plotly":
        # Fallback to matplotlib for now
        return _plot_scheherazade_matplotlib(power, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_scheherazade_matplotlib(power: int, **kwargs) -> Dict[str, Any]:
    """Plot Scheherazade number patterns using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Get Scheherazade number and analysis
    sche_number = scheherazade_power(power)
    analysis = analyze_scheherazade_ssrcd(power)

    num_str = analysis['number_string']
    digits = [int(d) for d in num_str]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(_config["figure_size"][0] * 1.5, _config["figure_size"][1] * 1.5))

    # Plot 1: Digit distribution
    ax1.hist(digits, bins=range(11), alpha=0.7, color=_config["colors"]["primary"], edgecolor='black')
    ax1.set_title(f'Scheherazade Number 1001^{power}')
    ax1.set_xlabel('Digit Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Digit sequence
    positions = list(range(len(digits)))
    ax2.plot(positions, digits, 'o-', alpha=0.7, color=_config["colors"]["secondary"], markersize=3)
    ax2.set_title('Digit Sequence')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Digit Value')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Running digit sum
    running_sum = np.cumsum(digits)
    ax3.plot(positions, running_sum, color=_config["colors"]["accent"], linewidth=2)
    ax3.set_title('Cumulative Digit Sum')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Cumulative Sum')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Pattern density heatmap
    window_size = min(10, len(digits) // 10)
    if window_size > 0:
        pattern_density = []
        for i in range(0, len(digits) - window_size + 1, window_size):
            window = digits[i:i + window_size]
            # Calculate pattern score (simplified)
            score = len(set(window)) / len(window)  # Diversity measure
            pattern_density.append(score)

        ax4.plot(range(len(pattern_density)), pattern_density, 's-', color=_config["colors"]["primary"])
        ax4.set_title(f'Pattern Diversity (Window Size {window_size})')
        ax4.set_xlabel('Window Position')
        ax4.set_ylabel('Diversity Score')
        ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Scheherazade Number Analysis: 1001^{power}', fontsize=14)
    plt.tight_layout()

    # Save plot using organized structure
    filename = f"scheherazade_pattern_1001_power_{power}.png"
    filepath = get_organized_output_path('numbers', 'scheherazade', filename)
    fig.savefig(filepath, 
                dpi=_config["dpi"], 
                facecolor=_config["png_options"]["facecolor"],
                bbox_inches=_config["png_options"]["bbox_inches"],
                pad_inches=_config["png_options"]["pad_inches"],
                transparent=_config["png_options"]["transparent"])
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'scheherazade_pattern',
            'power': power,
            'number_length': len(num_str),
            'is_palindromic': analysis['is_palindromic'],
            'palindromic_density': analysis['palindromic_density'],
            'pattern_count': len(analysis['palindromic_patterns']),
            'backend': 'matplotlib'
        }
    }


def _plot_scheherazade_ascii(power: int, **kwargs) -> Dict[str, Any]:
    """Create ASCII representation of Scheherazade number patterns."""
    # Get Scheherazade number and analysis
    sche_number = scheherazade_power(power)
    analysis = analyze_scheherazade_ssrcd(power)

    lines = []
    lines.append(f"Scheherazade Number Analysis: 1001^{power}")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Number: {analysis['number_string'][:100]}{'...' if len(analysis['number_string']) > 100 else ''}")
    lines.append(f"Length: {analysis['digit_count']} digits")
    lines.append(f"Is Palindromic: {analysis['is_palindromic']}")
    lines.append(f"Palindromic Density: {analysis['palindromic_density']:.3%}")
    lines.append("")

    # Special insights
    if 'special_insight' in analysis:
        insight = analysis['special_insight']
        lines.append("Special Insight:")
        lines.append(f"  {insight['description']}")
        if 'coefficients' in insight:
            lines.append(f"  Coefficients: {insight['coefficients']}")
        lines.append("")

    # Pattern summary
    lines.append("Pattern Analysis:")
    lines.append(f"  Palindromic Patterns: {len(analysis['palindromic_patterns'])}")
    lines.append(f"  Symmetric Patterns: {len(analysis['symmetric_patterns'])}")
    if analysis['repeated_patterns']:
        lines.append(f"  Repeated Digit Patterns: {len(analysis['repeated_patterns'])}")
    lines.append("")

    # Show first and last digits
    num_str = analysis['number_string']
    if len(num_str) > 20:
        lines.append(f"First 20 digits: {num_str[:20]}")
        lines.append(f"Last 20 digits:  {num_str[-20:]}")
    else:
        lines.append(f"All digits: {num_str}")
    lines.append("")

    # Digit statistics
    digits = [int(d) for d in num_str]
    digit_counts = {}
    for d in digits:
        digit_counts[d] = digit_counts.get(d, 0) + 1

    lines.append("Digit Frequency:")
    for digit in range(10):
        count = digit_counts.get(digit, 0)
        if count > 0:
            percentage = count / len(digits) * 100
            bars = "█" * int(percentage / 2)  # Simple bar chart
            lines.append(f"  {digit}: {count:4d} ({percentage:5.1f}%) {bars}")
        else:
            lines.append(f"  {digit}: {count:4d} ({0:5.1f}%) ")
    lines.append("")
    
    # Add entropy analysis for randomness assessment
    total_digits = len(digits)
    entropy = 0.0
    for count in digit_counts.values():
        if count > 0:
            prob = count / total_digits
            entropy -= prob * math.log2(prob)
    
    lines.append(f"Statistical Analysis:")
    lines.append(f"  Entropy: {entropy:.3f} bits (max: 3.322 for uniform distribution)")
    if entropy > 3.0:
        lines.append(f"  Assessment: High entropy - relatively uniform digit distribution")
    elif entropy > 2.0:
        lines.append(f"  Assessment: Medium entropy - some pattern structure present")
    else:
        lines.append(f"  Assessment: Low entropy - highly structured digit patterns")
    lines.append("")

    # Save to file using organized structure
    filename = f"scheherazade_pattern_1001_power_{power}_ascii.txt"
    filepath = get_organized_output_path('numbers', 'scheherazade', filename)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'scheherazade_pattern_ascii',
            'power': power,
            'number_length': len(analysis['number_string']),
            'is_palindromic': analysis['is_palindromic'],
            'palindromic_density': analysis['palindromic_density'],
            'backend': 'ascii'
        }
    }


def plot_primorial_distribution(max_n: int = 20,
                               backend: Optional[str] = None,
                               **kwargs) -> Dict[str, Any]:
    """
    Visualize the distribution and growth of primorials.

    Args:
        max_n: Maximum n for primorial calculation
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_primorials_matplotlib(max_n, **kwargs)
    elif backend == "ascii":
        return _plot_primorials_ascii(max_n, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_primorials_matplotlib(max_n: int = 20, **kwargs) -> Dict[str, Any]:
    """Plot primorial distribution using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Generate primorial data
    n_values = []
    primorial_values = []
    log_values = []

    for n in range(2, max_n + 1):
        try:
            p = primorial(n)
            n_values.append(n)
            primorial_values.append(float(p.value))
            log_values.append(math.log(float(p.value)))
        except:
            break  # Stop if primorial becomes too large

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(_config["figure_size"][0] * 1.5, _config["figure_size"][1] * 1.5))

    # Plot 1: Linear primorial values
    ax1.plot(n_values, primorial_values, 'o-', color=_config["colors"]["primary"], linewidth=2, markersize=6)
    ax1.set_title('Primorial Values')
    ax1.set_xlabel('n')
    ax1.set_ylabel('n#')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Logarithmic primorial values
    ax2.plot(n_values, log_values, 's-', color=_config["colors"]["secondary"], linewidth=2, markersize=6)
    ax2.set_title('Primorial Values (Log Scale)')
    ax2.set_xlabel('n')
    ax2.set_ylabel('log(n#)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Growth rate
    if len(log_values) > 1:
        growth_rates = []
        for i in range(1, len(log_values)):
            rate = log_values[i] - log_values[i-1]
            growth_rates.append(rate)

        ax3.plot(n_values[1:], growth_rates, '^-', color=_config["colors"]["accent"], linewidth=2, markersize=6)
        ax3.set_title('Growth Rate (log difference)')
        ax3.set_xlabel('n')
        ax3.set_ylabel('Δlog(n#)')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Ratio to factorial
    if len(n_values) > 0:
        factorial_ratios = []
        for n in n_values:
            if n <= 20:  # Factorial becomes very large
                fact = math.factorial(n)
                prime_fact = float(primorial(n).value)
                ratio = prime_fact / fact if fact > 0 else 0
                factorial_ratios.append(ratio)

        if factorial_ratios:
            ax4.plot(n_values[:len(factorial_ratios)], factorial_ratios, 'd-', color=_config["colors"]["primary"], linewidth=2, markersize=6)
            ax4.set_title('Primorial / Factorial Ratio')
            ax4.set_xlabel('n')
            ax4.set_ylabel('n# / n!')
            ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Primorial Distribution (n ≤ {max_n})', fontsize=14)
    plt.tight_layout()

    # Save plot using organized structure
    filename = f"primorial_distribution_max_n_{max_n}.png"
    filepath = get_organized_output_path('numbers', 'primorials', filename)
    fig.savefig(filepath, 
                dpi=_config["dpi"], 
                facecolor=_config["png_options"]["facecolor"],
                bbox_inches=_config["png_options"]["bbox_inches"],
                pad_inches=_config["png_options"]["pad_inches"],
                transparent=_config["png_options"]["transparent"])
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'primorial_distribution',
            'max_n': max_n,
            'primorials_calculated': len(n_values),
            'largest_primorial': str(primorial_values[-1]) if primorial_values else 'N/A',
            'backend': 'matplotlib'
        }
    }


def _plot_primorials_ascii(max_n: int = 20, **kwargs) -> Dict[str, Any]:
    """Create ASCII representation of primorial distribution."""
    lines = []
    lines.append(f"Primorial Distribution (n ≤ {max_n})")
    lines.append("=" * 50)
    lines.append("")

    lines.append("Primorial Values:")
    lines.append("n    n#")
    lines.append("-    --")

    # Calculate primorials and analyze digit patterns
    all_digits = []
    primorial_data = []
    
    for n in range(2, min(max_n + 1, 15)):  # Limit to avoid extremely large numbers
        try:
            p = primorial(n)
            p_str = str(p.value)
            primorial_data.append((n, p_str, len(p_str)))
            
            # Collect all digits for frequency analysis
            all_digits.extend([int(d) for d in p_str])
            
            # Show primorial values (truncated if very long)
            if len(p_str) <= 50:
                lines.append(f"{n:2d}   {p_str}")
            else:
                lines.append(f"{n:2d}   {p_str[:20]}...{p_str[-20:]} ({len(p_str)} digits)")
        except Exception as e:
            lines.append(f"{n:2d}   [Error calculating primorial: {str(e)[:30]}]")
            break
    
    lines.append("")
    
    # Digit frequency analysis across all primorials
    if all_digits:
        digit_counts = {}
        for d in all_digits:
            digit_counts[d] = digit_counts.get(d, 0) + 1
        
        total_digits = len(all_digits)
        lines.append(f"Collective Digit Frequency Analysis ({total_digits} total digits):")
        lines.append("=" * 50)
        
        for digit in range(10):
            count = digit_counts.get(digit, 0)
            if count > 0:
                percentage = count / total_digits * 100
                bars = "█" * int(percentage / 2)
                lines.append(f"  {digit}: {count:4d} ({percentage:5.1f}%) {bars}")
            else:
                lines.append(f"  {digit}: {count:4d} ({0:5.1f}%) ")
        
        # Calculate entropy for digit distribution
        entropy = 0.0
        for count in digit_counts.values():
            if count > 0:
                prob = count / total_digits
                entropy -= prob * math.log2(prob)
        
        lines.append("")
        lines.append(f"Statistical Properties:")
        lines.append(f"  Digit Entropy: {entropy:.3f} bits (max: 3.322 for uniform)")
        if entropy > 3.0:
            lines.append(f"  Assessment: High entropy - digits appear relatively random")
        elif entropy > 2.5:
            lines.append(f"  Assessment: Moderate entropy - some digit preferences")
        else:
            lines.append(f"  Assessment: Low entropy - strong digit patterns present")
        
        # Most and least common digits (among digits that actually appear)
        appearing_digits = [(k, v) for k, v in digit_counts.items() if v > 0]
        if appearing_digits:
            most_common = max(appearing_digits, key=lambda x: x[1])
            least_common = min(appearing_digits, key=lambda x: x[1])
            lines.append(f"  Most common digit: {most_common[0]} ({most_common[1]} times)")
            lines.append(f"  Least common digit: {least_common[0]} ({least_common[1]} times)")
        
        # Count of unique digits that appear
        unique_digits = len(appearing_digits)
        lines.append(f"  Unique digits used: {unique_digits}/10 possible digits")
    
    lines.append("")
    lines.append("Mathematical Properties:")
    lines.append("- Primorials grow faster than exponentials")
    lines.append("- Related to prime number distribution")
    lines.append("- Used in number theory and cryptography")
    lines.append("- Fuller's 14-illion number uses primorial factors")

    # Save to file using organized structure
    filename = f"primorial_distribution_max_n_{max_n}_ascii.txt"
    filepath = get_organized_output_path('numbers', 'primorials', filename)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'primorial_distribution_ascii',
            'max_n': max_n,
            'backend': 'ascii'
        }
    }


def plot_mnemonic_visualization(number: Union[int, SymergeticsNumber],
                               backend: Optional[str] = None,
                               **kwargs) -> Dict[str, Any]:
    """
    Visualize mnemonic encoding strategies for a number.

    Args:
        number: Number to create mnemonic visualizations for
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_mnemonic_matplotlib(number, **kwargs)
    elif backend == "ascii":
        return _plot_mnemonic_ascii(number, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_mnemonic_matplotlib(number: Union[int, SymergeticsNumber],
                             **kwargs) -> Dict[str, Any]:
    """Plot mnemonic strategies using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Get mnemonic aids
    aids = create_memory_aid(number)

    # Create visualization of different mnemonic strategies
    fig, axes = plt.subplots(2, 2, figsize=(_config["figure_size"][0] * 1.5, _config["figure_size"][1] * 1.5))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    strategies = ['grouped', 'scientific', 'words', 'patterns']
    titles = ['Grouped Digits', 'Scientific Notation', 'Word Representation', 'Pattern Analysis']

    for i, (strategy, title) in enumerate(zip(strategies, titles)):
        ax = axes[i]

        # Create a simple visualization for each strategy
        mnemonic = aids.get(strategy, 'N/A')

        # Word cloud-like visualization (simplified)
        ax.text(0.5, 0.5, mnemonic, ha='center', va='center',
               fontsize=12, wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor=_config["colors"]["background"]))

        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle(f'Mnemonic Encoding Strategies: {format_large_number(number)}', fontsize=14)
    plt.tight_layout()

    # Save plot using organized structure
    if isinstance(number, SymergeticsNumber):
        num_str = str(number.value.numerator)[:20]
    else:
        try:
            num_str = str(abs(int(number)))[:20]
        except (ValueError, TypeError):
            # Handle invalid input gracefully
            return {
                'files': [],
                'metadata': {
                    'error': f'Invalid input: {number}',
                    'backend': 'matplotlib',
                    'function': 'plot_mnemonic_visualization'
                }
            }

    filename = f"mnemonic_visualization_{num_str}.png"
    filepath = get_organized_output_path('numbers', 'mnemonics', filename)
    fig.savefig(filepath, 
                dpi=_config["dpi"], 
                facecolor=_config["png_options"]["facecolor"],
                bbox_inches=_config["png_options"]["bbox_inches"],
                pad_inches=_config["png_options"]["pad_inches"],
                transparent=_config["png_options"]["transparent"])
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'mnemonic_visualization',
            'number': str(number),
            'strategies': len(aids),
            'backend': 'matplotlib'
        }
    }


def plot_palindromic_heatmap(sequence_start: int, sequence_end: int,
                           backend: Optional[str] = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Create a heatmap visualization of palindromic patterns across a number sequence.

    Args:
        sequence_start: Starting number of the sequence
        sequence_end: Ending number of the sequence
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths

    Examples:
        >>> plot_palindromic_heatmap(100, 200)
        {'files': ['output/numbers/palindromes/palindromic_heatmap_100_200.png'], 'metadata': {...}}
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_palindromic_heatmap_matplotlib(sequence_start, sequence_end, **kwargs)
    else:
        raise ValueError(f"Heatmap visualization requires matplotlib backend, got: {backend}")


def _plot_palindromic_heatmap_matplotlib(sequence_start: int, sequence_end: int,
                                       **kwargs) -> Dict[str, Any]:
    """Create palindromic heatmap using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("matplotlib and seaborn are required for heatmap visualization")

    # Generate sequence data
    numbers = list(range(sequence_start, sequence_end + 1))

    # Calculate palindromic properties for each number
    palindromic_data = []
    digit_density = []
    pattern_complexity = []

    for num in numbers:
        # Check if number is palindromic
        is_pal = is_palindromic(num)
        palindromic_data.append(1 if is_pal else 0)

        # Calculate digit density (distribution of digits)
        num_str = str(abs(num))
        digit_counts = [num_str.count(str(d)) for d in range(10)]
        density = sum(digit_counts) / len(digit_counts) if digit_counts else 0
        digit_density.append(density)

        # Calculate pattern complexity (number of unique digit patterns)
        patterns = extract_palindromic_patterns(num, min_length=2)
        complexity = len(patterns)
        pattern_complexity.append(complexity)

    # Create heatmap data matrix
    data_matrix = np.array([palindromic_data, digit_density, pattern_complexity])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use seaborn for better heatmap
    sns.heatmap(data_matrix,
                cmap='viridis',
                cbar_kws={'label': 'Value'},
                ax=ax)

    # Customize labels
    ax.set_title(f'Palindromic Pattern Analysis\nNumbers {sequence_start} to {sequence_end}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Number Index')
    ax.set_ylabel('Property Type')

    # Set custom y-tick labels
    ax.set_yticklabels(['Palindromic', 'Digit Density', 'Pattern Complexity'])
    ax.set_xticks(np.arange(0, len(numbers), 50))  # Show every 50th number
    ax.set_xticklabels([numbers[i] for i in range(0, len(numbers), 50)], rotation=45)

    plt.tight_layout()

    # Save heatmap
    filename = f"palindromic_heatmap_{sequence_start}_{sequence_end}.png"
    filepath = get_organized_output_path('numbers', 'palindromes', filename)
    fig.savefig(filepath, dpi=_config["dpi"], bbox_inches='tight')
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'palindromic_heatmap',
            'sequence_start': sequence_start,
            'sequence_end': sequence_end,
            'total_numbers': len(numbers),
            'palindromic_count': sum(palindromic_data),
            'backend': 'matplotlib'
        }
    }


def plot_scheherazade_network(power: int,
                             backend: Optional[str] = None,
                             **kwargs) -> Dict[str, Any]:
    """
    Create a network visualization of relationships within Scheherazade numbers.

    Args:
        power: Power of 1001 to analyze
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths

    Examples:
        >>> plot_scheherazade_network(6)
        {'files': ['output/numbers/scheherazade/scheherazade_network_6.png'], 'metadata': {...}}
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_scheherazade_network_matplotlib(power, **kwargs)
    else:
        raise ValueError(f"Network visualization requires matplotlib backend, got: {backend}")


def _plot_scheherazade_network_matplotlib(power: int, **kwargs) -> Dict[str, Any]:
    """Create Scheherazade network visualization using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        raise ImportError("matplotlib is required for network visualization")

    # Get Scheherazade number
    scheherazade_num = scheherazade_power(power)
    num_str = str(scheherazade_num.value.numerator)

    # Analyze digit relationships
    digit_positions = {}
    for i, digit in enumerate(num_str):
        if digit not in digit_positions:
            digit_positions[digit] = []
        digit_positions[digit].append(i)

    # Create network visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create nodes for each digit
    digit_nodes = {}
    for i, digit in enumerate('0123456789'):
        x = (i % 3) * 4
        y = (i // 3) * 4
        digit_nodes[digit] = (x, y)

        # Draw node
        circle = Circle((x, y), 1.5, facecolor=_config["colors"]["primary"],
                       edgecolor=_config["colors"]["secondary"], alpha=0.7)
        ax.add_patch(circle)

        # Add digit label
        ax.text(x, y, digit, ha='center', va='center',
               fontsize=16, fontweight='bold', color='white')

    # Draw connections based on digit patterns
    for digit, positions in digit_positions.items():
        if len(positions) > 1:
            x1, y1 = digit_nodes[digit]
            for pos in positions[1:]:
                # Create connection lines (simplified)
                ax.plot([x1, x1 + np.random.uniform(-2, 2)],
                       [y1, y1 + np.random.uniform(-2, 2)],
                       color=_config["colors"]["accent"], alpha=0.3, linewidth=1)

    # Add pattern information
    info_text = f"Scheherazade Number: 1001^{power}\n"
    info_text += f"Total digits: {len(num_str)}\n"
    info_text += f"Unique digits: {len(digit_positions)}\n"
    info_text += f"Is palindromic: {is_palindromic(scheherazade_num)}"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))

    ax.set_title(f'Scheherazade Number Network: 1001^{power}',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 14)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    # Save network visualization
    filename = f"scheherazade_network_{power}.png"
    filepath = get_organized_output_path('numbers', 'scheherazade', filename)
    fig.savefig(filepath, dpi=_config["dpi"], bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'scheherazade_network',
            'power': power,
            'total_digits': len(num_str),
            'unique_digits': len(digit_positions),
            'is_palindromic': is_palindromic(scheherazade_num),
            'backend': 'matplotlib'
        }
    }


def plot_primorial_spectrum(max_n: int = 15,
                          backend: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
    """
    Create a spectrum visualization of primorial growth patterns.

    Args:
        max_n: Maximum n for primorial calculation
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths

    Examples:
        >>> plot_primorial_spectrum(20)
        {'files': ['output/numbers/primorials/primorial_spectrum_20.png'], 'metadata': {...}}
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_primorial_spectrum_matplotlib(max_n, **kwargs)
    else:
        raise ValueError(f"Spectrum visualization requires matplotlib backend, got: {backend}")


def _plot_primorial_spectrum_matplotlib(max_n: int = 15, **kwargs) -> Dict[str, Any]:
    """Create primorial spectrum visualization using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for spectrum visualization")

    # Validate input
    if max_n < 1:
        raise ValueError(f"max_n must be >= 1, got {max_n}")

    # Calculate primorials
    primorials = []
    for n in range(1, max_n + 1):
        p_n = primorial(n)
        primorials.append(float(p_n.value))

    # Create spectrum visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Spectrum 1: Raw primorial values
    x_vals = list(range(1, max_n + 1))
    colors = plt.cm.viridis(np.linspace(0, 1, len(primorials)))

    ax1.bar(x_vals, primorials, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_yscale('log')
    ax1.set_title('Primorial Spectrum - Raw Values', fontsize=12, fontweight='bold')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Primorial Value (log scale)')
    ax1.grid(True, alpha=0.3)

    # Add value labels for smaller primorials
    for i, (x, val) in enumerate(zip(x_vals, primorials)):
        if val < 1e10:  # Only label smaller values
            ax1.text(x, val * 1.1, f'{int(val)}', ha='center', va='bottom',
                    fontsize=8, rotation=45)

    # Spectrum 2: Growth rates
    growth_rates = []
    for i in range(1, len(primorials)):
        rate = primorials[i] / primorials[i-1] if primorials[i-1] > 0 else 0
        growth_rates.append(rate)

    ax2.plot(range(2, max_n + 1), growth_rates, 'ro-', linewidth=2, markersize=6)
    ax2.set_title('Primorial Growth Rates', fontsize=12, fontweight='bold')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Growth Rate (P(n)/P(n-1))')
    ax2.grid(True, alpha=0.3)

    # Add growth rate values
    for i, rate in enumerate(growth_rates):
        if rate < 100:  # Only label reasonable values
            ax2.text(i + 2, rate * 1.05, f'{rate:.1f}', ha='center', va='bottom', fontsize=8)

    # Spectrum 3: Digit length progression
    digit_lengths = [len(str(int(p))) for p in primorials]

    ax3.plot(x_vals, digit_lengths, 'go-', linewidth=2, markersize=8)
    ax3.fill_between(x_vals, digit_lengths, alpha=0.3, color='green')
    ax3.set_title('Primorial Digit Length Spectrum', fontsize=12, fontweight='bold')
    ax3.set_xlabel('n')
    ax3.set_ylabel('Number of Digits')
    ax3.grid(True, alpha=0.3)

    # Add digit count labels
    for i, (x, digits) in enumerate(zip(x_vals, digit_lengths)):
        ax3.text(x, digits + 0.1, str(digits), ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save spectrum visualization
    filename = f"primorial_spectrum_{max_n}.png"
    filepath = get_organized_output_path('numbers', 'primorials', filename)
    fig.savefig(filepath, dpi=_config["dpi"], bbox_inches='tight')
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'primorial_spectrum',
            'max_n': max_n,
            'total_primorials': len(primorials),
            'largest_digits': max(digit_lengths),
            'backend': 'matplotlib'
        }
    }


def _plot_mnemonic_ascii(number: Union[int, SymergeticsNumber],
                        **kwargs) -> Dict[str, Any]:
    """Create ASCII representation of mnemonic strategies."""
    # Get mnemonic aids
    aids = create_memory_aid(number)

    lines = []
    lines.append(f"Mnemonic Encoding Strategies: {format_large_number(number)}")
    lines.append("=" * 70)
    lines.append("")

    for strategy, mnemonic in aids.items():
        lines.append(f"{strategy.upper()}:")
        lines.append(f"  {mnemonic}")
        lines.append("")

    lines.append("Mnemonic Encoding Benefits:")
    lines.append("- Makes large numbers more memorable")
    lines.append("- Reveals mathematical patterns")
    lines.append("- Follows Fuller's synergetic principles")
    lines.append("- Aids in number theory research")

    # Save to file using organized structure
    if isinstance(number, SymergeticsNumber):
        num_str = str(number.value.numerator)[:20]
    else:
        try:
            num_str = str(abs(int(number)))[:20]
        except (ValueError, TypeError):
            # Handle invalid input gracefully
            return {
                'files': [],
                'metadata': {
                    'error': f'Invalid input: {number}',
                    'backend': 'ascii',
                    'function': 'plot_mnemonic_visualization'
                }
            }

    filename = f"mnemonic_visualization_{num_str}_ascii.txt"
    filepath = get_organized_output_path('numbers', 'mnemonics', filename)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'mnemonic_visualization_ascii',
            'number': str(number),
            'strategies': len(aids),
            'backend': 'ascii'
        }
    }
