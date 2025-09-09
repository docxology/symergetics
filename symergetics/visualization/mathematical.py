"""
Mathematical Visualizations for Synergetics

This module provides visualization capabilities for mathematical concepts and
algorithms in the Synergetics framework, including continued fractions,
base conversions, pattern analysis, and SSRCD visualizations.

Features:
- Continued fraction visualizations
- Base conversion diagrams
- Pattern analysis heatmaps
- SSRCD (Sublimely Rememberable Comprehensive Dividends) analysis
- Mathematical convergence plots
- Algorithm visualization

Author: Symergetics Team
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from pathlib import Path
import math

from ..utils.conversion import (
    continued_fraction_approximation, convergents_from_continued_fraction,
    convert_between_bases, best_rational_approximation
)
from ..computation.palindromes import analyze_scheherazade_ssrcd, find_repeated_patterns
from . import _config, ensure_output_dir, get_organized_output_path


def plot_continued_fraction(value: float,
                          max_terms: int = 10,
                          backend: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
    """
    Visualize the continued fraction expansion of a real number.

    Args:
        value: Real number to analyze
        max_terms: Maximum number of continued fraction terms to show
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_continued_fraction_matplotlib(value, max_terms, **kwargs)
    elif backend == "ascii":
        return _plot_continued_fraction_ascii(value, max_terms, **kwargs)
    elif backend == "plotly":
        # Fallback to matplotlib for compatibility
        return _plot_continued_fraction_matplotlib(value, max_terms, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_continued_fraction_matplotlib(value: float,
                                       max_terms: int = 10,
                                       **kwargs) -> Dict[str, Any]:
    """Plot continued fraction using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Get continued fraction terms and convergents
    terms = continued_fraction_approximation(value, max_terms)
    convergents = convergents_from_continued_fraction(terms)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(_config["figure_size"][0] * 1.5, _config["figure_size"][1] * 1.5))

    # Plot 1: Continued fraction terms
    term_positions = list(range(len(terms)))
    ax1.bar(term_positions, terms, color=_config["colors"]["primary"], alpha=0.7)
    ax1.set_title('Continued Fraction Terms')
    ax1.set_xlabel('Term Index')
    ax1.set_ylabel('Term Value')
    ax1.set_xticks(term_positions)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Convergents approximation error
    if convergents:
        errors = []
        for num, den in convergents:
            approx = num / den
            error = abs(approx - value)
            errors.append(error)

        ax2.semilogy(range(len(errors)), errors, 'o-', color=_config["colors"]["secondary"], linewidth=2)
        ax2.set_title('Convergence to True Value (Log Scale)')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Absolute Error')
        ax2.grid(True, alpha=0.3)

    # Plot 3: Fraction values over iterations
    if convergents:
        fraction_values = [num / den for num, den in convergents]
        ax3.plot(range(len(fraction_values)), fraction_values, 's-', color=_config["colors"]["accent"], linewidth=2)
        ax3.axhline(y=value, color='red', linestyle='--', alpha=0.7, label=f'True value: {value:.6f}')
        ax3.set_title('Fractional Approximations')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Approximated Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Term distribution
    if len(terms) > 1:
        ax4.hist(terms[1:], bins=max(5, len(set(terms[1:]))), alpha=0.7, color=_config["colors"]["primary"])
        ax4.set_title('Term Value Distribution')
        ax4.set_xlabel('Term Value')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Continued Fraction Analysis: {value:.6f}')
    plt.tight_layout()

    # Save plot using organized structure
    filename = f"continued_fraction_{value:.6f}_terms_{max_terms}.png"
    filepath = get_organized_output_path('mathematical', 'continued_fractions', filename)
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
            'type': 'continued_fraction',
            'value': value,
            'terms_calculated': len(terms),
            'convergents_found': len(convergents),
            'final_error': abs(convergents[-1][0] / convergents[-1][1] - value) if convergents else None,
            'backend': 'matplotlib'
        }
    }


def _plot_continued_fraction_ascii(value: float,
                                  max_terms: int = 10,
                                  **kwargs) -> Dict[str, Any]:
    """Create ASCII representation of continued fraction."""
    # Get continued fraction terms and convergents
    terms = continued_fraction_approximation(value, max_terms)
    convergents = convergents_from_continued_fraction(terms)

    lines = []
    lines.append(f"Continued Fraction Analysis: {value:.10f}")
    lines.append("=" * 60)
    lines.append("")

    # Show continued fraction representation
    lines.append("Continued Fraction:")
    cf_repr = f"{terms[0]}"
    for term in terms[1:]:
        cf_repr += f" + 1/({term}"
    cf_repr += ")" * (len(terms) - 1)
    lines.append(f"  {cf_repr}")
    lines.append("")

    # Show terms
    lines.append("Terms:")
    for i, term in enumerate(terms):
        lines.append(f"  a{i} = {term}")
    lines.append("")

    # Show convergents
    if convergents:
        lines.append("Convergents:")
        for i, (num, den) in enumerate(convergents):
            approx = num / den
            error = abs(approx - value)
            lines.append(f"  {i}: {num}/{den} = {approx:.10f} (error: {error:.2e})")
        lines.append("")

    # Statistics
    lines.append("Statistics:")
    lines.append(f"  Terms calculated: {len(terms)}")
    lines.append(f"  Final convergent: {convergents[-1][0]}/{convergents[-1][1]}" if convergents else "  No convergents")
    if convergents:
        final_error = abs(convergents[-1][0] / convergents[-1][1] - value)
        lines.append(f"  Final error: {final_error:.2e}")

    # Save to file using organized structure
    filename = f"continued_fraction_{value:.6f}_terms_{max_terms}_ascii.txt"
    filepath = get_organized_output_path('mathematical', 'continued_fractions', filename)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'continued_fraction_ascii',
            'value': value,
            'terms_calculated': len(terms),
            'convergents_found': len(convergents),
            'backend': 'ascii'
        }
    }


def plot_base_conversion(number: int,
                        from_base: int = 10,
                        to_base: int = 2,
                        backend: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
    """
    Visualize base conversion process.

    Args:
        number: Number to convert
        from_base: Source base
        to_base: Target base
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_base_conversion_matplotlib(number, from_base, to_base, **kwargs)
    elif backend == "ascii":
        return _plot_base_conversion_ascii(number, from_base, to_base, **kwargs)
    elif backend == "plotly":
        # Fallback to matplotlib for compatibility
        return _plot_base_conversion_matplotlib(number, from_base, to_base, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_base_conversion_matplotlib(number: int,
                                   from_base: int = 10,
                                   to_base: int = 2,
                                   **kwargs) -> Dict[str, Any]:
    """Plot base conversion process using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Perform conversion and collect steps
    result = convert_between_bases(number, from_base, to_base)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(_config["figure_size"][0] * 1.5, _config["figure_size"][1] * 1.5))

    # Plot 1: Number digit distribution
    num_str = str(number)
    digits = [int(d) for d in num_str]
    digit_counts = {}

    for d in digits:
        digit_counts[d] = digit_counts.get(d, 0) + 1

    digit_values = list(digit_counts.keys())
    digit_freqs = list(digit_counts.values())

    ax1.bar(digit_values, digit_freqs, color=_config["colors"]["primary"], alpha=0.7)
    ax1.set_title(f'Digit Distribution in Base {from_base}')
    ax1.set_xlabel('Digit Value')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks(range(max(digit_values) + 1))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Conversion steps (simplified visualization)
    steps = []
    temp = number
    while temp > 0:
        remainder = temp % to_base
        steps.append(remainder)
        temp //= to_base

    ax2.plot(range(len(steps)), steps[::-1], 'o-', color=_config["colors"]["secondary"], linewidth=2)
    ax2.set_title(f'Conversion Steps: Base {from_base} → Base {to_base}')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Remainder')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Bit/coefficient pattern (for binary)
    if to_base == 2:
        ax3.bar(range(len(steps)), steps[::-1], color=_config["colors"]["accent"], alpha=0.7)
        ax3.set_title('Binary Representation')
        ax3.set_xlabel('Bit Position')
        ax3.set_ylabel('Bit Value (0 or 1)')
        ax3.set_xticks(range(len(steps)))
        ax3.grid(True, alpha=0.3)
    else:
        # General coefficient visualization
        coeffs = steps[::-1]
        ax3.bar(range(len(coeffs)), coeffs, color=_config["colors"]["accent"], alpha=0.7)
        ax3.set_title(f'Coefficients in Base {to_base}')
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Coefficient Value')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Numerical relationships
    if to_base == 2:
        # Show powers of 2
        powers = [2**i for i in range(len(steps))]
        ax4.bar(range(len(powers)), powers, color=_config["colors"]["primary"], alpha=0.7)
        ax4.set_title('Powers of 2 (Binary Weights)')
        ax4.set_xlabel('Bit Position')
        ax4.set_ylabel('Weight Value')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Base Conversion: {number} ({from_base} → {to_base})')
    plt.tight_layout()

    # Save plot using organized structure
    filename = f"base_conversion_{number}_base_{from_base}_to_{to_base}.png"
    filepath = get_organized_output_path('mathematical', 'base_conversions', filename)
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
            'type': 'base_conversion',
            'number': number,
            'from_base': from_base,
            'to_base': to_base,
            'result': result,
            'steps': len(steps),
            'backend': 'matplotlib'
        }
    }


def _plot_base_conversion_ascii(number: int,
                               from_base: int = 10,
                               to_base: int = 2,
                               **kwargs) -> Dict[str, Any]:
    """Create ASCII representation of base conversion."""
    result = convert_between_bases(number, from_base, to_base)

    lines = []
    lines.append(f"Base Conversion: {number} (base {from_base}) → {result} (base {to_base})")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Original number: {number}")
    lines.append(f"From base: {from_base}")
    lines.append(f"To base: {to_base}")
    lines.append(f"Result: {result}")
    lines.append("")

    # Show conversion steps
    lines.append("Conversion Steps:")
    if from_base == 10 and to_base != 10:
        lines.append("(Dividing by target base and collecting remainders)")
        temp = number
        step = 0
        while temp > 0:
            remainder = temp % to_base
            quotient = temp // to_base
            lines.append(f"  Step {step}: {temp} ÷ {to_base} = {quotient} remainder {remainder}")
            temp = quotient
            step += 1
    elif to_base == 10 and from_base != 10:
        lines.append("(Multiplying digits by powers of source base)")
        # Simplified explanation
        lines.append(f"  Interpreting '{number}' as base {from_base} number")

    lines.append("")
    lines.append("Properties:")
    lines.append(f"- Number of digits in base {to_base}: {len(str(result))}")
    lines.append(f"- Maximum digit value: {to_base - 1}")

    # Save to file using organized structure
    filename = f"base_conversion_{number}_base_{from_base}_to_{to_base}_ascii.txt"
    filepath = get_organized_output_path('mathematical', 'base_conversions', filename)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'base_conversion_ascii',
            'number': number,
            'from_base': from_base,
            'to_base': to_base,
            'result': result,
            'backend': 'ascii'
        }
    }


def plot_pattern_analysis(number: Union[int, str],
                         pattern_type: str = 'palindrome',
                         backend: Optional[str] = None,
                         **kwargs) -> Dict[str, Any]:
    """
    Visualize pattern analysis for numbers.

    Args:
        number: Number or string to analyze
        pattern_type: Type of pattern to analyze ('palindrome', 'repeated', 'symmetric')
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_pattern_analysis_matplotlib(number, pattern_type, **kwargs)
    elif backend == "ascii":
        return _plot_pattern_analysis_ascii(number, pattern_type, **kwargs)
    elif backend == "plotly":
        # Fallback to matplotlib for compatibility
        return _plot_pattern_analysis_matplotlib(number, pattern_type, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_pattern_analysis_matplotlib(number: Union[int, str],
                                     pattern_type: str = 'palindrome',
                                     **kwargs) -> Dict[str, Any]:
    """Plot pattern analysis using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Convert to string for analysis
    if isinstance(number, int):
        num_str = str(number)
    else:
        num_str = str(number)

    digits = [int(d) for d in num_str]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(_config["figure_size"][0] * 1.5, _config["figure_size"][1] * 1.5))

    # Plot 1: Digit sequence
    positions = list(range(len(digits)))
    ax1.plot(positions, digits, 'o-', color=_config["colors"]["primary"], linewidth=2, markersize=6)
    ax1.set_title(f'Number: {num_str[:20]}{"..." if len(num_str) > 20 else ""}')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Digit Value')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Pattern highlighting based on type
    colors = [_config["colors"]["primary"]] * len(digits)

    if pattern_type == 'palindrome':
        # Highlight palindromic positions
        for i in range(len(digits)):
            for radius in range(1, min(i, len(digits) - 1 - i) + 1):
                if digits[i - radius] == digits[i + radius]:
                    colors[i] = _config["colors"]["accent"]
                    break
    elif pattern_type == 'repeated':
        # Highlight repeated digits
        for i in range(len(digits)):
            if digits.count(digits[i]) > 1:
                colors[i] = _config["colors"]["accent"]

    ax2.bar(positions, digits, color=colors, alpha=0.7)
    ax2.set_title(f'{pattern_type.title()} Pattern Analysis')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Digit Value')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Pattern density heatmap
    pattern_matrix = np.zeros((len(digits), len(digits)))

    if pattern_type == 'palindrome':
        # Palindrome similarity matrix
        for i in range(len(digits)):
            for j in range(len(digits)):
                if digits[i] == digits[j]:
                    pattern_matrix[i, j] = 1
    elif pattern_type == 'repeated':
        # Repetition matrix
        for i in range(len(digits)):
            for j in range(len(digits)):
                if digits[i] == digits[j]:
                    pattern_matrix[i, j] = 1

    im = ax3.imshow(pattern_matrix, cmap='Blues', aspect='equal')
    ax3.set_title('Pattern Similarity Matrix')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Position')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Plot 4: Statistical analysis
    digit_counts = {}
    for d in digits:
        digit_counts[d] = digit_counts.get(d, 0) + 1

    digit_values = list(digit_counts.keys())
    digit_freqs = list(digit_counts.values())

    ax4.bar(digit_values, digit_freqs, color=_config["colors"]["secondary"], alpha=0.7)
    ax4.set_title('Digit Frequency Distribution')
    ax4.set_xlabel('Digit')
    ax4.set_ylabel('Frequency')
    ax4.set_xticks(range(10))
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Pattern Analysis: {pattern_type.title()}', fontsize=14)
    plt.tight_layout()

    # Save plot using organized structure
    filename = f"pattern_analysis_{pattern_type}_{num_str[:20]}.png"
    filepath = get_organized_output_path('mathematical', 'pattern_analysis', filename)
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
            'type': 'pattern_analysis',
            'number': num_str,
            'pattern_type': pattern_type,
            'length': len(digits),
            'unique_digits': len(set(digits)),
            'backend': 'matplotlib'
        }
    }


def _plot_pattern_analysis_ascii(number: Union[int, str],
                                pattern_type: str = 'palindrome',
                                **kwargs) -> Dict[str, Any]:
    """Create ASCII representation of pattern analysis."""
    # Convert to string for analysis
    if isinstance(number, int):
        num_str = str(number)
    else:
        num_str = str(number)

    digits = [int(d) for d in num_str]

    lines = []
    lines.append(f"Pattern Analysis: {pattern_type.title()}")
    lines.append("=" * 50)
    lines.append("")

    lines.append(f"Number: {num_str}")
    lines.append(f"Length: {len(digits)} digits")
    lines.append("")

    # Show digit sequence with pattern highlighting
    lines.append("Digit Sequence:")
    pattern_marks = []

    if pattern_type == 'palindrome':
        for i in range(len(digits)):
            is_pattern = False
            for radius in range(1, min(i, len(digits) - 1 - i) + 1):
                if digits[i - radius] == digits[i + radius]:
                    is_pattern = True
                    break
            pattern_marks.append('*' if is_pattern else ' ')
    elif pattern_type == 'repeated':
        for i in range(len(digits)):
            is_pattern = digits.count(digits[i]) > 1
            pattern_marks.append('*' if is_pattern else ' ')

    lines.append(" ".join(str(d) for d in digits))
    lines.append(" ".join(pattern_marks))
    lines.append("(* indicates pattern position)")
    lines.append("")

    # Statistics
    digit_counts = {}
    for d in digits:
        digit_counts[d] = digit_counts.get(d, 0) + 1

    lines.append("Digit Statistics:")
    for digit in range(10):
        count = digit_counts.get(digit, 0)
        if count > 0:
            percentage = count / len(digits) * 100
            lines.append(f"  Digit {digit}: {count} times ({percentage:.1f}%)")
    lines.append("")

    # Save to file using organized structure
    filename = f"pattern_analysis_{pattern_type}_{num_str[:20]}_ascii.txt"
    filepath = get_organized_output_path('mathematical', 'pattern_analysis', filename)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'pattern_analysis_ascii',
            'number': num_str,
            'pattern_type': pattern_type,
            'length': len(digits),
            'backend': 'ascii'
        }
    }


def plot_ssrcd_analysis(power: int,
                       backend: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
    """
    Visualize SSRCD (Sublimely Rememberable Comprehensive Dividends) analysis.

    Args:
        power: Power of 1001 to analyze
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths
    """
    # Get SSRCD analysis
    analysis = analyze_scheherazade_ssrcd(power)

    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_ssrcd_matplotlib(analysis, **kwargs)
    elif backend == "ascii":
        return _plot_ssrcd_ascii(analysis, **kwargs)
    elif backend == "plotly":
        # Fallback to matplotlib for compatibility
        return _plot_ssrcd_matplotlib(analysis, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _plot_ssrcd_matplotlib(analysis: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Plot SSRCD analysis using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(_config["figure_size"][0] * 1.5, _config["figure_size"][1] * 1.5))

    num_str = analysis['number_string']
    digits = [int(d) for d in num_str]

    # Plot 1: Number structure
    positions = list(range(len(digits)))
    ax1.plot(positions, digits, 'o-', alpha=0.7, color=_config["colors"]["primary"], markersize=3)
    ax1.set_title(f'Scheherazade {analysis["scheherazade_power"]}^th Power')
    ax1.set_xlabel('Digit Position')
    ax1.set_ylabel('Digit Value')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Palindromic density visualization
    palindromic_positions = []
    for i in range(len(digits)):
        is_palindrome = False
        for radius in range(1, min(i, len(digits) - 1 - i) + 1):
            if digits[i - radius] == digits[i + radius]:
                is_palindrome = True
                break
        palindromic_positions.append(1 if is_palindrome else 0)

    ax2.bar(positions, palindromic_positions, color=_config["colors"]["accent"], alpha=0.7)
    ax2.set_title('Palindromic Positions')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Is Palindromic (0/1)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Pattern density
    window_size = min(20, len(digits) // 10)
    if window_size > 0:
        densities = []
        for i in range(0, len(digits) - window_size + 1, window_size):
            window = digits[i:i + window_size]
            density = sum(palindromic_positions[i:i + window_size]) / window_size
            densities.append(density)

        ax3.plot(range(len(densities)), densities, 's-', color=_config["colors"]["secondary"], linewidth=2)
        ax3.set_title(f'Palindromic Density (Window Size {window_size})')
        ax3.set_xlabel('Window Position')
        ax3.set_ylabel('Density')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Special insights
    if 'special_insight' in analysis:
        insight = analysis['special_insight']
        ax4.text(0.1, 0.8, f"Special Insight:", fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.6, insight['description'], fontsize=10, wrap=True)
        if 'coefficients' in insight:
            ax4.text(0.1, 0.4, f"Coefficients: {insight['coefficients']}", fontsize=10)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, "No special insights found", ha='center', va='center', fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

    plt.suptitle(f'SSRCD Analysis: 1001^{analysis["scheherazade_power"]}', fontsize=14)
    plt.tight_layout()

    # Save plot using organized structure
    filename = f"ssrcd_analysis_1001_power_{analysis['scheherazade_power']}.png"
    filepath = get_organized_output_path('mathematical', 'ssrcd', filename)
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
            'type': 'ssrcd_analysis',
            'scheherazade_power': analysis['scheherazade_power'],
            'number_length': analysis['digit_count'],
            'is_palindromic': analysis['is_palindromic'],
            'palindromic_density': analysis['palindromic_density'],
            'backend': 'matplotlib'
        }
    }


def _plot_ssrcd_ascii(analysis: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Create ASCII representation of SSRCD analysis."""
    lines = []
    lines.append(f"SSCRD Analysis: 1001^{analysis['scheherazade_power']}")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Number: {analysis['number_string'][:80]}{'...' if len(analysis['number_string']) > 80 else ''}")
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
        if 'mathematical_significance' in insight:
            lines.append(f"  Mathematical Significance: {insight['mathematical_significance']}")
        lines.append("")

    # Pattern summary
    lines.append("Pattern Analysis:")
    lines.append(f"  Palindromic Patterns: {len(analysis['palindromic_patterns'])}")
    lines.append(f"  Symmetric Patterns: {len(analysis['symmetric_patterns'])}")
    if analysis['repeated_patterns']:
        lines.append(f"  Repeated Digit Patterns: {len(analysis['repeated_patterns'])}")
    lines.append("")

    # Binomial patterns
    if analysis['binomial_patterns']:
        lines.append("Binomial Coefficient Patterns:")
        for pattern in analysis['binomial_patterns'][:5]:  # Show first 5
            lines.append(f"  Position {pattern['position']}: {pattern['description']}")
        if len(analysis['binomial_patterns']) > 5:
            lines.append(f"  ... and {len(analysis['binomial_patterns']) - 5} more")
    else:
        lines.append("No binomial coefficient patterns found.")
    lines.append("")

    # Save to file using organized structure
    filename = f"ssrcd_analysis_1001_power_{analysis['scheherazade_power']}_ascii.txt"
    filepath = get_organized_output_path('mathematical', 'ssrcd', filename)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'ssrcd_analysis_ascii',
            'scheherazade_power': analysis['scheherazade_power'],
            'number_length': analysis['digit_count'],
            'is_palindromic': analysis['is_palindromic'],
            'backend': 'ascii'
        }
    }


def plot_continued_fraction_convergence(value: float,
                                       max_terms: int = 20,
                                       backend: Optional[str] = None,
                                       **kwargs) -> Dict[str, Any]:
    """
    Create an enhanced convergence visualization showing how continued fractions approach the true value.

    Args:
        value: Real number to analyze
        max_terms: Maximum number of continued fraction terms
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths

    Examples:
        >>> plot_continued_fraction_convergence(3.14159, 15)
        {'files': ['output/mathematical/continued_fractions/convergence_3_14159_15.png'], 'metadata': {...}}
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_continued_fraction_convergence_matplotlib(value, max_terms, **kwargs)
    else:
        raise ValueError(f"Convergence visualization requires matplotlib backend, got: {backend}")


def _plot_continued_fraction_convergence_matplotlib(value: float,
                                                   max_terms: int = 20,
                                                   **kwargs) -> Dict[str, Any]:
    """Create enhanced convergence visualization using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Get continued fraction data
    terms = continued_fraction_approximation(value, max_terms)
    convergents = convergents_from_continued_fraction(terms)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Terms as a sequence
    ax1 = fig.add_subplot(gs[0, 0])
    positions = list(range(len(terms)))
    bars = ax1.bar(positions, terms, color=_config["colors"]["primary"], alpha=0.7)
    ax1.set_title('Continued Fraction Terms', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Term Position')
    ax1.set_ylabel('Term Value')
    ax1.set_xticks(positions)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, term in zip(bars, terms):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{term}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Convergents and their errors
    ax2 = fig.add_subplot(gs[0, 1])
    if convergents:
        approximations = [num/den for num, den in convergents]
        errors = [abs(approx - value) for approx in approximations]

        # Plot approximations
        ax2.plot(range(len(approximations)), approximations, 'b-o',
                linewidth=2, markersize=6, label='Convergents')
        ax2.axhline(y=value, color='r', linestyle='--', linewidth=2, label=f'True Value ({value})')
        ax2.set_title('Convergent Approximations', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Approximated Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Error convergence (log scale)
    ax3 = fig.add_subplot(gs[1, 0])
    if convergents:
        approximations = [num/den for num, den in convergents]
        errors = [abs(approx - value) for approx in approximations]

        ax3.semilogy(range(len(errors)), errors, 'r-o', linewidth=2, markersize=6)
        ax3.set_title('Convergence Error (Log Scale)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Absolute Error')
        ax3.grid(True, alpha=0.3)

        # Add error values as text
        for i, error in enumerate(errors):
            if error > 0:
                ax3.text(i, error * 1.2, '.2e', ha='center', va='bottom', fontsize=8)

    # Plot 4: Fraction representation
    ax4 = fig.add_subplot(gs[1, 1])
    if convergents:
        fractions_text = "Convergent Fractions:\n\n"
        for i, (num, den) in enumerate(convergents[:8]):  # Show first 8
            fractions_text += f"{i}: {num}/{den} = {num/den:.6f}\n"

        ax4.text(0.1, 0.5, fractions_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        ax4.set_title('Fraction Representations', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

    # Plot 5: Convergence rate analysis
    ax5 = fig.add_subplot(gs[2, :])
    if len(convergents) > 1:
        approximations = [num/den for num, den in convergents]
        errors = [abs(approx - value) for approx in approximations]

        # Calculate convergence rates
        convergence_rates = []
        for i in range(1, len(errors)):
            if errors[i-1] > 0 and errors[i] > 0:
                rate = errors[i-1] / errors[i]
                convergence_rates.append(rate)

        if convergence_rates:
            ax5.plot(range(1, len(convergence_rates) + 1), convergence_rates,
                    'g-o', linewidth=2, markersize=6)
            ax5.axhline(y=1, color='k', linestyle=':', alpha=0.5, label='Linear convergence')
            ax5.set_title('Convergence Rate Analysis', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Convergence Rate (Error[n-1]/Error[n])')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Continued Fraction Convergence Analysis: {value}',
                fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()

    # Save the convergence analysis
    value_str = str(value).replace('.', '_')
    filename = f"continued_fraction_convergence_{value_str}_{max_terms}.png"
    filepath = get_organized_output_path('mathematical', 'continued_fractions', filename)
    fig.savefig(filepath, dpi=_config["dpi"], bbox_inches='tight')
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'continued_fraction_convergence',
            'value': value,
            'max_terms': max_terms,
            'terms_found': len(terms),
            'convergents_found': len(convergents),
            'backend': 'matplotlib'
        }
    }


def plot_base_conversion_matrix(start_base: int = 2, end_base: int = 16,
                              number: int = 1001,
                              backend: Optional[str] = None,
                              **kwargs) -> Dict[str, Any]:
    """
    Create a matrix visualization showing number representations across different bases.

    Args:
        start_base: Starting base for conversion (minimum 2)
        end_base: Ending base for conversion (maximum 36)
        number: Number to convert across bases
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths

    Examples:
        >>> plot_base_conversion_matrix(2, 16, 1001)
        {'files': ['output/mathematical/base_conversions/matrix_2_16_1001.png'], 'metadata': {...}}
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_base_conversion_matrix_matplotlib(start_base, end_base, number, **kwargs)
    else:
        raise ValueError(f"Matrix visualization requires matplotlib backend, got: {backend}")


def _plot_base_conversion_matrix_matplotlib(start_base: int, end_base: int, number: int,
                                          **kwargs) -> Dict[str, Any]:
    """Create base conversion matrix using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Generate base conversions
    base_range = range(max(2, start_base), min(37, end_base + 1))
    conversions = {}

    for base in base_range:
        try:
            converted = convert_between_bases(number, 10, base)
            conversions[base] = converted
        except:
            conversions[base] = "ERROR"

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create a grid layout
    n_bases = len(base_range)
    cell_width = 1.0
    cell_height = 0.8

    # Draw cells and text
    for i, base in enumerate(base_range):
        x = i * cell_width
        y = 0

        # Draw cell background
        rect = patches.Rectangle((x, y), cell_width, cell_height,
                               linewidth=1, edgecolor='black',
                               facecolor=_config["colors"]["background"])
        ax.add_patch(rect)

        # Add base number
        ax.text(x + cell_width/2, y + cell_height - 0.2, f'Base {base}',
               ha='center', va='center', fontsize=10, fontweight='bold')

        # Add converted number
        conversion = conversions[base]
        ax.text(x + cell_width/2, y + cell_height/2, str(conversion),
               ha='center', va='center', fontsize=9, family='monospace')

        # Add digit count
        if conversion != "ERROR":
            digit_count = len(str(conversion))
            ax.text(x + cell_width/2, y + 0.2, f'{digit_count} digits',
                   ha='center', va='center', fontsize=8, color='gray')

    # Set axis limits and labels
    ax.set_xlim(0, n_bases * cell_width)
    ax.set_ylim(0, cell_height)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add title and metadata
    title = f'Number {number} in Different Bases (Base {start_base} to {end_base})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add statistics
    stats_text = f"Statistics:\n"
    stats_text += f"Total bases: {n_bases}\n"
    valid_conversions = sum(1 for conv in conversions.values() if conv != "ERROR")
    stats_text += f"Valid conversions: {valid_conversions}\n"
    avg_digits = sum(len(str(conv)) for conv in conversions.values() if conv != "ERROR") / max(1, valid_conversions)
    stats_text += ".1f"

    ax.text(n_bases * cell_width + 0.5, cell_height/2, stats_text,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    # Save matrix visualization
    filename = f"base_conversion_matrix_{start_base}_{end_base}_{number}.png"
    filepath = get_organized_output_path('mathematical', 'base_conversions', filename)
    fig.savefig(filepath, dpi=_config["dpi"], bbox_inches='tight')
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'base_conversion_matrix',
            'start_base': start_base,
            'end_base': end_base,
            'number': number,
            'total_bases': n_bases,
            'valid_conversions': valid_conversions,
            'backend': 'matplotlib'
        }
    }


def plot_pattern_analysis_radar(sequence: str,
                               backend: Optional[str] = None,
                               **kwargs) -> Dict[str, Any]:
    """
    Create a radar chart visualization of pattern analysis in a sequence.

    Args:
        sequence: String sequence to analyze for patterns
        backend: Visualization backend
        **kwargs: Additional parameters

    Returns:
        Dict: Visualization metadata and file paths

    Examples:
        >>> plot_pattern_analysis_radar("1001200130014001")
        {'files': ['output/mathematical/pattern_analysis/radar_1001200130014001.png'], 'metadata': {...}}
    """
    backend = backend or _config["backend"]

    if backend == "matplotlib":
        return _plot_pattern_analysis_radar_matplotlib(sequence, **kwargs)
    else:
        raise ValueError(f"Radar visualization requires matplotlib backend, got: {backend}")


def _plot_pattern_analysis_radar_matplotlib(sequence: str, **kwargs) -> Dict[str, Any]:
    """Create pattern analysis radar chart using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from math import pi
    except ImportError:
        raise ImportError("matplotlib is required for matplotlib backend")

    # Analyze sequence patterns
    digit_counts = {}
    for digit in sequence:
        if digit.isdigit():
            digit_counts[digit] = digit_counts.get(digit, 0) + 1

    # Calculate pattern metrics
    sequence_length = len(sequence)
    unique_digits = len(digit_counts)
    digit_entropy = 0

    for count in digit_counts.values():
        if count > 0:
            probability = count / sequence_length
            digit_entropy -= probability * math.log2(probability)

    # Pattern analysis metrics
    categories = ['Length', 'Unique Digits', 'Entropy', 'Digit Diversity', 'Pattern Density', 'Symmetry']
    values = [
        min(sequence_length / 100, 1),  # Normalized length
        unique_digits / 10,  # Max 10 digits (0-9)
        digit_entropy / 4,  # Max entropy for 10 symbols
        len(digit_counts) / 10,  # Digit diversity
        len(find_repeated_patterns(sequence)) / 20,  # Pattern count
        0.5  # Placeholder for symmetry measure
    ]

    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Number of variables
    N = len(categories)

    # Angle for each category
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    # Values for each category
    values += values[:1]  # Close the polygon

    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, label='Pattern Metrics', color=_config["colors"]["primary"])
    ax.fill(angles, values, alpha=0.25, color=_config["colors"]["primary"])

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Add value labels on the plot
    for angle, value, category in zip(angles[:-1], values[:-1], categories):
        ax.text(angle, value + 0.1, '.2f', ha='center', va='center',
               fontsize=9, fontweight='bold')

    # Set title
    ax.set_title(f'Pattern Analysis Radar\nSequence Length: {sequence_length}',
                size=14, fontweight='bold', pad=20)

    # Add statistics
    stats_text = f"Sequence Statistics:\n"
    stats_text += f"Length: {sequence_length}\n"
    stats_text += f"Unique digits: {unique_digits}\n"
    stats_text += ".3f"
    stats_text += ".3f"

    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save radar visualization
    seq_hash = hash(sequence) % 10000  # Simple hash for filename
    filename = f"pattern_analysis_radar_{seq_hash}.png"
    filepath = get_organized_output_path('mathematical', 'pattern_analysis', filename)
    fig.savefig(filepath, dpi=_config["dpi"], bbox_inches='tight')
    plt.close(fig)

    return {
        'files': [str(filepath)],
        'metadata': {
            'type': 'pattern_analysis_radar',
            'sequence_length': sequence_length,
            'unique_digits': unique_digits,
            'entropy': digit_entropy,
            'backend': 'matplotlib'
        }
    }
