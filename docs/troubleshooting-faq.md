# Troubleshooting installation, coordinate results and plot outputs

[API contracts](api/index.md) · [Installation](../README.md#install-and-run) · [Tests](testing-guide.md)

## Quick Start Issues

### Installation Problems

`symergetics` is not on PyPI. Install a checkout or a downloaded GitHub release
wheel using the [root instructions](../README.md#install-and-run). Scientific
work uses an explicitly selected Python 3.13 environment; the repository pin
need not change. NumPy is required by core coordinates. Plotting and scientific
libraries come from the `scientific` extra. A missing optional library is not
fixed by silently selecting a different backend or returning fake data.

### Import and Usage Issues

From the checkout, inspect the interpreter and package location:

```bash
uv run --python 3.13 --extra scientific python -c "import sys, symergetics; print(sys.executable); print(symergetics.__file__)"
```

An editable checkout and a release-wheel environment are different consumers.
Run wheel probes outside the checkout to avoid source shadowing. Check both
the distribution version and import path against the intended release interface.
Avoid naming local scripts `numpy.py`, `fractions.py` or `symergetics.py`.

## Mathematical Computation Issues

### Precision and Accuracy Problems

`SymergeticsNumber(0.1)` preserves the binary float's rational value.
`SymergeticsNumber("0.1")` expresses exactly 1/10. Irrational roots and constants
remain approximations even when stored as fractions. A denominator cap is not
a universal error certificate. See [numeric details](api/index.md#symergeticsnumber).

### Quadray normalization does not fix a component sum

The constructor rounds components to integers, halves away from zero, then
subtracts their minimum. At least one normalized component is zero; the sum
is **not** fixed. Common-offset aliases occupy the same position but tuple
equality compares stored labels. `add`/`sub` preserve raw component results;
canonicalize deliberately when using tuples as physical identities.

### Coordinate round trips are quantized

`from_xyz` selects the nearest full integer-quadray site. Arbitrary Cartesian
points need not round-trip, and a nearest result need not be FCC. Continuous
rotations, projections and fractional scale helpers can quantize. Use
`dot_exact` and squared-distance methods when the represented metric must
stay exact; do not assume a generally irrational distance is rational.

### Volume calculations returning unexpected results

A basis-corner tetrahedron has IVM volume 1/4, not one. The default regular
tetrahedron has native volume one; default scale-one XYZ volume is 8/3 times
native volume. Canonical cube/tetrahedron edges differ. Mutable vertices change
live measurements. Warped faces mean the declared triangle fan, not a convex
hull. Read [solid preconditions and units](api/index.md#indexed-surfaces-and-live-solid-analysis).

### Which centroid should I use?

`centroid()` is a floor-snapped lattice approximation. `centroid_exact()` is
the rational mean of declared vertices. `volume_centroid_exact()` is the
uniform-solid mass center and raises `ValueError` for zero volume. Returned
rational tuples must not be reconstructed through the snapping coordinate
constructor if their exact value is needed.

## Visualization Issues

### Plotting and Display Problems

Use `MPLBACKEND=Agg` for headless Matplotlib commands. Configuration takes one
dictionary: `set_config({"output_dir": "/tmp/my-figures", "dpi": 120})`, not
keyword arguments. Functions return dictionaries with `files` and `metadata`,
not figure objects. Use returned paths rather than guessing a flat `output/`
filename. Configure output outside existing user artifacts. Backend support
varies by function; the scientific poster and solid graphical abstract require
Matplotlib. See [visualization contracts](api/index.md#visualization-module).

### Missing sections, large coordinate labels or changed origins

Since 1.4.0, a mega abstract defaults to a focused six-panel `overview`;
request `section="all"` for seventeen topics. Earlier wheels retain their own
interfaces. Separate explicit paths prevent sections overwriting the
same primary filename. Solid plots are anchor-relative at scale one; exact
world coordinates remain in metadata. Bounded labels deliberately omit huge
exact strings from the drawing, not from the metadata. False optional-section
flags remove those sections; they do not replace them with empty panels.

### Excessive rendering memory

Reduce figure size/DPI or choose a focused section. The 100,000,000 nominal-pixel
guard is not a guarantee of peak RSS or final tight-bbox dimensions. Some
abstract APIs have fixed layouts instead of inheriting ordinary `figure_size`.
See [rendering cost](performance-optimization-guide.md#rendering-memory-and-output-size).

## Development and Testing Issues

The package-wide coverage floor can fail a one-file run even when every selected
test passes. Use `--no-cov` for a targeted diagnostic, and the full coverage
command for a gate. See [test commands](testing-guide.md#running-tests). Report
actual failures and tracebacks, not an inferred mismatch with an old saved count.

## Performance Issues

Large integers, denominator growth, substring analysis, closest-point search and
rendering all have input-dependent costs. Mnemonic formatting is not memory
compression. Measure the relevant computation separately from decimal conversion
and display. See [profiling guidance](performance-optimization-guide.md).

## Getting Help

Include the source revision or release filename, Python/dependency versions,
minimal complete input, selected backend, configuration, returned metadata and
full traceback. State native/XYZ units and whether vertices or labels were
mutated. Conceptual application questions belong with the
[research material](research/index.md), not an assumed biological/quantum API.
