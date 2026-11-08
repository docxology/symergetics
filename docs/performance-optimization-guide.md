# Measuring rational arithmetic, lattice projection and rendering cost

[Architecture](architecture/index.md) · [Advanced boundaries](advanced-topics-deep-dive.md)

## Understanding Performance Characteristics

There are no hardware-independent timing or speedup guarantees here. Python
integers grow with their bit length; an arithmetic operation is not constant
cost merely because it is written as one expression. Rational operations also
normalize numerators and denominators. Report operand sizes with timings.

| Operation | Relevant cost driver |
|---|---|
| `scheherazade_power(n)` | Exponent and resulting integer size; formatting has an additional cost |
| `primorial(n)` | Sieve upper bound, stored primes and growing product |
| `extract_palindromic_patterns` | Decimal digit length and substring population |
| `from_xyz` | Embedding geometry, exact coefficient size and closest-vector search |
| `analyze_geometry` | Declared triangles, shell count and enclosure work |
| Plotting | Layout, number of primitives, nominal pixels and encoder/backend work |

The default inverse has a BCC-specific fast path; skew embeddings use a bounded
three-dimensional closest-vector search. Do not infer a universal O(1) inverse
or a benchmark advantage over a different API with different rounding semantics.
Sources: [coordinates.py](../symergetics/core/coordinates.py),
[primorials.py](../symergetics/computation/primorials.py),
[polyhedra.py](../symergetics/geometry/polyhedra.py).

## Profiling and Benchmarking

Run this standalone measurement in a prepared Python 3.13 environment; it
reports actual samples rather than embedding expected machine-specific times:

```python
from statistics import median
from time import perf_counter
from symergetics import scheherazade_power

samples = []
for _ in range(7):
    started = perf_counter()
    value = scheherazade_power(100)
    samples.append(perf_counter() - started)
print({"median_seconds": median(samples), "samples_seconds": samples,
       "result_bits": value.value.numerator.bit_length()})
```

This measures computation, not decimal formatting, plotting or import time.
Record interpreter/package revision, OS, CPU, dependency versions, warm/cold
state and input parameters. Use `cProfile` for call attribution and `tracemalloc`
for traced Python allocations; traced allocations are not total process RSS or
all NumPy/renderer native memory. Small timing differences need repeated samples
and a like-for-like baseline, not a single stopwatch reading.

## Optimization Strategies

- Keep exact integers/rationals until a numerical consumer needs floats.
- Stream large populations in bounded batches rather than retaining every
  analysis dictionary or figure. Choose batch size from measured memory use.
- For repeated work, cache immutable input tuples and parameters in the
  application. Public vertices mutate; cached object identity is not a valid
  geometry cache key. Coordinate objects are mutable hash keys: do not mutate
  them while they are in a dictionary or set.
- For many XYZ conversions, a NumPy matrix product may be useful only when
  fixed-width/float range and precision are acceptable. Preserve displacement-
  first conversion for huge translations.
- Measure serialization, process startup and interprocess transfer before using
  multiprocessing. Global plotting configuration is not a per-request context.

## Rendering memory and output size

The nominal raster canvas is width × height × DPI squared. One RGBA buffer
alone uses four bytes per nominal pixel; layout, multiple buffers and encoders
add memory. Configuration rejects nonpositive/nonfinite dimensions and nominal
canvases exceeding 100,000,000 pixels. Tight bounding boxes can change the final
saved dimensions, so this is a guard, not an exact peak-memory bound.

Use fewer panels or lower DPI when appropriate. The unreleased main poster's
focused default is meant for legibility; `section="all"` restores all seventeen
topics and their larger layout. Ordinary figure configuration does not promise
to resize fixed-layout abstract APIs. See [poster contracts](api/index.md#scientific-poster-sections-unreleased-main).

## What is not an implemented acceleration facility

There is no package GPU backend, coordinate pool, distributed scheduler,
automatic cache, monitoring service or compressed-number storage API. Mnemonic
encoding formats numbers for people; it is not numeric compression and does not
remove the arithmetic/storage cost of the original integer. Numerical GPU or
float-vectorized alternatives require a separate accuracy comparison before
they can replace exact rational work.
