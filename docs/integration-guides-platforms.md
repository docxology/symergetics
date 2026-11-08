# Integrating rational geometry with arrays, notebooks and external consumers

[Installation](../README.md#install-and-run) · [API contracts](api/index.md) · [Suite exchange](synergetics-suite.md)

## Python Ecosystem Integration

The core requires NumPy. Scientific tools such as SciPy, SymPy and plotting
backends are installed with the `scientific` extra. Pandas, scikit-learn,
Flask, Django, Jupyter and ipywidgets are **not** installed by that extra and
are not package integration subsystems. Add them explicitly in an application's
own environment if needed; do not treat example adapters as exported APIs.

### Integration with NumPy

```python
import numpy as np
from symergetics import QuadrayCoordinate as Q

points = [Q(0, 0, 0, 0), Q(2, 1, 1, 0), Q(2, 1, 0, 1)]
anchor = points[0]
xyz = np.asarray([point.sub(anchor).to_xyz() for point in points], dtype=float)
assert xyz.shape == (3, 3)
exact_squared_distance = points[1].squared_distance_exact(points[2])
```

Arrays of floats intentionally cross the exactness boundary. For huge translated
coordinates, subtract exact quadray anchors first; a matrix product over absolute
float labels can destroy a small shape. `as_array()` is already floating point.
`dtype=object` can hold Python rational objects but does not turn general NumPy
or SciPy algorithms into exact arithmetic.

### Integration with SciPy

Use floating XYZ arrays for numerical solvers, preserving the native inputs
alongside them. A solver result is continuous: `Q.from_xyz(*result)` then makes
a separate nearest-lattice choice. Report projection error instead of assuming
optimization and snapping commute. The package does not provide a SciPy
optimizer or a statistical inference model.

### Integration with SymPy

```python
from sympy import Rational
from symergetics import SymergeticsNumber as N

value = N(22, 7).value
symbolic = Rational(value.numerator, value.denominator)
assert symbolic == Rational(22, 7)
```

This preserves the represented rational, not an intended irrational constant.
There is no `to_sympy_fraction()` method on `SymergeticsNumber`.

## Exact data exchange and JSON

JSON numbers can lose large integer precision in JavaScript. Encode integer
labels and rational numerator/denominator pairs as decimal strings, and record
the coordinate convention, embedding scale, units and surface indices. A small
application-owned encoding is:

```python
import json
from symergetics import QuadrayCoordinate as Q

q = Q(2, 1, 1, 0)
metric = q.squared_norm_exact()
payload = {
    "convention": "symergetics Urner scale 1; minimum-zero gauge",
    "quadray": [str(component) for component in q.as_tuple()],
    "squared_norm_xyz": {"numerator": str(metric.numerator), "denominator": str(metric.denominator)},
}
wire = json.dumps(payload)
restored = Q(*(int(component) for component in json.loads(wire)["quadray"]))
assert restored.as_tuple() == q.as_tuple()
```

This is an explicit adapter, **not** a package serializer. Do not serialize
solid metadata with a blanket float conversion and still call it exact.
For suite-native mapping and orientation differences use the
[suite interoperability contract](synergetics-suite.md).

## Scientific Computing Integration

### Integration with Jupyter Notebooks

Select a Python 3.13 kernel in the same environment as the checkout or downloaded
release wheel. Follow the [root installation routes](../README.md#install-and-run);
there is no supported `pip install symergetics` PyPI route. Notebook display
libraries must be installed separately. Use the file paths returned by a plot
for display; the plotting functions do not return an `Axes` object.

## Web Development Integration

A web service must define its own schema, input-size limits, authentication,
resource policy and error responses. Exact arithmetic is unbounded in integer
bit length; plotting also writes files and uses process-global configuration.
Do not mutate that configuration independently in simultaneous requests.
Prefer separate worker processes and request-specific output directories.
No Flask/Django routes, browser widget framework or server implementation ship
as part of this package.

## Data Science and Machine Learning Integration

Geometry can supply explicit features such as scale-declared squared distances
or IVM volume. It does not by itself validate a prediction target, justify a
coordinate representation or establish a causal model. Train/test splits,
units, quantization error and baseline comparisons remain application concerns.
See [conceptual research directions](research/index.md) for non-API context.
