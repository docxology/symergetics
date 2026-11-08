# Utilities: conversion boundaries, mnemonic text and report export

`conversion.py` provides rational/float, decimal-string, base and coordinate
conversions plus continued fractions. `mnemonics.py` formats numbers and builds
memory-aid dictionaries. `reporting.py` summarizes analysis dictionaries and
exports JSON, CSV and Markdown. See [API reference](../../docs/api/index.md).

`decimal_to_fraction(decimal_str)` parses an exact decimal string.
Despite its name, `float_to_exact_rational(value, max_denominator=1000000)`
limits the denominator: it approximates the supplied binary float rather than
recovering an intended decimal. `rational_to_float(...)` is a float boundary.
`xyz_to_quadray(...)` delegates to nearest-lattice `QuadrayCoordinate.from_xyz`;
`quadray_to_xyz(...)` delegates to its float embedding. Neither creates four
independent Euclidean axes; see [core contracts](../core/README.md).

`generate_statistical_summary(analysis_results, title=...)` consumes a list of
analysis dictionaries, not a list of `SymergeticsNumber` objects. Exporters take
explicit output paths; callers must choose safe destinations. Mnemonic text and
statistical summaries do not establish physical or pedagogical effectiveness.
